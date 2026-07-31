// 检查点测试工具。
//
// 所有图片按实际推理分辨率保存，不做任何上采样：
//   01_reference.png      — 1024x1024 参考图（模型实际输入）
//   02_main_processed.png — 256x256 主图（模型实际输入，应用相机滤镜后）
//   03_output.png         — 1024x1024 模型输出（模型实际输出）
//
// 用法：
//   cargo run --release --bin test_checkpoints

use std::fs;
use std::path::PathBuf;

use burn::backend::NdArray;
use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use image::imageops::FilterType;
use upscaler::augmentation::{self, CameraFilter};
use upscaler::rgb_image_to_tensor;
use upscaler::save_rgb_tensor;
use upscaler::RefGuidedUpsamplerConfig;

type Backend = NdArray;

fn main() -> Result<(), String> {
    let test_image_path = PathBuf::from("2026-07-27 22-34-38 (C,S1).jpg");
    let checkpoints_dir = PathBuf::from("checkpoints");
    let output_root = PathBuf::from("test");

    //加载测试图片
    let test_img = image::open(&test_image_path)
        .map_err(|e| format!("无法打开测试图片 {}: {}", test_image_path.display(), e))?
        .to_rgb8();
    let (tw, th) = test_img.dimensions();
    println!("测试图片: {}x{}", tw, th);

    if tw != 1024 || th != 1024 {
        eprintln!("警告: 测试图片尺寸 {}x{} 不是 1024x1024", tw, th);
    }

    //推理用图像（实际分辨率）
    //模型保持 4x 上采样比例：主图 x4 = 参考图 = 输出
    let ref_img = image::imageops::resize(&test_img, 1024, 1024, FilterType::Lanczos3);
    let main_img = image::imageops::resize(&test_img, 256, 256, FilterType::Lanczos3);
    println!("推理分辨率: main=256x256, ref=1024x1024, output=1024x1024");

    //收集检查点文件
    let mut checkpoint_files: Vec<PathBuf> = Vec::new();
    if checkpoints_dir.exists() {
        for entry in fs::read_dir(&checkpoints_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "bin") {
                checkpoint_files.push(path);
            }
        }
    }
    checkpoint_files.sort();
    println!("找到 {} 个检查点", checkpoint_files.len());

    if checkpoint_files.is_empty() {
        return Err("未找到任何检查点文件".to_string());
    }

    //准备后处理方式
    let filters = CameraFilter::all();

    //初始化模型（与训练配置一致：Nearest 插值）
    let device = Default::default();
    let model = RefGuidedUpsamplerConfig::new().init::<Backend>(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    println!("模型参数量: {}", model.num_params());

    //遍历每个检查点
    for cp_path in &checkpoint_files {
        let cp_name = cp_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        println!("检查点: {}", cp_name);

        // 加载权重
        let model = match model
            .clone()
            .load_file(cp_path.to_str().unwrap(), &recorder, &device)
        {
            Ok(m) => {
                println!("权重加载成功");
                m
            }
            Err(e) => {
                eprintln!("加载失败: {:?}", e);
                continue;
            }
        };

        let cp_dir = output_root.join(cp_name);
        fs::create_dir_all(&cp_dir).map_err(|e| e.to_string())?;

        //对每种后处理方式
        for filter in filters {
            let filter_name = format!("{:?}", filter).to_lowercase();
            let filter_dir = cp_dir.join(&filter_name);
            fs::create_dir_all(&filter_dir).map_err(|e| e.to_string())?;

            // 1. 参考图（实际推理分辨率：1024x1024）
            let ref_path = filter_dir.join("01_reference.png");
            ref_img.save(&ref_path)
                .map_err(|e| format!("保存参考图失败: {}", e))?;

            // 2. 主图（实际推理分辨率：256x256，应用相机滤镜）
            let main_processed = augmentation::apply_filter(&main_img, *filter);
            let main_path = filter_dir.join("02_main_processed.png");
            main_processed.save(&main_path)
                .map_err(|e| format!("保存主图失败: {}", e))?;

            // 3. 模型推理，输出实际分辨率：1024x1024
            let main_tensor = rgb_image_to_tensor::<Backend>(main_processed, &device);
            let ref_tensor = rgb_image_to_tensor::<Backend>(ref_img.clone(), &device);
            let output_tensor = model.forward(main_tensor, ref_tensor);

            let output_path = filter_dir.join("03_output.png");
            save_rgb_tensor(output_tensor, &output_path)
                .map_err(|e| format!("保存输出失败: {}", e))?;

            println!("ok{}", filter_name);
        }
    }
    println!("全部测试完成！结果保存在: {}", output_root.display());
    Ok(())
}