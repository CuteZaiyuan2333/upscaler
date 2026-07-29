// 检查点测试工具。
//
// 由于 GPU 显存不足以处理 4096x4096 全分辨率推理，本工具采用降分辨率
// CPU 推理+后上采样策略：
//   参考图: 1024×1024 (Lanczos3 从 4096 降采样)
//   主图:   256×256   (Lanczos3 从 1024 降采样)
//   模型输出: 1024×1024 > Lanczos3 上采样到 4096x4096
//
// 每个检查点、每种后处理方式输出三张图：
//   01_reference.png      — 4096x4096 参考图
//   02_main_processed.png — 1024x1024 主图（应用相机滤镜后）
//   03_output.png         — 4096x4096 模型输出
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
    println!("测试图片: {}×{}", tw, th);

    if tw != 1024 || th != 1024 {
        eprintln!("警告: 测试图片尺寸 {}×{} 不是 1024×1024", tw, th);
    }

    //生成参考图 (4096×4096, Lanczos3 上采样)
    let reference_full = image::imageops::resize(&test_img, 4096, 4096, FilterType::Lanczos3);
    println!("参考图: 4096×4096 (Lanczos3 上采样)");

    //推理用降分辨率图像
    //模型是全卷积的，保持 4s 上采样比例即可
    //参考: 4096 > 1024, 主图: 1024 > 256 > 模型输出 1024 > 上采样回 4096
    let ref_inference = image::imageops::resize(&reference_full, 1024, 1024, FilterType::Lanczos3);
    let main_base = image::imageops::resize(&test_img, 256, 256, FilterType::Lanczos3);
    println!("推理分辨率: main=256×256, ref=1024×1024");

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

    //初始化模型（必须与训练配置一致: base_channels=64, num_res_blocks=4）
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

        //加载权重
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

            //参考图 (4096x4096 全分辨率，始终不变)
            let ref_path = filter_dir.join("01_reference.png");
            if !ref_path.exists() {
                reference_full
                    .save(&ref_path)
                    .map_err(|e| format!("保存参考图失败: {}", e))?;
            }

            //应用滤镜后的主图 (1024x1024，展示用)
            let main_display = augmentation::apply_filter(&test_img, *filter);
            let main_path = filter_dir.join("02_main_processed.png");
            main_display
                .save(&main_path)
                .map_err(|e| format!("保存主图失败: {}", e))?;

            //模型推理 (降分辨率 > 推理 > 上采样回 4096)
            let main_filtered = augmentation::apply_filter(&main_base, *filter);
            let main_tensor = rgb_image_to_tensor::<Backend>(main_filtered, &device);
            let ref_tensor = rgb_image_to_tensor::<Backend>(ref_inference.clone(), &device);

            let output_1024 = model.forward(main_tensor, ref_tensor);

            //保存张量 > 图像 > 上采样到 4096
            let tmp_path = filter_dir.join("_tmp_output_1024.png");
            save_rgb_tensor(output_1024, &tmp_path)
                .map_err(|e| format!("保存临时输出失败: {}", e))?;

            let output_1024_img = image::open(&tmp_path)
                .map_err(|e| format!("读取临时输出失败: {}", e))?
                .to_rgb8();
            let output_4096_img = image::imageops::resize(
                &output_1024_img, 4096, 4096, FilterType::Lanczos3,
            );
            let output_path = filter_dir.join("03_output.png");
            output_4096_img
                .save(&output_path)
                .map_err(|e| format!("保存输出失败: {}", e))?;

            // 清理临时文件
            let _ = fs::remove_file(&tmp_path);

            println!("{}", filter_name);
        }
    }
    println!("全部测试完成，结果保存在: {}", output_root.display());
    Ok(())
}
