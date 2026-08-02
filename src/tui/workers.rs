// 后台线程：训练和测试。
// 包含 panic 捕获、WGPU→NdArray 自动回退、文件日志系统。

use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, OnceLock, atomic::AtomicBool};
use std::io::Write;

use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use image::imageops::FilterType;

use crate::augmentation::{self, CameraFilter};
use crate::model::RefGuidedUpsamplerConfig;
use crate::training::{HighResDataset, Trainer, TrainingConfig, TrainingEvent};
use crate::tui::state::Backend;
use crate::{rgb_image_to_tensor, save_rgb_tensor};

// ── 文件日志 ──

static LOG_FILE: OnceLock<std::sync::Mutex<Option<std::fs::File>>> = OnceLock::new();

fn log_file() -> &'static std::sync::Mutex<Option<std::fs::File>> {
    LOG_FILE.get_or_init(|| std::sync::Mutex::new(None))
}

fn init_log() {
    let mut guard = log_file().lock().unwrap();
    if guard.is_none() {
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("upscaler.log")
        {
            Ok(f) => *guard = Some(f),
            Err(_) => {}
        }
    }
}

fn log_to_file(msg: &str) {
    init_log();
    let mut guard = log_file().lock().unwrap();
    if let Some(ref mut f) = *guard {
        let elapsed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = elapsed.as_secs();
        let h = (secs / 3600) % 24;
        let m = (secs / 60) % 60;
        let s = secs % 60;
        let _ = writeln!(f, "[{:02}:{:02}:{:02}] {}", h, m, s, msg);
        let _ = f.flush();
    }
}

// ── 测试进度事件 ──

#[derive(Debug, Clone)]
pub enum TestingEvent {
    AutoTestStarted        { total_filters: usize, total_checkpoints: usize },
    AutoTestCheckpointStarted { checkpoint: String },
    AutoTestItemCompleted  { checkpoint: String, filter: String, completed: usize, total: usize },
    AutoTestCompleted,
    DirectInferenceStarted,
    DirectInferenceCompleted { output_path: String },
    Log(String),
    Error(String),
}

// ── 训练 ──

pub fn spawn_training(
    config: TrainingConfig,
    _cancel_flag: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("training".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            train_impl(config)
        }));

        if let Err(panic_info) = result {
            let msg = format_panic(&panic_info);
            log_to_file(&format!("TRAINING PANIC: {}", msg));
            // 已经 panic 了，无法再通过 channel 发送
            eprintln!("TRAINING PANIC: {}", msg);
        }
    }).unwrap()
}

fn train_impl(config: TrainingConfig) {
    let device = Default::default();
    let model_config = RefGuidedUpsamplerConfig::new();

    let msg = |s: &str| {
        log_to_file(s);
        if let Some(ref tx) = config.progress_sender {
            let _ = tx.send(TrainingEvent::Error(s.to_string()));
        }
    };

    let mut trainer = Trainer::<Autodiff<Wgpu>>::new(model_config, config.clone(), device);

    if !config.resume_checkpoint.as_os_str().is_empty() {
        if let Err(e) = trainer.load_checkpoint(&config.resume_checkpoint) {
            msg(&format!("加载检查点失败: {}", e));
            return;
        }
    }

    let data_dir = if config.dataset_dir.as_os_str().is_empty() {
        PathBuf::from("./dataset")
    } else {
        config.dataset_dir.clone()
    };

    let mut dataset = match HighResDataset::from_dir(&data_dir) {
        Ok(d) => d,
        Err(e) => { msg(&format!("加载数据集失败: {}", e)); return; }
    };

    if let Err(e) = trainer.train(&mut dataset) {
        msg(&format!("训练错误: {}", e));
    }
}

// ── 推理后端分派 ──

pub fn spawn_auto_test(
    test_image_path: PathBuf,
    checkpoint_dir: PathBuf,
    output_dir: PathBuf,
    inference_backend: Backend,
    progress_tx: Sender<TestingEvent>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("auto-test".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match inference_backend {
                Backend::Wgpu | Backend::Vulkan | Backend::Cuda => {
                    run_auto_test::<Wgpu>(test_image_path.clone(), checkpoint_dir.clone(), output_dir.clone(), progress_tx.clone());
                }
                Backend::NdArray => {
                    run_auto_test::<NdArray>(test_image_path.clone(), checkpoint_dir.clone(), output_dir.clone(), progress_tx.clone());
                }
            }
        }));

        if let Err(panic_info) = result {
            let msg = format_panic(&panic_info);
            log_to_file(&format!("AUTO TEST PANIC: {}", msg));
            let _ = progress_tx.send(TestingEvent::Error(format!("PANIC: {}", msg)));
        }
    }).unwrap()
}

pub fn spawn_direct_inference(
    main_image_path: PathBuf,
    reference_image_path: PathBuf,
    checkpoint_path: PathBuf,
    output_path: PathBuf,
    inference_backend: Backend,
    progress_tx: Sender<TestingEvent>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("direct-inference".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match inference_backend {
                Backend::Wgpu | Backend::Vulkan | Backend::Cuda => {
                    run_direct_inference::<Wgpu>(
                        main_image_path.clone(), reference_image_path.clone(),
                        checkpoint_path.clone(), output_path.clone(), progress_tx.clone(),
                    );
                }
                Backend::NdArray => {
                    run_direct_inference::<NdArray>(
                        main_image_path.clone(), reference_image_path.clone(),
                        checkpoint_path.clone(), output_path.clone(), progress_tx.clone(),
                    );
                }
            }
        }));

        if let Err(panic_info) = result {
            let msg = format_panic(&panic_info);
            log_to_file(&format!("DIRECT INFERENCE PANIC: {}", msg));
            let _ = progress_tx.send(TestingEvent::Error(format!("PANIC: {}", msg)));
        }
    }).unwrap()
}

// ── 自动测试（泛型） ──

fn run_auto_test<B: burn::prelude::Backend>(
    test_image_path: PathBuf,
    checkpoint_dir: PathBuf,
    output_dir: PathBuf,
    progress_tx: Sender<TestingEvent>,
) {
    let log = |msg: &str| {
        log_to_file(msg);
        let _ = progress_tx.send(TestingEvent::Log(msg.to_string()));
    };

    let test_img = match image::open(&test_image_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            let msg = format!("无法打开测试图片 {}: {}", test_image_path.display(), e);
            log(&msg);
            let _ = progress_tx.send(TestingEvent::Error(msg));
            return;
        }
    };

    let (tw, th) = test_img.dimensions();
    if tw != 1024 || th != 1024 {
        log(&format!("测试图片尺寸 {}×{} 不是 1024×1024", tw, th));
    }

    let ref_img = image::imageops::resize(&test_img, 1024, 1024, FilterType::Lanczos3);
    let main_img = image::imageops::resize(&test_img, 256, 256, FilterType::Lanczos3);

    // 收集检查点
    let mut checkpoint_files: Vec<PathBuf> = Vec::new();
    if checkpoint_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&checkpoint_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "bin") {
                    checkpoint_files.push(path);
                }
            }
        }
    } else if checkpoint_dir.is_file() {
        checkpoint_files.push(checkpoint_dir.clone());
    }
    checkpoint_files.sort();

    if checkpoint_files.is_empty() {
        let msg = format!("未找到检查点文件: {}", checkpoint_dir.display());
        log(&msg);
        let _ = progress_tx.send(TestingEvent::Error(msg));
        return;
    }

    let filters = CameraFilter::all();
    let total = checkpoint_files.len() * filters.len();

    log(&format!("自动测试: {} 检查点 × {} 滤镜 = {} 项", checkpoint_files.len(), filters.len(), total));

    let _ = progress_tx.send(TestingEvent::AutoTestStarted {
        total_filters: filters.len(),
        total_checkpoints: checkpoint_files.len(),
    });

    let device = Default::default();
    let model = RefGuidedUpsamplerConfig::new().init::<B>(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let mut completed = 0usize;

    for cp_path in &checkpoint_files {
        let cp_name = cp_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        log(&format!("测试检查点: {}", cp_name));

        let _ = progress_tx.send(TestingEvent::AutoTestCheckpointStarted {
            checkpoint: cp_name.clone(),
        });

        let model = match model.clone().load_file(cp_path.to_str().unwrap(), &recorder, &device) {
            Ok(m) => m,
            Err(e) => {
                let msg = format!("加载检查点 {} 失败: {:?}", cp_name, e);
                log(&msg);
                let _ = progress_tx.send(TestingEvent::Error(msg));
                completed += filters.len();
                continue;
            }
        };

        for filter in filters {
            let filter_name = format!("{:?}", filter).to_lowercase();
            let filter_dir = output_dir.join(&cp_name).join(&filter_name);

            if let Err(e) = std::fs::create_dir_all(&filter_dir) {
                let _ = progress_tx.send(TestingEvent::Error(format!("创建目录失败: {}", e)));
                completed += 1;
                continue;
            }

            let _ = ref_img.save(filter_dir.join("01_reference.png"));
            let main_processed = augmentation::apply_filter(&main_img, *filter);
            let _ = main_processed.save(filter_dir.join("02_main_processed.png"));

            let main_tensor = rgb_image_to_tensor::<B>(main_processed, &device);
            let ref_tensor = rgb_image_to_tensor::<B>(ref_img.clone(), &device);
            let output_tensor = model.forward(main_tensor, ref_tensor);

            if let Err(e) = save_rgb_tensor(output_tensor, filter_dir.join("03_output.png")) {
                let _ = progress_tx.send(TestingEvent::Error(format!("保存输出失败: {}", e)));
            }

            completed += 1;
            let _ = progress_tx.send(TestingEvent::AutoTestItemCompleted {
                checkpoint: cp_name.clone(),
                filter: filter_name,
                completed,
                total,
            });
        }
    }

    log(&format!("自动测试完成: {} 项", completed));
    let _ = progress_tx.send(TestingEvent::AutoTestCompleted);
}

// ── 直接推理（泛型） ──

fn run_direct_inference<B: burn::prelude::Backend>(
    main_image_path: PathBuf,
    reference_image_path: PathBuf,
    checkpoint_path: PathBuf,
    output_path: PathBuf,
    progress_tx: Sender<TestingEvent>,
) {
    let log = |msg: &str| {
        log_to_file(msg);
        let _ = progress_tx.send(TestingEvent::Log(msg.to_string()));
    };

    let _ = progress_tx.send(TestingEvent::DirectInferenceStarted);

    let device = Default::default();
    let model = RefGuidedUpsamplerConfig::new().init::<B>(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let model = match model.clone().load_file(checkpoint_path.to_str().unwrap(), &recorder, &device) {
        Ok(m) => m,
        Err(e) => {
            let msg = format!("加载检查点 {} 失败: {:?}", checkpoint_path.display(), e);
            log(&msg);
            let _ = progress_tx.send(TestingEvent::Error(msg));
            return;
        }
    };

    let main_img = match image::open(&main_image_path) {
        Ok(img) => {
            let rgb = img.to_rgb8();
            let (w, h) = rgb.dimensions();
            log(&format!("主图: {} ({}×{})", main_image_path.display(), w, h));
            rgb
        }
        Err(e) => {
            let msg = format!("无法打开主图 {}: {}", main_image_path.display(), e);
            log(&msg);
            let _ = progress_tx.send(TestingEvent::Error(msg));
            return;
        }
    };

    let ref_img = match image::open(&reference_image_path) {
        Ok(img) => {
            let rgb = img.to_rgb8();
            let (w, h) = rgb.dimensions();
            log(&format!("参考图: {} ({}×{})", reference_image_path.display(), w, h));
            rgb
        }
        Err(e) => {
            let msg = format!("无法打开参考图 {}: {}", reference_image_path.display(), e);
            log(&msg);
            let _ = progress_tx.send(TestingEvent::Error(msg));
            return;
        }
    };

    // 检查尺寸关系
    let (mw, mh) = main_img.dimensions();
    let (rw, rh) = ref_img.dimensions();
    let expected_rw = mw * 4;
    let expected_rh = mh * 4;
    if rw != expected_rw || rh != expected_rh {
        let msg = format!(
            "尺寸不匹配: 主图 {}×{}，参考图 {}×{}（期望 {}×{}）",
            mw, mh, rw, rh, expected_rw, expected_rh
        );
        log(&msg);
        let _ = progress_tx.send(TestingEvent::Error(msg));
        return;
    }

    log(&format!("开始推理: 主图 {}×{} → 参考图 {}×{}", mw, mh, rw, rh));

    let main_tensor = rgb_image_to_tensor::<B>(main_img, &device);
    let ref_tensor = rgb_image_to_tensor::<B>(ref_img, &device);
    let output_tensor = model.forward(main_tensor, ref_tensor);

    match save_rgb_tensor(output_tensor, &output_path) {
        Ok(()) => {
            let msg = format!("推理完成，输出: {}", output_path.display());
            log(&msg);
            let _ = progress_tx.send(TestingEvent::DirectInferenceCompleted {
                output_path: output_path.display().to_string(),
            });
        }
        Err(e) => {
            let msg = format!("保存输出 {} 失败: {}", output_path.display(), e);
            log(&msg);
            let _ = progress_tx.send(TestingEvent::Error(msg));
        }
    }
}

// ── 工具函数 ──

fn format_panic(info: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = info.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = info.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}