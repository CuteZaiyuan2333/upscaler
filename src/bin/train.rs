// 训练二进制入口。
//
// 用法：
//   cargo run --bin train --release
//
// 环境变量配置：
//   DATA_DIR         数据集目录（默认 ./dataset）
//   EPOCHS           训练轮数（默认 100）
//   STEPS_PER_EPOCH  每轮步数（默认 1000）
//   LR               学习率（默认 1e-4）
//   WEIGHT_DECAY     AdamW 权重衰减（默认 1e-4）
//   CROP_SIZE        主图裁剪尺寸（默认 256）
//   FULL_RES         全分辨率模式：1=1024→4096，0=裁剪模式（默认 0）
//   REF_NOISE        参考图噪声标准差（默认 0.02）
//   CHECKPOINT       从检查点目录恢复训练
//   SEED             随机种子（默认 42）

use std::env;
use std::path::PathBuf;

use burn::backend::{Autodiff, Wgpu};
use upscaler::RefGuidedUpsamplerConfig;
use upscaler::training::{HighResDataset, Trainer, TrainingConfig};

type Backend = Autodiff<Wgpu>;

fn main() -> Result<(), String> {
    let config = parse_config();

    println!("RefGuidedUpsampler 训练程序");
    println!("->后端: Autodiff<Wgpu>");
    println!("->模型: base_channels=64, num_res_blocks=4");

    let data_dir = env::var("DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./dataset"));

    println!("加载数据集: {}", data_dir.display());
    let mut dataset = HighResDataset::from_dir(&data_dir)?;
    println!("找到 {} 张图像", dataset.len());

    let device = Default::default();
    let model_config = RefGuidedUpsamplerConfig::new();
    let mut trainer = Trainer::<Backend>::new(model_config, config.clone(), device);

    if let Ok(checkpoint) = env::var("CHECKPOINT") {
        let cp_path = PathBuf::from(&checkpoint);
        println!("从检查点恢复: {}", cp_path.display());
        trainer.load_checkpoint(&cp_path)?;
    }

    trainer.train(&mut dataset)?;
    println!("训练完成");
    Ok(())
}

fn parse_config() -> TrainingConfig {
    let default = TrainingConfig::default();

    TrainingConfig {
        num_epochs: env::var("EPOCHS").ok().and_then(|s| s.parse().ok()).unwrap_or(default.num_epochs),
        steps_per_epoch: env::var("STEPS_PER_EPOCH").ok().and_then(|s| s.parse().ok()).unwrap_or(default.steps_per_epoch),
        learning_rate: env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(default.learning_rate),
        weight_decay: env::var("WEIGHT_DECAY").ok().and_then(|s| s.parse().ok()).unwrap_or(default.weight_decay),
        crop_main_size: env::var("CROP_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(default.crop_main_size),
        full_resolution: env::var("FULL_RES").ok().map(|s| s == "1" || s.to_lowercase() == "true").unwrap_or(default.full_resolution),
        ref_noise_std: env::var("REF_NOISE").ok().and_then(|s| s.parse().ok()).unwrap_or(default.ref_noise_std),
        seed: env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(default.seed),
        ..default
    }
}