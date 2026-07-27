// 训练模块 — RefGuidedUpsampler 自监督训练
//
// 训练策略：
//   从高分辨率图像（≥4096×4096）中随机裁剪配对区域：
//     main_crop   = 降采样到 256×256（模拟 Flux2 编辑后的低分辨率主图）
//     ref_crop    = 对应 1024×1024 区域（原始高分辨率参考图）
//     ground_truth = 对应 1024×1024 区域（训练目标）
//   模型学习从参考图中恢复高频细节，同时保留主图的语义内容。
//
// 显存优化（可在 8-16 GB VRAM GPU 上训练）：
//   - 默认使用 256×256 main / 1024×1024 ref 裁剪（而非全分辨率 4K）
//   - 全分辨率微调作为可选项（需 24+ GB VRAM）

use burn::{
    module::Module,
    nn::loss::MseLoss,
    optim::Optimizer,
    prelude::Backend,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::Tensor,
    tensor::backend::AutodiffBackend,
};

use image::RgbImage;
use rand::{Rng, SeedableRng};
use std::path::{Path, PathBuf};
use std::fs;

use crate::model::RefGuidedUpsamplerConfig;
use crate::RefGuidedUpsampler;
use crate::augmentation;

// ═══════════════════════════════════════════════════════════════════════════════
// 训练配置
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// 训练轮数
    pub num_epochs: usize,
    /// 每轮训练步数
    pub steps_per_epoch: usize,
    /// 学习率
    pub learning_rate: f64,
    /// AdamW 权重衰减
    pub weight_decay: f32,
    /// 主图裁剪尺寸（main 输入大小，ref 大小 = main × 4）
    pub crop_main_size: u32,
    /// 验证集比例（0.0 ~ 1.0）
    pub val_split: f32,
    /// 每 N 步保存一次检查点
    pub checkpoint_every_steps: usize,
    /// 检查点保存目录
    pub checkpoint_dir: PathBuf,
    /// 日志目录
    pub log_dir: PathBuf,
    /// 随机种子
    pub seed: u64,
    /// 是否使用全分辨率训练（1024→4096，需大量显存）
    pub full_resolution: bool,
    /// 训练时对参考图添加噪声扰动（帮助存在性门控学习）
    pub ref_noise_std: f32,
    /// L1 损失权重
    pub l1_weight: f32,
    /// 感知损失权重（当前为占位，需额外依赖）
    pub perceptual_weight: f32,
    /// 门控监督损失权重（当前为 0，预留）
    pub gate_supervision_weight: f32,
    /// 是否对主图应用相机滤镜增强（模拟 Flux2 编辑引入的色彩/光照变化）
    pub use_camera_filters: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 100,
            steps_per_epoch: 1000,
            learning_rate: 1e-4,
            weight_decay: 1e-4,
            crop_main_size: 256,
            val_split: 0.05,
            checkpoint_every_steps: 500,
            checkpoint_dir: PathBuf::from("checkpoints"),
            log_dir: PathBuf::from("logs"),
            seed: 42,
            full_resolution: false,
            ref_noise_std: 0.02,
            l1_weight: 1.0,
            perceptual_weight: 0.0,
            gate_supervision_weight: 0.0,
            use_camera_filters: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 数据集
// ═══════════════════════════════════════════════════════════════════════════════

/// 高分辨率图像数据集。
///
/// 从目录中读取所有 PNG/JPG 图像，训练时动态裁剪配对区域。
pub struct HighResDataset {
    image_paths: Vec<PathBuf>,
    rng: rand::rngs::StdRng,
}

impl HighResDataset {
    /// 扫描目录，收集所有 PNG/JPG 图像文件。
    pub fn from_dir(data_dir: &Path) -> Result<Self, String> {
        if !data_dir.exists() {
            return Err(format!("数据集目录不存在: {}", data_dir.display()));
        }

        let mut image_paths: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(data_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "png" || ext == "PNG" || ext == "jpg" || ext == "jpeg" {
                        image_paths.push(path);
                    }
                }
            }
        }

        if image_paths.is_empty() {
            return Err(format!("目录中未找到 PNG/JPG 图像: {}", data_dir.display()));
        }

        Ok(Self {
            image_paths,
            rng: rand::rngs::StdRng::seed_from_u64(42),
        })
    }

    /// 图像数量。
    pub fn len(&self) -> usize {
        self.image_paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.image_paths.is_empty()
    }

    /// 随机选择一张图像，加载并裁剪训练对。
    ///
    /// 返回 (main_crop, ref_crop, ground_truth, image_path)。
    pub fn random_crop_pair(
        &mut self,
        main_size: u32,
        use_camera_filters: bool,
    ) -> Result<(RgbImage, RgbImage, RgbImage, PathBuf), String> {
        let idx = self.rng.gen_range(0..self.image_paths.len());
        let path = &self.image_paths[idx];

        let img = image::open(path).map_err(|e| format!("打开 {} 失败: {}", path.display(), e))?;
        let rgb = img.to_rgb8();
        let (img_w, img_h) = rgb.dimensions();

        let ref_size = main_size * 4;

        if img_w < ref_size || img_h < ref_size {
            return Err(format!(
                "图像 {} 尺寸 {}×{} 小于所需 {}×{}",
                path.display(), img_w, img_h, ref_size, ref_size,
            ));
        }

        let max_x = img_w - ref_size;
        let max_y = img_h - ref_size;
        let x = if max_x > 0 { self.rng.gen_range(0..max_x) } else { 0 };
        let y = if max_y > 0 { self.rng.gen_range(0..max_y) } else { 0 };

        let ref_crop = image::imageops::crop_imm(&rgb, x, y, ref_size, ref_size).to_image();

        let main_crop = image::imageops::resize(
            &ref_crop, main_size, main_size, image::imageops::FilterType::Lanczos3,
        );

        // 模拟 Flux2 编辑引入的退化
        let main_crop = if self.rng.gen_bool(0.3) {
            image::imageops::blur(&main_crop, 0.5)
        } else {
            main_crop
        };

        // 相机滤镜增强：模拟 Flux2 编辑引入的色彩/光照变化
        // 80% 概率对主图应用随机相机滤镜，参考图保持不变
        let main_crop = if use_camera_filters && self.rng.gen_bool(0.8) {
            augmentation::apply_random_filter(&mut self.rng, &main_crop)
        } else {
            main_crop
        };

        // 随机翻转增强
        let (ref_crop, main_crop) = if self.rng.gen_bool(0.5) {
            (image::imageops::flip_horizontal(&ref_crop), image::imageops::flip_horizontal(&main_crop))
        } else {
            (ref_crop, main_crop)
        };
        let (ref_crop, main_crop) = if self.rng.gen_bool(0.5) {
            (image::imageops::flip_vertical(&ref_crop), image::imageops::flip_vertical(&main_crop))
        } else {
            (ref_crop, main_crop)
        };

        Ok((main_crop, ref_crop.clone(), ref_crop, path.clone()))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 张量转换工具
// ═══════════════════════════════════════════════════════════════════════════════

fn rgb_to_tensor<B: Backend>(img: &RgbImage, device: &B::Device) -> Tensor<B, 4> {
    let (w, h) = img.dimensions();
    let pixels = img.as_raw();
    let mut data = Vec::with_capacity(pixels.len());
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                data.push(pixels[((y * w + x) * 3 + c) as usize] as f32 / 255.0);
            }
        }
    }
    Tensor::from_data(
        burn::tensor::TensorData::new(data, [1, 3, h as usize, w as usize]),
        device,
    )
}

fn add_gaussian_noise<B: Backend>(tensor: Tensor<B, 4>, std: f32, device: &B::Device) -> Tensor<B, 4> {
    if std <= 0.0 {
        return tensor;
    }
    let shape = tensor.dims();
    let noise = Tensor::<B, 1>::random(
        [shape[1] * shape[2] * shape[3]],
        burn::tensor::Distribution::Normal(0.0, std as f64),
        device,
    );
    tensor + noise.reshape([shape[0], shape[1], shape[2], shape[3]])
}

// ═══════════════════════════════════════════════════════════════════════════════
// 损失函数
// ═══════════════════════════════════════════════════════════════════════════════

pub fn l1_loss<B: Backend>(pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    (pred - target).abs().mean()
}

pub fn mse_loss<B: Backend>(pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    let loss_module = MseLoss::new();
    loss_module.forward(pred, target, burn::nn::loss::Reduction::Mean)
}

// ═══════════════════════════════════════════════════════════════════════════════
// 训练器
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Default)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub step: usize,
    pub total_steps: usize,
    pub loss: f32,
    pub avg_loss: f32,
    pub best_val_loss: f32,
    pub samples_seen: usize,
}

pub struct Trainer<B: Backend> {
    model: RefGuidedUpsampler<B>,
    config: TrainingConfig,
    device: B::Device,
    metrics: TrainingMetrics,
}

impl<B: Backend> Trainer<B> {
    pub fn new(
        model_config: RefGuidedUpsamplerConfig,
        training_config: TrainingConfig,
        device: B::Device,
    ) -> Self {
        let model = model_config.init(&device);
        Self {
            model,
            config: training_config,
            device,
            metrics: TrainingMetrics { best_val_loss: f32::MAX, ..Default::default() },
        }
    }

    /// 从检查点恢复模型。
    pub fn load_checkpoint(&mut self, checkpoint_path: &Path) -> Result<(), String> {
        let model_path = checkpoint_path.join("model_latest.bin");
        if !model_path.exists() {
            return Err(format!("检查点文件不存在: {}", model_path.display()));
        }
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.model = self.model.clone().load_file(
            model_path.to_str().unwrap(),
            &recorder,
            &self.device,
        ).map_err(|e| format!("加载模型失败: {:?}", e))?;
        println!("✅ 从 {} 加载模型", model_path.display());
        Ok(())
    }

    /// 保存检查点。
    pub fn save_checkpoint(&self, step: usize) -> Result<(), String> {
        let dir = &self.config.checkpoint_dir;
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let model_path = dir.join(format!("model_step_{}.bin", step));
        let latest_path = dir.join("model_latest.bin");

        self.model.clone().save_file(model_path.to_str().unwrap(), &recorder)
            .map_err(|e| format!("保存模型失败: {:?}", e))?;
        fs::copy(&model_path, &latest_path).map_err(|e| e.to_string())?;

        println!("💾 检查点已保存: step={}", step);
        Ok(())
    }

    pub fn model(&self) -> &RefGuidedUpsampler<B> { &self.model }
    pub fn metrics(&self) -> &TrainingMetrics { &self.metrics }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 训练循环（需要 Autodiff 后端）
// ═══════════════════════════════════════════════════════════════════════════════

impl<B: AutodiffBackend> Trainer<B> {
    pub fn train(&mut self, dataset: &mut HighResDataset) -> Result<(), String> {
        let config = &self.config;

        let optim_config = burn::optim::AdamWConfig::new()
            .with_weight_decay(config.weight_decay);
        let mut optim = optim_config.init();

        fs::create_dir_all(&config.log_dir).map_err(|e| e.to_string())?;
        fs::create_dir_all(&config.checkpoint_dir).map_err(|e| e.to_string())?;

        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║           RefGuidedUpsampler 训练启动                    ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║  数据集图像数: {:>6}                                     ║", dataset.len());
        println!("║  训练轮数:     {:>6}                                     ║", config.num_epochs);
        println!("║  每轮步数:     {:>6}                                     ║", config.steps_per_epoch);
        println!("║  学习率:       {:>10.6}                                  ║", config.learning_rate);
        println!("║  裁剪尺寸:     {:>6}×{:>6} (main)                         ║",
            config.crop_main_size, config.crop_main_size);
        println!("║                 {:>6}×{:>6} (ref)                          ║",
            config.crop_main_size * 4, config.crop_main_size * 4);
        println!("║  全分辨率模式: {:>6}                                     ║",
            if config.full_resolution { "ON" } else { "OFF" });
        println!("║  参数量:       {:>6}                                     ║",
            self.model.num_params());
        println!("╚══════════════════════════════════════════════════════════╝");

        let main_size = if config.full_resolution { 1024 } else { config.crop_main_size };
        let mut global_step = 0usize;

        for epoch in 0..config.num_epochs {
            self.metrics.epoch = epoch;
            let mut epoch_loss = 0.0f32;
            let mut epoch_samples = 0usize;

            for _step_in_epoch in 0..config.steps_per_epoch {
                // 1. 加载训练样本
                let (main_img, ref_img, gt_img, _path) = match dataset.random_crop_pair(main_size, config.use_camera_filters) {
                    Ok(pair) => pair,
                    Err(e) => { eprintln!("⚠️  跳过: {}", e); continue; }
                };

                // 2. 转换为张量
                let main = rgb_to_tensor::<B>(&main_img, &self.device);
                let reference = rgb_to_tensor::<B>(&ref_img, &self.device);
                let ground_truth = rgb_to_tensor::<B>(&gt_img, &self.device);

                // 3. 参考图噪声扰动
                let reference = add_gaussian_noise(reference, config.ref_noise_std, &self.device);

                // 4. 前向传播
                let output = self.model.forward(main, reference);

                // 5. 计算损失
                let loss = l1_loss(output, ground_truth).mul_scalar(config.l1_weight);

                // 记录损失值（必须在 backward 之前，因为 backward 会消费 loss）
                let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];

                // 6. 反向传播
                let grads = loss.backward();

                // 7. 优化器步骤
                let grads_params = burn::optim::GradientsParams::from_grads(grads, &self.model);
                self.model = optim.step(config.learning_rate, self.model.clone(), grads_params);

                global_step += 1;
                epoch_loss += loss_val;
                epoch_samples += 1;
                self.metrics.samples_seen += 1;
                self.metrics.step = global_step;
                self.metrics.total_steps = global_step;
                self.metrics.loss = loss_val;

                // 日志
                if global_step % 50 == 0 || global_step == 1 {
                    println!(
                        "[Epoch {:>4} | Step {:>6}] loss={:.6}  avg_loss={:.6}  samples={}",
                        epoch + 1, global_step, self.metrics.loss,
                        epoch_loss / epoch_samples.max(1) as f32,
                        self.metrics.samples_seen,
                    );
                }

                // 检查点
                if global_step % config.checkpoint_every_steps == 0 && global_step > 0 {
                    self.save_checkpoint(global_step)?;
                }
            }

            let avg_epoch_loss = epoch_loss / epoch_samples.max(1) as f32;
            self.metrics.avg_loss = avg_epoch_loss;
            println!(
                "━━━ Epoch {} 完成 ━━━ avg_loss={:.6}  samples={} ━━━",
                epoch + 1, avg_epoch_loss, self.metrics.samples_seen,
            );
            self.save_checkpoint(global_step)?;
        }

        println!("✅ 训练完成！总步数: {}", global_step);
        self.save_checkpoint(global_step)?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_l1_loss_shape() {
        let device = Default::default();
        let pred = Tensor::<TestBackend, 4>::ones([1, 3, 64, 64], &device);
        let target = Tensor::<TestBackend, 4>::zeros([1, 3, 64, 64], &device);
        let loss = l1_loss(pred, target);
        assert_eq!(loss.dims().len(), 1);
        let val = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!((val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rgb_to_tensor_shape() {
        let device = Default::default();
        let img = RgbImage::new(64, 64);
        let tensor = rgb_to_tensor::<TestBackend>(&img, &device);
        assert_eq!(tensor.dims(), [1, 3, 64, 64]);
    }
}