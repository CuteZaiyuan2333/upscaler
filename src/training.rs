// 训练模块 — RefGuidedUpsampler 自监督训练。
//
// 从高分辨率图像中随机裁剪配对区域：
//   main_crop   = 降采样 4×（模拟 Flux2 编辑后的低分辨率主图）
//   ref_crop    = 对应 4× 区域（原始高分辨率参考图 = 训练目标）
// 模型学习从参考图中恢复高频细节。

use burn::{
    module::Module,
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
use std::sync::mpsc::Sender;
use std::sync::{Arc, atomic::AtomicBool};

use crate::model::RefGuidedUpsamplerConfig;
use crate::RefGuidedUpsampler;
use crate::augmentation;

// ── 训练进度事件 ──

#[derive(Debug, Clone)]
pub enum TrainingEvent {
    Started { dataset_size: usize, num_params: usize },
    EpochStarted { epoch: usize, total_epochs: usize },
    StepCompleted { epoch: usize, global_step: usize, loss: f32, avg_loss: f32, samples: usize },
    EpochCompleted { epoch: usize, avg_loss: f32 },
    CheckpointSaved { step: usize },
    Completed { total_steps: usize },
    Error(String),
}

// ── 训练配置 ──

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub steps_per_epoch: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub crop_main_size: u32,
    pub checkpoint_every_steps: usize,
    pub checkpoint_dir: PathBuf,
    pub seed: u64,
    pub full_resolution: bool,
    pub ref_noise_std: f32,
    pub use_camera_filters: bool,
    pub dataset_dir: PathBuf,
    pub resume_checkpoint: PathBuf,
    pub progress_sender: Option<Sender<TrainingEvent>>,
    pub cancel_flag: Option<Arc<AtomicBool>>,
}

// ── 数据集 ──

pub struct HighResDataset {
    image_paths: Vec<PathBuf>,
    rng: rand::rngs::StdRng,
}

impl HighResDataset {
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

        Ok(Self { image_paths, rng: rand::rngs::StdRng::seed_from_u64(42) })
    }

    pub fn len(&self) -> usize { self.image_paths.len() }

    pub fn random_crop_pair(
        &mut self, main_size: u32, use_camera_filters: bool,
    ) -> Result<(RgbImage, RgbImage, RgbImage, PathBuf), String> {
        let idx = self.rng.gen_range(0..self.image_paths.len());
        let path = &self.image_paths[idx];

        let img = image::open(path).map_err(|e| format!("打开 {} 失败: {}", path.display(), e))?;
        let rgb = img.to_rgb8();
        let (img_w, img_h) = rgb.dimensions();

        let ref_size = main_size * 4;

        if img_w < ref_size || img_h < ref_size {
            return Err(format!("图像 {} 尺寸 {}×{} 小于所需 {}×{}", path.display(), img_w, img_h, ref_size, ref_size));
        }

        let max_x = img_w - ref_size;
        let max_y = img_h - ref_size;
        let x = if max_x > 0 { self.rng.gen_range(0..max_x) } else { 0 };
        let y = if max_y > 0 { self.rng.gen_range(0..max_y) } else { 0 };

        let ref_crop = image::imageops::crop_imm(&rgb, x, y, ref_size, ref_size).to_image();

        let main_crop = image::imageops::resize(&ref_crop, main_size, main_size, image::imageops::FilterType::Lanczos3);

        // 模拟 Flux2 退化
        let main_crop = if self.rng.gen_bool(0.3) {
            image::imageops::blur(&main_crop, 0.5)
        } else {
            main_crop
        };

        // 相机滤镜增强
        let main_crop = if use_camera_filters && self.rng.gen_bool(0.8) {
            augmentation::apply_random_filter(&mut self.rng, &main_crop)
        } else {
            main_crop
        };

        // 随机翻转
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

// ── 张量工具 ──

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
    Tensor::from_data(burn::tensor::TensorData::new(data, [1, 3, h as usize, w as usize]), device)
}

fn add_gaussian_noise<B: Backend>(tensor: Tensor<B, 4>, std: f32, device: &B::Device) -> Tensor<B, 4> {
    if std <= 0.0 { return tensor; }
    let shape = tensor.dims();
    let noise = Tensor::<B, 1>::random(
        [shape[1] * shape[2] * shape[3]],
        burn::tensor::Distribution::Normal(0.0, std as f64),
        device,
    );
    tensor + noise.reshape([shape[0], shape[1], shape[2], shape[3]])
}

// ── 损失函数 ──

pub fn l1_loss<B: Backend>(pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    (pred - target).abs().mean()
}

// ── 训练器 ──

pub struct Trainer<B: Backend> {
    model: RefGuidedUpsampler<B>,
    config: TrainingConfig,
    device: B::Device,
}

impl<B: Backend> Trainer<B> {
    pub fn new(model_config: RefGuidedUpsamplerConfig, training_config: TrainingConfig, device: B::Device) -> Self {
        let model = model_config.init(&device);
        Self { model, config: training_config, device }
    }

    pub fn load_checkpoint(&mut self, checkpoint_path: &Path) -> Result<(), String> {
        let model_path = if checkpoint_path.is_dir() {
            let p = checkpoint_path.join("model_latest.bin");
            if !p.exists() { return Err(format!("检查点文件不存在: {}", p.display())); }
            p
        } else if checkpoint_path.is_file() {
            checkpoint_path.to_path_buf()
        } else {
            return Err(format!("检查点路径不存在: {}", checkpoint_path.display()));
        };
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.model = self.model.clone().load_file(model_path.to_str().unwrap(), &recorder, &self.device)
            .map_err(|e| format!("加载模型失败: {:?}", e))?;
        Ok(())
    }

    pub fn save_checkpoint(&self, step: usize) -> Result<(), String> {
        let dir = &self.config.checkpoint_dir;
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let model_path = dir.join(format!("model_step_{}.bin", step));
        let latest_path = dir.join("model_latest.bin");
        self.model.clone().save_file(model_path.to_str().unwrap(), &recorder)
            .map_err(|e| format!("保存模型失败: {:?}", e))?;
        fs::copy(&model_path, &latest_path).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn model(&self) -> &RefGuidedUpsampler<B> { &self.model }
}

// ── 训练循环 ──

impl<B: AutodiffBackend> Trainer<B> {
    fn is_cancelled(&self) -> bool {
        self.config.cancel_flag.as_ref().map_or(false, |f| f.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn send_event(&self, event: TrainingEvent) {
        if let Some(ref tx) = self.config.progress_sender {
            let _ = tx.send(event);
        }
    }

    pub fn train(&mut self, dataset: &mut HighResDataset) -> Result<(), String> {
        let config = &self.config;

        let optim_config = burn::optim::AdamWConfig::new().with_weight_decay(config.weight_decay);
        let mut optim = optim_config.init();

        fs::create_dir_all(&config.checkpoint_dir).map_err(|e| e.to_string())?;

        let num_params = self.model.num_params();

        self.send_event(TrainingEvent::Started {
            dataset_size: dataset.len(),
            num_params,
        });

        let main_size = if config.full_resolution { 1024 } else { config.crop_main_size };
        let mut global_step = 0usize;

        for epoch in 0..config.num_epochs {
            if self.is_cancelled() { return Ok(()); }

            let mut epoch_loss = 0.0f32;
            let mut epoch_samples = 0usize;

            self.send_event(TrainingEvent::EpochStarted { epoch: epoch + 1, total_epochs: config.num_epochs });

            for _ in 0..config.steps_per_epoch {
                if self.is_cancelled() { return Ok(()); }

                let (main_img, ref_img, gt_img, _path) = match dataset.random_crop_pair(main_size, config.use_camera_filters) {
                    Ok(pair) => pair,
                    Err(e) => {
                        self.send_event(TrainingEvent::Error(format!("跳过: {}", e)));
                        continue;
                    }
                };

                let main = rgb_to_tensor::<B>(&main_img, &self.device);
                let reference = rgb_to_tensor::<B>(&ref_img, &self.device);
                let ground_truth = rgb_to_tensor::<B>(&gt_img, &self.device);

                let reference = add_gaussian_noise(reference, config.ref_noise_std, &self.device);

                let output = self.model.forward(main, reference);
                let loss = l1_loss(output, ground_truth);

                let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];

                let grads = loss.backward();
                let grads_params = burn::optim::GradientsParams::from_grads(grads, &self.model);
                self.model = optim.step(config.learning_rate, self.model.clone(), grads_params);

                global_step += 1;
                epoch_loss += loss_val;
                epoch_samples += 1;

                if global_step % 50 == 0 || global_step == 1 {
                    self.send_event(TrainingEvent::StepCompleted {
                        epoch: epoch + 1,
                        global_step,
                        loss: loss_val,
                        avg_loss: epoch_loss / epoch_samples.max(1) as f32,
                        samples: global_step,
                    });
                }

                if global_step % config.checkpoint_every_steps == 0 && global_step > 0 {
                    self.save_checkpoint(global_step)?;
                    self.send_event(TrainingEvent::CheckpointSaved { step: global_step });
                }
            }

            let avg = epoch_loss / epoch_samples.max(1) as f32;
            self.send_event(TrainingEvent::EpochCompleted { epoch: epoch + 1, avg_loss: avg });
            self.save_checkpoint(global_step)?;
        }

        self.send_event(TrainingEvent::Completed { total_steps: global_step });
        Ok(())
    }
}

// ── 测试 ──

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_l1_loss_shape() {
        let device = Default::default();
        let pred = Tensor::<NdArray, 4>::ones([1, 3, 64, 64], &device);
        let target = Tensor::<NdArray, 4>::zeros([1, 3, 64, 64], &device);
        let loss = l1_loss(pred, target);
        assert_eq!(loss.dims().len(), 1);
        let val = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!((val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rgb_to_tensor_shape() {
        let device = Default::default();
        let img = RgbImage::new(64, 64);
        let tensor = rgb_to_tensor::<NdArray>(&img, &device);
        assert_eq!(tensor.dims(), [1, 3, 64, 64]);
    }
}