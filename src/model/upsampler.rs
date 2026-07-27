use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
    },
    prelude::Backend,
    tensor::{Tensor, activation},
};

use super::blocks::{
    DepthwiseSeparableConv, DepthwiseSeparableConvConfig, FusionBlock, FusionBlockConfig,
    ResBlock, ResBlockConfig, UpsampleStage, UpsampleStageConfig,
};
use super::presence_gate::{PresenceGate, PresenceGateConfig};

// 参考引导条件上采样器。

#[derive(Module, Debug)]
pub struct RefGuidedUpsampler<B: Backend> {
    main_stem: Conv2d<B>,
    main_encoder: MainEncoder<B>,
    ref_stem: Conv2d<B>,
    ref_encoder: RefEncoder<B>,
    fusion_bottleneck: FusionBlock<B>,
    upsample_to_2048: UpsampleStage<B>,
    upsample_to_4096: UpsampleStage<B>,
    detail_head: Conv2d<B>,
    presence_gate: PresenceGate<B>,
    upsample_2x: Interpolate2d,
    upsample_4x: Interpolate2d,
}

#[derive(Module, Debug)]
struct MainEncoder<B: Backend> {
    blocks: Vec<ResBlock<B>>,
    tail: Conv2d<B>,
}

#[derive(Module, Debug)]
struct RefEncoder<B: Backend> {
    // 4096 尺度浅层特征（仅 2 层 depthwise，控制显存）
    full_res_blocks: Vec<DepthwiseSeparableConv<B>>,
    // 4096 → 2048
    pool_to_2048: Conv2d<B>,
    blocks_2048: Vec<ResBlock<B>>,
    // 2048 → 1024
    pool_to_1024: Conv2d<B>,
    blocks_1024: Vec<ResBlock<B>>,
}

#[derive(Config, Debug)]
pub struct RefGuidedUpsamplerConfig {
    #[config(default = 64)]
    pub base_channels: usize,
    #[config(default = 4)]
    pub num_res_blocks: usize,
    #[config(default = 32)]
    pub gate_hidden_channels: usize,
}

impl RefGuidedUpsamplerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RefGuidedUpsampler<B> {
        let c = self.base_channels;
        let c2 = c * 2;

        RefGuidedUpsampler {
            main_stem: Conv2dConfig::new([3, c], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            main_encoder: MainEncoder {
                blocks: (0..self.num_res_blocks)
                    .map(|_| ResBlockConfig::new(c).init(device))
                    .collect(),
                tail: Conv2dConfig::new([c, c], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
            },
            ref_stem: Conv2dConfig::new([3, c], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            ref_encoder: RefEncoder {
                full_res_blocks: (0..2)
                    .map(|_| DepthwiseSeparableConvConfig::new(c, 3).init(device))
                    .collect(),
                pool_to_2048: Conv2dConfig::new([c, c2], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                blocks_2048: (0..self.num_res_blocks)
                    .map(|_| ResBlockConfig::new(c2).init(device))
                    .collect(),
                pool_to_1024: Conv2dConfig::new([c2, c2], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                blocks_1024: (0..self.num_res_blocks)
                    .map(|_| ResBlockConfig::new(c2).init(device))
                    .collect(),
            },
            fusion_bottleneck: FusionBlockConfig::new(c, c2, c).init(device),
            upsample_to_2048: UpsampleStageConfig::new(c, c2).init(device),
            upsample_to_4096: UpsampleStageConfig::new(c, c).init(device),
            detail_head: Conv2dConfig::new([c, 3], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            presence_gate: PresenceGateConfig::new(self.gate_hidden_channels).init(device),
            // 使用 Nearest 模式而非 Linear：CubeCL/WGPU 后端不支持双线性插值的反向传播。
            // UpsampleStage 中的 smooth 卷积层会补偿最近邻上采样带来的块状伪影。
            upsample_2x: Interpolate2dConfig::new()
                .with_scale_factor(Some([2.0, 2.0]))
                .with_mode(InterpolateMode::Nearest)
                .init(),
            upsample_4x: Interpolate2dConfig::new()
                .with_scale_factor(Some([4.0, 4.0]))
                .with_mode(InterpolateMode::Nearest)
                .init(),
        }
    }
}

impl<B: Backend> MainEncoder<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = input;
        for block in &self.blocks {
            x = block.forward(x);
        }
        self.tail.forward(x)
    }
}

impl<B: Backend> RefEncoder<B> {
    // 返回 (ref_f0@4096, ref_f1@2048, ref_f2@1024) 三级 skip 特征。
    fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let mut f0 = input;
        for block in &self.full_res_blocks {
            f0 = block.forward(f0);
        }

        let mut f1 = self.pool_to_2048.forward(f0.clone());
        f1 = activation::relu(f1);
        for block in &self.blocks_2048 {
            f1 = block.forward(f1);
        }

        let mut f2 = self.pool_to_1024.forward(f1.clone());
        f2 = activation::relu(f2);
        for block in &self.blocks_1024 {
            f2 = block.forward(f2);
        }

        (f0, f1, f2)
    }
}

impl<B: Backend> RefGuidedUpsampler<B> {
    // 前向推理。
    //
    // main：[B, 3, 1024, 1024] Flux2 编辑后的主图
    // reference: [B, 3, 4096, 4096] 原始参考图（与主图同场景、像素对齐）
    //
    // 返回 [B, 3, 4096, 4096] 成品图。
    pub fn forward(&self, main: Tensor<B, 4>, reference: Tensor<B, 4>) -> Tensor<B, 4> {
        let main_up = self.upsample_4x.forward(main.clone());

        // 主图编码 @1024
        let main_feat = activation::relu(self.main_stem.forward(main));
        let main_feat = self.main_encoder.forward(main_feat);

        // 参考图多尺度编码
        let ref_feat = self.ref_stem.forward(reference.clone());
        let (ref_f0, ref_f1, ref_f2) = self.ref_encoder.forward(ref_feat);

        // 瓶颈融合 @1024
        let mut feat = self.fusion_bottleneck.forward(main_feat, ref_f2);

        // 渐进上采样 + 参考 skip
        feat = self
            .upsample_to_2048
            .forward(feat, ref_f1, &self.upsample_2x);
        feat = self
            .upsample_to_4096
            .forward(feat, ref_f0, &self.upsample_2x);

        let delta = self.detail_head.forward(feat);
        let gate = self.presence_gate.forward(main_up.clone(), reference);

        clamp01(main_up + gate * delta)
    }
}

// 将 RGB 张量限制在 [0, 1]
fn clamp01<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    activation::relu(tensor).clamp_max(1.0)
}

// 训练损失建议（供训练脚本参考，非 Module 一部分）
#[allow(dead_code)]
pub struct TrainingLossHints {
    pub l1_weight: f32,
    pub perceptual_weight: f32,
    pub gate_supervision_weight: f32,
}

impl Default for TrainingLossHints {
    fn default() -> Self {
        Self {
            l1_weight: 1.0,
            perceptual_weight: 0.1,
            gate_supervision_weight: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //use burn::backend::NdArray;
    use burn::backend::Wgpu;
    use burn::prelude::Module;

    //type TestBackend = NdArray;
    type TestBackend = Wgpu;

    #[test]
    #[ignore = "全分辨率 4096 前向在 CPU 上较慢，使用 cargo test -- --ignored 运行"]
    fn forward_output_shape_is_4096() {
        let device = Default::default();
        let model = RefGuidedUpsamplerConfig::new().init::<TestBackend>(&device);

        let main = Tensor::<TestBackend, 4>::zeros([1, 3, 1024, 1024], &device);
        let reference = Tensor::<TestBackend, 4>::zeros([1, 3, 4096, 4096], &device);

        let output = model.forward(main, reference);
        assert_eq!(output.dims(), [1, 3, 4096, 4096]);
    }

    #[test]
    fn model_is_lightweight() {
        let device = Default::default();
        let model = RefGuidedUpsamplerConfig::new().init::<TestBackend>(&device);
        let params = model.num_params();
        // 默认配置应控制在 ~5M 参数以内
        assert!(params < 5_000_000, "params={params}");
    }
}
