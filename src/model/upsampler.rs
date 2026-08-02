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
    FusionBlock, FusionBlockConfig,
    ResBlock, ResBlockConfig, UpsampleStage, UpsampleStageConfig,
};

//参考引导条件上采样器

#[derive(Module, Debug)]
pub struct RefGuidedUpsampler<B: Backend> {
    main_stem: Conv2d<B>,
    main_encoder: MainEncoder<B>,
    ref_stem: Conv2d<B>,
    ref_encoder: RefEncoder<B>,
    fusion_bottleneck: FusionBlock<B>,
    upsample_to_2048: UpsampleStage<B>,
    upsample_to_4096: UpsampleStage<B>,
    detail_head: DetailHead<B>,
    /// 仅用于主图初始 4× 上采样（Bilinear 提供平滑基线）
    upsample_4x: Interpolate2d,
}

// 细节生成头：3 层卷积解码器
// 替代原来的单层 Conv2d，给模型足够容量从特征生成高频细节。

#[derive(Module, Debug)]
struct DetailHead<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
}

impl<B: Backend> DetailHead<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = activation::relu(self.conv1.forward(input));
        let x = activation::relu(self.conv2.forward(x));
        self.conv3.forward(x) // 无激活，让梯度自由流动
    }
}

//编码器 

#[derive(Module, Debug)]
struct MainEncoder<B: Backend> {
    blocks: Vec<ResBlock<B>>,
    tail: Conv2d<B>,
}

#[derive(Module, Debug)]
struct RefEncoder<B: Backend> {
    // 全分辨率特征：单层 Conv + ReLU（比 depthwise 强，比 ResBlock 轻）
    full_res_conv: Conv2d<B>,
    // 下采样到 1/2 分辨率
    pool_to_2048: Conv2d<B>,
    blocks_2048: Vec<ResBlock<B>>,
    // 下采样到 1/4 分辨率
    pool_to_1024: Conv2d<B>,
    blocks_1024: Vec<ResBlock<B>>,
}

//配置

#[derive(Config, Debug)]
pub struct RefGuidedUpsamplerConfig {
    #[config(default = 64)]
    pub base_channels: usize,
    #[config(default = 4)]
    pub num_res_blocks: usize,
}

impl RefGuidedUpsamplerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RefGuidedUpsampler<B> {
        let c = self.base_channels;
        let c2 = c * 2;
        let c_half = c / 2;

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
                // 全分辨率使用单层 Conv（比 depthwise 强，比 ResBlock 轻量）
                full_res_conv: Conv2dConfig::new([c, c], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
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
            detail_head: DetailHead {
                conv1: Conv2dConfig::new([c, c_half], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                conv2: Conv2dConfig::new([c_half, c_half], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                conv3: Conv2dConfig::new([c_half, 3], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
            },
            upsample_4x: Interpolate2dConfig::new()
                .with_scale_factor(Some([4.0, 4.0]))
                .with_mode(InterpolateMode::Linear)
                .init(),
        }
    }
}

//编码器 forward

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
    /// 返回 (ref_f0, ref_f1, ref_f2) 三级 skip 特征。
    fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let f0 = activation::relu(self.full_res_conv.forward(input));

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

//主 forward

impl<B: Backend> RefGuidedUpsampler<B> {
    // 前向推理。
    //
    // main：[B, 3, H, W] 主图（如 256×256）
    // reference: [B, 3, 4H, 4W] 参考图（如 1024×1024）
    //
    // 返回 [B, 3, 4H, 4W] 成品图。
    //
    // 公式：output = main_up + delta
    // 不做 clamp/sigmoid/tanh，让训练梯度自由流动。
    // 推理时 save_rgb_tensor 会 clamp 到 [0, 1]。
    pub fn forward(&self, main: Tensor<B, 4>, reference: Tensor<B, 4>) -> Tensor<B, 4> {
        // 主图 4× bilinear 上采样（平滑基线）
        let main_up = self.upsample_4x.forward(main.clone());

        // 主图编码
        let main_feat = activation::relu(self.main_stem.forward(main));
        let main_feat = self.main_encoder.forward(main_feat);

        // 参考图多尺度编码
        let ref_feat = self.ref_stem.forward(reference.clone());
        let (ref_f0, ref_f1, ref_f2) = self.ref_encoder.forward(ref_feat);

        // 瓶颈融合
        let mut feat = self.fusion_bottleneck.forward(main_feat, ref_f2);

        // 渐进上采样 + 参考 skip
        feat = self.upsample_to_2048.forward(feat, ref_f1);
        feat = self.upsample_to_4096.forward(feat, ref_f0);

        // 细节生成（无激活约束，梯度自由流动）
        let delta = self.detail_head.forward(feat);

        // 不做 clamp/sigmoid——训练时 L1 loss 会引导输出接近 [0,1]，
        // 推理时 save_rgb_tensor 做 clamp
        main_up + delta
    }
}

//测试

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::prelude::Module;

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
        // 默认配置应控制在 5M 参数以内
        assert!(params < 5_000_000, "params={params}");
    }
}