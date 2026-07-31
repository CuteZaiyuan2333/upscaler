use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::Backend,
    tensor::{Tensor, activation},
};

// 轻量残差块：Conv -> ReLU -> Conv -> 残差相加。
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    pub channels: usize,
}

impl ResBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResBlock<B> {
        let c = self.channels;
        ResBlock {
            conv1: Conv2dConfig::new([c, c], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([c, c], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
        }
    }
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = input.clone();
        let x = self.conv1.forward(input);
        let x = activation::relu(x);
        let x = self.conv2.forward(x);
        activation::relu(x + residual)
    }
}

// 双分支特征融合：concat _> 1x1 投影 -> 残差精炼。
#[derive(Module, Debug)]
pub struct FusionBlock<B: Backend> {
    project: Conv2d<B>,
    refine: ResBlock<B>,
}

#[derive(Config, Debug)]
pub struct FusionBlockConfig {
    pub main_channels: usize,
    pub ref_channels: usize,
    pub out_channels: usize,
}

impl FusionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FusionBlock<B> {
        let in_channels = self.main_channels + self.ref_channels;
        FusionBlock {
            project: Conv2dConfig::new([in_channels, self.out_channels], [1, 1]).init(device),
            refine: ResBlockConfig::new(self.out_channels).init(device),
        }
    }
}

impl<B: Backend> FusionBlock<B> {
    pub fn forward(&self, main: Tensor<B, 4>, reference: Tensor<B, 4>) -> Tensor<B, 4> {
        let fused = Tensor::cat(vec![main, reference], 1);
        let fused = self.project.forward(fused);
        self.refine.forward(fused)
    }
}

// 2x 上采样阶段：学习型转置卷积 → 卷积平滑 → 与参考 skip 融合。
//
// 使用 ConvTranspose2d（可学习上采样）替代 Interpolate2d（固定插值），
// 使模型能学会从低分辨率特征生成高频细节，而非只是复制/平均像素。
#[derive(Module, Debug)]
pub struct UpsampleStage<B: Backend> {
    up: ConvTranspose2d<B>,
    smooth: Conv2d<B>,
    fusion: FusionBlock<B>,
}

#[derive(Config, Debug)]
pub struct UpsampleStageConfig {
    pub channels: usize,
    pub ref_channels: usize,
}

impl UpsampleStageConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpsampleStage<B> {
        let c = self.channels;
        UpsampleStage {
            // kernel=4, stride=2, padding=1 实现精确 2× 上采样
            up: ConvTranspose2dConfig::new([c, c], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_bias(true)
                .init(device),
            smooth: Conv2dConfig::new([c, c], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            fusion: FusionBlockConfig::new(c, self.ref_channels, c).init(device),
        }
    }
}

impl<B: Backend> UpsampleStage<B> {
    pub fn forward(
        &self,
        features: Tensor<B, 4>,
        reference: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let upsampled = self.up.forward(features);
        let upsampled = activation::relu(upsampled);
        let upsampled = self.smooth.forward(upsampled);
        self.fusion.forward(upsampled, reference)
    }
}
