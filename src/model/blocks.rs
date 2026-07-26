use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        interpolate::Interpolate2d,
    },
    prelude::Backend,
    tensor::{Tensor, activation},
};

// 轻量残差块：Conv → ReLU → Conv → 残差相加。
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

// 双分支特征融合：concat → 1×1 投影 → 残差精炼。
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

// 深度可分离卷积：逐通道空间卷积 + 1×1 点卷积，用于 4K 参考图的高效浅层处理。。
#[derive(Module, Debug)]
pub struct DepthwiseSeparableConv<B: Backend> {
    depthwise: Conv2d<B>,
    pointwise: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct DepthwiseSeparableConvConfig {
    pub channels: usize,
    pub kernel_size: usize,
}

impl DepthwiseSeparableConvConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DepthwiseSeparableConv<B> {
        let k = self.kernel_size;
        DepthwiseSeparableConv {
            depthwise: Conv2dConfig::new([self.channels, self.channels], [k, k])
                .with_groups(self.channels)
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pointwise: Conv2dConfig::new([self.channels, self.channels], [1, 1]).init(device),
        }
    }
}

impl<B: Backend> DepthwiseSeparableConv<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.depthwise.forward(input);
        let x = activation::relu(x);
        self.pointwise.forward(x)
    }
}

// 2× 上采样阶段：双线性插值 → 卷积平滑 → 与参考 skip 融合。
#[derive(Module, Debug)]
pub struct UpsampleStage<B: Backend> {
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
        UpsampleStage {
            smooth: Conv2dConfig::new([self.channels, self.channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            fusion: FusionBlockConfig::new(self.channels, self.ref_channels, self.channels).init(device),
        }
    }
}

impl<B: Backend> UpsampleStage<B> {
    pub fn forward(
        &self,
        features: Tensor<B, 4>,
        reference: Tensor<B, 4>,
        upsample: &Interpolate2d,
    ) -> Tensor<B, 4> {
        let upsampled = upsample.forward(features);
        let upsampled = self.smooth.forward(upsampled);
        self.fusion.forward(upsampled, reference)
    }
}
