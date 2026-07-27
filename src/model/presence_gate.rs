use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::{Tensor, activation},
};

// 存在性门控网络。
// 
// 决定每个像素是否从参考图借用高频细节：
// 门控接近1：主图该区域有有效内容，可结合参考图恢复纹理
// 门控接近0：主图刻意消除了该结构（如支撑架），忽略参考图对应细节
//
// 输入通道（7）：
// main_up (3)：双线性上采样后的主图
// reference (3)：参考图
// luminance_diff (1)：亮度差异 |Y(main_up) - Y(ref)|，帮助区分是否是缺细节但结构一致还是结构被刻意移除
#[derive(Module, Debug)]
pub struct PresenceGate<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct PresenceGateConfig {
    pub hidden_channels: usize,
}

impl PresenceGateConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PresenceGate<B> {
        let h = self.hidden_channels;
        PresenceGate {
            conv1: Conv2dConfig::new([7, h], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([h, h], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv3: Conv2dConfig::new([h, 1], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
        }
    }
}

impl<B: Backend> PresenceGate<B> {
    pub fn forward(&self, main_up: Tensor<B, 4>, reference: Tensor<B, 4>) -> Tensor<B, 4> {
        let luminance_diff = luminance_abs_diff(&main_up, &reference);
        let input = Tensor::cat(vec![main_up, reference, luminance_diff], 1);

        let x = self.conv1.forward(input);
        let x = activation::relu(x);
        let x = self.conv2.forward(x);
        let x = activation::relu(x);
        activation::sigmoid(self.conv3.forward(x))
    }
}

// ITU-R BT.601 亮度近似，用于结构差异度量。
fn luminance_abs_diff<B: Backend>(a: &Tensor<B, 4>, b: &Tensor<B, 4>) -> Tensor<B, 4> {
    let ya = rgb_to_luminance(a.clone());
    let yb = rgb_to_luminance(b.clone());
    (ya - yb).abs()
}

fn rgb_to_luminance<B: Backend>(rgb: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, _, height, width] = rgb.dims();
    let r = rgb.clone().slice([0..batch, 0..1, 0..height, 0..width]);
    let g = rgb.clone().slice([0..batch, 1..2, 0..height, 0..width]);
    let b = rgb.slice([0..batch, 2..3, 0..height, 0..width]);
    r.mul_scalar(0.299) + g.mul_scalar(0.587) + b.mul_scalar(0.114)
}
