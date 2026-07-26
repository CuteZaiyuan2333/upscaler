// use burn::backend::NdArray;
// use burn::backend::Cuda
use burn::backend::Wgpu;
use burn::prelude::Module;
use upscaler::RefGuidedUpsamplerConfig;

// type Backend = NdArray;
// tyoe Backend = Cuda;
type Backend = Wgpu;

fn main() {
    let device = Default::default();
    let config = RefGuidedUpsamplerConfig::new();
    let model = config.init::<Backend>(&device);

    println!("=== RefGuidedUpsampler ===");
    println!("输入: main [1,3,1024,1024] + reference [1,3,4096,4096]");
    println!("输出: [1,3,4096,4096]");
    println!("参数量: {}", model.num_params());
    println!();
    println!("配置: base_channels={}, num_res_blocks={}",
        config.base_channels, config.num_res_blocks);
    println!();
    println!("推理示例（需准备 examples/main_1024.png 与 examples/ref_4096.png）:");
    println!("  见 upscaler::load_rgb_tensor / save_rgb_tensor API");
}
