// 参考引导的条件上采样模型
// 接收 Flux2 编辑后的 1024×1024 主图与原始 4096×4096 参考图
// 输出保留编辑语义、并从参考图恢复高频细节的 4096×4096 成品图

pub mod augmentation;
pub mod model;
pub mod tensor_image;
pub mod training;

pub use model::{
    RefGuidedUpsampler,
    RefGuidedUpsamplerConfig,
};
pub use tensor_image::{
    ImageTensorError,
    load_rgb_tensor,
    rgb_image_to_tensor,
    save_rgb_tensor,
};
