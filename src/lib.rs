// 参考引导的条件上采样模型。
// 接收 Flux2 编辑后的低分辨率主图与原始高分辨率参考图，
// 输出保留编辑语义、并从参考图恢复高频细节的成品图。

pub mod augmentation;
pub mod model;
pub mod tensor_image;
pub mod training;
pub mod tui;

pub use model::{RefGuidedUpsampler, RefGuidedUpsamplerConfig};
pub use tensor_image::{ImageTensorError, load_rgb_tensor, rgb_image_to_tensor, save_rgb_tensor};
pub use training::TrainingEvent;