use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use image::{DynamicImage, Rgb, RgbImage};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ImageTensorError {
    #[error("failed to open image: {0}")]
    Io(#[from] image::ImageError),
    #[error("expected RGB image, got {0:?}")]
    NotRgb(image::ColorType),
    #[error("image size mismatch: expected {expected_w}x{expected_h}, got {actual_w}x{actual_h}")]
    SizeMismatch {
        expected_w: u32,
        expected_h: u32,
        actual_w: u32,
        actual_h: u32,
    },
}

// 将 RGB 图像加载为 [1, 3, H, W]浮点张量，像素值归一化到 [0, 1]
pub fn load_rgb_tensor<B: Backend>(
    path: impl AsRef<Path>,
    expected_size: Option<(u32, u32)>,
    device: &B::Device,
) -> Result<Tensor<B, 4>, ImageTensorError> {
    let img = image::open(path)?;
    let rgb = to_rgb8(&img)?;

    if let Some((ew, eh)) = expected_size {
        let (aw, ah) = rgb.dimensions();
        if aw != ew || ah != eh {
            return Err(ImageTensorError::SizeMismatch {
                expected_w: ew,
                expected_h: eh,
                actual_w: aw,
                actual_h: ah,
            });
        }
    }

    Ok(rgb_image_to_tensor(rgb, device))
}
// 将 [1, 3, H, W] 张量保存为 8bit RGB PNG
pub fn save_rgb_tensor<B: Backend>(
    tensor: Tensor<B, 4>,
    path: impl AsRef<Path>,
) -> Result<(), ImageTensorError> {
    let data = tensor.into_data();
    let shape = data.shape.dims();
    let [batch, channels, height, width] = shape;
    if batch != 1 || channels != 3 {
        return Err(ImageTensorError::SizeMismatch {
            expected_w: 1,
            expected_h: 3,
            actual_w: batch as u32,
            actual_h: channels as u32,
        });
    }

    let values = data
        .to_vec::<f32>()
        .expect("tensor should contain f32 values");
    let mut img = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * 3;
            let r = (values[base].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (values[base + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (values[base + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    img.save(path)?;
    Ok(())
}

fn to_rgb8(img: &DynamicImage) -> Result<RgbImage, ImageTensorError> {
    match img.color() {
        image::ColorType::Rgb8 => Ok(img.to_rgb8()),
        other => Err(ImageTensorError::NotRgb(other)),
    }
}

fn rgb_image_to_tensor<B: Backend>(img: RgbImage, device: &B::Device) -> Tensor<B, 4> {
    let (width, height) = img.dimensions();
    let pixels = img.into_raw();
    let mut data = Vec::with_capacity(pixels.len());

    // HWC -》 CHW
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3 + c) as usize;
                data.push(pixels[idx] as f32 / 255.0);
            }
        }
    }

    let tensor_data = TensorData::new(data, [1, 3, height as usize, width as usize]);
    Tensor::from_data(tensor_data, device)
}
