// 相机滤镜数据增强模块。
//
// 对 1024×1024 主图施加各种相机滤镜效果，模拟 Flux2 编辑后的图像特征变化，
// 同时保持 4096×4096 参考图不变。模型需要学习从参考图中恢复原始细节，
// 而非简单地复制参考图。
//
// 所有滤镜基于image库的 RgbImage 操作，无需额外依赖。

use image::{Rgb, RgbImage};
use rand::Rng;

// 相机滤镜类型。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraFilter {
    // 原图（不处理）
    Original,
    // 暖色调：增加红/黄色温
    Warm,
    // 冷色调：增加蓝色温
    Cool,
    // 鲜艳：提高饱和度 + 对比度
    Vivid,
    // 褪色：降低对比度，略微提亮
    Fade,
    // 黑白：去饱和
    Mono,
    // 高对比度黑白
    Noir,
    // 铬黄：高饱和 + 高对比
    Chrome,
    // 老旧照片：棕褐色调 + 褪色
    Sepia,
    // 玩具相机：暗角 + 轻微模糊
    ToyCamera,
    // 胶片颗粒：添加随机噪声
    Grain,
    // 柔焦：轻微模糊
    SoftFocus,
    // 曝光不足
    Underexpose,
    // 曝光过度
    Overexpose,
    // 反转色
    Invert,
    // 海报化：减少颜色层次
    Posterize,
}

impl CameraFilter {
    // 所有滤镜的列表。
    pub fn all() -> &'static [CameraFilter] {
        &[
            CameraFilter::Original,
            CameraFilter::Warm,
            CameraFilter::Cool,
            CameraFilter::Vivid,
            CameraFilter::Fade,
            CameraFilter::Mono,
            CameraFilter::Noir,
            CameraFilter::Chrome,
            CameraFilter::Sepia,
            CameraFilter::ToyCamera,
            CameraFilter::Grain,
            CameraFilter::SoftFocus,
            CameraFilter::Underexpose,
            CameraFilter::Overexpose,
            CameraFilter::Invert,
            CameraFilter::Posterize,
        ]
    }
}

// 对图像应用指定的相机滤镜。
pub fn apply_filter(img: &RgbImage, filter: CameraFilter) -> RgbImage {
    match filter {
        CameraFilter::Original => img.clone(),
        CameraFilter::Warm => warm(img),
        CameraFilter::Cool => cool(img),
        CameraFilter::Vivid => vivid(img),
        CameraFilter::Fade => fade(img),
        CameraFilter::Mono => mono(img),
        CameraFilter::Noir => noir(img),
        CameraFilter::Chrome => chrome(img),
        CameraFilter::Sepia => sepia(img),
        CameraFilter::ToyCamera => toy_camera(img),
        CameraFilter::Grain => grain(img),
        CameraFilter::SoftFocus => soft_focus(img),
        CameraFilter::Underexpose => exposure(img, 0.5),
        CameraFilter::Overexpose => exposure(img, 1.8),
        CameraFilter::Invert => invert(img),
        CameraFilter::Posterize => posterize(img, 4),
    }
}

//随机选择一个滤镜并应用，有概率叠加多个滤镜。
pub fn apply_random_filter(rng: &mut impl Rng, img: &RgbImage) -> RgbImage {
    let all = CameraFilter::all();
    let idx = rng.gen_range(0..all.len());
    let primary = all[idx];

    let mut result = apply_filter(img, primary);

    // 30% 概率叠加第二个轻量滤镜
    if rng.gen_bool(0.3) {
        let secondary = match rng.gen_range(0..4) {
            0 => CameraFilter::Grain,
            1 => CameraFilter::SoftFocus,
            2 => CameraFilter::Vivid,
            _ => CameraFilter::Fade,
        };
        result = apply_filter(&result, secondary);
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// 基础像素操作
// ═══════════════════════════════════════════════════════════════════════════════

// 将 RGB 像素转换为 HSL 分量。
// 返回 (h: 0-360, s: 0.0-1.0, l: 0.0-1.0)
fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;

    let l = (max + min) / 2.0;

    if delta < 1e-5 {
        return (0.0, 0.0, l);
    }

    let s = if l > 0.5 {
        delta / (2.0 - max - min)
    } else {
        delta / (max + min)
    };

    let h = if (max - rf).abs() < 1e-5 {
        ((gf - bf) / delta) % 6.0
    } else if (max - gf).abs() < 1e-5 {
        (bf - rf) / delta + 2.0
    } else {
        (rf - gf) / delta + 4.0
    };

    (h * 60.0, s, l)
}

/// 将 HSL 分量转换为 RGB 像素。
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let h = h % 360.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (rf, gf, bf) = match h as i32 {
        h if h < 60 => (c, x, 0.0),
        h if h < 120 => (x, c, 0.0),
        h if h < 180 => (0.0, c, x),
        h if h < 240 => (0.0, x, c),
        h if h < 300 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (
        ((rf + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((gf + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((bf + m) * 255.0).clamp(0.0, 255.0) as u8,
    )
}

// 遍历每个像素，应用 HSL 变换。
fn map_hsl(img: &RgbImage, f: impl Fn(f32, f32, f32) -> (f32, f32, f32)) -> RgbImage {
    let mut out = RgbImage::new(img.width(), img.height());
    for (x, y, pix) in img.enumerate_pixels() {
        let (h, s, l) = rgb_to_hsl(pix[0], pix[1], pix[2]);
        let (h2, s2, l2) = f(h, s, l);
        let (r, g, b) = hsl_to_rgb(h2, s2, l2);
        out.put_pixel(x, y, Rgb([r, g, b]));
    }
    out
}

// 滤镜实现

// 暖色调：色调向红/黄偏移，略微提亮。
fn warm(img: &RgbImage) -> RgbImage {
    let mut out = img.clone();
    for (_, _, pix) in out.enumerate_pixels_mut() {
        pix[0] = pix[0].saturating_add(15).min(255); // 加红
        pix[1] = pix[1].saturating_add(5).min(255); // 微加绿
        pix[2] = pix[2].saturating_sub(10); // 减蓝
    }
    out
}

// 冷色调：色调向蓝偏移。
fn cool(img: &RgbImage) -> RgbImage {
    let mut out = img.clone();
    for (_, _, pix) in out.enumerate_pixels_mut() {
        pix[0] = pix[0].saturating_sub(10); // 减红
        pix[1] = pix[1].saturating_sub(5); // 微减绿
        pix[2] = pix[2].saturating_add(15).min(255); // 加蓝
    }
    out
}

// 鲜艳：提高饱和度 50% + 提高对比度。
fn vivid(img: &RgbImage) -> RgbImage {
    let saturated = map_hsl(img, |h, s, l| (h, (s * 1.5).min(1.0), l));
    image::imageops::contrast(&saturated, 20.0)
}

// 褪色：降低对比度，略微提亮，模拟长时间曝光或滤镜效果。
fn fade(img: &RgbImage) -> RgbImage {
    let low_contrast = image::imageops::contrast(img, -30.0);
    image::imageops::brighten(&low_contrast, 20)
}

// 黑白：完全去饱和。
fn mono(img: &RgbImage) -> RgbImage {
    let gray = image::imageops::grayscale(img);
    image::DynamicImage::ImageLuma8(gray).to_rgb8()
}

// 高对比度黑白：去饱和 + 提高对比度。
fn noir(img: &RgbImage) -> RgbImage {
    let gray = image::imageops::grayscale(img);
    let contrast = image::imageops::contrast(&gray, 50.0);
    image::DynamicImage::ImageLuma8(contrast).to_rgb8()
}

// 铬黄：高饱和 + 高对比 + 微暖色调。
fn chrome(img: &RgbImage) -> RgbImage {
    let vivid = map_hsl(img, |h, s, l| (h, (s * 1.8).min(1.0), l));
    let contrast = image::imageops::contrast(&vivid, 30.0);
    let mut out = contrast;
    for (_, _, pix) in out.enumerate_pixels_mut() {
        pix[0] = pix[0].saturating_add(5).min(255);
    }
    out
}

// 棕褐色调：经典的老照片效果。
fn sepia(img: &RgbImage) -> RgbImage {
    let mut out = RgbImage::new(img.width(), img.height());
    for (x, y, pix) in img.enumerate_pixels() {
        let r = pix[0] as f32;
        let g = pix[1] as f32;
        let b = pix[2] as f32;

        let out_r = (r * 0.393 + g * 0.769 + b * 0.189).min(255.0) as u8;
        let out_g = (r * 0.349 + g * 0.686 + b * 0.168).min(255.0) as u8;
        let out_b = (r * 0.272 + g * 0.534 + b * 0.131).min(255.0) as u8;

        out.put_pixel(x, y, Rgb([out_r, out_g, out_b]));
    }
    out
}

// 玩具相机：暗角 + 轻微模糊 + 略微提亮中心。
fn toy_camera(img: &RgbImage) -> RgbImage {
    let blurred = image::imageops::blur(img, 0.5);
    vignette(&blurred, 0.4)
}

// 胶片颗粒：添加随机噪声。
fn grain(img: &RgbImage) -> RgbImage {
    let mut rng = rand::thread_rng();
    let mut out = img.clone();
    for (_, _, pix) in out.enumerate_pixels_mut() {
        let noise: i16 = rng.gen_range(-15..=15);
        for c in 0..3 {
            let val = pix[c] as i16 + noise;
            pix[c] = val.clamp(0, 255) as u8;
        }
    }
    out
}

// 柔焦：轻微高斯模糊。
fn soft_focus(img: &RgbImage) -> RgbImage {
    image::imageops::blur(img, 1.0)
}

// 曝光调整：乘以一个系数。
fn exposure(img: &RgbImage, factor: f32) -> RgbImage {
    let mut out = img.clone();
    for (_, _, pix) in out.enumerate_pixels_mut() {
        for c in 0..3 {
            let val = (pix[c] as f32 * factor).clamp(0.0, 255.0);
            pix[c] = val as u8;
        }
    }
    out
}

// 反转色。
fn invert(img: &RgbImage) -> RgbImage {
    let mut out = img.clone();
    image::imageops::invert(&mut out);
    out
}

// 海报化：减少颜色层次。
fn posterize(img: &RgbImage, levels: u8) -> RgbImage {
    let step = 255.0 / (levels - 1) as f32;
    let mut out = img.clone();
    for (_, _, pix) in out.enumerate_pixels_mut() {
        for c in 0..3 {
            let val = (pix[c] as f32 / step).round() * step;
            pix[c] = val.clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// 暗角效果：从中心向外逐渐变暗。
fn vignette(img: &RgbImage, strength: f32) -> RgbImage {
    let mut out = img.clone();
    let (w, h) = out.dimensions();
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();

    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let factor = 1.0 - strength * (dist / max_dist).powi(2);
            let pix = out.get_pixel_mut(x, y);
            for c in 0..3 {
                pix[c] = (pix[c] as f32 * factor).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

// 测试

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image() -> RgbImage {
        let mut img = RgbImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let r = (x * 4) as u8;
                let g = (y * 4) as u8;
                let b = 128u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        img
    }

    #[test]
    fn test_all_filters_output_correct_size() {
        let img = make_test_image();
        for filter in CameraFilter::all() {
            let result = apply_filter(&img, *filter);
            assert_eq!(
                result.dimensions(),
                (64, 64),
                "滤镜 {:?} 输出尺寸错误",
                filter
            );
        }
    }

    #[test]
    fn test_rgb_hsl_roundtrip() {
        // 测试 RGB→HSL→RGB 往返一致性
        let test_cases = [
            (255, 0, 0),     // 纯红
            (0, 255, 0),     // 纯绿
            (0, 0, 255),     // 纯蓝
            (128, 128, 128), // 中灰
            (255, 255, 255), // 白
            (0, 0, 0),       // 黑
            (200, 100, 50),  // 橙色
        ];

        for (r, g, b) in test_cases {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let (r2, g2, b2) = hsl_to_rgb(h, s, l);
            // 允许 ±1 的舍入误差
            assert!(
                (r as i32 - r2 as i32).abs() <= 1
                    && (g as i32 - g2 as i32).abs() <= 1
                    && (b as i32 - b2 as i32).abs() <= 1,
                "RGB→HSL→RGB 往返失败: ({r},{g},{b}) → ({r2},{g2},{b2})"
            );
        }
    }

    #[test]
    fn test_exposure() {
        let img = make_test_image();
        let dark = exposure(&img, 0.5);
        let bright = exposure(&img, 2.0);

        // 变暗后，每个像素值应 ≤ 原值
        for (dp, op) in dark.pixels().zip(img.pixels()) {
            for c in 0..3 {
                assert!(dp[c] <= op[c], "曝光不足应使像素变暗");
            }
        }
        // 变亮后，每个像素值应 ≥ 原值（除非已饱和）
        for (bp, op) in bright.pixels().zip(img.pixels()) {
            for c in 0..3 {
                if op[c] < 127 {
                    assert!(bp[c] >= op[c], "曝光过度应使像素变亮");
                }
            }
        }
    }

    #[test]
    fn test_grayscale_is_actually_gray() {
        let img = make_test_image();
        let mono = mono(&img);
        for pix in mono.pixels() {
            assert_eq!(pix[0], pix[1], "R=G");
            assert_eq!(pix[1], pix[2], "G=B");
        }
    }
}
