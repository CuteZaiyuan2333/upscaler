# Upscaler — 参考引导的条件图像上采样器

> **本文档由 AI 辅助生成，内容基于代码库实际实现。如存在歧义，请以代码为准。**

---

## 项目概述

本项目实现了一个**参考图像引导的条件超分辨率神经网络**（`RefGuidedUpsampler`），用于将 Flux2 图像编辑管线的 1024×1024 编辑结果放大至 4096×4096，同时从原始高清参考图中恢复高频纹理细节。

**核心思路：** 编辑模型（Flux2）在低分辨率下操作效果最佳，但编辑后的图像需要放大到原始分辨率。简单的双线性上采样会导致细节丢失。本模型通过同时参考编辑结果（主图）和原始高分辨率照片（参考图），在放大的同时智能地恢复丢失的纹理信息。

---

## 设计动机

### 问题定义

```
输入：
  主图 (main)      [1, 3, 1024, 1024]  ← Flux2 编辑输出，保留编辑语义
  参考图 (reference) [1, 3, 4096, 4096]  ← 原始高分辨率照片，保留高频细节

输出：
  成品图            [1, 3, 4096, 4096]  ← 编辑语义 + 高频纹理
```

### 关键挑战

1. **信息不对称**：主图包含编辑后的内容（如移除物体、修改场景），参考图包含原始内容。直接融合会引入不应存在的细节。
2. **分辨率差距**：主图仅 1024px，需放大 4 倍至 4096px，传统方法会产生模糊。
3. **显存约束**：在 4096×4096 分辨率上进行全分辨率卷积运算会消耗大量显存。

### 解决方案

- **多尺度参考编码**：在 4096、2048、1024 三个尺度提取参考图特征，逐级融合。
- **存在性门控**：通过一个轻量门控网络判断每个像素是否应从参考图借用细节，避免将参考图中已被编辑移除的内容混入输出。
- **深度可分离卷积**：在 4096 全分辨率路径使用 depthwise-separable conv 控制显存占用。

---

## 模型架构

```
RefGuidedUpsampler
│
├─ MainEncoder
│   ├─ stem: Conv2d [3 → 64, 3×3]
│   ├─ 4 × ResBlock [64 channels]
│   └─ tail: Conv2d [64 → 64, 3×3]
│
├─ RefEncoder
│   ├─ ref_stem: Conv2d [3 → 64, 3×3]
│   │
│   ├─ @4096 (full_res)
│   │   └─ 2 × DepthwiseSeparableConv [64, kernel=3]
│   │
│   ├─ @2048 (mid_res)
│   │   ├─ pool_to_2048: Conv2d [64 → 128, stride=2]
│   │   └─ 4 × ResBlock [128 channels]
│   │
│   └─ @1024 (low_res)
│       ├─ pool_to_1024: Conv2d [128 → 128, stride=2]
│       └─ 4 × ResBlock [128 channels]
│
├─ FusionBlock (瓶颈融合 @1024)
│   ├─ concat(main_feat_64, ref_f2_128) → 192 channels
│   ├─ 1×1 投影 [192 → 64]
│   └─ ResBlock 精炼 [64]
│
├─ UpsampleStage × 2 (渐进上采样)
│   ├─ @1024 → @2048: bilinear 2× + conv 平滑 + fusion(ref_f1)
│   └─ @2048 → @4096: bilinear 2× + conv 平滑 + fusion(ref_f0)
│
├─ Detail Head
│   └─ Conv2d [64 → 3, 3×3] → 输出 delta (RGB 残差)
│
├─ PresenceGate (存在性门控)
│   ├─ 输入: [main_up_rgb(3), reference_rgb(3), luminance_diff(1)] = 7 channels
│   ├─ Conv2d [7 → 32, 3×3] → ReLU
│   ├─ Conv2d [32 → 32, 3×3] → ReLU
│   └─ Conv2d [32 → 1, 3×3] → Sigmoid → per-pixel gate [0, 1]
│
└─ 输出: clamp01( bilinear_4×(main) + gate × delta )
```

### 关键设计决策

| 决策 | 原因 |
|---|---|
| 参考编码器使用 3 级 skip connection | 不同尺度的特征信息对上采样各有价值，低层纹理 vs 高层语义 |
| 4096 全分辨率路径仅用 2 层 depthwise-sep conv | 全分辨率下标准 conv 显存占用过大，depthwise-sep 是轻量替代 |
| 存在性门控引入亮度差异特征 | 区分"缺细节但结构一致"与"结构被刻意移除"两种情况 |
| 最终输出为 `upsample(main) + gate × delta` | 基线为双线性上采样主图，delta 仅在门控允许时叠加，保证最差情况不会比简单上采样更差 |
| `clamp01` 使用 `ReLU + clamp_max(1.0)` | 将输出限制在有效 RGB 范围内 |

---

## 模块说明

### `model/blocks.rs` — 可复用构建块

- **`ResBlock`**：标准残差块（Conv→ReLU→Conv + skip connection），用于特征精炼。
- **`FusionBlock`**：将两个不同来源的特征图 concat 后投影融合，再用 ResBlock 精炼。
- **`DepthwiseSeparableConv`**：深度可分离卷积（depthwise + pointwise），用于 4K 分辨率下的高效浅层处理。
- **`UpsampleStage`**：上采样阶段，执行双线性插值 → 卷积平滑 → 与参考 skip 融合。

### `model/presence_gate.rs` — 存在性门控网络

决定每个像素是否从参考图借用高频细节：

- **门控 ≈ 1**：主图该区域有有效内容，可结合参考图恢复纹理。
- **门控 ≈ 0**：主图刻意消除了该结构（如删除支撑架），忽略参考图对应细节。

亮度差异基于 ITU-R BT.601 标准计算：`Y = 0.299R + 0.587G + 0.114B`。

### `tensor_image.rs` — 图像 ↔ Tensor 工具

- `load_rgb_tensor`：加载 RGB 图像为 `[1, 3, H, W]` 浮点张量（归一化到 [0, 1]），可选验证尺寸。
- `save_rgb_tensor`：将 `[1, 3, H, W]` 张量保存为 8-bit RGB PNG。
- `ImageTensorError`：统一错误类型，包含 I/O、颜色类型、尺寸不匹配三种情况。

---

## 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `base_channels` | 64 | 基础特征通道数 |
| `num_res_blocks` | 4 | 每个 ResBlock 堆叠的数量 |
| `gate_hidden_channels` | 32 | 门控网络隐藏层通道数 |

---

## 技术栈

| 组件 | 技术 |
|---|---|
| 语言 | Rust (2024 edition) |
| 深度学习框架 | Burn v0.21.0 |
| GPU 后端 | WGPU（默认）/ CUDA |
| CPU 后端 | NdArray（测试用） |
| 图像处理 | `image` crate v0.25.10 |
| 错误处理 | `thiserror` v2.0.12 |

---

## 构建与运行

### 环境要求

- Rust 工具链（建议通过 [rustup](https://rustup.rs/) 安装）
- 支持 WGPU 的 GPU（或使用 NdArray CPU 后端）

### 编译

```bash
cargo build --release
```

### 运行（模型信息）

```bash
cargo run --release
```

### 测试

```bash
# 运行常规测试
cargo test

# 运行 4096 全分辨率前向测试（CPU 较慢）
cargo test -- --ignored
```

---

## 使用示例

```rust
use burn::backend::Wgpu;
use upscaler::{RefGuidedUpsamplerConfig, load_rgb_tensor, save_rgb_tensor};

type Backend = Wgpu;

fn main() {
    let device = Default::default();
    let model = RefGuidedUpsamplerConfig::new().init::<Backend>(&device);

    // 加载输入图像
    let main = load_rgb_tensor::<Backend>(
        "examples/main_1024.png",
        Some((1024, 1024)),
        &device,
    ).expect("加载主图失败");

    let reference = load_rgb_tensor::<Backend>(
        "examples/ref_4096.png",
        Some((4096, 4096)),
        &device,
    ).expect("加载参考图失败");

    // 推理
    let output = model.forward(main, reference);

    // 保存结果
    save_rgb_tensor(output, "output_4096.png").expect("保存失败");
}
```

---

## 项目结构

```
upscaler/
├── Cargo.toml                  # 依赖与项目配置
├── Cargo.lock                  # 锁定依赖版本
├── README.md                   # 本文档
└── src/
    ├── lib.rs                  # 公共 API 导出
    ├── main.rs                 # 二进制入口（模型信息展示）
    ├── tensor_image.rs         # 图像 ↔ Tensor 转换工具
    └── model/
        ├── mod.rs              # 模块组织与导出
        ├── upsampler.rs        # 主模型 RefGuidedUpsampler + 训练损失建议
        ├── presence_gate.rs    # 存在性门控子网络
        └── blocks.rs           # 可复用 NN 构建块（ResBlock, FusionBlock 等）
```

---

## 训练计划

代码中预留了 `TrainingLossHints` 结构体（`src/model/upsampler.rs:199`），建议的训练损失配置：

| 损失项 | 权重 | 说明 |
|---|---|---|
| L1 Loss | 1.0 | 像素级重建损失 |
| Perceptual Loss | 0.1 | 感知特征匹配损失 |
| Gate Supervision | 0.0 | 门控监督损失（预留，当前未启用） |

> **注意：** 训练代码尚未实现，当前仅包含模型架构定义和推理入口。

---

## 许可证

暂未指定。
