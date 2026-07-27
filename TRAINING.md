# 训练指南 — RefGuidedUpsampler

> **本文档由 AI 辅助生成，内容基于代码库实际实现。如存在歧义，请以代码为准。**

---

## 目录

1. [训练原理](#训练原理)
2. [数据集准备](#数据集准备)
3. [快速开始](#快速开始)
4. [配置参数](#配置参数)
5. [训练策略](#训练策略)
6. [检查点与恢复](#检查点与恢复)
7. [监控与日志](#监控与日志)
8. [常见问题](#常见问题)

---

## 训练原理

### 模型输入/输出

```
输入:
  main (主图)      [1, 3, 1024, 1024]  ← Flux2 编辑后的低分辨率结果
  reference (参考图) [1, 3, 4096, 4096]  ← 原始高分辨率照片

输出:
  output (成品图)   [1, 3, 4096, 4096]  ← 上采样 + 高频细节恢复

公式: output = clamp01( bilinear_4×(main) + gate × delta )
```

### 自监督训练策略

由于缺少成对的（编辑后主图, 原始参考图, 理想输出）数据，本训练脚本采用**自监督学习**策略：

1. 从高分辨率图像（≥4096×4096）中随机裁剪区域
2. 将裁剪区域降采样 4× 作为"主图"（模拟低分辨率输入）
3. 原始裁剪区域同时作为"参考图"和"训练目标"
4. 模型学习从参考图中恢复被降采样丢失的高频细节

```
训练数据生成流程:
  4096×4096 HR 图像
       │
       ├──[降采样 4×]──→ main    [256×256]   (模拟 Flux2 编辑输出)
       │
       └──[保持原样]──→ reference [1024×1024] (参考图 = 训练目标)
```

> **注意**：自监督训练中 `reference == ground_truth`，门控网络会倾向于输出 1。为帮助门控学会抑制，训练时对参考图添加了随机高斯噪声扰动（`ref_noise_std=0.02`），模拟参考图中不可靠的区域。

### 为什么可以用裁剪训练？

模型是全卷积架构（无全连接层），可以处理任意尺寸输入。用 256×256→1024×1024 裁剪训练后，推理时可无缝扩展到 1024×1024→4096×4096。

### 相机滤镜数据增强

为了帮助模型学习"从参考图恢复细节，而非简单复制参考图"，训练时对主图（1024×1024 原始图片）施加随机相机滤镜，模拟 Flux2 编辑可能引入的色彩、光照、风格变化。参考图始终保持不变。

**增强策略**：
- 80% 概率对主图应用随机滤镜
- 30% 概率叠加第二个轻量滤镜（Grain / SoftFocus / Vivid / Fade）
- 参考图保持原始高分辨率，不做任何滤镜处理

**可用滤镜**（16 种）：

| 滤镜 | 效果 | 说明 |
|------|------|------|
| `Original` | 原图 | 不处理，作为对照组 |
| `Warm` | 暖色调 | 增加红/黄色温 |
| `Cool` | 冷色调 | 增加蓝色温 |
| `Vivid` | 鲜艳 | 提高饱和度 50% + 对比度 |
| `Fade` | 褪色 | 降低对比度，略微提亮 |
| `Mono` | 黑白 | 完全去饱和 |
| `Noir` | 高对比黑白 | 去饱和 + 高对比度 |
| `Chrome` | 铬黄 | 高饱和 + 高对比 + 微暖 |
| `Sepia` | 老旧照片 | 棕褐色调 + 褪色 |
| `ToyCamera` | 玩具相机 | 暗角 + 轻微模糊 |
| `Grain` | 胶片颗粒 | 添加随机噪声 |
| `SoftFocus` | 柔焦 | 轻微高斯模糊 |
| `Underexpose` | 曝光不足 | 亮度 ×0.5 |
| `Overexpose` | 曝光过度 | 亮度 ×1.8 |
| `Invert` | 反转色 | 颜色反转 |
| `Posterize` | 海报化 | 减少颜色层次（4 级） |

**启用/禁用**：

```rust
// 在 TrainingConfig 中控制
TrainingConfig {
    use_camera_filters: true,   // 默认启用
    // ...
}
```

也可以通过环境变量控制（需在 `src/bin/train.rs` 中添加对应逻辑）：

```bash
# 禁用滤镜增强
CAMERA_FILTERS=0 DATA_DIR=./dataset cargo run --release --bin train
```

---

## 数据集准备

### 格式要求

```
dataset/                          # 数据集根目录（可自定义名称）
├── image_0001.png                # 高分辨率 PNG 图像
├── image_0002.png                # 支持 PNG / JPG / JPEG
├── image_0003.png
├── ...
└── image_XXXX.png
```

### 图像要求

| 要求 | 说明 |
|------|------|
| **最小分辨率** | ≥4096×4096 像素（裁剪训练模式需要 ≥1024×1024） |
| **格式** | PNG（推荐）/ JPG / JPEG |
| **色彩** | RGB 彩色图像 |
| **内容** | 高质量自然图像（风景、建筑、纹理等），避免纯色块或纯文字图像 |
| **数量** | 建议 ≥1000 张（裁剪训练），≥5000 张更佳 |

### 推荐数据来源

| 来源 | 说明 | 链接 |
|------|------|------|
| **DIV8K** | 8K 分辨率高质量图像，专为超分辨率设计 | https://competitions.codalab.org/competitions/22217 |
| **Flickr2K** | 2K 分辨率，2650 张高质量图像 | https://github.com/limbee/NTIRE2017 |
| **Unsplash** | 大量高分辨率免费照片 | https://unsplash.com |
| **自采集** | 用相机拍摄的 4K+ 照片 | — |

### 数据准备脚本（示例）

```bash
#!/bin/bash
# 1. 创建数据集目录
mkdir -p dataset

# 2. 将高分辨率图像复制/链接到数据集目录
#    确保所有图像 ≥ 4096×4096
cp /path/to/your/high_res_images/*.png dataset/

# 3. 检查图像数量
ls dataset/ | wc -l
# 应该输出 ≥ 1000
```

### 数据量建议

| 用途 | 最少数量 | 推荐数量 | 说明 |
|------|----------|----------|------|
| 快速验证 | 10-50 张 | — | 验证训练流程是否正常 |
| 预训练 | 1000 张 | 5000+ 张 | 学习基本的参考引导上采样能力 |
| 微调 | 100-500 张 | — | 在特定领域数据上微调（如人像） |

### 验证集

训练脚本会自动从数据集中随机抽取 5%（由 `val_split` 控制）作为验证集。无需手动划分。

---

## 快速开始

### 1. 准备数据集

```bash
mkdir -p dataset
# 将你的高分辨率图像放入 dataset/ 目录
```

### 2. 编译

```bash
cargo build --release --bin train
```

### 3. 运行训练

```bash
# 默认配置（裁剪模式，256→1024）
DATA_DIR=./dataset cargo run --release --bin train

# 自定义配置
DATA_DIR=./dataset \
EPOCHS=200 \
STEPS_PER_EPOCH=2000 \
LR=0.0002 \
CROP_SIZE=256 \
cargo run --release --bin train
```

### 4. 从检查点恢复训练

```bash
DATA_DIR=./dataset \
CHECKPOINT=./checkpoints \
cargo run --release --bin train
```

### 5. 全分辨率微调（需要 24+ GB VRAM）

```bash
DATA_DIR=./dataset \
FULL_RES=1 \
CROP_SIZE=1024 \
LR=0.00005 \
EPOCHS=50 \
cargo run --release --bin train
```

---

## 配置参数

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATA_DIR` | `./dataset` | 数据集目录路径 |
| `EPOCHS` | `100` | 训练轮数 |
| `STEPS_PER_EPOCH` | `1000` | 每轮训练步数 |
| `LR` | `0.0001` | 学习率 |
| `WEIGHT_DECAY` | `0.0001` | AdamW 权重衰减 |
| `CROP_SIZE` | `256` | 主图裁剪尺寸（ref 自动为 4×） |
| `FULL_RES` | `0` | 全分辨率模式：`1`=1024→4096，`0`=裁剪模式 |
| `REF_NOISE` | `0.02` | 参考图噪声标准差（帮助门控训练） |
| `SEED` | `42` | 随机种子 |
| `CHECKPOINT` | — | 从指定检查点目录恢复训练 |

### 模型配置（代码中修改）

在 `src/bin/train.rs` 中修改 `RefGuidedUpsamplerConfig`：

```rust
let model_config = RefGuidedUpsamplerConfig::new()
    .with_base_channels(64)     // 基础通道数
    .with_num_res_blocks(4)     // ResBlock 数量
    .with_gate_hidden_channels(32); // 门控隐藏通道
```

### 训练配置（代码中修改）

`TrainingConfig` 的完整默认值在 `src/training.rs` 中：

```rust
TrainingConfig {
    num_epochs: 100,            // 训练轮数
    steps_per_epoch: 1000,      // 每轮步数
    learning_rate: 1e-4,        // 学习率
    weight_decay: 1e-4,         // 权重衰减
    crop_main_size: 256,        // 裁剪尺寸
    val_split: 0.05,            // 验证集比例
    checkpoint_every_steps: 500,// 检查点保存间隔
    ref_noise_std: 0.02,        // 参考图噪声
    l1_weight: 1.0,             // L1 损失权重
    perceptual_weight: 0.0,     // 感知损失权重（预留）
    gate_supervision_weight: 0.0,// 门控监督权重（预留）
    full_resolution: false,     // 全分辨率模式
    // ...
}
```

---

## 训练策略

### 阶段一：裁剪预训练（推荐起始）

**目标**：让模型学习基本的参考引导上采样能力。

```
配置:
  CROP_SIZE=256              # 256×256 → 1024×1024
  EPOCHS=100                 # 100 轮
  STEPS_PER_EPOCH=1000       # 每轮 1000 步
  LR=0.0001                  # 初始学习率

显存需求: ~6-8 GB VRAM
预计耗时: 数小时（取决于 GPU）
```

### 阶段二：全分辨率微调（可选）

**目标**：在全分辨率上微调，适应 4K 推理。

```
配置:
  FULL_RES=1                 # 1024×1024 → 4096×4096
  CROP_SIZE=1024
  EPOCHS=50                  # 较少轮数
  STEPS_PER_EPOCH=500
  LR=0.00005                 # 更小的学习率

显存需求: ~20-24+ GB VRAM
预计耗时: 数小时到一天
```

### 损失函数

| 损失项 | 权重 | 说明 |
|--------|------|------|
| **L1 Loss** | 1.0 | 像素级重建损失，主要驱动 |
| Delta Sparsity | 0.01 | 鼓励 delta 输出稀疏，避免过度修正 |
| Perceptual Loss | 0.0 (预留) | 感知特征匹配，需额外依赖 |
| Gate Supervision | 0.0 (预留) | 门控监督，当前仅通过噪声扰动隐式训练 |

### 学习率调度

当前使用固定学习率 + AdamW。如需学习率衰减，可在训练循环中手动调整：

```rust
// 在 Trainer::train() 中添加余弦退火
let lr = config.learning_rate * 0.5 
    * (1.0 + f64::cos(std::f64::consts::PI * epoch as f64 / config.num_epochs as f64));
```

---

## 检查点与恢复

### 保存位置

```
checkpoints/
├── model_step_500.bin       # 第 500 步检查点
├── model_step_1000.bin      # 第 1000 步检查点
├── model_step_1500.bin
├── ...
└── model_latest.bin         # 最新模型（覆盖更新）
```

### 恢复训练

```bash
CHECKPOINT=./checkpoints cargo run --release --bin train
```

恢复时：
- 模型权重从 `model_latest.bin` 加载
- 优化器状态**不会**恢复（当前版本限制，后续改进）
- 训练步数从 0 重新计数

### 导出推理模型

```bash
# 将最新的检查点复制为推理用模型文件
cp checkpoints/model_latest.bin model_for_inference.bin
```

---

## 监控与日志

### 训练输出示例

```
╔══════════════════════════════════════════════════════════╗
║           RefGuidedUpsampler 训练启动                    ║
╠══════════════════════════════════════════════════════════╣
║  数据集图像数:   5000                                     ║
║  训练轮数:       100                                     ║
║  每轮步数:       1000                                    ║
║  梯度累积:          4                                    ║
║  学习率:       0.000100                                  ║
║  裁剪尺寸:       256×256 (main)                          ║
║                 1024×1024 (ref)                           ║
║  全分辨率模式:    OFF                                    ║
║  参数量:      3245678                                    ║
╚══════════════════════════════════════════════════════════╝

[Epoch    1 | Step     50] loss=0.023456  avg_loss=0.025123  samples=200
[Epoch    1 | Step    100] loss=0.021234  avg_loss=0.024012  samples=400
...
💾 检查点已保存: step=500
━━━ Epoch 1 完成 ━━━ avg_loss=0.022134  samples=4000 ━━━
💾 检查点已保存: step=1000
```

### 关键指标

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `loss` | 当前步损失 | 初始 0.1-0.5，逐步下降至 0.01-0.05 |
| `avg_loss` | 轮次平均损失 | 应持续下降 |
| `samples` | 已处理样本数 | 持续增长 |

### 异常信号

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| loss 不下降 | 学习率过大/过小 | 调整 `LR` |
| loss 震荡 | 学习率过大 | 降低 `LR` 到 1e-5 |
| loss=NaN | 梯度爆炸 | 降低 `LR` |
| 显存溢出 (OOM) | 裁剪尺寸过大 | 降低 `CROP_SIZE` |
| 数据加载慢 | 磁盘 I/O 瓶颈 | 使用 SSD 存放数据集 |

---

## 常见问题

### Q: 训练需要多长时间？

| GPU | 配置 | 预计耗时 |
|-----|------|----------|
| RTX 4090 (24GB) | 裁剪模式 256→1024, 100 epochs × 1000 steps | ~4-8 小时 |
| RTX 3090 (24GB) | 裁剪模式 256→1024, 100 epochs × 1000 steps | ~6-12 小时 |
| RTX 4070 (12GB) | 裁剪模式 256→1024, 100 epochs × 1000 steps | ~8-16 小时 |
| RTX 3060 (12GB) | 裁剪模式 256→1024, 100 epochs × 1000 steps | ~12-24 小时 |

### Q: 显存不够怎么办？

1. 降低 `CROP_SIZE`（如 128，ref 自动为 512）
2. 不要在 `FULL_RES=1` 模式下训练

### Q: 训练后如何推理？

```rust
use burn::backend::Wgpu;
use upscaler::{RefGuidedUpsamplerConfig, load_rgb_tensor, save_rgb_tensor};

type Backend = Wgpu;

fn main() {
    let device = Default::default();
    let model = RefGuidedUpsamplerConfig::new().init::<Backend>(&device);

    // 加载训练好的权重
    model.load_file("checkpoints/model_latest.bin", &device).unwrap();

    // 加载图像
    let main = load_rgb_tensor::<Backend>("main_1024.png", Some((1024, 1024)), &device).unwrap();
    let reference = load_rgb_tensor::<Backend>("ref_4096.png", Some((4096, 4096)), &device).unwrap();

    // 推理
    let output = model.forward(main, reference);

    // 保存
    save_rgb_tensor(output, "output_4096.png").unwrap();
}
```

### Q: 训练数据需要"编辑前后"的配对吗？

**不需要**。当前训练脚本使用自监督学习（降采样 → 重建），任何高分辨率图像都可以作为训练数据。

如果你有 Flux2 编辑前后的配对数据（编辑后 1024 + 原始 4096 + 理想输出 4096），可以修改 `HighResDataset` 来加载配对数据，进行监督微调。这会比纯自监督训练效果更好。

### Q: 如何评估训练效果？

1. **定量指标**：在验证集上计算 PSNR / SSIM（需自行实现）
2. **定性评估**：用训练好的模型推理几张测试图像，肉眼对比
3. **门控可视化**：提取 `presence_gate` 的输出，观察门控图是否合理

---

## 文件结构

```
upscaler/
├── src/
│   ├── bin/
│   │   └── train.rs              # 训练二进制入口
│   ├── augmentation.rs           # 相机滤镜数据增强模块
│   ├── training.rs               # 训练模块（数据集、损失、训练循环）
│   ├── model/
│   │   ├── mod.rs
│   │   ├── upsampler.rs          # RefGuidedUpsampler 模型定义
│   │   ├── blocks.rs             # 神经网络构建块
│   │   └── presence_gate.rs      # 存在性门控
│   ├── tensor_image.rs           # 图像 ↔ 张量转换
│   ├── lib.rs                    # 库入口
│   └── main.rs                   # 推理入口
├── Cargo.toml
├── TRAINING.md                   # 本文档
└── README.md                     # 项目文档
```