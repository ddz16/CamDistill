# 运镜识别多模型 SFT 训练

基于 ms-swift 4.3.0 的运镜识别模型全参数微调，支持 4 种多模态模型。

## 项目概述

**任务**：输入一段视频 → 输出结构化 JSON 运镜分析结果（时间段、运镜类型、方向、速度、特殊技法）

**方法**：冻结 ViT + Aligner，全参数微调 LLM 部分，让多模态大模型学会从视频帧序列中识别运镜。

**标签体系**：15 种基础运镜（Basic Movement）+ 15 种特殊技法（Special Movement），详见 `common.py`。

---

## 支持的模型

| 模型 | 快捷名称 | HuggingFace ID | 参数量 | 特点 |
|------|----------|----------------|--------|------|
| Qwen3-VL-4B | `qwen3vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` | ~4B | 轻量高效 |
| Qwen3-VL-8B | `qwen3vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | ~8B | 性能均衡 |
| Qwen3.5-4B | `qwen35-4b` | `Qwen/Qwen3.5-4B` | ~4B | Linear Attention，长序列优化 |
| Qwen3.5-9B | `qwen35-9b` | `Qwen/Qwen3.5-9B` | ~9B | Linear Attention，长序列优化 |

> 所有模型均为多模态模型（支持视频输入），首次运行时自动从 HuggingFace/ModelScope 下载权重。

---

## 环境信息

| 项目 | 值 |
|------|-----|
| conda 环境 | `cm` |
| Python | 3.12.13 |
| ms-swift | 4.3.0.dev0 |
| transformers | 5.8.1 |
| flash-linear-attention | 0.5.0 |
| causal_conv1d | 1.6.2.post1 |

激活环境：`conda activate cm`

---

## 数据路径说明

### 训练数据（已处理好的 swift SFT 格式 JSONL）

位置：`/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/train_data/`

| 文件 | 大小 | 说明 |
|------|------|------|
| `camera_movement_train_human_all_167k.jsonl` | 1.1GB | **全量数据**（所有人工标注），当前默认使用 |
| `camera_movement_train_human_excluded_149k.jsonl` | 929MB | 排除老版本标注后的数据 |
| `camera_movement_train_human_excluded_balanced_114k.jsonl` | 721MB | 排除 + 类别均衡采样 |

> 本项目 `train_data/` 目录下有软链接指向上述文件，可直接使用。

### 原始标注数据（人工标注源文件）

| 来源 | 路径 | 规模 |
|------|------|------|
| 云雀标注 | `/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/train_data/human_labels/yunque/` | 51 个 jsonl |
| 混元标注 | `/group/40059/yyjyu/data/aigc/camera_data/human_labels/hunyuan/results/` | 19 个 jsonl |

### 视频文件（.mp4）

训练脚本按以下顺序查找视频，找到即用：

| 优先级 | 路径 | 规模 |
|--------|------|------|
| 1 | `/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos/` | **25.4万** 个 mp4，584GB |
| 2 | `/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_21k_video_shot_cameraMove/videos/` | 2.1万 个 |
| 3 | `/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_realfilm_30w_260401_deduped_final/videos/` | 30万 个 |

> 如果以上目录都找不到对应视频，脚本会尝试从标注文件中的 `cos_url` 自动下载。

### 测试集

位置：`/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/train_data/human_labels/yunque/testset_results/`

| 测试集 | 条数 | 用途 |
|--------|------|------|
| `testset_500_refined_reviewed_450_sampled_300_refined_v1.jsonl` | 300 | 主测试集 |
| `testset_300_refined_v3.jsonl` | 300 | 精标测试集 v3 |
| `testset_300_balanced.jsonl` | 300 | 均衡测试集 |
| `testset_arc_zoom_roll_200.jsonl` | ~200 | 稀有类（Arc/Zoom/Roll）专项 |

---

## 快速开始

### 1. 准备数据

```bash
cd /group/40009/dazhaodu/ms-swift
conda activate cm

# 从人工标注生成训练数据（同时输出三个版本）
python camera_movement_sft/prepare_human_labels.py

# 或直接使用已有数据（已有软链接，无需额外操作）
ls camera_movement_sft/train_data/
```

### 2. 启动训练

```bash
cd /group/40009/dazhaodu/ms-swift
conda activate cm

# 选择模型训练 (输出目录自动生成, 无需手动指定)
bash camera_movement_sft/train.sh qwen3vl-4b
bash camera_movement_sft/train.sh qwen3vl-8b
bash camera_movement_sft/train.sh qwen35-4b
bash camera_movement_sft/train.sh qwen35-9b

# 后台启动（推荐）
bash camera_movement_sft/start_train.sh qwen3vl-8b
# 训练完成后终端会打印: "输出目录: output/camera_sft_qwen3vl_8b/v0-20260525-143000"
```

### 3. 评测

```bash
# 查看训练输出了什么版本
ls output/camera_sft_qwen3vl_8b/

# 一键评测 (只需指定训练输出目录, 其他全部自动)
TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_8b/v0-20260525-143000 \
bash camera_movement_sft/eval/run_batch_checkpoints.sh
# → 自动: 推理所有 checkpoint + 后处理 + 评测打分
# → 结果: eval/eval_results/qwen3vl_8b_v0_testset_300_refined_v1/eval_results.json
```

#### A) 训练：离线 cache 模式（原来方式）

  VGGT_MODE=cache \
  VGGT_CACHE_DIR=/path/to/vggt_cache \
  bash camera_movement_sft/train_caminject.sh qwen3vl-8b

  #### B) 训练：在线提取模式（无需预提取）

  VGGT_MODE=online \
  VGGT_TEACHER_TYPE=vggt_omega \
  VGGT_MODEL_PATH=/group/40009/dazhaodu/vggt-omega/checkpoints/vggt_omega_1b_512.pt \
  bash camera_movement_sft/train_caminject.sh qwen3vl-8b

  ———

  #### C) 评测：离线 cache 模式

  USE_CAMINJECT=1 \
  VGGT_MODE=cache \
  VGGT_CACHE_DIR=/path/to/vggt_cache \
  CAMINJECT_MODEL_TYPE=qwen3_vl_caminject \
  TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_8b_caminject/v0-xxxxxx \
  bash camera_movement_sft/eval/run_batch_checkpoints.sh

  #### D) 评测：在线提取模式（你要的“直接跑推理”）

  USE_CAMINJECT=1 \
  VGGT_MODE=online \
  VGGT_TEACHER_TYPE=vggt_omega \
  VGGT_MODEL_PATH=/group/40009/dazhaodu/vggt-omega/checkpoints/vggt_omega_1b_512.pt \
  CAMINJECT_MODEL_TYPE=qwen3_vl_caminject \
  TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_8b_caminject/v0-xxxxxx \
  bash camera_movement_sft/eval/run_batch_checkpoints.sh

  如果你评的是 qwen3.5 系列，把 CAMINJECT_MODEL_TYPE 改成：

  CAMINJECT_MODEL_TYPE=qwen3_5_caminject

  ———

  如果你愿意，我下一步可以再给你补一个 camera_movement_sft/eval/run_batch_checkpoints_caminject.sh 专用薄封装脚本，这样你就不用每次记这么多环
  境变量。

### 4. 监控与管理

```bash
# 查看训练状态 + GPU 使用 + 最新 loss
bash camera_movement_sft/monitor_train.sh

# 查看实时日志
tail -f camera_movement_sft/logs/train_qwen3vl-8b_*.log

# 停止训练
bash camera_movement_sft/stop_train.sh
```

---

## 自定义训练参数

所有参数都支持通过环境变量覆盖：

```bash
# 自定义数据路径
DATASET_PATH=/path/to/data.jsonl bash camera_movement_sft/train.sh qwen3vl-8b

# 自定义 GPU（例如只用 4 卡）
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash camera_movement_sft/train.sh qwen3vl-4b

# 调整学习率和 batch size
LEARNING_RATE=2e-5 PER_DEVICE_BATCH_SIZE=2 bash camera_movement_sft/train.sh qwen3vl-4b

# 使用 ZeRO3（显存不足时）
DEEPSPEED_STAGE=zero3 bash camera_movement_sft/train.sh qwen35-9b

# 从 checkpoint 续训
RESUME=true \
RESUME_OUTPUT_DIR=output/camera_sft_qwen3vl_8b/v0-20260520-120000 \
RESUME_CHECKPOINT=output/camera_sft_qwen3vl_8b/v0-20260520-120000/checkpoint-5000 \
bash camera_movement_sft/train.sh qwen3vl-8b
```

---

## 训练配置

### 各模型差异

| 参数 | Qwen3-VL-4B | Qwen3-VL-8B | Qwen3.5-4B | Qwen3.5-9B |
|------|-------------|-------------|------------|------------|
| 学习率 | 2e-5 | 1.5e-5 | 2e-5 | 1.5e-5 |
| Per-device Batch | 2 | 1 | 2 | 1 |
| Gradient Accum | 2 | 2 | 2 | 2 |
| 有效 Batch Size | 32 | 16 | 32 | 16 |
| DeepSpeed | ZeRO-2 | ZeRO-2 | ZeRO-2 | ZeRO-2 |

### 所有模型共同配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 训练方式 | `tuner_type=full` | 全参数微调 LLM 部分 |
| 冻结 ViT | Yes | 视觉编码器不训练 |
| 冻结 Aligner | No | 视觉-语言投影层参与训练 |
| 训练轮数 | 2 epochs | |
| Max Length | 16384 tokens | 超长样本直接丢弃 |
| 视频 FPS | 5 | 每秒抽 5 帧 |
| 最大帧数 | 100 | 支持 20 秒视频 |
| 调度器 | Cosine | 余弦退火 |
| Warmup | 5% | |
| Weight Decay | 0.01 | |
| 精度 | bfloat16 | |
| 注意力机制 | Flash Attention 2 | 高效注意力实现 |
| 保存策略 | 每 1000 步 | 最多 15 个 checkpoint |
| 最佳模型 | eval_loss 最小 | 训练结束加载最佳 |
| 验证集 | 5% 自动划分 | |

---

## 评测

### 自动命名机制

训练和评测的目录名全部自动生成，**不需要手动命名**：

**训练输出目录** — 由脚本自动决定，格式为 `output/camera_sft_{方案}_{模型}/v{N}-{时间戳}`：

| 训练命令 | 自动生成的输出目录 |
|---------|-------------------|
| `train.sh qwen3vl-8b` | `output/camera_sft_qwen3vl_8b/v0-20260525-143000` |
| `train_camdistill.sh qwen3vl-8b` | `output/camera_sft_qwen3vl_8b_camdistill/v0-20260525-...` |
| `train_caminject.sh qwen3vl-8b` | `output/camera_sft_qwen3vl_8b_caminject/v0-20260525-...` |
| `train.sh qwen35-9b` | `output/camera_sft_qwen35_9b/v0-20260525-...` |

> 多次训练同一方案会自动递增版本号: `v0`, `v1`, `v2`...

**评测结果目录** — 从训练目录路径自动推导 `TRAIN_TAG`：

```
训练目录: output/camera_sft_qwen3vl_8b/v0-20260525-143000
                         └── 自动提取 ──┘ └─ 取 ─┘
→ TRAIN_TAG = "qwen3vl_8b_v0"
→ 评测结果: eval/eval_results/qwen3vl_8b_v0_testset_300_refined_v1/
```

不同方案的评测结果互不覆盖：
```
eval/eval_results/
├── qwen3vl_8b_v0_testset_300_refined_v1/          ← baseline SFT
├── qwen3vl_8b_camdistill_v0_testset_300_.../      ← CamDistill
├── qwen3vl_8b_caminject_v0_testset_300_.../     ← CamInject
└── qwen35_9b_v0_testset_300_.../                  ← 其他模型
```

### 评测步骤

训练完成后，终端会打印输出目录路径。复制该路径即可一键评测：

```bash
# 1. 查看训练输出 (找到具体的版本目录)
ls output/camera_sft_qwen3vl_8b/
# → v0-20260525-143000

# 2. 一键评测所有 checkpoint (默认测试集)
TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_8b/v0-20260525-143000 \
bash camera_movement_sft/eval/run_batch_checkpoints.sh

# 3. 指定测试集
TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_8b/v0-20260525-143000 \
bash camera_movement_sft/eval/run_batch_checkpoints.sh \
    /group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/train_data/human_labels/yunque/testset_results/testset_300_refined_v3.jsonl

# 4. 多个测试集一起评
TRAIN_OUTPUT_DIR=... bash camera_movement_sft/eval/run_batch_checkpoints.sh \
    /path/to/testset_300_refined_v3.jsonl \
    /path/to/testset_arc_zoom_roll_200.jsonl

# 5. 强制重跑 (覆盖旧结果)
TRAIN_OUTPUT_DIR=... bash camera_movement_sft/eval/run_batch_checkpoints.sh --force
```

### 评测流程 (脚本自动完成)

```
run_batch_checkpoints.sh 自动执行 3 步:

Step 0: 数据转换 (prepare_test_data.py)
   原始测试集 JSONL → swift 推理格式 + 视频路径解析

Step 1: 模型推理 (swift infer, 8 卡并行)
   逐个 checkpoint 推理, 已有结果自动跳过

Step 2: 评测打分 (evaluate_camera_movement_fixed.py)
   Frame-level + Segment-level 评测, 输出对比表格
```

### 评测输出结构

```
eval/eval_results/{TRAIN_TAG}_{测试集名}/
├── test_for_infer.jsonl       # swift 推理格式数据
├── video_id_mapping.jsonl     # ID 映射
├── checkpoint-1000/
│   ├── infer_results.jsonl    # 推理原始输出
│   └── eval_pred.jsonl        # 后处理后的预测
├── checkpoint-2000/
│   └── ...
├── eval_results.json          # 所有 checkpoint 的评测结果汇总 ← 看这个
└── eval_log.txt               # 评测日志 (含对比表格)
```

### 核心评测指标

**主指标**: Macro-average 的 `basic_movement_with_direction` F1
- 视频等权 (每个视频贡献相同)
- 带方向 (Pan Left ≠ Pan Right)
- 同时报告 Frame-level 和 Segment-level (IoU=0.3/0.5/0.7)

### 外部 Benchmark 评测

除自有测试集外，还支持在 3 个公开 Benchmark 上评测：

| Benchmark | 任务格式 | 评测指标 | 特点 |
|-----------|---------|---------|------|
| **CameraBench** | Yes/No 二分类 | Accuracy | NeurIPS 2025, 15种运动, ~7.7k 样本 |
| **RefineShot** | 4选项 MCQ (ABCD) | Accuracy | 8维影视理解, 含 camera_movement |
| **MultiCamVideo** | 4选项 MCQ (ABCD) | Accuracy | 合成视频, 运镜组合识别 |

```bash
# 一键跑所有外部 benchmark
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh <checkpoint>

# 只跑 CameraBench
BENCHMARKS="camerabench" \
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh <checkpoint>

# 只跑 RefineShot (camera_movement 类别)
BENCHMARKS="refineshot" \
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh <checkpoint>

# 只跑 MultiCamVideo
BENCHMARKS="multicam" \
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh <checkpoint>
```

**数据路径** (已配置默认值，首次需下载数据):
- CameraBench 视频: `/group/40009/dazhaodu/t2v_metrics/camerabench/data/videos/`
- RefineShot 数据: `/group/40009/dazhaodu/RefineShot/evaluation/data/ShotBench/test.tsv`
- MultiCamVideo: 需从 HuggingFace 下载 `fengyee/camera-motion-dataset-and-benchmark`

**注意**: 这些 benchmark 都是选择题/Yes-No 格式。SFT 全参数微调后模型可能倾向输出 JSON 而非选项字母，评测结果可能偏低。这是预期行为。

详细说明见 `eval/benchmarks/benchmark.md`。

---

## 文件结构

```
camera_movement_sft/
├── common.py                  # 公共模块（15种运镜闭集定义、System Prompt、标签标准化）
├── prepare_human_labels.py    # 人工标注数据 → swift 格式（含三版本输出 + 均衡采样）
├── prepare_data_jsonl.py      # 通用 JSONL 格式转换
├── train.sh                   # 统一训练入口（通过参数选择模型）
├── train_qwen3vl_4b.sh        # 快捷启动: Qwen3-VL-4B
├── train_qwen3vl_8b.sh        # 快捷启动: Qwen3-VL-8B
├── train_qwen35_4b.sh         # 快捷启动: Qwen3.5-4B
├── train_qwen35_9b.sh         # 快捷启动: Qwen3.5-9B
├── train_camdistill.sh        # CamDistill 蒸馏训练
├── train_caminject.sh       # CamInject baseline 训练
├── start_train.sh             # 后台启动训练
├── stop_train.sh              # 停止训练
├── monitor_train.sh           # 监控训练状态
├── train_data/                # 训练数据（软链接到源数据）
├── logs/                      # 训练日志
├── plugins/                   # Camera Token 增强插件
│   ├── camdistill_plugin.py          # 插件入口: 注册 model types + Template encode/collator/forward_context patch
│   ├── camdistill_model.py           # CameraTokenModule (交替注意力, ~200M) + align_proj
│   ├── camdistill_loss.py            # SFT + 余弦蒸馏 Loss (从 _last_video_ids 加载 teacher target)
│   ├── caminject_model.py          # CamInjectAdapter + VGGTProjector (~12.6M, fail-fast)  [已重命名为 caminject_model.py]
│   ├── modeling_qwen3_vl_camdistill.py   # Qwen3-VL 修改版 (preexpanded 一致性检查 + camera position)
│   ├── modeling_qwen3_5_camdistill.py    # Qwen3.5 修改版 (同上)
│   └── vggt_feature_extractor.py     # VGGT 离线预提取 (多 GPU 并行)
├── eval/                      # 评测
│   ├── run_batch_checkpoints.sh          # 自有测试集评测（一键推理+评测）
│   ├── evaluate_camera_movement_fixed.py # Frame-level + Segment-level 评测
│   ├── eval_macro_average.py             # Macro-average 评测
│   ├── postprocess_results.py            # 推理结果后处理
│   ├── prepare_test_data.py              # 测试数据格式转换
│   └── benchmarks/                       # 外部 Benchmark 评测
│       ├── run_benchmarks.sh             # 一键跑 CameraBench/RefineShot/MultiCam
│       ├── eval_camerabench.py           # CameraBench (Yes/No 二分类)
│       ├── eval_refineshot.py            # RefineShot (MCQ 选择题)
│       ├── eval_multicam.py              # MultiCamVideo (MCQ 选择题)
│       └── benchmark.md                  # 详细说明文档
├── CamDistill_注入实现报告.md  # Camera Token 蒸馏技术方案详解
└── README.md                  # 本文档
```

---

## Camera Token 增强训练（CamDistill / CamInject）

除普通 SFT 外，本项目还支持两种 Camera Token 增强方案，为 LLM 注入 3D 空间感知能力：

### 三种方案对比

| | 普通 SFT | CamDistill (蒸馏) | CamInject (直接注入) |
|---|---|---|---|
| **启动脚本** | `train.sh` | `train_camdistill.sh` | `train_caminject.sh` |
| **Camera Token** | 无 | 自学习模块（交替注意力） | VGGT 冻结输出 |
| **推理需 VGGT** | 否 | **否** | 是 |
| **需要训练** | 是 | 是 | 是 |
| **额外可训练参数** | 0 | ~200M | ~12.6M |
| **Loss** | SFT | SFT + 余弦蒸馏 | 仅 SFT |
| **原理** | 纯视觉理解 | 学习空间特征 | 注入空间特征 |

---

### 方案一：CamDistill (Camera Token 蒸馏)

**核心思想**: 训练一个 `CameraTokenModule`，从 ViT 中间层特征中学习生成 camera token。训练时用 VGGT 预提取的 features 做蒸馏 target（余弦相似度 loss）。**推理时不需要 VGGT**。

**架构图**:
```
ViT (冻结) → 中间层缓存 (6层) → CameraTokenModule (可训练, ~200M)
    │                                     │
    │                               camera_embeds (T, llm_dim)
    │                                     │
    ViT → Merger → video_embeds ──────────┘
                                          │
                           合并: [CAM₁, vis_frame1, CAM₂, vis_frame2, ...]
                                          │
                           position_ids (camera = 帧中心 3D RoPE)
                                          │
                           LLM forward → SFT loss + λ·distill_loss
```

**CameraTokenModule 结构** (交替注意力):
- 输入: ViT 中间 6 层的 hidden states → 拼接/投影得到 camera queries
- 交替重复 `depth=6` 次:
  - `FrameCrossAttention`: 每个 camera token attend 本帧的 visual tokens (单向)
  - `GlobalCameraSelfAttention`: 所有帧的 camera tokens 互相 attend (全局时序建模)
- 输出: `(T_total, llm_dim)` 的 camera embeddings

**可训练参数**: ~200M (CameraTokenModule) + LLM 全参数
**Loss**: `L = L_sft + λ × L_distill`
- `L_distill = 1 - cos_sim(camera_features, VGGT_target)`
- `λ = 0.3` (默认, 通过 `CAMDISTILL_LAMBDA` 环境变量控制)

**关键文件**:
| 文件 | 作用 |
|------|------|
| `plugins/camdistill_model.py` | CameraTokenModule 定义 (交替注意力) |
| `plugins/camdistill_loss.py` | SFT + 余弦蒸馏 Loss |
| `plugins/camdistill_plugin.py` | 插件入口, 注册 Loader |
| `plugins/modeling_qwen3_vl_camdistill.py` | Qwen3-VL 修改版 (注入逻辑) |
| `plugins/modeling_qwen3_5_camdistill.py` | Qwen3.5 修改版 (注入逻辑) |

**使用方式**:
```bash
# 前提: 已预提取 VGGT features (见下方)
VGGT_CACHE_DIR=/path/to/vggt_cache \
CAMDISTILL_DEPTH=6 \
CAMDISTILL_LAMBDA=0.3 \
bash camera_movement_sft/train_camdistill.sh qwen3vl-8b
```

---

### 方案二：CamInject (直接注入)

**核心思想**: 直接使用 VGGT 预提取的 camera token (2048D)，通过可训练的 Projector 投影到 LLM 维度后，作为独立 token 插入 LLM 序列。训练时只训练 Projector + LLM，**推理时需要 VGGT**。

**架构图**:
```
VGGT (离线预提取) → camera_features (S, 2048) → 保存为 .pt cache
                                                      │
                              [训练时从 cache 加载, 帧对齐]
                                                      │
                              VGGTProjector (可训练, ~12.6M)
                              LayerNorm → Linear(2048→2048) → GELU → Linear(2048→llm_dim)
                                                      │
                                               camera_embeds (T, llm_dim)
                                                      │
ViT (冻结) → Merger → video_embeds ──────────────────┘
                                                      │
                               合并: [CAM₁, vis_frame1, CAM₂, vis_frame2, ...]
                                                      │
                               position_ids (camera = 帧中心 3D RoPE)
                                                      │
                               LLM forward → 仅 SFT loss
```

**帧数对齐**:
- VGGT cache 存原始帧数 S (fps=5, max=100)
- Qwen3-VL 使用 `temporal_patch_size=2`，模型帧数 T = S/2
- Adapter 自动做 2帧平均池化对齐: `(S, 2048) → (T, 2048)`

**可训练参数**: ~12.6M (VGGTProjector) + LLM 全参数
**Loss**: 仅标准 SFT Cross-Entropy

**关键文件**:
| 文件 | 作用 |
|------|------|
| `plugins/caminject_model.py` | CamInjectAdapter + VGGTProjector |
| `plugins/camdistill_plugin.py` | 插件入口, 注册 CamInject Loader |
| `plugins/modeling_qwen3_vl_camdistill.py` | 共用注入逻辑 (`_camdistill_mode='direct'`) |

**使用方式**:
```bash
# 前提: 已预提取 VGGT features (见下方)
VGGT_CACHE_DIR=/path/to/vggt_cache \
bash camera_movement_sft/train_caminject.sh qwen3vl-8b
```

---

### 两种方案共同的注入机制

CamDistill 和 CamInject 使用 **完全相同的 camera token 注入逻辑**，区别仅在于 camera_embeds 的来源：

**注入方式**: 每帧 visual tokens 前面独立插入 1 个 camera token

```
原始序列: [text] <ts> [vis_frame1 × 36] <ts> [vis_frame2 × 36] ... [text]
注入后:   [text] <ts> [CAM₁] [vis_frame1 × 36] <ts> [CAM₂] [vis_frame2 × 36] ... [text]
```

**架构改动 (2026-06 重构)**：扩展逻辑由 model.forward 内部移至 **Template.encode 阶段**，以 monkey-patch 方式实现。collator 看到的就是已扩展的最终序列，不再需要 outer forward wrapper。

**Template.encode 阶段做的事** (`_expand_single_sample_for_camera`):
- `input_ids`: 每个连续 video_token 段前插入 1 个 video_token_id 作为 camera placeholder
- `labels`: 同步插入 `-100`（camera 位置不计算 SFT loss）
- `loss_scale`: 同步插入 `0`
- `mm_token_type_ids`: camera 位置标记为 `2`
- 所有扩展严格按 `dst_positions` 散列，**保证 input_ids 与 labels 永远对齐**

**video_ids 三段式打通** (替换之前不稳定的 collator 注入):
1. **Template.encode** monkey-patch：从 `inputs.videos` 提取路径 → 生成 `video_id` 注入到 sample features
2. **Template._data_collator_mm_data** monkey-patch：从 batch 收集 `video_ids` 到 batch tensor
3. **Template.forward_context** monkey-patch (ctx manager)：进入 forward 前把 `video_ids` 塞进 adapter._pending_video_ids，退出时恢复，**避免跨 batch 污染**

**Position IDs (3D M-RoPE)**:
- Camera token 位置: `(temporal=帧时间, height=H_grid//2, width=W_grid//2)` — 帧中心
- direct 模式（CamInject）：collator 阶段就启用 camera position（因为序列已扩展）
- learn 模式（CamDistill）：保留旧逻辑兼容

**Fail-fast 校验** (训练时立即报错而不是默默 silently 跑):
- collator: 有视频输入但 `video_ids` 缺失/数量不匹配 → raise
- inner forward: `(input_ids == video_token_id).sum() != video_embeds.shape[0]` → raise
- CamInject adapter: cache miss / shape 错 / NaN/Inf / projector 输出 NaN → raise
- CamDistill loss: teacher target 缺失 / 形状错 / NaN → raise

**首次训练前清掉 dataset map cache**: 因为 encode 阶段改了 input_ids 内容，但 ms-swift 的 cache key 不区分 model_type，可能命中旧 cache 导致序列没扩展。

```bash
# 第一次跑前清 cache (路径根据实际)
rm -rf ~/.cache/modelscope/datasets/map_cache/*
# 之后训练即可正常使用 --load_from_cache_file true
```

---

### 使用前提：VGGT / VGGT-Omega 预提取

CamDistill 和 CamInject 都需要先对训练视频预提取 camera token features。支持两种 Teacher 模型：

| Teacher | 参数量 | Backbone | 图像尺寸 | Camera Token 维度 |
|---------|--------|----------|----------|-----------------|
| VGGT | 1.26B | DINOv2 ViT-L/14 | 518px | 2048 |
| VGGT-Omega | 1.14B | DINOv3 ViT-L/16 | 512px | 2048 |

两者输出格式完全相同（2048 维），下游 pipeline 无需修改，可无缝切换。

**脚本位置**: `camera_movement_sft/plugins/vggt_feature_extractor.py`

**GPU 显存需求**:

| 模型 | 参数量 | 推理显存 (batch=1, 100帧) |
|------|--------|--------------------------|
| VGGT | 1.26B | ~13GB (含模型+帧+中间激活) |
| VGGT-Omega | 1.14B | ~13GB (含模型+帧+中间激活) |

**多 GPU 并行**: 默认使用 8 卡并行 (`--num_gpus 8`)，自动将视频列表均分到各 GPU。每张 96GB 卡可轻松容纳单个视频推理 (~13GB)。

**模型权重下载位置**:
- VGGT (`facebook/VGGT-1B`): 通过 HuggingFace `from_pretrained` 自动下载到 `HF_HOME` 目录
- VGGT-Omega: 默认找本地 `/group/40009/dazhaodu/vggt-omega/checkpoints/vggt_omega_1b_512.pt`，不存在则通过 `hf_hub_download` 下载到 `HF_HOME`

**环境变量** (已内置于脚本中):
```bash
export HF_HOME=/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/hf
export HF_TOKEN=***REMOVED***
```

**磁盘空间估算**:
- 单模型: 167k 视频 × ~400KB/file ≈ **~32-65GB** (取决于平均帧数)
- 两个模型都提取: **~64-130GB** 总磁盘

#### 提取命令 (建议两个模型都提取)

```bash
# VGGT, 8卡并行 (存到 /group/40009/dazhaodu/vggt_cache/)
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl camera_movement_sft/train_data/camera_movement_train_diverse_50k.jsonl \
    --output_dir /group/40009/dazhaodu/vggt_cache/ \
    --teacher vggt \
    --save_pose \
    --num_gpus 8

# VGGT-Omega, 8卡并行 (存到 /group/40009/dazhaodu/vggt_omega_cache/)
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl camera_movement_sft/train_data/camera_movement_train_human_all_167k.jsonl \
    --output_dir /group/40009/dazhaodu/vggt_omega_cache/ \
    --teacher vggt_omega \
    --save_pose \
    --num_gpus 8

# 单卡模式 (兼容旧用法)
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl ... --output_dir ... --num_gpus 1 --device cuda:0
```

**时间估算** (167k 视频):

| GPU 数量 | 预估时间 |
|----------|----------|
| 1 GPU    | ~10 天   |
| 4 GPU    | ~2.5 天  |
| 8 GPU    | ~1.25 天 |

支持中断续跑 (`--skip_existing` 默认开启)，随时可以 Ctrl+C 中断，下次自动跳过已完成的视频。

#### 输出文件格式与目录结构

**保存位置**: 由 `--output_dir` 指定，每个视频生成一个独立的 `.pt` 文件。

```
{output_dir}/
├── abc123.pt          # video_id = "abc123"
├── def456.pt          # video_id = "def456"
├── xyz789.pt
└── ...                # 每个视频一个文件，总数 = 训练集中不重复视频数
```

**video_id 命名规则**: 取视频文件名（去掉扩展名）。例如视频路径为 `/path/to/videos/abc123.mp4` → `video_id = "abc123"` → 保存为 `{output_dir}/abc123.pt`。

**保存方式**: `torch.save(dict, path)` — 每个 `.pt` 文件是一个 Python **字典 (dict)**。

**文件内容**:

```python
# 不加 --save_pose (默认，用于蒸馏):
{
    "camera_features": torch.Tensor  # shape (S, 2048), dtype float16
}

# 加 --save_pose (同时保存解码后的内外参):
{
    "camera_features": torch.Tensor,  # shape (S, 2048), dtype float16
    "pose_enc": torch.Tensor,         # shape (S, 9), dtype float32
}
```

其中 **S = 该视频实际抽取的帧数**（由 fps × 视频时长决定，上限 max_frames=100，下限 4）。

**读取示例**:

```python
import torch

# 读取单个视频的 cache
data = torch.load("/path/to/vggt_cache/abc123.pt", map_location="cpu")

camera_features = data["camera_features"]  # shape (S, 2048), float16
print(f"帧数: {camera_features.shape[0]}, 特征维度: {camera_features.shape[1]}")

# 如果有 pose
if "pose_enc" in data:
    pose = data["pose_enc"]  # shape (S, 9), float32
    translation = pose[:, 0:3]   # 每帧的 3D 平移
    quaternion = pose[:, 3:7]    # 每帧的四元数旋转
    fov = pose[:, 7:9]           # 每帧的视场角
```

**`camera_features` (S, 2048) 说明**:
- 来自 Aggregator 最后一层 (layer 23) 的 camera token
- 2048 = frame_attention_output(1024) + global_attention_output(1024) 拼接
- 包含模型对多帧几何关系的全部理解（丰富的空间特征）
- 用于 CamDistill 蒸馏 和 CamInject 注入

**`pose_enc` (S, 9) 说明**:

| 维度 | 含义 | 类型 | 说明 |
|------|------|------|------|
| `[0:3]` | Translation | 外参 | 绝对平移 (世界坐标系下的相机位置) |
| `[3:7]` | Quaternion | 外参 | 四元数旋转 (相机朝向) |
| `[7:9]` | Field of View | 内参 | 水平/垂直视场角, 单位 radian, 经 ReLU+0.01 保正 |

- 是 Camera Head 对 `camera_features` 的 9D 解码结果
- 用于任意需要相机位姿数字的下游任务（比如转成欧拉角文本注入 prompt）
- 蒸馏时不需要（蒸馏直接用 2048D features 更信息丰富）

#### 训练时的读取方式

训练只跑 1 epoch，每个视频的 `.pt` 文件在训练过程中被读取一次。训练代码通过 video_id 直接 `torch.load` 对应文件：

```python
# camdistill_loss.py 中的逻辑 (无需手动调用):
cache_path = f"{VGGT_CACHE_DIR}/{video_id}.pt"
data = torch.load(cache_path, map_location="cpu")
features = data["camera_features"]  # (S, 2048) → .to(gpu) 用完释放
```

- 每个文件 ~400KB，读取耗时 ~5-50ms (ceph)
- 167k 样本累计 I/O 约 **14-28 分钟**，分散在整个训练过程中
- **不占用额外 GPU 显存**（加载到 CPU，只有当前 batch 的小 tensor 临时到 GPU）
- **不占用大量 CPU 内存**（读完即用即释放，不缓存）

```bash
# 完整工作流 (8卡并行提取 + 训练)
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl camera_movement_sft/train_data/camera_movement_train_human_all_167k.jsonl \
    --output_dir /path/to/vggt_cache/ \
    --teacher vggt \
    --num_gpus 8

VGGT_CACHE_DIR=/path/to/vggt_cache bash camera_movement_sft/train_camdistill.sh qwen3vl-8b
```

### 启动训练

```bash
# 方案 0: 普通 SFT (baseline, 无 camera token)
bash camera_movement_sft/train.sh qwen3vl-8b

# 方案 1: CamDistill 蒸馏（推理时不需要 VGGT，推荐）
VGGT_CACHE_DIR=/group/40009/dazhaodu/vggt_cache \
CAMDISTILL_DEPTH=6 \
bash camera_movement_sft/train_camdistill.sh qwen3vl-8b

# 方案 2: CamInject（实现简单, 推理时需要 VGGT）
VGGT_CACHE_DIR=/group/40009/dazhaodu/vggt_cache \
bash camera_movement_sft/train_caminject.sh qwen3vl-8b

# 使用 VGGT-Omega cache（只需换目录，代码不变）
VGGT_CACHE_DIR=/group/40009/dazhaodu/vggt_omega_cache \
bash camera_movement_sft/train_camdistill.sh qwen3vl-8b

# 其他模型
VGGT_CACHE_DIR=/path/to/cache bash camera_movement_sft/train_camdistill.sh qwen35-9b
VGGT_CACHE_DIR=/path/to/cache bash camera_movement_sft/train_caminject.sh qwen35-4b
```

**CamDistill 额外环境变量**:
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CAMDISTILL_DEPTH` | 6 | CameraTokenModule 交替注意力层数 |
| `CAMDISTILL_LAMBDA` | 0.3 | 蒸馏 loss 权重 (L = L_sft + λ·L_distill) |
| `CAMDISTILL_STRICT_CACHE` | 1 | 1=teacher cache 缺失/损坏直接 raise; 0=warn 后跳过蒸馏 |
| `VGGT_CACHE_DIR` | (必填) | 预提取 cache 目录 |

**CamInject 额外环境变量** (fail-fast 配置, 防止 silent fallback 到零特征):
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VGGT_CACHE_DIR` | (必填) | 预提取 cache 目录 |
| `CAMINJECT_STRICT_IDS` | 1 | 1=video_ids 数量与 video 不匹配直接 raise |
| `CAMINJECT_STRICT_CACHE` | 1 | 1=cache miss/损坏/NaN 直接 raise |
| `CAMINJECT_MAX_MISS_RATIO` | 0.0 | cache miss 比例阈值 (>阈值时 raise), 0.0=零容忍 |
| `CAMINJECT_MIN_RATIO_SAMPLES` | 32 | 至少跑过多少样本才检查 miss ratio |
| `CAMINJECT_LOG_EVERY` | 50 | 每多少步打印一次 cache 命中率统计 |

所有方案均支持 4 个模型：`qwen3vl-4b`, `qwen3vl-8b`, `qwen35-4b`, `qwen35-9b`
**详细文档**:
- `CamDistill_注入实现报告.md` — Camera Token 注入的详细实现报告 (5处修改说明)
- `VGGT_架构详解.md` — VGGT/VGGT-Omega 模型架构分析
- `VG-LLM_参考实现.md` — VG-LLM 参考实现分析
- `SpatialStack_参考实现.md` — SpatialStack 参考实现分析

---

## 常见问题

### Q: 显存不足？

```bash
# 方案1: 使用 ZeRO3
DEEPSPEED_STAGE=zero3 bash camera_movement_sft/train.sh qwen35-9b

# 方案2: 减少视频帧数
FPS_MAX_FRAMES=50 bash camera_movement_sft/train.sh qwen35-9b

# 方案3: 减小 batch + 增大梯度累积
PER_DEVICE_BATCH_SIZE=1 GRADIENT_ACCUMULATION=4 bash camera_movement_sft/train.sh qwen3vl-8b
```

### Q: Qwen3.5 视频训练卡住？

decord 后端可能导致视频读取卡住，使用 torchcodec 替代：

```bash
pip install torchcodec
export FORCE_QWENVL_VIDEO_READER=torchcodec
```

### Q: 如何切换训练数据版本？

```bash
# 使用均衡版（下采头部 + 上采尾部类别）
DATASET_PATH=/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/train_data/camera_movement_train_human_excluded_balanced_114k.jsonl \
bash camera_movement_sft/train.sh qwen3vl-8b
```

---

## 与原项目（ms-swift 3.x）的主要区别

| 项目 | 原版 (swift 3.12.4) | 本版 (swift 4.3.0) |
|------|---------------------|---------------------|
| 位置 | `/group/40059/yyjyu/code/cv/swift-3.12.4/` | `/group/40009/dazhaodu/ms-swift/` |
| 全参数微调参数 | `--train_type full` | `--tuner_type full` |
| 模型来源 | 本地路径 (`/data/Qwen3-VL-8B-Instruct`) | HuggingFace ID + `--use_hf true` |
| 支持模型 | 仅 Qwen3-VL-8B | 4 种模型（统一脚本切换） |
| 数据 | 相同 | 相同（软链接引用） |
| 视频 | 相同 | 相同路径 |
