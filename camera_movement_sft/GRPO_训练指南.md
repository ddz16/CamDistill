# GRPO 强化学习训练指南

## 1. 概述

GRPO (Group Relative Policy Optimization) 是一种无需 Critic Model 的强化学习算法，通过**组内相对排名**来估计 advantage。

**核心思想**: 对每个 prompt 生成 N 个候选答案，用 Reward 函数评分，分数高于组内平均的答案获得正 advantage（强化），低于平均的获得负 advantage（抑制）。

**SFT → GRPO 的完整 pipeline**:
```
原始 Qwen3-VL-8B → SFT (学习格式+基础能力) → GRPO (精调决策边界)
                         ↓                          ↓
                   学会输出 JSON              提升 F1, 减少格式错误
                   学会识别运镜              修正 SFT 过拟合
```

---

## 2. 快速开始

```bash
cd /group/40009/dazhaodu/ms-swift
conda activate cm

# Step 1: SFT (已完成)
bash camera_movement_sft/train.sh qwen3vl-8b
# → output/camera_sft_qwen3vl_8b/v0-xxx/checkpoint-5000

# Step 2: 数据准备 (首次自动执行, 也可手动)
python camera_movement_sft/rl/prepare_grpo_data.py
# → camera_movement_sft/train_data/camera_movement_grpo_167k.jsonl

# Step 3: GRPO (在 SFT 模型上继续)
bash camera_movement_sft/rl/train_grpo.sh \
    output/camera_sft_qwen3vl_8b/v0-xxx/checkpoint-5000

# Step 4: 评测
TRAIN_OUTPUT_DIR=output/camera_grpo_v0-xxx/v0-yyy \
bash camera_movement_sft/eval/run_batch_checkpoints.sh
```

**就这几条命令。数据准备脚本会在首次训练时自动执行（如果 GRPO 数据不存在）。**

---

## 3. Reward 函数设计

### 3.1 总体策略

使用 4 个 reward 函数组合，各司其职：

| Reward | 权重 | 作用 | 范围 |
|--------|------|------|------|
| `camera_accuracy` | 1.0 | 核心: 运镜标签 F1 | [0, 1] |
| `camera_format` | 0.3 | 格式: JSON 合规性 | [0, 1] |
| `camera_segment_iou` | 0.2 | 时间: segment 对齐 | [0, 1] |
| `repetition` | 0.1 | 多样性: 防止重复输出 | [-1, 0] |

**最终 reward** = 1.0×accuracy + 0.3×format + 0.2×iou + 0.1×repetition

### 3.2 CameraMovementAccuracy (核心)

逐帧对比模型输出与 GT 的 basic_movement 复合标签（type + direction），计算 F1：

```
模型输出: {"segments": [{"start_time": 0, "end_time": 3, "basic_movement": [{"type": "Pan", "direction": "left"}]}]}
GT:      {"segments": [{"start_time": 0, "end_time": 3, "basic_movement": [{"type": "Pan", "direction": "left"}]}]}

逐帧(每0.5秒采样):
  t=0.0: pred={"Pan_left"}, gt={"Pan_left"} → TP=1
  t=0.5: pred={"Pan_left"}, gt={"Pan_left"} → TP=1
  ...

F1 = 2*P*R / (P+R) = 1.0 → reward = 1.0
```

**评分规则**:
- 完美匹配: reward ≈ 1.0
- 方向错误 (Pan_left vs Pan_right): reward ≈ 0.5
- 类型错误: reward ≈ 0.0
- 输出非法 JSON: reward = 0.0

### 3.3 CameraFormatReward (格式)

检查输出是否为合法 JSON 并符合 schema，**防止 GRPO 训练中格式退化**：

```
评分 (累加):
  能解析为 JSON        → +0.3
  有 segments 数组     → +0.2
  有 start/end_time    → +0.2
  有 basic_movement    → +0.2
  type 属于闭集        → +0.1
```

**为什么需要**: GRPO 可能让模型为了追求高 accuracy 而输出不规范内容。format reward 保证格式稳定。

### 3.4 CameraSegmentIoU (时间)

评估时间段的对齐质量。对每个 GT segment，找 IoU 最大的 pred segment：

```
GT:   [0.0 -------- 3.0]
Pred: [0.5 ------- 3.5]

IoU = intersection / union = 2.5 / 3.5 = 0.71
```

**为什么需要**: accuracy 可能在时间段偏移时仍然得到较高分（因为逐帧采样），segment IoU 显式奖励精确的时间边界。

### 3.5 RepetitionPenalty (多样性)

使用 ms-swift 内置的 `repetition` reward，惩罚 n-gram 重复：
- 如果输出中有大量重复片段 → 负奖励
- 正常输出 → 0（不惩罚也不奖励）

---

## 4. 超参数说明

### GRPO 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_generations` | 8 | 每个 prompt 生成的候选数 (越多越稳定, 但越慢) |
| `temperature` | 0.9 | 生成温度 (0.7-1.0 合理, 太低没有多样性) |
| `beta` | 0.04 | KL 散度惩罚系数 (越大越保守) |
| `learning_rate` | 1e-6 | 学习率 (**必须远小于 SFT 的 1.5e-5**) |
| `max_completion_length` | 4096 | 最大生成 token 数 |
| `steps_per_generation` | 4 | 每隔 N 步重新生成候选 |
| `num_iterations` | 1 | 每次生成后训练几轮 |
| `epsilon` | 0.2 | PPO clip 下界 |
| `epsilon_high` | 0.28 | PPO clip 上界 |
| `scale_rewards` | none | 不做 reward 归一化 (运镜 reward 已在 [0,1]) |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `overlong_filter` | false | 不过滤超长输出 (视频模型输出可能较长) |
| `sleep_level` | 1 | vLLM colocate 模式的 GPU 内存释放策略 |

### vLLM 配置 (用于加速生成)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vllm_mode` | colocate | 生成和训练共用 GPU (单机最简方案) |
| `vllm_gpu_memory_utilization` | 0.5 | vLLM 占 GPU 显存比例 |
| `vllm_max_model_len` | 16384 | vLLM 最大序列长度 |
| `sleep_level` | 1 | 生成完毕后释放 vLLM 内存给训练 |

### 显存预估 (8×96GB)

```
训练 (DeepSpeed ZeRO-2): ~40GB/卡 (8B 模型 + 优化器)
vLLM 生成 (colocate):    ~40GB/卡 (模型副本 + KV cache)
合计:                     ~80GB/卡 → 96GB 卡足够
```

> 注: `sleep_level=1` 时，vLLM 在不生成时会释放大部分 GPU 显存给训练使用，
> 实际峰值不会同时占 80GB。

---

## 5. SFT → GRPO 数据策略

### 直接复用 SFT 数据 (需预处理)

ms-swift GRPO 要求数据中有独立的 `solution` 字段供 reward 函数使用。SFT 数据的 GT 在 `messages[assistant]` 中，需要提取出来：

```bash
# 自动转换 (train_grpo.sh 首次运行时会自动执行)
python camera_movement_sft/rl/prepare_grpo_data.py
```

转换后的格式:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<video>请分析..."},
    {"role": "assistant", "content": "{\"segments\": [...]}"}
  ],
  "videos": ["path/to/video.mp4"],
  "solution": "{\"segments\": [...]}"
}
```

**`solution` 字段** = assistant 最后一轮回复的副本。ms-swift GRPO 会：
1. 用 messages 构建 prompt（到 user 为止）
2. 模型自由生成新的 completions
3. reward 函数收到 `completions` + `solution` → 对比打分

### 为什么可以复用

- SFT: 模型被 **强制** 输出 reference answer (teacher forcing)
- GRPO: 模型 **自由** 生成答案, reward 函数评判好坏
- 同一个 prompt+video, 但学习方式完全不同
- GRPO 能修正 SFT 的过拟合（因为看到了自己的错误并从中学习）

---

## 6. 关于指令跟随能力

### SFT 后模型能否做选择题 VQA？

| 场景 | 答案 |
|------|------|
| 运镜识别 JSON 输出 | ✅ 正常工作 (SFT 专门训练的) |
| 简单选择题 (A/B/C/D) | ⚠️ 可能退化 (全参数微调偏移了) |
| 复杂对话/指令跟随 | ⚠️ 可能退化 |
| 其他视频理解任务 | ⚠️ 取决于偏移程度 |

### 保护措施

1. **Format Reward (0.3 权重)**: GRPO 中显式奖励 JSON 格式，防止格式退化
2. **低学习率 (1e-6)**: GRPO 只微调决策边界，不大幅改变模型
3. **KL 约束 (beta=0.04)**: 惩罚策略偏离 SFT 太远
4. **如需选择题能力**: 建议使用 LoRA 而非全参数 SFT，或在 SFT 数据中混入 5-10% 选择题样本

---

## 7. 训练监控

### 关键指标

| 指标 | 期望趋势 | 异常信号 |
|------|----------|----------|
| `reward/mean` | 稳步上升 | 突然跳高 (reward hacking) |
| `reward/std` | 先高后低 | 持续为 0 (mode collapse) |
| `kl` | 缓慢增长 | 快速增大 (>10 说明偏移过大) |
| `completions/mean_length` | 稳定 | 暴涨 (长度 hacking) |
| `camera_accuracy` | 上升 | — |
| `camera_format` | 接近 1.0 | 下降 (格式退化) |

### 常见问题

**Q: Reward 不增长?**
- 学习率太小 → 增大到 2e-6
- 温度太低 → 增大到 1.0
- num_generations 太少 → 增大到 16

**Q: 格式退化 (camera_format 下降)?**
- 增大 format reward 权重: `REWARD_WEIGHTS="1.0 0.5 0.2 0.1"`
- 减小 beta: 让模型更自由探索
- 降低温度: 减少极端输出

**Q: KL 散度过大?**
- 增大 beta
- 减小学习率
- 更频繁保存 checkpoint

**Q: Mode Collapse (所有输出一样)?**
- 增大温度
- 增大 num_generations
- 检查 reward 函数是否有 bug (是否总是给同一个模式最高分)

---

## 8. 环境变量速查

```bash
# 完整示例 (带所有可调参数)
NUM_GENERATIONS=8 \
TEMPERATURE=0.9 \
LEARNING_RATE=1e-6 \
BETA=0.04 \
REWARD_WEIGHTS="1.0 0.3 0.2 0.1" \
VLLM_GPU_UTIL=0.5 \
bash camera_movement_sft/rl/train_grpo.sh \
    output/camera_sft_qwen3vl_8b/v0-xxx/checkpoint-5000
```

---

## 9. 文件结构

```
camera_movement_sft/rl/
├── rewards.py              # Reward 函数定义 (4 个 ORM, 通过 --external_plugins 加载)
├── prepare_grpo_data.py    # SFT → GRPO 数据转换 (添加 solution 字段)
└── train_grpo.sh           # GRPO 训练脚本 (一条命令启动)

camera_movement_sft/train_data/
└── camera_movement_grpo_167k.jsonl  # 转换后的 GRPO 数据 (首次训练自动生成)

注册的 4 个自定义 reward (通过 --external_plugins 加载到 orms 字典):
  - camera_accuracy      (核心: Frame-level F1, 逐帧对比运镜标签)
  - camera_format        (格式: JSON schema 合规性检查)
  - camera_segment_iou   (时间: Segment temporal IoU)
  - camera_segment_count (段数: 防过多/过少 segments)
+ ms-swift 内置:
  - repetition           (多样性: n-gram 重复惩罚)
```

---

## 10. 与其他方案的组合

```
              ┌─── 普通 SFT ──────── GRPO ──── 最终模型 (推荐)
              │
原始模型 ────┤─── CamDistill SFT ─── GRPO ──── CamDistill + GRPO
              │
              └─── CamInject SFT ── GRPO ──── CamInject + GRPO
```

所有 SFT 变体都可以接 GRPO。只需将对应的 checkpoint 路径传给 `train_grpo.sh`。
