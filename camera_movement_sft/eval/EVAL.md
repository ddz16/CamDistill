# Benchmark 评测指南

外部 benchmark 评测，统一三段式：`prepare → swift infer → evaluate`。

支持的 benchmark：**CameraBench**（Yes/No 二分类）、**MultiCamVideo**（4 选 1 MCQ）。

---

## 数据状态（已下载/处理）

| Benchmark | 标注 | 视频 |
|---|---|---|
| CameraBench | `/group/40009/dazhaodu/t2v_metrics/camerabench/data/binary_classification/*.jsonl`（15 split, 共 ~7679 条） | `/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/CameraBench`（969 mp4，命中率 100%） |
| MultiCamVideo | `cam-motion.jsonl`（12274 条 MCQ）+ `{train,val,test}.json` 三份 manifest | tar shard 已解压到 `/apdcephfs_gy2/.../Multicam_Benchmark/extracted/{train,val,test}/`（jpg 序列），prepare 时会拼成 mp4 |

MultiCam 的 prepared 数据：
- `prepared/multicam_all/`  — 12274 条（train+val+test 全集，自家用）
- `prepared/multicam_test/` — 1228 条（标准 test split，与 leaderboard 可比）

---

## 一键评测

```bash
cd /group/40009/dazhaodu/ms-swift

# 官方原版 Qwen3-VL-4B-Instruct
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh qwen3vl-4b

# 官方原版 Qwen3-VL-8B-Instruct
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh qwen3vl-8b

# chancharikm cam-motion 微调模型（Qwen2.5-VL 基础上微调，从 HF 直接拉）
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh cam-motion-7b
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh cam-motion-72b

# 自家训练的 checkpoint（绝对路径）
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh \
    /group/40009/dazhaodu/ms-swift/output/camera_sft_qwen3vl_4b_caminject/v23-20260627-023518/checkpoint-1260
```

结果落到：`camera_movement_sft/eval/benchmarks/results/<TAG>/{camerabench,multicam}/`。

### 支持的模型短名

| 短名 | 实际模型 | 来源 |
|---|---|---|
| `qwen3vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` | HF 官方原版 |
| `qwen3vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | HF 官方原版 |
| `qwen2.5vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF 官方原版 |
| `qwen2.5vl-72b` | `Qwen/Qwen2.5-VL-72B-Instruct` | HF 官方原版 |
| `cam-motion-7b` | `chancharikm/qwen2.5-vl-7b-cam-motion` | HF，Qwen2.5-VL 微调 |
| `cam-motion-72b` | `chancharikm/qwen2.5-vl-72b-cam-motion` | HF，Qwen2.5-VL 微调 |
| `<绝对路径>` | 该 checkpoint | 自家训练产物 |

> 短名/checkpoint 都统一走 `--use_hf true` + 同一套 prompt 模板，跨模型可比。

### 常用环境变量

| 变量 | 默认 | 用途 |
|---|---|---|
| `BENCHMARKS` | `"camerabench multicam"` | 子集，如 `BENCHMARKS=camerabench` 只跑一个 |
| `MULTICAM_PREPARED` | `prepared/multicam_all` | 改成 `multicam_test` 即用 1228 test split |
| `TAG` | 从模型路径派生 | 结果目录名后缀 |
| `FORCE_PREPARE` / `FORCE_INFER` / `FORCE_EVAL` | 0 | 强制重跑某一步（默认中间产物存在就跳过） |
| `NPROC_PER_NODE` | 8 | 数据并行的 GPU 数（每卡各跑一份完整模型）；`MODEL_PARALLEL=1` 时被强制为 1 |
| `MODEL_PARALLEL` | 72B 自动为 1，其余 0 | 1=单模型跨卡切分（`device_map auto`），0=数据并行。见下文「72B 多卡」 |

例：

```bash
# 只 camerabench
BENCHMARKS=camerabench bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh qwen3vl-4b

# multicam test 1228 条
MULTICAM_PREPARED=$(realpath camera_movement_sft/eval/benchmarks/prepared/multicam_test) \
BENCHMARKS=multicam \
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh qwen3vl-4b
```

### 72B 多卡（单模型跨卡切分）

72B 模型 bf16 约 144GB，**单卡（H20 ~96GB）放不下**。默认的数据并行（`NPROC_PER_NODE=8`）是
每个 torchrun 进程在单卡上各加载一份完整模型，72B 会直接 OOM。

因此 `cam-motion-72b` / `qwen2.5vl-72b`（以及路径含 `72b` 的 checkpoint）会**自动**切到单模型
跨卡切分模式：

- 加 `--device_map auto`，一个进程把模型按层分散到全部可见 GPU；
- 强制 `NPROC_PER_NODE=1`（关掉数据并行，否则每进程仍各加载一份完整模型 → OOM）。

banner 里会打印当前模式（`model-parallel` / `data-parallel`）。用法不变：

```bash
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh cam-motion-72b
bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh qwen2.5vl-72b

# 手动覆盖：给某个大 checkpoint 强开跨卡
MODEL_PARALLEL=1 bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh \
    output/xxx/checkpoint-5000
```

> ⚠️ **速度**：`device_map auto` 是按层切分（pipeline），同一时刻基本只有一张卡在算，对 12k+
> 视频偏慢。真正快的是 vLLM 张量并行（`--vllm_tensor_parallel_size 8`），但**当前环境未安装
> vLLM**，故先用 device_map 这条能直接跑通、且与其余模型同一套 pipeline（logprobs/解码/eval
> 都不变）的路径。若要 vLLM 提速需另装 vLLM 并加后端分支。

---

## Prompt 模板

### CameraBench
```
<video>{question} Please only answer Yes or No.
```
`{question}` 来自标注 jsonl 的 `question` 字段；包含 "(not tilting up)" / "(not zooming in)" 这类消歧括注，逼模型区分位移 vs 旋转 vs 焦距变化。

### MultiCamVideo
```
<video>Identify the camera motion depicted in the video using standard cinematographic terminology.
Options:
(A) ...
(B) ...
(C) ...
(D) ...

Please directly output the answer letter (A, B, C, or D).
```
原题来自 `cam-motion.jsonl` 的 `human.value`，仅去 `<video>\n` 的换行 + 追加"只输出字母"指令；4 个选项保持原文。

---

## 报告指标

### CameraBench（`eval_camerabench.py`）
- per-split: **Accuracy / Precision / Recall / F1 / N**
- 汇总:
  - **Overall**：全部样本 micro accuracy
  - **Mean (macro avg)**：15 个 split 的 accuracy 平均
- ⚠️ 我们用生成式硬决策 acc，**与论文 mAP（VQAScore = P(Yes) logprob）不可直接对位**，但模型相对强弱关系一致

### MultiCamVideo（`eval_multicam.py`）
- 主指标：**Overall Accuracy**（correct / total）
- 切片：per-split (train/val/test) / per-GT-letter (A/B/C/D 各自命中率) / Pred letter 分布

---

## 文件结构

```
camera_movement_sft/eval/benchmarks/
├── extract_multicam.sh        # 解压 MultiCam tar shard（已跑过）
├── prepare_camerabench.py     # CameraBench prepare（生成 infer/meta jsonl）
├── prepare_multicam.py        # MultiCam prepare（解 tar/已解压目录都支持）
├── eval_camerabench.py        # CameraBench evaluate
├── eval_multicam.py           # MultiCam evaluate
├── run_benchmarks.sh          # 一键串起两个 benchmark
├── prepared/                  # prepare 产出的中间 jsonl
│   ├── camerabench/
│   ├── multicam_all/
│   └── multicam_test/
└── results/                   # 评测产出
    └── <TAG>/{camerabench,multicam}/{infer_results.jsonl, *_results.json}
```

---

## 单步骤命令（如果要拆开跑）

### CameraBench
```bash
# 1. prepare（首次会自动跑；存在则跳过）
python camera_movement_sft/eval/benchmarks/prepare_camerabench.py
# → prepared/camerabench/{camerabench_infer.jsonl, camerabench_meta.jsonl}

# 2. infer
swift infer \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --val_dataset camera_movement_sft/eval/benchmarks/prepared/camerabench/camerabench_infer.jsonl \
    --result_path camera_movement_sft/eval/benchmarks/results/qwen3vl-4b-base/camerabench/infer_results.jsonl \
    --temperature 0 --max_new_tokens 16 --max_batch_size 16

# 3. evaluate
python camera_movement_sft/eval/benchmarks/eval_camerabench.py \
    --pred_file camera_movement_sft/eval/benchmarks/results/qwen3vl-4b-base/camerabench/infer_results.jsonl \
    --meta_file camera_movement_sft/eval/benchmarks/prepared/camerabench/camerabench_meta.jsonl
```

### MultiCamVideo
```bash
# 1. prepare（已跑过的话跳过）
python camera_movement_sft/eval/benchmarks/prepare_multicam.py \
    --dataset_root /apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/Multicam_Benchmark \
    --split all \
    --extracted_dir /apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/Multicam_Benchmark/extracted \
    --video_out /apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/Multicam_Benchmark/all_videos \
    --output_dir camera_movement_sft/eval/benchmarks/prepared/multicam_all

# 2. infer
swift infer \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --val_dataset camera_movement_sft/eval/benchmarks/prepared/multicam_all/multicam_infer.jsonl \
    --result_path camera_movement_sft/eval/benchmarks/results/qwen3vl-4b-base/multicam/infer_results.jsonl \
    --temperature 0 --max_new_tokens 32 --max_batch_size 16

# 3. evaluate
python camera_movement_sft/eval/benchmarks/eval_multicam.py \
    --pred_file camera_movement_sft/eval/benchmarks/results/qwen3vl-4b-base/multicam/infer_results.jsonl \
    --meta_file camera_movement_sft/eval/benchmarks/prepared/multicam_all/multicam_meta.jsonl
```

---

## 估算耗时（8×A100，bf16）

| Benchmark | 样本 | 估时 |
|---|---|---|
| CameraBench | 7679 | 30–45 min |
| MultiCam (test) | 1228 | 5–8 min |
| MultiCam (all) | 12274 | 50–70 min |
