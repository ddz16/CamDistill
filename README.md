# CamDistill

Camera movement understanding via knowledge distillation from VGGT.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [User Configuration](#2-user-configuration)
3. [Quick Start with Pretrained Checkpoints](#3-quick-start-with-pretrained-checkpoints)
4. [Prepare Training Data](#4-prepare-training-data)
5. [Extract VGGT Features](#5-extract-vggt-features)  *(CamDistill / CamInject only)*
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)

---

## 1. Environment Setup

> Tested with **PyTorch 2.10.0 + CUDA 12.8**.

```bash
conda create -n cm python=3.12 -y
conda activate cm

# Install ms-swift (this repo) in editable mode
cd /path/to/CamDistill
pip install uv
uv pip install -e . --torch-backend=auto

# Core dependencies
uv pip install vllm==0.19.0
uv pip install "flash-linear-attention>=0.4.2" --no-build-isolation
uv pip install "causal-conv1d" --no-build-isolation
uv pip install deepspeed
uv pip install "qwen_vl_utils>=0.0.14"
uv pip install decord
uv pip install torchcodec
conda install "ffmpeg" -y

# Logging / metrics
uv pip install wandb
uv pip install scikit-learn

# Attention kernel (match your CUDA version)
uv pip install "flash-attn==2.8.3" --no-build-isolation --no-cache-dir

# Transformers
uv pip install transformers==5.8.1
```

---

## 2. User Configuration

All private paths and tokens are managed in a single file.

```bash
cp camera_movement_sft/env.sh.example camera_movement_sft/env.sh
```

Then edit `camera_movement_sft/env.sh` and fill in your values:

| Variable | Description |
|---|---|
| `http_proxy` / `https_proxy` | Network proxy (leave empty to disable) |
| `HF_HOME` | HuggingFace cache directory |
| `HF_TOKEN` | HuggingFace access token |
| `WANDB_API_KEY` | WandB API key (leave empty to disable WandB) |
| `VGGT_REPO` | Local path to the VGGT repo (for feature extraction) |
| `VGGT_OMEGA_REPO` | Local path to the VGGT-Omega repo (for feature extraction) |
| `VGGT_CACHE_DIR` | Pre-extracted feature cache directory (CamDistill / CamInject) |
| `DATASET_PATH` | Training dataset path (optional; defaults to the built-in 50k set) |


---

## 3. Quick Start with Pretrained Checkpoints

We provide pretrained checkpoints on HuggingFace. You can directly download and run inference
**without any training**.

| Model | HF Repo | Description |
|---|---|---|
| CamSFT-4B | `ddz16/CamSFT-4B` | Standard SFT baseline |
| CamDistill-4B | `ddz16/CamDistill-4B` | SFT + Camera Token Distillation |
| CamInject-4B | `ddz16/CamInject-4B` | Frozen VGGT token injection |

### 3.1 Evaluate on Our Benchmark

Make sure the benchmark test set is placed at the path configured in your environment,
then run:

**CamSFT-4B:**
```bash
BASE_MODEL=ddz16/CamSFT-4B MODEL_TAG=camsft_4b \
bash camera_movement_sft/eval/run_batch_checkpoints.sh \
    /path/to/OurBenchmark/youtube_benchmark_subset.jsonl
```

**CamDistill-4B:**
```bash
BASE_MODEL=ddz16/CamDistill-4B MODEL_TAG=camdistill_4b \
bash camera_movement_sft/eval/run_batch_checkpoints.sh \
    /path/to/OurBenchmark/youtube_benchmark_subset.jsonl
```

**CamInject-4B** (requires VGGT features, always runs online):
```bash
USE_CAMINJECT=1 VGGT_TEACHER_TYPE=vggt_omega \
BASE_MODEL=ddz16/CamInject-4B MODEL_TAG=caminject_4b \
bash camera_movement_sft/eval/run_batch_checkpoints.sh \
    /path/to/OurBenchmark/youtube_benchmark_subset.jsonl
```

Results are saved to `camera_movement_sft/eval/eval_results/<run_name>/eval_results.json`.

### 3.2 Inference on a Custom Video

You can run camera movement analysis on any video with a custom prompt.

**CamSFT-4B / CamDistill-4B** (standard models, no plugin needed):
```bash
swift infer \
    --model ddz16/CamSFT-4B \
    --use_hf true \
    --infer_backend pt \
    --stream true
# Then enter your query interactively, e.g.:
# <video>/path/to/your/video.mp4</video>
# Describe the camera movement in this video.
```

Or pass a single query non-interactively:
```bash
swift infer \
    --model ddz16/CamDistill-4B \
    --use_hf true \
    --infer_backend pt \
    --val_dataset '[{"messages": [{"role": "user", "content": "<video>Describe the camera movement in this video."}], "videos": ["/path/to/your/video.mp4"]}]'
```

**CamInject-4B** (requires the CamInject plugin, always runs VGGT online):
```bash
VGGT_TEACHER_TYPE=vggt_omega \
swift infer \
    --model ddz16/CamInject-4B \
    --model_type qwen3_vl_caminject \
    --external_plugins camera_movement_sft/plugins/camdistill_plugin.py \
    --use_hf true \
    --infer_backend pt \
    --val_dataset '[{"messages": [{"role": "user", "content": "<video>Describe the camera movement in this video."}], "videos": ["/path/to/your/video.mp4"]}]'
```

---

## 4. Prepare Training Data

Place the training JSONL file at:

```
camera_movement_sft/train_data/camera_movement_train_diverse_50k_en.jsonl
```

Or set `DATASET_PATH` in `env.sh` to point to your own dataset.

Place the benchmark test set at:

```
/group/40009/dazhaodu/OurBenchmark/youtube_benchmark_subset.jsonl
```

---

## 5. Extract VGGT Features

> **Required for CamDistill and CamInject (cache mode) only. Skip for plain SFT.**

Make sure `VGGT_REPO` / `VGGT_OMEGA_REPO` are set in `env.sh`.

**VGGT (default):**

```bash
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl camera_movement_sft/train_data/camera_movement_train_diverse_50k_en.jsonl \
    --output_dir /path/to/vggt_cache/ \
    --teacher vggt \
    --num_gpus 8
```

**VGGT-Omega (recommended):**

```bash
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl camera_movement_sft/train_data/camera_movement_train_diverse_50k_en.jsonl \
    --output_dir /path/to/vggt_omega_cache/ \
    --teacher vggt_omega \
    --save_pose \
    --num_gpus 8
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--teacher` | `vggt` | `vggt` or `vggt_omega` |
| `--output_dir` | — | Directory to save `.pt` feature files |
| `--num_gpus` | `1` | Number of GPUs for parallel extraction |
| `--fps` | `5` | Frame sampling rate |
| `--max_frames` | `100` | Maximum frames per video |
| `--save_pose` | off | Also save 9D pose encoding (required for VGGT-Omega distillation) |
| `--skip_existing` | off | Resume an interrupted run |

After extraction, set `VGGT_CACHE_DIR` in `env.sh` to the `--output_dir` path.

---

## 6. Training

All training scripts are under `camera_movement_sft/`. Activate the conda environment first:

```bash
conda activate cm
```

### 6.1 CamSFT — Standard Supervised Fine-Tuning

```bash
bash camera_movement_sft/train.sh qwen3vl-4b
# or
bash camera_movement_sft/train.sh qwen3vl-8b
```

### 6.2 CamDistill — SFT + Camera Token Distillation

Requires pre-extracted VGGT features (`VGGT_CACHE_DIR` must be set).

```bash
VGGT_CACHE_DIR=/path/to/vggt_omega_cache \
bash camera_movement_sft/train_camdistill.sh qwen3vl-4b
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `VGGT_CACHE_DIR` | — | Pre-extracted feature cache (**required**) |
| `VGGT_TEACHER_TYPE` | `vggt` | `vggt` or `vggt_omega` |
| `CAMDISTILL_LAMBDA` | `0.3` | Distillation loss weight |
| `CAMDISTILL_DEPTH` | `6` | Number of Camera Token Module layers |
| `CAMDISTILL_WARMUP_STEPS` | `200` | Steps before distillation loss is enabled |
| `CAMERA_TOKEN_INSERT_POSITION` | `front` | Insert camera token at `front` or `back` of each frame |

### 6.3 CamInject — Inject Frozen VGGT Tokens into LLM

**Cache mode** (recommended, requires pre-extracted features):

```bash
VGGT_MODE=cache \
VGGT_CACHE_DIR=/path/to/vggt_omega_cache \
bash camera_movement_sft/train_caminject.sh qwen3vl-4b
```

**Online mode** (no pre-extraction, VGGT runs per batch):

```bash
VGGT_MODE=online \
VGGT_TEACHER_TYPE=vggt_omega \
bash camera_movement_sft/train_caminject.sh qwen3vl-4b
```

### Common Training Options

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | GPUs to use |
| `NPROC_PER_NODE` | `8` | Number of processes |
| `NUM_EPOCHS` | `2` | Training epochs |
| `LEARNING_RATE` | `2e-5` (4B) / `1.5e-5` (8B) | Learning rate |
| `PER_DEVICE_BATCH_SIZE` | `2` | Batch size per GPU |
| `GRADIENT_ACCUMULATION` | `4` | Gradient accumulation steps |
| `OUTPUT_DIR` | `output/camera_sft_<model>` | Output directory |

---

## 7. Evaluation

### 7.1 Our Benchmark

Evaluate a trained checkpoint on the internal benchmark:

```bash
TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_4b/v0-<timestamp> \
bash camera_movement_sft/eval/run_batch_checkpoints.sh \
    /group/40009/dazhaodu/OurBenchmark/youtube_benchmark_subset.jsonl
```

By default only the **last checkpoint** is evaluated. To evaluate all checkpoints:

```bash
ONLY_LAST=false TRAIN_OUTPUT_DIR=... \
bash camera_movement_sft/eval/run_batch_checkpoints.sh <testset.jsonl>
```

To evaluate a **base model** (no fine-tuning):

```bash
BASE_MODEL=Qwen/Qwen3-VL-4B-Instruct MODEL_TAG=base_qwen3vl_4b \
bash camera_movement_sft/eval/run_batch_checkpoints.sh <testset.jsonl>
```

To evaluate a **CamInject checkpoint** (always online mode):

```bash
USE_CAMINJECT=1 \
VGGT_TEACHER_TYPE=vggt_omega \
TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_4b_caminject/v0-<timestamp> \
bash camera_movement_sft/eval/run_batch_checkpoints.sh <testset.jsonl>
```

Results are saved to `camera_movement_sft/eval/eval_results/<run_name>/eval_results.json`.
