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

### Clone the VGGT Repositories

`vggt` and `vggt_omega` are not available on PyPI and must be cloned from GitHub. After cloning, set the corresponding paths in `env.sh` (see next section).

**VGGT:**
```bash
git clone https://github.com/facebookresearch/vggt.git /path/to/vggt
# Then in env.sh:
# export VGGT_REPO="/path/to/vggt"
```

**VGGT-Omega:**
```bash
git clone https://github.com/facebookresearch/vggt-omega.git /path/to/vggt_omega
# Then in env.sh:
# export VGGT_OMEGA_REPO="/path/to/vggt_omega"
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
| `DATASET_PATH` | Training dataset path (optional; overrides the default `train_data/train_swift.jsonl`) |


---

## 3. Quick Start with Pretrained Checkpoints

We provide pretrained checkpoints on HuggingFace. You can directly download and run inference.

| Model | HF Repo | Description |
|---|---|---|
| CamSFT-4B | [`ddz16/CamSFT-4B`](https://huggingface.co/ddz16/CamSFT-4B) | Standard SFT baseline |
| CamDistill-4B | [`ddz16/CamDistill-4B`](https://huggingface.co/ddz16/CamDistill-4B) | Camera Token Distillation |
| CamInject-4B | [`ddz16/CamInject-4B`](https://huggingface.co/ddz16/CamInject-4B) | Frozen VGGT token injection |
| CamSFT-8B | [`ddz16/CamSFT-8B`](https://huggingface.co/ddz16/CamSFT-8B) | Standard SFT baseline |
| CamDistill-8B | [`ddz16/CamDistill-8B`](https://huggingface.co/ddz16/CamDistill-8B) | Camera Token Distillation |
| CamInject-8B | [`ddz16/CamInject-8B`](https://huggingface.co/ddz16/CamInject-8B) | Frozen VGGT token injection |

### 3.1 Download the Benchmark Dataset

Our benchmark, **[CamChoreo](https://huggingface.co/datasets/ddz16/CamChoreo)** (4229 clips), is released on HuggingFace. The dataset is gated — first request access on the dataset page and
wait for approval, then download it (requires `HF_TOKEN`):

```bash
huggingface-cli download ddz16/CamChoreo \
    --repo-type dataset \
    --local-dir /path/to/CamChoreo

# Unzip the videos
unzip /path/to/CamChoreo/videos.zip -d /path/to/CamChoreo/
```

After downloading you will have:

```
/path/to/CamChoreo/
├── annotations.jsonl        # one record per clip (video_id / local_path / is_doubt / segments)
└── videos/<video_id>.mp4    # 4229 video clips
```

### 3.2 Evaluate on Our Benchmark

Make sure the benchmark test set is placed at the path configured in your environment,
then run:

**CamSFT-4B:**
```bash
BASE_MODEL=ddz16/CamSFT-4B MODEL_TAG=camsft_4b \
bash camera_movement_sft/eval/run_batch_checkpoints.sh \
    /path/to/CamChoreo/annotations.jsonl
```

**CamDistill-4B** (needs the CamDistill plugin; no online VGGT):
```bash
USE_CAMDISTILL=1 \
BASE_MODEL=ddz16/CamDistill-4B MODEL_TAG=camdistill_4b \
bash camera_movement_sft/eval/run_batch_checkpoints.sh \
    /path/to/CamChoreo/annotations.jsonl
```

**CamInject-4B** (requires VGGT features, always runs online):
```bash
USE_CAMINJECT=1 VGGT_TEACHER_TYPE=vggt_omega \
BASE_MODEL=ddz16/CamInject-4B MODEL_TAG=caminject_4b \
bash camera_movement_sft/eval/run_batch_checkpoints.sh \
    /path/to/CamChoreo/annotations.jsonl
```

Results are saved to `camera_movement_sft/eval/eval_results/<run_name>/eval_results.json`.

### 3.3 Inference on a Custom Video

Run camera-movement analysis on any video. `infer_single.py` uses the **official
prompt** (the same system/user prompt used for training and evaluation), so the model
outputs the full structured JSON — every camera-movement segment with its time span,
basic-movement type / direction / speed, and special techniques.

**CamSFT-4B** (standard model, no plugin):
```bash
python camera_movement_sft/infer_single.py \
    --model ddz16/CamSFT-4B \
    --video /path/to/your/video.mp4
```

**CamDistill-4B** (needs the plugin; camera tokens are generated internally, **no
online VGGT required**):
```bash
python camera_movement_sft/infer_single.py \
    --model ddz16/CamDistill-4B \
    --video /path/to/your/video.mp4 \
    --variant camdistill
```

**CamInject-4B** (needs the plugin **and** runs VGGT online):
```bash
VGGT_TEACHER_TYPE=vggt_omega \
python camera_movement_sft/infer_single.py \
    --model ddz16/CamInject-4B \
    --video /path/to/your/video.mp4 \
    --variant caminject
```

> Replace `-4B` with `-8B` for the larger checkpoints.

---

## 4. Prepare Training Data

Our internal training set is **not publicly released**, but it shares the exact same
JSON schema as our benchmark ([CamChoreo](https://huggingface.co/datasets/ddz16/CamChoreo)),
so you can train on your own data with no code changes.

Prepare your data as an annotation JSONL (one object per line, same schema as the benchmark):

```json
{"video_id": "xxx", "local_path": "./videos/xxx.mp4", "segments": [...]}
```

`local_path` is resolved relative to the JSONL's own directory — keep the videos in a
`videos/` folder next to it. Convert it to the ms-swift SFT format:

```bash
python camera_movement_sft/prepare_train_data.py \
    --input  /path/to/your_annotations.jsonl \
    --output camera_movement_sft/train_data/train_swift.jsonl
```

Then point `DATASET_PATH` at the produced file when training (see Section 6), e.g.:

```bash
DATASET_PATH=camera_movement_sft/train_data/train_swift.jsonl \
    bash camera_movement_sft/train.sh qwen3vl-4b
```

> See `camera_movement_sft/train_data/README.md` for more details.

---

## 5. Extract VGGT Features

> **Required for CamDistill and CamInject (cache mode) only. Skip for plain SFT.**

Run this on your prepared training JSONL (from Section 4). The extractor reads video
paths from either the ms-swift `videos` field or a CamChoreo-style `local_path`.

**VGGT (default):**

```bash
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl camera_movement_sft/train_data/train_swift.jsonl \
    --output_dir /path/to/vggt_cache/ \
    --teacher vggt \
    --num_gpus 8
```

**VGGT-Omega (recommended):**

```bash
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl camera_movement_sft/train_data/train_swift.jsonl \
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
| `--num_gpus` | `8` | Number of GPUs for parallel extraction |
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

### 6.2 CamDistill — Camera Token Distillation

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
| `OUTPUT_DIR` | `output/camera_sft_<model>[_camdistill\|_caminject]` | Output directory (a `v0-<timestamp>` version subdir is added automatically) |

---

## 7. Evaluation

### 7.1 Our Benchmark

Evaluate a trained checkpoint on the internal benchmark:

```bash
TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_4b/v0-<timestamp> \
bash camera_movement_sft/eval/run_batch_checkpoints.sh \
    /path/to/CamChoreo/annotations.jsonl
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

To evaluate a **CamDistill checkpoint** (needs the plugin; no online VGGT):

```bash
USE_CAMDISTILL=1 \
TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_4b_camdistill/v0-<timestamp> \
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
