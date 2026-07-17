#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # Default to decord (overridable via env), avoids torchcodec's CUDA (libnvrtc.so.13) version mismatch
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/python3.12/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
# ============================================================================
# CamInject Baseline training script.
# ============================================================================
# Inject frozen VGGT camera tokens directly into the LLM without distillation.
# Two feature-source modes are supported:
#   1) cache  (default): read pre-extracted .pt features; high throughput, stable.
#   2) online: run VGGT/VGGT-Omega per batch to extract camera tokens (no pre-extraction).
#
# Trainable params: VGGTProjector (~8M) + LLM
# Loss: standard SFT cross-entropy only.
#
# Usage:
#   # Offline cache mode (default).
#   VGGT_MODE=cache VGGT_CACHE_DIR=/path/to/cache \
#   bash camera_movement_sft/train_caminject.sh qwen3vl-8b
#
#   # Online extraction mode (no pre-extraction).
#   VGGT_MODE=online VGGT_TEACHER_TYPE=vggt_omega \
#   VGGT_MODEL_PATH=/group/40009/dazhaodu/vggt-omega/checkpoints/vggt_omega_1b_512.pt \
#   bash camera_movement_sft/train_caminject.sh qwen3vl-8b
#
#   # Insert camera token at the end of each frame's visual tokens.
#   CAMERA_TOKEN_INSERT_POSITION=back \
#   VGGT_MODE=cache VGGT_CACHE_DIR=/path/to/cache \
#   bash camera_movement_sft/train_caminject.sh qwen3vl-8b
#
# Supported models: qwen3vl-4b, qwen3vl-8b
# ============================================================================

set -e

# NOTE: activate the conda environment manually before running (conda activate cm).

# ================================
# Model selection
# ================================
MODEL_NAME="${1:-qwen3vl-8b}"

case "${MODEL_NAME}" in
    qwen3vl-4b|qwen3-vl-4b)
        MODEL="Qwen/Qwen3-VL-4B-Instruct"
        MODEL_TYPE="qwen3_vl_caminject"
        MODEL_SHORT="qwen3vl_4b_caminject"
        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
        GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
        LEARNING_RATE=${LEARNING_RATE:-2e-5}
        DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}
        ;;
    qwen3vl-8b|qwen3-vl-8b)
        MODEL="Qwen/Qwen3-VL-8B-Instruct"
        MODEL_TYPE="qwen3_vl_caminject"
        MODEL_SHORT="qwen3vl_8b_caminject"
        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
        GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
        LEARNING_RATE=${LEARNING_RATE:-1.5e-5}
        DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}
        ;;
    *)
        echo "Error: unknown model '${MODEL_NAME}'"
        echo "Supported: qwen3vl-4b, qwen3vl-8b"
        exit 1
        ;;
esac

# ================================
# VGGT configuration
# ================================
export VGGT_MODE="${VGGT_MODE:-cache}"
export VGGT_TEACHER_TYPE="${VGGT_TEACHER_TYPE:-vggt}"  # "vggt" or "vggt_omega"
export VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-}"
if [ -z "${VGGT_MODEL_PATH}" ]; then
    if [ "${VGGT_TEACHER_TYPE}" = "vggt_omega" ]; then
        export VGGT_MODEL_PATH="facebook/VGGT-Omega"
    else
        export VGGT_MODEL_PATH="facebook/VGGT-1B"
    fi
fi
export VGGT_CACHE_DIR="${VGGT_CACHE_DIR:-}"

if [ "${VGGT_MODE}" = "cache" ]; then
    if [ -z "${VGGT_CACHE_DIR}" ]; then
        echo "Error: VGGT_CACHE_DIR must be set when VGGT_MODE=cache"
        exit 1
    fi
elif [ "${VGGT_MODE}" = "online" ]; then
    # Online mode does not depend on the cache; if strict_cache is not explicitly configured,
    # the behavior on online extraction failure falls back to strict_cache.
    export VGGT_CACHE_DIR=""
else
    echo "Error: unsupported VGGT_MODE=${VGGT_MODE} (only 'cache' or 'online' are supported)"
    exit 1
fi

# CamInject fail-fast configuration (prevents silent fallback to zero features).
export CAMINJECT_STRICT_IDS="${CAMINJECT_STRICT_IDS:-1}"
export CAMINJECT_STRICT_CACHE="${CAMINJECT_STRICT_CACHE:-1}"
export CAMINJECT_MAX_MISS_RATIO="${CAMINJECT_MAX_MISS_RATIO:-0.0}"
export CAMINJECT_MIN_RATIO_SAMPLES="${CAMINJECT_MIN_RATIO_SAMPLES:-32}"
export CAMINJECT_LOG_EVERY="${CAMINJECT_LOG_EVERY:-50}"
export CAMERA_TOKEN_INSERT_POSITION="${CAMERA_TOKEN_INSERT_POSITION:-front}"

if [ "${CAMERA_TOKEN_INSERT_POSITION}" != "front" ] && [ "${CAMERA_TOKEN_INSERT_POSITION}" != "back" ]; then
    echo "Error: CAMERA_TOKEN_INSERT_POSITION must be either 'front' or 'back', got: ${CAMERA_TOKEN_INSERT_POSITION}"
    exit 1
fi

# ================================
# Path configuration
# ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PLUGIN_PATH="${SCRIPT_DIR}/plugins/camdistill_plugin.py"

cd "${PROJECT_ROOT}"

TRAIN_DATA="${DATASET_PATH:-${SCRIPT_DIR}/train_data/camera_movement_train_diverse_50k_en.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/camera_sft_${MODEL_SHORT}}"

# ================================
# Load user environment (proxy, tokens, paths)
# ================================
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

# ================================
# GPU configuration
# ================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ================================
# Video processing configuration
# ================================
export FPS="${FPS:-5}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-100}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-100352}"
export VGGT_ONLINE_FPS="${VGGT_ONLINE_FPS:-${FPS}}"
export VGGT_ONLINE_MAX_FRAMES="${VGGT_ONLINE_MAX_FRAMES:-${FPS_MAX_FRAMES}}"

# ================================
# Training hyperparameters
# ================================
NUM_EPOCHS="${NUM_EPOCHS:-2}"
MAX_LENGTH="${MAX_LENGTH:-16384}"

# ================================
# Print training info
# ================================
echo "============================================"
echo "CamInject Baseline training"
echo "============================================"
echo "Model:            ${MODEL} (${MODEL_TYPE})"
echo "VGGT:             ${VGGT_MODEL_PATH} (mode=${VGGT_MODE})"
echo "VGGT teacher:     ${VGGT_TEACHER_TYPE}"
echo "VGGT cache:       ${VGGT_CACHE_DIR:-<disabled>}"
echo "Online FPS:       ${VGGT_ONLINE_FPS}"
echo "Online MaxFrames: ${VGGT_ONLINE_MAX_FRAMES}"
echo "Strict IDs:       ${CAMINJECT_STRICT_IDS}"
echo "Strict Cache:     ${CAMINJECT_STRICT_CACHE}"
echo "Max miss ratio:   ${CAMINJECT_MAX_MISS_RATIO}"
echo "Camera position:  ${CAMERA_TOKEN_INSERT_POSITION}"
echo "Data:             ${TRAIN_DATA}"
echo "Output:           ${OUTPUT_DIR}"
echo "GPUs:             ${NPROC_PER_NODE}"
echo "Effective batch:  $((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Trainable:        VGGTProjector (~8M) + LLM"
echo "Loss:             SFT only (no distillation)"
echo "============================================"

# Sanity checks.
if [ ! -f "${PLUGIN_PATH}" ]; then
    echo "Error: plugin file not found: ${PLUGIN_PATH}"; exit 1
fi
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Error: training data not found: ${TRAIN_DATA}"; exit 1
fi
if [ "${VGGT_MODE}" = "cache" ] && [ ! -d "${VGGT_CACHE_DIR}" ]; then
    echo "Error: VGGT_CACHE_DIR directory not found: ${VGGT_CACHE_DIR}"; exit 1
fi

# ================================
# Start training (standard SFT)
# ================================

swift sft \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --use_hf true \
    --external_plugins "${PLUGIN_PATH}" \
    --dataset "${TRAIN_DATA}" \
    --split_dataset_ratio 0.05 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_aligner true \
    --deepspeed "${DEEPSPEED_STAGE}" \
    --attn_impl flash_attention_2 \
    --gradient_checkpointing true \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_length ${MAX_LENGTH} \
    --truncation_strategy delete \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_only_model false \
    --load_best_model_at_end false \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --load_from_cache_file true \
    --report_to wandb

echo "============================================"
echo "CamInject training finished!"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================"

# ===== After training, launch the GPU-occupation program on every node. =====
echo "===== Training finished, launching GPU-occupation program ====="
cd /group/40009/dazhaodu
python run.py --size 25000 --gpus 0,1,2,3,4,5,6,7 --interval 0.002
