#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # Default to decord (overridable via env), avoids torchcodec's CUDA (libnvrtc.so.13) version mismatch
# ============================================================================
# CamDistill training script.
# ============================================================================
# Adds Camera Token distillation on top of standard SFT:
#   - Load the CamDistill plugin (CameraTokenModule + distillation loss).
#   - ViT is frozen; Camera Token Module and LLM are trainable.
#   - Requires running VGGT pre-extraction first (see plugins/vggt_feature_extractor.py).
#
# Usage:
#   bash camera_movement_sft/train_camdistill.sh <model_name>
#
#   model_name: qwen3vl-4b, qwen3vl-8b
#
# Environment variables:
#   VGGT_CACHE_DIR    -- VGGT pre-extraction feature cache directory (required).
#   CAMDISTILL_LAMBDA -- distillation loss weight (constant, default 0.3).
#   CAMDISTILL_WARMUP_STEPS -- steps to disable distillation for at the start (default 200).
#   CAMDISTILL_DEPTH  -- number of Camera Token Module layers (default 6).
#   CAMDISTILL_EXTRACT_LAYERS -- explicit comma-separated ViT layer indices;
#                                length must equal CAMDISTILL_DEPTH.
#   CAMDISTILL_PER_HALF -- 1=compute cosine on the two concat halves separately then average
#                          (default), 0=compute over the full vector.
#   CAMDISTILL_STRICT_CACHE -- 1=raise immediately on cache miss / corruption (default 1).
#   CAMERA_TOKEN_INSERT_POSITION -- camera-token insertion position: front|back (default front).
#
# Example:
#   VGGT_CACHE_DIR=/path/to/vggt_cache \
#   bash camera_movement_sft/train_camdistill.sh qwen3vl-8b
# ============================================================================

set -e

# Use the current shell environment (conda switching removed).

# ================================
# Load user environment (proxy, tokens, paths)
# ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

# ================================
# Model selection
# ================================
MODEL_NAME="${1:-qwen3vl-8b}"

case "${MODEL_NAME}" in
    qwen3vl-4b|qwen3-vl-4b)
        MODEL="Qwen/Qwen3-VL-4B-Instruct"
        MODEL_TYPE="qwen3_vl_camdistill"
        MODEL_SHORT="qwen3vl_4b_camdistill"
        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
        GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
        LEARNING_RATE=${LEARNING_RATE:-2e-5}
        DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}
        ;;
    qwen3vl-8b|qwen3-vl-8b)
        MODEL="Qwen/Qwen3-VL-8B-Instruct"
        MODEL_TYPE="qwen3_vl_camdistill"
        MODEL_SHORT="qwen3vl_8b_camdistill"
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
# CamDistill configuration
# ================================
export VGGT_CACHE_DIR="${VGGT_CACHE_DIR:?Please set VGGT_CACHE_DIR to point at the pre-extraction cache directory}"
export CAMDISTILL_LAMBDA="${CAMDISTILL_LAMBDA:-0.3}"
export CAMDISTILL_DEPTH="${CAMDISTILL_DEPTH:-6}"
export CAMDISTILL_EXTRACT_LAYERS="${CAMDISTILL_EXTRACT_LAYERS:-}"
export CAMDISTILL_WARMUP_STEPS="${CAMDISTILL_WARMUP_STEPS:-200}"
export CAMDISTILL_PER_HALF="${CAMDISTILL_PER_HALF:-1}"    # 1=compute cosine on the two concat halves separately then average (default), 0=full-vector
export VGGT_TEACHER_TYPE="${VGGT_TEACHER_TYPE:-vggt}"  # "vggt" or "vggt_omega"
export CAMERA_TOKEN_INSERT_POSITION="${CAMERA_TOKEN_INSERT_POSITION:-front}"

if [ "${CAMERA_TOKEN_INSERT_POSITION}" != "front" ] && [ "${CAMERA_TOKEN_INSERT_POSITION}" != "back" ]; then
    echo "Error: CAMERA_TOKEN_INSERT_POSITION must be either 'front' or 'back', got: ${CAMERA_TOKEN_INSERT_POSITION}"
    exit 1
fi

# CamDistill fail-fast configuration (teacher cache must be available).
export CAMDISTILL_STRICT_CACHE="${CAMDISTILL_STRICT_CACHE:-1}"

# ================================
# Path configuration
# ================================
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PLUGIN_PATH="${SCRIPT_DIR}/plugins/camdistill_plugin.py"

cd "${PROJECT_ROOT}"

# Training data.
TRAIN_DATA="${DATASET_PATH:-${SCRIPT_DIR}/train_data/train_swift.jsonl}"

# Output directory.
OUTPUT_DIR="${OUTPUT_DIR:-output/camera_sft_${MODEL_SHORT}}"

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

# ================================
# Training hyperparameters
# ================================
NUM_EPOCHS="${NUM_EPOCHS:-2}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"

# ================================
# Print training info
# ================================
echo "============================================"
echo "CamDistill training (Camera Token Distillation)"
echo "============================================"
echo "Model:            ${MODEL}"
echo "Plugin:           ${PLUGIN_PATH}"
echo "VGGT cache:       ${VGGT_CACHE_DIR}"
echo "Distill lambda:   ${CAMDISTILL_LAMBDA}"
echo "CamDistill depth: ${CAMDISTILL_DEPTH}"
echo "Extract layers:   ${CAMDISTILL_EXTRACT_LAYERS:-<auto>}"
echo "Distill warmup:   ${CAMDISTILL_WARMUP_STEPS}"
echo "Per-half compute: ${CAMDISTILL_PER_HALF}"
echo "Camera position:  ${CAMERA_TOKEN_INSERT_POSITION}"
echo "Strict cache:     ${CAMDISTILL_STRICT_CACHE}"
echo "Data:             ${TRAIN_DATA}"
echo "Output:           ${OUTPUT_DIR}"
echo "GPUs:             ${NPROC_PER_NODE}"
echo "Effective batch:  $((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "============================================"

# Sanity checks.
if [ ! -f "${PLUGIN_PATH}" ]; then
    echo "Error: plugin file not found: ${PLUGIN_PATH}"
    exit 1
fi
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Error: training data not found: ${TRAIN_DATA}"
    exit 1
fi
if [ ! -d "${VGGT_CACHE_DIR}" ]; then
    echo "Error: VGGT cache directory not found: ${VGGT_CACHE_DIR}"
    exit 1
fi

# ================================
# Start training
# ================================
swift sft \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --use_hf true \
    --external_plugins "${PLUGIN_PATH}" \
    --loss_type camdistill \
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
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
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
echo "CamDistill training finished!"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================"
