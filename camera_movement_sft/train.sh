#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # Default to decord (overridable via env), avoids torchcodec's CUDA (libnvrtc.so.13) version mismatch
# ============================================================================
# Camera-movement SFT training - unified entry script.
# ============================================================================
# Supported models:
#   1. Qwen3-VL-4B-Instruct  (multimodal, ~4B params)
#   2. Qwen3-VL-8B-Instruct  (multimodal, ~8B params)
#
# Usage:
#   bash camera_movement_sft/train.sh <model_name>
#
#   model_name values:
#     qwen3vl-4b   -> Qwen/Qwen3-VL-4B-Instruct
#     qwen3vl-8b   -> Qwen/Qwen3-VL-8B-Instruct
#
# Examples:
#   bash camera_movement_sft/train.sh qwen3vl-8b
#   RESUME=true bash camera_movement_sft/train.sh qwen3vl-8b
# ============================================================================

set -e

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
        MODEL_SHORT="qwen3vl_4b"
        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
        GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
        LEARNING_RATE=${LEARNING_RATE:-2e-5}
        DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}
        ;;
    qwen3vl-8b|qwen3-vl-8b)
        MODEL="Qwen/Qwen3-VL-8B-Instruct"
        MODEL_SHORT="qwen3vl_8b"
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
# Path configuration (overridable via env)
# ================================
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Switch to the project root (ms-swift must be launched from the root).
cd "${PROJECT_ROOT}"

# Training data path.
TRAIN_DATA="${DATASET_PATH:-${SCRIPT_DIR}/train_data/train_swift.jsonl}"

# Resume toggle.
RESUME="${RESUME:-false}"

if [ "${RESUME}" = "true" ]; then
    OUTPUT_DIR="${RESUME_OUTPUT_DIR:?Please set RESUME_OUTPUT_DIR to point at an existing training directory}"
    RESUME_CKPT="${RESUME_CHECKPOINT:?Please set RESUME_CHECKPOINT to point at a checkpoint path}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-output/camera_sft_${MODEL_SHORT}}"
    RESUME_CKPT=""
fi


# ================================
# GPU configuration (overridable via env)
# ================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Memory optimization.
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ================================
# Video processing configuration (camera-movement analysis needs more frames)
# ================================
# Frame-sampling logic: nframes = min(video_duration * FPS, FPS_MAX_FRAMES)
# FPS=5: 5 frames per second, captures camera-movement changes.
# FPS_MIN_FRAMES=4: at least 4 frames even for very short videos.
# FPS_MAX_FRAMES=100: up to 100 frames, enough for 20s videos.
# VIDEO_MAX_PIXELS=100352: ~ 336x336 pixels per frame.
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
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-15}"

# ================================
# Print training info
# ================================
echo "============================================"
echo "Camera-movement SFT training"
echo "============================================"
echo "Model:            ${MODEL}"
echo "Model short:      ${MODEL_SHORT}"
echo "Training mode:    full parameter fine-tuning (full, ViT+Aligner frozen)"
echo "Data:             ${TRAIN_DATA}"
echo "Output:           ${OUTPUT_DIR}"
if [ "${RESUME}" = "true" ]; then
    echo "Resume ckpt:      ${RESUME_CKPT}"
fi
echo "GPUs:             ${NPROC_PER_NODE}"
echo "DeepSpeed:        ${DEEPSPEED_STAGE}"
echo "Learning rate:    ${LEARNING_RATE}"
echo "Epochs:           ${NUM_EPOCHS}"
echo "Effective batch:  $((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Max length:       ${MAX_LENGTH}"
echo "Video params:     FPS=${FPS}, MAX_FRAMES=${FPS_MAX_FRAMES}"
echo "============================================"

# Check that training data exists.
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Error: training data file not found: ${TRAIN_DATA}"
    echo ""
    echo "Prepare training data from CamChoreo-schema annotations first:"
    echo "  python camera_movement_sft/prepare_train_data.py \\"
    echo "      --input /path/to/your_annotations.jsonl \\"
    echo "      --output camera_movement_sft/train_data/train_swift.jsonl"
    echo ""
    echo "Then point DATASET_PATH at the produced file:"
    echo "  DATASET_PATH=camera_movement_sft/train_data/train_swift.jsonl bash camera_movement_sft/train.sh ${MODEL_NAME}"
    exit 1
fi

# ================================
# Build resume args
# ================================
RESUME_ARGS=""
if [ "${RESUME}" = "true" ]; then
    RESUME_ARGS="--resume_from_checkpoint ${RESUME_CKPT} --add_version false"
fi

# ================================
# Start training
# ================================
swift sft \
    --model "${MODEL}" \
    --use_hf true \
    ${RESUME_ARGS} \
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
echo "Training finished!"
echo "Output dir: ${OUTPUT_DIR}"
echo "WandB: training logs synced to wandb.ai, project: camera-movement"
echo "============================================"
