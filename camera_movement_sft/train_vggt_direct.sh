#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # 默认 decord(可被环境变量覆盖), 避开 torchcodec 的 CUDA(libnvrtc.so.13)版本不匹配
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/python3.12/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
# ============================================================================
# VGGT-Direct Baseline 训练脚本
# ============================================================================
# 直接使用 VGGT 冻结输出的 camera token 注入 LLM，不做蒸馏。
# 支持两种特征来源模式：
#   1) cache  (默认): 读取预提取 .pt 特征，吞吐高、稳定
#   2) online: 每个 batch 在线跑 VGGT/VGGT-Omega 提取 camera token（无需预提取）
#
# 可训练参数: VGGTProjector (~8M) + LLM
# Loss: 仅标准 SFT Cross-Entropy
#
# 用法:
#   # 离线 cache 模式（默认）
#   VGGT_MODE=cache VGGT_CACHE_DIR=/path/to/cache \
#   bash camera_movement_sft/train_vggt_direct.sh qwen3vl-8b
#
#   # 在线提取模式（无需预提取）
#   VGGT_MODE=online VGGT_TEACHER_TYPE=vggt_omega \
#   VGGT_MODEL_PATH=/group/40009/dazhaodu/vggt-omega/checkpoints/vggt_omega_1b_512.pt \
#   bash camera_movement_sft/train_vggt_direct.sh qwen3vl-8b
#
#   # 指定 camera token 插入到每帧 visual tokens 末尾
#   CAMERA_TOKEN_INSERT_POSITION=back \
#   VGGT_MODE=cache VGGT_CACHE_DIR=/path/to/cache \
#   bash camera_movement_sft/train_vggt_direct.sh qwen3vl-8b
#
# 支持模型: qwen3vl-4b, qwen3vl-8b, qwen35-4b, qwen35-9b
# ============================================================================

set -e

# 注意：请在运行脚本前手动激活 conda 环境（conda activate cm）

# ================================
# 模型选择
# ================================
MODEL_NAME="${1:-qwen3vl-8b}"

case "${MODEL_NAME}" in
    qwen3vl-4b|qwen3-vl-4b)
        MODEL="Qwen/Qwen3-VL-4B-Instruct"
        MODEL_TYPE="qwen3_vl_vggt_direct"
        MODEL_SHORT="qwen3vl_4b_vggt_direct"
        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
        GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
        LEARNING_RATE=${LEARNING_RATE:-2e-5}
        DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}
        ;;
    qwen3vl-8b|qwen3-vl-8b)
        MODEL="Qwen/Qwen3-VL-8B-Instruct"
        MODEL_TYPE="qwen3_vl_vggt_direct"
        MODEL_SHORT="qwen3vl_8b_vggt_direct"
        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
        GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
        LEARNING_RATE=${LEARNING_RATE:-1.5e-5}
        DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}
        ;;
    qwen35-4b|qwen3.5-4b)
        MODEL="Qwen/Qwen3.5-4B"
        MODEL_TYPE="qwen3_5_vggt_direct"
        MODEL_SHORT="qwen35_4b_vggt_direct"
        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
        GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
        LEARNING_RATE=${LEARNING_RATE:-2e-5}
        DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}
        ;;
    qwen35-9b|qwen3.5-9b)
        MODEL="Qwen/Qwen3.5-9B"
        MODEL_TYPE="qwen3_5_vggt_direct"
        MODEL_SHORT="qwen35_9b_vggt_direct"
        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
        GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
        LEARNING_RATE=${LEARNING_RATE:-1.5e-5}
        DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}
        ;;
    *)
        echo "错误: 未知模型 '${MODEL_NAME}'"
        echo "支持的模型: qwen3vl-4b, qwen3vl-8b, qwen35-4b, qwen35-9b"
        exit 1
        ;;
esac

# ================================
# VGGT 配置
# ================================
export VGGT_MODE="${VGGT_MODE:-cache}"
export VGGT_TEACHER_TYPE="${VGGT_TEACHER_TYPE:-vggt}"  # "vggt" 或 "vggt_omega"
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
        echo "错误: VGGT_MODE=cache 时必须设置 VGGT_CACHE_DIR"
        exit 1
    fi
elif [ "${VGGT_MODE}" = "online" ]; then
    # online 模式不依赖 cache；若未显式配置 strict_cache，则默认允许在线提取失败时回退由 strict_cache 控制。
    export VGGT_CACHE_DIR=""
else
    echo "错误: 不支持的 VGGT_MODE=${VGGT_MODE}（仅支持 cache 或 online）"
    exit 1
fi

# VGGT-Direct fail-fast 配置（避免 silent fallback 到零特征）
export VGGT_DIRECT_STRICT_IDS="${VGGT_DIRECT_STRICT_IDS:-1}"
export VGGT_DIRECT_STRICT_CACHE="${VGGT_DIRECT_STRICT_CACHE:-1}"
export VGGT_DIRECT_MAX_MISS_RATIO="${VGGT_DIRECT_MAX_MISS_RATIO:-0.0}"
export VGGT_DIRECT_MIN_RATIO_SAMPLES="${VGGT_DIRECT_MIN_RATIO_SAMPLES:-32}"
export VGGT_DIRECT_LOG_EVERY="${VGGT_DIRECT_LOG_EVERY:-50}"
export CAMERA_TOKEN_INSERT_POSITION="${CAMERA_TOKEN_INSERT_POSITION:-front}"
# SFT loss 对"答案值 token"提权
# 默认关闭：仅在显式设置 CAMERA_VALUE_LOSS_W 时开启
# 可选 CAMERA_LOSS_SCALE 覆盖策略名（默认 camera_value，设置 none 可关闭）

if [ "${CAMERA_TOKEN_INSERT_POSITION}" != "front" ] && [ "${CAMERA_TOKEN_INSERT_POSITION}" != "back" ]; then
    echo "错误: CAMERA_TOKEN_INSERT_POSITION 仅支持 front 或 back，当前值: ${CAMERA_TOKEN_INSERT_POSITION}"
    exit 1
fi

# ================================
# 路径配置
# ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PLUGIN_PATH="${SCRIPT_DIR}/plugins/camdistill_plugin.py"

cd "${PROJECT_ROOT}"

TRAIN_DATA="${DATASET_PATH:-${SCRIPT_DIR}/train_data/camera_movement_train_diverse_50k_en.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/camera_sft_${MODEL_SHORT}}"

# ================================
# 网络代理配置
# ================================
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

# ================================
# WandB 配置
# ================================
export WANDB_API_KEY="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"
export WANDB_PROJECT="camera-movement"

# ================================
# HuggingFace 配置
# ================================
export HF_HOME="/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/hf"
export USE_HF="${USE_HF:-1}"  # 默认用 HuggingFace hub(命中 HF_HOME 缓存); 设 USE_HF=0 切回 ModelScope
export HF_TOKEN="${HF_TOKEN:-***REMOVED***}"

# ================================
# GPU 配置
# ================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ================================
# 视频处理配置
# ================================
export FPS="${FPS:-5}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-100}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-100352}"
export VGGT_ONLINE_FPS="${VGGT_ONLINE_FPS:-${FPS}}"
export VGGT_ONLINE_MAX_FRAMES="${VGGT_ONLINE_MAX_FRAMES:-${FPS_MAX_FRAMES}}"

# ================================
# 训练超参数
# ================================
NUM_EPOCHS="${NUM_EPOCHS:-2}"
MAX_LENGTH="${MAX_LENGTH:-16384}"

# ================================
# 打印训练信息
# ================================
echo "============================================"
echo "VGGT-Direct Baseline 训练"
echo "============================================"
echo "模型:           ${MODEL} (${MODEL_TYPE})"
echo "VGGT:           ${VGGT_MODEL_PATH} (mode=${VGGT_MODE})"
echo "VGGT Teacher:   ${VGGT_TEACHER_TYPE}"
echo "VGGT Cache:     ${VGGT_CACHE_DIR:-<disabled>}"
echo "Online FPS:     ${VGGT_ONLINE_FPS}"
echo "Online MaxFrm:  ${VGGT_ONLINE_MAX_FRAMES}"
echo "Strict IDs:     ${VGGT_DIRECT_STRICT_IDS}"
echo "Strict Cache:   ${VGGT_DIRECT_STRICT_CACHE}"
echo "Max Miss Ratio: ${VGGT_DIRECT_MAX_MISS_RATIO}"
echo "Camera位置:     ${CAMERA_TOKEN_INSERT_POSITION}"
echo "数据:           ${TRAIN_DATA}"
echo "输出:           ${OUTPUT_DIR}"
echo "GPU 数量:       ${NPROC_PER_NODE}"
echo "有效 Batch:     $((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "可训练:         VGGTProjector (~8M) + LLM"
echo "Loss:           仅 SFT (无蒸馏)"
echo "============================================"

# 检查
if [ ! -f "${PLUGIN_PATH}" ]; then
    echo "错误: 插件文件不存在: ${PLUGIN_PATH}"; exit 1
fi
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "错误: 训练数据不存在: ${TRAIN_DATA}"; exit 1
fi
if [ "${VGGT_MODE}" = "cache" ] && [ ! -d "${VGGT_CACHE_DIR}" ]; then
    echo "错误: VGGT_CACHE_DIR 目录不存在: ${VGGT_CACHE_DIR}"; exit 1
fi

# ================================
# 开始训练 (标准 SFT)
# ================================

# SFT loss 加权 (答案值 token 提权)
# 默认关闭：仅在显式设置 CAMERA_VALUE_LOSS_W 时开启
LOSS_SCALE_ARGS=""
if [ -n "${CAMERA_VALUE_LOSS_W+x}" ]; then
    CAMERA_LOSS_SCALE_NAME="${CAMERA_LOSS_SCALE:-camera_value}"
    if [ -n "${CAMERA_LOSS_SCALE_NAME}" ] && [ "${CAMERA_LOSS_SCALE_NAME}" != "none" ]; then
        LOSS_SCALE_ARGS="--loss_scale ${CAMERA_LOSS_SCALE_NAME}"
        echo "SFT loss 加权: ${CAMERA_LOSS_SCALE_NAME} (值 token W=${CAMERA_VALUE_LOSS_W})"
    fi
fi

swift sft \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --use_hf true \
    ${LOSS_SCALE_ARGS} \
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
echo "VGGT-Direct 训练完成！"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================"

# ===== 训练完成后，所有节点启动占卡程序 =====
echo "===== 训练完成，启动占卡程序 ====="
cd /group/40009/dazhaodu
python run.py --size 25000 --gpus 0,1,2,3,4,5,6,7 --interval 0.002
