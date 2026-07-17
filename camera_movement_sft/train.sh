#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # 默认 decord(可被环境变量覆盖), 避开 torchcodec 的 CUDA(libnvrtc.so.13)版本不匹配
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/python3.12/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
# ============================================================================
# 运镜识别 SFT 训练 - 统一入口脚本
# ============================================================================
# 支持模型:
#   1. Qwen3-VL-4B-Instruct  (多模态, ~4B 参数)
#   2. Qwen3-VL-8B-Instruct  (多模态, ~8B 参数)
#   3. Qwen3.5-4B            (多模态, ~4B 参数, Linear Attention)
#   4. Qwen3.5-9B            (多模态, ~9B 参数, Linear Attention)
#
# 使用方法:
#   bash camera_movement_sft/train.sh <model_name>
#
#   model_name 可选值:
#     qwen3vl-4b   → Qwen/Qwen3-VL-4B-Instruct
#     qwen3vl-8b   → Qwen/Qwen3-VL-8B-Instruct
#     qwen35-4b    → Qwen/Qwen3.5-4B
#     qwen35-9b    → Qwen/Qwen3.5-9B
#
# 示例:
#   bash camera_movement_sft/train.sh qwen3vl-8b
#   bash camera_movement_sft/train.sh qwen35-9b
#   RESUME=true bash camera_movement_sft/train.sh qwen3vl-8b
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
    qwen35-4b|qwen3.5-4b)
        MODEL="Qwen/Qwen3.5-4B"
        MODEL_SHORT="qwen35_4b"
        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
        GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
        LEARNING_RATE=${LEARNING_RATE:-2e-5}
        DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}
        ;;
    qwen35-9b|qwen3.5-9b)
        MODEL="Qwen/Qwen3.5-9B"
        MODEL_SHORT="qwen35_9b"
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
# 路径配置（可通过环境变量覆盖）
# ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# 切换到项目根目录（ms-swift 需要在根目录运行）
cd "${PROJECT_ROOT}"

# 训练数据路径
TRAIN_DATA="${DATASET_PATH:-${SCRIPT_DIR}/train_data/camera_movement_train_diverse_50k_en.jsonl}"

# 续训开关
RESUME="${RESUME:-false}"

if [ "${RESUME}" = "true" ]; then
    OUTPUT_DIR="${RESUME_OUTPUT_DIR:?请设置 RESUME_OUTPUT_DIR 环境变量指向已有训练目录}"
    RESUME_CKPT="${RESUME_CHECKPOINT:?请设置 RESUME_CHECKPOINT 环境变量指向 checkpoint 路径}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-output/camera_sft_${MODEL_SHORT}}"
    RESUME_CKPT=""
fi

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
# GPU 配置（可通过环境变量覆盖）
# ================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# 显存优化
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ================================
# 视频处理配置（运镜分析需要更多帧）
# ================================
# 抽帧逻辑: nframes = min(视频时长 × FPS, FPS_MAX_FRAMES)
# FPS=5: 每秒5帧，捕捉运镜变化
# FPS_MIN_FRAMES=4: 极短视频最少4帧
# FPS_MAX_FRAMES=100: 最多100帧，支持20秒视频
# VIDEO_MAX_PIXELS=100352: 每帧约336×336像素
export FPS="${FPS:-5}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-100}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-100352}"

# ================================
# 训练超参数
# ================================
NUM_EPOCHS="${NUM_EPOCHS:-2}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-15}"

# ================================
# 打印训练信息
# ================================
echo "============================================"
echo "运镜识别 SFT 训练"
echo "============================================"
echo "模型:           ${MODEL}"
echo "模型简称:       ${MODEL_SHORT}"
echo "训练方式:       全参数微调 (full, 冻结 ViT+Aligner)"
echo "数据:           ${TRAIN_DATA}"
echo "输出:           ${OUTPUT_DIR}"
if [ "${RESUME}" = "true" ]; then
    echo "续训 checkpoint: ${RESUME_CKPT}"
fi
echo "GPU 数量:       ${NPROC_PER_NODE}"
echo "DeepSpeed:      ${DEEPSPEED_STAGE}"
echo "学习率:         ${LEARNING_RATE}"
echo "训练轮数:       ${NUM_EPOCHS}"
echo "有效 Batch:     $((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Max Length:     ${MAX_LENGTH}"
echo "视频参数:       FPS=${FPS}, MAX_FRAMES=${FPS_MAX_FRAMES}"
echo "============================================"

# 检查训练数据是否存在
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "错误: 训练数据文件不存在: ${TRAIN_DATA}"
    echo ""
    echo "请先准备数据:"
    echo "  python camera_movement_sft/prepare_human_labels.py"
    echo ""
    echo "或指定已有数据路径:"
    echo "  DATASET_PATH=/path/to/data.jsonl bash camera_movement_sft/train.sh ${MODEL_NAME}"
    exit 1
fi

# ================================
# 构建续训参数
# ================================
RESUME_ARGS=""
if [ "${RESUME}" = "true" ]; then
    RESUME_ARGS="--resume_from_checkpoint ${RESUME_CKPT} --add_version false"
fi

# ================================
# 开始训练
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
echo "训练完成！"
echo "输出目录: ${OUTPUT_DIR}"
echo "WandB: 训练日志已同步到 wandb.ai，项目: camera-movement"
echo "============================================"

# ===== 训练完成后，所有节点启动占卡程序 =====
echo "===== 训练完成，启动占卡程序 ====="
cd /group/40009/dazhaodu
python run.py --size 25000 --gpus 0,1,2,3,4,5,6,7 --interval 0.002
