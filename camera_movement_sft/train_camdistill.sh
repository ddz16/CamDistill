#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # 默认 decord(可被环境变量覆盖), 避开 torchcodec 的 CUDA(libnvrtc.so.13)版本不匹配
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/python3.12/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
# ============================================================================
# CamDistill 训练脚本
# ============================================================================
# 在普通 SFT 基础上增加 Camera Token 蒸馏：
#   - 加载 CamDistill 插件（CameraTokenModule + 蒸馏 Loss）
#   - ViT 冻结，Camera Token Module 和 LLM 可训练
#   - 需要先运行 VGGT 预提取（见 plugins/vggt_feature_extractor.py）
#
# 用法:
#   bash camera_movement_sft/train_camdistill.sh <model_name>
#
#   model_name: qwen3vl-4b, qwen3vl-8b, qwen35-4b, qwen35-9b
#
# 环境变量:
#   VGGT_CACHE_DIR    — VGGT 预提取特征缓存目录（必需）
#   CAMDISTILL_LAMBDA — 蒸馏 loss 权重（常数, 默认 0.3）
#   CAMDISTILL_WARMUP_STEPS — 前多少步关闭蒸馏（默认 200）
#   CAMDISTILL_DEPTH  — Camera Token Module 层数（默认 6）
#   CAMDISTILL_EXTRACT_LAYERS — 显式指定抽取的 ViT 层索引，逗号分隔；长度必须等于 CAMDISTILL_DEPTH
#   CAMDISTILL_PER_HALF — 1=按 concat 两半分别算 cosine 再平均(默认), 0=整段计算
#   CAMDISTILL_STRICT_CACHE — 1=cache miss/损坏直接 raise（默认 1）
#   CAMERA_TOKEN_INSERT_POSITION — camera token 插入位置: front|back（默认 front）
#
# 示例:
#   VGGT_CACHE_DIR=/path/to/vggt_cache \
#   bash camera_movement_sft/train_camdistill.sh qwen3vl-8b
# ============================================================================

set -e

# 使用当前终端环境（已移除conda环境切换）

# ================================
# 模型选择
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
    qwen35-4b|qwen3.5-4b)
        MODEL="Qwen/Qwen3.5-4B"
        MODEL_TYPE="qwen3_5_camdistill"
        MODEL_SHORT="qwen35_4b_camdistill"
        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
        GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
        LEARNING_RATE=${LEARNING_RATE:-2e-5}
        DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}
        ;;
    qwen35-9b|qwen3.5-9b)
        MODEL="Qwen/Qwen3.5-9B"
        MODEL_TYPE="qwen3_5_camdistill"
        MODEL_SHORT="qwen35_9b_camdistill"
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
# CamDistill 配置
# ================================
export VGGT_CACHE_DIR="${VGGT_CACHE_DIR:?请设置 VGGT_CACHE_DIR 环境变量指向预提取缓存目录}"
export CAMDISTILL_LAMBDA="${CAMDISTILL_LAMBDA:-0.3}"
export CAMDISTILL_DEPTH="${CAMDISTILL_DEPTH:-6}"
export CAMDISTILL_EXTRACT_LAYERS="${CAMDISTILL_EXTRACT_LAYERS:-}"
export CAMDISTILL_WARMUP_STEPS="${CAMDISTILL_WARMUP_STEPS:-200}"
export CAMDISTILL_PER_HALF="${CAMDISTILL_PER_HALF:-1}"    # 1=按 concat 两半分别算 cosine 再平均(默认), 0=整段计算
export VGGT_TEACHER_TYPE="${VGGT_TEACHER_TYPE:-vggt}"  # "vggt" 或 "vggt_omega"
export CAMERA_TOKEN_INSERT_POSITION="${CAMERA_TOKEN_INSERT_POSITION:-front}"

if [ "${CAMERA_TOKEN_INSERT_POSITION}" != "front" ] && [ "${CAMERA_TOKEN_INSERT_POSITION}" != "back" ]; then
    echo "错误: CAMERA_TOKEN_INSERT_POSITION 仅支持 front 或 back，当前值: ${CAMERA_TOKEN_INSERT_POSITION}"
    exit 1
fi

# CamDistill fail-fast 配置（teacher 缓存必须可用）
export CAMDISTILL_STRICT_CACHE="${CAMDISTILL_STRICT_CACHE:-1}"

# ================================
# 路径配置
# ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PLUGIN_PATH="${SCRIPT_DIR}/plugins/camdistill_plugin.py"

cd "${PROJECT_ROOT}"

# 训练数据
TRAIN_DATA="${DATASET_PATH:-${SCRIPT_DIR}/train_data/camera_movement_train_diverse_50k_en.jsonl}"

# 输出目录
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

# ================================
# 训练超参数
# ================================
NUM_EPOCHS="${NUM_EPOCHS:-2}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"

# ================================
# 打印训练信息
# ================================
echo "============================================"
echo "CamDistill 训练 (Camera Token Distillation)"
echo "============================================"
echo "模型:           ${MODEL}"
echo "Plugin:         ${PLUGIN_PATH}"
echo "VGGT Cache:     ${VGGT_CACHE_DIR}"
echo "蒸馏权重:       ${CAMDISTILL_LAMBDA}"
echo "CamDistill层数: ${CAMDISTILL_DEPTH}"
echo "抽取层索引:     ${CAMDISTILL_EXTRACT_LAYERS:-<auto>}"
echo "蒸馏预热步数:   ${CAMDISTILL_WARMUP_STEPS}"
echo "按半计算:       ${CAMDISTILL_PER_HALF}"
echo "Camera位置:     ${CAMERA_TOKEN_INSERT_POSITION}"
echo "Strict Cache:   ${CAMDISTILL_STRICT_CACHE}"
echo "数据:           ${TRAIN_DATA}"
echo "输出:           ${OUTPUT_DIR}"
echo "GPU 数量:       ${NPROC_PER_NODE}"
echo "有效 Batch:     $((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "============================================"

# 检查
if [ ! -f "${PLUGIN_PATH}" ]; then
    echo "错误: 插件文件不存在: ${PLUGIN_PATH}"
    exit 1
fi
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "错误: 训练数据不存在: ${TRAIN_DATA}"
    exit 1
fi
if [ ! -d "${VGGT_CACHE_DIR}" ]; then
    echo "错误: VGGT 缓存目录不存在: ${VGGT_CACHE_DIR}"
    exit 1
fi

# ================================
# 开始训练
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
echo "CamDistill 训练完成！"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================"

# ===== 训练完成后，所有节点启动占卡程序 =====
# echo "===== 训练完成，启动占卡程序 ====="
# cd /group/40009/dazhaodu
# python run.py --size 25000 --gpus 0,1,2,3,4,5,6,7 --interval 0.002
