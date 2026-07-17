#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # 默认 decord(可被环境变量覆盖), 避开 torchcodec 的 CUDA(libnvrtc.so.13)版本不匹配
# ============================================================================
# 后台启动训练
# ============================================================================
# 用法:
#   bash camera_movement_sft/start_train.sh <model_name>
#
# 示例:
#   bash camera_movement_sft/start_train.sh qwen3vl-8b
#   bash camera_movement_sft/start_train.sh qwen35-9b
#   bash camera_movement_sft/start_train.sh              # 默认 qwen3vl-8b
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${1:-qwen3vl-8b}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SCRIPT_DIR}/logs/train_${MODEL_NAME}_${TIMESTAMP}.log"

# 网络代理配置
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

# 确保日志目录存在
mkdir -p "${SCRIPT_DIR}/logs"

# 检查是否已有训练在运行
if [ -f "${SCRIPT_DIR}/logs/train.pid" ]; then
    OLD_PID=$(cat "${SCRIPT_DIR}/logs/train.pid")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "警告: 已有训练任务在运行 (PID: ${OLD_PID})"
        echo "请先停止: bash camera_movement_sft/stop_train.sh"
        exit 1
    fi
fi

# 后台启动训练
echo "启动后台训练..."
echo "  模型: ${MODEL_NAME}"
echo "  日志: ${LOG_FILE}"

nohup bash "${SCRIPT_DIR}/train.sh" "${MODEL_NAME}" > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

echo "${TRAIN_PID}" > "${SCRIPT_DIR}/logs/train.pid"

echo ""
echo "训练任务已在后台启动！"
echo "  进程ID: ${TRAIN_PID}"
echo "  日志文件: ${LOG_FILE}"
echo ""
echo "常用操作:"
echo "  查看日志: tail -f ${LOG_FILE}"
echo "  监控状态: bash camera_movement_sft/monitor_train.sh"
echo "  停止训练: bash camera_movement_sft/stop_train.sh"
