#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # Default to decord (overridable via env), avoids torchcodec CUDA (libnvrtc.so.13) version mismatch
# ============================================================================
# Launch training in the background.
# ============================================================================
# Usage:
#   bash camera_movement_sft/start_train.sh <model_name>
#
# Examples:
#   bash camera_movement_sft/start_train.sh qwen3vl-8b
#   bash camera_movement_sft/start_train.sh qwen35-9b
#   bash camera_movement_sft/start_train.sh              # default: qwen3vl-8b
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${1:-qwen3vl-8b}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SCRIPT_DIR}/logs/train_${MODEL_NAME}_${TIMESTAMP}.log"

# Network proxy configuration.
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

# Make sure the log directory exists.
mkdir -p "${SCRIPT_DIR}/logs"

# Check whether a training job is already running.
if [ -f "${SCRIPT_DIR}/logs/train.pid" ]; then
    OLD_PID=$(cat "${SCRIPT_DIR}/logs/train.pid")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "Warning: another training job is already running (PID: ${OLD_PID})"
        echo "Stop it first: bash camera_movement_sft/stop_train.sh"
        exit 1
    fi
fi

# Launch training in the background.
echo "Launching training in the background..."
echo "  Model: ${MODEL_NAME}"
echo "  Log:   ${LOG_FILE}"

nohup bash "${SCRIPT_DIR}/train.sh" "${MODEL_NAME}" > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

echo "${TRAIN_PID}" > "${SCRIPT_DIR}/logs/train.pid"

echo ""
echo "Training job launched in the background!"
echo "  PID:      ${TRAIN_PID}"
echo "  Log file: ${LOG_FILE}"
echo ""
echo "Common operations:"
echo "  Follow log:      tail -f ${LOG_FILE}"
echo "  Monitor status:  bash camera_movement_sft/monitor_train.sh"
echo "  Stop training:   bash camera_movement_sft/stop_train.sh"
