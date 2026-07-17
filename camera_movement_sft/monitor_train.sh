#!/bin/bash
# ============================================================================
# Monitor training status.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/train.pid"

echo "============================================"
echo "Training monitor"
echo "============================================"

# 1. Check training process status.
echo ""
echo "[Process status]"
if [ -f "${PID_FILE}" ]; then
    TRAIN_PID=$(cat "${PID_FILE}")
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "  [OK] Training process is running (PID: ${TRAIN_PID})"
        RUN_TIME=$(ps -o etime= -p "${TRAIN_PID}" 2>/dev/null | tr -d ' ')
        echo "  Elapsed time: ${RUN_TIME}"
    else
        echo "  [FAIL] Training process has exited (PID: ${TRAIN_PID})"
    fi
else
    echo "  [FAIL] Training PID file not found"
fi

# 2. GPU usage.
echo ""
echo "[GPU usage]"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"

# 3. Latest log.
echo ""
echo "[Latest training log (last 15 lines)]"
LATEST_LOG=$(ls -t "${SCRIPT_DIR}"/logs/train_*.log 2>/dev/null | head -1)
if [ -n "${LATEST_LOG}" ]; then
    echo "  Log file: ${LATEST_LOG}"
    echo "  ---"
    tail -15 "${LATEST_LOG}"
else
    echo "  Log file not found"
fi

# 4. Loss trend.
echo ""
echo "[Loss trend (most recent 10 entries)]"
if [ -n "${LATEST_LOG}" ]; then
    grep "{'loss':" "${LATEST_LOG}" 2>/dev/null | tail -10 || echo "  No loss records yet"
fi

echo ""
echo "============================================"
