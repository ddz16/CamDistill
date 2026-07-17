#!/bin/bash
# ============================================================================
# Stop training.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/train.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "Training PID file not found: ${PID_FILE}"
    echo "Trying to locate the process manually..."
    ps aux | grep "swift sft" | grep -v grep
    exit 1
fi

TRAIN_PID=$(cat "${PID_FILE}")

if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "Stopping training process (PID: ${TRAIN_PID})..."
    # Try a graceful stop first.
    kill "${TRAIN_PID}" 2>/dev/null
    sleep 3
    # If still running, force-kill it.
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "Graceful stop timed out; force killing..."
        kill -9 "${TRAIN_PID}" 2>/dev/null
    fi
    # Also kill possible child processes (deepspeed workers).
    pkill -P "${TRAIN_PID}" 2>/dev/null
    echo "Training stopped."
else
    echo "Training process (PID: ${TRAIN_PID}) is no longer running."
fi

rm -f "${PID_FILE}"
