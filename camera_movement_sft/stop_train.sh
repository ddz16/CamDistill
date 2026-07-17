#!/bin/bash
# ============================================================================
# 停止训练
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/train.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "未找到训练进程ID文件: ${PID_FILE}"
    echo "尝试手动查找..."
    ps aux | grep "swift sft" | grep -v grep
    exit 1
fi

TRAIN_PID=$(cat "${PID_FILE}")

if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "正在停止训练进程 (PID: ${TRAIN_PID})..."
    # 先尝试优雅停止
    kill "${TRAIN_PID}" 2>/dev/null
    sleep 3
    # 如果还在运行，强制停止
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "优雅停止超时，强制终止..."
        kill -9 "${TRAIN_PID}" 2>/dev/null
    fi
    # 同时杀掉可能的子进程 (deepspeed workers)
    pkill -P "${TRAIN_PID}" 2>/dev/null
    echo "训练已停止。"
else
    echo "训练进程 (PID: ${TRAIN_PID}) 已不在运行。"
fi

rm -f "${PID_FILE}"
