#!/bin/bash
# ============================================================================
# 监控训练状态
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/train.pid"

echo "============================================"
echo "训练监控"
echo "============================================"

# 1. 检查训练进程状态
echo ""
echo "【进程状态】"
if [ -f "${PID_FILE}" ]; then
    TRAIN_PID=$(cat "${PID_FILE}")
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "  ✅ 训练进程运行中 (PID: ${TRAIN_PID})"
        RUN_TIME=$(ps -o etime= -p "${TRAIN_PID}" 2>/dev/null | tr -d ' ')
        echo "  运行时间: ${RUN_TIME}"
    else
        echo "  ❌ 训练进程已退出 (PID: ${TRAIN_PID})"
    fi
else
    echo "  ❌ 未找到训练进程ID文件"
fi

# 2. GPU 使用情况
echo ""
echo "【GPU 使用情况】"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 不可用"

# 3. 最新日志
echo ""
echo "【最新训练日志 (最后 15 行)】"
LATEST_LOG=$(ls -t "${SCRIPT_DIR}"/logs/train_*.log 2>/dev/null | head -1)
if [ -n "${LATEST_LOG}" ]; then
    echo "  日志文件: ${LATEST_LOG}"
    echo "  ---"
    tail -15 "${LATEST_LOG}"
else
    echo "  未找到日志文件"
fi

# 4. Loss 趋势
echo ""
echo "【Loss 趋势 (最近 10 条)】"
if [ -n "${LATEST_LOG}" ]; then
    grep "{'loss':" "${LATEST_LOG}" 2>/dev/null | tail -10 || echo "  暂无 loss 记录"
fi

echo ""
echo "============================================"
