#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # 默认 decord, 避开 torchcodec 的 CUDA(libnvrtc.so.13) 版本不匹配
# ============================================================================
# Pose-Text 方案批量评测：VGGT-Omega 估计相机位姿 -> 文本注入 prompt -> base/SFT 模型推理
# ============================================================================
# 与 run_batch_checkpoints.sh 同结构，但多两步(位姿提取 + 文本注入)，且推理是纯文本
# (不加 VGGT 插件、不改 model_type)。复用 prepare_test_data / postprocess / evaluate。
#
# 用法:
#   # 评测 base 模型
#   BASE_MODEL=Qwen/Qwen3-VL-4B-Instruct MODEL_TAG=base_qwen3vl_4b \
#   bash camera_movement_sft/eval/run_batch_posetext.sh /path/to/benchmark.jsonl
#
#   # 评测 SFT checkpoint(自动扫描 TRAIN_OUTPUT_DIR)
#   TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_4b/vX-... \
#   bash camera_movement_sft/eval/run_batch_posetext.sh /path/to/benchmark.jsonl
#
#   # --force 覆盖旧结果; ONLY_LAST=false 评估全部 checkpoint
#
# 关键环境变量:
#   VGGT_MODEL_PATH   VGGT-Omega 权重 (默认 facebook/VGGT-Omega, 命中 HF 缓存)
#   POSE_CACHE_DIR    位姿缓存目录 (默认 <eval>/pose_cache/<测试集名>，与模型无关，可跨模型复用)
#   POSE_ON_MISSING   缺位姿时 keep(原样保留)|skip(丢弃)，默认 keep
#   VIDEO_DIRS_EXTRA  额外本地视频候选目录(空格分隔)
# ============================================================================

# ================================
# 解析命令行参数：--force + 测试集路径列表
# ================================
FORCE=false
RAW_TEST_DATA_LIST=()
for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
        *) RAW_TEST_DATA_LIST+=("$arg") ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${PROJECT_ROOT}")"
cd "${REPO_ROOT}"

# ================================
# 模型选择：BASE_MODEL(原始基座) 或 TRAIN_OUTPUT_DIR(SFT checkpoint)
# ================================
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-output/camera_sft_qwen3vl_4b/v0-latest}"
BASE_MODEL="${BASE_MODEL:-}"

if [ -n "${BASE_MODEL}" ]; then
    TRAIN_TAG="${MODEL_TAG:-$(basename "${BASE_MODEL}")}"
    CHECKPOINTS=("base")
    echo "[INFO] BASE_MODEL 模式，评测原始模型: ${BASE_MODEL}（标签: ${TRAIN_TAG}）"
else
    TRAIN_PARENT=$(basename "$(dirname "${TRAIN_OUTPUT_DIR}")")
    TRAIN_LEAF=$(basename "${TRAIN_OUTPUT_DIR}")
    TRAIN_VER="${TRAIN_LEAF%%-*}"
    TRAIN_TAG="${TRAIN_PARENT#output_}_${TRAIN_VER}"
    TRAIN_TAG="${TRAIN_TAG#camera_sft_}"

    CHECKPOINTS=()
    for d in "${TRAIN_OUTPUT_DIR}"/checkpoint-*; do
        [ -d "$d" ] && CHECKPOINTS+=("$(basename "$d")")
    done
    IFS=$'\n' CHECKPOINTS=($(printf '%s\n' "${CHECKPOINTS[@]}" | sort -t'-' -k2 -n)); unset IFS

    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "[ERROR] 在 ${TRAIN_OUTPUT_DIR} 下未找到任何 checkpoint-* 目录！"
        echo "  设置 TRAIN_OUTPUT_DIR，或用 BASE_MODEL 评测原始基座模型。"
        exit 1
    fi

    ONLY_LAST="${ONLY_LAST:-true}"
    if [ "${ONLY_LAST}" = "true" ]; then
        CHECKPOINTS=("${CHECKPOINTS[-1]}")
        echo "[INFO] ONLY_LAST=true，仅评估: ${CHECKPOINTS[0]}（设 ONLY_LAST=false 评估全部）"
    fi
fi

# ================================
# 运行环境(与 run_batch_checkpoints.sh 一致)
# ================================
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/python3.12/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_HOME="${HF_HOME:-/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/hf}"
export USE_HF="${USE_HF:-1}"
export HF_TOKEN="${HF_TOKEN:-***REMOVED***}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 视频/推理配置(与训练/标准评测一致)
export FPS="${FPS:-5}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-100}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-100352}"
INFER_BACKEND="${INFER_BACKEND:-pt}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-24576}"   # 位姿文本会显著拉长 prompt，给足上下文预算避免截断

# Pose-Text 专用配置
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-facebook/VGGT-Omega}"
POSE_ON_MISSING="${POSE_ON_MISSING:-keep}"
POSE_INJECT_SCRIPT="${SCRIPT_DIR}/pose_text_inject.py"

# 本地视频候选目录
VIDEO_DIRS=(
    "/group/40009/dazhaodu/OurBenchmark/YouTube_Videos"
    ${VIDEO_DIRS_EXTRA:-}
)

# 可选：显式 model_type(base 模型自动匹配有歧义时用)。pose-text 不用 VGGT 插件。
MODEL_TYPE="${MODEL_TYPE:-}"
MODEL_TYPE_ARGS=()
if [ -n "${MODEL_TYPE}" ]; then
    MODEL_TYPE_ARGS+=(--model_type "${MODEL_TYPE}" --template "${MODEL_TYPE}")
fi

if [ ${#RAW_TEST_DATA_LIST[@]} -eq 0 ]; then
    echo "[ERROR] 未指定测试集。用法: bash $(basename "$0") <benchmark.jsonl> [更多...] [--force]"
    exit 1
fi

EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_camera_movement_fixed.py"

echo "============================================"
echo "Pose-Text 批量评测 (VGGT-Omega 位姿 -> 文本 prompt)"
echo "============================================"
echo "训练标签 (TRAIN_TAG): ${TRAIN_TAG}"
echo "Checkpoints:  ${CHECKPOINTS[*]}"
echo "VGGT-Omega:   ${VGGT_MODEL_PATH}"
echo "缺位姿策略:   ${POSE_ON_MISSING}"
echo "推理后端:     ${INFER_BACKEND}"
echo "测试集 (${#RAW_TEST_DATA_LIST[@]} 个):"
for td in "${RAW_TEST_DATA_LIST[@]}"; do echo "  - $(basename "$td")"; done
[ "$FORCE" = true ] && echo "模式: --force（覆盖旧结果）"
echo "============================================"

# ============================================================================
# 逐测试集
# ============================================================================
TESTSET_TOTAL=${#RAW_TEST_DATA_LIST[@]}
TESTSET_IDX=0

for RAW_TEST_DATA in "${RAW_TEST_DATA_LIST[@]}"; do
    TESTSET_IDX=$((TESTSET_IDX + 1))
    TESTSET_BASENAME=$(basename "${RAW_TEST_DATA}" .jsonl)
    TESTSET_NAME="${TRAIN_TAG}_posetext_${TESTSET_BASENAME}"

    RUN_DIR="${SCRIPT_DIR}/eval_results/${TESTSET_NAME}"
    TEST_DATA_SWIFT="${RUN_DIR}/test_for_infer.jsonl"
    MAPPING_FILE="${RUN_DIR}/video_id_mapping.jsonl"
    POSE_TEST_DATA="${RUN_DIR}/test_for_infer_pose.jsonl"
    # 位姿缓存与模型无关，按测试集名共享(可被 POSE_CACHE_DIR 覆盖)
    POSE_CACHE_DIR="${POSE_CACHE_DIR:-${SCRIPT_DIR}/pose_cache/${TESTSET_BASENAME}}"

    echo ""
    echo "████████████████████████████████████████████████████████████████"
    echo "██  测试集 [${TESTSET_IDX}/${TESTSET_TOTAL}]: ${TESTSET_NAME}"
    echo "██  路径: ${RAW_TEST_DATA}"
    echo "████████████████████████████████████████████████████████████████"

    if [ ! -f "${RAW_TEST_DATA}" ]; then
        echo "[ERROR] 测试集不存在: ${RAW_TEST_DATA}，跳过！"; continue
    fi

    if [ "$FORCE" = true ]; then
        echo "[FORCE] 清理旧结果: ${RUN_DIR}/ (位姿缓存 ${POSE_CACHE_DIR} 保留)"
        rm -rf "${RUN_DIR}"
    fi
    mkdir -p "${RUN_DIR}" "${POSE_CACHE_DIR}"
    ALL_PRED_PATHS=()

    # ---- Step 0: 标准测试数据转换 ----
    if [ ! -f "$TEST_DATA_SWIFT" ]; then
        echo "[Step 0] 转换测试数据..."
        python "${SCRIPT_DIR}/prepare_test_data.py" \
            --input "${RAW_TEST_DATA}" \
            --output_dir "${RUN_DIR}" \
            --use_local_video \
            --video_dir "${VIDEO_DIRS[@]}" \
            --auto_download || { echo "[ERROR] 数据转换失败，跳过"; continue; }
    else
        echo "[Step 0] 测试数据已存在: ${TEST_DATA_SWIFT}"
    fi

    # ---- Step 0.5a: VGGT-Omega 在线估位姿(与模型无关，缓存复用) ----
    echo "[Step 0.5a] 提取 VGGT-Omega 位姿 -> ${POSE_CACHE_DIR}"
    VGGT_MODEL_PATH="${VGGT_MODEL_PATH}" python "${POSE_INJECT_SCRIPT}" extract \
        --mapping "${MAPPING_FILE}" \
        --pose_dir "${POSE_CACHE_DIR}" \
        --video_dir "${VIDEO_DIRS[@]}" \
        --vggt_model "${VGGT_MODEL_PATH}" \
        --fps "${FPS}" --max_frames "${FPS_MAX_FRAMES}" \
        || { echo "[ERROR] 位姿提取失败，跳过此测试集"; continue; }

    # ---- Step 0.5b: 位姿文本注入 prompt ----
    echo "[Step 0.5b] 注入位姿文本 -> ${POSE_TEST_DATA}"
    python "${POSE_INJECT_SCRIPT}" inject \
        --test_jsonl "${TEST_DATA_SWIFT}" \
        --mapping "${MAPPING_FILE}" \
        --pose_dir "${POSE_CACHE_DIR}" \
        --output "${POSE_TEST_DATA}" \
        --fps "${FPS}" --on_missing "${POSE_ON_MISSING}" \
        || { echo "[ERROR] 位姿注入失败，跳过此测试集"; continue; }

    # ---- 逐 checkpoint 推理 + 后处理 ----
    TOTAL=${#CHECKPOINTS[@]}
    IDX=0
    for CKPT in "${CHECKPOINTS[@]}"; do
        IDX=$((IDX + 1))
        if [ -n "${BASE_MODEL}" ]; then MODEL="${BASE_MODEL}"; else MODEL="${TRAIN_OUTPUT_DIR}/${CKPT}"; fi
        CKPT_DIR="${RUN_DIR}/${CKPT}"
        RESULT_PATH="${CKPT_DIR}/infer_results.jsonl"
        PRED_PATH="${CKPT_DIR}/eval_pred.jsonl"

        echo ""
        echo "╔══════════════════════════════════════════════╗"
        echo "║  [${TESTSET_IDX}/${TESTSET_TOTAL}] ${TESTSET_NAME} — [${IDX}/${TOTAL}] ${CKPT}"
        echo "╚══════════════════════════════════════════════╝"
        echo "  模型: ${MODEL}"
        mkdir -p "${CKPT_DIR}"

        # ---- 推理(纯文本，无 VGGT 插件) ----
        if [ -f "${RESULT_PATH}" ]; then
            echo "  [推理] 结果已存在，跳过: ${RESULT_PATH}"
        else
            echo "  [推理] 开始..."
            START_TIME=$(date +%s)
            MASTER_PORT="${MASTER_PORT:-29502}" \
            NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
            swift infer \
                --model ${MODEL} \
                "${MODEL_TYPE_ARGS[@]}" \
                --infer_backend ${INFER_BACKEND} \
                --val_dataset ${POSE_TEST_DATA} \
                --result_path ${RESULT_PATH} \
                --max_length ${MAX_LENGTH} \
                --temperature ${TEMPERATURE} \
                --max_new_tokens ${MAX_NEW_TOKENS} \
                --max_batch_size ${MAX_BATCH_SIZE} || true
            wait 2>/dev/null
            echo "  [推理] 耗时: $(( $(date +%s) - START_TIME ))s"
            if [ ! -f "${RESULT_PATH}" ]; then
                echo "  [ERROR] 推理失败！结果文件不存在，跳过此 checkpoint"; continue
            fi
        fi

        # ---- 后处理 ----
        if [ -f "${PRED_PATH}" ]; then
            echo "  [后处理] 结果已存在，跳过"
        else
            echo "  [后处理] 生成 eval_pred..."
            python "${SCRIPT_DIR}/postprocess_results.py" \
                --infer_result "${RESULT_PATH}" \
                --mapping "${MAPPING_FILE}" \
                --output_dir "${CKPT_DIR}"
            if [ ! -f "${PRED_PATH}" ]; then
                echo "  [ERROR] 后处理失败！eval_pred.jsonl 未生成，跳过"; continue
            fi
        fi

        ALL_PRED_PATHS+=("${PRED_PATH}")
        echo "  [完成] ${CKPT} ✓"
    done

    # ---- 汇总评测 ----
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  汇总评测 — ${TESTSET_NAME}"
    echo "╚══════════════════════════════════════════════╝"
    if [ ${#ALL_PRED_PATHS[@]} -eq 0 ]; then
        echo "[ERROR] ${TESTSET_NAME}: 没有任何有效预测结果，跳过评测！"; continue
    fi
    SUMMARY_OUTPUT="${RUN_DIR}/eval_results.json"
    python3 "${EVAL_SCRIPT}" \
        --gt "${RAW_TEST_DATA}" \
        --pred ${ALL_PRED_PATHS[@]} \
        --iou_thresh 0.3 0.5 0.7 \
        --output "${SUMMARY_OUTPUT}" \
        2>&1 | tee "${RUN_DIR}/eval_log.txt"

    echo "[完成] ${TESTSET_NAME} -> ${SUMMARY_OUTPUT}"
done

echo ""
echo "============================================"
echo "Pose-Text 批量评测完成！"
echo "============================================"
