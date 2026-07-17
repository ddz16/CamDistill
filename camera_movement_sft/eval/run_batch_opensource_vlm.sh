#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # (对这些开源模型无影响, 它们不走 qwen_vl_utils)
# ============================================================================
# 开源视频 VLM 基线评测: LLaVA-OneVision-7B / LLaVA-NeXT-Video-7B / InternVL3.5-8B
# ============================================================================
# 与本地 Qwen 模型同一 benchmark、同一英文 prompt、同一评测口径。
# 复用 prepare_test_data → swift infer → postprocess → evaluate。
#
# ⚠️ 帧采样说明: 这些模型是"固定帧数"采样(不是 fps)。swift 用环境变量控制帧数:
#     LLaVA(onevision/next-video) -> NUM_FRAMES ; InternVL -> VIDEO_SEGMENTS
#   为尽量对齐 Qwen 的 fps=5(本 benchmark 平均 ~31 帧/中位 24 帧), 默认取 FRAMES=32。
#   这是"固定帧数 ≈ fps=5 均值"的近似, 无法与真 fps=5 完全一致(公平性注意)。
#
# 依赖: av(已装), decord(已装); InternVL 还需 timm(如未装先 pip install timm)。
#
# 用法:
#   bash run_batch_opensource_vlm.sh <model> [testset.jsonl ...] [--force]
#     <model>: llava-ov-7b | llava-video-7b | internvl3.5-8b | 或直接给 HF id
#   例:
#     bash run_batch_opensource_vlm.sh llava-ov-7b /group/40009/dazhaodu/OurBenchmark/youtube_benchmark_subset.jsonl
#     FRAMES=24 bash run_batch_opensource_vlm.sh internvl3.5-8b <testset.jsonl>
# ============================================================================

# ---- 解析参数: 第1个是 model, 其余是 testset(--force 可夹在任意位置) ----
FORCE=false
MODEL_KEY=""
RAW_TEST_DATA_LIST=()
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        *) if [ -z "$MODEL_KEY" ]; then MODEL_KEY="$arg"; else RAW_TEST_DATA_LIST+=("$arg"); fi ;;
    esac
done

if [ -z "$MODEL_KEY" ]; then
    echo "[ERROR] 未指定模型。用法: bash $(basename "$0") <llava-ov-7b|llava-video-7b|internvl3.5-8b|HF id> [testset.jsonl ...]"
    exit 1
fi
if [ ${#RAW_TEST_DATA_LIST[@]} -eq 0 ]; then
    RAW_TEST_DATA_LIST=("/group/40009/dazhaodu/OurBenchmark/youtube_benchmark_subset.jsonl")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${PROJECT_ROOT}")"
cd "${REPO_ROOT}"

# ---- 模型映射(HF id + model_type + template) ----
MODEL_TYPE=""
TEMPLATE=""
case "$MODEL_KEY" in
    llava-ov-7b|llava-onevision-7b)
        MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
        MODEL_TYPE="llava_onevision_hf"; TEMPLATE="llava_onevision_hf"
        MODEL_TAG="${MODEL_TAG:-llava_ov_7b}" ;;
    llava-video-7b|llava-next-video-7b)
        MODEL="llava-hf/LLaVA-NeXT-Video-7B-hf"
        MODEL_TYPE="llava_next_video_hf"; TEMPLATE="llava_next_video_hf"
        MODEL_TAG="${MODEL_TAG:-llava_video_7b}" ;;
    internvl3.5-8b|internvl3_5-8b|internvl3.5-8b-hf)
        # HF 原生格式(InternVLForConditionalGeneration + tokenizer.json), 兼容 transformers 5.x
        MODEL="OpenGVLab/InternVL3_5-8B-HF"
        MODEL_TYPE="internvl"; TEMPLATE="internvl_hf"   # model_type=internvl, template=internvl_hf
        MODEL_TAG="${MODEL_TAG:-internvl3_5_8b_hf}" ;;
    internvl3.5-8b-instruct|internvl3_5-8b-instruct)
        # trust_remote_code: model_type=internvl_chat(InternVLLoader); 帧数用 VIDEO_SEGMENTS(load_video_internvl); 需 timm
        MODEL="OpenGVLab/InternVL3_5-8B-Instruct"
        MODEL_TYPE="internvl_chat"; TEMPLATE="internvl3_5"
        MODEL_TAG="${MODEL_TAG:-internvl3_5_8b_instruct}" ;;
    *)
        # 直接传 HF id: model_type/template 交给 swift 自动匹配(可用 MODEL_TYPE/TEMPLATE 环境变量覆盖)
        MODEL="$MODEL_KEY"
        MODEL_TYPE="${MODEL_TYPE_OVERRIDE:-}"; TEMPLATE="${TEMPLATE_OVERRIDE:-}"
        MODEL_TAG="${MODEL_TAG:-$(echo "$MODEL_KEY" | tr '/:' '__')}" ;;
esac

# ---- 帧数(近似 fps=5): LLaVA 用 NUM_FRAMES, InternVL 用 VIDEO_SEGMENTS, 两个都设 ----
FRAMES="${FRAMES:-32}"
export NUM_FRAMES="${NUM_FRAMES:-$FRAMES}"
export VIDEO_SEGMENTS="${VIDEO_SEGMENTS:-$FRAMES}"

# ---- 运行环境 ----
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/python3.12/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_HOME="${HF_HOME:-/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/hf}"
export USE_HF="${USE_HF:-1}"
export HF_TOKEN="${HF_TOKEN:-***REMOVED***}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

INFER_BACKEND="${INFER_BACKEND:-pt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-24576}"   # 视频视觉 token 多, 给足上下文

VIDEO_DIRS=(
    "/group/40009/dazhaodu/OurBenchmark/YouTube_Videos"
    "/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos"
)
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_camera_movement_fixed.py"

# model_type/template 参数(可能为空 -> 交给 swift 自动匹配)
MTYPE_ARGS=()
[ -n "$MODEL_TYPE" ] && MTYPE_ARGS+=(--model_type "$MODEL_TYPE")
[ -n "$TEMPLATE" ] && MTYPE_ARGS+=(--template "$TEMPLATE")

echo "============================================"
echo "开源视频 VLM 基线评测"
echo "============================================"
echo "模型:      $MODEL"
echo "model_type/template: ${MODEL_TYPE:-<auto>} / ${TEMPLATE:-<auto>}"
echo "标签:      $MODEL_TAG"
echo "帧数:      NUM_FRAMES=$NUM_FRAMES / VIDEO_SEGMENTS=$VIDEO_SEGMENTS (近似 fps=5)"
echo "测试集:    ${#RAW_TEST_DATA_LIST[@]} 个"
[ "$FORCE" = true ] && echo "模式: --force"
echo "============================================"
case "$MODEL_KEY" in internvl3.5-8b|internvl3_5-8b) echo "[NOTE] InternVL 需要 timm, 若报 No module named 'timm' 请先 pip install timm";; esac

# ============================================================================
for RAW_TEST_DATA in "${RAW_TEST_DATA_LIST[@]}"; do
    TESTSET_BASENAME=$(basename "${RAW_TEST_DATA}" .jsonl)
    TESTSET_NAME="${MODEL_TAG}_${TESTSET_BASENAME}"
    RUN_DIR="${SCRIPT_DIR}/eval_results/${TESTSET_NAME}"
    TEST_DATA_SWIFT="${RUN_DIR}/test_for_infer.jsonl"
    MAPPING_FILE="${RUN_DIR}/video_id_mapping.jsonl"
    CKPT_DIR="${RUN_DIR}/base"
    RESULT_PATH="${CKPT_DIR}/infer_results.jsonl"
    PRED_PATH="${CKPT_DIR}/eval_pred.jsonl"

    echo ""
    echo "████████ ${TESTSET_NAME} ████████"
    [ -f "${RAW_TEST_DATA}" ] || { echo "[ERROR] 测试集不存在: ${RAW_TEST_DATA}"; continue; }
    if [ "$FORCE" = true ]; then echo "[FORCE] 清理 ${RUN_DIR}"; rm -rf "${RUN_DIR}"; fi
    mkdir -p "${CKPT_DIR}"

    # Step 0: 数据转换(与其它评测一致)
    if [ ! -f "$TEST_DATA_SWIFT" ]; then
        echo "[Step 0] 转换测试数据..."
        python "${SCRIPT_DIR}/prepare_test_data.py" \
            --input "${RAW_TEST_DATA}" --output_dir "${RUN_DIR}" \
            --use_local_video --video_dir "${VIDEO_DIRS[@]}" --auto_download \
            || { echo "[ERROR] 数据转换失败"; continue; }
    else
        echo "[Step 0] 测试数据已存在"
    fi

    # 推理
    if [ -f "${RESULT_PATH}" ]; then
        echo "[推理] 已存在, 跳过: ${RESULT_PATH}"
    else
        echo "[推理] 开始 (${MODEL})..."
        MASTER_PORT="${MASTER_PORT:-29503}" \
        NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
        swift infer \
            --model "${MODEL}" \
            "${MTYPE_ARGS[@]}" \
            --use_hf true \
            --infer_backend ${INFER_BACKEND} \
            --val_dataset "${TEST_DATA_SWIFT}" \
            --result_path "${RESULT_PATH}" \
            --max_length ${MAX_LENGTH} \
            --temperature 0 \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --max_batch_size ${MAX_BATCH_SIZE} || true
        wait 2>/dev/null
        [ -f "${RESULT_PATH}" ] || { echo "[ERROR] 推理失败, 结果文件不存在"; continue; }
    fi

    # 后处理
    if [ -f "${PRED_PATH}" ]; then
        echo "[后处理] 已存在, 跳过"
    else
        python "${SCRIPT_DIR}/postprocess_results.py" \
            --infer_result "${RESULT_PATH}" --mapping "${MAPPING_FILE}" --output_dir "${CKPT_DIR}"
        [ -f "${PRED_PATH}" ] || { echo "[ERROR] 后处理失败"; continue; }
    fi

    # 评测(同口径)
    SUMMARY_OUTPUT="${RUN_DIR}/eval_results.json"
    echo "[评测] -> ${SUMMARY_OUTPUT}"
    python3 "${EVAL_SCRIPT}" \
        --gt "${RAW_TEST_DATA}" --pred "${PRED_PATH}" \
        --iou_thresh 0.3 0.5 0.7 --output "${SUMMARY_OUTPUT}" \
        2>&1 | tee "${RUN_DIR}/eval_log.txt"
    echo "[完成] ${TESTSET_NAME}"
done

echo ""
echo "开源 VLM 基线评测完成！"
