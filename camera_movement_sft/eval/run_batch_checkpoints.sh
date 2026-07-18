#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # Default to decord (overridable via env), avoids torchcodec CUDA (libnvrtc.so.13) version mismatch
# ============================================================================
# Batch-evaluate multiple checkpoints -- serial inference + post-processing + auto eval comparison.
# ============================================================================
# Usage:
#   bash camera_movement_sft/eval/run_batch_checkpoints.sh                          # use default test set
#   bash camera_movement_sft/eval/run_batch_checkpoints.sh testset_a.jsonl          # single test set
#   bash camera_movement_sft/eval/run_batch_checkpoints.sh --force testset_a.jsonl  # force re-run
#
# Directory structure:
#   eval/eval_results/{testset_name}/
#   ├── test_for_infer.jsonl       # swift inference data
#   ├── video_id_mapping.jsonl     # mapping file
#   ├── checkpoint-*/
#   │   ├── infer_results.jsonl    # raw inference output
#   │   └── eval_pred.jsonl        # model predictions
#   ├── eval_results.json          # aggregated evaluation results
#   └── eval_log.txt               # evaluation log
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"

cd "${PROJECT_ROOT}"
# Use the current shell environment (conda switching removed).

# ================================
# Parse command-line arguments: --force + test-set path list.
# ================================
FORCE=false
RAW_TEST_DATA_LIST=()
for arg in "$@"; do
    case $arg in
        --force)
            FORCE=true
            ;;
        *)
            RAW_TEST_DATA_LIST+=("$arg")
            ;;
    esac
done

# ================================
# Training output root directory (auto-scans all checkpoint-* subdirectories).
# Adjust this path to match your actual training output.
#
# To evaluate the raw base model (no SFT): set BASE_MODEL to a local path or HF name;
# checkpoint scanning is skipped and the model is evaluated directly.
# Use MODEL_TAG to customise the output directory label.
#   BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct MODEL_TAG=base_qwen3vl_8b bash camera_movement_sft/eval/run_batch_checkpoints.sh testset.jsonl
# ================================
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-output/camera_sft_qwen3vl_8b/v0-latest}"
BASE_MODEL="${BASE_MODEL:-}"

if [ -n "${BASE_MODEL}" ]; then
    # ---- Base-model mode: skip checkpoint scanning, evaluate the raw model directly. ----
    TRAIN_TAG="${MODEL_TAG:-$(basename "${BASE_MODEL}")}"
    CHECKPOINTS=("base")
    echo "[INFO] BASE_MODEL mode, evaluating raw model: ${BASE_MODEL} (tag: ${TRAIN_TAG})"
else
    # ---- SFT checkpoint mode. ----
    # Training version identifier (used for output directory naming).
    TRAIN_PARENT=$(basename "$(dirname "${TRAIN_OUTPUT_DIR}")")
    TRAIN_LEAF=$(basename "${TRAIN_OUTPUT_DIR}")
    TRAIN_VER="${TRAIN_LEAF%%-*}"
    TRAIN_TAG="${TRAIN_PARENT#output_}_${TRAIN_VER}"
    TRAIN_TAG="${TRAIN_TAG#camera_sft_}"

    # Auto-scan all checkpoint-* directories, sorted numerically.
    CHECKPOINTS=()
    for d in "${TRAIN_OUTPUT_DIR}"/checkpoint-*; do
        if [ -d "$d" ]; then
            CHECKPOINTS+=("$(basename "$d")")
        fi
    done

    IFS=$'\n' CHECKPOINTS=($(printf '%s\n' "${CHECKPOINTS[@]}" | sort -t'-' -k2 -n)); unset IFS

    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "[ERROR] No checkpoint-* directories found under ${TRAIN_OUTPUT_DIR}!"
        echo "Set the TRAIN_OUTPUT_DIR environment variable, e.g.:"
        echo "  TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_8b/v0-20260520-120000 bash camera_movement_sft/eval/run_batch_checkpoints.sh"
        echo "Or evaluate the raw base model:"
        echo "  BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct bash camera_movement_sft/eval/run_batch_checkpoints.sh"
        exit 1
    fi

    # Evaluate only the last checkpoint by default (set ONLY_LAST=false to evaluate all).
    ONLY_LAST="${ONLY_LAST:-true}"
    if [ "${ONLY_LAST}" = "true" ]; then
        CHECKPOINTS=("${CHECKPOINTS[-1]}")
        echo "[INFO] ONLY_LAST=true, evaluating only: ${CHECKPOINTS[0]} (set ONLY_LAST=false to evaluate all)"
    fi
fi


export HF_HOME=/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/hf
export USE_HF="${USE_HF:-1}"  # Default: use HuggingFace hub (hits HF_HOME cache); set USE_HF=0 to fall back to ModelScope
export HF_TOKEN=***REMOVED***

# ================================
# Dynamic library path configuration.
# ================================
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/python3.12/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"

# ================================
# GPU configuration.
# ================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ================================
# Video processing configuration (must match training exactly).
# ================================
export FPS=5
export FPS_MIN_FRAMES=4
export FPS_MAX_FRAMES=100
export VIDEO_MAX_PIXELS=100352

# ================================
# Inference configuration.
# ================================
INFER_BACKEND="pt"
TEMPERATURE=0
MAX_NEW_TOKENS=4096
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"

# ================================
# Optional CamInject evaluation configuration.
# ================================
USE_CAMINJECT="${USE_CAMINJECT:-0}"                     # 1: use the caminject plugin for inference
VGGT_MODE="online"                                      # always online mode for inference
VGGT_TEACHER_TYPE="${VGGT_TEACHER_TYPE:-vggt}"          # vggt or vggt_omega
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-}"                  # used in online mode; auto-selected when empty
if [ -z "${VGGT_MODEL_PATH}" ]; then
    if [ "${VGGT_TEACHER_TYPE}" = "vggt_omega" ]; then
        VGGT_MODEL_PATH="facebook/VGGT-Omega"
    else
        VGGT_MODEL_PATH="facebook/VGGT-1B"
    fi
fi
CAMINJECT_MODEL_TYPE="${CAMINJECT_MODEL_TYPE:-qwen3_vl_caminject}"  # use qwen3_5_caminject for Qwen3.5
VGGT_ONLINE_FPS="${VGGT_ONLINE_FPS:-${FPS}}"
VGGT_ONLINE_MAX_FRAMES="${VGGT_ONLINE_MAX_FRAMES:-${FPS_MAX_FRAMES}}"
CAMINJECT_STRICT_IDS="${CAMINJECT_STRICT_IDS:-1}"
CAMINJECT_STRICT_CACHE="${CAMINJECT_STRICT_CACHE:-1}"
CAMINJECT_MAX_MISS_RATIO="${CAMINJECT_MAX_MISS_RATIO:-0.0}"
CAMINJECT_MIN_RATIO_SAMPLES="${CAMINJECT_MIN_RATIO_SAMPLES:-32}"
CAMINJECT_LOG_EVERY="${CAMINJECT_LOG_EVERY:-50}"
CAMINJECT_PLUGIN_PATH="${CAMINJECT_PLUGIN_PATH:-camera_movement_sft/plugins/camdistill_plugin.py}"

if [ "${USE_CAMINJECT}" = "1" ]; then
    export VGGT_MODE VGGT_TEACHER_TYPE VGGT_MODEL_PATH
    export VGGT_ONLINE_FPS VGGT_ONLINE_MAX_FRAMES
    export CAMINJECT_STRICT_IDS CAMINJECT_STRICT_CACHE
    export CAMINJECT_MAX_MISS_RATIO CAMINJECT_MIN_RATIO_SAMPLES CAMINJECT_LOG_EVERY
    export VGGT_CACHE_DIR=""  # always online; cache dir unused

    if [ ! -f "${CAMINJECT_PLUGIN_PATH}" ]; then
        echo "[ERROR] CamInject plugin not found: ${CAMINJECT_PLUGIN_PATH}"
        exit 1
    fi
fi

CAMINJECT_INFER_EXTRA_ARGS=()
if [ "${USE_CAMINJECT}" = "1" ]; then
    CAMINJECT_INFER_EXTRA_ARGS+=(--model_type "${CAMINJECT_MODEL_TYPE}")
    CAMINJECT_INFER_EXTRA_ARGS+=(--external_plugins "${CAMINJECT_PLUGIN_PATH}")
fi

# ================================
# Explicit model_type override (optional).
# Some HF fine-tuned models have ambiguous model_type auto-detection in swift (e.g. the same
# architecture is registered under multiple types such as qwen2_5_vl / mimo_vl); specify manually.
#   Example: evaluate cam-motion-7b
#   BASE_MODEL=chancharikm/qwen2.5-vl-7b-cam-motion MODEL_TAG=cam_motion_7b MODEL_TYPE=qwen2_5_vl bash ...
# Note: when USE_CAMINJECT=1, model_type is already set by the CamInject branch; do not set MODEL_TYPE.
# ================================
MODEL_TYPE="${MODEL_TYPE:-}"
MODEL_TYPE_ARGS=()
if [ -n "${MODEL_TYPE}" ] && [ "${USE_CAMINJECT}" != "1" ]; then
    MODEL_TYPE_ARGS+=(--model_type "${MODEL_TYPE}" --template "${MODEL_TYPE}")
fi

# ================================
# Data paths.
# ================================
if [ ${#RAW_TEST_DATA_LIST[@]} -eq 0 ]; then
    RAW_TEST_DATA_LIST=(
        "/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/test/500评测集_人工筛选后_with_split_clips.jsonl"
    )
fi

# Local video candidate directories.
VIDEO_DIRS=(
    "/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos"
    "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_21k_video_shot_cameraMove/videos"
    "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_realfilm_30w_260401_deduped_final/videos"
)

EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_camera_movement_fixed.py"

echo "============================================"
echo "Batch evaluation: multiple checkpoints x multiple test sets"
echo "============================================"
echo "Training dir:  ${TRAIN_OUTPUT_DIR}"
echo "Training tag:  ${TRAIN_TAG}"
echo "Checkpoints:   ${CHECKPOINTS[*]}"
echo "Infer backend: ${INFER_BACKEND}"
echo "USE_CAMINJECT: ${USE_CAMINJECT}"
if [ "${USE_CAMINJECT}" = "1" ]; then
    echo "VGGT mode:     online (fixed)"
    echo "VGGT teacher:  ${VGGT_TEACHER_TYPE}"
    echo "VGGT model:    ${VGGT_MODEL_PATH}"
    echo "CamInject type:${CAMINJECT_MODEL_TYPE}"
    echo "VGGT online fps/max_frames: ${VGGT_ONLINE_FPS}/${VGGT_ONLINE_MAX_FRAMES}"
fi
echo "Test sets (${#RAW_TEST_DATA_LIST[@]}):"
for td in "${RAW_TEST_DATA_LIST[@]}"; do
    echo "  - $(basename "$td")"
done
if [ "$FORCE" = true ]; then
echo "Mode: --force (overwrite existing results)"
fi
echo "============================================"
echo ""

# ============================================================================
# Outer loop: iterate over test sets.
# ============================================================================
TESTSET_TOTAL=${#RAW_TEST_DATA_LIST[@]}
TESTSET_IDX=0

for RAW_TEST_DATA in "${RAW_TEST_DATA_LIST[@]}"; do
    TESTSET_IDX=$((TESTSET_IDX + 1))

    TESTSET_BASENAME=$(basename "${RAW_TEST_DATA}" .jsonl)
    TESTSET_NAME="${TRAIN_TAG}_${TESTSET_BASENAME}"

    RUN_DIR="${SCRIPT_DIR}/eval_results/${TESTSET_NAME}"
    TEST_DATA_SWIFT="${RUN_DIR}/test_for_infer.jsonl"
    MAPPING_FILE="${RUN_DIR}/video_id_mapping.jsonl"

    echo ""
    echo "████████████████████████████████████████████████████████████████"
    echo "██  Test set [${TESTSET_IDX}/${TESTSET_TOTAL}]: ${TESTSET_NAME}"
    echo "██  Path: ${RAW_TEST_DATA}"
    echo "████████████████████████████████████████████████████████████████"

    if [ ! -f "${RAW_TEST_DATA}" ]; then
        echo "[ERROR] Test set file not found: ${RAW_TEST_DATA}, skipping!"
        continue
    fi

    mkdir -p "${RUN_DIR}"

    ALL_PRED_PATHS=()

    if [ "$FORCE" = true ]; then
        echo "[FORCE] Cleaning old results: ${RUN_DIR}/"
        rm -rf "${RUN_DIR}"
        mkdir -p "${RUN_DIR}"
    fi

    # ================================
    # Step 0: Data conversion.
    # ================================
    if [ ! -f "$TEST_DATA_SWIFT" ]; then
        echo "[Step 0] Test data not found, converting..."
        python ${SCRIPT_DIR}/prepare_test_data.py \
            --input "${RAW_TEST_DATA}" \
            --output_dir "${RUN_DIR}" \
            --use_local_video \
            --video_dir ${VIDEO_DIRS[@]} \
            --auto_download
        if [ $? -ne 0 ]; then
            echo "[ERROR] Data conversion failed, skipping this test set!"
            continue
        fi
    else
        echo "[Step 0] Test data already exists: ${TEST_DATA_SWIFT}"
    fi

    # ================================
    # Per-checkpoint inference + post-processing.
    # ================================
    TOTAL=${#CHECKPOINTS[@]}
    IDX=0

    for CKPT in "${CHECKPOINTS[@]}"; do
        IDX=$((IDX + 1))
        if [ -n "${BASE_MODEL}" ]; then
            MODEL="${BASE_MODEL}"
        else
            MODEL="${TRAIN_OUTPUT_DIR}/${CKPT}"
        fi
        CKPT_DIR="${RUN_DIR}/${CKPT}"
        RESULT_PATH="${CKPT_DIR}/infer_results.jsonl"
        PRED_PATH="${CKPT_DIR}/eval_pred.jsonl"

        echo ""
        echo "╔══════════════════════════════════════════════╗"
        echo "║  [${TESTSET_IDX}/${TESTSET_TOTAL}] ${TESTSET_NAME} — [${IDX}/${TOTAL}] ${CKPT}"
        echo "╚══════════════════════════════════════════════╝"
        echo "  Model:  ${MODEL}"
        echo "  Output: ${CKPT_DIR}"

        mkdir -p "${CKPT_DIR}"

        # ---- Inference ----
        if [ -f "${RESULT_PATH}" ]; then
            echo "  [Infer] Result already exists, skipping: ${RESULT_PATH}"
        else
            echo "  [Infer] Starting..."
            START_TIME=$(date +%s)

            MASTER_PORT="${MASTER_PORT:-29501}" \
            NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
            swift infer \
                --model ${MODEL} \
                "${CAMINJECT_INFER_EXTRA_ARGS[@]}" \
                "${MODEL_TYPE_ARGS[@]}" \
                --infer_backend ${INFER_BACKEND} \
                --val_dataset ${TEST_DATA_SWIFT} \
                --result_path ${RESULT_PATH} \
                --temperature ${TEMPERATURE} \
                --max_new_tokens ${MAX_NEW_TOKENS} \
                --max_batch_size ${MAX_BATCH_SIZE} || true
            wait 2>/dev/null

            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "  [Infer] Elapsed: ${ELAPSED}s"

            if [ ! -f "${RESULT_PATH}" ]; then
                echo "  [ERROR] Inference failed! Result file not found, skipping this checkpoint."
                continue
            fi
        fi

        # ---- Post-processing ----
        if [ -f "${PRED_PATH}" ]; then
            echo "  [Post-process] Result already exists, skipping"
        else
            echo "  [Post-process] Generating eval_pred..."
            python ${SCRIPT_DIR}/postprocess_results.py \
                --infer_result ${RESULT_PATH} \
                --mapping ${MAPPING_FILE} \
                --output_dir ${CKPT_DIR}

            if [ ! -f "${PRED_PATH}" ]; then
                echo "  [ERROR] Post-processing failed! eval_pred.jsonl not generated, skipping this checkpoint."
                continue
            fi
        fi

        ALL_PRED_PATHS+=("${PRED_PATH}")
        echo "  [Done] ${CKPT} ✓"
    done

    # ================================
    # Aggregated evaluation.
    # ================================
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Aggregated evaluation — ${TESTSET_NAME}"
    echo "╚══════════════════════════════════════════════╝"

    if [ ${#ALL_PRED_PATHS[@]} -eq 0 ]; then
        echo "[ERROR] ${TESTSET_NAME}: no valid prediction results, skipping evaluation!"
        continue
    fi

    SUMMARY_OUTPUT="${RUN_DIR}/eval_results.json"

    # Detect split_clips test sets; if found, merge clip predictions back to the original video
    # dimension before evaluating.
    if echo "${TESTSET_BASENAME}" | grep -q "split_clips"; then
        MERGE_EVAL_SCRIPT="${SCRIPT_DIR}/merge_split_clips_eval.py"
        # Hard-coded GT file path for the split_clips test set.
        SPLIT_CLIPS_GT="/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/test/500评测集-0519-gt结果-2026-05-21--标注结果459条--人工筛选后.jsonl"

        echo "  [Merge+Eval] Detected split_clips test set, running merge before evaluation."
        echo "  [Merge+Eval] Split-clips data: ${RAW_TEST_DATA}"
        echo "  [Merge+Eval] GT file:          ${SPLIT_CLIPS_GT}"

        python3 ${MERGE_EVAL_SCRIPT} \
            --split_clips_file "${RAW_TEST_DATA}" \
            --pred ${ALL_PRED_PATHS[@]} \
            --gt "${SPLIT_CLIPS_GT}" \
            --eval_script "${EVAL_SCRIPT}" \
            --iou_thresh 0.3 0.5 0.7 \
            --output "${SUMMARY_OUTPUT}" \
            2>&1 | tee "${RUN_DIR}/eval_log.txt"
    else
        # Standard test set: evaluate directly.
        python3 ${EVAL_SCRIPT} \
            --gt "${RAW_TEST_DATA}" \
            --pred ${ALL_PRED_PATHS[@]} \
            --iou_thresh 0.3 0.5 0.7 \
            --output "${SUMMARY_OUTPUT}" \
            2>&1 | tee "${RUN_DIR}/eval_log.txt"
    fi

    echo ""
    echo "  ✓ ${TESTSET_NAME} evaluation complete"
    echo "    Run dir:     ${RUN_DIR}/"
    echo "    Eval result: ${SUMMARY_OUTPUT}"
    echo "    Eval log:    ${RUN_DIR}/eval_log.txt"

done

# ================================
# Final summary.
# ================================
echo ""
echo "============================================"
echo "All done! Processed ${TESTSET_TOTAL} test set(s) x ${#CHECKPOINTS[@]} checkpoint(s)."
echo "============================================"
