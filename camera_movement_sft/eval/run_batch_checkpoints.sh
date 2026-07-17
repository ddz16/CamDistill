#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # 默认 decord(可被环境变量覆盖), 避开 torchcodec 的 CUDA(libnvrtc.so.13)版本不匹配
# ============================================================================
# 批量评测多个 checkpoint —— 串行推理 + 后处理 + 自动评测对比
# ============================================================================
# 用法:
#   bash camera_movement_sft/eval/run_batch_checkpoints.sh                          # 用默认测试集
#   bash camera_movement_sft/eval/run_batch_checkpoints.sh testset_a.jsonl          # 单个测试集
#   bash camera_movement_sft/eval/run_batch_checkpoints.sh --force testset_a.jsonl  # 强制重跑
#
# 目录结构:
#   eval/eval_results/{测试集名}/
#   ├── test_for_infer.jsonl       # swift 推理数据
#   ├── video_id_mapping.jsonl     # 映射文件
#   ├── checkpoint-*/
#   │   ├── infer_results.jsonl    # 推理原始输出
#   │   └── eval_pred.jsonl        # 模型预测
#   ├── eval_results.json          # 汇总评测结果
#   └── eval_log.txt               # 评测日志
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"

cd "${PROJECT_ROOT}"
# 使用当前终端环境（已移除conda环境切换）

# ================================
# 解析命令行参数：--force + 测试集路径列表
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
# 训练输出根目录（自动扫描所有 checkpoint-* 目录）
# 请根据实际训练输出修改此路径
#
# 评测原始基座模型（未 SFT）：设置 BASE_MODEL 指向本地路径或 HF 名称，
# 此时跳过 checkpoint 扫描，直接评测该模型；可用 MODEL_TAG 自定义输出目录标签。
#   BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct MODEL_TAG=base_qwen3vl_8b bash camera_movement_sft/eval/run_batch_checkpoints.sh testset.jsonl
# ================================
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-output/camera_sft_qwen3vl_8b/v0-latest}"
BASE_MODEL="${BASE_MODEL:-}"

if [ -n "${BASE_MODEL}" ]; then
    # ---- 基座模型模式：不扫描 checkpoint，直接评测原始模型 ----
    TRAIN_TAG="${MODEL_TAG:-$(basename "${BASE_MODEL}")}"
    CHECKPOINTS=("base")
    echo "[INFO] BASE_MODEL 模式，评测原始模型: ${BASE_MODEL}（标签: ${TRAIN_TAG}）"
else
    # ---- SFT checkpoint 模式 ----
    # 训练版本标识（用于输出目录命名）
    TRAIN_PARENT=$(basename "$(dirname "${TRAIN_OUTPUT_DIR}")")
    TRAIN_LEAF=$(basename "${TRAIN_OUTPUT_DIR}")
    TRAIN_VER="${TRAIN_LEAF%%-*}"
    TRAIN_TAG="${TRAIN_PARENT#output_}_${TRAIN_VER}"
    TRAIN_TAG="${TRAIN_TAG#camera_sft_}"

    # 自动扫描所有 checkpoint-* 目录，按数字排序
    CHECKPOINTS=()
    for d in "${TRAIN_OUTPUT_DIR}"/checkpoint-*; do
        if [ -d "$d" ]; then
            CHECKPOINTS+=("$(basename "$d")")
        fi
    done

    IFS=$'\n' CHECKPOINTS=($(printf '%s\n' "${CHECKPOINTS[@]}" | sort -t'-' -k2 -n)); unset IFS

    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "[ERROR] 在 ${TRAIN_OUTPUT_DIR} 下未找到任何 checkpoint-* 目录！"
        echo "请设置 TRAIN_OUTPUT_DIR 环境变量，例如:"
        echo "  TRAIN_OUTPUT_DIR=output/camera_sft_qwen3vl_8b/v0-20260520-120000 bash camera_movement_sft/eval/run_batch_checkpoints.sh"
        echo "或评测原始基座模型:"
        echo "  BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct bash camera_movement_sft/eval/run_batch_checkpoints.sh"
        exit 1
    fi

    # 只评估最后一个 checkpoint（默认开启，设 ONLY_LAST=false 评估全部）
    ONLY_LAST="${ONLY_LAST:-true}"
    if [ "${ONLY_LAST}" = "true" ]; then
        CHECKPOINTS=("${CHECKPOINTS[-1]}")
        echo "[INFO] ONLY_LAST=true，仅评估: ${CHECKPOINTS[0]}（设 ONLY_LAST=false 评估全部）"
    fi
fi


export HF_HOME=/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/hf
export USE_HF="${USE_HF:-1}"  # 默认用 HuggingFace hub(命中 HF_HOME 缓存); 设 USE_HF=0 切回 ModelScope
export HF_TOKEN=***REMOVED***

# ================================
# 动态库路径配置
# ================================
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/python3.12/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"

# ================================
# GPU 配置
# ================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ================================
# 视频处理配置（与训练完全一致）
# ================================
export FPS=5
export FPS_MIN_FRAMES=4
export FPS_MAX_FRAMES=100
export VIDEO_MAX_PIXELS=100352

# ================================
# 推理配置
# ================================
INFER_BACKEND="pt"
TEMPERATURE=0
MAX_NEW_TOKENS=4096
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"

# ================================
# CamInject 评测可选配置
# ================================
USE_CAMINJECT="${USE_CAMINJECT:-0}"                     # 1: 使用 caminject 插件推理
VGGT_MODE="${VGGT_MODE:-cache}"                         # cache 或 online
VGGT_CACHE_DIR="${VGGT_CACHE_DIR:-}"                    # cache 模式必填
VGGT_TEACHER_TYPE="${VGGT_TEACHER_TYPE:-vggt}"          # vggt 或 vggt_omega
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-}"                  # online 模式用；留空时按 teacher 自动选择
if [ -z "${VGGT_MODEL_PATH}" ]; then
    if [ "${VGGT_TEACHER_TYPE}" = "vggt_omega" ]; then
        VGGT_MODEL_PATH="facebook/VGGT-Omega"
    else
        VGGT_MODEL_PATH="facebook/VGGT-1B"
    fi
fi
CAMINJECT_MODEL_TYPE="${CAMINJECT_MODEL_TYPE:-qwen3_vl_caminject}"  # qwen3.5 请改成 qwen3_5_caminject
VGGT_ONLINE_FPS="${VGGT_ONLINE_FPS:-${FPS}}"
VGGT_ONLINE_MAX_FRAMES="${VGGT_ONLINE_MAX_FRAMES:-${FPS_MAX_FRAMES}}"
CAMINJECT_STRICT_IDS="${CAMINJECT_STRICT_IDS:-1}"
CAMINJECT_STRICT_CACHE="${CAMINJECT_STRICT_CACHE:-1}"
CAMINJECT_MAX_MISS_RATIO="${CAMINJECT_MAX_MISS_RATIO:-0.0}"
CAMINJECT_MIN_RATIO_SAMPLES="${CAMINJECT_MIN_RATIO_SAMPLES:-32}"
CAMINJECT_LOG_EVERY="${CAMINJECT_LOG_EVERY:-50}"
CAMINJECT_PLUGIN_PATH="${CAMINJECT_PLUGIN_PATH:-camera_movement_sft/plugins/camdistill_plugin.py}"

if [ "${USE_CAMINJECT}" = "1" ]; then
    export VGGT_MODE VGGT_CACHE_DIR VGGT_TEACHER_TYPE VGGT_MODEL_PATH
    export VGGT_ONLINE_FPS VGGT_ONLINE_MAX_FRAMES
    export CAMINJECT_STRICT_IDS CAMINJECT_STRICT_CACHE
    export CAMINJECT_MAX_MISS_RATIO CAMINJECT_MIN_RATIO_SAMPLES CAMINJECT_LOG_EVERY

    if [ "${VGGT_MODE}" = "cache" ]; then
        if [ -z "${VGGT_CACHE_DIR}" ]; then
            echo "[ERROR] USE_CAMINJECT=1 且 VGGT_MODE=cache 时必须设置 VGGT_CACHE_DIR"
            exit 1
        fi
        if [ ! -d "${VGGT_CACHE_DIR}" ]; then
            echo "[ERROR] VGGT_CACHE_DIR 不存在: ${VGGT_CACHE_DIR}"
            exit 1
        fi
    elif [ "${VGGT_MODE}" = "online" ]; then
        export VGGT_CACHE_DIR=""
    else
        echo "[ERROR] 不支持的 VGGT_MODE=${VGGT_MODE}（仅支持 cache 或 online）"
        exit 1
    fi

    if [ ! -f "${CAMINJECT_PLUGIN_PATH}" ]; then
        echo "[ERROR] CamInject 插件不存在: ${CAMINJECT_PLUGIN_PATH}"
        exit 1
    fi
fi

CAMINJECT_INFER_EXTRA_ARGS=()
if [ "${USE_CAMINJECT}" = "1" ]; then
    CAMINJECT_INFER_EXTRA_ARGS+=(--model_type "${CAMINJECT_MODEL_TYPE}")
    CAMINJECT_INFER_EXTRA_ARGS+=(--external_plugins "${CAMINJECT_PLUGIN_PATH}")
fi

# ================================
# 显式指定 model_type（可选）
# 某些 HF 微调模型 swift 自动匹配 model_type 有歧义（同一 architecture 被多个 type 注册，
# 如 Qwen2_5_VLForConditionalGeneration 同时匹配 qwen2_5_vl / mimo_vl），需手动指定。
#   例: 评测 cam-motion-7b
#   BASE_MODEL=chancharikm/qwen2.5-vl-7b-cam-motion MODEL_TAG=cam_motion_7b MODEL_TYPE=qwen2_5_vl bash ...
# 注意: USE_CAMINJECT=1 时 model_type 已由 CamInject 分支指定，勿再设 MODEL_TYPE。
# ================================
MODEL_TYPE="${MODEL_TYPE:-}"
MODEL_TYPE_ARGS=()
if [ -n "${MODEL_TYPE}" ] && [ "${USE_CAMINJECT}" != "1" ]; then
    MODEL_TYPE_ARGS+=(--model_type "${MODEL_TYPE}" --template "${MODEL_TYPE}")
fi

# ================================
# 数据路径
# ================================
if [ ${#RAW_TEST_DATA_LIST[@]} -eq 0 ]; then
    RAW_TEST_DATA_LIST=(
        "/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/test/500评测集_人工筛选后_with_split_clips.jsonl"
    )
fi

# 本地视频候选目录
VIDEO_DIRS=(
    "/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos"
    "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_21k_video_shot_cameraMove/videos"
    "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_realfilm_30w_260401_deduped_final/videos"
)

EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_camera_movement_fixed.py"

echo "============================================"
echo "批量评测多个 checkpoint × 多个测试集"
echo "============================================"
echo "训练目录: ${TRAIN_OUTPUT_DIR}"
echo "训练标签 (TRAIN_TAG): ${TRAIN_TAG}"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "推理后端: ${INFER_BACKEND}"
echo "USE_CAMINJECT: ${USE_CAMINJECT}"
if [ "${USE_CAMINJECT}" = "1" ]; then
    echo "VGGT mode: ${VGGT_MODE}"
    echo "VGGT teacher: ${VGGT_TEACHER_TYPE}"
    echo "VGGT model: ${VGGT_MODEL_PATH}"
    echo "CamInject model_type: ${CAMINJECT_MODEL_TYPE}"
    echo "VGGT cache: ${VGGT_CACHE_DIR:-<disabled>}"
    echo "VGGT online fps/max_frames: ${VGGT_ONLINE_FPS}/${VGGT_ONLINE_MAX_FRAMES}"
fi
echo "测试集 (${#RAW_TEST_DATA_LIST[@]} 个):"
for td in "${RAW_TEST_DATA_LIST[@]}"; do
    echo "  - $(basename "$td")"
done
if [ "$FORCE" = true ]; then
echo "模式: --force（覆盖旧结果）"
fi
echo "============================================"
echo ""

# ============================================================================
# 外层循环：逐个测试集
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
    echo "██  测试集 [${TESTSET_IDX}/${TESTSET_TOTAL}]: ${TESTSET_NAME}"
    echo "██  路径: ${RAW_TEST_DATA}"
    echo "████████████████████████████████████████████████████████████████"

    if [ ! -f "${RAW_TEST_DATA}" ]; then
        echo "[ERROR] 测试集文件不存在: ${RAW_TEST_DATA}，跳过！"
        continue
    fi

    mkdir -p "${RUN_DIR}"

    ALL_PRED_PATHS=()

    if [ "$FORCE" = true ]; then
        echo "[FORCE] 清理旧结果: ${RUN_DIR}/"
        rm -rf "${RUN_DIR}"
        mkdir -p "${RUN_DIR}"
    fi

    # ================================
    # Step 0: 数据转换
    # ================================
    if [ ! -f "$TEST_DATA_SWIFT" ]; then
        echo "[Step 0] 测试数据不存在，先转换..."
        python ${SCRIPT_DIR}/prepare_test_data.py \
            --input "${RAW_TEST_DATA}" \
            --output_dir "${RUN_DIR}" \
            --use_local_video \
            --video_dir ${VIDEO_DIRS[@]} \
            --auto_download
        if [ $? -ne 0 ]; then
            echo "[ERROR] 数据转换失败，跳过此测试集！"
            continue
        fi
    else
        echo "[Step 0] 测试数据已存在: ${TEST_DATA_SWIFT}"
    fi

    # ================================
    # 逐个 checkpoint 推理 + 后处理
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
        echo "  模型: ${MODEL}"
        echo "  输出: ${CKPT_DIR}"

        mkdir -p "${CKPT_DIR}"

        # ---- 推理 ----
        if [ -f "${RESULT_PATH}" ]; then
            echo "  [推理] 结果已存在，跳过: ${RESULT_PATH}"
        else
            echo "  [推理] 开始..."
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
            echo "  [推理] 耗时: ${ELAPSED}s"

            if [ ! -f "${RESULT_PATH}" ]; then
                echo "  [ERROR] 推理失败！结果文件不存在，跳过此 checkpoint"
                continue
            fi
        fi

        # ---- 后处理 ----
        if [ -f "${PRED_PATH}" ]; then
            echo "  [后处理] 结果已存在，跳过"
        else
            echo "  [后处理] 生成 eval_pred..."
            python ${SCRIPT_DIR}/postprocess_results.py \
                --infer_result ${RESULT_PATH} \
                --mapping ${MAPPING_FILE} \
                --output_dir ${CKPT_DIR}

            if [ ! -f "${PRED_PATH}" ]; then
                echo "  [ERROR] 后处理失败！eval_pred.jsonl 未生成，跳过此 checkpoint"
                continue
            fi
        fi

        ALL_PRED_PATHS+=("${PRED_PATH}")
        echo "  [完成] ${CKPT} ✓"
    done

    # ================================
    # 汇总评测
    # ================================
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  汇总评测对比 — ${TESTSET_NAME}"
    echo "╚══════════════════════════════════════════════╝"

    if [ ${#ALL_PRED_PATHS[@]} -eq 0 ]; then
        echo "[ERROR] ${TESTSET_NAME}: 没有任何有效的预测结果，跳过评测！"
        continue
    fi

    SUMMARY_OUTPUT="${RUN_DIR}/eval_results.json"

    # 判断是否为 split_clips 评测集，如果是则执行 merge_and_eval
    if echo "${TESTSET_BASENAME}" | grep -q "split_clips"; then
        # split_clips 评测集：merge 切片预测回原视频维度后再评测
        MERGE_EVAL_SCRIPT="${SCRIPT_DIR}/merge_split_clips_eval.py"
        # 写死对应的 GT 文件路径
        SPLIT_CLIPS_GT="/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/test/500评测集-0519-gt结果-2026-05-21--标注结果459条--人工筛选后.jsonl"

        echo "  [Merge+Eval] 检测到 split_clips 评测集，执行 merge 后评测"
        echo "  [Merge+Eval] 切片数据: ${RAW_TEST_DATA}"
        echo "  [Merge+Eval] GT 文件:  ${SPLIT_CLIPS_GT}"

        python3 ${MERGE_EVAL_SCRIPT} \
            --split_clips_file "${RAW_TEST_DATA}" \
            --pred ${ALL_PRED_PATHS[@]} \
            --gt "${SPLIT_CLIPS_GT}" \
            --eval_script "${EVAL_SCRIPT}" \
            --iou_thresh 0.3 0.5 0.7 \
            --output "${SUMMARY_OUTPUT}" \
            2>&1 | tee "${RUN_DIR}/eval_log.txt"
    else
        # 普通评测集：直接评测
        python3 ${EVAL_SCRIPT} \
            --gt "${RAW_TEST_DATA}" \
            --pred ${ALL_PRED_PATHS[@]} \
            --iou_thresh 0.3 0.5 0.7 \
            --output "${SUMMARY_OUTPUT}" \
            2>&1 | tee "${RUN_DIR}/eval_log.txt"
    fi

    echo ""
    echo "  ✓ ${TESTSET_NAME} 评测完成"
    echo "    运行目录: ${RUN_DIR}/"
    echo "    汇总评测: ${SUMMARY_OUTPUT}"
    echo "    评测日志: ${RUN_DIR}/eval_log.txt"

done

# ================================
# 最终总结
# ================================
echo ""
echo "============================================"
echo "全部完成！共处理 ${TESTSET_TOTAL} 个测试集 × ${#CHECKPOINTS[@]} 个 checkpoint"
echo "============================================"
