#!/usr/bin/env python3
"""
推理结果后处理：将 swift infer 的输出解析为评测用的 eval_pred.jsonl

swift infer 输出格式（每行）:
    {"response": "模型输出文本", "labels": "ground_truth文本", ...}

本脚本输出:
    eval_pred.jsonl — 模型预测，可直接用于 evaluate_camera_movement_fixed.py

GT 不需要本脚本生成，评测时直接用原始标注 JSONL（RAW_TEST_DATA）即可。

使用方法:
    python camera_movement_sft/eval/postprocess_results.py \
        --infer_result path/to/infer_results.jsonl \
        --mapping path/to/video_id_mapping.jsonl \
        --output_dir path/to/checkpoint_dir

    # 同时生成 debug 文件（含 raw_response，用于排查问题）
    python camera_movement_sft/eval/postprocess_results.py \
        --infer_result path/to/infer_results.jsonl \
        --mapping path/to/video_id_mapping.jsonl \
        --output_dir path/to/checkpoint_dir \
        --debug
"""

import argparse
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_json_response(raw_text: str) -> dict:
    """尝试从模型输出中解析 JSON"""
    if not raw_text:
        return {"raw": raw_text}

    # 直接解析
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 块
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取第一个 { ... } 块
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"raw": raw_text}


def load_jsonl(path: str) -> list:
    """加载 JSONL 文件"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _vid_from_videos_field(videos_field):
    """从 swift infer 输出的 'videos' 字段抽 video_id (mp4 文件名去扩展名)。
    videos_field 形如 ['/path/to/<video_id>.mp4'] 或 ['http://.../<video_id>.mp4?...']。
    抽不到时返回 None。"""
    if not videos_field:
        return None
    p = videos_field[0]
    if not isinstance(p, str):
        return None
    p = p.split("?")[0]
    fname = p.rsplit("/", 1)[-1]
    if "." in fname:
        fname = fname.rsplit(".", 1)[0]
    return fname or None


def _normalize_video_source(path_or_url):
    """归一化视频路径/URL，用于在 infer 结果与 mapping 间做稳健对齐。"""
    if not isinstance(path_or_url, str):
        return None
    p = path_or_url.strip()
    if not p:
        return None
    return p.split("?")[0]


def postprocess(infer_result_path, mapping_path, output_dir, debug=False):
    """解析 swift infer 结果，产出 eval_pred.jsonl

    video_id 来源(优先级):
      1. 从 infer_result 每行的 'videos' 字段提取(mp4 文件名)。
         这样即使 infer_results 比 mapping 少几行(漏推理),也不会发生错位。
      2. 兜底: 按行号 i 取 mappings[i].video_id (旧逻辑)。
    """
    mappings = load_jsonl(mapping_path)
    results = load_jsonl(infer_result_path)

    print(f"[INFO] 映射条数: {len(mappings)}")
    print(f"[INFO] 推理结果: {len(results)}")
    if len(mappings) != len(results):
        print(f"[WARN] 行数不一致 (map={len(mappings)} vs infer={len(results)})，"
              f"将优先用 infer.videos 字段抽 video_id 以避免错位")

    # 为了兼容 clip 场景（GT video_id 如 "xxx__seg0"，而实际视频文件名可能是 "xxx_0.mp4"），
    # 先基于 mapping 构建“视频源路径 -> GT video_id”的索引。
    mapping_by_source = {}
    for m in mappings:
        vid = m.get("video_id")
        src = _normalize_video_source(m.get("cos_url"))
        if isinstance(vid, str) and vid and src:
            mapping_by_source[src] = vid

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "eval_pred.jsonl")

    stats = {"success": 0, "parse_error": 0,
             "vid_from_mapping_src": 0,
             "vid_from_mapping_idx": 0,
             "vid_from_videos": 0,
             "vid_unknown": 0}
    debug_items = [] if debug else None

    with open(pred_path, "w", encoding="utf-8") as f_pred:
        for i, result in enumerate(results):
            raw_response = result.get("response", "")

            # 获取对应的 video_id（优先级）:
            # 1) infer.videos[0] 路径与 mapping.cos_url 精确对齐
            # 2) 行号对齐（仅当 infer 与 mapping 等长时，避免错位）
            # 3) 从 infer.videos 文件名兜底抽取
            # 4) unknown
            video_id = None
            videos_field = result.get("videos")
            src = _normalize_video_source(videos_field[0]) if isinstance(videos_field, list) and videos_field else None
            if src and src in mapping_by_source:
                video_id = mapping_by_source[src]
                stats["vid_from_mapping_src"] += 1
            elif len(results) == len(mappings) and i < len(mappings):
                video_id = mappings[i].get("video_id")
                if video_id:
                    stats["vid_from_mapping_idx"] += 1

            if not video_id:
                video_id = _vid_from_videos_field(videos_field)
                if video_id:
                    stats["vid_from_videos"] += 1
                else:
                    video_id = f"unknown_{i}"
                    stats["vid_unknown"] += 1

            # 解析模型输出
            prediction = parse_json_response(raw_response)

            if "segments" in prediction:
                stats["success"] += 1
                pred_segs = prediction["segments"]
            elif "raw" in prediction:
                stats["parse_error"] += 1
                pred_segs = []
            else:
                stats["success"] += 1
                pred_segs = prediction.get("segments", [])

            # 写 eval_pred
            f_pred.write(json.dumps(
                {"video_id": video_id, "segments": pred_segs},
                ensure_ascii=False
            ) + "\n")

            # debug 模式：记录完整信息
            if debug:
                debug_items.append({
                    "video_id": video_id,
                    "raw_response": raw_response,
                    "parsed_segments": pred_segs,
                    "status": "success" if "raw" not in prediction else "parse_error",
                })

    print(f"\n[INFO] 后处理完成!")
    print(f"  成功解析: {stats['success']}")
    print(f"  解析失败: {stats['parse_error']}")
    print(
        "  video_id 来源: "
        f"mapping[cos_url]={stats['vid_from_mapping_src']}  "
        f"mapping[idx]={stats['vid_from_mapping_idx']}  "
        f"videos字段={stats['vid_from_videos']}  "
        f"unknown={stats['vid_unknown']}"
    )
    print(f"  eval_pred: {pred_path}")

    # debug 模式：写完整调试文件
    if debug and debug_items:
        debug_path = os.path.join(output_dir, "debug_full_results.jsonl")
        with open(debug_path, "w", encoding="utf-8") as f:
            for item in debug_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  debug文件: {debug_path}")


def main():
    parser = argparse.ArgumentParser(description="swift infer 结果后处理")
    parser.add_argument(
        "--infer_result", type=str, required=True,
        help="swift infer 输出的 JSONL 文件"
    )
    parser.add_argument(
        "--mapping", type=str, required=True,
        help="video_id 映射文件"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出目录（eval_pred.jsonl 写到这里）"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="同时生成 debug_full_results.jsonl（含 raw_response，用于排查问题）"
    )
    # 兼容旧的 --output 参数（忽略）
    parser.add_argument("--output", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no_eval_files", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # 兼容旧调用方式：如果用了 --output 而没用 --output_dir，从 --output 推导
    if args.output and not args.output_dir:
        args.output_dir = os.path.dirname(args.output) or "."

    postprocess(args.infer_result, args.mapping, args.output_dir, args.debug)


if __name__ == "__main__":
    main()
