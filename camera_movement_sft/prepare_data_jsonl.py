#!/usr/bin/env python3
"""
数据格式转换脚本: 通用 JSONL → ms-swift 视频训练格式。

输入:
    JSONL 文件，每行: {"video_id": "xxx", "cos_url": "...", "segments": [...]}

输出:
    ms-swift SFT 格式 JSONL:
    {"messages": [...], "videos": ["/path/to/video.mp4"]}
"""

import argparse
import json
import os
import sys

# 添加父目录到路径以导入 common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_movement_sft.common import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    normalize_segments,
)


def convert_one(item: dict, video_root: str) -> dict | None:
    """将单条原始标注转换为 ms-swift SFT 格式。"""
    video_id = item.get("video_id", "")
    segments = item.get("segments", [])

    if not segments:
        return None

    # 查找视频文件
    video_path = os.path.join(video_root, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return None

    # 标准化 segments
    normalized = normalize_segments(segments)
    if not normalized.get("segments"):
        return None

    # 构建 swift 格式
    assistant_content = json.dumps(normalized, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<video>{USER_PROMPT}"},
            {"role": "assistant", "content": assistant_content},
        ],
        "videos": [video_path],
    }


def main():
    parser = argparse.ArgumentParser(description="通用 JSONL 数据转换为 ms-swift 训练格式")
    parser.add_argument("--input_jsonl", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--video_root", type=str, required=True, help="视频文件根目录")
    parser.add_argument("--output_jsonl", type=str, required=True, help="输出 JSONL 文件路径")
    args = parser.parse_args()

    count = 0
    skipped = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            result = convert_one(item, args.video_root)
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
            else:
                skipped += 1

    print(f"转换完成: {count} 条成功, {skipped} 条跳过")
    print(f"输出文件: {args.output_jsonl}")


if __name__ == "__main__":
    main()
