#!/usr/bin/env python3
"""
Data-format conversion: generic JSONL -> ms-swift video training format.

Input:
    JSONL file, one entry per line: {"video_id": "xxx", "cos_url": "...", "segments": [...]}

Output:
    ms-swift SFT JSONL:
    {"messages": [...], "videos": ["/path/to/video.mp4"]}
"""

import argparse
import json
import os
import sys

# Add the parent directory to sys.path so `common` can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_movement_sft.common import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    normalize_segments,
)


def convert_one(item: dict, video_root: str) -> dict | None:
    """Convert a single raw annotation into the ms-swift SFT format."""
    video_id = item.get("video_id", "")
    segments = item.get("segments", [])

    if not segments:
        return None

    # Look up the video file.
    video_path = os.path.join(video_root, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return None

    # Normalize segments.
    normalized = normalize_segments(segments)
    if not normalized.get("segments"):
        return None

    # Build the swift-format entry.
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
    parser = argparse.ArgumentParser(description="Convert generic JSONL data into ms-swift training format")
    parser.add_argument("--input_jsonl", type=str, required=True, help="input JSONL file path")
    parser.add_argument("--video_root", type=str, required=True, help="root directory of video files")
    parser.add_argument("--output_jsonl", type=str, required=True, help="output JSONL file path")
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

    print(f"Conversion done: {count} succeeded, {skipped} skipped")
    print(f"Output file: {args.output_jsonl}")


if __name__ == "__main__":
    main()
