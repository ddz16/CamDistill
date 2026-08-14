#!/usr/bin/env python3
"""
Convert training data: CamChoreo-style annotation JSONL -> ms-swift SFT JSONL.

The training set uses the **same schema as our benchmark** (CamChoreo's
annotations.jsonl). Each input line is:

    {"video_id": "xxx",
     "local_path": "./videos/xxx.mp4",   # relative to this JSONL's directory
     "segments": [...]}

and is converted to the ms-swift chat format expected by `swift sft`:

    {"messages": [{"role": "system",    "content": "<SYSTEM_PROMPT>"},
                  {"role": "user",      "content": "<video><USER_PROMPT>"},
                  {"role": "assistant", "content": "<ground_truth_json>"}],
     "videos": ["/abs/path/to/xxx.mp4"]}

Our internal training set is not publicly released, but you can train on your own
data simply by preparing it in the CamChoreo annotation schema above and running
this script; then point `DATASET_PATH` at the produced JSONL.

Usage:
    # local_path is resolved relative to the input JSONL directory
    python camera_movement_sft/prepare_train_data.py \
        --input  /path/to/your_annotations.jsonl \
        --output camera_movement_sft/train_data/train_swift.jsonl

    # then train:
    DATASET_PATH=camera_movement_sft/train_data/train_swift.jsonl \
        bash camera_movement_sft/train.sh qwen3vl-4b
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from common import SYSTEM_PROMPT, USER_PROMPT, normalize_segments


def convert(input_path, output_path, normalize=True):
    """Convert an annotation JSONL into ms-swift SFT format."""
    input_dir = os.path.dirname(os.path.abspath(input_path))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n_in = 0
    n_out = 0
    n_missing_video = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON parse error on line {line_no}: {e}")
                continue

            video_id = item.get("video_id", "")
            local_path = item.get("local_path", "")
            if not local_path:
                print(f"[WARN] line {line_no} ({video_id}) has no local_path, skipping")
                continue

            # Resolve local_path relative to the JSONL directory (e.g. ./videos/<id>.mp4).
            video_path = local_path if os.path.isabs(local_path) \
                else os.path.normpath(os.path.join(input_dir, local_path))
            if not os.path.exists(video_path):
                n_missing_video += 1

            segments = item.get("segments", [])
            ground_truth = normalize_segments(segments) if normalize else {"segments": segments}
            gt_json = json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":"))

            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<video>{USER_PROMPT}"},
                    {"role": "assistant", "content": gt_json},
                ],
                "videos": [video_path],
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[INFO] Conversion done!")
    print(f"  Input lines:   {n_in}")
    print(f"  Output samples:{n_out}")
    if n_missing_video:
        print(f"  [WARN] {n_missing_video} video file(s) not found on disk "
              f"(paths kept; make sure videos/ is unzipped next to the JSONL)")
    print(f"  Output file:   {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert annotation JSONL to ms-swift SFT format")
    parser.add_argument("--input", type=str, required=True,
                        help="input annotation JSONL (CamChoreo schema)")
    parser.add_argument("--output", type=str, required=True,
                        help="output ms-swift SFT JSONL")
    parser.add_argument("--no_normalize", action="store_true",
                        help="keep segments verbatim instead of normalising to the closed-set format")
    args = parser.parse_args()
    convert(args.input, args.output, normalize=not args.no_normalize)


if __name__ == "__main__":
    main()
