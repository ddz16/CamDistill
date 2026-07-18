#!/usr/bin/env python3
"""
Convert test data: annotation JSONL -> ms-swift infer standard JSONL.

Input format (annotation result):
    {"video_id": "xxx", "cos_url": "http://...", "segments": [...]}

Output format (ms-swift standard):
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "<video>..."},
                  {"role": "assistant", "content": "<ground_truth_json>"}],
     "videos": ["http://...cos_url..."]}

Notes:
    - The videos field uses cos_url directly; swift supports URL input.
    - The assistant message contains the ground truth; swift infer will use it as labels automatically.
    - A video_id mapping file is also generated for downstream result alignment.

Usage:
    python camera_movement_sft/eval/prepare_test_data.py
    python camera_movement_sft/eval/prepare_test_data.py --use_local_video --video_dir /path/to/videos1 /path/to/videos2
    python camera_movement_sft/eval/prepare_test_data.py --use_local_video --video_dir /path/to/videos1 /path/to/videos2 --auto_download
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Add the project directory to sys.path so the common module can be imported.
sys.path.insert(0, PROJECT_DIR)

# Import from the common module.
from common import SYSTEM_PROMPT, USER_PROMPT

# Test data paths.
DEFAULT_TEST_DATA = "/group/40009/dazhaodu/OurBenchmark/youtube_benchmark_subset.jsonl"
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_data_swift")


def download_video(url, save_path):
    """Download a video from a URL to the specified path. Returns True on success."""
    if not url:
        return False

    # Some test-set cos_urls are actually local absolute paths; reuse them directly.
    if os.path.isabs(url) and os.path.exists(url):
        print(f"[INFO] cos_url is a local file, using directly: {url}")
        return url

    tmp_path = save_path + ".downloading"
    try:
        print(f"[INFO] Downloading: {url}")
        print(f"       Saving to:   {save_path}")
        result = subprocess.run(
            ["wget", "-q", "--timeout=60", "--tries=2", "-O", tmp_path, url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=120,
        )
        if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            os.rename(tmp_path, save_path)
            print(f"[INFO] Download succeeded: {os.path.basename(save_path)} ({os.path.getsize(save_path) / 1024 / 1024:.1f} MB)")
            return True
        else:
            print(f"[WARN] Download failed (wget exit code {result.returncode}): {url}")
            if result.stderr:
                print(f"       stderr: {result.stderr.strip()}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
    except subprocess.TimeoutExpired:
        print(f"[WARN] Download timed out: {url}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    except Exception as e:
        print(f"[WARN] Download error: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def find_local_video(video_id, video_dirs, cos_url=None, auto_download=False):
    """Search several candidate directories for a local video file.

    Args:
        video_id: video ID.
        video_dirs: list of candidate directories.
        cos_url: COS URL used for downloading when the file is not found locally.
        auto_download: whether to automatically download missing videos to the first directory.

    Returns:
        Path if found/downloaded successfully, otherwise None.
    """
    for d in video_dirs:
        path = os.path.join(d, f"{video_id}.mp4")
        if os.path.exists(path):
            return path

    # cos_url may already be a usable local path.
    if cos_url and os.path.isabs(cos_url) and os.path.exists(cos_url):
        return cos_url

    # Not found locally; try downloading to the first directory.
    if auto_download and cos_url and video_dirs:
        download_dir = video_dirs[0]
        os.makedirs(download_dir, exist_ok=True)
        save_path = os.path.join(download_dir, f"{video_id}.mp4")
        dl_result = download_video(cos_url, save_path)
        if isinstance(dl_result, str):
            return dl_result
        if dl_result:
            return save_path

    return None


def convert_test_data(input_path, output_dir, use_local_video=False, video_dirs=None, auto_download=False):
    """Convert annotation format to the swift infer standard format.

    Args:
        video_dirs: list of local video candidate directories, ordered by priority (highest first).
        auto_download: whether to automatically download from cos_url when not found locally.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_jsonl = os.path.join(output_dir, "test_for_infer.jsonl")
    mapping_jsonl = os.path.join(output_dir, "video_id_mapping.jsonl")

    samples = []
    mappings = []
    n_local = 0
    n_downloaded = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON parse error on line {line_no}: {e}")
                continue

            video_id = item["video_id"]
            cos_url = item.get("cos_url", "")
            ground_truth = {"segments": item.get("segments", [])}

            # Video source: prefer a usable local absolute path from cos_url; otherwise search
            # candidate directories, then download on demand.
            if use_local_video and video_dirs:
                # If cos_url is already a usable local path, reuse it directly.
                if os.path.isabs(cos_url) and os.path.exists(cos_url):
                    video_source = cos_url
                    n_local += 1
                else:
                    # Try to find the video in the local candidate directories by video_id.
                    existing_path = None
                    for d in video_dirs:
                        p = os.path.join(d, f"{video_id}.mp4")
                        if os.path.exists(p):
                            existing_path = p
                            break

                    if existing_path:
                        video_source = existing_path
                        n_local += 1
                    else:
                        # Not found locally; try downloading.
                        local_path = find_local_video(video_id, video_dirs, cos_url=cos_url, auto_download=auto_download)
                        if local_path:
                            video_source = local_path
                            n_downloaded += 1
                        else:
                            searched_dirs = "\n".join(f"    - {d}" for d in video_dirs)
                            msg = (
                                f"\n[ERROR] Video file not found: {video_id}.mp4\n"
                                f"  Searched {len(video_dirs)} director(ies):\n{searched_dirs}\n"
                            )
                            if auto_download:
                                msg += f"  Download from cos_url also failed: {cos_url}\n"
                            else:
                                msg += (
                                    f"  Tip: add --auto_download to download missing videos from cos_url automatically.\n"
                                    f"  cos_url: {cos_url}\n"
                                )
                            raise FileNotFoundError(msg)
            else:
                video_source = cos_url

            # Build the ground-truth JSON as the assistant message (swift will extract it as labels).
            gt_json = json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":"))

            swift_sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<video>{USER_PROMPT}"},
                    {"role": "assistant", "content": gt_json},
                ],
                "videos": [video_source],
            }
            samples.append(swift_sample)

            # Mapping: row number -> video_id + ground_truth (used for downstream result alignment).
            mappings.append({
                "line_no": line_no,
                "video_id": video_id,
                "cos_url": cos_url,
                "ground_truth": ground_truth,
            })

    # Write the swift inference data.
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Write the mapping file.
    with open(mapping_jsonl, "w", encoding="utf-8") as f:
        for m in mappings:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[INFO] Conversion done!")
    print(f"  Input samples: {len(samples)}")
    if use_local_video and video_dirs:
        print(f"  Found locally: {n_local}, newly downloaded: {n_downloaded}")
        if auto_download and n_downloaded > 0:
            print(f"  (downloaded videos saved to: {video_dirs[0]})")
    print(f"  swift infer data: {output_jsonl}")
    print(f"  mapping file:     {mapping_jsonl}")
    return output_jsonl, mapping_jsonl


def main():
    parser = argparse.ArgumentParser(description="Convert test data format")
    parser.add_argument(
        "--input", type=str, default=DEFAULT_TEST_DATA,
        help="input test data JSONL path"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="output directory"
    )
    parser.add_argument(
        "--use_local_video", action="store_true",
        help="use local video files instead of cos_url"
    )
    parser.add_argument(
        "--video_dir", type=str, nargs="+", default=None,
        help="local video candidate directories, multiple allowed (ordered by priority, used with --use_local_video)"
    )
    parser.add_argument(
        "--auto_download", action="store_true",
        help="automatically download missing videos from cos_url to the first --video_dir directory"
    )
    args = parser.parse_args()
    convert_test_data(args.input, args.output_dir, args.use_local_video, args.video_dir, args.auto_download)


if __name__ == "__main__":
    main()
