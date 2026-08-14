#!/usr/bin/env python3
"""
Convert benchmark data: CamChoreo annotation JSONL -> ms-swift infer standard JSONL.

Input format (CamChoreo annotations.jsonl):
    {"video_id": "xxx",
     "local_path": "./videos/xxx.mp4",   # relative to this JSONL's directory
     "is_doubt": false,
     "segments": [...]}

Output format (ms-swift standard):
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "<video>..."},
                  {"role": "assistant", "content": "<ground_truth_json>"}],
     "videos": ["/abs/path/to/xxx.mp4"]}

Notes:
    - local_path is resolved relative to the input JSONL directory. Unzip videos.zip
      next to annotations.jsonl (so ./videos/ sits beside it) and no flags are needed.
    - The assistant message contains the ground truth; swift infer will use it as labels automatically.
    - A video_id mapping file is also generated for downstream result alignment.

Usage:
    # Zero-config: annotations.jsonl and videos/ live in the same directory.
    python camera_movement_sft/eval/prepare_test_data.py --input /path/to/CamChoreo/annotations.jsonl

    # Extra fallback video directories (searched by <video_id>.mp4):
    python camera_movement_sft/eval/prepare_test_data.py --use_local_video --video_dir /path/to/videos1 /path/to/videos2
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
# BENCHMARK_PATH env var overrides; otherwise fall back to the in-repo default.
DEFAULT_TEST_DATA = os.environ.get(
    "BENCHMARK_PATH",
    os.path.join(PROJECT_DIR, "data", "CamChoreo", "annotations.jsonl"),
)
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

    Video path resolution (in priority order) for each sample:
      1. ``local_path`` field — resolved relative to the directory containing the
         input JSONL when it is a relative path (e.g. CamChoreo's ``./videos/<id>.mp4``).
         This is the recommended layout: place ``annotations.jsonl`` and the unzipped
         ``videos/`` folder side by side and everything works with zero configuration.
      2. ``--video_dir`` candidate directories (searched by ``<video_id>.mp4``).
      3. ``cos_url`` field — a remote URL passed straight to swift, or a local abs path.

    Args:
        video_dirs: extra local video candidate directories, ordered by priority.
        auto_download: whether to download from cos_url when the file is not found locally.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Relative local_path entries are resolved against the JSONL's own directory.
    input_dir = os.path.dirname(os.path.abspath(input_path))

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
            local_path = item.get("local_path", "")
            ground_truth = {"segments": item.get("segments", [])}

            # Resolve local_path (may be relative to the JSONL directory, e.g. ./videos/<id>.mp4).
            resolved_local = ""
            if local_path:
                resolved_local = local_path if os.path.isabs(local_path) \
                    else os.path.normpath(os.path.join(input_dir, local_path))

            if resolved_local and os.path.exists(resolved_local):
                # Preferred path: the dataset's own local_path resolved next to the JSONL.
                video_source = resolved_local
                n_local += 1
            elif use_local_video and video_dirs:
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
                        found = find_local_video(video_id, video_dirs, cos_url=cos_url, auto_download=auto_download)
                        if found:
                            video_source = found
                            n_downloaded += 1
                        else:
                            searched_dirs = "\n".join(f"    - {d}" for d in video_dirs)
                            msg = (
                                f"\n[ERROR] Video file not found: {video_id}.mp4\n"
                                f"  local_path in JSONL: {local_path or '(none)'}\n"
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
            elif resolved_local:
                # local_path given but the file is missing and no other source configured.
                raise FileNotFoundError(
                    f"\n[ERROR] Video file not found for {video_id}: {resolved_local}\n"
                    f"  local_path in JSONL: {local_path}\n"
                    f"  Make sure videos.zip has been unzipped next to the JSONL "
                    f"(so that {os.path.join(input_dir, 'videos')}/ exists).\n"
                )
            else:
                # No local_path and no local dirs: fall back to cos_url (remote URL for swift).
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
                "video_source": video_source,
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
