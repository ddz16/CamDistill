#!/usr/bin/env python3
"""
Data Download Script for CameraBench Videos
Downloads videos from HuggingFace repository for camera motion understanding.

Default download path: /apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/CameraBench
Repo:                 syCen/Videos4CameraBench (~3000 mp4 files)

Usage:
    python camera_movement_sft/eval/data_download.py
    python camera_movement_sft/eval/data_download.py --output_dir /custom/path
    python camera_movement_sft/eval/data_download.py --repo_id syCen/CameraBench
"""

import argparse
import os
import time
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Installing huggingface_hub...")
    os.system("pip install huggingface_hub")
    from huggingface_hub import snapshot_download


DEFAULT_OUTPUT_DIR = "/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/CameraBench"
DEFAULT_REPO_ID = "syCen/Videos4CameraBench"


def main():
    parser = argparse.ArgumentParser(description="Download CameraBench videos from HuggingFace.")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID,
                        help=f"HuggingFace dataset repo (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Local download directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Concurrent download workers (default: 4)")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max retry attempts on failure (default: 5)")
    args = parser.parse_args()

    repo_id = args.repo_id
    output_dir = args.output_dir

    # Default to the HF mirror; keep whatever the user set if already configured.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_TOKEN", "***REMOVED***")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print("=" * 60)
    print("Downloading CameraBench videos...")
    print(f"  Repository : {repo_id}")
    print(f"  Local dir  : {output_dir}")
    print(f"  HF endpoint: {os.environ.get('HF_ENDPOINT')}")
    print(f"  Workers    : {args.max_workers}")
    print("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    max_retries = args.max_retries
    base_delay = 2

    for attempt in range(max_retries):
        try:
            print(f"\nDownload attempt {attempt + 1}/{max_retries}...")
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)

            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=output_dir,
                tqdm_class=None,
                max_workers=args.max_workers,
                token=os.environ.get("HF_TOKEN"),
                resume_download=True,
            )

            print(f"\n✅ Videos downloaded successfully to: {output_dir}")
            print("\nNext steps:")
            print(f"  ls {output_dir}")
            print(f"  export CAMERABENCH_VIDEO_DIR={output_dir}")
            print("  bash camera_movement_sft/eval/benchmarks/run_benchmarks.sh <ckpt>")
            return

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                print("This might be due to rate limiting. Retrying with delay...")
            else:
                print("\nAll download attempts failed.")
                print("Troubleshooting:")
                print("  1. Check HF_ENDPOINT / HF_TOKEN env vars")
                print("  2. Try lowering --max_workers to 1")
                print("  3. Re-run; snapshot_download supports resume")
                print("  4. Manual: huggingface-cli download --repo-type dataset \\")
                print(f"             {repo_id} --local-dir {output_dir}")


if __name__ == "__main__":
    main()
