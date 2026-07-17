#!/usr/bin/env python3
"""统计训练集中视频的真实时长分布 (用 ffprobe)"""

import argparse
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_video_duration(video_path: str) -> float:
    """用 ffprobe 获取视频时长 (秒). 失败返回 -1"""
    if not os.path.exists(video_path):
        return -1.0
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        return -1.0
    except Exception:
        return -1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str,
                        default="/group/40009/dazhaodu/ms-swift/camera_movement_sft/train_data/camera_movement_train_diverse_50k.jsonl")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    print(f"读取 {args.input_jsonl}...")
    video_paths = []
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            for v in data.get("videos", []):
                if isinstance(v, str):
                    video_paths.append(v)
    print(f"  共 {len(video_paths)} 个视频")
    print(f"  并行 workers: {args.workers}")

    durations = []
    failed = 0
    try:
        from tqdm import tqdm
        progress = tqdm(total=len(video_paths), desc="probing")
    except ImportError:
        progress = None

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(get_video_duration, p) for p in video_paths]
        for fut in as_completed(futures):
            d = fut.result()
            if d > 0:
                durations.append(d)
            else:
                failed += 1
            if progress:
                progress.update(1)
    if progress: progress.close()

    print(f"\n=== 统计结果 ===")
    print(f"  成功: {len(durations)}, 失败: {failed}")

    if not durations:
        return

    import numpy as np
    arr = np.array(durations)
    print(f"\n时长统计 (秒):")
    print(f"  min:    {arr.min():.2f}")
    print(f"  max:    {arr.max():.2f}")
    print(f"  mean:   {arr.mean():.2f}")
    print(f"  median: {np.median(arr):.2f}")
    print(f"  std:    {arr.std():.2f}")
    print(f"\n分位数:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(arr, p):.2f}s")

    print(f"\n时长区间分布:")
    bins = [0, 1, 2, 3, 5, 10, 15, 20, 30, 60, 120, 300, 1e9]
    bin_labels = ['<1s', '1-2s', '2-3s', '3-5s', '5-10s', '10-15s', '15-20s',
                  '20-30s', '30-60s', '1-2m', '2-5m', '>5m']
    counts, _ = np.histogram(arr, bins=bins)
    total = len(arr)
    for label, cnt in zip(bin_labels, counts):
        pct = cnt / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {label:>8}: {cnt:>6} ({pct:5.2f}%) {bar}")


if __name__ == "__main__":
    main()
