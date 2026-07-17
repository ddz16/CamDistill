#!/usr/bin/env python3
"""
预过滤训练集中的坏视频。

逻辑:
  - 用 decord 实际解码每个视频
  - 出错或总帧数 < 2 的视频, 从 jsonl 中删除
  - 输出过滤后的新 jsonl

用法:
  python filter_bad_videos.py \
      --input camera_movement_sft/train_data/camera_movement_train_human_all_292k_fixed.jsonl \
      --output camera_movement_sft/train_data/camera_movement_train_human_all_292k_filtered.jsonl \
      --num_workers 32

  # 只检查 (不输出新文件), 报告坏视频数量:
  python filter_bad_videos.py --input <jsonl> --check_only
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEO_DIRS = [
    "/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos",
    "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_21k_video_shot_cameraMove/videos",
    "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_realfilm_30w_260401_deduped_final/videos",
]


def find_video_path(video_id: str) -> str:
    """在多个目录中查找视频"""
    for vdir in VIDEO_DIRS:
        path = os.path.join(vdir, f"{video_id}.mp4")
        if os.path.exists(path):
            return path
    return None


def check_video(video_path: str) -> tuple:
    """
    用 decord 检查视频是否可读。
    返回 (is_good: bool, reason: str)
    """
    if not os.path.exists(video_path):
        return False, "file_not_found"

    # 文件太小 (< 5KB) 直接判定为坏
    try:
        size = os.path.getsize(video_path)
        if size < 5000:
            return False, f"too_small_{size}B"
    except OSError:
        return False, "stat_failed"

    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        n_frames = len(vr)

        if n_frames < 2:
            return False, f"only_{n_frames}_frames"

        # 严格检查: 读首/中/末 3 帧 (能覆盖大部分中间损坏的情况)
        # 如果只有 1-2 帧, 上面已 return; 这里至少有 2 帧
        test_indices = [0, n_frames // 2, n_frames - 1]
        test_indices = sorted(set(min(i, n_frames - 1) for i in test_indices))
        _ = vr.get_batch(test_indices).asnumpy()
        return True, "ok"
    except Exception as e:
        return False, f"decord_error: {str(e)[:80]}"


def check_one_sample(args):
    """检查 jsonl 中一条样本的所有视频"""
    line_num, line = args
    line = line.strip()
    if not line:
        return line_num, None, None, "empty_line"

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return line_num, None, None, "bad_json"

    videos = data.get("videos", [])
    if not videos:
        # 纯文本样本, 保留
        return line_num, line, None, "no_video"

    # 检查所有 videos
    for v in videos:
        if not isinstance(v, str):
            continue

        # 视频路径: 如果是相对路径或仅文件名, 用 video_dirs 查找
        if os.path.isabs(v) and os.path.exists(v):
            video_path = v
        else:
            vid = os.path.splitext(os.path.basename(v))[0]
            video_path = find_video_path(vid)
            if video_path is None:
                return line_num, None, vid, "video_file_not_found"

        is_good, reason = check_video(video_path)
        if not is_good:
            vid = os.path.splitext(os.path.basename(video_path))[0]
            return line_num, None, vid, reason

    # 全部 ok
    return line_num, line, None, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--check_only", action="store_true",
                        help="只统计, 不输出新文件")
    parser.add_argument("--report", type=str, default=None,
                        help="坏视频报告输出路径 (默认: <output>.bad_report.txt)")
    args = parser.parse_args()

    if not args.check_only and args.output is None:
        # 自动生成输出路径: xxx.jsonl -> xxx_filtered.jsonl
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_filtered{ext}"

    if args.report is None and args.output:
        args.report = f"{args.output}.bad_report.txt"

    # 读取 jsonl
    print(f"读取 {args.input}...")
    with open(args.input, "r") as f:
        lines = f.readlines()
    print(f"  总行数: {len(lines)}")

    # 并行检查
    print(f"\n开始检查 (并行 {args.num_workers})...")
    tasks = list(enumerate(lines))

    good_lines = [None] * len(lines)
    bad_records = []  # (line_num, video_id, reason)
    no_video_count = 0
    ok_count = 0

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(tasks), desc="checking")
    except ImportError:
        progress = None

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(check_one_sample, t): t[0] for t in tasks}
        for future in as_completed(futures):
            line_num, valid_line, bad_vid, reason = future.result()

            if valid_line is not None:
                good_lines[line_num] = valid_line
                if reason == "no_video":
                    no_video_count += 1
                else:
                    ok_count += 1
            else:
                bad_records.append((line_num, bad_vid, reason))

            if progress:
                progress.update(1)

    if progress:
        progress.close()

    # 统计坏视频原因
    from collections import Counter
    reasons = Counter(r[2] for r in bad_records)

    print(f"\n=== 结果 ===")
    print(f"  ✅ 视频正常: {ok_count}")
    print(f"  📝 纯文本(无视频): {no_video_count}")
    print(f"  ❌ 损坏样本: {len(bad_records)}")
    print(f"\n坏视频原因分布:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")

    # 写报告
    if args.report:
        with open(args.report, "w") as f:
            f.write(f"# 坏视频报告\n")
            f.write(f"# 输入: {args.input}\n")
            f.write(f"# 总样本: {len(lines)}, 坏样本: {len(bad_records)}\n\n")
            f.write(f"line_num\tvideo_id\treason\n")
            for line_num, vid, reason in sorted(bad_records):
                f.write(f"{line_num}\t{vid}\t{reason}\n")
        print(f"\n报告: {args.report}")

    # 写过滤后的 jsonl
    if not args.check_only:
        with open(args.output, "w") as f:
            for line in good_lines:
                if line is not None:
                    if not line.endswith("\n"):
                        line += "\n"
                    f.write(line)
        kept = sum(1 for l in good_lines if l is not None)
        print(f"\n过滤后输出: {args.output} ({kept} 条)")


if __name__ == "__main__":
    main()
