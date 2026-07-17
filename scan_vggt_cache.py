#!/usr/bin/env python3
"""扫描 VGGT cache, 找出含 NaN/Inf 的 .pt 文件"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import torch


def check_one(path: str) -> tuple:
    """检查单个 .pt 文件. 返回 (path, status, info)"""
    try:
        size = os.path.getsize(path)
        if size < 100:
            return path, "too_small", f"{size}B"

        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            feats = data.get("camera_features")
            if feats is None:
                return path, "no_camera_features", "missing key"
        else:
            feats = data

        if not isinstance(feats, torch.Tensor):
            return path, "not_tensor", str(type(feats))

        if feats.numel() == 0:
            return path, "empty", str(feats.shape)

        # 检查 NaN/Inf
        nan_count = torch.isnan(feats).sum().item()
        inf_count = torch.isinf(feats).sum().item()
        if nan_count > 0 or inf_count > 0:
            return path, "has_nan_inf", f"shape={list(feats.shape)}, nan={nan_count}, inf={inf_count}"

        # 也检查 pose_enc (如果有)
        if isinstance(data, dict) and "pose_enc" in data:
            pose = data["pose_enc"]
            if not torch.isfinite(pose).all():
                return path, "pose_has_nan_inf", f"pose shape={list(pose.shape)}"

        return path, "ok", f"shape={list(feats.shape)}"

    except Exception as e:
        return path, "load_error", str(e)[:100]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--report", type=str, default=None,
                        help="保存坏文件列表的报告路径")
    parser.add_argument("--delete_bad", action="store_true",
                        help="直接删除坏文件 (谨慎使用)")
    parser.add_argument("--input_jsonl", type=str, default=None,
                        help="只检查这个 jsonl 中的 video_id 对应的 cache (而不是整个 cache_dir)")
    args = parser.parse_args()

    if args.report is None:
        args.report = f"{args.cache_dir.rstrip('/')}_bad_report.txt"

    # 确定要检查的文件列表
    if args.input_jsonl:
        import json
        print(f"读取 {args.input_jsonl}, 提取 video_ids...")
        video_ids = set()
        with open(args.input_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for v in data.get("videos", []):
                    if isinstance(v, str):
                        vid = os.path.splitext(os.path.basename(v))[0]
                        video_ids.add(vid)
        print(f"  共 {len(video_ids)} 个 unique video_id")
        files = []
        missing = 0
        for vid in sorted(video_ids):
            p = os.path.join(args.cache_dir, f"{vid}.pt")
            if os.path.exists(p):
                files.append(p)
            else:
                missing += 1
        print(f"  cache 中存在: {len(files)}, 缺失: {missing}")
    else:
        print(f"扫描目录: {args.cache_dir}")
        files = sorted(glob(os.path.join(args.cache_dir, "*.pt")))
        print(f"  共 {len(files)} 个 .pt 文件")

    print(f"  并行 workers: {args.workers}")

    bad = []  # (path, status, info)
    ok_count = 0

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(files), desc="checking")
    except ImportError:
        progress = None

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(check_one, p) for p in files]
        for fut in as_completed(futures):
            path, status, info = fut.result()
            if status == "ok":
                ok_count += 1
            else:
                bad.append((path, status, info))
                # 实时打印前 50 个坏文件
                if len(bad) <= 50:
                    print(f"  {status}: {os.path.basename(path)} ({info})")
            if progress:
                progress.update(1)

    if progress:
        progress.close()

    # 统计
    from collections import Counter
    status_counter = Counter(b[1] for b in bad)
    print(f"\n=== 扫描结果 ===")
    print(f"  ✅ 正常: {ok_count}")
    print(f"  ❌ 异常: {len(bad)}")
    for st, cnt in status_counter.most_common():
        print(f"    {st}: {cnt}")

    # 写报告
    if bad:
        with open(args.report, "w") as f:
            f.write(f"# VGGT cache 扫描报告\n")
            f.write(f"# 目录: {args.cache_dir}\n")
            f.write(f"# 总数: {len(files)}, 异常: {len(bad)}\n\n")
            f.write("path\tstatus\tinfo\n")
            for path, status, info in sorted(bad):
                f.write(f"{path}\t{status}\t{info}\n")
        print(f"\n报告: {args.report}")

        if args.delete_bad:
            print(f"\n删除 {len(bad)} 个坏文件...")
            for path, _, _ in bad:
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"  删除失败 {path}: {e}")
            print("删除完成 (重新跑提取脚本会重新生成)")
    else:
        print("\n✅ 所有 cache 文件都正常!")


if __name__ == "__main__":
    main()
