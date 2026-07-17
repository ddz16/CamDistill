#!/usr/bin/env python3
"""
将切片（split_clips）的预测结果 merge 回原视频维度，然后与 GT 进行评测。

逻辑：
1. 从原始切片数据文件中获取每个 seg 的 scene_start 时间偏移
2. 从 eval_pred.jsonl 中读取每个 seg 的预测 segments
3. 将每个 seg 的 segments 时间加上 scene_start 偏移，还原到原视频时间轴
4. 按 orig_video_id 分组，合并所有 segments，按时间排序
5. 合并相邻且运镜类型完全相同的 segments（消除切片边界导致的碎片）
6. 输出 merged pred JSONL，然后调用评测脚本

用法:
    python merge_split_clips_eval.py \
        --split_clips_file /path/to/split_clips.jsonl \
        --pred /path/to/ckpt1/eval_pred.jsonl /path/to/ckpt2/eval_pred.jsonl \
        --gt /path/to/gt.jsonl \
        --eval_script /path/to/evaluate_camera_movement_fixed.py \
        --output /path/to/output/eval_results.json \
        --iou_thresh 0.3 0.5 0.7
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict


def load_split_clips_info(filepath):
    """加载切片数据，提取每个 seg 的 scene_start 偏移和 orig_video_id"""
    seg_info = {}  # video_id -> {scene_start, scene_end, orig_video_id, scene_idx}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            video_id = item['video_id']
            seg_info[video_id] = {
                'orig_video_id': item['orig_video_id'],
                'scene_start': item['scene_start'],
                'scene_end': item['scene_end'],
                'scene_idx': item['scene_idx'],
                'duration': item['duration'],
            }
    return seg_info


def load_eval_pred(filepath):
    """加载切片维度的预测结果"""
    preds = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            video_id = item['video_id']
            preds[video_id] = item
    return preds


def segments_are_same_movement(seg1, seg2):
    """判断两个 segment 的运镜类型是否完全相同（basic_movement + special_movement）"""
    bm1 = seg1.get('basic_movement', [])
    bm2 = seg2.get('basic_movement', [])
    sm1 = seg1.get('special_movement', [])
    sm2 = seg2.get('special_movement', [])

    # 比较 basic_movement
    if len(bm1) != len(bm2):
        return False
    for m1, m2 in zip(bm1, bm2):
        if m1.get('type') != m2.get('type'):
            return False
        if m1.get('direction') != m2.get('direction'):
            return False
        if m1.get('speed') != m2.get('speed'):
            return False

    # 比较 special_movement
    def normalize_sm(sm):
        if not sm:
            return []
        return sorted([s for s in sm if s is not None])

    if normalize_sm(sm1) != normalize_sm(sm2):
        return False

    return True


def merge_adjacent_segments(segments):
    """合并相邻且运镜类型完全相同的 segments"""
    if not segments:
        return segments

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        # 判断是否相邻（允许 0.05s 的容差，因为切片边界可能有微小误差）
        time_gap = abs(seg['start_time'] - last['end_time'])
        if time_gap <= 0.05 and segments_are_same_movement(last, seg):
            # 合并：扩展 end_time
            last['end_time'] = seg['end_time']
        else:
            merged.append(seg.copy())

    return merged


def merge_predictions(seg_info, preds):
    """将切片预测结果 merge 回原视频维度"""
    # 按 orig_video_id 分组
    video_segments = defaultdict(list)

    for video_id, pred in preds.items():
        if video_id not in seg_info:
            print(f"[WARN] 预测中的 {video_id} 在切片信息中找不到，跳过")
            continue

        info = seg_info[video_id]
        orig_video_id = info['orig_video_id']
        scene_start = info['scene_start']
        scene_idx = info['scene_idx']

        # 将每个 segment 的时间加上 scene_start 偏移
        pred_segments = pred.get('segments', [])
        offset_segments = []
        for seg in pred_segments:
            new_seg = seg.copy()
            new_seg['start_time'] = round(seg['start_time'] + scene_start, 4)
            new_seg['end_time'] = round(seg['end_time'] + scene_start, 4)
            offset_segments.append(new_seg)

        video_segments[orig_video_id].append((scene_idx, offset_segments))

    # 对每个原视频，按 scene_idx 排序后合并 segments
    merged_results = {}
    for orig_video_id, seg_list in video_segments.items():
        seg_list.sort(key=lambda x: x[0])

        all_segments = []
        for _, segments in seg_list:
            all_segments.extend(segments)

        # 按 start_time 排序
        all_segments.sort(key=lambda x: x['start_time'])

        # 合并相邻且运镜类型相同的 segments
        merged_segments = merge_adjacent_segments(all_segments)

        merged_results[orig_video_id] = {
            'video_id': orig_video_id,
            'segments': merged_segments,
        }

    return merged_results


def merge_single_pred(seg_info, pred_path, output_dir):
    """对单个 eval_pred.jsonl 执行 merge，返回 merged_pred 路径"""
    preds = load_eval_pred(pred_path)
    merged = merge_predictions(seg_info, preds)

    # 统计信息
    total_segments = sum(len(v['segments']) for v in merged.values())
    print(f"    Merge 后视频数: {len(merged)}, 总 segments: {total_segments}")

    # 输出 merged pred JSONL
    merged_pred_path = os.path.join(output_dir, "merged_pred.jsonl")
    with open(merged_pred_path, 'w', encoding='utf-8') as f:
        for video_id in sorted(merged.keys()):
            f.write(json.dumps(merged[video_id], ensure_ascii=False) + "\n")

    return merged_pred_path


def main():
    parser = argparse.ArgumentParser(description="将切片预测结果 merge 回原视频维度后评测")
    parser.add_argument("--split_clips_file", required=True,
                        help="原始切片数据文件（包含 scene_start/scene_end 信息）")
    parser.add_argument("--pred", nargs="+", required=True,
                        help="切片维度的 eval_pred.jsonl 路径（支持多个）")
    parser.add_argument("--gt", required=True,
                        help="GT 文件（原视频维度）")
    parser.add_argument("--eval_script", required=True,
                        help="评测脚本路径")
    parser.add_argument("--iou_thresh", nargs="+", type=float, default=[0.3, 0.5, 0.7],
                        help="IoU 阈值列表")
    parser.add_argument("--output", required=True,
                        help="评测结果输出路径（JSON）")
    args = parser.parse_args()

    print("=" * 60)
    print("  切片预测结果 Merge + 评测")
    print("=" * 60)

    # Step 1: 加载切片信息
    print(f"\n[Step 1] 加载切片信息: {args.split_clips_file}")
    seg_info = load_split_clips_info(args.split_clips_file)
    print(f"  切片总数: {len(seg_info)}")
    orig_videos = set(v['orig_video_id'] for v in seg_info.values())
    print(f"  原视频数: {len(orig_videos)}")

    # Step 2: 对每个 pred 文件执行 merge
    all_merged_pred_paths = []
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    for pred_path in args.pred:
        print(f"\n[Step 2] Merge 预测: {pred_path}")
        # 每个 pred 的 merged 结果放在其同级目录下
        pred_dir = os.path.dirname(pred_path)
        merged_pred_path = merge_single_pred(seg_info, pred_path, pred_dir)
        all_merged_pred_paths.append(merged_pred_path)
        print(f"    Merged 预测已保存: {merged_pred_path}")

    # Step 3: 调用评测脚本
    print(f"\n[Step 3] 调用评测脚本...")
    print(f"  GT:   {args.gt}")
    print(f"  Pred: {all_merged_pred_paths}")
    print(f"  输出: {args.output}")
    print()

    cmd = [
        sys.executable,
        args.eval_script,
        "--gt", args.gt,
        "--pred",
    ] + all_merged_pred_paths + [
        "--iou_thresh",
    ] + [str(t) for t in args.iou_thresh] + [
        "--output", args.output,
    ]

    print(f"  执行命令: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Merge + 评测完成！结果保存在: {args.output}")
    else:
        print(f"\n✗ 评测失败，返回码: {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
