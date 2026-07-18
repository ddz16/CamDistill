#!/usr/bin/env python3
"""
Macro-average evaluation script: compute precision/recall weighted equally per video.

Unlike the existing micro-average (which accumulates TP/FP/FN directly at the frame level),
this script first computes P/R/F1 for each video individually, then averages across all videos.

This gives every video the same weight regardless of its length.

Usage:
    # Use the default GT (read from debug_full_results.jsonl)
    python eval_macro_average.py

    # Specify an external GT file
    python eval_macro_average.py --gt /path/to/gt.jsonl

    # Compare two GT files
    python eval_macro_average.py --gt /path/to/gt1.jsonl --gt2 /path/to/gt2.jsonl
"""

import json
import sys
import os
import argparse
from collections import defaultdict
from typing import Dict, Set, Optional, List, Tuple, Counter as CounterType
from collections import Counter

# ==========================================================================
# Reuse helper functions from evaluate_camera_movement_fixed.py
# ==========================================================================

DIRECTIONAL_TYPES = {'Pan', 'Tilt', 'Truck', 'Crane', 'Arc', 'Roll'}

TYPE_NORMALIZATION = {
    'Truck Left':     ('Truck', 'left'),
    'Truck Right':    ('Truck', 'right'),
    'Pedestal Up':    ('Crane', 'up'),
    'Pedestal Down':  ('Crane', 'down'),
}

def normalize_null(val):
    if isinstance(val, str):
        stripped = val.strip()
        if stripped == '' or stripped.lower() == 'null':
            return None
    return val

def normalize_movement(raw_type, raw_direction):
    raw_direction = normalize_null(raw_direction)
    if raw_type in TYPE_NORMALIZATION:
        return TYPE_NORMALIZATION[raw_type]
    return raw_type, raw_direction

def get_segment_at_time(segments, t):
    for seg in segments:
        if seg['start_time'] <= t < seg['end_time']:
            return seg
    if segments and t == segments[-1]['end_time']:
        return segments[-1]
    return None

def get_segments(video_data):
    return video_data.get('segments') or video_data.get('segments_anno') or []

def get_basic_movement_labels(seg):
    if seg is None:
        return set()
    labels = set()
    for m in seg.get('basic_movement', []):
        t = normalize_null(m.get('type'))
        if t is None:
            continue
        d = m.get('direction')
        norm_t, norm_d = normalize_movement(t, d)
        if norm_t in DIRECTIONAL_TYPES and norm_d is not None:
            labels.add(f"{norm_t}_{norm_d}")
        else:
            labels.add(norm_t)
    return labels

def get_basic_movement_types(seg):
    if seg is None:
        return set()
    types = set()
    for m in seg.get('basic_movement', []):
        t = normalize_null(m.get('type'))
        if t is None:
            continue
        norm_t, _ = normalize_movement(t, m.get('direction'))
        types.add(norm_t)
    return types

def get_special_movements(seg):
    if seg is None:
        return set()
    result = set()
    for s in seg.get('special_movement', []):
        normalized = normalize_null(s)
        if normalized is not None:
            result.add(normalized)
    return result

def get_speed(seg):
    if seg is None:
        return None
    top_speed = seg.get('speed')
    if top_speed is not None:
        return normalize_null(top_speed)
    speeds = [normalize_null(m.get('speed')) for m in seg.get('basic_movement', [])
              if normalize_null(m.get('speed')) is not None]
    if not speeds:
        return None
    return Counter(speeds).most_common(1)[0][0]

def get_video_time_range(segments):
    if not segments:
        return 0.0, 0.0
    start = min(s['start_time'] for s in segments)
    end = max(s['end_time'] for s in segments)
    return start, end

def load_jsonl(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            vid = item.get('video_id')
            if vid:
                data[vid] = item
    return data


# ==========================================================================
# Macro-average: compute P/R/F1 per video independently, then average across all videos.
# ==========================================================================

def evaluate_single_video_frame_level(gt_video, pred_video, sample_step=0.1):
    """Run frame-level evaluation on a single video and return its TP/FP/FN statistics."""
    gt_segs = get_segments(gt_video)
    pred_segs = get_segments(pred_video)

    if not gt_segs:
        return None

    gt_start, gt_end = get_video_time_range(gt_segs)
    pred_start, pred_end = get_video_time_range(pred_segs) if pred_segs else (0, 0)

    eval_start = max(gt_start, pred_start)
    eval_end = min(gt_end, pred_end)

    if eval_start >= eval_end:
        return None

    # Statistics.
    bm_label = {'tp': 0, 'fp': 0, 'fn': 0}  # basic_movement with direction
    bm_type  = {'tp': 0, 'fp': 0, 'fn': 0}  # basic_movement type only
    sm       = {'tp': 0, 'fp': 0, 'fn': 0}  # special_movement
    speed_correct = 0
    speed_total = 0
    per_label = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    n_frames = 0

    t = eval_start
    while t < eval_end:
        gt_seg = get_segment_at_time(gt_segs, t)
        pred_seg = get_segment_at_time(pred_segs, t)

        if gt_seg is None and pred_seg is None:
            t = round(t + sample_step, 4)
            continue

        gt_labels = get_basic_movement_labels(gt_seg)
        pred_labels = get_basic_movement_labels(pred_seg)
        gt_types = get_basic_movement_types(gt_seg)
        pred_types = get_basic_movement_types(pred_seg)
        gt_sm = get_special_movements(gt_seg)
        pred_sm = get_special_movements(pred_seg)

        # basic_movement with direction
        bm_label['tp'] += len(gt_labels & pred_labels)
        bm_label['fp'] += len(pred_labels - gt_labels)
        bm_label['fn'] += len(gt_labels - pred_labels)

        # basic_movement type only
        bm_type['tp'] += len(gt_types & pred_types)
        bm_type['fp'] += len(pred_types - gt_types)
        bm_type['fn'] += len(gt_types - pred_types)

        # special_movement
        sm['tp'] += len(gt_sm & pred_sm)
        sm['fp'] += len(pred_sm - gt_sm)
        sm['fn'] += len(gt_sm - pred_sm)

        # per label
        all_labels = gt_labels | pred_labels
        for label in all_labels:
            if label in gt_labels and label in pred_labels:
                per_label[label]['tp'] += 1
            elif label in pred_labels:
                per_label[label]['fp'] += 1
            else:
                per_label[label]['fn'] += 1

        # speed
        if gt_seg is not None and pred_seg is not None:
            gt_sp = get_speed(gt_seg)
            pred_sp = get_speed(pred_seg)
            if gt_sp is not None and pred_sp is not None:
                speed_total += 1
                if gt_sp == pred_sp:
                    speed_correct += 1

        n_frames += 1
        t = round(t + sample_step, 4)

    return {
        'n_frames': n_frames,
        'bm_label': bm_label,
        'bm_type': bm_type,
        'sm': sm,
        'speed_correct': speed_correct,
        'speed_total': speed_total,
        'per_label': dict(per_label),
    }


def calc_prf(tp, fp, fn):
    """Compute precision, recall, and F1."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def macro_average_evaluation(gt_data, pred_data, sample_step=0.1):
    """
    Macro-average evaluation: compute P/R/F1 per video, then average.
    Also returns micro-average results for comparison.
    """
    common_videos = sorted(set(gt_data.keys()) & set(pred_data.keys()))

    # Collect P/R/F1 per video.
    video_results = []
    all_labels_set = set()

    for vid in common_videos:
        res = evaluate_single_video_frame_level(gt_data[vid], pred_data[vid], sample_step)
        if res is None:
            continue
        video_results.append((vid, res))
        all_labels_set.update(res['per_label'].keys())

    n_videos = len(video_results)

    # ---- Macro-average: compute P/R/F1 per video, then average ----
    macro = {
        'bm_label': {'precision': [], 'recall': [], 'f1': []},
        'bm_type': {'precision': [], 'recall': [], 'f1': []},
        'sm': {'precision': [], 'recall': [], 'f1': []},
        'speed_acc': [],
    }
    macro_per_label = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})

    # ---- Micro-average: accumulate TP/FP/FN globally ----
    micro = {
        'bm_label': {'tp': 0, 'fp': 0, 'fn': 0},
        'bm_type': {'tp': 0, 'fp': 0, 'fn': 0},
        'sm': {'tp': 0, 'fp': 0, 'fn': 0},
        'speed_correct': 0,
        'speed_total': 0,
    }
    micro_per_label = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for vid, res in video_results:
        # Micro accumulation.
        for key in ['bm_label', 'bm_type', 'sm']:
            micro[key]['tp'] += res[key]['tp']
            micro[key]['fp'] += res[key]['fp']
            micro[key]['fn'] += res[key]['fn']
        micro['speed_correct'] += res['speed_correct']
        micro['speed_total'] += res['speed_total']

        for label, counts in res['per_label'].items():
            micro_per_label[label]['tp'] += counts['tp']
            micro_per_label[label]['fp'] += counts['fp']
            micro_per_label[label]['fn'] += counts['fn']

        # Macro: P/R/F1 per video.
        for key in ['bm_label', 'bm_type', 'sm']:
            p, r, f1 = calc_prf(res[key]['tp'], res[key]['fp'], res[key]['fn'])
            macro[key]['precision'].append(p)
            macro[key]['recall'].append(r)
            macro[key]['f1'].append(f1)

        if res['speed_total'] > 0:
            macro['speed_acc'].append(res['speed_correct'] / res['speed_total'])

        # Macro per label: only include a video when it has GT or Pred for that label.
        for label in all_labels_set:
            if label in res['per_label']:
                c = res['per_label'][label]
                p, r, f1 = calc_prf(c['tp'], c['fp'], c['fn'])
                macro_per_label[label]['precision'].append(p)
                macro_per_label[label]['recall'].append(r)
                macro_per_label[label]['f1'].append(f1)

    # Aggregate macro.
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    result = {
        'n_videos': n_videos,
        'macro': {},
        'micro': {},
    }

    for key, display in [('bm_label', 'basic_movement_with_direction'),
                          ('bm_type', 'basic_movement_type_only'),
                          ('sm', 'special_movement')]:
        result['macro'][display] = {
            'precision': round(avg(macro[key]['precision']), 4),
            'recall': round(avg(macro[key]['recall']), 4),
            'f1': round(avg(macro[key]['f1']), 4),
        }
        p, r, f1 = calc_prf(micro[key]['tp'], micro[key]['fp'], micro[key]['fn'])
        result['micro'][display] = {
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f1, 4),
        }

    result['macro']['speed'] = {'accuracy': round(avg(macro['speed_acc']), 4)}
    result['micro']['speed'] = {
        'accuracy': round(micro['speed_correct'] / micro['speed_total'], 4) if micro['speed_total'] > 0 else 0.0
    }

    # Per-label
    result['macro']['per_label'] = {}
    result['micro']['per_label'] = {}
    for label in sorted(all_labels_set):
        if label in macro_per_label:
            result['macro']['per_label'][label] = {
                'precision': round(avg(macro_per_label[label]['precision']), 4),
                'recall': round(avg(macro_per_label[label]['recall']), 4),
                'f1': round(avg(macro_per_label[label]['f1']), 4),
                'n_videos': len(macro_per_label[label]['f1']),
            }
        mp = micro_per_label.get(label, {'tp': 0, 'fp': 0, 'fn': 0})
        p, r, f1 = calc_prf(mp['tp'], mp['fp'], mp['fn'])
        result['micro']['per_label'][label] = {
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f1, 4),
        }

    return result


# ==========================================================================
# GT loading functions
# ==========================================================================

# Default GT file path.
DEFAULT_GT_V1 = "/group/40009/dazhaodu/OurBenchmark/youtube_benchmark_subset.jsonl"
DEFAULT_GT_V2 = "/group/40009/dazhaodu/OurBenchmark/youtube_benchmark_subset.jsonl"


def load_gt_from_external_file(gt_path: str) -> Dict[str, dict]:
    """Load GT from an external file (format: video_id, segments)."""
    gt_data = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            vid = item.get('video_id')
            if vid:
            # GT file has a segments field directly.
                gt_data[vid] = {'segments': item.get('segments', [])}
    return gt_data


def load_gt_from_debug_file(debug_path: str) -> Dict[str, dict]:
    """Load GT from debug_full_results.jsonl (legacy format compatibility)."""
    gt_data = {}
    with open(debug_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            vid = item.get('video_id')
            gt = item.get('ground_truth', {})
            if vid and gt:
                gt_data[vid] = gt
    return gt_data


# ==========================================================================
# Main
# ==========================================================================

def run_evaluation(gt_data: Dict[str, dict], gt_name: str, checkpoints: List[str], base_dir: str):
    """Run evaluation and print results."""
    print(f"\n{'='*120}")
    print(f"  GT: {gt_name}")
    print(f"  Videos: {len(gt_data)}")
    print(f"{'='*120}\n")

    all_results = {}
    for ckpt in checkpoints:
        pred_path = os.path.join(base_dir, ckpt, "eval_pred.jsonl")
        pred_data = load_jsonl(pred_path)
        result = macro_average_evaluation(gt_data, pred_data)
        all_results[ckpt] = result

    # 1. Overall comparison table (Macro vs Micro).
    print("=" * 120)
    print("  Per-checkpoint overall metrics (Macro-average vs Micro-average)")
    print("=" * 120)

    header = f"{'Checkpoint':<20} | {'Mode':<6} | {'BM+Dir P':>9} {'BM+Dir R':>9} {'BM+Dir F1':>10} | {'BM Type P':>10} {'BM Type R':>10} {'BM Type F1':>11} | {'SM F1':>7} | {'Speed Acc':>10}"
    print(header)
    print("-" * 120)

    for ckpt in checkpoints:
        res = all_results[ckpt]
        for mode, label in [('macro', 'Macro'), ('micro', 'Micro')]:
            bm = res[mode]['basic_movement_with_direction']
            bt = res[mode]['basic_movement_type_only']
            sm = res[mode]['special_movement']
            sp = res[mode]['speed']
            print(f"{ckpt:<20} | {label:<6} | {bm['precision']:>9.4f} {bm['recall']:>9.4f} {bm['f1']:>10.4f} | {bt['precision']:>10.4f} {bt['recall']:>10.4f} {bt['f1']:>11.4f} | {sm['f1']:>7.4f} | {sp['accuracy']:>10.4f}")
        print("-" * 120)

    # 2. Macro-only ranking table.
    print("\n")
    print("=" * 100)
    print("  Macro-average (equal weight per video) ranking")
    print("=" * 100)
    print(f"{'Checkpoint':<20} | {'BM+Dir P':>9} {'BM+Dir R':>9} {'BM+Dir F1':>10} | {'BM Type F1':>11} | {'SM F1':>7} | {'Speed Acc':>10}")
    print("-" * 100)

    sorted_ckpts = sorted(checkpoints,
                          key=lambda c: all_results[c]['macro']['basic_movement_with_direction']['f1'],
                          reverse=True)
    for ckpt in sorted_ckpts:
        res = all_results[ckpt]['macro']
        bm = res['basic_movement_with_direction']
        bt = res['basic_movement_type_only']
        sm = res['special_movement']
        sp = res['speed']
        print(f"{ckpt:<20} | {bm['precision']:>9.4f} {bm['recall']:>9.4f} {bm['f1']:>10.4f} | {bt['f1']:>11.4f} | {sm['f1']:>7.4f} | {sp['accuracy']:>10.4f}")
    
    best_macro = sorted_ckpts[0]
    print(f"\n  >>> Best Macro-average: {best_macro} (BM+Dir F1 = {all_results[best_macro]['macro']['basic_movement_with_direction']['f1']:.4f})")
    return all_results, best_macro


def main():
    parser = argparse.ArgumentParser(description="Macro-average evaluation script")
    parser.add_argument("--gt", type=str, default=DEFAULT_GT_V1,
                        help="GT file path (default: 500-sample test set)")
    parser.add_argument("--gt2", type=str, default=None,
                        help="second GT file path (for comparison)")
    parser.add_argument("--compare-default", action="store_true",
                        help="compare the two default GT files (v1 and v2)")
    parser.add_argument("--base-dir", type=str,
                        default="/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/eval/output_testset_300_balanced",
                        help="checkpoint output directory")
    parser.add_argument("--checkpoints", type=str, nargs="+",
                        default=["checkpoint-2500", "checkpoint-3000", "checkpoint-3500",
                                 "checkpoint-4000", "checkpoint-4500", "checkpoint-4536"],
                        help="list of checkpoints to evaluate")
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    CHECKPOINTS = args.checkpoints

    # Determine which GT files to evaluate.
    gt_files = []
    if args.compare_default:
        gt_files = [
            (DEFAULT_GT_V1, "500-sample test set (v1)"),
            (DEFAULT_GT_V2, "500-sample test set (v2)"),
        ]
    elif args.gt2:
        gt_files = [
            (args.gt, os.path.basename(args.gt)),
            (args.gt2, os.path.basename(args.gt2)),
        ]
    else:
        gt_files = [(args.gt, os.path.basename(args.gt))]

    # Run evaluation.
    all_gt_results = {}
    for gt_path, gt_name in gt_files:
        print(f"\nLoading GT: {gt_path}")
        gt_data = load_gt_from_external_file(gt_path)
        results, best = run_evaluation(gt_data, gt_name, CHECKPOINTS, BASE_DIR)
        all_gt_results[gt_name] = {'results': results, 'best': best}

    # If there are multiple GTs, print a comparison summary.
    if len(gt_files) > 1:
        print("\n" + "=" * 120)
        print("  GT comparison summary (best checkpoint per GT)")
        print("=" * 120)
        print(f"{'GT file':<45} | {'Best Ckpt':<18} | {'BM+Dir F1':>10} | {'BM Type F1':>11} | {'SM F1':>7} | {'Speed Acc':>10}")        print("-" * 120)
        for gt_name, data in all_gt_results.items():
            best_ckpt = data['best']
            res = data['results'][best_ckpt]['macro']
            bm = res['basic_movement_with_direction']
            bt = res['basic_movement_type_only']
            sm = res['special_movement']
            sp = res['speed']
            print(f"{gt_name:<45} | {best_ckpt:<18} | {bm['f1']:>10.4f} | {bt['f1']:>11.4f} | {sm['f1']:>7.4f} | {sp['accuracy']:>10.4f}")


if __name__ == '__main__':
    main()
