#!/usr/bin/env python3
"""
Temporal camera-movement annotation evaluation script (fixed v2).

Evaluates the accuracy of predicted camera-movement annotations against GT annotations.

Core design:
1. basic_movement uses composite labels (type+direction combined) for precision/recall:
   - Directional movements (Pan/Tilt/Truck/Crane/Arc/Roll) -> "Pan_left", "Tilt_up", etc.
   - Non-directional movements (Static/Dolly In/Zoom In, etc.) -> type directly
   - Also keeps type-only precision/recall as a reference comparison
2. Fixes the frame-skip issue: when GT has a segment but Pred does not, GT labels count as FN;
   the reverse counts as FP.
3. direction is no longer evaluated separately; it is folded into the composite-label P/R.
4. Evaluation is performed only within the time range covered by both GT and Pred.

Usage:    python evaluate_camera_movement_fixed.py --gt gt.jsonl --pred pred1.jsonl [pred2.jsonl ...]
"""

import json
import argparse
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set


# ============================================================================
# Data loading
# ============================================================================

def load_jsonl(filepath: str) -> Dict[str, dict]:
    """Load a JSONL file and return a {video_id: data} dict."""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {filepath} line {line_num} JSON parse error: {e}", file=sys.stderr)
                continue
            vid = item.get('video_id')
            if vid:
                data[vid] = item
    return data


# ============================================================================
# Helper functions
# ============================================================================

def get_segment_at_time(segments: List[dict], t: float) -> Optional[dict]:
    """Return the segment that covers time t."""
    for seg in segments:
        if seg['start_time'] <= t < seg['end_time']:
            return seg
    # Handle the end_time boundary of the last segment.
    if segments and t == segments[-1]['end_time']:
        return segments[-1]
    return None


def get_segments(video_data: dict) -> List[dict]:
    """Extract segments from video data, supporting both 'segments' and 'segments_anno' field names."""
    segs = video_data.get('segments') or video_data.get('segments_anno') or []
    return segs


def normalize_null(val):
    """Normalise the string 'null' / 'Null' / 'NULL' / empty string to None."""
    if isinstance(val, str):
        stripped = val.strip()
        if stripped == '' or stripped.lower() == 'null':
            return None
    return val


# According to the annotation spec, the following movement types require a direction:
# Pan(left/right), Tilt(up/down), Truck(left/right), Crane(up/down),
# Arc(clockwise/counterclockwise), Roll(clockwise/counterclockwise)
# All other types must have direction=null:
# Static, Unstable, Dolly In, Dolly Out, Zoom In, Zoom Out, Follow, Focus Shift, Free Fly
DIRECTIONAL_TYPES = {'Pan', 'Tilt', 'Truck', 'Crane', 'Arc', 'Roll'}

# ---- Label normalisation map ----
# The model output may use type names that differ from the annotation spec;
# map them to the canonical names used in the spec (GT format).
#
# Known inconsistencies:
# 1. Pred uses "Truck Left" / "Truck Right" (direction embedded in type, direction=None)
#    GT uses "Truck" + direction="left"/"right"
# 2. Pred uses "Pedestal Up" / "Pedestal Down" (direction embedded in type, direction=None)
#    GT uses "Crane" + direction="up"/"down" (vertical movement is always Crane in the spec)
#
# Map format: { "raw_type": ("normalised_type", "normalised_direction") }
# If a raw type needs to be split into type+direction, map it to a tuple.
TYPE_NORMALIZATION = {
    'Truck Left':     ('Truck', 'left'),
    'Truck Right':    ('Truck', 'right'),
    'Pedestal Up':    ('Crane', 'up'),
    'Pedestal Down':  ('Crane', 'down'),
}


def normalize_movement(raw_type: str, raw_direction) -> Tuple[str, Optional[str]]:
    """Normalise a raw (type, direction) pair to the canonical annotation-spec naming.

    Handles two inconsistencies:
    1. Pred uses "Truck Left"/"Truck Right" -> normalised to ("Truck", "left"/"right")
    2. Pred uses "Pedestal Up"/"Pedestal Down" -> normalised to ("Crane", "up"/"down")
    """
    raw_direction = normalize_null(raw_direction)
    
    if raw_type in TYPE_NORMALIZATION:
        norm_type, norm_dir = TYPE_NORMALIZATION[raw_type]
        return norm_type, norm_dir
    
    return raw_type, raw_direction


def get_basic_movement_types(seg: Optional[dict]) -> Set[str]:
    """Extract the set of basic_movement types from a segment (type only, no direction), normalised."""
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


def get_basic_movement_labels(seg: Optional[dict]) -> Set[str]:
    """Extract the set of composite basic_movement labels from a segment (normalised).

    For directional movement types (Pan/Tilt/Truck/Crane/Arc/Roll), the label is
    "type_direction" (e.g. "Pan_left"); for non-directional types (Static/Dolly In, etc.)
    the label is the type itself.

    If a directional type is missing its direction, it degrades gracefully to type-only
    (lenient handling).
    """
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


def get_basic_movement_with_direction(seg: Optional[dict]) -> List[Tuple[str, Optional[str]]]:
    """Extract a normalised list of (type, direction) pairs from a segment."""
    if seg is None:
        return []
    result = []
    for m in seg.get('basic_movement', []):
        t = normalize_null(m.get('type'))
        if t is not None:
            norm_t, norm_d = normalize_movement(t, m.get('direction'))
            result.append((norm_t, norm_d))
    return result


def get_special_movements(seg: Optional[dict]) -> Set[str]:
    """Extract the set of special_movement values from a segment, filtering out null/None/empty strings."""
    if seg is None:
        return set()
    result = set()
    for s in seg.get('special_movement', []):
        normalized = normalize_null(s)
        if normalized is not None:
            result.add(normalized)
    return result


def get_speed(seg: Optional[dict]) -> Optional[str]:
    """Extract speed from a segment, supporting two formats:
    1. Legacy format: top-level speed field on the segment.
    2. New format: speed inside each basic_movement item.
    When there are multiple basic_movement items, return the most frequent speed (excluding None).
    """
    if seg is None:
        return None
    # Try the top-level field first.
    top_speed = seg.get('speed')
    if top_speed is not None:
        return normalize_null(top_speed)
    # Extract from basic_movement items (new format).
    speeds = [normalize_null(m.get('speed')) for m in seg.get('basic_movement', [])
              if normalize_null(m.get('speed')) is not None]
    if not speeds:
        return None
    # Return the most frequent speed.
    return Counter(speeds).most_common(1)[0][0]


def is_slow_segment(seg: dict) -> bool:
    """Determine whether a segment is a 'slow' segment.

    Criterion: every basic_movement in the segment has speed == "slow".
    i.e. all movements in the segment are slow.
    Note: other speeds (zero / medium / fast) are not filtered.
    A segment with no basic_movement is not considered slow (it is kept).
    """
    movements = seg.get('basic_movement', [])
    if not movements:
        return False
    for m in movements:
        sp = normalize_null(m.get('speed'))
        if sp is None or sp != 'slow':
            return False
    return True


def filter_slow_segments(gt_data: Dict[str, dict]) -> Dict[str, dict]:
    """Filter GT data: remove all slow segments (segments where every speed is 'slow').

    Note: segments with other speeds (zero / medium / fast) are kept.
    Returns a new dict; the original data is not modified. Videos that have no segments
    after filtering are still kept (with empty segments).
    """
    filtered = {}
    total_removed = 0
    total_kept = 0
    for vid, item in data.items():
        segs = get_segments(item)
        new_segs = [s for s in segs if not is_slow_segment(s)]
        total_removed += len(segs) - len(new_segs)
        total_kept += len(new_segs)
        new_item = dict(item)
        # Support both 'segments' and 'segments_anno' field names.
        if 'segments_anno' in item:
            new_item['segments_anno'] = new_segs
        else:
            new_item['segments'] = new_segs
        filtered[vid] = new_item
    print(f"  [filter_slow_segments] Before: {total_removed + total_kept} GT segments, "
          f"removed {total_removed} slow segments, kept {total_kept}")
    return filtered


def get_video_time_range(segments: List[dict]) -> Tuple[float, float]:
    """Get the time range of a list of segments."""
    if not segments:
        return 0.0, 0.0
    start = min(s['start_time'] for s in segments)
    end = max(s['end_time'] for s in segments)
    return start, end


def compute_iou(seg_a: dict, seg_b: dict) -> float:
    """Compute the IoU of two time intervals."""
    start = max(seg_a['start_time'], seg_b['start_time'])
    end = min(seg_a['end_time'], seg_b['end_time'])
    intersection = max(0, end - start)
    union = (seg_a['end_time'] - seg_a['start_time']) + \
            (seg_b['end_time'] - seg_b['start_time']) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


# ============================================================================
# Multi-label Precision / Recall / F1 accumulators
# ============================================================================

class MultiLabelMetrics:
    """Multi-label (set-level) Precision / Recall / F1 accumulator."""
    def __init__(self, name: str = ""):
        self.name = name
        self.tp = 0    # true positive:  pred ∩ gt
        self.fp = 0    # false positive:  pred - gt
        self.fn = 0    # false negative:  gt - pred

    def update(self, gt_set: Set[str], pred_set: Set[str]):
        self.tp += len(gt_set & pred_set)
        self.fp += len(pred_set - gt_set)
        self.fn += len(gt_set - pred_set)

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'precision': round(self.precision(), 4),
            'recall': round(self.recall(), 4),
            'f1': round(self.f1(), 4),
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
        }


class AccuracyMetrics:
    """Single-select attribute accuracy accumulator."""
    def __init__(self, name: str = ""):
        self.name = name
        self.correct = 0
        self.total = 0

    def update(self, gt_val, pred_val):
        self.total += 1
        if gt_val == pred_val:
            self.correct += 1

    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'accuracy': round(self.accuracy(), 4),
            'correct': self.correct,
            'total': self.total,
        }


# ============================================================================
# Frame-level evaluation (fixed)
# ============================================================================

def evaluate_frame_level(gt_data: Dict[str, dict],
                         pred_data: Dict[str, dict],
                         sample_step: float = 0.1,
                         exclude_slow_gt: bool = False) -> dict:
    """
    Frame-level evaluation by sampling (fixed v2).

    Key changes:
    1. basic_movement uses composite labels (type+direction combined) for P/R:
       - Directional movements (Pan/Tilt/Truck/Crane/Arc/Roll) -> "Pan_left", "Tilt_up", etc.
       - Non-directional movements (Static/Dolly In/Zoom In, etc.) -> type directly
    2. Fixes the frame-skip issue: GT has segment but Pred does not -> GT labels all count as FN;
       the reverse counts as FP.
    3. direction is no longer evaluated separately (folded into composite-label P/R).
    4. When exclude_slow_gt=True, skip frames covered by slow GT segments (segment-level skip,
       not video-level skip).
    """
    # basic_movement composite-label (type+direction) P/R
    basic_mv_label = MultiLabelMetrics("basic_movement_label")
    # type-only P/R (ignores direction, for reference comparison)
    basic_mv_type = MultiLabelMetrics("basic_movement_type_only")
    special_mv = MultiLabelMetrics("special_movement")
    speed_acc = AccuracyMetrics("speed")

    # per-composite-label breakdown P/R
    per_label_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    common_videos = set(gt_data.keys()) & set(pred_data.keys())
    n_frames = 0
    n_videos_evaluated = len(common_videos)
    n_videos_skipped = 0

    for vid in sorted(common_videos):
        gt_segs = get_segments(gt_data[vid])
        pred_segs = get_segments(pred_data[vid])

        # exclude_slow_gt: filter out slow GT segments (segment-level skip, not video-level)
        if exclude_slow_gt:
            gt_segs = [s for s in gt_segs if not is_slow_segment(s)]

        if not gt_segs and not pred_segs:
            n_videos_skipped += 1
            continue

        # Evaluate only within the time range covered by both GT and Pred.
        gt_start, gt_end = get_video_time_range(gt_segs) if gt_segs else (0, 0)
        pred_start, pred_end = get_video_time_range(pred_segs) if pred_segs else (0, 0)
        
        if gt_segs and pred_segs:
            # Compute the intersection range.
            eval_start = max(gt_start, pred_start)
            eval_end = min(gt_end, pred_end)
        elif pred_segs and not gt_segs:
            # GT is empty after filtering but Pred has segments -> evaluate over Pred range (all Pred = FP).
            eval_start = pred_start
            eval_end = pred_end
        else:
            # Both empty; already skipped above.
            eval_start = gt_start
            eval_end = gt_end
        
        if eval_start >= eval_end:
            # No overlapping region.
            n_videos_skipped += 1
            continue

        # Frame-by-frame sampling (within the intersection range only).
        t = eval_start
        while t < eval_end:
            gt_seg = get_segment_at_time(gt_segs, t)
            pred_seg = get_segment_at_time(pred_segs, t)

            # Fixed: no longer skip; correctly compute FN / FP.
            # GT has segment but Pred does not -> all GT labels are FN (missed).
            # GT has no segment but Pred does -> all Pred labels are FP (false alarm).
            # Both have no segment -> skip (does not affect any metric).
            if gt_seg is None and pred_seg is None:
                t = round(t + sample_step, 4)
                continue

            gt_labels = get_basic_movement_labels(gt_seg)
            pred_labels = get_basic_movement_labels(pred_seg)
            gt_types = get_basic_movement_types(gt_seg)
            pred_types = get_basic_movement_types(pred_seg)
            gt_sm = get_special_movements(gt_seg)
            pred_sm = get_special_movements(pred_seg)

            # Multi-label: basic_movement composite label (type+direction).
            basic_mv_label.update(gt_labels, pred_labels)

            # Multi-label: basic_movement type only (ignores direction, for comparison).
            basic_mv_type.update(gt_types, pred_types)

            # Per-composite-label breakdown.
            all_labels = gt_labels | pred_labels
            for label in all_labels:
                if label in gt_labels and label in pred_labels:
                    per_label_metrics[label]['tp'] += 1
                elif label in pred_labels:
                    per_label_metrics[label]['fp'] += 1
                else:
                    per_label_metrics[label]['fn'] += 1

            # Multi-label: special_movement.
            special_mv.update(gt_sm, pred_sm)

            # Single-select: speed (evaluated only when both sides have a valid segment and speed).
            if gt_seg is not None and pred_seg is not None:
                gt_sp = get_speed(gt_seg)
                pred_sp = get_speed(pred_seg)
                if gt_sp is not None and pred_sp is not None:
                    speed_acc.update(gt_sp, pred_sp)

            n_frames += 1
            t = round(t + sample_step, 4)

    # ---- Per-class macro: compute P/R/F1 per composite label, then average equally across classes ----
    # (Contrast with the micro version in basic_mv_label.to_dict(): micro accumulates by label instance
    #  and is dominated by high-frequency classes; macro weights each movement class equally,
    #  reflecting performance on rare classes.)
    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    _mp, _mr, _mf = [], [], []
    for _v in per_label_metrics.values():
        _p, _r, _f = _prf(_v['tp'], _v['fp'], _v['fn'])
        _mp.append(_p); _mr.append(_r); _mf.append(_f)
    _ncls = len(per_label_metrics)
    basic_mv_label_macro = {
        'precision': round(sum(_mp) / _ncls, 4) if _ncls else 0.0,
        'recall': round(sum(_mr) / _ncls, 4) if _ncls else 0.0,
        'f1': round(sum(_mf) / _ncls, 4) if _ncls else 0.0,
        'n_classes': _ncls,
    }

    return {
        'n_videos': n_videos_evaluated,
        'n_videos_skipped': n_videos_skipped,
        'n_frames': n_frames,
        # basic_movement(type+direction): micro (original key, accumulated by label instance) + macro (equal weight per class)
        'basic_movement_with_direction': basic_mv_label.to_dict(),
        'basic_movement_with_direction_macro': basic_mv_label_macro,
        'basic_movement_type_only': basic_mv_type.to_dict(),
        'special_movement': special_mv.to_dict(),
        'speed': speed_acc.to_dict(),
        'per_label_basic_movement': {
            k: {
                'precision': round(v['tp'] / (v['tp'] + v['fp']), 4) if (v['tp'] + v['fp']) > 0 else 0.0,
                'recall': round(v['tp'] / (v['tp'] + v['fn']), 4) if (v['tp'] + v['fn']) > 0 else 0.0,
                'f1': round(2 * v['tp'] / (2 * v['tp'] + v['fp'] + v['fn']), 4) if (2 * v['tp'] + v['fp'] + v['fn']) > 0 else 0.0,
                'tp': v['tp'],
                'fp': v['fp'],
                'fn': v['fn'],
            }
            for k, v in sorted(per_label_metrics.items())
        }
    }


# ============================================================================
# Segment-level evaluation
# ============================================================================

def match_segments_greedy(gt_segs: List[dict],
                          pred_segs: List[dict],
                          iou_thresh: float) -> List[Tuple[int, int, float]]:
    """
    Greedy matching: pair GT segments and Pred segments in descending IoU order.
    Each segment is matched at most once.
    Returns [(gt_idx, pred_idx, iou), ...]
    """
    pairs = []
    for i, gs in enumerate(gt_segs):
        for j, ps in enumerate(pred_segs):
            iou = compute_iou(gs, ps)
            if iou >= iou_thresh:
                pairs.append((i, j, iou))

    # Sort by IoU descending; greedy matching.
    pairs.sort(key=lambda x: -x[2])
    matched = []
    used_gt, used_pred = set(), set()
    for gi, pi, iou in pairs:
        if gi not in used_gt and pi not in used_pred:
            matched.append((gi, pi, iou))
            used_gt.add(gi)
            used_pred.add(pi)

    return matched


def evaluate_segment_level(gt_data: Dict[str, dict],
                           pred_data: Dict[str, dict],
                           iou_thresh: float = 0.5,
                           exclude_slow_gt: bool = False) -> dict:
    """
    Segment-level evaluation:
    1. Greedy matching with IoU >= threshold.
    2. Compute segment-matching P/R/F1.
    3. Evaluate attribute accuracy on matched pairs (basic_movement uses composite labels).
    4. When exclude_slow_gt=True, filter out slow GT segments (excluded from matching and GT count).
    """
    total_gt_segs = 0
    total_pred_segs = 0
    total_matched = 0
    # Coupled detection: temporal match (IoU>=threshold) AND basic_movement(type+direction) label sets identical -> TP
    total_strict_matched = 0

    # Attribute evaluation on matched pairs.
    matched_basic_mv_label = MultiLabelMetrics("seg_basic_movement_label")
    matched_basic_mv_type = MultiLabelMetrics("seg_basic_movement_type_only")
    matched_special_mv = MultiLabelMetrics("seg_special_movement")
    matched_speed_acc = AccuracyMetrics("seg_speed")

    common_videos = set(gt_data.keys()) & set(pred_data.keys())

    for vid in sorted(common_videos):
        gt_segs = get_segments(gt_data[vid])
        pred_segs = get_segments(pred_data[vid])

        # exclude_slow_gt: filter out slow GT segments (segment-level skip).
        if exclude_slow_gt:
            gt_segs = [s for s in gt_segs if not is_slow_segment(s)]

        total_gt_segs += len(gt_segs)
        total_pred_segs += len(pred_segs)

        matched = match_segments_greedy(gt_segs, pred_segs, iou_thresh)
        total_matched += len(matched)

        for gi, pi, iou in matched:
            gs = gt_segs[gi]
            ps = pred_segs[pi]

            # basic_movement composite label (type+direction).
            gt_labels = get_basic_movement_labels(gs)
            pred_labels = get_basic_movement_labels(ps)
            matched_basic_mv_label.update(gt_labels, pred_labels)

            # Coupled detection: on top of temporal match, also require identical basic_movement composite label sets.
            if gt_labels == pred_labels:
                total_strict_matched += 1

            # basic_movement type only (reference comparison).
            gt_types = get_basic_movement_types(gs)
            pred_types = get_basic_movement_types(ps)
            matched_basic_mv_type.update(gt_types, pred_types)

            # special_movement
            gt_sm = get_special_movements(gs)
            pred_sm = get_special_movements(ps)
            matched_special_mv.update(gt_sm, pred_sm)

            # speed
            gt_sp = get_speed(gs)
            pred_sp = get_speed(ps)
            if gt_sp is not None and pred_sp is not None:
                matched_speed_acc.update(gt_sp, pred_sp)

    # Segment-matching P/R/F1.
    seg_precision = total_matched / total_pred_segs if total_pred_segs > 0 else 0.0
    seg_recall = total_matched / total_gt_segs if total_gt_segs > 0 else 0.0
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall) \
        if (seg_precision + seg_recall) > 0 else 0.0

    # Coupled detection (strict): IoU>=threshold AND basic_movement(type+direction) label sets identical.
    strict_precision = total_strict_matched / total_pred_segs if total_pred_segs > 0 else 0.0
    strict_recall = total_strict_matched / total_gt_segs if total_gt_segs > 0 else 0.0
    strict_f1 = 2 * strict_precision * strict_recall / (strict_precision + strict_recall) \
        if (strict_precision + strict_recall) > 0 else 0.0

    return {
        'iou_threshold': iou_thresh,
        'n_videos': len(common_videos),
        'total_gt_segments': total_gt_segs,
        'total_pred_segments': total_pred_segs,
        'total_matched': total_matched,
        # Segment Localization (Loc): temporal localisation only, label-agnostic (class-agnostic).
        'segment_localization': {
            'precision': round(seg_precision, 4),
            'recall': round(seg_recall, 4),
            'f1': round(seg_f1, 4),
        },
        # Segment Detection (Det): IoU>=threshold AND basic_movement(type+direction) label sets identical (mAP-style).
        'segment_detection': {
            'precision': round(strict_precision, 4),
            'recall': round(strict_recall, 4),
            'f1': round(strict_f1, 4),
            'total_matched': total_strict_matched,
            'criterion': 'IoU>=thresh AND basic_movement(type+direction) label-set exact match',
        },
        'matched_basic_movement_with_direction': matched_basic_mv_label.to_dict(),
        'matched_basic_movement_type_only': matched_basic_mv_type.to_dict(),
        'matched_special_movement': matched_special_mv.to_dict(),
        'matched_speed': matched_speed_acc.to_dict(),
    }


# ============================================================================
# Aggregation & printing
# ============================================================================

def print_separator(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_results(pred_name: str, frame_res: dict, seg_res: dict):
    """Print the full evaluation results for a single prediction file."""
    print_separator(f"Evaluation results: {pred_name}")

    # ---- Data overview ----
    print(f"\n  Videos evaluated: {frame_res['n_videos']} (skipped: {frame_res['n_videos_skipped']})")
    print(f"  Frame-level sampled frames: {frame_res['n_frames']}")
    print(f"  GT segments: {seg_res['total_gt_segments']}, "
          f"Pred segments: {seg_res['total_pred_segments']}, "
          f"matched (IoU>={seg_res['iou_threshold']}): {seg_res['total_matched']}")

    # ---- Frame-level ----
    print_separator("Frame-level evaluation (sampled every 0.1s)")

    print(f"\n  {'Attribute (micro)':<36} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
    print(f"  {'-' * 92}")
    for attr_name, display_name in [
        ('basic_movement_with_direction', 'basic_movement(type+dir)'),
        ('basic_movement_type_only', 'basic_movement(type_only)'),
        ('special_movement', 'special_movement'),
    ]:
        m = frame_res[attr_name]
        print(f"  {display_name:<36} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['tp']:>8} {m['fp']:>8} {m['fn']:>8}")

    # Per-class macro (equal weight per movement class, reflects rare-class performance).
    bm_macro = frame_res.get('basic_movement_with_direction_macro')
    if bm_macro:
        print(f"  {'basic_movement(type+dir) [MACRO]':<36} {bm_macro['precision']:>10.4f} {bm_macro['recall']:>10.4f} {bm_macro['f1']:>10.4f}  ({bm_macro['n_classes']} classes)")

    print(f"\n  {'Attribute':<36} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print(f"  {'-' * 68}")
    m = frame_res['speed']
    print(f"  {'speed':<36} {m['accuracy']:>10.4f} {m['correct']:>10} {m['total']:>10}")

    # Per-composite-label breakdown.
    if frame_res.get('per_label_basic_movement'):
        print(f"\n  [basic_movement per-composite-label breakdown (type+direction)]")
        print(f"  {'Label':<36} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
        print(f"  {'-' * 92}")
        for label, m in frame_res['per_label_basic_movement'].items():
            print(f"  {label:<36} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['tp']:>8} {m['fp']:>8} {m['fn']:>8}")

    # ---- Segment-level ----
    print_separator(f"Segment-level evaluation (IoU >= {seg_res['iou_threshold']})")

    m = seg_res['segment_localization']
    print(f"\n  [Segment Localization (Loc-F1)] (temporal localisation only, label-agnostic, class-agnostic)")
    print(f"  {'Precision':<16} {'Recall':<16} {'F1':<16}")
    print(f"  {m['precision']:<16.4f} {m['recall']:<16.4f} {m['f1']:<16.4f}")

    sd = seg_res.get('segment_detection')
    if sd:
        print(f"\n  [Segment Detection (Det-F1)] (IoU>=threshold AND basic_movement(type+dir) label sets identical, mAP-style)")
        print(f"  {'Precision':<16} {'Recall':<16} {'F1':<16} {'Matched':<10}")
        print(f"  {sd['precision']:<16.4f} {sd['recall']:<16.4f} {sd['f1']:<16.4f} {sd['total_matched']:<10}")

    print(f"\n  [Attribute evaluation on matched pairs]")
    print(f"  {'Attribute':<36} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
    print(f"  {'-' * 92}")
    for attr_name, display_name in [
        ('matched_basic_movement_with_direction', 'basic_movement(type+dir)'),
        ('matched_basic_movement_type_only', 'basic_movement(type_only)'),
        ('matched_special_movement', 'special_movement'),
    ]:
        m = seg_res[attr_name]
        print(f"  {display_name:<36} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['tp']:>8} {m['fp']:>8} {m['fn']:>8}")

    print(f"\n  {'Attribute':<36} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print(f"  {'-' * 68}")
    m = seg_res['matched_speed']
    print(f"  {'speed':<36} {m['accuracy']:>10.4f} {m['correct']:>10} {m['total']:>10}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Temporal camera-movement annotation evaluation (fixed)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GT can be the raw annotation JSONL directly (needs video_id + segments)
  python evaluate_camera_movement_fixed.py --gt testset_500.jsonl --pred pred1.jsonl
  python evaluate_camera_movement_fixed.py --gt testset_500.jsonl --pred pred1.jsonl pred2.jsonl
  python evaluate_camera_movement_fixed.py --gt testset_500.jsonl --pred pred1.jsonl --iou_thresh 0.3 0.5 0.7
  python evaluate_camera_movement_fixed.py --gt testset_500.jsonl --pred pred1.jsonl --output results.json
  python evaluate_camera_movement_fixed.py --gt testset_500.jsonl --pred pred1.jsonl --exclude_slow_gt
        """)
    parser.add_argument('--gt', required=True, help='GT JSONL file path (raw annotation file is fine; needs video_id + segments)')
    parser.add_argument('--pred', nargs='+', required=True, help='prediction JSONL file path(s) (multiple supported)')
    parser.add_argument('--iou_thresh', nargs='+', type=float, default=[0.3, 0.5, 0.7],
                        help='Segment-level IoU threshold(s) (default 0.3 0.5 0.7, multiple supported)')
    parser.add_argument('--sample_step', type=float, default=0.1,
                        help='Frame-level sampling step in seconds (default 0.1)')
    parser.add_argument('--output', type=str, default=None,
                        help='output JSON file path (optional)')
    parser.add_argument('--exclude_slow_gt', action='store_true', default=False,
                        help='filter out GT segments where every speed is slow; '
                             'evaluate only non-slow movements (zero/medium/fast are kept)')

    args = parser.parse_args()
    gt_path = args.gt

    # Load GT.
    print(f"Loading GT: {gt_path}")
    gt_data = load_jsonl(gt_path)
    print(f"  GT videos: {len(gt_data)}")

    # Notify the user if --exclude_slow_gt was specified.
    if args.exclude_slow_gt:
        print(f"\n  [--exclude_slow_gt] Segments where every speed is slow will be skipped during evaluation (segment-level skip, not video-level).")

    all_results = {}

    for pred_path in args.pred:
        print(f"\nLoading Pred: {pred_path}")
        pred_data = load_jsonl(pred_path)
        print(f"  Pred videos: {len(pred_data)}")

        common = set(gt_data.keys()) & set(pred_data.keys())
        only_gt = set(gt_data.keys()) - set(pred_data.keys())
        only_pred = set(pred_data.keys()) - set(gt_data.keys())
        print(f"  Common videos: {len(common)}, GT-only: {len(only_gt)}, Pred-only: {len(only_pred)}")

        if not common:
            print(f"  [WARN] No common videos, skipping")
            continue

        # Frame-level evaluation.
        frame_res = evaluate_frame_level(gt_data, pred_data, sample_step=args.sample_step,
                                         exclude_slow_gt=args.exclude_slow_gt)

        # Segment-level evaluation (possibly multiple IoU thresholds).
        seg_results = {}
        for iou_t in args.iou_thresh:
            seg_res = evaluate_segment_level(gt_data, pred_data, iou_thresh=iou_t,
                                             exclude_slow_gt=args.exclude_slow_gt)
            seg_results[f"iou_{iou_t}"] = seg_res

        # Print results (for each IoU threshold).
        for iou_key, seg_res in seg_results.items():
            print_results(pred_path, frame_res, seg_res)

        all_results[pred_path] = {
            'frame_level': frame_res,
            'segment_level': seg_results,
        }

    # Multi-file comparison summary.
    if len(args.pred) > 1 and len(all_results) > 1:
        print_separator("Multi-file comparison summary")
        print(f"\n  {'File':<40} {'BM+Dir-microF1':>15} {'BM+Dir-macroF1':>15} {'SM-F1':>8} {'Spd-Acc':>8} {'Loc-F1@.5':>10} {'Det-F1@.5':>10}")
        print(f"  {'-' * 108}")
        for pred_path, res in all_results.items():
            parts = pred_path.rstrip('/').split('/')
            fname = '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            fr = res['frame_level']
            # Summary uses IoU=0.5 (falls back to the first threshold if 0.5 is not available).
            seg = res['segment_level'].get('iou_0.5') or list(res['segment_level'].values())[0]
            bm_macro = fr.get('basic_movement_with_direction_macro', {}).get('f1', 0.0)
            loc = seg.get('segment_localization', {}).get('f1', 0.0)
            det = seg.get('segment_detection', {}).get('f1', 0.0)
            print(f"  {fname:<40} "
                  f"{fr['basic_movement_with_direction']['f1']:>15.4f} "
                  f"{bm_macro:>15.4f} "
                  f"{fr['special_movement']['f1']:>8.4f} "
                  f"{fr['speed']['accuracy']:>8.4f} "
                  f"{loc:>10.4f} "
                  f"{det:>10.4f}")

    # Save JSON.
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
