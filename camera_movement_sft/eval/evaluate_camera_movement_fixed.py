#!/usr/bin/env python3
"""
时序运镜标注评测脚本（修复版 v2）
==================
评估预测的运镜标注与 GT 标注之间的准确性。

核心设计：
1. basic_movement 使用复合标签（type+direction 一体化）评测准召：
   - 有方向的运镜（Pan/Tilt/Truck/Crane/Arc/Roll）→ "Pan_left", "Tilt_up" 等
   - 无方向的运镜（Static/Dolly In/Zoom In 等）→ 直接用 type
   - 同时保留仅 type 的准召作为对比参考
2. 修复帧跳过问题：GT有segment但Pred没有时，GT标签计入FN；反之计入FP
3. direction 不再单独评测，已融合进 basic_movement 复合标签的准召中
4. 只在GT和Pred都有segment的时间范围内评估

用法:
    python evaluate_camera_movement_fixed.py --gt gt.jsonl --pred pred1.jsonl [pred2.jsonl ...]
"""

import json
import argparse
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set


# ============================================================================
# 数据加载
# ============================================================================

def load_jsonl(filepath: str) -> Dict[str, dict]:
    """加载 JSONL 文件，返回 {video_id: data} 字典"""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {filepath} 第 {line_num} 行 JSON 解析失败: {e}", file=sys.stderr)
                continue
            vid = item.get('video_id')
            if vid:
                data[vid] = item
    return data


# ============================================================================
# 辅助函数
# ============================================================================

def get_segment_at_time(segments: List[dict], t: float) -> Optional[dict]:
    """获取时刻 t 所在的 segment"""
    for seg in segments:
        if seg['start_time'] <= t < seg['end_time']:
            return seg
    # 处理最后一个 segment 的 end_time 边界
    if segments and t == segments[-1]['end_time']:
        return segments[-1]
    return None


def get_segments(video_data: dict) -> List[dict]:
    """从视频数据中提取 segments，兼容 'segments' 和 'segments_anno' 两种字段名"""
    segs = video_data.get('segments') or video_data.get('segments_anno') or []
    return segs


def normalize_null(val):
    """将字符串 'null' / 'Null' / 'NULL' / 空字符串 统一为 None"""
    if isinstance(val, str):
        stripped = val.strip()
        if stripped == '' or stripped.lower() == 'null':
            return None
    return val


# 根据标注文档，以下运镜类型需要带方向：
# Pan(left/right), Tilt(up/down), Truck(left/right), Crane(up/down),
# Arc(clockwise/counterclockwise), Roll(clockwise/counterclockwise)
# 其余类型 direction 必须为 null：
# Static, Unstable, Dolly In, Dolly Out, Zoom In, Zoom Out, Follow, Focus Shift, Free Fly
DIRECTIONAL_TYPES = {'Pan', 'Tilt', 'Truck', 'Crane', 'Arc', 'Roll'}

# ---- 标签归一化映射 ----
# 模型输出可能使用不同于标注规范的 type 名称，需要统一映射到标注规范中的标准命名。
# 标准命名参考标注文档（GT 的格式）。
#
# 已知的不一致：
# 1. Pred 使用 "Truck Left" / "Truck Right" (type自带方向, direction=None)
#    GT 使用 "Truck" + direction="left"/"right"
# 2. Pred 使用 "Pedestal Up" / "Pedestal Down" (type自带方向, direction=None)
#    GT 使用 "Crane" + direction="up"/"down" (标注规范中垂直升降统一用 Crane)
#
# 映射格式: { "原始type": ("归一化type", "归一化direction") }
# 如果某个原始type需要拆分为 type+direction，则映射到元组
TYPE_NORMALIZATION = {
    'Truck Left':     ('Truck', 'left'),
    'Truck Right':    ('Truck', 'right'),
    'Pedestal Up':    ('Crane', 'up'),
    'Pedestal Down':  ('Crane', 'down'),
}


def normalize_movement(raw_type: str, raw_direction) -> Tuple[str, Optional[str]]:
    """将原始 (type, direction) 归一化为标注规范中的标准命名。
    
    处理两种不一致：
    1. Pred 使用 "Truck Left"/"Truck Right" → 归一化为 ("Truck", "left"/"right")
    2. Pred 使用 "Pedestal Up"/"Pedestal Down" → 归一化为 ("Crane", "up"/"down")
    """
    raw_direction = normalize_null(raw_direction)
    
    if raw_type in TYPE_NORMALIZATION:
        norm_type, norm_dir = TYPE_NORMALIZATION[raw_type]
        return norm_type, norm_dir
    
    return raw_type, raw_direction


def get_basic_movement_types(seg: dict) -> Set[str]:
    """从 segment 提取 basic_movement type 集合（仅 type，不含方向），经过归一化"""
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


def get_basic_movement_labels(seg: dict) -> Set[str]:
    """从 segment 提取 basic_movement 的复合标签集合（经过归一化）。
    对于有方向的运镜类型（Pan/Tilt/Truck/Crane/Arc/Roll），
    标签为 "type_direction"（如 "Pan_left"）；
    对于无方向的类型（Static/Dolly In 等），标签就是 type 本身。
    
    如果有方向的类型缺少 direction，仍然退化为只用 type（宽容处理）。
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


def get_basic_movement_directions(seg: dict) -> List[Tuple[str, Optional[str]]]:
    """从 segment 提取归一化后的 (type, direction) 列表"""
    if seg is None:
        return []
    result = []
    for m in seg.get('basic_movement', []):
        t = normalize_null(m.get('type'))
        if t is not None:
            norm_t, norm_d = normalize_movement(t, m.get('direction'))
            result.append((norm_t, norm_d))
    return result


def get_special_movements(seg: dict) -> Set[str]:
    """从 segment 提取 special_movement 集合，过滤掉 null / None / 空字符串"""
    if seg is None:
        return set()
    result = set()
    for s in seg.get('special_movement', []):
        normalized = normalize_null(s)
        if normalized is not None:
            result.add(normalized)
    return result


def get_speed(seg: dict) -> Optional[str]:
    """从 segment 提取 speed，兼容两种格式:
    1. 旧格式: segment 顶层 speed 字段
    2. 新格式: speed 在 basic_movement 的每个 item 内部
    如果有多个 basic_movement，取出现次数最多的 speed（排除 None）
    """
    if seg is None:
        return None
    # 先尝试顶层
    top_speed = seg.get('speed')
    if top_speed is not None:
        return normalize_null(top_speed)
    # 从 basic_movement 中提取 (新格式)
    speeds = [normalize_null(m.get('speed')) for m in seg.get('basic_movement', [])
              if normalize_null(m.get('speed')) is not None]
    if not speeds:
        return None
    # 取出现最多的 speed
    return Counter(speeds).most_common(1)[0][0]


def is_slow_segment(seg: dict) -> bool:
    """判断一个 segment 是否为 'slow' segment。
    
    判定标准：该 segment 的所有 basic_movement 的 speed 都是 "slow"。
    即：segment 中每个运动都是慢速的。
    注意：zero / medium / fast 等其他速度不会被过滤。
    如果 segment 没有 basic_movement，不视为 slow（保留）。
    """
    movements = seg.get('basic_movement', [])
    if not movements:
        return False
    for m in movements:
        sp = normalize_null(m.get('speed'))
        if sp is None or sp != 'slow':
            return False
    return True


def filter_slow_segments(data: Dict[str, dict]) -> Dict[str, dict]:
    """过滤 GT 数据：去掉所有 slow segment（speed 全部为 slow 的 segment）。
    
    注意：zero / medium / fast 等其他速度的 segment 会保留。
    返回一份新的 dict，不修改原始数据。对于过滤后没有任何 segment 的视频，仍保留（空 segments）。
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
        # 兼容 'segments' 和 'segments_anno' 两种字段名
        if 'segments_anno' in item:
            new_item['segments_anno'] = new_segs
        else:
            new_item['segments'] = new_segs
        filtered[vid] = new_item
    print(f"  [filter_slow_segments] 过滤前 {total_removed + total_kept} 个 GT segments, "
          f"去掉 {total_removed} 个 slow segments, 保留 {total_kept} 个")
    return filtered


def get_video_time_range(segments: List[dict]) -> Tuple[float, float]:
    """获取segments的时间范围"""
    if not segments:
        return 0.0, 0.0
    start = min(s['start_time'] for s in segments)
    end = max(s['end_time'] for s in segments)
    return start, end


def compute_iou(seg_a: dict, seg_b: dict) -> float:
    """计算两个时间段的 IoU"""
    start = max(seg_a['start_time'], seg_b['start_time'])
    end = min(seg_a['end_time'], seg_b['end_time'])
    intersection = max(0, end - start)
    union = (seg_a['end_time'] - seg_a['start_time']) + \
            (seg_b['end_time'] - seg_b['start_time']) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


# ============================================================================
# 多标签 Precision / Recall / F1 计算器
# ============================================================================

class MultiLabelMetrics:
    """多标签 (集合级别) 的 Precision / Recall / F1 累加器"""

    def __init__(self, name: str = ""):
        self.name = name
        self.tp = 0    # 真阳: pred ∩ gt
        self.fp = 0    # 假阳: pred - gt
        self.fn = 0    # 漏报: gt - pred

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
    """单选属性的 Accuracy 累加器"""

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
# Frame-level 评测（修复版）
# ============================================================================

def evaluate_frame_level(gt_data: Dict[str, dict],
                         pred_data: Dict[str, dict],
                         sample_step: float = 0.1,
                         exclude_slow_gt: bool = False) -> dict:
    """
    逐帧采样评测（修复版 v2）。
    
    核心修改：
    1. basic_movement 使用复合标签（type+direction 一体化）评测准召：
       - 有方向的运镜（Pan/Tilt/Truck/Crane/Arc/Roll）→ "Pan_left", "Tilt_up" 等
       - 无方向的运镜（Static/Dolly In/Zoom In 等）→ 直接用 type
    2. 修复帧跳过问题：GT有segment但Pred没有 → GT标签全部算FN；反之算FP
    3. direction 不再单独评测（已融合进复合标签的准召中）
    4. exclude_slow_gt=True 时，跳过 slow GT segment 覆盖的时间帧（segment 级跳过，不跳过视频）
    """
    # basic_movement 复合标签（type+direction）的准召
    basic_mv_label = MultiLabelMetrics("basic_movement_label")
    # 仅 type 的准召（不考虑方向，用于对比参考）
    basic_mv_type = MultiLabelMetrics("basic_movement_type_only")
    special_mv = MultiLabelMetrics("special_movement")
    speed_acc = AccuracyMetrics("speed")

    # 按复合标签统计细分准召
    per_label_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    common_videos = set(gt_data.keys()) & set(pred_data.keys())
    n_frames = 0
    n_videos_evaluated = len(common_videos)
    n_videos_skipped = 0

    for vid in sorted(common_videos):
        gt_segs = get_segments(gt_data[vid])
        pred_segs = get_segments(pred_data[vid])

        # exclude_slow_gt: 过滤掉 slow 的 GT segment（segment 级跳过，不跳过视频）
        if exclude_slow_gt:
            gt_segs = [s for s in gt_segs if not is_slow_segment(s)]

        if not gt_segs and not pred_segs:
            n_videos_skipped += 1
            continue

        # 只在GT和Pred都有覆盖的时间范围内评估
        gt_start, gt_end = get_video_time_range(gt_segs) if gt_segs else (0, 0)
        pred_start, pred_end = get_video_time_range(pred_segs) if pred_segs else (0, 0)
        
        if gt_segs and pred_segs:
            # 取交集范围
            eval_start = max(gt_start, pred_start)
            eval_end = min(gt_end, pred_end)
        elif pred_segs and not gt_segs:
            # GT 过滤后为空，但 Pred 有 → 用 Pred 范围评估（Pred 全算 FP）
            eval_start = pred_start
            eval_end = pred_end
        else:
            # 都没有，已在上面跳过了
            eval_start = gt_start
            eval_end = gt_end
        
        if eval_start >= eval_end:
            # 没有重叠区域
            n_videos_skipped += 1
            continue

        # 逐帧采样（只在交集范围内）
        t = eval_start
        while t < eval_end:
            gt_seg = get_segment_at_time(gt_segs, t)
            pred_seg = get_segment_at_time(pred_segs, t)

            # 修复：不再跳过，而是正确计算 FN / FP
            # GT 有但 Pred 没有 → GT 的标签全部为 FN（漏报）
            # GT 没有但 Pred 有 → Pred 的标签全部为 FP（误报）
            # 双方都没有 → 跳过（不影响任何指标）
            if gt_seg is None and pred_seg is None:
                t = round(t + sample_step, 4)
                continue

            gt_labels = get_basic_movement_labels(gt_seg)
            pred_labels = get_basic_movement_labels(pred_seg)
            gt_types = get_basic_movement_types(gt_seg)
            pred_types = get_basic_movement_types(pred_seg)
            gt_sm = get_special_movements(gt_seg)
            pred_sm = get_special_movements(pred_seg)

            # 多标签：basic_movement 复合标签（type+direction）
            basic_mv_label.update(gt_labels, pred_labels)

            # 多标签：basic_movement 仅 type（不考虑方向，用于对比）
            basic_mv_type.update(gt_types, pred_types)

            # 按复合标签细分
            all_labels = gt_labels | pred_labels
            for label in all_labels:
                if label in gt_labels and label in pred_labels:
                    per_label_metrics[label]['tp'] += 1
                elif label in pred_labels:
                    per_label_metrics[label]['fp'] += 1
                else:
                    per_label_metrics[label]['fn'] += 1

            # 多标签：special_movement
            special_mv.update(gt_sm, pred_sm)

            # 单选：speed（只在双方都有有效 segment 且都有 speed 时评估）
            if gt_seg is not None and pred_seg is not None:
                gt_sp = get_speed(gt_seg)
                pred_sp = get_speed(pred_seg)
                if gt_sp is not None and pred_sp is not None:
                    speed_acc.update(gt_sp, pred_sp)

            n_frames += 1
            t = round(t + sample_step, 4)

    # ---- 按类别 macro: 每个复合标签单独算 P/R/F1, 再对所有类别等权平均 ----
    # (与 basic_mv_label.to_dict() 的 micro 版本对比: micro 按标签实例累加, 被高频类主导;
    #  macro 每个运镜类别等权, 反映稀有类表现)
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
        # basic_movement(type+direction): micro (原键, 按标签实例累加) + macro (按类别等权)
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
# Segment-level 评测
# ============================================================================

def match_segments_greedy(gt_segs: List[dict],
                          pred_segs: List[dict],
                          iou_thresh: float) -> List[Tuple[int, int, float]]:
    """
    贪心匹配: 按 IoU 从高到低配对 GT segment 和 Pred segment。
    每个 segment 只匹配一次。
    返回 [(gt_idx, pred_idx, iou), ...]
    """
    pairs = []
    for i, gs in enumerate(gt_segs):
        for j, ps in enumerate(pred_segs):
            iou = compute_iou(gs, ps)
            if iou >= iou_thresh:
                pairs.append((i, j, iou))

    # 按 IoU 降序排列，贪心匹配
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
    Segment-level 评测:
    1. 用 IoU >= 阈值 做贪心匹配
    2. 统计 segment 匹配的 P/R/F1
    3. 在匹配对上评估属性准确性（basic_movement 使用复合标签）
    4. exclude_slow_gt=True 时，过滤掉 slow GT segment（不参与匹配也不计入 GT 总数）
    """
    total_gt_segs = 0
    total_pred_segs = 0
    total_matched = 0
    # 耦合检测: 时间匹配(IoU>=阈值) 且 basic_movement(type+direction) 标签集完全一致 才算 TP
    total_strict_matched = 0

    # 匹配对上的属性评估
    matched_basic_mv_label = MultiLabelMetrics("seg_basic_movement_label")
    matched_basic_mv_type = MultiLabelMetrics("seg_basic_movement_type_only")
    matched_special_mv = MultiLabelMetrics("seg_special_movement")
    matched_speed_acc = AccuracyMetrics("seg_speed")

    common_videos = set(gt_data.keys()) & set(pred_data.keys())

    for vid in sorted(common_videos):
        gt_segs = get_segments(gt_data[vid])
        pred_segs = get_segments(pred_data[vid])

        # exclude_slow_gt: 过滤掉 slow 的 GT segment（segment 级跳过）
        if exclude_slow_gt:
            gt_segs = [s for s in gt_segs if not is_slow_segment(s)]

        total_gt_segs += len(gt_segs)
        total_pred_segs += len(pred_segs)

        matched = match_segments_greedy(gt_segs, pred_segs, iou_thresh)
        total_matched += len(matched)

        for gi, pi, iou in matched:
            gs = gt_segs[gi]
            ps = pred_segs[pi]

            # basic_movement 复合标签（type+direction）
            gt_labels = get_basic_movement_labels(gs)
            pred_labels = get_basic_movement_labels(ps)
            matched_basic_mv_label.update(gt_labels, pred_labels)

            # 耦合检测: 在时间匹配对基础上, 进一步要求 basic_movement 复合标签集完全一致
            if gt_labels == pred_labels:
                total_strict_matched += 1

            # basic_movement 仅 type（对比参考）
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

    # segment 匹配 P/R/F1
    seg_precision = total_matched / total_pred_segs if total_pred_segs > 0 else 0.0
    seg_recall = total_matched / total_gt_segs if total_gt_segs > 0 else 0.0
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall) \
        if (seg_precision + seg_recall) > 0 else 0.0

    # 耦合检测 (strict detection): IoU>=阈值 且 basic_movement(type+direction) 标签集完全一致
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
        # Segment Localization (Loc): 仅时间定位, 标签无关 (class-agnostic)
        'segment_localization': {
            'precision': round(seg_precision, 4),
            'recall': round(seg_recall, 4),
            'f1': round(seg_f1, 4),
        },
        # Segment Detection (Det): IoU>=阈值 且 basic_movement(type+direction) 标签集完全一致 (mAP 式)
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
# 汇总 & 打印
# ============================================================================

def print_separator(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_results(pred_name: str, frame_res: dict, seg_res: dict):
    """打印单个预测文件的完整评测结果"""
    print_separator(f"评测结果: {pred_name}")

    # ---- 数据概览 ----
    print(f"\n  共评测视频: {frame_res['n_videos']} 个（跳过 {frame_res['n_videos_skipped']} 个）")
    print(f"  Frame-level 采样帧数: {frame_res['n_frames']}")
    print(f"  GT segments: {seg_res['total_gt_segments']}, "
          f"Pred segments: {seg_res['total_pred_segments']}, "
          f"匹配数 (IoU>={seg_res['iou_threshold']}): {seg_res['total_matched']}")

    # ---- Frame-level ----
    print_separator("Frame-level 评测 (逐 0.1s 采样)")

    print(f"\n  {'属性 (micro)':<36} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
    print(f"  {'-' * 92}")
    for attr_name, display_name in [
        ('basic_movement_with_direction', 'basic_movement(type+dir)'),
        ('basic_movement_type_only', 'basic_movement(type_only)'),
        ('special_movement', 'special_movement'),
    ]:
        m = frame_res[attr_name]
        print(f"  {display_name:<36} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['tp']:>8} {m['fp']:>8} {m['fn']:>8}")

    # 按类别 macro (每个运镜类别等权平均, 反映稀有类)
    bm_macro = frame_res.get('basic_movement_with_direction_macro')
    if bm_macro:
        print(f"  {'basic_movement(type+dir) [MACRO]':<36} {bm_macro['precision']:>10.4f} {bm_macro['recall']:>10.4f} {bm_macro['f1']:>10.4f}  ({bm_macro['n_classes']} classes)")

    print(f"\n  {'属性':<36} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print(f"  {'-' * 68}")
    m = frame_res['speed']
    print(f"  {'speed':<36} {m['accuracy']:>10.4f} {m['correct']:>10} {m['total']:>10}")

    # 按复合标签细分
    if frame_res.get('per_label_basic_movement'):
        print(f"\n  [basic_movement 按复合标签细分 (type+direction)]")
        print(f"  {'Label':<36} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
        print(f"  {'-' * 92}")
        for label, m in frame_res['per_label_basic_movement'].items():
            print(f"  {label:<36} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['tp']:>8} {m['fp']:>8} {m['fn']:>8}")

    # ---- Segment-level ----
    print_separator(f"Segment-level 评测 (IoU >= {seg_res['iou_threshold']})")

    m = seg_res['segment_localization']
    print(f"\n  [Segment Localization (Loc-F1)]（仅时间定位, 与标签无关, class-agnostic）")
    print(f"  {'Precision':<16} {'Recall':<16} {'F1':<16}")
    print(f"  {m['precision']:<16.4f} {m['recall']:<16.4f} {m['f1']:<16.4f}")

    sd = seg_res.get('segment_detection')
    if sd:
        print(f"\n  [Segment Detection (Det-F1)]（IoU>=阈值 且 basic_movement(type+dir) 标签集完全一致, mAP 式）")
        print(f"  {'Precision':<16} {'Recall':<16} {'F1':<16} {'匹配数':<10}")
        print(f"  {sd['precision']:<16.4f} {sd['recall']:<16.4f} {sd['f1']:<16.4f} {sd['total_matched']:<10}")

    print(f"\n  [匹配后属性评估]")
    print(f"  {'属性':<36} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
    print(f"  {'-' * 92}")
    for attr_name, display_name in [
        ('matched_basic_movement_with_direction', 'basic_movement(type+dir)'),
        ('matched_basic_movement_type_only', 'basic_movement(type_only)'),
        ('matched_special_movement', 'special_movement'),
    ]:
        m = seg_res[attr_name]
        print(f"  {display_name:<36} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['tp']:>8} {m['fp']:>8} {m['fn']:>8}")

    print(f"\n  {'属性':<36} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print(f"  {'-' * 68}")
    m = seg_res['matched_speed']
    print(f"  {'speed':<36} {m['accuracy']:>10.4f} {m['correct']:>10} {m['total']:>10}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='时序运镜标注评测（修复版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # GT 可以直接用原始标注 JSONL（包含 video_id + segments 即可）
  python evaluate_camera_movement_fixed.py --gt testset_500.jsonl --pred pred1.jsonl
  python evaluate_camera_movement_fixed.py --gt testset_500.jsonl --pred pred1.jsonl pred2.jsonl
  python evaluate_camera_movement_fixed.py --gt testset_500.jsonl --pred pred1.jsonl --iou_thresh 0.3 0.5 0.7
  python evaluate_camera_movement_fixed.py --gt testset_500.jsonl --pred pred1.jsonl --output results.json
  python evaluate_camera_movement_fixed.py --gt testset_500.jsonl --pred pred1.jsonl --exclude_slow_gt
        """)
    parser.add_argument('--gt', required=True, help='GT JSONL 文件路径（原始标注文件即可，包含 video_id + segments）')
    parser.add_argument('--pred', nargs='+', required=True, help='预测 JSONL 文件路径 (支持多个)')
    parser.add_argument('--iou_thresh', nargs='+', type=float, default=[0.3, 0.5, 0.7],
                        help='Segment-level IoU 阈值 (默认 0.3 0.5 0.7，支持多个)')
    parser.add_argument('--sample_step', type=float, default=0.1,
                        help='Frame-level 采样步长/秒 (默认 0.1)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 JSON 文件路径 (可选)')
    parser.add_argument('--exclude_slow_gt', action='store_true', default=False,
                        help='过滤掉 GT 中 speed 全部为 slow 的 segments，'
                             '只评测非 slow 运动的部分（zero/medium/fast 等保留）')

    args = parser.parse_args()
    gt_path = args.gt

    # 加载 GT
    print(f"加载 GT: {gt_path}")
    gt_data = load_jsonl(gt_path)
    print(f"  GT 视频数: {len(gt_data)}")

    # 如果指定了 --exclude_slow_gt，提示用户
    if args.exclude_slow_gt:
        print(f"\n  [--exclude_slow_gt] 评测时将跳过 GT 中 speed 全部为 slow 的 segments（segment 级跳过，不跳过视频）")

    all_results = {}

    for pred_path in args.pred:
        print(f"\n加载 Pred: {pred_path}")
        pred_data = load_jsonl(pred_path)
        print(f"  Pred 视频数: {len(pred_data)}")

        common = set(gt_data.keys()) & set(pred_data.keys())
        only_gt = set(gt_data.keys()) - set(pred_data.keys())
        only_pred = set(pred_data.keys()) - set(gt_data.keys())
        print(f"  共同视频: {len(common)}, GT独有: {len(only_gt)}, Pred独有: {len(only_pred)}")

        if not common:
            print(f"  [WARN] 没有共同视频，跳过")
            continue

        # Frame-level 评测
        frame_res = evaluate_frame_level(gt_data, pred_data, sample_step=args.sample_step,
                                         exclude_slow_gt=args.exclude_slow_gt)

        # Segment-level 评测 (可能多个 IoU 阈值)
        seg_results = {}
        for iou_t in args.iou_thresh:
            seg_res = evaluate_segment_level(gt_data, pred_data, iou_thresh=iou_t,
                                             exclude_slow_gt=args.exclude_slow_gt)
            seg_results[f"iou_{iou_t}"] = seg_res

        # 打印结果 (对每个 IoU 阈值都打印)
        for iou_key, seg_res in seg_results.items():
            print_results(pred_path, frame_res, seg_res)

        all_results[pred_path] = {
            'frame_level': frame_res,
            'segment_level': seg_results,
        }

    # 多文件对比摘要
    if len(args.pred) > 1 and len(all_results) > 1:
        print_separator("多文件对比摘要")
        print(f"\n  {'文件':<40} {'BM+Dir-microF1':>15} {'BM+Dir-macroF1':>15} {'SM-F1':>8} {'Spd-Acc':>8} {'Loc-F1@.5':>10} {'Det-F1@.5':>10}")
        print(f"  {'-' * 108}")
        for pred_path, res in all_results.items():
            parts = pred_path.rstrip('/').split('/')
            fname = '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            fr = res['frame_level']
            # 摘要取 IoU=0.5 (若无则取第一个阈值)
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

    # 保存 JSON
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")

    print("\n评测完成!")


if __name__ == '__main__':
    main()
