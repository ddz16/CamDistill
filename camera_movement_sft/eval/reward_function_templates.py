"""
GRPO Reward Function Templates for Camera Movement Evaluation
Based on the camera_movement_sft evaluation logic
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np


# ============================================================================
# Helper: Label extraction (same logic as evaluate_camera_movement_fixed.py)
# ============================================================================

DIRECTIONAL_TYPES = {'Pan', 'Tilt', 'Truck', 'Crane', 'Arc', 'Roll'}

def get_composite_labels(segment: Optional[Dict]) -> Set[str]:
    """Extract composite labels (type_direction) from a segment"""
    if segment is None:
        return set()
    
    labels = set()
    for movement in segment.get('basic_movement', []):
        movement_type = movement.get('type')
        direction = movement.get('direction')
        
        if movement_type is None:
            continue
        
        # Build composite label
        if movement_type in DIRECTIONAL_TYPES and direction:
            labels.add(f"{movement_type}_{direction}")
        else:
            labels.add(movement_type)
    
    return labels


# ============================================================================
# OPTION 1: Basic F1-based Reward (Simplest)
# ============================================================================

def reward_basic_f1(
    pred_segment: Optional[Dict],
    gt_segment: Optional[Dict]
) -> float:
    """
    Simplest reward: F1 score of composite labels at this time point
    
    Use case: Straightforward, gradient-friendly
    Range: [0, 1]
    """
    pred_labels = get_composite_labels(pred_segment)
    gt_labels = get_composite_labels(gt_segment)
    
    # Handle empty cases
    if not gt_labels and not pred_labels:
        return 1.0  # Both silent = correct
    if not gt_labels or not pred_labels:
        return 0.0  # One has content, other silent = wrong
    
    # Compute F1
    tp = len(gt_labels & pred_labels)
    fp = len(pred_labels - gt_labels)
    fn = len(gt_labels - pred_labels)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1


# ============================================================================
# OPTION 2: Per-Label Binary Reward (For selective credit)
# ============================================================================

def reward_per_label(
    pred_segment: Optional[Dict],
    gt_segment: Optional[Dict],
    direction_penalty: float = 0.3
) -> float:
    """
    Decompose by label: each label gets 0 (FN/FP) or 1 (TP).
    Wrong direction gets partial credit (e.g., 0.3)
    
    Use case: Fine-grained learning signal
    Range: [0, 1]
    """
    pred_labels = get_composite_labels(pred_segment)
    gt_labels = get_composite_labels(gt_segment)
    
    if not gt_labels and not pred_labels:
        return 1.0
    
    all_labels = gt_labels | pred_labels
    rewards = []
    
    for label in all_labels:
        if label in gt_labels and label in pred_labels:
            rewards.append(1.0)  # TP
        elif label in gt_labels:
            rewards.append(0.0)  # FN
        else:
            # FP
            rewards.append(0.0)
    
    return np.mean(rewards) if rewards else 0.0


# ============================================================================
# OPTION 3: Hierarchical Reward (Type + Direction Awareness)
# ============================================================================

def reward_hierarchical(
    pred_segment: Optional[Dict],
    gt_segment: Optional[Dict],
    type_weight: float = 0.7,
    direction_weight: float = 0.3
) -> float:
    """
    Two-level grading:
    - Level 1: Type correctness (0.7 weight)
    - Level 2: Direction correctness (0.3 weight, only if type is right)
    
    Use case: Encourage type learning first, then direction refinement
    Range: [0, 1]
    """
    pred_labels = get_composite_labels(pred_segment)
    gt_labels = get_composite_labels(gt_segment)
    
    if not gt_labels and not pred_labels:
        return 1.0
    if not gt_labels or not pred_labels:
        return 0.0
    
    type_rewards = []
    direction_rewards = []
    
    all_labels = gt_labels | pred_labels
    
    for gt_label in gt_labels:
        # Parse GT label
        if '_' in gt_label:
            gt_type, gt_dir = gt_label.split('_', 1)
        else:
            gt_type, gt_dir = gt_label, None
        
        # Check if type is correct
        type_found = False
        for pred_label in pred_labels:
            if '_' in pred_label:
                pred_type, pred_dir = pred_label.split('_', 1)
            else:
                pred_type, pred_dir = pred_label, None
            
            if pred_type == gt_type:
                type_found = True
                # Type is correct
                type_rewards.append(1.0)
                
                # Check direction (only if GT has direction requirement)
                if gt_dir is not None:
                    if pred_dir == gt_dir:
                        direction_rewards.append(1.0)
                    else:
                        direction_rewards.append(0.0)  # Wrong direction
                break
        
        if not type_found:
            type_rewards.append(0.0)
            direction_rewards.append(0.0)
    
    # False positives
    for pred_label in pred_labels:
        if '_' in pred_label:
            pred_type, _ = pred_label.split('_', 1)
        else:
            pred_type = pred_label
        
        if not any(
            (l.split('_')[0] if '_' in l else l) == pred_type 
            for l in gt_labels
        ):
            type_rewards.append(0.0)  # FP type
    
    type_score = np.mean(type_rewards) if type_rewards else 0.0
    direction_score = np.mean(direction_rewards) if direction_rewards else 0.0
    
    return type_weight * type_score + direction_weight * direction_score


# ============================================================================
# OPTION 4: Segment-Level IoU Reward (Temporal Grounding)
# ============================================================================

def compute_iou(start_pred: float, end_pred: float, 
                start_gt: float, end_gt: float) -> float:
    """Compute Intersection over Union of two time intervals"""
    intersection = max(0, min(end_pred, end_gt) - max(start_pred, start_gt))
    union = (end_pred - start_pred) + (end_gt - start_gt) - intersection
    return intersection / union if union > 0 else 0.0


def reward_segment_matching(
    pred_segment: Optional[Dict],
    gt_segment: Optional[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Reward segment-level temporal alignment.
    High IoU = close match = higher reward
    
    Use case: Prevents arbitrary segmentation boundaries
    Range: [0, 1]
    """
    if pred_segment is None or gt_segment is None:
        return 0.0
    
    iou = compute_iou(
        pred_segment['start_time'],
        pred_segment['end_time'],
        gt_segment['start_time'],
        gt_segment['end_time']
    )
    
    if iou >= iou_threshold:
        # Normalize IoU to [0, 1] range starting from threshold
        return (iou - iou_threshold) / (1.0 - iou_threshold)
    else:
        return 0.0


# ============================================================================
# OPTION 5: Composite Reward (Multi-Signal)
# ============================================================================

def reward_composite(
    pred_segments: List[Optional[Dict]],
    gt_segments: List[Optional[Dict]],
    weights: Dict[str, float] = None
) -> float:
    """
    Aggregate multiple reward signals with configurable weights.
    
    Default weights:
    - frame_level_f1: 0.50 (primary)
    - segment_matching: 0.15 (temporal)
    - special_movement: 0.15 (auxiliary)
    - speed_accuracy: 0.10 (secondary)
    - type_only_f1: 0.10 (robustness)
    
    Use case: Balanced, comprehensive training signal
    Range: [0, 1]
    """
    if weights is None:
        weights = {
            'frame_level_f1': 0.50,
            'segment_matching': 0.15,
            'special_movement': 0.15,
            'speed_accuracy': 0.10,
            'type_only_f1': 0.10,
        }
    
    rewards = {}
    
    # 1. Frame-level F1 (composite labels)
    frame_f1_scores = []
    for pred_seg, gt_seg in zip(pred_segments, gt_segments):
        frame_f1_scores.append(reward_basic_f1(pred_seg, gt_seg))
    rewards['frame_level_f1'] = np.mean(frame_f1_scores) if frame_f1_scores else 0.0
    
    # 2. Segment matching (temporal)
    segment_iou_scores = []
    for pred_seg, gt_seg in zip(pred_segments, gt_segments):
        segment_iou_scores.append(reward_segment_matching(pred_seg, gt_seg))
    rewards['segment_matching'] = np.mean(segment_iou_scores) if segment_iou_scores else 0.0
    
    # 3. Special movement (if available)
    # TODO: Implement special_movement F1
    rewards['special_movement'] = 0.5  # Placeholder
    
    # 4. Speed accuracy
    # TODO: Implement speed classification accuracy
    rewards['speed_accuracy'] = 0.5  # Placeholder
    
    # 5. Type-only F1 (direction robustness check)
    # TODO: Implement type-only F1
    rewards['type_only_f1'] = rewards['frame_level_f1']  # Placeholder
    
    # Weighted sum
    total_reward = sum(
        weights.get(key, 0) * value 
        for key, value in rewards.items()
    )
    
    return total_reward


# ============================================================================
# OPTION 6: Confidence-Weighted Reward
# ============================================================================

def reward_confidence_weighted(
    pred_segment: Optional[Dict],
    gt_segment: Optional[Dict],
    base_reward_fn=reward_basic_f1
) -> float:
    """
    Penalize wrong high-confidence predictions more than low-confidence ones.
    Reward correct high-confidence predictions.
    
    Use case: Encourages calibrated uncertainty
    Range: [0, 1]
    """
    if pred_segment is None:
        return 0.0 if gt_segment else 1.0
    
    base_reward = base_reward_fn(pred_segment, gt_segment)
    confidence_str = pred_segment.get('confidence', 'medium')
    
    # Map confidence to multiplier
    confidence_map = {
        'low': 0.5,      # Mistakes less penalized, correct less rewarded
        'medium': 1.0,   # Normal reward/penalty
        'high': 1.5      # Mistakes more penalized, correct more rewarded
    }
    
    multiplier = confidence_map.get(confidence_str, 1.0)
    
    # Clip to prevent reward > 1
    return min(1.0, base_reward * multiplier)


# ============================================================================
# OPTION 7: Class-Balanced Reward (For imbalanced datasets)
# ============================================================================

def reward_class_balanced(
    pred_segments: List[Optional[Dict]],
    gt_segments: List[Optional[Dict]],
    class_weights: Dict[str, float] = None
) -> float:
    """
    Weight underrepresented movement types higher (e.g., Arc, Free Fly).
    Common types (Pan, Tilt) get normal weight.
    Rare types get higher weight for better learning.
    
    Use case: Handling class imbalance in camera movements
    Range: [0, 1]
    """
    if class_weights is None:
        # Example weights (adjust based on actual distribution)
        class_weights = {
            'Pan': 1.0,
            'Tilt': 1.0,
            'Static': 1.0,
            'Dolly In': 1.2,
            'Dolly Out': 1.2,
            'Truck': 1.2,
            'Crane': 1.2,
            'Arc': 1.5,        # Rare
            'Free Fly': 1.5,   # Rare
            'Roll': 1.3,
            'Zoom In': 1.2,
            'Zoom Out': 1.2,
            'Focus Shift': 1.5,
            'Follow': 1.3,
            'Unstable': 1.0,
        }
    
    weighted_rewards = []
    
    for pred_seg, gt_seg in zip(pred_segments, gt_segments):
        base_reward = reward_basic_f1(pred_seg, gt_seg)
        
        # Get type from GT segment
        if gt_seg:
            for movement in gt_seg.get('basic_movement', []):
                movement_type = movement.get('type')
                weight = class_weights.get(movement_type, 1.0)
                weighted_rewards.append(base_reward * weight)
        
        # Also account for predicted types (FP penalty)
        if pred_seg:
            for movement in pred_seg.get('basic_movement', []):
                movement_type = movement.get('type')
                # Check if this is a false positive
                if gt_seg:
                    gt_types = {m.get('type') for m in gt_seg.get('basic_movement', [])}
                    if movement_type not in gt_types:
                        weight = class_weights.get(movement_type, 1.0)
                        weighted_rewards.append((1 - base_reward) * weight)
    
    if not weighted_rewards:
        return 0.0
    
    return np.mean(weighted_rewards)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example segments
    gt_seg = {
        'start_time': 0.0,
        'end_time': 2.5,
        'basic_movement': [
            {'type': 'Pan', 'direction': 'left', 'speed': 'medium'},
            {'type': 'Dolly In', 'direction': None, 'speed': 'slow'},
        ],
        'special_movement': ['Handheld'],
        'confidence': 'high'
    }
    
    pred_seg = {
        'start_time': 0.1,
        'end_time': 2.4,
        'basic_movement': [
            {'type': 'Pan', 'direction': 'left', 'speed': 'medium'},
            # Missing Dolly In
        ],
        'special_movement': [],
        'confidence': 'high'
    }
    
    print("=== Reward Function Options ===\n")
    print(f"1. Basic F1: {reward_basic_f1(pred_seg, gt_seg):.4f}")
    print(f"2. Per-Label: {reward_per_label(pred_seg, gt_seg):.4f}")
    print(f"3. Hierarchical: {reward_hierarchical(pred_seg, gt_seg):.4f}")
    print(f"4. Segment Matching (IoU): {reward_segment_matching(pred_seg, gt_seg):.4f}")
    print(f"5. Confidence-Weighted: {reward_confidence_weighted(pred_seg, gt_seg):.4f}")
    print(f"6. Composite: {reward_composite([pred_seg], [gt_seg]):.4f}")

