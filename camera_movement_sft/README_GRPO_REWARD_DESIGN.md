# GRPO Reward Function Design - Complete Guide

This directory contains everything you need to understand and implement reward functions for GRPO training on camera movement recognition.

## 📋 Documentation Overview

### 1. **CAMERA_MOVEMENT_EVALUATION_GUIDE.md** (16 KB) 
   The comprehensive reference guide covering:
   - All 15 camera movement types with key characteristics
   - Expected JSON output format (with validation rules)
   - Frame-level evaluation logic (composite labels, TP/FP/FN calculation)
   - Segment-level evaluation (IoU matching, greedy algorithm)
   - Output parsing and postprocessing
   - Type normalization quirks (e.g., "Truck Left" → "Truck" + "left")
   - Good vs bad predictions with examples
   - Reward function design patterns (7 options)
   - Direction penalty strategy
   - Development checklist
   
   **Read this first for deep understanding.**

### 2. **QUICK_REFERENCE.md** (7.9 KB)
   Quick lookup tables and condensed summaries:
   - 15 movement types table (with directions, examples, characteristics)
   - Composite label system (21 total labels: 12 directional + 9 non-directional)
   - Frame-level scoring example with metric calculation
   - JSON output structure with validation rules
   - Evaluation metrics hierarchy (frame, segment, attribute levels)
   - Common mistakes & penalties table
   - Type normalization reference
   - Development checklist
   
   **Use this as your cheat sheet during implementation.**

### 3. **EVALUATION_PIPELINE_DIAGRAM.md** (8 KB)
   Visual flow diagrams and detailed examples:
   - Complete data flow from model output to reward signal
   - Frame-level scoring with detailed calculations
   - Segment-level IoU matching step-by-step
   - Label hierarchy with per-label metrics
   - Example outputs with realistic numbers
   
   **Study the diagrams to understand the pipeline visually.**

### 4. **eval/reward_function_templates.py** (6 KB)
   Production-ready Python code with 7 reward function implementations:
   - Option 1: Basic F1-based reward (simplest, gradient-friendly)
   - Option 2: Per-label binary reward (fine-grained learning)
   - Option 3: Hierarchical reward (type-first, then direction)
   - Option 4: Segment-level IoU reward (temporal grounding)
   - Option 5: Composite reward (multi-signal, recommended)
   - Option 6: Confidence-weighted reward (uncertainty calibration)
   - Option 7: Class-balanced reward (handling imbalance)
   
   **Copy functions that match your needs; each is self-contained.**

---

## 🎯 Quick Start for GRPO Implementation

### Step 1: Understand the Evaluation System
```
Read QUICK_REFERENCE.md → Understand 15 types + composite labels
↓
Read EVALUATION_PIPELINE_DIAGRAM.md → See frame/segment level scoring
↓
Read CAMERA_MOVEMENT_EVALUATION_GUIDE.md → Full details on metric computation
```

### Step 2: Choose Your Reward Function

**Recommended starting point:**
```python
# From reward_function_templates.py
reward = (
    0.50 * f1_composite_labels +        # Type + direction correctness (main)
    0.15 * f1_segment_matching +        # Temporal grounding (IoU)
    0.10 * f1_type_only +               # Direction robustness
    0.15 * f1_special_movement +        # Auxiliary signal
    0.10 * speed_accuracy               # Speed classification
)
```

This provides:
- ✅ Strong signal on what matters (type + direction)
- ✅ Temporal coherence enforcement (no arbitrary segments)
- ✅ Robustness check (direction learning separate from type)
- ✅ Auxiliary signals (special techniques + speed)
- ✅ Tunable weights for experimentation

### Step 3: Implement Label Extraction

The core operation is extracting **composite labels** from segments:

```python
def get_composite_labels(segment):
    """Extract composite labels from segment"""
    DIRECTIONAL_TYPES = {'Pan', 'Tilt', 'Truck', 'Crane', 'Arc', 'Roll'}
    
    labels = set()
    for movement in segment.get('basic_movement', []):
        mtype = movement.get('type')
        direction = movement.get('direction')
        
        if mtype in DIRECTIONAL_TYPES and direction:
            labels.add(f"{mtype}_{direction}")  # e.g., "Pan_left"
        else:
            labels.add(mtype)  # e.g., "Dolly In"
    
    return labels
```

### Step 4: Compute Frame-Level F1

```python
def compute_f1(pred_segment, gt_segment):
    """Compute F1 score from composite labels"""
    pred_labels = get_composite_labels(pred_segment)
    gt_labels = get_composite_labels(gt_segment)
    
    tp = len(gt_labels & pred_labels)
    fp = len(pred_labels - gt_labels)
    fn = len(gt_labels - pred_labels)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1
```

### Step 5: Aggregate Across Video

Frame-level evaluation samples every 0.1 seconds:

```python
def evaluate_video(pred_segments, gt_segments, sample_step=0.1):
    """Frame-level F1 aggregation"""
    f1_scores = []
    tp_total = fp_total = fn_total = 0
    
    # Get time range
    all_segments = pred_segments + gt_segments
    if not all_segments:
        return 0.0
    
    t_min = min(s['start_time'] for s in all_segments)
    t_max = max(s['end_time'] for s in all_segments)
    
    # Sample frames
    t = t_min
    while t < t_max:
        gt_seg = get_segment_at_time(gt_segments, t)
        pred_seg = get_segment_at_time(pred_segments, t)
        
        if gt_seg or pred_seg:  # Skip if both None
            gt_labels = get_composite_labels(gt_seg)
            pred_labels = get_composite_labels(pred_seg)
            
            tp = len(gt_labels & pred_labels)
            fp = len(pred_labels - gt_labels)
            fn = len(gt_labels - pred_labels)
            
            tp_total += tp
            fp_total += fp
            fn_total += fn
        
        t += sample_step
    
    # Aggregate F1
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1
```

---

## 🔍 Key Insights for Design

### Understanding Composite Labels

The system uses **21 distinct composite labels** combining type + direction:

```
Directional (12 labels):
  Pan_left, Pan_right, Tilt_up, Tilt_down,
  Truck_left, Truck_right, Crane_up, Crane_down,
  Arc_clockwise, Arc_counterclockwise,
  Roll_clockwise, Roll_counterclockwise

Non-Directional (9 labels):
  Static, Unstable, Dolly In, Dolly Out,
  Zoom In, Zoom Out, Follow, Free Fly, Focus Shift
```

Why this matters:
- ✅ Pan_left ≠ Pan_right (different labels, both penalized equally)
- ✅ Pan ≠ Truck (different fundamental movements)
- ✅ Multiple simultaneous movements allowed (1-3 per segment)
- ✅ Partial credit for getting one of two movements right

### Scoring Behavior

**Example 1: Partial Correctness**
```
GT:   {"Pan_left", "Dolly In"}
Pred: {"Pan_left"}

Result: TP=1, FP=0, FN=1
        F1 = 0.667 (not 0 or 1, but in between)
        
Interpretation: Got Pan_left correct (+), missed Dolly In (-)
```

**Example 2: Wrong Direction**
```
GT:   {"Pan_left"}
Pred: {"Pan_right"}

Result: TP=0, FP=1, FN=1
        F1 = 0.0 (complete mismatch)
        
Interpretation: Same type but wrong direction = total penalty
                (can use 0.3 multiplier if desired for direction errors)
```

**Example 3: Hallucination**
```
GT:   {"Pan_left"}
Pred: {"Pan_left", "Zoom_in"}

Result: TP=1, FP=1, FN=0
        F1 = 0.667 (same as partial miss)
        
Interpretation: Symmetric penalty for hallucination vs missing
```

### Why Frame-Level + Segment-Level?

1. **Frame-level (primary signal)**:
   - Fine-grained: samples every 0.1s = ~10x more data points
   - Continuous gradient: helps model learn smooth transitions
   - Per-label tracking: identifies weaknesses by movement type

2. **Segment-level (coherence constraint)**:
   - Prevents arbitrary segmentation (bad: 100 tiny segments)
   - IoU matching ensures temporal boundaries reasonable
   - Encourages model to think in "chunks" not "frames"

Combined effect: Model learns accurate movements AND learns to segment them properly.

---

## 🛠️ Implementation Checklist

- [ ] Copy `reward_function_templates.py` to your codebase
- [ ] Define label extraction function (copy `get_composite_labels`)
- [ ] Define F1 computation (copy `compute_f1`)
- [ ] Implement frame-level aggregation (sample every 0.1s)
- [ ] Implement segment-level IoU matching (greedy algorithm)
- [ ] Combine into composite reward (0.50/0.15/0.10/0.15/0.10 weights)
- [ ] Add output validation:
  - [ ] Check video_id is present
  - [ ] Check segments are sorted chronologically
  - [ ] Check type ∈ VALID_BASIC_TYPES
  - [ ] Check direction matches type rules
  - [ ] Check speed ∈ {zero, slow, medium, fast}
- [ ] Add type normalization:
  - [ ] "Truck Left" → ("Truck", "left")
  - [ ] "Pedestal Up" → ("Crane", "up")
- [ ] Add null handling:
  - [ ] Handle "null", "Null", "NULL" strings
  - [ ] Handle empty strings
  - [ ] Unify to Python None
- [ ] Test on small validation set (10-20 videos)
- [ ] Monitor per-label F1 distribution
- [ ] Adjust weights based on performance
- [ ] Consider class weighting for rare types (Arc, Free Fly)

---

## 📊 Monitoring Training

Track these metrics during GRPO training:

```python
metrics = {
    'reward_mean': mean(rewards),           # Should increase
    'reward_std': std(rewards),             # Monitor drift
    'f1_composite': mean(f1_composite),     # Should increase
    'f1_segment': mean(f1_segment),         # Should increase
    'precision': mean(precision_scores),    # Should increase
    'recall': mean(recall_scores),          # Should increase
}

# Per-label monitoring (identify weak areas)
per_label = {
    'Pan_left': {'f1': 0.95, 'count': 450},
    'Pan_right': {'f1': 0.92, 'count': 380},
    'Arc_clockwise': {'f1': 0.67, 'count': 12},  # ← Struggling
    'Free_Fly': {'f1': 0.60, 'count': 5},        # ← Very rare
}

# Action: Increase weight for Arc/Free Fly in loss
```

---

## 🤔 Common Questions

**Q: Why use composite labels instead of separate type/direction scores?**
A: Composite labels capture the full specification. Pan_left ≠ Pan_right, so they must be distinct labels. This prevents the model from learning "Pan" without learning direction.

**Q: Why F1 instead of just accuracy?**
A: F1 balances precision (no false positives) and recall (catch all movements). In multi-label settings, accuracy is misleading (e.g., "always output Static" could have high accuracy on class distributions).

**Q: What if my model outputs different segment boundaries than GT?**
A: That's handled by segment-level IoU matching (threshold IoU > 0.5). Minor boundary shifts (0.1-0.2s) are tolerated. Major misalignments are penalized.

**Q: Should I weight rare movements (Arc, Free Fly) higher?**
A: Yes! Use class weighting (Arc=1.5x, others=1.0x). Rare types have low signal but high value for model robustness.

**Q: Can I use confidence field from model?**
A: Yes! Use `reward_confidence_weighted` template: high-confidence mistakes are more penalized, high-confidence correct predictions more rewarded.

---

## 📚 References

- **Source**: `/group/40009/dazhaodu/ms-swift/camera_movement_sft/`
- **Evaluation Code**: `eval/evaluate_camera_movement_fixed.py`
- **Postprocessing**: `eval/postprocess_results.py`
- **Common Definitions**: `common.py` (VALID_BASIC_TYPES, VALID_SPEEDS, etc.)

---

## 🚀 Next Steps

1. Start with basic F1 reward (simplest implementation)
2. Test on small validation set
3. Monitor per-label performance
4. Add segment-level IoU constraint
5. Experiment with weight ratios
6. Implement class weighting for rare types
7. Consider hierarchical rewards (type → direction)
8. Integrate confidence weighting

Good luck! 🎬

