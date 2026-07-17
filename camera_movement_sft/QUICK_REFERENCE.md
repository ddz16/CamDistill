# Quick Reference: Camera Movement Evaluation

## The 15 Camera Movement Types

| # | Type | Direction | Example | Key Characteristic |
|---|------|-----------|---------|-------------------|
| 1 | **Static** | null | No movement | Machine/orientation unchanged |
| 2 | **Unstable** | null | Handheld shaking | Uncontrolled jittering |
| 3 | **Pan** | ✓ left/right | Horizontal rotation | Fixed pivot, horizontal scan |
| 4 | **Tilt** | ✓ up/down | Vertical rotation | Fixed pivot, vertical scan |
| 5 | **Truck** | ✓ left/right | Lateral movement | Depth parallax (near fast, far slow) |
| 6 | **Crane** | ✓ up/down | Vertical elevation | Horizon changes, perspective shift |
| 7 | **Dolly In** | null | Moving forward | Parallax present (not zoom) |
| 8 | **Dolly Out** | null | Moving backward | Parallax present (not zoom) |
| 9 | **Zoom In** | null | Focal length ↑ | Uniform scaling, NO parallax |
| 10 | **Zoom Out** | null | Focal length ↓ | Uniform scaling, NO parallax |
| 11 | **Roll** | ✓ clockwise/cc | Optical axis rotation | Horizon tilts |
| 12 | **Arc** | ✓ clockwise/cc | Orbital movement | Subject centered, angle changes |
| 13 | **Follow** | - | Tracks subject | Background continuous change |
| 14 | **Free Fly** | - | Complex 3D path | No clear classifiable motion |
| 15 | **Focus Shift** | null | Focus plane change | Depth layer virtualization |

## Composite Label System

The evaluation uses **composite labels** combining type + direction:

### Directional Movements (6 types)
```
Pan_left        Pan_right
Tilt_up         Tilt_down
Truck_left      Truck_right
Crane_up        Crane_down
Arc_clockwise   Arc_counterclockwise
Roll_clockwise  Roll_counterclockwise
```

### Non-Directional Movements (9 types)
```
Static          Unstable        Dolly In        Dolly Out
Zoom In         Zoom Out        Follow          Free Fly
Focus Shift
```

**Total possible labels: 12 directional + 9 non-directional = 21 distinct labels**

---

## Scoring at Each Time Point (Frame-Level)

### Example Scenario
```
Time: t = 1.5 seconds

GT Segment:
  basic_movement: [
    {"type": "Pan", "direction": "left", "speed": "medium"},
    {"type": "Dolly In", "direction": null, "speed": "slow"}
  ]
  → Labels: {"Pan_left", "Dolly In"}

Prediction:
  basic_movement: [
    {"type": "Pan", "direction": "left", "speed": "medium"}
  ]
  → Labels: {"Pan_left"}
```

### Metric Calculation
```
TP (True Positive)  = |{"Pan_left"} ∩ {"Pan_left", "Dolly In"}| = 1
FP (False Positive) = |{"Pan_left"} - {"Pan_left", "Dolly In"}| = 0
FN (False Negative) = |{"Pan_left", "Dolly In"} - {"Pan_left"}| = 1

Precision = TP / (TP + FP) = 1 / (1 + 0) = 1.0
Recall    = TP / (TP + FN) = 1 / (1 + 1) = 0.5
F1        = 2 * P * R / (P + R) = 2 * 1.0 * 0.5 / 1.5 = 0.667
```

**Insight**: Model got Pan_left correct (TP=1) but missed Dolly In (FN=1)
- **Precision is high (1.0)**: No false positives
- **Recall is low (0.5)**: Missed half the ground truth
- **F1 = 0.667**: Balanced penalty for the miss

---

## Output JSON Structure

### Required Fields
```json
{
  "video_id": "unique_identifier",
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.5,
      
      "basic_movement": [
        {
          "type": "Pan",              // Required: Type from 15 types
          "direction": "left",        // Required if type in {Pan,Tilt,Truck,Crane,Arc,Roll}
                                      // Otherwise: null
          "speed": "medium"           // Required: {zero, slow, medium, fast}
        }
      ],
      
      "special_movement": ["Handheld"],  // Optional: Special techniques
      "visual_evidence": "...",          // Recommended: Description
      "confidence": "high"               // Optional: {high, medium, low}
    }
  ]
}
```

### Validation Rules
- ✅ Segments must be **chronologically ordered**
- ✅ **No overlapping** segments (by design, each frame belongs to one segment)
- ✅ type must be in VALID_BASIC_TYPES (the 15 types)
- ✅ direction must match type-specific rules
- ✅ speed must be in {zero, slow, medium, fast}
- ✅ basic_movement array: 1-3 elements (max 3 simultaneous movements)

---

## Evaluation Metrics Hierarchy

### Frame-Level (Primary)
- **Metric**: Composite label F1 score
- **Sample Frequency**: Every 0.1 seconds
- **Output**: Per-frame precision, recall, F1
- **Per-Label Breakdown**: F1 separately for each label (Pan_left, Pan_right, etc.)

### Segment-Level (Secondary)
- **Metric**: IoU-based segment matching
- **Matching**: Greedy (highest IoU first)
- **Threshold**: IoU > 0.5 (default)
- **Output**: Segment P/R/F1 + attribute F1 for matched pairs

### Attribute-Level (Tertiary)
- **Speed**: Multi-class accuracy {zero, slow, medium, fast}
- **Special Movement**: Multi-label F1
- **Type-Only**: F1 ignoring direction (robustness check)

---

## GRPO Reward Function Design

### Recommended Composite Reward
```python
reward = (
    0.50 * f1_composite_labels +        # Type + direction correctness
    0.15 * f1_segment_matching +        # Temporal grounding (IoU)
    0.10 * f1_type_only +               # Direction robustness
    0.15 * f1_special_movement +        # Auxiliary techniques
    0.10 * speed_accuracy               # Speed classification
)
```

### Reward Function Options (Complexity vs Informativeness)

| Option | Complexity | Gradient Quality | Use Case |
|--------|-----------|------------------|----------|
| **Basic F1** | Low | Smooth | Quick training |
| **Per-Label** | Medium | Fine-grained | Selective credit for partial correctness |
| **Hierarchical** | Medium | Two-stage | Type learning first, then direction |
| **Segment IoU** | Low-Medium | Smooth | Enforce temporal coherence |
| **Confidence-Weighted** | Medium | Task-aware | Uncertainty calibration |
| **Class-Balanced** | Medium | Reweighted | Handle imbalanced movement types |
| **Composite** | High | Multi-signal | Best for comprehensive training |

---

## Common Mistakes & Penalties

### Type Errors (Worst)
```
GT: Pan_left  →  Pred: Truck_left
TP = 0 (types don't match)
F1 = 0.0
```
**Penalty**: Full – These are not interchangeable

### Direction Errors (Moderate)
```
GT: Pan_left  →  Pred: Pan_right
TP = 0 (labels don't match)
F1 = 0.0
```
**Penalty**: Full F1 but could use 0.3 multiplier if desired (same type)

### Missing Movements (Low)
```
GT: {Pan_left, Dolly In}  →  Pred: {Pan_left}
TP = 1, FN = 1
F1 = 2*1*0.5 / 1.5 = 0.667
```
**Penalty**: Partial – Got one right, missed one

### Hallucinations (Moderate)
```
GT: {Pan_left}  →  Pred: {Pan_left, Zoom_in}
TP = 1, FP = 1
F1 = 2*1*0.5 / 1.5 = 0.667
```
**Penalty**: Partial – Same as missing (symmetrical)

### Silent Output (Worst)
```
GT: {Pan_left}  →  Pred: {}
TP = 0, FN = 1
F1 = 0.0
```
**Penalty**: Full – Model predicted no movement when there is

---

## Type Normalization Quirks

Watch for these during post-processing:

| Model Output | Standard Form | Conversion |
|--------------|--------------|-----------|
| "Truck Left" | "Truck", direction="left" | Split type |
| "Truck Right" | "Truck", direction="right" | Split type |
| "Pedestal Up" | "Crane", direction="up" | Rename type |
| "Pedestal Down" | "Crane", direction="down" | Rename type |

These are **automatically normalized** by the evaluation script.

---

## Development Checklist

- [ ] Define composite reward function weights (start with 0.50/0.15/0.10/0.15/0.10)
- [ ] Decide on class weighting (Arc/Free Fly higher? Arc=1.5, others=1.0?)
- [ ] Determine segment matching threshold (default IoU=0.5)
- [ ] Choose frame sampling frequency (default 0.1s)
- [ ] Decide if using exclude_slow_gt flag (recommended for clearer signals)
- [ ] Implement output parsing with JSON fallback
- [ ] Add type normalization (Truck Left → Truck + left)
- [ ] Test on small validation set first
- [ ] Monitor per-label F1 distribution (ensure balanced learning)
- [ ] Consider direction penalty multiplier (0.3 for wrong direction vs 0.0 for wrong type)

