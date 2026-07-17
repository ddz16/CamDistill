# Camera Movement Evaluation Pipeline (Visual Guide)

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GRPO TRAINING LOOP                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Model Inference       │
                    │  (Generates JSON)       │
                    └────────────┬────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        POSTPROCESSING (postprocess_results.py)          │
│                                                                          │
│  Raw Model Output                                                       │
│  ├─ Try direct JSON parse                                              │
│  ├─ Try ```json ... ``` block extraction                              │
│  ├─ Try first {...} block extraction                                  │
│  └─ Fallback: {"raw": raw_text}                                       │
│                                                                          │
│  Result: {"video_id": "...", "segments": [...]}                       │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  NORMALIZATION & VALIDATION                             │
│                                                                          │
│  Type Normalization:                                                    │
│  ├─ "Truck Left" → ("Truck", "left")                                  │
│  ├─ "Pedestal Up" → ("Crane", "up")                                   │
│  ├─ Validate type ∈ VALID_BASIC_TYPES (15 types)                     │
│  └─ Validate direction ∈ VALID_DIRECTIONS                             │
│                                                                          │
│  Null Normalization:                                                    │
│  ├─ Handle "null", "Null", "NULL" strings                             │
│  ├─ Handle empty strings ""                                            │
│  └─ Unify to Python None                                              │
│                                                                          │
│  Speed Validation:                                                      │
│  └─ Validate speed ∈ {zero, slow, medium, fast}                       │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              COMPOSITE LABEL EXTRACTION                                 │
│                                                                          │
│  For each segment's basic_movement array:                              │
│                                                                          │
│  ┌──────────────────────┐         ┌──────────────────┐                 │
│  │ Directional Type?    │         │ Non-Directional  │                │
│  │ Pan/Tilt/Truck/...   │         │ Dolly/Zoom/...   │                │
│  └──────────┬───────────┘         └────────┬─────────┘                 │
│             │                              │                            │
│    ┌────────▼────────────┐        ┌────────▼──────────┐                │
│    │ Combine type+dir    │        │ Use type only     │                │
│    │ "Pan_left"          │        │ "Dolly In"        │                │
│    │ "Crane_up"          │        │ "Static"          │                │
│    │ "Arc_clockwise"     │        │ "Focus Shift"     │                │
│    └─────────────────────┘        └───────────────────┘                │
│                                                                          │
│  Result: Labels = {"Pan_left", "Dolly In"} for this segment            │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              EVALUATION LEVEL 1: FRAME-LEVEL (Every 0.1s)              │
│                                                                          │
│  For each sampled time point t in [0, video_length]:                   │
│                                                                          │
│  Step 1: Get GT segment at time t                                      │
│          ├─ None: silent (no movement)                                 │
│          └─ Segment: extract composite labels                          │
│                                                                          │
│  Step 2: Get Pred segment at time t                                    │
│          ├─ None: silent (no movement)                                 │
│          └─ Segment: extract composite labels                          │
│                                                                          │
│  Step 3: Multi-label set comparison                                    │
│                                                                          │
│    gt_labels = {"Pan_left", "Dolly In"}                                │
│    pred_labels = {"Pan_left"}                                          │
│                                                                          │
│    TP = |{"Pan_left"} ∩ {"Pan_left", "Dolly In"}| = 1                 │
│    FP = |{"Pan_left"} - {"Pan_left", "Dolly In"}| = 0                 │
│    FN = |{"Pan_left", "Dolly In"} - {"Pan_left"}| = 1                 │
│                                                                          │
│    Precision = TP / (TP + FP) = 1.0                                    │
│    Recall = TP / (TP + FN) = 0.5                                       │
│    F1 = 2*P*R / (P+R) = 0.667                                          │
│                                                                          │
│  Step 4: Aggregate across all frames                                   │
│          ├─ Sum TP, FP, FN                                             │
│          ├─ Compute overall P, R, F1                                   │
│          └─ Per-label metrics                                          │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
        ┌────────────────┴─────────────────┐
        ▼                                   ▼
┌──────────────────────────┐       ┌──────────────────────────┐
│ LEVEL 2: SEGMENT-LEVEL   │       │ LEVEL 3: ATTRIBUTES      │
│ (Temporal Alignment)     │       │ (Speed, Special Movement)│
│                          │       │                          │
│ Greedy IoU Matching:     │       │ ├─ Speed: Accuracy      │
│ ├─ Compute all pairs     │       │ ├─ Special: Multi-label │
│ ├─ Sort by IoU (desc)    │       │ └─ Type-only: F1        │
│ ├─ Greedy matching       │       │                          │
│ └─ Match rate P/R/F1     │       │ (Only when both have     │
│                          │       │  segments at same time)  │
└────────────────┬─────────┘       └──────────────┬───────────┘
                 │                                │
                 └────────────────┬────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              REWARD SIGNAL GENERATION (GRPO)                            │
│                                                                          │
│  Composite Reward Calculation:                                         │
│                                                                          │
│  reward = (                                                             │
│      0.50 * f1_composite_labels +        # Type + direction             │
│      0.15 * f1_segment_matching +        # Temporal IoU                 │
│      0.10 * f1_type_only +               # Direction robustness         │
│      0.15 * f1_special_movement +        # Auxiliary signal             │
│      0.10 * speed_accuracy               # Speed classification         │
│  )                                                                       │
│                                                                          │
│  Range: [0.0, 1.0] where 1.0 = perfect prediction                      │
└─────────────────────┬────────────────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   GRPO Gradient Update      │
        │   ∇_θ R(θ) via PPO/etc      │
        └─────────────────────────────┘
```

---

## Frame-Level Scoring Detail

```
TIME: 1.5 seconds

┌─────────────────────────────────────────────────────────────────┐
│ Ground Truth Segment                                            │
├─────────────────────────────────────────────────────────────────┤
│ basic_movement: [                                               │
│   {"type": "Pan", "direction": "left", "speed": "medium"},     │
│   {"type": "Dolly In", "direction": null, "speed": "slow"}     │
│ ]                                                               │
│                                                                 │
│ → Composite Labels: {"Pan_left", "Dolly In"}                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Prediction Segment                                              │
├─────────────────────────────────────────────────────────────────┤
│ basic_movement: [                                               │
│   {"type": "Pan", "direction": "left", "speed": "medium"}      │
│   # Missing Dolly In!                                           │
│ ]                                                               │
│                                                                 │
│ → Composite Labels: {"Pan_left"}                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Set Operations                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Intersection (TP):     {"Pan_left"} ∩ {"Pan_left", "Dolly In"} │
│                        = {"Pan_left"} → count=1                │
│                                                                 │
│ False Positives (FP):  {"Pan_left"} - {"Pan_left", "Dolly In"} │
│                        = {} → count=0                           │
│                                                                 │
│ False Negatives (FN):  {"Pan_left", "Dolly In"} - {"Pan_left"} │
│                        = {"Dolly In"} → count=1                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Metrics                                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Precision = TP / (TP + FP) = 1 / 1 = 1.0                       │
│   ✓ What was predicted is correct (no false positives)         │
│                                                                 │
│ Recall = TP / (TP + FN) = 1 / 2 = 0.5                          │
│   ✗ Missed 50% of the ground truth movements                   │
│                                                                 │
│ F1 = 2*P*R / (P+R) = 2*1.0*0.5 / 1.5 = 0.667                  │
│   = Balanced measure: not penalizing false positives,          │
│   but penalizing false negatives (missing movements)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Segment-Level IoU Matching

```
GT Segments:
  Seg 1: [0.0, 2.5]  ──►  Type: Pan_left, Dolly In
  Seg 2: [2.5, 5.0]  ──►  Type: Truck_right
  Seg 3: [5.0, 7.5]  ──►  Type: Crane_up

Pred Segments:
  Seg A: [0.1, 2.4]  ──►  Type: Pan_left
  Seg B: [2.4, 5.1]  ──►  Type: Truck_right, Dolly In
  Seg C: [5.1, 7.6]  ──►  Type: Crane_up, Roll_clockwise

┌────────────────────────────────────────────────────┐
│ IoU Computation (All Pairs)                         │
├────────────────────────────────────────────────────┤
│                                                    │
│ (GT1[0,2.5], PredA[0.1,2.4]):                     │
│   intersection = min(2.5, 2.4) - max(0, 0.1)     │
│               = 2.4 - 0.1 = 2.3                   │
│   union = (2.5-0) + (2.4-0.1) - 2.3               │
│         = 2.5 + 2.3 - 2.3 = 2.5                   │
│   IoU = 2.3 / 2.5 = 0.92 ✓ > 0.5                  │
│                                                    │
│ (GT1[0,2.5], PredB[2.4,5.1]):                     │
│   intersection = max(0, 2.4 - 2.5) = 0            │
│   IoU = 0 / union = 0.0 ✗ < 0.5                   │
│                                                    │
│ (GT1[0,2.5], PredC[5.1,7.6]):                     │
│   intersection = 0                                │
│   IoU = 0.0 ✗ < 0.5                               │
│                                                    │
│ ...etc for all 9 pairs...                         │
│                                                    │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ Greedy Matching (Sort by IoU, highest first)       │
├────────────────────────────────────────────────────┤
│                                                    │
│ Matches ≥ 0.5:                                    │
│  1. GT1 + PredA: IoU=0.92                         │
│  2. GT2 + PredB: IoU=0.80                         │
│  3. GT3 + PredC: IoU=0.78                         │
│                                                    │
│ Selected (greedy, each used once):                │
│  ✓ GT1 ↔ PredA (IoU=0.92)                         │
│  ✓ GT2 ↔ PredB (IoU=0.80)                         │
│  ✓ GT3 ↔ PredC (IoU=0.78)                         │
│                                                    │
│ Metrics:                                          │
│  matched = 3                                      │
│  seg_precision = 3/3 = 1.0                        │
│  seg_recall = 3/3 = 1.0                           │
│  seg_f1 = 1.0                                     │
│                                                    │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ Attribute Evaluation on Matched Pairs              │
├────────────────────────────────────────────────────┤
│                                                    │
│ Matched Pair 1: GT1 ↔ PredA                       │
│   GT:   [Pan_left, Dolly In]                      │
│   Pred: [Pan_left]                                │
│   → TP=1, FP=0, FN=1 → F1=0.667                   │
│                                                    │
│ Matched Pair 2: GT2 ↔ PredB                       │
│   GT:   [Truck_right]                             │
│   Pred: [Truck_right, Dolly In]                   │
│   → TP=1, FP=1, FN=0 → F1=0.667                   │
│                                                    │
│ Matched Pair 3: GT3 ↔ PredC                       │
│   GT:   [Crane_up]                                │
│   Pred: [Crane_up, Roll_clockwise]                │
│   → TP=1, FP=1, FN=0 → F1=0.667                   │
│                                                    │
│ Overall matched F1 = (0.667+0.667+0.667)/3 = 0.667
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## Label Hierarchy

```
┌─────────────────────────────────────────────┐
│      21 Possible Composite Labels           │
└─────────────────────────────────────────────┘

Directional (12 labels):
├─ Pan:  Pan_left, Pan_right
├─ Tilt: Tilt_up, Tilt_down
├─ Truck: Truck_left, Truck_right
├─ Crane: Crane_up, Crane_down
├─ Arc: Arc_clockwise, Arc_counterclockwise
└─ Roll: Roll_clockwise, Roll_counterclockwise

Non-Directional (9 labels):
├─ Static
├─ Unstable
├─ Dolly In
├─ Dolly Out
├─ Zoom In
├─ Zoom Out
├─ Follow
├─ Free Fly
└─ Focus Shift

Per-Label F1 Tracking:
├─ Pan_left:     TP=45, FP=2, FN=3  → P=0.96, R=0.94, F1=0.95
├─ Pan_right:    TP=38, FP=1, FN=5  → P=0.97, R=0.88, F1=0.92
├─ Tilt_up:      TP=30, FP=0, FN=8  → P=1.0, R=0.79, F1=0.88
├─ Truck_left:   TP=20, FP=3, FN=7  → P=0.87, R=0.74, F1=0.80
├─ ...
├─ Arc_clockwise: TP=8, FP=0, FN=4  → P=1.0, R=0.67, F1=0.80  ← Underrepresented
├─ Free_Fly:     TP=3, FP=1, FN=2   → P=0.75, R=0.60, F1=0.67  ← Rare!
└─ ...

Analysis:
  - Pan/Tilt: Well-learned (F1 > 0.90)
  - Dolly/Zoom: Moderate (F1 ≈ 0.75)
  - Arc/Free Fly: Poor (F1 < 0.70) → Need class weighting!
```

---

## Example Output

### Frame-Level Results
```
Metric                               Precision  Recall  F1      TP    FP    FN
─────────────────────────────────────────────────────────────────────────────
basic_movement(type+direction)       0.8934     0.8456  0.8689  1245  147   227
basic_movement(type_only)            0.9123     0.8901  0.9010  1287  125   161
special_movement                     0.7234     0.6890  0.7057  412   158   184
speed                                          Accuracy: 0.8234  (492 correct / 598 total)
```

### Per-Label Breakdown
```
Label               Precision  Recall  F1      TP   FP   FN
─────────────────────────────────────────────────────────
Pan_left            0.9600     0.9400  0.9500  47   2    3
Pan_right           0.9700     0.8800  0.9200  44   1    6
Tilt_up             1.0000     0.7900  0.8800  30   0    8
Tilt_down           0.9500     0.8300  0.8800  35   2    7
Truck_left          0.8700     0.7400  0.8000  20   3    7
Truck_right         0.9000     0.7500  0.8182  18   2    6
Crane_up            0.8333     0.7143  0.7692  25   5    10
Crane_down          0.7500     0.6667  0.7059  20   7    10
...
Arc_clockwise       1.0000     0.6667  0.8000  8    0    4    ← Rare, high precision
Free_Fly            0.7500     0.6000  0.6667  3    1    2    ← Rare, low recall
```

### Segment-Level Results
```
Segment Matching (IoU >= 0.5):
  Precision: 0.9231 (36/39 predicted segments matched)
  Recall:    0.8235 (36/43 GT segments matched)
  F1:        0.8696

Matched Attributes:
  basic_movement(type+dir): P=0.8945, R=0.8667, F1=0.8804
  special_movement:         P=0.7500, R=0.7143, F1=0.7317
  speed:                    Accuracy=0.8333
```

