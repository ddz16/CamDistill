# Camera Movement Evaluation Logic & GRPO Reward Function Design Guide

## 1. The 15 Camera Movement Types (From `common.py`)

### Basic Movement Types (VALID_BASIC_TYPES)
The system defines **15 basic camera movement types**:

1. **Static** - Fixed camera position/orientation
2. **Unstable** - Uncontrolled jittering/shaking
3. **Dolly In** - Camera physically moves forward
4. **Dolly Out** - Camera physically moves backward
5. **Truck** - Camera lateral movement (left/right)
6. **Crane** - Camera vertical movement (up/down)
7. **Follow** - Camera tracks a moving subject
8. **Arc** - Camera orbits around a subject
9. **Free Fly** - Complex 3D free-form movement
10. **Pan** - Camera rotates horizontally (left/right)
11. **Tilt** - Camera rotates vertically (up/down)
12. **Roll** - Camera rotates around optical axis
13. **Zoom In** - Focal length increases (not camera movement)
14. **Zoom Out** - Focal length decreases (not camera movement)
15. **Focus Shift** - Focus layer change (not camera movement)

### Directional vs Non-Directional Types
**Require Direction Field:**
- `Pan` → left/right
- `Tilt` → up/down
- `Truck` → left/right
- `Crane` → up/down
- `Arc` → clockwise/counterclockwise
- `Roll` → clockwise/counterclockwise

**Forbid Direction Field (must be null):**
- Static, Unstable, Dolly In, Dolly Out, Follow, Free Fly, Zoom In, Zoom Out, Focus Shift

---

## 2. Expected JSON Output Format

### Model Output (eval_pred.jsonl format)

```json
{
  "video_id": "video_123",
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.5,
      "visual_evidence": "Background buildings pan left uniformly",
      "basic_movement": [
        {
          "type": "Pan",
          "direction": "left",
          "speed": "medium"
        }
      ],
      "special_movement": ["Handheld"],
      "confidence": "high"
    },
    {
      "start_time": 2.5,
      "end_time": 5.0,
      "visual_evidence": "Camera moves forward with depth parallax",
      "basic_movement": [
        {
          "type": "Dolly In",
          "direction": null,
          "speed": "slow"
        }
      ],
      "special_movement": [],
      "confidence": "medium"
    }
  ]
}
```

### Key Structure Rules
1. **video_id**: Unique video identifier
2. **segments**: Array of temporal segments (must be chronologically ordered)
3. **start_time / end_time**: Float timestamps in seconds
4. **basic_movement**: Array of 1-3 movement objects (can be multiple simultaneous movements)
5. **direction**: Must follow type-specific rules (required for Pan/Tilt/Truck/Crane/Arc/Roll, null for others)
6. **speed**: One of {zero, slow, medium, fast}
7. **special_movement**: Optional array (Handheld, Steadicam, Shaky, Aerial, etc.)
8. **confidence**: high/medium/low (for quality estimation)
9. **visual_evidence**: Text explanation of visual cues

---

## 3. Evaluation Metrics & Scoring Logic

### Two Levels of Evaluation

#### A. Frame-Level Evaluation (Sampled every 0.1s)

**Core Scoring Approach: Multi-Label Set Comparison**

For each sampled time point `t`:
1. Extract GT segment at time `t` (if exists)
2. Extract Pred segment at time `t` (if exists)
3. Convert segments to label sets

**Label Composition:**

The evaluation uses **composite labels** that combine type + direction:

```python
# For directional types (Pan/Tilt/Truck/Crane/Arc/Roll):
# Labels like: "Pan_left", "Pan_right", "Truck_left", "Crane_up", "Arc_clockwise"

# For non-directional types (Static/Dolly In/Zoom In, etc.):
# Labels are just the type: "Static", "Dolly In", "Zoom In"

# Example segment:
basic_movement = [
    {"type": "Pan", "direction": "left", "speed": "medium"},
    {"type": "Dolly In", "direction": null, "speed": "slow"}
]
# → Labels: {"Pan_left", "Dolly In"}
```

**Metric Calculation (Per Sampled Frame):**

```python
# At time t:
gt_labels = extract_labels(gt_segment_at_t)
pred_labels = extract_labels(pred_segment_at_t)

TP = |gt_labels ∩ pred_labels|          # Correct predictions
FP = |pred_labels - gt_labels|          # False positives
FN = |gt_labels - pred_labels|          # False negatives (missed labels)

Precision = TP / (TP + FP)  if (TP + FP) > 0 else 0
Recall = TP / (TP + FN)     if (TP + FN) > 0 else 0
F1 = 2 * P * R / (P + R)    if (P + R) > 0 else 0
```

**Key Points:**
- **Partial Credit**: Getting Pan_left correct but missing Dolly In = TP=1, FN=1 (not F1=0)
- **Per-label Granularity**: Each composite label (e.g., "Pan_left") is evaluated independently
- **Silent Segments**: If both GT and Pred have no segment at time `t`, that frame is skipped
- **Asymmetric Penalties**: 
  - GT has movement but Pred doesn't → FN (missed movement)
  - Pred has movement but GT doesn't → FP (hallucinated movement)

**Three Metrics Tracked:**
1. **basic_movement_with_direction**: Full composite labels (Pan_left, Truck_right, etc.)
2. **basic_movement_type_only**: Only type, ignoring direction (for direction robustness check)
3. **special_movement**: Special techniques (Handheld, Aerial, etc.)
4. **speed**: Classification accuracy (zero/slow/medium/fast exact match)

**Filtering Option:**
- `exclude_slow_gt=True`: Removes GT segments where ALL basic_movements have speed="slow"
  - Rationale: Slow movements are harder to annotate reliably

---

#### B. Segment-Level Evaluation

**Matching Strategy: Greedy Matching by IoU**

```python
# Step 1: Compute IoU for all (GT, Pred) pairs
for each (gt_segment, pred_segment):
    intersection = overlap_duration
    union = gt_duration + pred_duration - intersection
    IoU = intersection / union

# Step 2: Greedy matching (highest IoU first)
matches = []
used_gt, used_pred = set(), set()
for (gt_idx, pred_idx, iou) in sorted_by_iou_descending:
    if gt_idx not in used_gt and pred_idx not in used_pred:
        matches.append((gt_idx, pred_idx, iou))
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)

# Step 3: Evaluate matched pairs
seg_precision = len(matches) / len(pred_segments)
seg_recall = len(matches) / len(gt_segments)
seg_f1 = 2 * P * R / (P + R)
```

**Segment-Level Metrics (with default IoU threshold = 0.5):**

```
Precision = matched_segments / total_pred_segments
Recall = matched_segments / total_gt_segments
F1 = 2 * P * R / (P + R)
```

**For Each Matched Pair:**
- Evaluate attribute agreement (using same frame-level metrics)
- Compute basic_movement F1, special_movement F1, speed accuracy

**Not About Exact Timing:**
- No penalty for 0.4s vs 0.5s duration if IoU > threshold
- Only penalized if IoU < threshold (significant temporal mismatch)

---

### Output Parsing (`postprocess_results.py`)

The model output is parsed with **graceful fallback**:

```python
def parse_json_response(raw_text: str) -> dict:
    # Try 1: Direct JSON parsing
    try:
        return json.loads(raw_text)
    except:
        pass
    
    # Try 2: Extract ```json ... ``` block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # Try 3: Extract first {...} block
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    # Fallback: Return raw text
    return {"raw": raw_text}
```

**Video ID Extraction Fallback:**
```
Priority 1: From 'videos' field in infer result
            (extracts filename without extension)
Priority 2: From 'video_id_mapping.jsonl' by line number
Priority 3: Generate "unknown_{i}" identifier
```

---

## 4. What Makes a "Good" vs "BAD" Prediction?

### Good Predictions ✓
1. **Correct composite labels**: Pan_left matched vs Pan_left in GT
2. **Complete coverage**: All GT movements labeled (high recall)
3. **Confident labels**: No extraneous movements (high precision)
4. **Proper timing**: Segments with IoU > 0.5
5. **Direction accuracy**: Pan_right correctly identified (not Pan_left)
6. **Speed consistency**: "medium" matches "medium" (not "slow")

### Bad Predictions ✗
1. **Wrong direction**: Predicting Pan_left when GT is Pan_right (FN + FP)
2. **Missing movements**: GT has [Pan_left, Dolly In] but Pred only has [Pan_left] (FN for Dolly In)
3. **Hallucinated movements**: Pred has movements not in GT (FP)
4. **Timing misalignment**: IoU < 0.5 (segments don't match temporally)
5. **Wrong type**: Predicting Truck when GT is Pan (complete type mismatch)
6. **Silent output**: Predicting [] when GT has movements (all are FN)

---

## 5. Designing GRPO Reward Functions

### Reward Structure Template

Based on the metrics above, design rewards as:

```python
def compute_reward(pred_segments, gt_segments):
    """
    GRPO reward combining multiple signal sources
    """
    rewards = {
        'f1_composite_labels': compute_frame_level_f1(pred, gt),     # Primary signal
        'f1_type_only': compute_type_only_f1(pred, gt),               # Robustness check
        'segment_matching_f1': compute_segment_f1(pred, gt),          # Temporal awareness
        'special_movement_f1': compute_special_f1(pred, gt),          # Aux signal
        'speed_accuracy': compute_speed_accuracy(pred, gt),           # Secondary
    }
    
    # Composite reward (tunable weights)
    return (
        0.50 * rewards['f1_composite_labels'] +      # Main objective
        0.15 * rewards['segment_matching_f1'] +      # Temporal quality
        0.10 * rewards['f1_type_only'] +             # Direction robustness
        0.15 * rewards['special_movement_f1'] +      # Technique detection
        0.10 * rewards['speed_accuracy']             # Speed classification
    )
```

### Key Reward Signal Options

**1. Frame-Level Composite Label F1** (Primary)
- Incentivizes correct (type, direction) pairs
- Captures both precision (no hallucinations) and recall (complete labeling)
- **Good for**: Most accurate, granular feedback
- **Range**: [0, 1]

**2. Segment-Level F1** (Temporal Grounding)
- Ensures temporal boundaries are reasonable
- Greedy matching with IoU > 0.5 threshold
- **Good for**: Preventing arbitrary segmentation
- **Range**: [0, 1]

**3. Per-Label F1** (Fine-grained)
- Compute F1 separately for each label (e.g., "Pan_left" only)
- **Good for**: Balancing across underrepresented movements
- **Action**: Use weighted average or focal loss on underrepresented labels

**4. Direction Accuracy** (Selective)
- Only for Pan/Tilt/Truck/Crane/Arc/Roll
- Penalize wrong direction more than missing direction
- **Good for**: Directional robustness
- **Formula**: Separate reward for directional types

**5. Speed Accuracy** (Secondary)
- Only count when both GT and Pred have a segment
- Multi-class accuracy: {zero, slow, medium, fast}
- **Good for**: Speed classification refinement

### Handling Partial Correctness

**Example Scenario:**
```
GT: {"Pan_left", "Dolly In"}
Pred: {"Pan_left"}

Naive F1: TP=1, FP=0, FN=1
         Precision = 1/1 = 1.0
         Recall = 1/2 = 0.5
         F1 = 2*1.0*0.5 / 1.5 = 0.67
```

**Options for GRPO:**
1. **Use raw F1 directly**: Gives F1=0.67 reward
2. **Decompose by label**: 
   - Pan_left: TP (reward +1)
   - Dolly In: FN (reward -0.5)
3. **Use confidence weighting**:
   - If pred has low confidence → reduced penalty
   - If pred has high confidence → full penalty for FN

### Direction Penalty Strategy

```python
def direction_aware_penalty(gt_label, pred_label):
    """
    Different penalties for direction errors vs type errors
    """
    if gt_label == pred_label:
        return 1.0  # Correct
    
    gt_type, gt_dir = parse_label(gt_label)
    pred_type, pred_dir = parse_label(pred_label)
    
    if gt_type != pred_type:
        return 0.0   # Type completely wrong → no credit
    
    if gt_type in DIRECTIONAL_TYPES and gt_dir != pred_dir:
        return 0.3   # Wrong direction → partial credit (e.g., 30%)
    
    return 0.0
```

---

## 6. Normalization & Edge Cases

### Type Normalization (from `evaluate_camera_movement_fixed.py`)

Model outputs sometimes use different conventions:
```python
TYPE_NORMALIZATION = {
    'Truck Left':      ('Truck', 'left'),      # Model uses "Truck Left"
    'Truck Right':     ('Truck', 'right'),     # GT uses type="Truck", dir="left/right"
    'Pedestal Up':     ('Crane', 'up'),        # Model uses "Pedestal" naming
    'Pedestal Down':   ('Crane', 'down'),      # GT uses "Crane" (standard)
}
```

**Action for GRPO:**
- Apply normalization BEFORE computing metrics
- Ensures fair comparison between model outputs and GT

### Null Handling

```python
def normalize_null(val):
    """Unify null representations"""
    if isinstance(val, str):
        if val.strip() in ('', 'null', 'Null', 'NULL'):
            return None
    return val
```

**Action:**
- Parse model output with this function
- Handle empty strings, "null" strings, None, etc.

### Speed Classification (4 levels)

```python
VALID_SPEEDS = {"zero", "slow", "medium", "fast"}

# Static-specific:
# - "zero": completely static
# - "slow": has imperceptible micro-jitter

# Others:
# - "slow": can observe movement but subtle
# - "medium": clear, perceptible movement
# - "fast": rapid, strong visual impact
```

---

## 7. Example Frame-Level Scoring

```
Time: 1.5s

GT Segment:
  basic_movement: [
    {"type": "Pan", "direction": "left", "speed": "medium"},
    {"type": "Dolly In", "direction": null, "speed": "slow"}
  ]
  special_movement: ["Handheld"]

GT Labels: {"Pan_left", "Dolly In"}

---

Pred Segment:
  basic_movement: [
    {"type": "Pan", "direction": "right", "speed": "medium"}  # WRONG direction!
  ]
  special_movement: []

Pred Labels: {"Pan_right"}

---

Scoring:
  TP = {"Pan_left"} ∩ {"Pan_right"} = {}                          (0)
  FP = {"Pan_right"} - {"Pan_left", "Dolly In"} = {"Pan_right"}  (1)
  FN = {"Pan_left", "Dolly In"} - {"Pan_right"} = {"Pan_left", "Dolly In"}  (2)
  
  Precision = 0 / (0 + 1) = 0.0
  Recall = 0 / (0 + 2) = 0.0
  F1 = 0.0
  
  Frame reward = 0.0 (harsh but fair - completely wrong prediction)
```

---

## 8. Summary for GRPO Implementation

### Input Expected
- **Format**: Same as GT (video_id + segments array)
- **Fields**: Must include type, direction (if applicable), start_time, end_time
- **Validation**: Segments must be ordered chronologically

### Evaluation Pipeline
```
Raw Model Output
    ↓
Postprocess (JSON parsing + normalization)
    ↓
Normalize types (Truck Left → Truck, left)
    ↓
Extract composite labels (Pan + left → Pan_left)
    ↓
Frame-level sampling (every 0.1s)
    ↓
Compute F1: TP, FP, FN
    ↓
Compute multiple metrics (composite, type-only, segment, speed)
    ↓
Aggregate into reward signal
```

### Recommended Reward Function Weights

```python
reward = (
    0.50 * f1_composite_labels +           # Core: type+direction
    0.15 * f1_segment_matching +           # Temporal grounding
    0.10 * f1_type_only +                  # Robustness check
    0.15 * f1_special_movement +           # Auxiliary signal
    0.10 * speed_accuracy                  # Fine classification
)
```

### Critical Considerations for Reward Design

1. **Balanced Training**: Weight underrepresented movements (Arc, Free Fly) higher
2. **Direction Sensitivity**: Consider 0.3 multiplier for direction errors vs type errors
3. **Temporal Coherence**: Include segment-level metrics to avoid arbitrary segmentation
4. **Confidence Integration**: Use model's confidence field if available
5. **Frame-level Sampling**: Sample at 0.1s intervals for computational efficiency
6. **Exclude Slow GT**: May want `exclude_slow_gt=True` to focus on clearer movements

