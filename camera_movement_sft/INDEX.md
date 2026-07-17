# Camera Movement Evaluation & GRPO Reward Design - Document Index

## 📚 Complete Documentation Package

All files are located in `/group/40009/dazhaodu/ms-swift/camera_movement_sft/`

### Core Documents

| File | Size | Purpose | Audience |
|------|------|---------|----------|
| **README_GRPO_REWARD_DESIGN.md** | 8 KB | Start here! Overview and quick start guide | Everyone |
| **QUICK_REFERENCE.md** | 7.9 KB | Cheat sheets and lookup tables | Implementers |
| **CAMERA_MOVEMENT_EVALUATION_GUIDE.md** | 16 KB | Deep technical reference | Technical leads |
| **EVALUATION_PIPELINE_DIAGRAM.md** | 8 KB | Visual explanations and examples | Visual learners |
| **eval/reward_function_templates.py** | 6 KB | Production-ready Python code | Engineers |

### Total Package Size: ~46 KB (highly condensed, zero padding)

---

## 🗺️ Reading Paths

### Path 1: Quick Implementation (2 hours)
```
1. README_GRPO_REWARD_DESIGN.md (Quick Start section)
2. eval/reward_function_templates.py (copy Option 5: Composite Reward)
3. QUICK_REFERENCE.md (bookmark for reference)
4. Start coding!
```

### Path 2: Deep Understanding (4 hours)
```
1. README_GRPO_REWARD_DESIGN.md (full read)
2. QUICK_REFERENCE.md (learn all 15 types + composite labels)
3. EVALUATION_PIPELINE_DIAGRAM.md (visual understanding)
4. CAMERA_MOVEMENT_EVALUATION_GUIDE.md (comprehensive technical reference)
5. eval/reward_function_templates.py (code walkthrough)
6. Implement with full confidence
```

### Path 3: Architecture Review (30 mins)
```
1. EVALUATION_PIPELINE_DIAGRAM.md (data flow diagram)
2. QUICK_REFERENCE.md (evaluation metrics hierarchy)
3. README_GRPO_REWARD_DESIGN.md (summary section)
```

---

## 🎯 Key Concepts at a Glance

### The 15 Camera Movement Types
See **QUICK_REFERENCE.md** Table 1

### Composite Label System
- **21 total labels** combining type + direction
- **12 directional**: Pan_left, Pan_right, Tilt_up, Tilt_down, etc.
- **9 non-directional**: Static, Dolly In, Zoom In, etc.
- See **QUICK_REFERENCE.md** "Composite Label System"

### Scoring Formula (Frame-Level)
```
For each sampled time point:

gt_labels = extract_composite_labels(gt_segment)
pred_labels = extract_composite_labels(pred_segment)

TP = |gt_labels ∩ pred_labels|
FP = |pred_labels - gt_labels|
FN = |gt_labels - pred_labels|

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * P * R / (P + R)
```
See **QUICK_REFERENCE.md** "Scoring at Each Time Point"

### Recommended Reward Function
```python
reward = (
    0.50 * f1_composite_labels +      # Type + direction (main)
    0.15 * f1_segment_matching +      # Temporal grounding
    0.10 * f1_type_only +             # Direction robustness
    0.15 * f1_special_movement +      # Auxiliary signal
    0.10 * speed_accuracy             # Speed classification
)
```
See **README_GRPO_REWARD_DESIGN.md** "Step 2: Choose Your Reward Function"

---

## 🔧 Quick Implementation Template

```python
# 1. Extract composite labels
from eval.reward_function_templates import get_composite_labels

# 2. Compute frame-level F1
gt_labels = get_composite_labels(gt_segment)
pred_labels = get_composite_labels(pred_segment)
tp = len(gt_labels & pred_labels)
fp = len(pred_labels - gt_labels)
fn = len(gt_labels - pred_labels)
f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

# 3. Aggregate with other signals
from eval.reward_function_templates import reward_composite
reward = reward_composite(pred_segments, gt_segments)

# 4. Use in GRPO training
loss = -reward  # or feed to PPO, etc.
```

See **eval/reward_function_templates.py** for complete implementations

---

## 📊 Evaluation Hierarchy

```
Frame-Level (Primary Signal)
├─ Composite Label F1: Type + direction accuracy
├─ Per-Label F1: Individual movement type tracking
└─ Sampled every 0.1s

Segment-Level (Temporal Coherence)
├─ IoU Matching: Greedy algorithm
├─ Segment P/R/F1: Boundary accuracy
└─ Threshold: IoU > 0.5

Attribute-Level (Auxiliary)
├─ Speed Accuracy: {zero, slow, medium, fast}
├─ Special Movement F1: Auxiliary techniques
└─ Type-Only F1: Direction robustness check
```

See **QUICK_REFERENCE.md** "Evaluation Metrics Hierarchy"
See **EVALUATION_PIPELINE_DIAGRAM.md** "Data Flow Diagram"

---

## ✅ Validation Checklist

Before implementing, verify you understand:

- [ ] 15 movement types (see **QUICK_REFERENCE.md** Table)
- [ ] Which types require direction (6 types: Pan, Tilt, Truck, Crane, Arc, Roll)
- [ ] Composite label format (e.g., "Pan_left" not "Pan")
- [ ] Frame-level scoring (multi-label set comparison)
- [ ] Segment-level IoU matching (greedy algorithm)
- [ ] Type normalization quirks ("Truck Left" → "Truck" + "left")
- [ ] Output JSON structure (video_id + segments array)
- [ ] Recommended reward weights (0.50/0.15/0.10/0.15/0.10)

---

## 🔍 Finding Specific Information

**"How do I compute F1?"**
→ QUICK_REFERENCE.md: "Scoring at Each Time Point"

**"What's the complete evaluation pipeline?"**
→ EVALUATION_PIPELINE_DIAGRAM.md: "Data Flow Diagram"

**"Which reward function should I use?"**
→ README_GRPO_REWARD_DESIGN.md: "Step 2: Choose Your Reward Function"

**"What are the 15 movement types?"**
→ QUICK_REFERENCE.md: Table 1, "The 15 Camera Movement Types"

**"How do I implement composite labels?"**
→ eval/reward_function_templates.py: `get_composite_labels()`

**"What's a good reward function?"**
→ eval/reward_function_templates.py: `reward_composite()` (Option 5)

**"How do I match segments?"**
→ EVALUATION_PIPELINE_DIAGRAM.md: "Segment-Level IoU Matching"

**"What makes a good vs bad prediction?"**
→ CAMERA_MOVEMENT_EVALUATION_GUIDE.md: Section 4, "What Makes a Good vs Bad Prediction"

**"How do I handle rare movements?"**
→ eval/reward_function_templates.py: `reward_class_balanced()` (Option 7)

---

## 📝 Version History

- **v1.0** (2025-05-25): Initial complete documentation package
  - 5 comprehensive documents
  - 7 reward function implementations
  - Full validation checklist
  - Ready for GRPO implementation

---

## 🚀 Getting Started

1. **First time?** Read **README_GRPO_REWARD_DESIGN.md** (5 min)
2. **Need details?** Read **QUICK_REFERENCE.md** (10 min)
3. **Want visuals?** Read **EVALUATION_PIPELINE_DIAGRAM.md** (15 min)
4. **Need code?** Copy from **eval/reward_function_templates.py**
5. **Going deep?** Read **CAMERA_MOVEMENT_EVALUATION_GUIDE.md** (30 min)

**Total onboarding time: ~1 hour for basic implementation, 2-3 hours for mastery**

---

## 📞 Support

For questions on specific aspects:

| Question | Answer Location |
|----------|-----------------|
| How to compute metrics? | CAMERA_MOVEMENT_EVALUATION_GUIDE.md § 3 |
| How to parse model output? | CAMERA_MOVEMENT_EVALUATION_GUIDE.md § 3 Output Parsing |
| How to normalize types? | CAMERA_MOVEMENT_EVALUATION_GUIDE.md § 6 |
| How to design reward? | README_GRPO_REWARD_DESIGN.md § 🎯 Quick Start |
| How to implement labels? | eval/reward_function_templates.py line 10 |
| How to aggregate F1? | README_GRPO_REWARD_DESIGN.md § Step 5 |

---

## 📌 Pro Tips

1. **Start simple**: Use `reward_basic_f1()` first, add complexity later
2. **Monitor per-label**: Track F1 separately for each composite label
3. **Class weighting**: Use 1.5x weight for Arc, Free Fly (rare)
4. **Temporal coherence**: Always include segment-level IoU in reward
5. **Direction matters**: Wrong direction = full penalty (or 0.3x if lenient)
6. **Test on small set**: Validate on 10-20 videos before full training
7. **Watch for distribution drift**: Ensure rewards remain in [0, 1] range

