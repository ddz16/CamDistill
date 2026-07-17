# Camera Movement Understanding — Evaluation Metrics

This document defines the evaluation protocol of the benchmark. It is written to be
directly reusable in a paper's *Evaluation Metrics* section. All formulas are standard;
the two novel design points are (i) treating **camera movement as a multi-label temporal
labeling problem** and (ii) reporting **two orthogonal axes** — dense *frame-level*
recognition and *segment-level* localization/detection.

> **中文速览**:评测分两条轴。**Frame-level**(逐 0.1s 采样)把定位与识别耦合在一起,报
> multi-label 的 **micro / macro P/R/F1**;**Segment-level**(按时间段)解耦成两把尺子——
> **Localization (Loc)** 只看时间段切得准不准(标签无关),**Detection (Det)** 要求时间对
> *且* 运镜标签集完全一致(mAP 式)。Segment 指标在 IoU ∈ {0.3, 0.5, 0.7} 各报一次。

---

## 1. Notation & Problem Setup

A video is annotated as a sequence of non-overlapping temporal **segments**. Each segment
`s = (t^start, t^end, L)` carries a **set of camera-movement labels** `L` (a segment may
contain multiple simultaneous movements, e.g. a pan while dollying in).

**Compound label (type + direction).** The benchmark defines **15 basic movement types**.
The 6 directional types (`Pan, Tilt, Truck, Crane, Arc, Roll`) are encoded jointly with
their direction, e.g. `Pan_left`, `Crane_up`; the 9 non-directional types
(`Static, Unstable, Dolly In, Dolly Out, Follow, Free Fly, Zoom In, Zoom Out, Focus Shift`)
use the type alone. This yields the compound label space

```
|C| = 9 (non-directional) + 6 (directional) × 2 (directions) = 21 classes.
```

Evaluating on the compound label folds *direction correctness* into the recognition
metrics, so a type-correct / direction-wrong prediction is **not** rewarded (it incurs both
a false positive on the wrong compound label and a false negative on the correct one).
Both micro and macro P/R/F1 are computed over this **same 21-class space**. A secondary,
direction-agnostic variant collapses each compound label to its type, giving a **15-class**
type-only space (reported as `basic_movement_type_only`) for isolating direction errors.

We evaluate a prediction against ground truth (GT) over the set of videos common to both.

---

## 2. Frame-Level Metrics (dense recognition)

The timeline of each video is sampled at a fixed step `Δ = 0.1 s`. At each sampled
timestamp `t`, let `G_t ⊆ C` and `P_t ⊆ C` be the GT and predicted compound-label sets
of the segment covering `t` (empty if none). Frames where both are empty are skipped.

This axis is *dense* and **entangles localization with recognition** (a boundary error
shifts many frames' labels), analogous to frame-wise accuracy in temporal action
segmentation.

### 2.1 Multi-label counts

Accumulated over all sampled frames `t`:

```
TP = Σ_t |G_t ∩ P_t|
FP = Σ_t |P_t \ G_t|
FN = Σ_t |G_t \ P_t|
```

### 2.2 Micro-averaged P / R / F1

Micro aggregation pools counts over all label instances (dominated by frequent classes):

```
P_micro = TP / (TP + FP)
R_micro = TP / (TP + FN)
F1_micro = 2·P_micro·R_micro / (P_micro + R_micro)
```

### 2.3 Macro-averaged P / R / F1

Macro treats every movement class equally (reveals rare classes such as `Roll`, `Arc`).
For each class `c ∈ C` compute per-class counts `TP_c, FP_c, FN_c` (a frame contributes to
class `c` as TP if `c ∈ G_t ∩ P_t`, FP if `c ∈ P_t \ G_t`, FN if `c ∈ G_t \ P_t`), then:

```
P_c = TP_c / (TP_c + FP_c),   R_c = TP_c / (TP_c + FN_c),   F1_c = 2·P_c·R_c / (P_c + R_c)

P_macro  = (1/|C|) Σ_c P_c
R_macro  = (1/|C|) Σ_c R_c
F1_macro = (1/|C|) Σ_c F1_c
```

The class set `C` is the fixed **21-class** type+direction vocabulary defined in §1 (all 21
occur in the benchmark GT, so the macro denominator is stable and comparable across models).
Micro and macro therefore share the same class space; they differ only in aggregation
(instance-weighted vs class-weighted).

### 2.4 Auxiliary frame-level metrics

- **special_movement**: same multi-label micro/macro P/R/F1 over the special-movement label set.
- **speed**: a *single-label* ordinal attribute (`zero < slow < medium < fast`); reported as
  **accuracy** over frames where both GT and prediction provide a speed. (Optionally an
  adjacent-tolerant accuracy / ordinal MAE.)

---

## 3. Segment-Level Metrics (localization & detection)

This axis operates on whole segments and **decouples** temporal localization from
recognition, mirroring temporal action *localization* vs *detection*.

### 3.1 Temporal IoU and matching

For a GT segment `g` and predicted segment `p`, temporal IoU is

```
IoU(g, p) = |g ∩ p| / |g ∪ p|
          = max(0, min(g^end,p^end) − max(g^start,p^start)) / (|g| + |p| − intersection)
```

Given a threshold `τ`, candidate pairs with `IoU ≥ τ` are matched **greedily in
descending IoU order**, each GT and predicted segment used at most once. Let `N_G`, `N_P`
be the numbers of GT and predicted segments.

### 3.2 Segment Localization — **Loc-F1** (class-agnostic)

A predicted segment is a true positive iff it is matched to some GT segment with
`IoU ≥ τ`, **regardless of labels**. With `M_loc` = number of matched pairs:

```
P_loc = M_loc / N_P,   R_loc = M_loc / N_G,   Loc-F1 = 2·P_loc·R_loc / (P_loc + R_loc)
```

Loc-F1 measures pure temporal segmentation quality (are the boundaries right?).

### 3.3 Segment Detection — **Det-F1** (localization + recognition)

A predicted segment is a true positive iff **both** conditions hold: `IoU ≥ τ` **and** its
compound label set exactly equals the matched GT's, i.e. `L_p = L_g` (exact multi-label set
match on basic_movement type+direction). This is the temporal analogue of detection mAP
(localize *and* classify correctly). With `M_det` = number of such pairs:

```
P_det = M_det / N_P,   R_det = M_det / N_G,   Det-F1 = 2·P_det·R_det / (P_det + R_det)
```

By construction `M_det ≤ M_loc`, hence **Det-F1 ≤ Loc-F1** at every `τ`. Det-F1 is
all-or-nothing per segment (one direction flip or one missing co-occurring movement fails
the whole segment), so it is naturally much lower than the partial-credit frame-level F1;
this is expected, not a regression.

### 3.4 Thresholds

All segment-level metrics are reported at **`τ ∈ {0.3, 0.5, 0.7}`** (no averaging), to
expose the localization-tightness trade-off.

---

## 4. Main Results Table (headline)

Report per model: frame-level multi-label **micro** and **macro** P/R/F1 for camera-movement
recognition (compound type+direction), plus segment-level **Loc-F1** and **Det-F1** at each
IoU threshold. `speed` accuracy and `special_movement` F1 may be added as extra columns.

*(Numbers below are illustrative placeholders.)*

**Frame-level (dense recognition)**

| Model | micro-P | micro-R | micro-F1 | macro-P | macro-R | macro-F1 |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B (zero-shot) | 0.512 | 0.437 | 0.471 | 0.401 | 0.352 | 0.375 |
| + CamDistill SFT (ours)  | 0.723 | 0.615 | **0.665** | 0.625 | 0.515 | **0.541** |
| GPT-4o (API)             | 0.588 | 0.502 | 0.541 | 0.489 | 0.430 | 0.458 |

**Segment-level (localization / detection), F1 @ IoU**

| Model | Loc@0.3 | Loc@0.5 | Loc@0.7 | Det@0.3 | Det@0.5 | Det@0.7 |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B (zero-shot) | 0.712 | 0.658 | 0.531 | 0.241 | 0.223 | 0.184 |
| + CamDistill SFT (ours)  | **0.864** | **0.803** | **0.667** | **0.397** | **0.381** | **0.336** |
| GPT-4o (API)             | 0.775 | 0.706 | 0.574 | 0.288 | 0.265 | 0.211 |

> The gap between Loc-F1 (≈0.80) and Det-F1 (≈0.38) is the key diagnostic: the model
> localizes segments well but assigns the exact movement label set correctly only about
> half the time — direction confusion and missed co-occurring movements dominate the
> remaining error.

---

## 5. Diagnostic / Appendix Metrics (optional)

- **Direction cost**: `basic_movement(type-only)` F1 vs `type+direction` F1 — isolates
  direction errors.
- **Recognition given localization**: multi-label F1 computed **only on Loc-matched pairs**
  (`matched_*` in the code). Because it is conditioned on correct localization, it is **not
  comparable across models** with different Loc-F1 and must be reported alongside Loc-F1.
- **Per-class F1**: the macro breakdown, showing which movements are hardest.
- **speed**: confusion matrix / adjacent-tolerant accuracy (ordinal).

---

## 6. Reproducibility

```bash
python evaluate_camera_movement_fixed.py \
    --gt   <benchmark_gt>.jsonl \
    --pred <predictions>.jsonl \
    --iou_thresh 0.3 0.5 0.7 \
    --output results.json
```

Output JSON keys: `frame_level.{basic_movement_with_direction (micro),
basic_movement_with_direction_macro, special_movement, speed}` and
`segment_level.iou_{τ}.{segment_localization, segment_detection}`.
Metrics are deterministic (greedy IoU matching, fixed 0.1 s sampling).
