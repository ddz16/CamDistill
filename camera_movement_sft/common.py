#!/usr/bin/env python3
"""
Common module for camera-movement training data processing.

Contains:
    - System Prompt / User Prompt (shared between training and inference)
    - Label normalisation maps and functions
    - Closed-set definitions
"""

from typing import Any, Dict, List, Optional

# ============================================================
# Closed-set definitions
# ============================================================

# Prompt closed set - basic movements
VALID_BASIC_TYPES = {
    "Static", "Unstable", "Dolly In", "Dolly Out", "Truck", "Crane",
    "Follow", "Arc", "Pan", "Tilt", "Roll",
    "Zoom In", "Zoom Out", "Focus Shift",
}

# Prompt closed set - special techniques
VALID_SPECIAL_TYPES = {
    "Handheld", "Steadicam", "Shaky", "360 Orbit", "Dive", "POV",
    "FPV", "Aerial", "Whip Pan", "Dolly Zoom", "Slow Shutter",
    "Bullet Time", "Car Grip", "SnorriCam", "Top-down Shot",
}

VALID_DIRECTIONS = {"left", "right", "up", "down", "clockwise", "counterclockwise"}
VALID_SPEEDS = {"zero", "slow", "medium", "fast"}

# Basic movements that require a direction.
DIRECTION_REQUIRED = {"Truck", "Crane", "Pan", "Tilt", "Arc", "Roll"}
# Basic movements whose direction must be null.
DIRECTION_FORBIDDEN = {
    "Static", "Unstable", "Dolly In", "Dolly Out", "Follow",
    "Free Fly", "Zoom In", "Zoom Out", "Focus Shift",
}


# ============================================================
# Label normalisation maps (align training-data labels to the prompt closed set)
# ============================================================

# Labels with a direction suffix in the raw data -> (canonical type, direction)
MOVEMENT_NORMALIZE_MAP = {
    "Pedestal Down": ("Crane", "down"),
    "Pedestal Up": ("Crane", "up"),
    "Truck Left": ("Truck", "left"),
    "Truck Right": ("Truck", "right"),
}

# Default directions (used when a direction is required but missing)
DEFAULT_DIRECTIONS = {
    "Truck": "right",
    "Crane": "down",
    "Pan": "right",
    "Tilt": "up",
    "Arc": "clockwise",
    "Roll": "clockwise",
}

# Special-technique name normalisation map (lowercase -> canonical name).
# Fixes case inconsistencies in annotation data, e.g. "Top-down shot" -> "Top-down Shot".
# ============================================================
# System Prompt (shared between training and inference; all scripts auto-sync on edit)
# ============================================================

SYSTEM_PROMPT_EN = """You are a senior film cinematographer.SYSTEM_PROMPT_EN = """You are a senior film cinematographer. After watching the video, determine which camera movements make up this video, locate their time spans, and output structured JSON.

**Core principle: judge only the motion of the camera (lens) itself, not the motion of objects within the frame.** People walking or cars driving inside the frame does not mean the camera is moving — observe whether the **background and frame edges** are moving.

## Basic movement basic_movement (required, array, 1-3 items)
Each element: {"type": "...", "direction": ..., "speed": "..."}

### type closed set and professional discrimination points

**Static and non-steady:**
- `Static` (fixed shot): camera position and lens orientation essentially unchanged, allowing extremely small, barely visible micro-jitter. **Boundary with Unstable: whether the shake can be clearly felt. Barely visible -> Static; clearly felt -> Unstable.** Fill speed as zero when completely still, and slow when there is extremely slight jitter
- `Unstable` (unstable shot): the background has **clearly perceptible** irregular shaking, with **no** classifiable stable direction of motion. If a clear sustained direction can be identified (e.g. the background sweeps to the left as a whole), it should be classified as the corresponding movement (e.g. Pan) + Handheld/Shaky

**Horizontal motion discrimination:**
- `Pan` (panning): camera position fixed, lens rotates horizontally. Features: uniform panoramic translation, **foreground and background move at the same speed** (no depth parallax), subject size unchanged. **Tip**: if you can only see background at a single depth and cannot judge parallax, prefer Pan
- `Truck` (lateral tracking): camera position moves sideways horizontally. Using the frame (camera facing) as reference, left means the frame moves left, right means the frame moves right. Features: produces obvious **depth parallax** (foreground fast, background slow)

**Vertical motion discrimination:**
- `Tilt` (tilting): camera position fixed, lens rotates vertically. Features: vertical translation, **no accompanying perspective change**. **Tip**: if the frame only translates up/down, choose Tilt
- `Crane` (crane up/down): camera position rises/falls vertically. Using the frame (camera facing) as reference, up means the frame moves up, down means the frame moves down. Features: horizon height changes, **accompanied by perspective change** (pitch angle changes). **Tip**: if there is an obvious change of looking-down/looking-up angle, choose Crane

**Depth motion discrimination:**
- `Dolly In` (push in): camera physically moves forward. Features: nearby objects **expand faster** than distant ones (perspective change), with parallax
- `Dolly Out` (pull back): camera physically moves backward. Features: nearby objects **shrink faster** than distant ones
- `Zoom In` (zoom in): focal length changes, camera does not move. Features: the frame **scales uniformly**, subject and background enlarge **in sync**, no parallax
- `Zoom Out` (zoom out): Features: the frame scales down uniformly, all elements shrink **at the same ratio**
- **Tip**: if near and far scale up/down at different speeds (parallax) -> Dolly; if all elements scale uniformly -> Zoom

**Other spatial motion:**
- `Roll` (rolling): the lens rotates around the optical axis, the horizon keeps tilting. Even if the tilt is small, as long as the horizon angle keeps changing it is Roll
- `Arc` (arcing): camera moves in an arc around the subject, the subject stays centered but the viewing angle changes continuously. Direction judged by the frame: more of the background on the subject's left is gradually revealed -> clockwise
- `Follow` (following): camera moves along with a moving subject, **necessary condition: the background changes continuously**. High bar: prefer labeling a specific movement such as Truck
- `Free Fly` (free traversal): 3D free flight, complex path that cannot be classified into other types. Almost only appears in FPV drone fly-throughs
- `Focus Shift` (focus shift): focus switches between sharp and blurred across different depth layers

**Note**: Arc and Follow must also annotate the basic movements involved during the arcing/following process (such as Truck, Pan, Dolly In, etc.)

### direction rules
- Direction required: Truck->left/right, Crane->up/down, Pan->left/right, Tilt->up/down, Arc->clockwise/counterclockwise, Roll->clockwise/counterclockwise
- For all other types, fill direction as null

### speed rules
- Static completely still -> `zero`; Static with extremely slight jitter -> `slow`; others -> `slow`/`medium`/`fast`
- slow: motion confirmable only on careful inspection | medium: motion clearly perceived | fast: rapid changes, with a sense of speed impact

### Compound movement
When multiple motions **occur simultaneously and are all observable**, fill in 2-3 items. For example: the camera is pushing forward (Dolly In) while rotating horizontally (Pan)

## Special technique special_movement (optional, 0-3 items)
Closed set: Handheld (handheld), Steadicam (Steadicam-stabilized), Shaky (shaky), 360 Orbit (360° orbit around subject), Dive (diving/plunging), POV (subjective point-of-view), FPV (first-person fly-through), Aerial (aerial), Whip Pan (whip pan), Dolly Zoom (Hitchcock zoom), Slow Shutter (slow shutter / long exposure), Bullet Time (bullet time), Car Grip (vehicle-mounted rig), SnorriCam (locked to the body), Top-down Shot (top-down / bird's-eye view)

Rules:
- Handheld/Steadicam/Shaky are mutually exclusive
- When Static, the special technique **must not include** Shaky (allow no special technique, or Handheld)
- When Unstable, the special technique **must include** Handheld or Shaky (choose one)
- FPV/POV/Aerial/Car Grip/SnorriCam are modifier layers, they do not replace the main movement

## Output JSON format (example; actual output depends on the video content)
```json
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.5,
      "visual_evidence": "background buildings translate to the left at constant speed, foreground and background move at the same speed, with slight jitter",
      "basic_movement": [
        {"type": "Pan", "direction": "right", "speed": "medium"}
      ],
      "special_movement": ["Handheld"],
      "confidence": "high"
    }
  ]
}
```

Notes:
- The segments array contains **all camera movement segments** in the video, ordered by time; the count is determined by the actual video content
- confidence: high (clear evidence) / medium (some uncertainty) / low (weak evidence)

Output only JSON, do not output anything else."""

USER_PROMPT_EN = "Analyze the camera movement in this video and output JSON following the system prompt rules. Output only JSON."

SYSTEM_PROMPT = SYSTEM_PROMPT_EN
USER_PROMPT = USER_PROMPT_EN


# ============================================================
# Label normalisation functions
# ============================================================

def normalize_basic_movement(bm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise a raw movement label into the closed-set format.

    Args:
        bm: raw basic_movement dict containing type, direction, speed.

    Returns:
        Normalised dict, or None if the type is not in the closed set.
    """
    raw_type = bm.get("type", "")
    direction = bm.get("direction")
    speed = bm.get("speed", "medium")

    # Map labels that carry a direction suffix.
    if raw_type in MOVEMENT_NORMALIZE_MAP:
        std_type, std_direction = MOVEMENT_NORMALIZE_MAP[raw_type]
        # If the raw data already has a valid direction, keep it.
        if direction and direction in VALID_DIRECTIONS:
            pass
        else:
            direction = std_direction
        raw_type = std_type

    # Skip types that are not in the closed set.
    if raw_type not in VALID_BASIC_TYPES:
        return None

    # Enforce direction rules.
    if raw_type in DIRECTION_REQUIRED:
        if not direction or direction not in VALID_DIRECTIONS:
            direction = DEFAULT_DIRECTIONS.get(raw_type, "right")
    elif raw_type in DIRECTION_FORBIDDEN:
        direction = None

    # Enforce speed rules.
    if raw_type == "Static":
        # Static allows zero (completely still) or slow (extremely slight jitter).
        if speed not in {"zero", "slow"}:
            speed = "zero"
    elif speed not in VALID_SPEEDS or speed == "zero":
        speed = "medium"

    return {"type": raw_type, "direction": direction, "speed": speed}


def normalize_special_movements(specials: List[str]) -> List[str]:
    """Filter to keep only special techniques in the closed set, and fix case inconsistencies.

    Example: "Top-down shot" -> "Top-down Shot"

    Args:
        specials: raw list of special techniques.

    Returns:
        Filtered and normalised list containing only valid techniques.
    """
    if not specials:
        return []
    result = []
    for s in specials:
        if not s:
            continue
        if s in VALID_SPECIAL_TYPES:
            result.append(s)
        else:
            # Try to fix case inconsistencies via the lowercase map.
            normalized = _SPECIAL_NORMALIZE_MAP.get(s.lower())
            if normalized:
                result.append(normalized)
    return result


def normalize_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalise a segments list so it conforms to the prompt closed-set definition.

    Args:
        segments: raw segments list.

    Returns:
        Normalised result in {"segments": [...]} format.
    """
    new_segments = []
    for seg in segments:
        new_basics = []
        for bm in seg.get("basic_movement", []):
            normalized = normalize_basic_movement(bm)
            if normalized:
                new_basics.append(normalized)
        if not new_basics:
            # Skip this segment if all movements were filtered out.
            continue

        # Deduplicate (a single segment should not contain duplicate types).
        seen_types = set()
        deduped_basics = []
        for bm in new_basics:
            if bm["type"] not in seen_types:
                seen_types.add(bm["type"])
                deduped_basics.append(bm)

        # Cap at 3 items.
        deduped_basics = deduped_basics[:3]

        new_seg = {
            "start_time": round(float(seg.get("start_time", 0.0)), 1),
            "end_time": round(float(seg.get("end_time", 0.0)), 1),
            "visual_evidence": seg.get("visual_evidence", ""),
            "basic_movement": deduped_basics,
            "special_movement": normalize_special_movements(seg.get("special_movement", []))[:3],
            "confidence": seg.get("confidence", "high") if seg.get("confidence") else "high",
        }
        new_segments.append(new_seg)

    # Align adjacent segment times: previous end_time = next start_time.
    for i in range(len(new_segments) - 1):
        next_start = new_segments[i + 1]["start_time"]
        new_segments[i]["end_time"] = next_start

    return {"segments": new_segments}


def normalize_label_with_nested_segments(label: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a label in the nested-segments format (SpatialVID data format).

    Input format: {"segments": [...]}

    Args:
        label: dict containing a "segments" key.

    Returns:
        Normalised {"segments": [...]}.
    """
    return normalize_segments(label.get("segments", []))
