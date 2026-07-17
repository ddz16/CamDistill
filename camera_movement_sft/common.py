#!/usr/bin/env python3
"""
运镜训练数据处理公共模块。

包含:
    - System Prompt / User Prompt（训练和推理共用）
    - 标签标准化映射和函数
    - 闭集定义
"""

from typing import Any, Dict, List, Optional

# ============================================================
# 闭集定义
# ============================================================

# prompt 闭集 - 基础运镜
VALID_BASIC_TYPES = {
    "Static", "Unstable", "Dolly In", "Dolly Out", "Truck", "Crane",
    "Follow", "Arc", "Free Fly", "Pan", "Tilt", "Roll",
    "Zoom In", "Zoom Out", "Focus Shift",
}

# prompt 闭集 - 特殊技法
VALID_SPECIAL_TYPES = {
    "Handheld", "Steadicam", "Shaky", "360 Orbit", "Dive", "POV",
    "FPV", "Aerial", "Whip Pan", "Dolly Zoom", "Slow Shutter",
    "Bullet Time", "Car Grip", "SnorriCam", "Top-down Shot",
}

VALID_DIRECTIONS = {"left", "right", "up", "down", "clockwise", "counterclockwise"}
VALID_SPEEDS = {"zero", "slow", "medium", "fast"}

# 需要方向的基础运镜
DIRECTION_REQUIRED = {"Truck", "Crane", "Pan", "Tilt", "Arc", "Roll"}
# 方向必须为 null 的基础运镜
DIRECTION_FORBIDDEN = {
    "Static", "Unstable", "Dolly In", "Dolly Out", "Follow",
    "Free Fly", "Zoom In", "Zoom Out", "Focus Shift",
}

# ============================================================
# 标签标准化映射（将训练数据标签对齐到 prompt 闭集定义）
# ============================================================

# 原始数据中带方向后缀的标签 → (标准type, direction)
MOVEMENT_NORMALIZE_MAP = {
    "Pedestal Down": ("Crane", "down"),
    "Pedestal Up": ("Crane", "up"),
    "Truck Left": ("Truck", "left"),
    "Truck Right": ("Truck", "right"),
}

# 默认方向（当需要方向但缺失时使用）
DEFAULT_DIRECTIONS = {
    "Truck": "right",
    "Crane": "down",
    "Pan": "right",
    "Tilt": "up",
    "Arc": "clockwise",
    "Roll": "clockwise",
}

# 特殊技法名称标准化映射（小写 → 标准名称）
# 用于修正标注数据中的大小写不一致问题，如 "Top-down shot" → "Top-down Shot"
_SPECIAL_NORMALIZE_MAP = {s.lower(): s for s in VALID_SPECIAL_TYPES}


# ============================================================
# System Prompt（训练和推理共用，修改后所有脚本自动同步）
#
# 语言开关: 环境变量 CAMERA_PROMPT_LANG = en(默认) | zh
#   - SYSTEM_PROMPT / USER_PROMPT 按该开关自动指向 zh 或 en 版本
#   - 数据再生脚本可直接 import SYSTEM_PROMPT_EN / USER_PROMPT_EN
#   注意: 枚举值(Static/Zoom In/...)、方向/速度、JSON 字段名两版完全一致, 只译自然语言说明。
# ============================================================
import os

SYSTEM_PROMPT_ZH = """你是一位资深影视摄影指导。观看视频后，判断这段视频由哪些运镜组成，定位时间段，输出结构化 JSON。

**核心原则：只判断镜头（相机）本身的运动，不是画面内物体的运动。** 画面中人在走、车在开，不代表镜头在动——请观察**背景和画面边缘**是否在移动。

## 基础运镜 basic_movement（必填，数组，1~3 个）
每个元素：{"type": "...", "direction": ..., "speed": "..."}

### type 闭集与专业鉴别要点

**静止与非稳态：**
- `Static`（固定镜头）：机位与镜头朝向基本不变，允许幅度极小、几乎看不见的微晃。**与 Unstable 的界限：晃动是否能被明显感受到。几乎看不见 → Static；明显感受到 → Unstable。** speed 完全静止时填 zero，存在极微小晃动时填 slow
- `Unstable`（不稳定镜头）：背景有**可以明显感受到的**无规律晃动，**没有**可归类的稳定运动方向。如果能识别出明确的持续方向（如背景整体向左扫过），应归类为对应运镜（如 Pan）+ Handheld/Shaky

**水平运动鉴别：**
- `Pan`（摇镜头）：机位固定，镜头水平旋转。特征：全景等速平移，**前后景移速一致**（无深度视差），主体大小不变。**区分技巧**：如果只能看到单一深度的背景，无法判断视差时，优先选 Pan
- `Truck`（移镜头）：机位水平侧移。以画面（相机朝向）为参考系，left 表示画面向左移动，right 表示画面向右移动。特征：产生明显的**深度视差**（前景快，后景慢）

**垂直运动鉴别：**
- `Tilt`（俯仰镜头）：机位固定，镜头垂直旋转。特征：垂直平移，**不伴随透视变化**。**区分技巧**：如果只是画面上下平移，选 Tilt
- `Crane`（升降镜头）：机位垂直升降。以画面（相机朝向）为参考系，up 表示画面向上移动，down 表示画面向下移动。特征：地平线高度改变，**伴随透视变化**（俯仰角度变化）。**区分技巧**：如果画面有明显的俯视/仰视角度变化，选 Crane

**纵深运动鉴别：**
- `Dolly In`（推镜头）：机位物理前移。特征：近处物体比远处**膨胀更快**（透视变化），有视差
- `Dolly Out`（拉镜头）：机位物理后移。特征：近处物体比远处**收缩更快**
- `Zoom In`（变焦推进）：焦距变化，机位不动。特征：画面**等比缩放**，主体和背景**同步**放大，无视差
- `Zoom Out`（变焦拉远）：特征：画面等比缩小，所有元素**同比例**缩小
- **区分技巧**：如果近远景放大/缩小速度不同（有视差）→ Dolly；如果所有元素等比缩放 → Zoom

**其他空间运动：**
- `Roll`（旋转镜头）：镜头绕光轴旋转，地平线持续倾斜。即使倾斜幅度不大，只要地平线角度在持续变化就是 Roll
- `Arc`（环绕镜头）：机位绕主体弧形运动，主体居中但视角连续改变。方向以画面判断：主体左侧背景逐渐露出更多 → clockwise
- `Follow`（跟镜头）：机位伴随移动主体，**必要条件：背景在连续变化**。门槛高：优先标具体运镜如 Truck
- `Free Fly`（自由穿越）：三维自由穿行，路径复杂无法归入其他类型。几乎只在 FPV 无人机穿越中出现
- `Focus Shift`（焦点转移）：焦点在不同景深层之间虚实切换

**注意**：Arc 和 Follow 必须同时标注环绕/跟随过程中涉及的基础运镜（如 Truck、Pan、Dolly In 等）

### direction 规则
- 必填方向：Truck→left/right, Crane→up/down, Pan→left/right, Tilt→up/down, Arc→clockwise/counterclockwise, Roll→clockwise/counterclockwise
- 其余 type 的 direction 填 null

### speed 规则
- Static 完全静止→`zero`；Static 有极微小晃动→`slow`；其余→`slow`/`medium`/`fast`
- slow：需仔细看才能确认在动 | medium：清晰感知在动 | fast：快速更替，有速度冲击感

### 复合运镜
当多种运动**同时进行且都能观察到**时填 2-3 个。例如：镜头在向前推进（Dolly In）的同时在水平旋转（Pan）

## 特殊技法 special_movement（可选，0~3 个）
闭集：Handheld（手持）, Steadicam（斯坦尼康）, Shaky（晃动）, 360 Orbit（360环绕）, Dive（俯冲）, POV（主观视角）, FPV（第一人称穿越）, Aerial（航拍）, Whip Pan（甩镜头）, Dolly Zoom（希区柯克变焦）, Slow Shutter（慢门）, Bullet Time（子弹时间）, Car Grip（载具挂拍）, SnorriCam（身体锁定）, Top-down Shot（上帝视角）

规则：
- Handheld/Steadicam/Shaky 三者互斥
- Static 时特殊技法**不可包含** Shaky（允许无特殊技法或 Handheld）
- Unstable 时特殊技法**必须包含** Handheld 或 Shaky（二选一）
- FPV/POV/Aerial/Car Grip/SnorriCam 是修饰层，不替代主运镜

## 输出 JSON 格式（示例，实际根据视频内容输出）
```json
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.5,
      "visual_evidence": "背景建筑群向左匀速平移，前后景移速一致，有轻微晃动",
      "basic_movement": [
        {"type": "Pan", "direction": "right", "speed": "medium"}
      ],
      "special_movement": ["Handheld"],
      "confidence": "high"
    }
  ]
}
```

说明：
- segments 数组包含视频中**所有运镜片段**，按时间顺序排列，数量根据实际视频内容确定
- confidence: high（证据清楚）/ medium（有不确定性）/ low（证据弱）

只输出 JSON，不要输出任何其他内容。"""

USER_PROMPT_ZH = "请分析这段视频的运镜，按 system prompt 规则输出 JSON。只输出 JSON。"


# 英文版（意思与中文版完全一致；枚举值 / 方向 / 速度 / JSON 字段名保持不变）
SYSTEM_PROMPT_EN = """You are a senior film cinematographer. After watching the video, determine which camera movements make up this video, locate their time spans, and output structured JSON.

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


# 按环境变量选择语言 (默认 en)。所有 import SYSTEM_PROMPT/USER_PROMPT 的脚本自动跟随。
_PROMPT_LANG = os.environ.get("CAMERA_PROMPT_LANG", "en").strip().lower()
if _PROMPT_LANG == "zh":
    SYSTEM_PROMPT = SYSTEM_PROMPT_ZH
    USER_PROMPT = USER_PROMPT_ZH
else:
    SYSTEM_PROMPT = SYSTEM_PROMPT_EN
    USER_PROMPT = USER_PROMPT_EN


# ============================================================
# 标签标准化函数
# ============================================================

def normalize_basic_movement(bm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """将原始标签中的运镜标准化为闭集格式。
    
    Args:
        bm: 原始 basic_movement 字典，包含 type, direction, speed
        
    Returns:
        标准化后的字典，如果类型不在闭集中则返回 None
    """
    raw_type = bm.get("type", "")
    direction = bm.get("direction")
    speed = bm.get("speed", "medium")

    # 映射带方向后缀的标签
    if raw_type in MOVEMENT_NORMALIZE_MAP:
        std_type, std_direction = MOVEMENT_NORMALIZE_MAP[raw_type]
        # 如果原始数据有 direction 且有效，优先使用原始 direction
        if direction and direction in VALID_DIRECTIONS:
            pass
        else:
            direction = std_direction
        raw_type = std_type

    # 跳过不在闭集中的类型
    if raw_type not in VALID_BASIC_TYPES:
        return None

    # 修正 direction 规则
    if raw_type in DIRECTION_REQUIRED:
        if not direction or direction not in VALID_DIRECTIONS:
            direction = DEFAULT_DIRECTIONS.get(raw_type, "right")
    elif raw_type in DIRECTION_FORBIDDEN:
        direction = None

    # 修正 speed 规则
    if raw_type == "Static":
        # Static 允许 zero（完全静止）或 slow（极微小晃动）
        if speed not in {"zero", "slow"}:
            speed = "zero"
    elif speed not in VALID_SPEEDS or speed == "zero":
        speed = "medium"

    return {"type": raw_type, "direction": direction, "speed": speed}


def normalize_special_movements(specials: List[str]) -> List[str]:
    """过滤只保留闭集内的特殊技法，同时修正大小写不一致。
    
    例如: "Top-down shot" → "Top-down Shot"
    
    Args:
        specials: 原始特殊技法列表
        
    Returns:
        过滤并标准化后只包含有效技法的列表
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
            # 尝试通过小写映射修正大小写不一致
            normalized = _SPECIAL_NORMALIZE_MAP.get(s.lower())
            if normalized:
                result.append(normalized)
    return result


def normalize_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """标准化 segments 列表，使其符合 prompt 闭集定义。
    
    Args:
        segments: 原始 segments 列表
        
    Returns:
        {"segments": [...]} 格式的标准化结果
    """
    new_segments = []
    for seg in segments:
        new_basics = []
        for bm in seg.get("basic_movement", []):
            normalized = normalize_basic_movement(bm)
            if normalized:
                new_basics.append(normalized)
        if not new_basics:
            # 如果所有运镜都被过滤掉了，跳过此段
            continue

        # 去重（同一 segment 内不应有重复 type）
        seen_types = set()
        deduped_basics = []
        for bm in new_basics:
            if bm["type"] not in seen_types:
                seen_types.add(bm["type"])
                deduped_basics.append(bm)

        # 限制最多 3 个
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

    # 对齐相邻 segments 的时间：前一个 end_time = 后一个 start_time
    for i in range(len(new_segments) - 1):
        next_start = new_segments[i + 1]["start_time"]
        new_segments[i]["end_time"] = next_start

    return {"segments": new_segments}


def normalize_label_with_nested_segments(label: Dict[str, Any]) -> Dict[str, Any]:
    """标准化嵌套格式的 label（SpatialVID 数据格式）。
    
    输入格式: {"segments": [...]}
    
    Args:
        label: 包含 segments 键的字典
        
    Returns:
        标准化后的 {"segments": [...]}
    """
    return normalize_segments(label.get("segments", []))
