#!/usr/bin/env python3
"""
人工标注运镜数据 → ms-swift 标准 jsonl 格式 转换脚本。

同时输出三个版本的训练数据:
    1. 全量数据（所有输入文件，shuffle 后）
    2. exclude 后数据（排除指定文件名模式后，shuffle）
    3. exclude 后 + 类别均衡数据（排除 + 上下采样）

用法:
    # 默认: 自适应模式，自动计算采样比例
    python prepare_human_labels.py
    
    # 排除特定文件名模式（默认已排除老版本原始标注）
    python prepare_human_labels.py --exclude_patterns \
        '运镜标注-正式标注0324-1w-*' \
        '正式标注0327-3w-4000条-part*-标注--*'
    
    # 手动指定采样比例（覆盖自适应）
    python prepare_human_labels.py --manual_ratios \\
        --downsample_static_pure 0.30 \\
        --downsample_static_mixed 0.70 \\
        --downsample_pan_pure 0.60 \\
        --downsample_pan_mixed 0.90 \\
        --upsample_follow 2.0 \\
        --upsample_roll 2.5 \\
        --upsample_arc 2.5 \\
        --upsample_zoom_in 3.0 \\
        --upsample_zoom_out 4.0

采样策略（自适应模式，默认，温和最终版 面向 VLM 训练优化）:
    以各类别视频数的"对数均值"为锚点，自动判断哪些类别需要上/下采样:
    - 超过锚点 2x  → 下采样到锚点 3.0x 附近（温和, 头部不腰斩）
    - 低于锚点 0.45x → 上采样到锚点 0.7x 附近（温和, 护稀缺类, 避免 Unstable/Follow 等被误上采）
    - 下采样 pure 最低 55%, mixed 最低 90%
    - 上采样最高 2x（单视频最多复制 1 次，严格避免过拟合）
    - 上采黑名单: Unstable, Follow（视觉/语义特征易过拟合，禁止复制）
    - fast 占比上限 10%（Zoom Out 原始 fast% 高达 13.6%，防过补偿）
    - Static 采用显式特化: pure=0.40, mixed=0.90
    
    结构性改进（解决隐藏问题）:
        1) 一次性下采决策: 每个样本在所有命中的 type 里取"最高保留率", 避免级联双杀
           （旧逻辑一个样本同时属于 Static/Pan/Tilt 会被丢 3 次, 实际保留率 = 0.9³=0.73）
        2) 高价值保护: segment 数 >= 3 或 含组合 segment(同段多 type) 的视频, 下采时强制完全保留
           （保护时序复杂 + 多类别共现这两类对 VLM 极有价值的样本）
        3) 稀缺方向保护: 某方向视频数 < 该类别的 15% 时, 该方向视频全部保留
        4) 上采基数锚定原始数量: 避免"下采误伤 → 上采补偿不足"的反向放大
    
    方向+速度联合均衡:
        - 按(方向, 速度)联合分组, 少数组合优先保留/多重复
        - 多方向样本归入"自身最稀缺"的组
        - 下采样时: fast 样本完全保护, 其余按(方向,速度)均匀保留
        - 上采样时: 按(方向,速度)联合分组, 少数组获得更多重复
        - 适用于有方向属性的类别: Pan, Tilt, Truck, Crane, Arc, Roll
    
    稀疏类别: Follow, Roll, Arc, Zoom In, Zoom Out, Free Fly（混合样本会被保护）

原始数据格式 (人工标注):
    JSONL: 每行一个 {"video_id": "xxx", "cos_url": "...", "segments": [...]}
    segments 中直接包含 basic_movement 和 special_movement

目标格式 (ms-swift 多模态 SFT):
    {"messages": [{"role": "system", "content": "<system_prompt>"},
                  {"role": "user", "content": "<video>请分析这段视频的运镜..."},
                  {"role": "assistant", "content": "<json_output>"}],
     "videos": ["/absolute/path/to/video.mp4"]}

关键处理:
    1. 将原始标签中的 "Pedestal Down/Up" 映射为 "Crane" + direction
    2. 将 "Truck Left/Right" 映射为 "Truck" + direction
    3. 生成符合 prompt 闭集定义的标准化 JSON 输出
    4. 跳过视频文件不存在的样本
    5. 支持类别均衡（上采样稀疏类别 + 下采样头部类别）
"""

import argparse
import fnmatch
import glob
import json
import os
import random
import requests
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# 从公共模块导入
from common import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    normalize_segments,
    normalize_basic_movement,
    VALID_BASIC_TYPES,
    DIRECTION_REQUIRED,
)

# ============================================================
# 类别均衡配置（自适应 + 温和策略，面向 VLM 训练优化）
# ============================================================
# 核心设计原则:
#   1) 温和: 头部不砍狠, 尾部不复制太夸张(<=2.5x), 避免过拟合
#   2) 护中部: 用对数均值做锚点, 避免 Tilt/Truck/Dolly 等中部类别被误伤
#   3) 护 segment 多样性: mixed_floor 回到 0.70, 保护多类别共现样本
#   4) 护稀缺细粒度: fast/少数方向完全保留, 仅温和补充
#   5) 护 Static 特殊性: 显式特化(不经过公式), 因它是 VLM 的默认类&关键负样本

# --- 下采样参数(全自适应: 下采目标 = anchor * DOWN_TARGET_MULT, 数据换了会自适应) ---
ADAPTIVE_DOWN_THRESHOLD = 2.0    # ratio_to_anchor > 2.0 触发下采
ADAPTIVE_DOWN_TARGET_MULT = 2.5  # 下采目标 = anchor * 2.5 (头部更贴近中部, 比 3.0 更均衡)
ADAPTIVE_DOWN_MIN_RATIO = 0.30   # pure 保留下限 30% (放宽, 让 Static/Pan 能真正按比例下降)
ADAPTIVE_DOWN_MIXED_FLOOR = 0.75 # mixed 保留下限 75% (允许头部 Pan/Tilt 的 hv/mixed 样本也被按比例降下来)

# --- 上采样参数 ---
ADAPTIVE_UP_THRESHOLD = 0.45     # ratio_to_anchor < 0.45 触发上采
ADAPTIVE_UP_TARGET_MULT = 0.7    # 上采目标 = anchor * 0.7
ADAPTIVE_UP_MAX_MULT = 2.0       # 上采最高 2x(单视频最多被复制一次)
ADAPTIVE_FAST_CAP = 0.10         # 上采后 fast 占比上限(防 Zoom Out 13.6% fast 被过度复制)

# --- 上采黑名单: 显式禁止上采的类别 ---
# Unstable: 抖动是视觉底层特征, 复制同一视频极易让 VLM 过拟合抖动模式
# (Follow 虽也有语义风险, 但 1872 条本就不在上采触发区间(ratio=0.66>0.45), 无需写入黑名单)
UPSAMPLE_BLACKLIST = {"Unstable"}

# 不参与自适应的类别(数据极少, 质量存疑)
ADAPTIVE_SKIP = {"Free Fly"}

# 稀疏类别集合(混合样本中包含这些类别时会被保护不下采样)
# 完全自适应: 运行时根据 ratio_to_anchor < RARE_TYPE_RATIO 动态判定, 不再写死类别名
# 这样新数据来了, 即使 Zoom In 变多、Dolly Out 变少, 稀疏集合也会自动更新
RARE_TYPE_RATIO = 0.8  # cnt/anchor < 0.8 的类别视为稀疏类(稀疏类共现样本受软保护)

# --- 高价值样本保护阈值 ---
# 视频 segment 数 >= 此值时, 视为"多段时序高价值样本", 下采时强制保留
# 设为 3: 2 段视频太普通(37%), 3 段视频明显更有时序价值(13%)
QUALITY_MIN_SEGMENTS = 3
# 稀缺方向保护: 某方向视频数 < 该类别的此比例时, 视为稀缺方向, 下采完全保留
RARE_DIRECTION_RATIO = 0.15
# 稀有组合保护阈值: 采用"总样本数 × RARE_COMBO_RATIO"自适应计算(数据量越大, 阈值越高)
# 这保证"各类型数据的组合保留"——稀有组合一条都不丢, 且不是写死的魔法数字
RARE_COMBO_RATIO = 0.0006         # 51407 × 0.0006 ≈ 30 (当前数据下与旧阈值等价)
RARE_COMBO_MIN_THRESHOLD = 20     # 即使数据再少, 至少按 20 判稀有


def _compute_anchor(counts: List[int]) -> float:
    """锚点 = 对数均值 (geometric mean).

    相较于中位数, 对数均值:
      - 对极端长尾更鲁棒(不会因头部过多被拉高, 也不会因尾部过多被拉低)
      - 不需要人为剔除某几个类别(如 Static/Pan)
      - 对"头部密集 + 尾部稀疏"的分布给出更合理的中间值
    """
    import math
    if not counts:
        return 0.0
    log_sum = sum(math.log(max(1, c)) for c in counts)
    return math.exp(log_sum / len(counts))


def compute_adaptive_ratios(
    samples: List[Dict[str, Any]],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """基于数据分布自适应计算上下采样比例(温和策略, 面向 VLM 训练优化)。

    策略(全自适应, 无任何写死的类别特化):
    - 锚点 = 各类别视频数的"对数均值"(对长尾鲁棒, 不用人为剔除主导类别)
    - ratio > ADAPTIVE_DOWN_THRESHOLD(2.0) → 下采: pure=max(MIN_RATIO, anchor*DOWN_MULT/cnt)
    - ratio < ADAPTIVE_UP_THRESHOLD(0.45)  → 上采: mult=anchor*UP_MULT/cnt, clamp 到 [1, MAX_MULT]
    - 下采 pure_ratio  clamp 到 [DOWN_MIN_RATIO, 1.0]        (默认 [0.30, 1.0])
    - 下采 mixed_ratio clamp 到 [DOWN_MIXED_FLOOR, 1.0]      (默认 [0.85, 1.0])
    - 上采倍数 clamp 到 [1.0, ADAPTIVE_UP_MAX_MULT]          (默认 [1.0, 2.0], 单视频最多复制 1 次)
    - Unstable 在上采黑名单内(抖动复制极易过拟合底层视觉)
    - Free Fly 在 ADAPTIVE_SKIP(极稀有 + 质量存疑)

    数据换了自动适应:
    - 所有阈值都是"相对于 anchor 的比例", 数据量变化时策略自动缩放
    - 稀有组合阈值 = N * RARE_COMBO_RATIO, 数据越多阈值越高, 仍然正比

    Returns:
        (downsample_ratios, upsample_ratios)
    """
    # 统计每个类别的样本数
    type_counts: Dict[str, int] = Counter()
    for s in samples:
        for bt in get_sample_basic_types(s):
            type_counts[bt] += 1

    # 用对数均值作为锚点(排除极稀有类别)
    counts_for_anchor = [
        cnt for t, cnt in type_counts.items()
        if t not in ADAPTIVE_SKIP and cnt > 0
    ]
    if not counts_for_anchor:
        return {}, {}

    anchor = _compute_anchor(counts_for_anchor)
    print(f"\n[自适应] 锚点(对数均值) = {anchor:.0f}  (参与计算类别数: {len(counts_for_anchor)})")
    print(f"[自适应] 下采目标 = anchor * {ADAPTIVE_DOWN_TARGET_MULT} = {anchor * ADAPTIVE_DOWN_TARGET_MULT:.0f}")
    print(f"[自适应] 上采目标 = anchor * {ADAPTIVE_UP_TARGET_MULT} = {anchor * ADAPTIVE_UP_TARGET_MULT:.0f}")

    downsample_ratios: Dict[str, Tuple[float, float]] = {}
    upsample_ratios: Dict[str, float] = {}

    for t in sorted(VALID_BASIC_TYPES):
        cnt = type_counts.get(t, 0)
        if cnt == 0 or t in ADAPTIVE_SKIP:
            continue

        ratio_to_anchor = cnt / anchor

        if ratio_to_anchor > ADAPTIVE_DOWN_THRESHOLD:
            # 下采: 目标量 = anchor * 1.5 (温和)
            target = anchor * ADAPTIVE_DOWN_TARGET_MULT
            pure_ratio = max(ADAPTIVE_DOWN_MIN_RATIO, target / cnt)
            pure_ratio = min(pure_ratio, 1.0)
            # mixed 更温和: (pure + 1) / 2 是"pure 和 1.0 的均值", 再兜底到 FLOOR
            mixed_ratio = max(ADAPTIVE_DOWN_MIXED_FLOOR, (pure_ratio + 1.0) / 2)
            mixed_ratio = min(mixed_ratio, 1.0)
            downsample_ratios[t] = (round(pure_ratio, 2), round(mixed_ratio, 2))
            print(f"  {t:<15}: {cnt:>6} ({ratio_to_anchor:.1f}x 锚点) → 下采 "
                  f"pure={pure_ratio:.2f}, mixed={mixed_ratio:.2f} "
                  f"(目标~{int(cnt * (pure_ratio * 0.3 + mixed_ratio * 0.7))})")

        elif ratio_to_anchor < ADAPTIVE_UP_THRESHOLD:
            if t in UPSAMPLE_BLACKLIST:
                # 黑名单: Unstable/Follow 即使稀少也不上采(语义/视觉易过拟合)
                print(f"  {t:<15}: {cnt:>6} ({ratio_to_anchor:.1f}x 锚点) → [黑名单]保持不变(禁止上采)")
                continue
            # 上采: 目标量 = anchor * 0.7
            target = anchor * ADAPTIVE_UP_TARGET_MULT
            multiplier = target / cnt
            multiplier = max(1.0, min(multiplier, ADAPTIVE_UP_MAX_MULT))
            upsample_ratios[t] = round(multiplier, 2)
            print(f"  {t:<15}: {cnt:>6} ({ratio_to_anchor:.1f}x 锚点) → 上采 "
                  f"{multiplier:.2f}x (目标 {int(cnt * multiplier)})")
        else:
            print(f"  {t:<15}: {cnt:>6} ({ratio_to_anchor:.1f}x 锚点) → 保持不变")

    return downsample_ratios, upsample_ratios


def get_sample_basic_types(item: Dict[str, Any]) -> Set[str]:
    """提取一个样本中包含的所有 basic_movement 类型。"""
    types = set()
    for seg in item.get("segments", []):
        for bm in seg.get("basic_movement", []):
            t = bm.get("type", "")
            if t and t in VALID_BASIC_TYPES:
                types.add(t)
    return types


def has_combo_segment(item: Dict[str, Any]) -> bool:
    """视频是否含有"组合 segment"(单 segment 内同时出现多种 type)。

    这是 VLM 学"同时段多运镜"的关键样本, 下采时必须保护。
    """
    for seg in item.get("segments", []):
        types_in_seg = set()
        for bm in seg.get("basic_movement", []):
            normalized = normalize_basic_movement(bm)
            if normalized and normalized["type"] in VALID_BASIC_TYPES:
                types_in_seg.add(normalized["type"])
        if len(types_in_seg) >= 2:
            return True
    return False


def is_high_value_sample(item: Dict[str, Any]) -> bool:
    """高价值样本判定:
        1) segment 数 >= QUALITY_MIN_SEGMENTS (时序复杂)
        2) 含组合 segment (单段多类别)
    两者任一满足即视为高价值, 下采时直接跳过(完全保留)。
    """
    segs = item.get("segments", [])
    if len(segs) >= QUALITY_MIN_SEGMENTS:
        return True
    return has_combo_segment(item)


def get_sample_direction_speed_profile(
    item: Dict[str, Any],
) -> Dict[str, List[Tuple[Optional[str], str]]]:
    """提取一个样本中每种 basic_movement 的 (direction, speed) 列表。

    对原始标签做标准化后再提取，确保与输出一致。

    Returns:
        {type: [(direction, speed), ...]}  同一 segment 内同类型去重
    """
    profile: Dict[str, List[Tuple[Optional[str], str]]] = defaultdict(list)
    for seg in item.get("segments", []):
        seen_in_seg: Set[str] = set()
        for bm in seg.get("basic_movement", []):
            normalized = normalize_basic_movement(bm)
            if normalized is None:
                continue
            t = normalized["type"]
            if t in seen_in_seg:
                continue
            seen_in_seg.add(t)
            d = normalized.get("direction")
            s = normalized.get("speed", "medium")
            profile[t].append((d, s))
    return dict(profile)


def balance_samples(
    samples: List[Dict[str, Any]],
    downsample_ratios: Dict[str, Tuple[float, float]] = None,
    upsample_ratios: Dict[str, float] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    对样本进行类别均衡处理（自适应上采样 + 分层下采样）。

    默认采用自适应模式：以中位数为锚点自动计算各类别上下采样比例。
    也可通过参数手动指定比例覆盖自适应。

    下采样策略（分层）：
    - 纯类别样本：按 ratio[0] 下采样（较激进）
    - 混合稀疏类别样本：完全保留
    - 混合常见类别样本：按 ratio[1] 下采样（较温和）
    - fast 样本在下采样中被完全保护
    - 按(方向,速度)联合分层，确保少数组合不被过度丢弃

    上采样策略：
    - 按(方向,速度)联合分组，少数组获得更多重复
    - fast 占比限制在 ADAPTIVE_FAST_CAP 以内，防止过补偿

    核心原则：上下采样都优先保留类别少的。

    Args:
        samples: 原始样本列表
        downsample_ratios: 下采样比例配置（None=自适应计算）
        upsample_ratios: 上采样倍数配置（None=自适应计算）
        seed: 随机种子

    Returns:
        均衡后的样本列表
    """
    # 自适应模式：如果没有手动指定比例，则自动计算
    if downsample_ratios is None and upsample_ratios is None:
        downsample_ratios, upsample_ratios = compute_adaptive_ratios(samples)
    else:
        if downsample_ratios is None:
            downsample_ratios = {}
        if upsample_ratios is None:
            upsample_ratios = {}

    random.seed(seed)

    # 动态稀疏类别集合: 完全基于数据的自适应判定(无硬编码类别名)
    # 1) 被上采样的类别必然是稀疏类
    # 2) cnt/anchor < RARE_TYPE_RATIO 的类别也视为稀疏
    # 3) ADAPTIVE_SKIP(Free Fly 这种极稀有的) 也算稀疏
    type_counts_for_rare: Counter = Counter()
    for s in samples:
        for bt in get_sample_basic_types(s):
            type_counts_for_rare[bt] += 1
    anchor_for_rare = _compute_anchor([
        c for t, c in type_counts_for_rare.items()
        if t not in ADAPTIVE_SKIP and c > 0
    ]) if type_counts_for_rare else 1.0
    rare_types = set(upsample_ratios.keys()) | ADAPTIVE_SKIP
    for t, c in type_counts_for_rare.items():
        if anchor_for_rare > 0 and c / anchor_for_rare < RARE_TYPE_RATIO:
            rare_types.add(t)
    print(f"\n[INFO] 自适应稀疏类集合(cnt/anchor<{RARE_TYPE_RATIO} 或被上采): "
          f"{sorted(rare_types)}")

    # Step 1: 为每个样本提取类别集合 + 方向速度特征 + 高价值标志
    sample_with_id = []
    high_value_flags: List[bool] = []
    for i, s in enumerate(samples):
        types = get_sample_basic_types(s)
        profile = get_sample_direction_speed_profile(s)
        sample_with_id.append((i, s, types, profile))
        high_value_flags.append(is_high_value_sample(s))

    # 类别 -> 包含该类别的样本索引列表
    type_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, _, types, _ in sample_with_id:
        for t in types:
            type_to_indices[t].append(idx)

    # 打印原始分布
    print(f"\n[INFO] 原始类别分布:")
    for t in sorted(VALID_BASIC_TYPES):
        cnt = len(type_to_indices.get(t, []))
        print(f"  {t:<15}: {cnt:>6} 个视频")

    n_high_value = sum(high_value_flags)
    print(f"\n[INFO] 高价值样本(≥{QUALITY_MIN_SEGMENTS} 段 或 含组合 segment): "
          f"{n_high_value}/{len(samples)} ({n_high_value / max(1, len(samples)) * 100:.1f}%)")

    # Step 2: 一次性下采决策(无级联双杀)
    # 核心思路: 对每个样本, 遍历它所有命中的下采 type, 取"最高保留率"作为该样本的最终保留率。
    #          即: 只要该样本在某一个 type 里是宽松类别(mixed_rare/高价值/fast/稀缺方向), 就按该宽松策略处理。
    #          不再"每个 type 都丢一次"。

    # 预计算每个下采 type 的稀缺方向(数量 < 该类别总数 * RARE_DIRECTION_RATIO)
    rare_directions: Dict[str, Set[Optional[str]]] = {}
    for t in downsample_ratios.keys():
        if t not in DIRECTION_REQUIRED:
            continue
        dir_cnt: Counter = Counter()
        for idx in type_to_indices.get(t, []):
            _, _, _, profile = sample_with_id[idx]
            for d, _ in profile.get(t, []):
                dir_cnt[d] += 1
        total_t = sum(dir_cnt.values())
        thresh = total_t * RARE_DIRECTION_RATIO
        rare_directions[t] = {d for d, c in dir_cnt.items() if c < thresh}

    # 预计算稀有组合: 按"类型组合签名(frozenset)"统计视频数,
    # 阈值 = max(RARE_COMBO_MIN_THRESHOLD, N * RARE_COMBO_RATIO), 数据量越大阈值越高(自适应)
    # 小于阈值的组合视为稀有, 完全保留(保证各类型组合的多样性)
    combo_cnt: Counter = Counter()
    sample_combo: Dict[int, frozenset] = {}
    for idx, _, types, _ in sample_with_id:
        if not types:
            continue
        combo = frozenset(types)
        combo_cnt[combo] += 1
        sample_combo[idx] = combo
    combo_threshold = max(RARE_COMBO_MIN_THRESHOLD, int(len(samples) * RARE_COMBO_RATIO))
    rare_combos: Set[frozenset] = {c for c, n in combo_cnt.items() if n < combo_threshold}
    n_rare_combo_videos = sum(combo_cnt[c] for c in rare_combos)
    print(f"\n[INFO] 类型组合签名: 共 {len(combo_cnt)} 种, "
          f"稀有阈值={combo_threshold} (自适应: max({RARE_COMBO_MIN_THRESHOLD}, N×{RARE_COMBO_RATIO})), "
          f"稀有组合 {len(rare_combos)} 种, 覆盖 {n_rare_combo_videos} 条视频 → 全部保留")

    # 为每个样本计算 "针对每个下采 type 的保留率", 再取 max
    # 保留率含义: 1.0=必留, 0.9=90%随机留, 等
    sample_keep_prob: Dict[int, float] = {}
    # 辅助统计
    stats_per_type: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # 记录每个样本 "贡献该保留率的 type" (用于按(方向,速度)均衡的辅助信息)
    sample_driver_type: Dict[int, str] = {}

    all_down_indices: Set[int] = set()
    for t in downsample_ratios.keys():
        all_down_indices.update(type_to_indices.get(t, []))

    for idx in all_down_indices:
        _, _, types, profile = sample_with_id[idx]
        is_hv = high_value_flags[idx]
        is_rare_combo = sample_combo.get(idx) in rare_combos

        best_prob = 0.0
        best_driver = None

        for t, (pure_ratio, mixed_ratio) in downsample_ratios.items():
            if t not in types:
                continue

            # 0) 稀有组合样本: 完全保留(优先级最高, 保证类型组合多样性)
            if is_rare_combo:
                prob_here = 1.0
                stats_per_type[t]["kept_by_rare_combo"] += 1
            # 1) fast 样本 → 完全保留(fast 在多数类别里 <2%, 是真稀缺)
            elif any(spd == "fast" for _, spd in profile.get(t, [])):
                prob_here = 1.0
                stats_per_type[t]["kept_by_fast"] += 1
            # 2) 稀缺方向 → 完全保留(<该类别 15%, 是真稀缺)
            elif t in rare_directions and any(
                (d in rare_directions[t]) for d, _ in profile.get(t, [])
            ):
                prob_here = 1.0
                stats_per_type[t]["kept_by_rare_direction"] += 1
            else:
                # 3) 分层: 高价值 + mixed_rare 享受"软保护"(即 mixed_ratio), 而不是硬保留 1.0
                #    这样 Pan/Static 即使 hv 比例高, 整体保留率仍能按公式下降;
                #    同时又比"纯 pure 层"的 pure_ratio 更宽松(保留 mixed 级的 85%)
                has_rare = any(rt in types for rt in rare_types if rt != t)
                if is_hv:
                    prob_here = mixed_ratio
                    stats_per_type[t]["layer_highvalue"] += 1
                elif has_rare:
                    # mixed_rare: 稀疏类别共现, 也按 mixed_ratio 软保护
                    prob_here = mixed_ratio
                    stats_per_type[t]["layer_mixed_rare"] += 1
                elif len(types) == 1:
                    prob_here = pure_ratio
                    stats_per_type[t]["layer_pure"] += 1
                else:
                    prob_here = mixed_ratio
                    stats_per_type[t]["layer_mixed_common"] += 1

            if prob_here > best_prob:
                best_prob = prob_here
                best_driver = t

        sample_keep_prob[idx] = best_prob
        if best_driver:
            sample_driver_type[idx] = best_driver

    # Step 3: 根据 keep_prob 做"全局一次性随机采样"
    # 为了保证方向/速度均衡, 对每个 type 的"被随机采样的层"做分组抽样;
    # 但由于已在 keep_prob 里把 fast/高价值/稀缺方向/mixed_rare 保留率标为 1.0,
    # 这里只需要对 keep_prob < 1.0 的样本按概率丢弃即可。

    final_indices: Set[int] = set(range(len(samples)))
    # 先把不在任何下采 type 里的样本全部保留(无变动)

    # 对 keep_prob<1.0 的样本: 按 (driver_type, 方向, 速度) 分组, 分组抽样以保均衡
    to_be_sampled: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for idx, prob in sample_keep_prob.items():
        if prob >= 1.0:
            continue  # 保留
        driver = sample_driver_type.get(idx, "__unknown__")
        _, _, _, profile = sample_with_id[idx]
        ds_list = profile.get(driver, [])
        # 取第一个 (dir, speed) 作为分组 key (fast 已被预保护, 这里只剩 medium/slow)
        d, s = (ds_list[0] if ds_list else ("null", "medium"))
        to_be_sampled[(driver, d or "null", s or "medium")].append(idx)

    # 按组, 按该组对应的 prob(同组 prob 相同, 因为同一 driver 的同一分层) 下采
    # 但严格说同组可能混合不同 prob(如一个样本 driver=Pan prob=0.55, 另一个 driver=Pan prob=0.9)
    # 所以按 (driver, layer_prob) 再细分
    refined_groups: Dict[Tuple[str, float, str, str], List[int]] = defaultdict(list)
    for (driver, d, s), idxs in to_be_sampled.items():
        for i in idxs:
            refined_groups[(driver, round(sample_keep_prob[i], 4), d, s)].append(i)

    removed_cnt = 0
    for (driver, prob, d, s), group in refined_groups.items():
        keep_count = int(round(len(group) * prob))
        keep_count = max(0, min(keep_count, len(group)))
        if keep_count == len(group):
            continue
        kept = set(random.sample(group, keep_count))
        for i in group:
            if i not in kept:
                final_indices.discard(i)
                removed_cnt += 1

    # 打印下采摘要(每个 type 的分层统计 + 最终数量)
    print(f"\n[INFO] 一次性下采决策(无级联双杀):")
    for t in sorted(downsample_ratios.keys()):
        pr, mr = downsample_ratios[t]
        st = stats_per_type[t]
        cnt_before = len(type_to_indices.get(t, []))
        cnt_after = sum(1 for idx in type_to_indices.get(t, []) if idx in final_indices)
        print(f"  {t:<14} (pure={pr}, mixed={mr}): {cnt_before} → {cnt_after} "
              f"[硬保留: rare_combo={st.get('kept_by_rare_combo', 0)} "
              f"fast={st.get('kept_by_fast', 0)} "
              f"rare_dir={st.get('kept_by_rare_direction', 0)} "
              f"| 软保护(按mixed_ratio): hv={st.get('layer_highvalue', 0)} "
              f"mx_rare={st.get('layer_mixed_rare', 0)} "
              f"| 抽样(按pure): pure={st.get('layer_pure', 0)} "
              f"(按mixed): mx_common={st.get('layer_mixed_common', 0)}]")
    print(f"  [INFO] 下采共移除: {removed_cnt} 条")

    # Step 4: 按倍数上采样尾部类别（方向感知）
    # 关键: target_count 基于"原始数量"而非"当前保留数量", 避免下采误伤反向放大上采不足
    upsample_samples = []
    upsample_info = []
    for t, multiplier in upsample_ratios.items():
        if t in UPSAMPLE_BLACKLIST:
            continue  # 黑名单: Unstable/Follow 禁止上采
        if t not in type_to_indices:
            continue
        # 原始数量(下采前的完整样本池, 用于算 target)
        original_count = len(type_to_indices[t])
        # 当前保留的样本索引(实际可复制源)
        current_indices = [idx for idx in type_to_indices[t] if idx in final_indices]
        if len(current_indices) == 0:
            continue

        # 目标数量 = 原始 × 倍数(不受下采误伤影响)
        target_count = int(original_count * multiplier)
        need_more = target_count - len(current_indices)

        if need_more <= 0:
            continue

        if t in DIRECTION_REQUIRED:
            # 方向+速度感知上采样：按(方向,速度)联合分组，少数组获得更多重复
            extra_indices = _direction_speed_aware_upsample(
                current_indices, sample_with_id, t, need_more
            )
        else:
            # 无方向属性的类别：随机重复采样
            extra_indices = random.choices(current_indices, k=need_more)

        for idx in extra_indices:
            upsample_samples.append(samples[idx])
        upsample_info.append((t, len(current_indices), target_count, multiplier, need_more))

    if upsample_info:
        print(f"\n[INFO] 上采样处理 (按倍数):")
        for t, before, after, mult, added in upsample_info:
            print(f"  {t:<15}: {before:>6} → {after:>6} ({mult}x, 重复采样 +{added} 条)")

    # Step 5: 合并结果
    balanced_samples = [samples[idx] for idx in sorted(final_indices)]
    balanced_samples.extend(upsample_samples)

    # Step 6: 充分 shuffle
    random.shuffle(balanced_samples)

    # 打印最终分布
    final_type_counts = Counter()
    final_dir_counts: Dict[str, Counter] = defaultdict(Counter)
    final_speed_counts: Dict[str, Counter] = defaultdict(Counter)
    for s in balanced_samples:
        profile = get_sample_direction_speed_profile(s)
        for bt in get_sample_basic_types(s):
            final_type_counts[bt] += 1
        for bt, ds_list in profile.items():
            for d, spd in ds_list:
                if d:
                    final_dir_counts[bt][d] += 1
                final_speed_counts[bt][spd] += 1

    print(f"\n[INFO] 均衡后类别分布:")
    total = sum(final_type_counts.values())
    for t in sorted(VALID_BASIC_TYPES):
        cnt = final_type_counts.get(t, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {t:<15}: {cnt:>6} 个视频 ({pct:>5.2f}%)")

    # ======= 运镜类别数量 原始 vs 采样后 对比 =======
    n_before = len(samples)
    n_after = len(balanced_samples)
    before_counts = {t: len(type_to_indices.get(t, [])) for t in VALID_BASIC_TYPES}
    print(f"\n[INFO] ========== 运镜类别数量对比（原始 → 采样后）==========")
    print(f"  {'类别':<14} {'原数量':>8} {'原占比':>8}  {'采样后':>8} {'后占比':>8}  {'Δ数量':>8}  {'保留率':>7}  策略")
    print(f"  {'-' * 95}")
    # 按原数量从大到小排序展示
    sorted_types = sorted(VALID_BASIC_TYPES, key=lambda x: -before_counts.get(x, 0))
    for t in sorted_types:
        b = before_counts.get(t, 0)
        a = final_type_counts.get(t, 0)
        if b == 0 and a == 0:
            continue
        b_pct = b / n_before * 100 if n_before else 0
        a_pct = a / n_after * 100 if n_after else 0
        delta = a - b
        keep_rate = (a / b * 100) if b > 0 else 0.0
        if t in downsample_ratios:
            pr, mr = downsample_ratios[t]
            strat = f"下采 p={pr}/m={mr}"
        elif t in upsample_ratios:
            strat = f"上采 {upsample_ratios[t]:.2f}x"
        elif t in UPSAMPLE_BLACKLIST:
            strat = "[黑名单]不上采"
        elif t in ADAPTIVE_SKIP:
            strat = "跳过"
        else:
            strat = "保持"
        print(f"  {t:<14} {b:>8} {b_pct:>7.2f}%  {a:>8} {a_pct:>7.2f}%  {delta:>+8}  {keep_rate:>6.1f}%  {strat}")
    print(f"  {'-' * 95}")
    print(f"  {'总计':<14} {n_before:>8} {100.0:>7.2f}%  {n_after:>8} {100.0:>7.2f}%  "
          f"{n_after - n_before:>+8}  {n_after / max(1, n_before) * 100:>6.1f}%")

    # 关键指标对比
    pos_before = [c for c in before_counts.values() if c > 0]
    pos_after = [c for c in final_type_counts.values() if c > 0]
    pos_before_no_skip = [c for t, c in before_counts.items()
                          if c > 0 and t not in ADAPTIVE_SKIP]
    pos_after_no_skip = [c for t, c in final_type_counts.items()
                         if c > 0 and t not in ADAPTIVE_SKIP]
    max_min_before = max(pos_before) / min(pos_before) if pos_before else 0
    max_min_after = max(pos_after) / min(pos_after) if pos_after else 0
    max_min_before_ns = (max(pos_before_no_skip) / min(pos_before_no_skip)
                         if pos_before_no_skip else 0)
    max_min_after_ns = (max(pos_after_no_skip) / min(pos_after_no_skip)
                        if pos_after_no_skip else 0)
    print(f"\n[INFO] 关键均衡指标:")
    print(f"  最大/最小比值(含极稀有): {max_min_before:>8.1f}x → {max_min_after:>8.1f}x")
    print(f"  最大/最小比值(排 ADAPTIVE_SKIP): {max_min_before_ns:>8.1f}x → {max_min_after_ns:>8.1f}x")
    # 头部/尾部 = 采样策略实际被下采/上采的类别(去重到样本级, 不重复计数)
    head_types = list(downsample_ratios.keys())
    tail_types = list(upsample_ratios.keys())
    if head_types:
        head_set_before = {idx for t in head_types for idx in type_to_indices.get(t, [])}
        head_set_after = sum(1 for s in balanced_samples
                             if any(t in get_sample_basic_types(s) for t in head_types))
        print(f"  头部(下采类: {','.join(head_types)}) 样本占比: "
              f"{len(head_set_before) / n_before * 100:.2f}% → {head_set_after / n_after * 100:.2f}%")
    if tail_types:
        tail_set_before = {idx for t in tail_types for idx in type_to_indices.get(t, [])}
        tail_set_after = sum(1 for s in balanced_samples
                             if any(t in get_sample_basic_types(s) for t in tail_types))
        print(f"  尾部(上采类: {','.join(tail_types)}) 样本占比: "
              f"{len(tail_set_before) / n_before * 100:.2f}% → {tail_set_after / n_after * 100:.2f}%")

    # 打印方向分布
    print(f"\n[INFO] 均衡后方向分布:")
    for t in sorted(DIRECTION_REQUIRED):
        dirs = final_dir_counts.get(t, {})
        if not dirs:
            continue
        total_t = sum(dirs.values())
        parts = [f"{d}={cnt}({cnt/total_t*100:.1f}%)" for d, cnt in sorted(dirs.items())]
        print(f"  {t:<15}: {' | '.join(parts)}")

    # 打印速度分布
    print(f"\n[INFO] 均衡后速度分布（fast 占比）:")
    for t in sorted(VALID_BASIC_TYPES):
        speeds = final_speed_counts.get(t, {})
        if not speeds:
            continue
        total_t = sum(speeds.values())
        fast_cnt = speeds.get("fast", 0)
        if fast_cnt > 0:
            print(f"  {t:<15}: fast={fast_cnt}({fast_cnt/total_t*100:.1f}%), "
                  f"medium={speeds.get('medium', 0)}, slow={speeds.get('slow', 0)}")

    print(f"\n[INFO] 均衡后总样本数: {len(balanced_samples)}")

    return balanced_samples


def _assign_sample_to_rarest_group(
    ds_list: List[Tuple[Optional[str], str]],
    group_freq: Dict[Tuple[str, str], int],
) -> Tuple[str, str]:
    """把一个样本（可能包含多个 (direction, speed) 组合）归入它自身最稀缺的那个组。

    这样多方向样本（如先 Pan Left 再 Pan Right）会被归入数量最少的那个方向，
    有助于保护真正的少数方向。
    """
    if not ds_list:
        return ("null", "medium")
    best = None
    best_freq = float("inf")
    for d, s in ds_list:
        key = (d or "null", s or "medium")
        freq = group_freq.get(key, 0)
        if freq < best_freq:
            best_freq = freq
            best = key
    return best if best is not None else ("null", "medium")


def _build_group_frequency(
    indices: List[int],
    sample_with_id: List[Tuple],
    target_type: str,
) -> Dict[Tuple[str, str], int]:
    """统计全体 indices 中各 (dir, speed) 组合的样本频次（用于判断稀缺度）。

    按"第一个 segment"粗统计即可——只是用来衡量相对稀缺度。
    """
    freq: Dict[Tuple[str, str], int] = defaultdict(int)
    for idx in indices:
        _, _, _, profile = sample_with_id[idx]
        for d, s in profile.get(target_type, []):
            freq[(d or "null", s or "medium")] += 1
    return freq


def _direction_speed_aware_downsample(
    indices: List[int],
    sample_with_id: List[Tuple],
    target_type: str,
    keep_total: int,
) -> Set[int]:
    """方向+速度感知的下采样：按 (方向, 速度) 联合分组，各组均匀保留。

    核心原则：优先保留类别少的组合。
    - fast 样本极少 → 保留比例更高
    - 少数方向样本少 → 保留比例更高

    Args:
        indices: 待下采样的样本索引列表
        sample_with_id: (idx, sample, types, profile) 列表
        target_type: 目标运镜类别
        keep_total: 总共要保留的数量

    Returns:
        要保留的索引集合
    """
    # 先统计全局频次，用来把多方向样本归入最稀缺的组
    group_freq = _build_group_frequency(indices, sample_with_id, target_type)

    # 按 (方向, 速度) 联合分组（每个样本归入它自己最稀缺的那个组）
    joint_groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx in indices:
        _, _, _, profile = sample_with_id[idx]
        ds_list = profile.get(target_type, [])
        key = _assign_sample_to_rarest_group(ds_list, group_freq)
        joint_groups[key].append(idx)

    if len(joint_groups) <= 1:
        # 只有一组，无需均衡
        keep_count = min(keep_total, len(indices))
        return set(random.sample(indices, keep_count))

    # 目标：每组均等分配保留名额，但不超过该组实际数量
    n_groups = len(joint_groups)
    per_group_target = keep_total // n_groups
    keep_set = set()
    remainder_pool = []

    for key, group in joint_groups.items():
        if len(group) <= per_group_target:
            # 该组样本数 ≤ 配额，全部保留
            keep_set.update(group)
        else:
            selected = set(random.sample(group, per_group_target))
            keep_set.update(selected)
            remainder_pool.extend([idx for idx in group if idx not in selected])

    # 如果还有名额剩余，从 remainder_pool 补充
    remaining_quota = keep_total - len(keep_set)
    if remaining_quota > 0 and remainder_pool:
        extra = random.sample(remainder_pool, min(remaining_quota, len(remainder_pool)))
        keep_set.update(extra)

    return keep_set


def _direction_speed_aware_upsample(
    indices: List[int],
    sample_with_id: List[Tuple],
    target_type: str,
    need_more: int,
    fast_cap: float = ADAPTIVE_FAST_CAP,
) -> List[int]:
    """方向+速度感知的上采样：按 (方向, 速度) 联合分组，少数组获得更多重复。

    核心原则：优先保留/多重复类别少的。
    - fast 样本极少 → 获得更多重复
    - 少数方向样本少 → 获得更多重复
    - fast 占比限制在 fast_cap 以内，防止小类别过补偿

    Args:
        indices: 当前保留的该类别样本索引
        sample_with_id: (idx, sample, types, profile) 列表
        target_type: 目标运镜类别
        need_more: 需要额外采样的数量
        fast_cap: fast 占比上限（默认 0.15）

    Returns:
        额外采样的索引列表
    """
    # 先统计全局频次，用来把多方向样本归入最稀缺的组
    group_freq = _build_group_frequency(indices, sample_with_id, target_type)

    # 按 (方向, 速度) 联合分组（每个样本归入它自己最稀缺的那个组）
    joint_groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx in indices:
        _, _, _, profile = sample_with_id[idx]
        ds_list = profile.get(target_type, [])
        key = _assign_sample_to_rarest_group(ds_list, group_freq)
        joint_groups[key].append(idx)

    if len(joint_groups) <= 1:
        return random.choices(indices, k=need_more)

    # 先按方向分组（不区分速度），用于方向均衡
    dir_groups: Dict[str, List[int]] = defaultdict(list)
    for (d, s), group in joint_groups.items():
        dir_groups[d].extend(group)

    # 计算 fast 上限：上采样后 fast 总数不超过 total_after * fast_cap
    total_after = len(indices) + need_more
    fast_groups = {k: v for k, v in joint_groups.items() if k[1] == "fast"}
    non_fast_groups = {k: v for k, v in joint_groups.items() if k[1] != "fast"}

    current_fast_count = sum(len(v) for v in fast_groups.values())
    max_fast_total = int(total_after * fast_cap)

    # fast 组最多能补多少
    fast_extra_budget = max(0, max_fast_total - current_fast_count)

    extra = []
    remaining_quota = need_more

    # 阶段 1：先给 fast 组补一些（受 cap 严格限制）
    if fast_groups and fast_extra_budget > 0:
        # 按方向均匀分配 fast 预算（不用 max(1,...) 避免击穿 cap）
        fast_dirs = list(fast_groups.keys())
        per_fast_dir = fast_extra_budget // len(fast_dirs) if fast_dirs else 0
        for key in sorted(fast_dirs, key=lambda k: len(fast_groups[k])):
            if per_fast_dir <= 0:
                break
            group = fast_groups[key]
            actual = min(per_fast_dir, remaining_quota, fast_extra_budget)
            if actual > 0:
                extra.extend(random.choices(group, k=actual))
                remaining_quota -= actual
                fast_extra_budget -= actual

    # 阶段 2：剩余名额分配给 non-fast 组（按方向均衡）
    if remaining_quota > 0 and non_fast_groups:
        n_nf_groups = len(non_fast_groups)
        per_nf_target = remaining_quota // n_nf_groups if n_nf_groups > 0 else remaining_quota

        nf_needs = []
        for key, group in sorted(non_fast_groups.items(), key=lambda x: len(x[1])):
            nf_needs.append((key, group, per_nf_target))

        for key, group, alloc in nf_needs:
            actual = min(alloc, remaining_quota)
            if actual > 0:
                extra.extend(random.choices(group, k=actual))
                remaining_quota -= actual

    # 阶段 3：如果还有剩余，随机分配到 non-fast
    if remaining_quota > 0:
        non_fast_indices = [idx for k, v in non_fast_groups.items() for idx in v]
        if non_fast_indices:
            extra.extend(random.choices(non_fast_indices, k=remaining_quota))
        else:
            extra.extend(random.choices(indices, k=remaining_quota))

    return extra


def build_sample(
    video_path: str,
    segments: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """构建一条 ms-swift 标准格式的训练样本。"""
    normalized = normalize_segments(segments)
    if not normalized["segments"]:
        return None

    response_json = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))

    sample = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<video>{USER_PROMPT}"},
            {"role": "assistant", "content": response_json},
        ],
        "videos": [video_path],
    }
    return sample


def download_video(url: str, save_path: str, timeout: int = 120) -> bool:
    """从 URL 下载视频到指定路径，成功返回 True。"""
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        tmp_path = save_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.rename(tmp_path, save_path)
        return True
    except Exception:
        # 清理可能的临时文件
        tmp_path = save_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def process_single_item(
    item: Dict[str, Any], video_roots: List[str], check_video_exists: bool,
    auto_download: bool = False, download_dir: Optional[str] = None
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """处理单个样本（用于并发执行）。

    返回: (状态, 样本, video_id)
        状态: "success", "video_not_found", "download_success", "download_failed", "empty_after_normalize"
    """
    video_id = item.get("video_id", "")
    segments = item.get("segments", [])

    # 在多个目录中查找视频文件
    video_path = None
    for video_root in video_roots:
        candidate_path = os.path.join(video_root, f"{video_id}.mp4")
        if os.path.exists(candidate_path):
            video_path = candidate_path
            break
    
    # 如果没找到，尝试自动下载
    if video_path is None:
        if auto_download and download_dir:
            cos_url = item.get("cos_url", "")
            if cos_url:
                target_path = os.path.join(download_dir, f"{video_id}.mp4")
                if download_video(cos_url, target_path):
                    video_path = target_path
                else:
                    return ("download_failed", None, video_id)
            else:
                return ("video_not_found", None, video_id)
        elif check_video_exists:
            return ("video_not_found", None, video_id)
        else:
            video_path = os.path.join(video_roots[0], f"{video_id}.mp4")

    sample = build_sample(video_path, segments)
    if sample is None:
        return ("empty_after_normalize", None, video_id)

    status = "success"
    return (status, sample, video_id)


def main():
    parser = argparse.ArgumentParser(
        description="人工标注运镜数据转换为 ms-swift jsonl 格式"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        nargs="+",
        default=[
            "/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/train_data/human_labels/yunque/",
            "/group/40059/yyjyu/data/aigc/camera_data/human_labels/hunyuan/results",
        ],
        help="人工标注数据目录（包含多个 jsonl 文件，可指定多个目录）",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        nargs="+",
        default=[
            "/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos",
            "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_21k_video_shot_cameraMove/videos",
            "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_realfilm_30w_260401_deduped_final/videos",
        ],
        help="视频文件根目录（可指定多个，会依次查找）",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="输出 jsonl 文件路径（默认根据数据量自动命名，如 camera_movement_train_human_17k.jsonl）",
    )
    parser.add_argument(
        "--exclude_jsonl",
        type=str,
        nargs="+",
        default=[
            "/group/40059/yyjyu/code/cv/swift-3.12.4/camera_movement_sft/train_data/human_labels/yunque/testset_results/testset_500_refined_reviewed_450_sampled_300_refined_v1.jsonl",
            "/group/40059/yyjyu/data/aigc/camera_data/human_labels/yunque/testset_results/testset_arc_zoom_roll_200.jsonl",
        ],
        help="要排除的 video_id 列表文件（jsonl 格式，可指定多个，用于剔除测试集）",
    )
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        nargs="+",
        default=[
            "运镜标注-正式标注0324-1w-*",
            "正式标注0327-3w-4000条-part*-标注--*",
        ],
        help="要排除的输入文件名模式（支持通配符，可指定多个）。"
             "默认排除已被返修版替代的原始标注文件。"
             "传 --exclude_patterns 不带参数可清空排除列表",
    )
    parser.add_argument(
        "--check_video_exists",
        action="store_true",
        default=True,
        help="是否检查视频文件存在（默认检查）",
    )
    parser.add_argument(
        "--auto_download",
        action="store_true",
        default=True,
        help="当视频不存在时，自动从 jsonl 中的 cos_url 下载到 --download_dir（默认开启）",
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        default=False,
        help="禁用自动下载（覆盖 --auto_download）",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos",
        help="自动下载视频的存放目录（默认: video_root 的第一个目录）",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="并发进程数（默认为 32）",
    )
    parser.add_argument(
        "--manual_ratios",
        action="store_true",
        default=False,
        help="使用手动指定的采样比例（而非自适应计算）。"
             "需要配合 --downsample_* 和 --upsample_* 参数使用",
    )
    parser.add_argument(
        "--downsample_static_pure",
        type=float,
        default=0.30,
        help="[手动模式] 纯 Static 样本保留比例（默认 0.30）",
    )
    parser.add_argument(
        "--downsample_static_mixed",
        type=float,
        default=0.70,
        help="[手动模式] Static + 常见类别 混合样本保留比例（默认 0.70）",
    )
    parser.add_argument(
        "--downsample_pan_pure",
        type=float,
        default=0.60,
        help="[手动模式] 纯 Pan 样本保留比例（默认 0.60）",
    )
    parser.add_argument(
        "--downsample_pan_mixed",
        type=float,
        default=0.90,
        help="[手动模式] Pan + 常见类别 混合样本保留比例（默认 0.90）",
    )
    parser.add_argument(
        "--upsample_follow",
        type=float,
        default=2.0,
        help="[手动模式] Follow 类别上采样倍数（默认 2.0x）",
    )
    parser.add_argument(
        "--upsample_roll",
        type=float,
        default=2.5,
        help="[手动模式] Roll 类别上采样倍数（默认 2.5x）",
    )
    parser.add_argument(
        "--upsample_arc",
        type=float,
        default=2.5,
        help="[手动模式] Arc 类别上采样倍数（默认 2.5x）",
    )
    parser.add_argument(
        "--upsample_zoom_in",
        type=float,
        default=3.0,
        help="[手动模式] Zoom In 类别上采样倍数（默认 3.0x）",
    )
    parser.add_argument(
        "--upsample_zoom_out",
        type=float,
        default=4.0,
        help="[手动模式] Zoom Out 类别上采样倍数（默认 4.0x）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )

    args = parser.parse_args()

    # --no_download 覆盖 --auto_download
    if args.no_download:
        args.auto_download = False

    # 默认并发数（下载时用更多线程，IO 密集型）
    if args.num_workers is None:
        if args.auto_download:
            args.num_workers = 100
        else:
            args.num_workers = min(32, os.cpu_count() or 16)
    print(f"[INFO] 使用 {args.num_workers} 个线程并发处理")

    # 打印视频根目录信息
    print(f"[INFO] 视频根目录列表 ({len(args.video_root)} 个):")
    for vr in args.video_root:
        exists_mark = "✓" if os.path.isdir(vr) else "✗"
        print(f"  {exists_mark} {vr}")

    # 读取所有人工标注文件（支持多个输入目录）
    input_files = []
    print(f"[INFO] 输入目录列表 ({len(args.input_dir)} 个):")
    for input_d in args.input_dir:
        exists_mark = "✓" if os.path.isdir(input_d) else "✗"
        files_in_dir = glob.glob(os.path.join(input_d, "*.jsonl"))
        input_files.extend(files_in_dir)
        print(f"  {exists_mark} {input_d} ({len(files_in_dir)} 个 jsonl)")

    # 标记哪些文件被 exclude_patterns 匹配（不立即过滤，后续分组输出用）
    excluded_file_set = set()
    if args.exclude_patterns:
        for f in input_files:
            basename = os.path.basename(f)
            if any(fnmatch.fnmatch(basename, pat) for pat in args.exclude_patterns):
                excluded_file_set.add(f)
        if excluded_file_set:
            print(f"[INFO] 以下 {len(excluded_file_set)} 个文件匹配 exclude_patterns (将生成排除版本):")
            for ef in sorted(excluded_file_set):
                print(f"  ✗ {os.path.basename(ef)}")

    if not input_files:
        print(f"[ERROR] 在所有输入目录下均未找到任何 .jsonl 文件")
        return

    included_files = [f for f in input_files if f not in excluded_file_set]
    print(f"[INFO] 共找到 {len(input_files)} 个标注文件 (保留 {len(included_files)} + 排除 {len(excluded_file_set)}):")
    for f in sorted(input_files):
        mark = "✗" if f in excluded_file_set else "✓"
        print(f"  {mark} {os.path.basename(f)}")

    # 读取要排除的 video_id 列表（测试集）
    exclude_video_ids = set()
    if args.exclude_jsonl:
        print(f"[INFO] 排除列表文件 ({len(args.exclude_jsonl)} 个):")
        for exclude_file in args.exclude_jsonl:
            exclude_file = exclude_file.strip()  # 防御性处理: 去除路径首尾空格
            if os.path.exists(exclude_file):
                count_before = len(exclude_video_ids)
                with open(exclude_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                vid = item.get("video_id", "")
                                if vid:
                                    exclude_video_ids.add(vid)
                            except json.JSONDecodeError:
                                continue
                count_added = len(exclude_video_ids) - count_before
                print(f"  ✓ {os.path.basename(exclude_file)}: +{count_added} 个 video_id")
            else:
                print(f"  ✗ {exclude_file} (文件不存在)")
        print(f"[INFO] 排除列表合计: {len(exclude_video_ids)} 个 video_id")

    # 读取所有数据（按 video_id 去重，根据文件修改时间优先保留最新的记录）
    # 同时记录每条数据是否来自被 exclude 的文件

    # 按文件修改时间从旧到新排序（新文件在后面，后出现的覆盖旧的）
    input_files_sorted = sorted(input_files, key=lambda f: os.path.getmtime(f))
    print(f"[INFO] 按文件修改时间排序（旧→新，去重时优先保留新文件的数据）:")
    for f in input_files_sorted:
        mtime = os.path.getmtime(f)
        from datetime import datetime
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {mtime_str} {os.path.basename(f)}")

    raw_data = []
    raw_data_excluded_flags = []  # 与 raw_data 一一对应，True 表示来自被排除的文件
    # video_id -> (index in raw_data) 用于去重时覆盖旧记录
    video_id_to_idx: Dict[str, int] = {}
    excluded_count = 0
    duplicate_count = 0
    for input_file in input_files_sorted:
        is_excluded_file = input_file in excluded_file_set
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        # 检查是否在排除列表中
                        vid = item.get("video_id", "")
                        if vid and vid in exclude_video_ids:
                            excluded_count += 1
                            continue
                        # 按 video_id 去重，优先保留新文件的记录（覆盖旧的）
                        if vid and vid in video_id_to_idx:
                            # 新文件的数据覆盖旧文件的数据
                            old_idx = video_id_to_idx[vid]
                            raw_data[old_idx] = item
                            raw_data_excluded_flags[old_idx] = is_excluded_file
                            duplicate_count += 1
                            continue
                        if vid:
                            video_id_to_idx[vid] = len(raw_data)
                        raw_data.append(item)
                        raw_data_excluded_flags.append(is_excluded_file)
                    except json.JSONDecodeError as e:
                        print(f"[WARN] 解析失败: {e}")
                        continue

    excluded_file_data_count = sum(1 for flag in raw_data_excluded_flags if flag)
    included_file_data_count = len(raw_data) - excluded_file_data_count
    print(f"[INFO] 原始样本数: {len(raw_data)} (保留文件: {included_file_data_count}, 排除文件: {excluded_file_data_count})"
          + (f" (已排除测试集 {excluded_count} 条)" if excluded_count > 0 else "")
          + (f" (已去重 {duplicate_count} 条)" if duplicate_count > 0 else ""))

    stats = Counter()
    missing_videos = []

    # 如果开启自动下载，确保下载目录存在
    if args.auto_download:
        os.makedirs(args.download_dir, exist_ok=True)
        print(f"[INFO] 自动下载已开启，缺失视频将下载到: {args.download_dir}")

    # 创建处理函数（固定参数）
    process_fn = partial(
        process_single_item,
        video_roots=args.video_root,
        check_video_exists=args.check_video_exists,
        auto_download=args.auto_download,
        download_dir=args.download_dir,
    )

    # 并发处理
    results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_fn, item): i for i, item in enumerate(raw_data)}

        for future in tqdm(as_completed(futures), total=len(raw_data), desc="处理中"):
            status, sample, video_id = future.result()
            stats[status] += 1
            if sample is not None:
                results.append((futures[future], sample))
            elif status in ("video_not_found", "download_failed"):
                missing_videos.append(video_id)

    # 按原始顺序排序后准备输出
    print("[INFO] 排序并准备输出...")
    results.sort(key=lambda x: x[0])
    
    # 提取成功的样本，同时保留 excluded 标记
    all_success_samples = []       # 全量成功样本
    included_success_samples = []  # 排除 exclude 文件后的成功样本
    all_success_raw = []           # 全量成功原始数据（用于均衡采样）
    included_success_raw = []      # exclude 后成功原始数据
    
    # 建立 video_id -> (sample, raw_item, is_excluded) 映射
    success_video_id_to_info = {}
    for orig_idx, sample in results:
        video_path = sample["videos"][0]
        video_id = os.path.basename(video_path).replace(".mp4", "")
        is_excluded = raw_data_excluded_flags[orig_idx]
        success_video_id_to_info[video_id] = (sample, raw_data[orig_idx], is_excluded)
        
        all_success_samples.append(sample)
        all_success_raw.append(raw_data[orig_idx])
        
        if not is_excluded:
            included_success_samples.append(sample)
            included_success_raw.append(raw_data[orig_idx])
    
    # 确定输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 格式化数量：17891 -> 17k, 8500 -> 8k
    def format_count(count):
        if count >= 1000:
            return f"{count // 1000}k"
        return str(count)
    
    # 确定基础输出路径前缀
    output_dir = os.path.join(script_dir, "train_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # 输出版本 1：全量数据（所有输入文件，shuffle 后）
    # ============================================================
    random.seed(args.seed)
    shuffled_all = all_success_samples.copy()
    random.shuffle(shuffled_all)
    
    output_all = args.output_jsonl or os.path.join(
        output_dir,
        f"camera_movement_train_human_all_{format_count(len(shuffled_all))}.jsonl"
    )
    with open(output_all, "w", encoding="utf-8") as fout:
        for sample in shuffled_all:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"[版本1] 全量数据:")
    print(f"  样本数: {len(shuffled_all)}")
    print(f"  输出文件: {output_all}")
    
    # ============================================================
    # 输出版本 2：exclude 后数据（排除指定文件名模式后，shuffle）
    # ============================================================
    random.seed(args.seed)
    shuffled_included = included_success_samples.copy()
    random.shuffle(shuffled_included)
    
    output_excluded = os.path.join(
        output_dir,
        f"camera_movement_train_human_excluded_{format_count(len(shuffled_included))}.jsonl"
    )
    with open(output_excluded, "w", encoding="utf-8") as fout:
        for sample in shuffled_included:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"[版本2] exclude 后数据:")
    print(f"  样本数: {len(shuffled_included)} (排除了 {len(all_success_samples) - len(included_success_samples)} 条)")
    print(f"  输出文件: {output_excluded}")
    
    # ============================================================
    # 输出版本 3：exclude 后 + 自适应类别均衡数据
    # ============================================================
    if args.manual_ratios:
        # 手动模式：使用用户指定的固定比例
        downsample_ratios = {
            "Static": (args.downsample_static_pure, args.downsample_static_mixed),
            "Pan": (args.downsample_pan_pure, args.downsample_pan_mixed),
        }
        upsample_ratios = {
            "Follow": args.upsample_follow,
            "Roll": args.upsample_roll,
            "Arc": args.upsample_arc,
            "Zoom In": args.upsample_zoom_in,
            "Zoom Out": args.upsample_zoom_out,
        }
        print(f"\n[INFO] 使用手动采样比例")
    else:
        # 自适应模式：传 None 让 balance_samples 自动计算
        downsample_ratios = None
        upsample_ratios = None
        print(f"\n[INFO] 使用自适应采样比例（基于中位数锚点）")
    
    # 对 exclude 后的数据进行类别均衡
    balanced_raw = balance_samples(
        included_success_raw,
        downsample_ratios=downsample_ratios,
        upsample_ratios=upsample_ratios,
        seed=args.seed,
    )
    
    # 转换为输出格式
    video_id_to_sample = {
        os.path.basename(s["videos"][0]).replace(".mp4", ""): s
        for s in included_success_samples
    }
    balanced_samples = []
    for item in balanced_raw:
        vid = item.get("video_id", "")
        if vid in video_id_to_sample:
            balanced_samples.append(video_id_to_sample[vid])
    
    output_balanced = os.path.join(
        output_dir,
        f"camera_movement_train_human_excluded_balanced_{format_count(len(balanced_samples))}.jsonl"
    )
    with open(output_balanced, "w", encoding="utf-8") as fout:
        for sample in balanced_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"[版本3] exclude 后 + 类别均衡数据:")
    print(f"  样本数: {len(balanced_samples)}")
    print(f"  输出文件: {output_balanced}")
    
    # ============================================================
    # 统计摘要
    # ============================================================
    print(f"\n{'='*60}")
    print(f"[统计摘要]")
    print(f"  处理成功: {stats['success']}")
    if args.auto_download:
        print(f"  自动下载成功: {stats.get('download_success', 0)}")
        print(f"  下载失败: {stats.get('download_failed', 0)}")
    print(f"  视频未找到: {stats.get('video_not_found', 0)} (已剔除)")
    print(f"  标准化后为空: {stats.get('empty_after_normalize', 0)}")
    print(f"  输出文件:")
    print(f"    1. 全量:           {output_all} ({len(shuffled_all)} 条)")
    print(f"    2. exclude后:      {output_excluded} ({len(shuffled_included)} 条)")
    print(f"    3. exclude+均衡:   {output_balanced} ({len(balanced_samples)} 条)")

    # 保存缺失视频列表到文件
    if missing_videos:
        missing_file = os.path.join(output_dir, "missing_videos.txt")
        with open(missing_file, "w", encoding="utf-8") as f:
            for vid in sorted(missing_videos):
                f.write(f"{vid}\n")
        print(f"\n[WARN] 缺失的视频文件 ({len(missing_videos)} 个):")
        print(f"  已保存列表到: {missing_file}")
        # 打印前 30 个示例
        for vid in missing_videos[:30]:
            print(f"    - {vid}.mp4")
        if len(missing_videos) > 30:
            print(f"    ... 等共 {len(missing_videos)} 个文件 (完整列表见上述文件)")


if __name__ == "__main__":
    main()
