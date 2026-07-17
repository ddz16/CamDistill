#!/usr/bin/env python3
"""
从 filter 后的训练集中, 按来源(video_type) + 运镜类别采样 ~40k 子集。

策略:
1. 统计每条样本的 video_type 和 basic_movement type 集合
2. 按 (video_type, primary_movement_type) 联合分桶
3. 在每个桶内采样, 确保:
   - 来源分布: 每个 video_type 都覆盖到
   - 类别分布: 每个运镜类别都覆盖到
   - 数量均衡: 大类下采样, 小类全部保留 (上限 cap)

用法:
  # 1. 先统计分布 (不采样)
  python sample_diverse_subset.py --input <jsonl> --analyze_only

  # 2. 采样
  python sample_diverse_subset.py \
      --input camera_movement_sft/train_data/camera_movement_train_human_all_292k_with_video_type.jsonl \
      --output camera_movement_sft/train_data/camera_movement_train_diverse_40k.jsonl \
      --target_size 40000
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict


def parse_movement_types(assistant_content: str) -> list:
    """从 assistant 输出 JSON 中提取所有 basic_movement type 列表"""
    try:
        data = json.loads(assistant_content)
    except (json.JSONDecodeError, TypeError):
        # 尝试从混合文本中提取 JSON
        match = re.search(r'\{.*\}', assistant_content, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    types = []
    for seg in data.get('segments', []):
        for m in seg.get('basic_movement', []):
            t = m.get('type')
            if t:
                types.append(t)
    return types


def get_sample_info(line: str) -> dict:
    """提取一条样本的 video_type + movement types"""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    video_type = data.get('video_type', '未知')

    # 找到 assistant 的回复
    assistant_content = ''
    for m in data.get('messages', []):
        if m.get('role') == 'assistant':
            assistant_content = m.get('content', '')
            break

    types = parse_movement_types(assistant_content)
    return {
        'video_type': video_type,
        'types': types,  # 该样本包含的所有 movement type
        'primary_type': types[0] if types else 'NoMovement',  # 主类别
    }


def analyze(samples: list):
    """统计分布"""
    print(f"\n=== 数据分布统计 ===")
    print(f"总样本数: {len(samples)}\n")

    # 按 video_type 统计
    vt_counter = Counter(s['video_type'] for s in samples)
    print(f"--- video_type 分布 ---")
    for vt, cnt in vt_counter.most_common():
        print(f"  {vt:>10}: {cnt:>7} ({cnt / len(samples) * 100:.2f}%)")

    # 按 primary movement type 统计
    mv_counter = Counter(s['primary_type'] for s in samples)
    print(f"\n--- primary movement 分布 ---")
    for mv, cnt in mv_counter.most_common():
        print(f"  {mv:>15}: {cnt:>7} ({cnt / len(samples) * 100:.2f}%)")

    # 联合分布
    joint = Counter((s['video_type'], s['primary_type']) for s in samples)
    print(f"\n--- (video_type, primary_movement) 联合分布 (top 20) ---")
    for (vt, mv), cnt in joint.most_common(20):
        print(f"  ({vt:>8}, {mv:>15}): {cnt:>7}")
    print(f"  ... (共 {len(joint)} 个组合)")

    return vt_counter, mv_counter, joint


def diverse_sample(samples: list, target_size: int, seed: int = 42,
                   cap_ratio: float = 2.5, base_ratio: float = 1.0) -> list:
    """
    分桶均衡采样:
    - 桶 = (video_type, primary_movement_type)
    - base_per_bucket = target / n_buckets * base_ratio
    - cap = base_per_bucket * cap_ratio (单桶上限)
    - 小桶全保留, 大桶随机下采样
    - 如果总和不够 target, 从大桶补足 (但不超 cap)

    参数说明:
      cap_ratio: 单桶上限倍数 (越大越偏向原始分布, 越小越均匀)
                 推荐 2.0~5.0
      base_ratio: 第一轮基础配额倍数
                  > 1.0: 第一轮就能采到更多, 大桶贡献更多
                  推荐保持 1.0
    """
    random.seed(seed)

    # 1. 分桶
    buckets = defaultdict(list)
    for s in samples:
        key = (s['video_type'], s['primary_type'])
        buckets[key].append(s)

    n_buckets = len(buckets)
    print(f"\n=== 采样策略 ===")
    print(f"target_size={target_size}, n_buckets={n_buckets}")

    # 2. 配额计算
    base_per_bucket = max(1, int(target_size / n_buckets * base_ratio))
    cap = max(int(base_per_bucket * cap_ratio), 50)
    print(f"base_per_bucket={base_per_bucket}, cap={cap}")
    print(f"  (cap_ratio={cap_ratio}, base_ratio={base_ratio})")

    # 3. 第一轮: 每个桶取 min(桶大小, base_per_bucket)
    selected = []
    leftover_buckets = []
    for key, items in buckets.items():
        k = min(len(items), base_per_bucket)
        chosen = random.sample(items, k) if k < len(items) else list(items)
        selected.extend(chosen)
        # 还剩下的, 留到下一轮补足
        remaining = [s for s in items if s not in chosen]
        if remaining:
            leftover_buckets.append((key, remaining))

    print(f"第一轮采样: {len(selected)} 条")

    # 4. 第二轮: 用剩余的样本补足到 target_size
    #    优先从已经有 base_per_bucket 但还有余量的桶继续抽 (上限 cap)
    deficit = target_size - len(selected)
    if deficit > 0 and leftover_buckets:
        # 计算每个桶还能补多少 (cap - 已选)
        bucket_used = Counter()
        for s in selected:
            bucket_used[(s['video_type'], s['primary_type'])] += 1

        # 给每个桶分配二次配额: 让大桶多补点 (按桶剩余量比例)
        total_remaining = sum(len(items) for _, items in leftover_buckets)
        for key, items in leftover_buckets:
            if deficit <= 0:
                break
            # 这个桶剩余量占比 × deficit, 上限 cap - 已用
            quota = max(1, int(deficit * len(items) / total_remaining))
            quota = min(quota, cap - bucket_used[key], len(items))
            if quota > 0:
                chosen = random.sample(items, quota) if quota < len(items) else items
                selected.extend(chosen)
                deficit -= quota

    print(f"最终采样: {len(selected)} 条")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="过滤后的 jsonl (可能没有 video_type)")
    parser.add_argument("--video_type_source", type=str, default=None,
                        help="提供 video_type 的 jsonl (按 video_id 匹配)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--target_size", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap_ratio", type=float, default=2.5,
                        help="单桶上限 = base × cap_ratio. 越大越偏向原始分布 (默认 2.5)")
    parser.add_argument("--base_ratio", type=float, default=1.0,
                        help="第一轮基础配额倍数 (默认 1.0)")
    parser.add_argument("--analyze_only", action="store_true",
                        help="只做分布分析, 不采样")
    args = parser.parse_args()

    if not args.analyze_only and args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_diverse_{args.target_size // 1000}k{ext}"

    # 0. 如果指定了 video_type_source, 先建立 video_id → video_type 映射
    video_id_to_type = {}
    if args.video_type_source:
        print(f"读取 video_type 映射: {args.video_type_source}...")
        with open(args.video_type_source, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vt = data.get('video_type', '未知')
                for v in data.get('videos', []):
                    if isinstance(v, str):
                        vid = os.path.splitext(os.path.basename(v))[0]
                        video_id_to_type[vid] = vt
        print(f"  共 {len(video_id_to_type)} 个 video_id 有 video_type")

    # 1. 读取所有样本 (保留原始行 + 统计信息)
    print(f"\n读取 {args.input}...")
    samples = []
    n_no_vtype = 0
    with open(args.input, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            info = get_sample_info(line)
            if info is None:
                continue

            # 如果原始行没有 video_type, 从映射中查
            if info['video_type'] == '未知' and video_id_to_type:
                # 取第一个视频的 vid
                try:
                    data = json.loads(line)
                    videos = data.get('videos', [])
                    if videos and isinstance(videos[0], str):
                        vid = os.path.splitext(os.path.basename(videos[0]))[0]
                        if vid in video_id_to_type:
                            info['video_type'] = video_id_to_type[vid]
                        else:
                            n_no_vtype += 1
                except (json.JSONDecodeError, IndexError):
                    n_no_vtype += 1

            info['_line'] = line
            samples.append(info)
            if (i + 1) % 50000 == 0:
                print(f"  已读 {i+1}...")

    print(f"  读取完成: {len(samples)} 条样本")
    if n_no_vtype > 0:
        print(f"  ⚠️ {n_no_vtype} 条没有匹配到 video_type (按 '未知' 处理)")

    # 2. 统计分布
    analyze(samples)

    if args.analyze_only:
        return

    # 3. 采样
    selected = diverse_sample(samples, args.target_size, seed=args.seed,
                              cap_ratio=args.cap_ratio, base_ratio=args.base_ratio)

    # 4. 统计采样后分布
    print(f"\n=== 采样后分布 ===")
    analyze(selected)

    # 5. 写入: 输出原始行, 同时把 video_type 注入到每条记录中 (便于后续追踪)
    print(f"\n写入 {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for s in selected:
            # 把 video_type 写入原始 json
            try:
                data = json.loads(s['_line'])
                if 'video_type' not in data:
                    data['video_type'] = s['video_type']
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                f.write(s['_line'] + "\n")

    print(f"✅ 完成: {len(selected)} 条 → {args.output}")


if __name__ == "__main__":
    main()
