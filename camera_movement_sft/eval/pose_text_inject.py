#!/usr/bin/env python3
"""
Pose-Text 推理数据构建：VGGT-Omega 相机位姿 -> 文本 -> 注入 prompt

方案：用 VGGT-Omega 在线估计逐帧相机位姿(9D: 平移3 + 四元数4 + FoV2)，把位姿以逐帧文本
形式加进 user prompt，然后用 base / SFT 模型推理评测。纯文本注入，不改模型结构、不用插件。

两个子命令：
  extract : 对 video_id_mapping.jsonl 里每个视频在线跑 VGGT-Omega，存 9D pose_enc 到
            {pose_dir}/{video_id}.pt（跳过已存在，可断点续跑）。
  inject  : 把位姿文本注入标准 test_for_infer.jsonl 的 user 消息(在首个 <video> 之后)，
            输出 test_for_infer_pose.jsonl（system / assistant(gt) 原样保留，评测标签仍对齐）。

pose->text 逻辑内置(不依赖 plugins/pose_prompt_baseline.py)；抽帧/预处理/模型加载复用
plugins/vggt_feature_extractor.py。

用法：
  python pose_text_inject.py extract \
      --mapping <run_dir>/video_id_mapping.jsonl \
      --pose_dir <pose_cache_dir> \
      --video_dir /group/40009/dazhaodu/OurBenchmark/YouTube_Videos [more dirs...] \
      --vggt_model facebook/VGGT-Omega --fps 5 --max_frames 100

  python pose_text_inject.py inject \
      --test_jsonl <run_dir>/test_for_infer.jsonl \
      --mapping <run_dir>/video_id_mapping.jsonl \
      --pose_dir <pose_cache_dir> \
      --output <run_dir>/test_for_infer_pose.jsonl --fps 5 --on_missing keep
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# plugins 目录(复用 vggt_feature_extractor)
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_EVAL_DIR)
_PLUGINS_DIR = os.path.join(_PROJECT_DIR, "plugins")


# ============================================================================
# 位姿 -> 文本 (逐帧完整格式)
# ============================================================================
def quaternion_to_euler_degrees(quat: np.ndarray):
    """四元数 [x, y, z, w] (VGGT/Hamilton) -> 欧拉角 (roll, pitch, yaw) 单位度"""
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    # Roll (x)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (y)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    # Yaw (z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def format_pose_as_text(pose_sequence: np.ndarray, fps: int = 5) -> str:
    """(S, 9) pose_enc = [T(3), quat(4), FoV(2)] -> 逐帧人类可读文本"""
    if pose_sequence is None or pose_sequence.ndim != 2 or pose_sequence.shape[0] < 2:
        return ""
    S = pose_sequence.shape[0]
    lines = ["以下是由3D视觉模型(VGGT)估计的逐帧相机位姿信息（仅供参考，可能存在误差）："]
    origin = pose_sequence[0, :3]
    for i in range(S):
        t_sec = i / float(fps)
        T = pose_sequence[i, :3]
        quat = pose_sequence[i, 3:7]
        fov = pose_sequence[i, 7:9]
        roll, pitch, yaw = quaternion_to_euler_degrees(quat)
        T_rel = np.zeros(3) if i == 0 else (T - origin)
        lines.append(
            f"  t={t_sec:.1f}s: "
            f"位移=({T_rel[0]:.2f},{T_rel[1]:.2f},{T_rel[2]:.2f}), "
            f"旋转=(roll={roll:.1f}°,pitch={pitch:.1f}°,yaw={yaw:.1f}°), "
            f"FoV=({math.degrees(float(fov[0])):.1f}°,{math.degrees(float(fov[1])):.1f}°)"
        )
    total_translation = float(np.linalg.norm(pose_sequence[-1, :3] - pose_sequence[0, :3]))
    lines.append(f"  总位移量: {total_translation:.3f}")
    return "\n".join(lines)


# ============================================================================
# extract：在线跑 VGGT-Omega 估位姿
# ============================================================================
def _resolve_video_path(video_id: str, cos_url: str, video_dirs):
    """优先用 cos_url(本地绝对路径)，否则在候选目录里按 {video_id}.mp4 查找。"""
    if cos_url and os.path.isabs(cos_url) and os.path.exists(cos_url):
        return cos_url
    for d in (video_dirs or []):
        p = os.path.join(d, f"{video_id}.mp4")
        if os.path.exists(p):
            return p
    return None


def _read_mapping(mapping_path: str):
    """读取 video_id_mapping.jsonl -> [(video_id, cos_url), ...] (保序、去重保留首次)"""
    items = []
    seen = set()
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            vid = d.get("video_id", "")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            items.append((vid, d.get("cos_url", "")))
    return items


def cmd_extract(args):
    import torch

    os.makedirs(args.pose_dir, exist_ok=True)
    if _PLUGINS_DIR not in sys.path:
        sys.path.insert(0, _PLUGINS_DIR)
    from vggt_feature_extractor import (
        load_vggt_omega_model,
        extract_frames_from_video,
        preprocess_for_vggt_omega,
    )

    mapping = _read_mapping(args.mapping)
    todo = [(vid, url) for vid, url in mapping
            if not os.path.exists(os.path.join(args.pose_dir, f"{vid}.pt"))]
    print(f"[pose-extract] 视频总数: {len(mapping)}, 待提取: {len(todo)} "
          f"(已缓存 {len(mapping) - len(todo)})")
    if not todo:
        print("[pose-extract] 全部已缓存，跳过。")
        return

    print(f"[pose-extract] 加载 VGGT-Omega: {args.vggt_model}")
    model = load_vggt_omega_model(args.vggt_model, device=args.device)

    ok, miss_video, fail = 0, 0, 0
    try:
        from tqdm import tqdm
        it = tqdm(todo, desc="extract pose")
    except Exception:
        it = todo

    for vid, cos_url in it:
        vpath = _resolve_video_path(vid, cos_url, args.video_dir)
        if not vpath:
            miss_video += 1
            continue
        try:
            frames = extract_frames_from_video(vpath, fps=args.fps, max_frames=args.max_frames)
            if frames is None:
                fail += 1
                continue
            images = preprocess_for_vggt_omega(frames, img_size=512).to(args.device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(images)
                pose_enc = pred["pose_enc"]  # (1, S, 9)
            pose = pose_enc.squeeze(0).float().cpu().half()  # (S, 9) fp16
            torch.save(pose, os.path.join(args.pose_dir, f"{vid}.pt"))
            ok += 1
        except Exception as e:
            print(f"[pose-extract] {vid} 失败: {type(e).__name__}: {str(e)[:150]}")
            fail += 1

    print(f"[pose-extract] 完成。成功 {ok}, 视频缺失 {miss_video}, 提取失败 {fail} "
          f"-> {args.pose_dir}")


# ============================================================================
# inject：把位姿文本注入 prompt
# ============================================================================
def cmd_inject(args):
    import torch

    # test_for_infer.jsonl 与 video_id_mapping.jsonl 由 prepare_test_data.py 同循环生成，逐行对齐
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        test_lines = [ln for ln in (l.rstrip("\n") for l in f) if ln.strip()]
    map_ids = []
    with open(args.mapping, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                map_ids.append(json.loads(line).get("video_id", ""))

    if len(test_lines) != len(map_ids):
        print(f"[pose-inject] 警告: test 行数({len(test_lines)}) != mapping 行数({len(map_ids)})，"
              f"按较短长度对齐。")
    n = min(len(test_lines), len(map_ids))

    out_lines = []
    injected, missing = 0, 0
    for i in range(n):
        sample = json.loads(test_lines[i])
        vid = map_ids[i]
        pose_path = os.path.join(args.pose_dir, f"{vid}.pt")

        pose_text = ""
        if os.path.exists(pose_path):
            pose = torch.load(pose_path, map_location="cpu", weights_only=True)
            pose_text = format_pose_as_text(pose.float().numpy(), fps=args.fps)

        if not pose_text:
            missing += 1
            if args.on_missing == "skip":
                continue
            # keep: 原样保留(不注入位姿)
            out_lines.append(json.dumps(sample, ensure_ascii=False))
            continue

        # 在首个 user 消息的第一个 <video> 之后插入位姿文本
        block = f"\n\n{pose_text}\n\n"
        done = False
        for msg in sample.get("messages", []):
            if msg.get("role") == "user" and "<video>" in msg.get("content", ""):
                msg["content"] = msg["content"].replace("<video>", "<video>" + block, 1)
                done = True
                break
        if done:
            injected += 1
        else:
            # 没找到 <video>，保底注入到 user 开头
            for msg in sample.get("messages", []):
                if msg.get("role") == "user":
                    msg["content"] = pose_text + "\n\n" + msg.get("content", "")
                    injected += 1
                    done = True
                    break
            if not done:
                missing += 1
        out_lines.append(json.dumps(sample, ensure_ascii=False))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"[pose-inject] 输出 {len(out_lines)} 条 (注入位姿 {injected}, 缺位姿 {missing}, "
          f"on_missing={args.on_missing}) -> {args.output}")


# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Pose-Text 推理数据构建 (VGGT-Omega 位姿 -> 文本 -> prompt)")
    sub = parser.add_subparsers(dest="command")

    pe = sub.add_parser("extract", help="在线跑 VGGT-Omega 估位姿并缓存")
    pe.add_argument("--mapping", required=True, help="video_id_mapping.jsonl (prepare_test_data 生成)")
    pe.add_argument("--pose_dir", required=True, help="位姿缓存输出目录 ({video_id}.pt)")
    pe.add_argument("--video_dir", nargs="+", default=[
        "/group/40009/dazhaodu/OurBenchmark/YouTube_Videos",
    ], help="本地视频候选目录 (cos_url 为本地绝对路径时优先用 cos_url)")
    pe.add_argument("--vggt_model", default=os.environ.get("VGGT_MODEL_PATH", "facebook/VGGT-Omega"),
                    help="VGGT-Omega 权重: 本地 .pt 或 HF repo (默认 facebook/VGGT-Omega, 命中 HF 缓存)")
    pe.add_argument("--fps", type=int, default=int(os.environ.get("FPS", "5")))
    pe.add_argument("--max_frames", type=int, default=int(os.environ.get("FPS_MAX_FRAMES", "100")))
    pe.add_argument("--device", default="cuda")

    pi = sub.add_parser("inject", help="把位姿文本注入 test_for_infer.jsonl")
    pi.add_argument("--test_jsonl", required=True, help="标准 test_for_infer.jsonl")
    pi.add_argument("--mapping", required=True, help="video_id_mapping.jsonl (与 test_jsonl 逐行对齐)")
    pi.add_argument("--pose_dir", required=True, help="位姿缓存目录 ({video_id}.pt)")
    pi.add_argument("--output", required=True, help="输出 test_for_infer_pose.jsonl")
    pi.add_argument("--fps", type=int, default=int(os.environ.get("FPS", "5")))
    pi.add_argument("--on_missing", choices=["keep", "skip"], default="keep",
                    help="缺位姿时: keep=原样保留(不注入), skip=丢弃该样本 (默认 keep)")

    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "inject":
        cmd_inject(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
