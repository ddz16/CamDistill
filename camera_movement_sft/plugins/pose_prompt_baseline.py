"""
Pose-Prompt Baseline: 将 VGGT/VGGT-Omega 预测的相机位姿以文本形式注入 prompt

无需训练！直接对已训练好的 SFT 模型做推理时，在 user prompt 中加入每帧的相机位姿信息。
验证相机空间信息是否能通过纯文本方式帮助运镜识别。

用法:
    # Step 1: 提取 pose (如果还没有)
    python camera_movement_sft/plugins/pose_prompt_baseline.py extract_pose \
        --input_jsonl test_data.jsonl \
        --output_dir /path/to/pose_cache/ \
        --teacher vggt

    # Step 2: 生成带 pose prompt 的推理数据
    python camera_movement_sft/plugins/pose_prompt_baseline.py generate_infer_data \
        --input_jsonl test_data.jsonl \
        --pose_dir /path/to/pose_cache/ \
        --output_jsonl test_with_pose.jsonl

    # Step 3: 用已有 SFT 模型推理
    swift infer --model <checkpoint> --val_dataset test_with_pose.jsonl ...
"""

import argparse
import json
import math
import os
import sys
import numpy as np
import torch
from typing import List, Tuple, Optional

# 路径
VGGT_REPO = "/group/40009/dazhaodu/vggt"
VGGT_OMEGA_REPO = "/group/40009/dazhaodu/vggt-omega"

# 添加 common.py 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from camera_movement_sft.common import SYSTEM_PROMPT, USER_PROMPT


def quaternion_to_euler_degrees(quat: np.ndarray) -> Tuple[float, float, float]:
    """四元数 [w, x, y, z] 或 [x, y, z, w] → 欧拉角 (度)"""
    # VGGT 使用 [x, y, z, w] 格式 (Hamilton convention)
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def format_pose_as_text(pose_sequence: np.ndarray, fps: int = 5) -> str:
    """
    将 pose 序列格式化为人类可读的文本描述

    Args:
        pose_sequence: (S, 9) numpy array — [T(3), quat(4), FoV(2)] per frame
        fps: 帧率（用于计算时间戳）

    Returns:
        str: 格式化的相机运动文本描述
    """
    S = pose_sequence.shape[0]
    if S < 2:
        return ""

    lines = []
    lines.append("以下是由3D视觉模型估计的逐帧相机位姿信息（供参考）：")

    for i in range(S):
        time_sec = i / fps
        T = pose_sequence[i, :3]
        quat = pose_sequence[i, 3:7]
        fov = pose_sequence[i, 7:9]

        # 转欧拉角
        roll, pitch, yaw = quaternion_to_euler_degrees(quat)

        # 相对于第一帧的位移
        if i == 0:
            T_rel = np.zeros(3)
        else:
            T_rel = T - pose_sequence[0, :3]

        lines.append(
            f"  t={time_sec:.1f}s: "
            f"位移=({T_rel[0]:.2f},{T_rel[1]:.2f},{T_rel[2]:.2f}), "
            f"旋转=(roll={roll:.1f}°,pitch={pitch:.1f}°,yaw={yaw:.1f}°), "
            f"FoV=({math.degrees(fov[0]):.1f}°,{math.degrees(fov[1]):.1f}°)"
        )

    # 添加整体运动摘要
    total_translation = np.linalg.norm(pose_sequence[-1, :3] - pose_sequence[0, :3])
    lines.append(f"  总位移量: {total_translation:.3f}")

    return "\n".join(lines)


def extract_pose_for_video(
    model,
    video_path: str,
    teacher_type: str = "vggt",
    fps: int = 5,
    max_frames: int = 100,
    device: str = "cuda",
) -> Optional[np.ndarray]:
    """
    对单个视频提取完整 pose encoding (经过 Camera Head)

    Returns:
        pose: (S, 9) numpy array, 或 None
    """
    from vggt_feature_extractor import extract_frames_from_video, preprocess_for_vggt, preprocess_for_vggt_omega

    frames = extract_frames_from_video(video_path, fps=fps, max_frames=max_frames)
    if frames is None:
        return None

    if teacher_type == "vggt_omega":
        images = preprocess_for_vggt_omega(frames, img_size=512).to(device)
    else:
        images = preprocess_for_vggt(frames, img_size=518).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        predictions = model(images)
        pose_enc = predictions["pose_enc"]  # (1, S, 9)

    return pose_enc.squeeze(0).float().cpu().numpy()  # (S, 9)


def cmd_extract_pose(args):
    """子命令: 提取 pose"""
os.environ.setdefault("HF_HOME", "/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/hf")
    os.environ.setdefault("HF_TOKEN", "***REMOVED***")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vggt_feature_extractor import (
        load_vggt_model, load_vggt_omega_model, find_video_path
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型（完整模型含 camera head）
    if args.teacher == "vggt_omega":
        if VGGT_OMEGA_REPO not in sys.path:
            sys.path.insert(0, VGGT_OMEGA_REPO)
        from vggt_omega.models.vggt_omega import VGGTOmega
        model = VGGTOmega()
        model_path = args.vggt_model or "/group/40009/dazhaodu/vggt-omega/checkpoints/vggt_omega_1b_512.pt"
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(args.device).eval()
        model.requires_grad_(False)
    else:
        if VGGT_REPO not in sys.path:
            sys.path.insert(0, VGGT_REPO)
        from vggt.models.vggt import VGGT
        model_path = args.vggt_model or "facebook/VGGT-1B"
        if os.path.isfile(model_path):
            model = VGGT()
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        else:
            model = VGGT.from_pretrained(model_path)
        model = model.to(args.device).eval()
        model.requires_grad_(False)

    print(f"[Pose-Prompt] Loaded {args.teacher} model")

    # 读取 video_ids
    video_ids = set()
    with open(args.input_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            vid = data.get("video_id", "")
            if not vid:
                # 尝试从 videos 字段提取
                videos = data.get("videos", [])
                for v in videos:
                    if isinstance(v, str):
                        vid = os.path.splitext(os.path.basename(v))[0]
                        break
            if vid:
                video_ids.add(vid)

    # 跳过已存在
    existing = set(f[:-3] for f in os.listdir(args.output_dir) if f.endswith(".pt"))
    todo = video_ids - existing
    print(f"[Pose-Prompt] Total: {len(video_ids)}, existing: {len(existing)}, todo: {len(todo)}")

    from tqdm import tqdm
    success, failed = 0, 0
    for vid in tqdm(sorted(todo), desc="Extracting pose"):
        video_path = find_video_path(vid, args.video_dirs)
        if not video_path:
            failed += 1
            continue

        pose = extract_pose_for_video(model, video_path, args.teacher, args.fps, args.max_frames, args.device)
        if pose is not None:
            torch.save(torch.from_numpy(pose).half(), os.path.join(args.output_dir, f"{vid}.pt"))
            success += 1
        else:
            failed += 1

    print(f"[Pose-Prompt] Done. Success: {success}, Failed: {failed}")


def cmd_generate_infer_data(args):
    """子命令: 生成带 pose 信息的推理数据"""
    pose_cache = {}

    # 读取原始数据并注入 pose
    output_lines = []
    skipped = 0

    with open(args.input_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # 提取 video_id
            vid = data.get("video_id", "")
            cos_url = data.get("cos_url", "")
            if not vid and cos_url:
                vid = os.path.splitext(os.path.basename(cos_url))[0]

            # 加载 pose
            pose_path = os.path.join(args.pose_dir, f"{vid}.pt")
            if not os.path.exists(pose_path):
                skipped += 1
                continue

            pose_tensor = torch.load(pose_path, map_location="cpu", weights_only=True)
            pose_np = pose_tensor.float().numpy()  # (S, 9)

            # 格式化 pose 文本
            pose_text = format_pose_as_text(pose_np, fps=args.fps)

            # 构建带 pose 的 swift 推理格式
            # 找到视频路径
            video_path = None
            for vdir in args.video_dirs:
                p = os.path.join(vdir, f"{vid}.mp4")
                if os.path.exists(p):
                    video_path = p
                    break

            if not video_path:
                skipped += 1
                continue

            # 构建 messages: system + user(含pose) + (无 assistant，让模型生成)
            user_content = f"<video>\n\n{pose_text}\n\n{USER_PROMPT}"

            output_item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "videos": [video_path],
            }
            output_lines.append(json.dumps(output_item, ensure_ascii=False))

    # 写出
    with open(args.output_jsonl, "w") as f:
        f.write("\n".join(output_lines) + "\n")

    print(f"[Pose-Prompt] Generated {len(output_lines)} items, skipped {skipped}")
    print(f"[Pose-Prompt] Output: {args.output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Pose-Prompt Baseline")
    subparsers = parser.add_subparsers(dest="command")

    # extract_pose
    p1 = subparsers.add_parser("extract_pose", help="提取 VGGT/Omega pose encoding")
    p1.add_argument("--input_jsonl", type=str, required=True)
    p1.add_argument("--output_dir", type=str, required=True)
    p1.add_argument("--teacher", type=str, default="vggt", choices=["vggt", "vggt_omega"])
    p1.add_argument("--vggt_model", type=str, default=None)
    p1.add_argument("--fps", type=int, default=5)
    p1.add_argument("--max_frames", type=int, default=100)
    p1.add_argument("--device", type=str, default="cuda")
    p1.add_argument("--video_dirs", type=str, nargs="+", default=[
        "/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos",
        "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_21k_video_shot_cameraMove/videos",
        "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_realfilm_30w_260401_deduped_final/videos",
    ])

    # generate_infer_data
    p2 = subparsers.add_parser("generate_infer_data", help="生成带 pose 的推理数据")
    p2.add_argument("--input_jsonl", type=str, required=True)
    p2.add_argument("--pose_dir", type=str, required=True)
    p2.add_argument("--output_jsonl", type=str, required=True)
    p2.add_argument("--fps", type=int, default=5)
    p2.add_argument("--video_dirs", type=str, nargs="+", default=[
        "/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos",
        "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_21k_video_shot_cameraMove/videos",
        "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_realfilm_30w_260401_deduped_final/videos",
    ])

    args = parser.parse_args()
    if args.command == "extract_pose":
        cmd_extract_pose(args)
    elif args.command == "generate_infer_data":
        cmd_generate_infer_data(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
