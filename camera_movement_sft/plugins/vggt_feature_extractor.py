"""
VGGT / VGGT-Omega Camera Token 特征预提取脚本

对训练视频离线提取 Camera Token features，存为 .pt 文件。
训练时通过 video_id 查表获取蒸馏 target，避免实时跑模型。

支持两种 teacher 模型:
  - VGGT (facebook/VGGT-1B): 1.26B 参数, DINOv2, patch=14, img=518
  - VGGT-Omega (facebook/VGGT-Omega): 1.14B 参数, DINOv3, patch=16, img=512

两者 camera token 输出维度相同 (2048)，下游 pipeline 无需修改。

多 GPU 并行支持:
  - 默认使用所有可见 GPU (通过 --num_gpus 控制)
  - 自动将视频列表均匀分片，每张 GPU 处理自己的分片
  - 支持中断续跑 (--skip_existing)

用法:
    # 使用 VGGT (默认), 8卡并行
    python camera_movement_sft/plugins/vggt_feature_extractor.py \
        --input_jsonl camera_movement_sft/train_data/camera_movement_train_human_all_167k.jsonl \
        --output_dir /path/to/vggt_cache/ \
        --teacher vggt \
        --vggt_model facebook/VGGT-1B \
        --fps 5 --max_frames 100 \
        --num_gpus 8

    # 使用 VGGT-Omega, 8卡并行
    python camera_movement_sft/plugins/vggt_feature_extractor.py \
        --input_jsonl camera_movement_sft/train_data/camera_movement_train_human_all_167k.jsonl \
        --output_dir /path/to/vggt_omega_cache/ \
        --teacher vggt_omega \
        --vggt_model facebook/VGGT-Omega \
        --fps 5 --max_frames 100 \
        --num_gpus 8

    # 单卡模式 (兼容旧用法)
    python camera_movement_sft/plugins/vggt_feature_extractor.py \
        --input_jsonl ... --output_dir ... --num_gpus 1 --device cuda:0
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 模型路径
VGGT_REPO = "/group/40009/dazhaodu/vggt"
VGGT_OMEGA_REPO = "/group/40009/dazhaodu/vggt-omega"


def load_vggt_model(model_path: str, device: str = "cuda"):
    """加载 VGGT 模型"""
    import os
    os.environ.setdefault("HF_HOME", "/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/hf")
    os.environ.setdefault("HF_TOKEN", "***REMOVED***")

    if VGGT_REPO not in sys.path:
        sys.path.insert(0, VGGT_REPO)
    from vggt.models.vggt import VGGT

    print(f"[VGGT][{device}] Loading model from {model_path}...")
    if os.path.isfile(model_path):
        # 本地 .pt 文件
        model = VGGT()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        # HuggingFace Hub ID (如 "facebook/VGGT-1B")
        model = VGGT.from_pretrained(model_path)

    model = model.to(device).eval()
    model.requires_grad_(False)
    print(f"[VGGT][{device}] Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model


def load_vggt_omega_model(model_path: str, device: str = "cuda"):
    """加载 VGGT-Omega 模型"""
    import os
    os.environ.setdefault("HF_HOME", "/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/hf")
    os.environ.setdefault("HF_TOKEN", "***REMOVED***")

    if VGGT_OMEGA_REPO not in sys.path:
        sys.path.insert(0, VGGT_OMEGA_REPO)
    from vggt_omega.models.vggt_omega import VGGTOmega

    print(f"[VGGT-Omega][{device}] Loading model from {model_path}...")
    model = VGGTOmega()
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    else:
        # 尝试从 HuggingFace 下载
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=model_path, filename="vggt_omega_1b_512.pt")
        state_dict = torch.load(local_path, map_location="cpu", weights_only=True)

    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    model.requires_grad_(False)
    print(f"[VGGT-Omega][{device}] Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model


def extract_frames_from_video(video_path: str, fps: int = 5, max_frames: int = 100):
    """
    从视频抽帧，与 Qwen3-VL 训练时一致的逻辑。

    Returns:
        frames: numpy array (S, H, W, 3) in [0, 255] uint8
        或 None (如果视频无法读取)
    """
    try:
        import decord
        decord.bridge.set_bridge("native")

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        duration = total_frames / video_fps

        # 计算抽帧数 (与训练一致: nframes = min(duration * fps, max_frames))
        n_frames = min(int(duration * fps), max_frames)
        n_frames = max(n_frames, 4)  # 最少 4 帧

        # 均匀抽帧
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # (S, H, W, 3)

        return frames

    except Exception as e:
        print(f"  Error reading {video_path}: {e}")
        return None


def preprocess_for_vggt(frames: np.ndarray, img_size: int = 518) -> torch.Tensor:
    """
    预处理帧用于 VGGT 输入 (518px, patch=14)

    Args:
        frames: (S, H, W, 3) uint8 [0, 255]

    Returns:
        tensor: (1, S, 3, img_size, img_size) float32 [0, 1]
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    processed = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i])
        img = TF.resize(img, [img_size, img_size])
        tensor = TF.to_tensor(img)  # (3, H, W) [0, 1]
        processed.append(tensor)

    batch = torch.stack(processed, dim=0)  # (S, 3, H, W)
    return batch.unsqueeze(0)  # (1, S, 3, H, W)


def preprocess_for_vggt_omega(frames: np.ndarray, img_size: int = 512) -> torch.Tensor:
    """
    预处理帧用于 VGGT-Omega 输入 (512px, patch=16)

    VGGT-Omega 使用 "balanced" resize 模式:
    - 短边 resize 到 img_size，保持宽高比
    - 然后 center crop 到 img_size x img_size

    Args:
        frames: (S, H, W, 3) uint8 [0, 255]

    Returns:
        tensor: (1, S, 3, img_size, img_size) float32 [0, 1]
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    processed = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i])
        # Resize shortest side to img_size, then center crop
        w, h = img.size
        scale = img_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, [new_h, new_w])
        img = TF.center_crop(img, [img_size, img_size])
        tensor = TF.to_tensor(img)  # (3, H, W) [0, 1]
        processed.append(tensor)

    batch = torch.stack(processed, dim=0)  # (S, 3, H, W)
    return batch.unsqueeze(0)  # (1, S, 3, H, W)


def extract_features_for_video(
    model,
    video_path: str,
    teacher_type: str = "vggt",
    fps: int = 5,
    max_frames: int = 100,
    device: str = "cuda",
    save_pose: bool = False,
) -> dict:
    """
    对单个视频提取 camera token features 和（可选的）解码后的 pose。

    Args:
        model: VGGT 或 VGGT-Omega 模型
        teacher_type: "vggt" 或 "vggt_omega"
        save_pose: 是否同时运行 Camera Head 获取解码后的 9D pose

    Returns:
        dict with:
          "camera_features": (S, 2048) float16 — Aggregator 最后一层 camera token
          "pose_enc": (S, 9) float32 — 解码后的 9D pose [T(3) + Quat(4) + FoV(2)]
                      (仅当 save_pose=True 时存在)
        或 None (视频读取失败)
    """
    # 1. 抽帧
    frames = extract_frames_from_video(video_path, fps=fps, max_frames=max_frames)
    if frames is None:
        return None

    # 2. 预处理（不同模型不同尺寸）
    if teacher_type == "vggt_omega":
        images = preprocess_for_vggt_omega(frames, img_size=512).to(device)
    else:
        images = preprocess_for_vggt(frames, img_size=518).to(device)

    # 3. 前向推理
    result = {}
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        if save_pose:
            # 跑完整 model.forward() 获取 pose_enc
            predictions = model(images)
            if teacher_type == "vggt_omega":
                if "camera_and_register_tokens" in predictions:
                    camera_features = predictions["camera_and_register_tokens"][:, :, 0, :]
                else:
                    output_list, _ = model.aggregator(images)
                    camera_features = output_list[-1][:, :, 0, :]
                pose_enc = predictions.get("pose_enc")  # (1, S, 9)
            else:
                # VGGT: forward 返回 predictions dict
                camera_features = None
                pose_enc = predictions.get("pose_enc")  # (1, S, 9) — 最后一次迭代
                # 需要从 aggregator 的输出拿 camera_features
                # VGGT forward 内部已经跑了 aggregator，但没暴露中间结果
                # 重新跑 aggregator 取 camera token
                output_list, _ = model.aggregator(images)
                camera_features = output_list[-1][:, :, 0, :]
        else:
            # 只跑 aggregator（不跑 camera head，节省计算）
            if teacher_type == "vggt_omega":
                predictions = model(images)
                if "camera_and_register_tokens" in predictions:
                    camera_features = predictions["camera_and_register_tokens"][:, :, 0, :]
                else:
                    output_list, _ = model.aggregator(images)
                    camera_features = output_list[-1][:, :, 0, :]
            else:
                output_list, _ = model.aggregator(images)
                camera_features = output_list[-1][:, :, 0, :]
            pose_enc = None

    result["camera_features"] = camera_features.squeeze(0).half().cpu()  # (S, 2048)
    if pose_enc is not None:
        result["pose_enc"] = pose_enc.squeeze(0).float().cpu()  # (S, 9)

    return result


def find_video_path(video_id: str, video_dirs: list) -> str:
    """在多个目录中查找视频文件"""
    for vdir in video_dirs:
        path = os.path.join(vdir, f"{video_id}.mp4")
        if os.path.exists(path):
            return path
    return None


# ============================================================
# 多 GPU Worker
# ============================================================

def gpu_worker(gpu_id: int, args, video_id_shard: list):
    """
    单个 GPU worker 进程。加载模型到指定 GPU，处理分配给它的 video_id 列表。

    Args:
        gpu_id: GPU 编号 (0-7)
        args: 命令行参数
        video_id_shard: 该 worker 需要处理的 video_id 列表
    """
    device = f"cuda:{gpu_id}"

    # 设置当前进程使用的 GPU
    torch.cuda.set_device(gpu_id)

    # 加载模型
    if args.teacher == "vggt_omega":
        model = load_vggt_omega_model(args.vggt_model, device)
    else:
        model = load_vggt_model(args.vggt_model, device)

    # 逐个处理
    success = 0
    failed = 0
    skipped = 0

    pbar = tqdm(
        sorted(video_id_shard),
        desc=f"[GPU {gpu_id}] {args.teacher}",
        position=gpu_id,
        leave=True,
    )

    for video_id in pbar:
        output_path = os.path.join(args.output_dir, f"{video_id}.pt")

        # 跳过已有文件
        if args.skip_existing and os.path.exists(output_path):
            skipped += 1
            continue

        # 视频路径: 严格使用 jsonl 里的路径
        video_path = getattr(args, '_video_id_to_path', {}).get(video_id)
        if video_path is None or not os.path.exists(video_path):
            failed += 1
            continue

        # 提取特征
        try:
            result = extract_features_for_video(
                model,
                video_path,
                teacher_type=args.teacher,
                fps=args.fps,
                max_frames=args.max_frames,
                device=device,
                save_pose=args.save_pose,
            )

            if result is not None:
                torch.save(result, output_path)
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {video_id}: {e}")
            failed += 1
            # 清理 GPU 显存碎片
            torch.cuda.empty_cache()

        pbar.set_postfix(ok=success, fail=failed, skip=skipped)

    print(f"[GPU {gpu_id}] Done! Success: {success}, Failed: {failed}, Skipped: {skipped}")
    return success, failed


def gpu_worker_wrapper(args_tuple):
    """mp.spawn 的入口包装（接收 tuple 解包）"""
    gpu_id, args, video_id_shard = args_tuple
    return gpu_worker(gpu_id, args, video_id_shard)


# ============================================================
# 主流程
# ============================================================

def cmd_extract(args):
    """提取特征的主流程（支持多 GPU 并行）"""

    # 默认模型路径
    if args.vggt_model is None:
        if args.teacher == "vggt":
            args.vggt_model = "facebook/VGGT-1B"
        else:
            args.vggt_model = "facebook/VGGT-Omega"

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 读取训练数据，提取所有 video_id 和对应路径
    print(f"Reading {args.input_jsonl}...")
    video_ids = set()
    video_id_to_path = {}  # video_id -> 来自 jsonl 的绝对路径 (优先使用)
    with open(args.input_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # 从 videos 字段提取 video_id
            videos = data.get("videos", [])
            for v in videos:
                if isinstance(v, str):
                    vid = os.path.splitext(os.path.basename(v))[0]
                    video_ids.add(vid)
                    # 保留 jsonl 里的原始路径 (优先于 video_dirs 查找)
                    if os.path.isabs(v) and vid not in video_id_to_path:
                        video_id_to_path[vid] = v

    # 注入到 args, 让 gpu_worker 能访问
    args._video_id_to_path = video_id_to_path
    print(f"  Found {len(video_id_to_path)} videos with absolute paths in jsonl")

    print(f"Total unique videos: {len(video_ids)}")
    if args.save_pose:
        print(f"Mode: save camera_features (2048D) + pose_enc (9D: Translation+Quaternion+FoV)")
    else:
        print(f"Mode: save camera_features (2048D) only")

    # 检查已完成的
    if args.skip_existing:
        existing = set(f[:-3] for f in os.listdir(args.output_dir) if f.endswith(".pt"))
        todo = sorted(video_ids - existing)
        print(f"Already cached: {len(existing)}, remaining: {len(todo)}")
    else:
        todo = sorted(video_ids)

    if len(todo) == 0:
        print("All videos already processed! Nothing to do.")
        return

    # 确定使用的 GPU 数量
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus <= 0:
        print("ERROR: No CUDA GPUs available!")
        return

    print(f"\n{'='*60}")
    print(f"Parallel extraction with {num_gpus} GPUs")
    print(f"Videos per GPU: ~{len(todo) // num_gpus}")
    print(f"Estimated speedup: {num_gpus}x (from ~{len(todo) * 6 / 3600:.1f}h to ~{len(todo) * 6 / 3600 / num_gpus:.1f}h)")
    print(f"{'='*60}\n")

    # 单 GPU 模式: 直接在当前进程跑 (无需 spawn)
    if num_gpus == 1:
        gpu_id = int(args.device.replace("cuda:", "")) if ":" in args.device else 0
        gpu_worker(gpu_id, args, todo)
        _print_summary(args)
        return

    # 多 GPU 模式: 均匀分片
    shards = [[] for _ in range(num_gpus)]
    for i, vid in enumerate(todo):
        shards[i % num_gpus].append(vid)

    for i, shard in enumerate(shards):
        print(f"  GPU {i}: {len(shard)} videos")

    # 启动多进程
    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, args, shards[gpu_id]),
        )
        p.start()
        processes.append(p)
        # 错开启动，避免同时下载模型权重
        if gpu_id == 0:
            time.sleep(5)

    # 等待所有进程完成
    for p in processes:
        p.join()

    _print_summary(args)


def _print_summary(args):
    """打印最终汇总信息"""
    total_cached = len([f for f in os.listdir(args.output_dir) if f.endswith(".pt")])
    print(f"\n{'='*60}")
    print(f"All GPUs finished!")
    print(f"Cache dir: {args.output_dir}")
    print(f"Total cached: {total_cached} files")
    print(f"\nSaved format per .pt file:")
    print(f"  result['camera_features']: (S, 2048) float16 — Aggregator camera token")
    if args.save_pose:
        print(f"  result['pose_enc']:         (S, 9) float32 — Decoded pose")
        print(f"    [0:3] = Translation (absolute, world coords)")
        print(f"    [3:7] = Quaternion rotation (wxyz or xyzw)")
        print(f"    [7:9] = Field of View (horizontal, vertical)")
    print(f"\n  训练时使用:")
    print(f"    VGGT_CACHE_DIR={args.output_dir} bash camera_movement_sft/train_camdistill.sh qwen3vl-8b")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="VGGT / VGGT-Omega Camera Token 特征预提取 (多 GPU 并行)")
    parser.add_argument("--input_jsonl", type=str, required=True, help="训练数据 JSONL")
    parser.add_argument("--output_dir", type=str, required=True, help="输出缓存目录")
    parser.add_argument("--teacher", type=str, default="vggt", choices=["vggt", "vggt_omega"],
                        help="Teacher 模型: vggt (518px) 或 vggt_omega (512px)")
    parser.add_argument("--vggt_model", type=str, default=None,
                        help="模型路径或 HF ID (默认: vggt->facebook/VGGT-1B)")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="单 GPU 模式时使用的设备 (仅 --num_gpus 1 时生效)")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="使用的 GPU 数量 (默认 8)")
    parser.add_argument("--save_pose", action="store_true", default=False,
                        help="同时保存 Camera Head 解码出的 9D pose")
    parser.add_argument("--video_dirs", type=str, nargs="+", default=[
        "/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos",
        "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_21k_video_shot_cameraMove/videos",
        "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_realfilm_30w_260401_deduped_final/videos",
    ])
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已存在的缓存文件 (支持中断续跑)")
    args = parser.parse_args()
    cmd_extract(args)


if __name__ == "__main__":
    main()
