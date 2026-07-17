"""
VGGT / VGGT-Omega Camera Token feature pre-extraction script.

Extracts Camera Token features from training videos offline and saves them as .pt files.
During training we look up the distillation target by video_id to avoid running the model
online.

Two teacher models are supported:
  - VGGT (facebook/VGGT-1B): 1.26B parameters, DINOv2, patch=14, img=518
  - VGGT-Omega (facebook/VGGT-Omega): 1.14B parameters, DINOv3, patch=16, img=512

Both emit camera tokens with the same output dimension (2048); downstream pipelines need
no changes.

Multi-GPU support:
  - Uses all visible GPUs by default (controlled by --num_gpus).
  - Automatically shards the video list evenly; each GPU processes its own shard.
  - Supports resuming an interrupted run (--skip_existing).

Usage:
    # VGGT (default), 8 GPUs in parallel
    python camera_movement_sft/plugins/vggt_feature_extractor.py \
        --input_jsonl camera_movement_sft/train_data/camera_movement_train_human_all_167k.jsonl \
        --output_dir /path/to/vggt_cache/ \
        --teacher vggt \
        --vggt_model facebook/VGGT-1B \
        --fps 5 --max_frames 100 \
        --num_gpus 8

    # VGGT-Omega, 8 GPUs in parallel
    python camera_movement_sft/plugins/vggt_feature_extractor.py \
        --input_jsonl camera_movement_sft/train_data/camera_movement_train_human_all_167k.jsonl \
        --output_dir /path/to/vggt_omega_cache/ \
        --teacher vggt_omega \
        --vggt_model facebook/VGGT-Omega \
        --fps 5 --max_frames 100 \
        --num_gpus 8

    # Single-GPU mode (legacy usage)
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

# Model paths.
VGGT_REPO = "/group/40009/dazhaodu/vggt"
VGGT_OMEGA_REPO = "/group/40009/dazhaodu/vggt-omega"


def load_vggt_model(model_path: str, device: str = "cuda"):
    """Load the VGGT model."""
    import os
    os.environ.setdefault("HF_HOME", "/apdcephfs_gy2/share_303094921/hunyuan/yujiazhang/dazhaodu/hf")
    os.environ.setdefault("HF_TOKEN", "***REMOVED***")

    if VGGT_REPO not in sys.path:
        sys.path.insert(0, VGGT_REPO)
    from vggt.models.vggt import VGGT

    print(f"[VGGT][{device}] Loading model from {model_path}...")
    if os.path.isfile(model_path):
        # Local .pt file.
        model = VGGT()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        # HuggingFace Hub id (e.g. "facebook/VGGT-1B").
        model = VGGT.from_pretrained(model_path)

    model = model.to(device).eval()
    model.requires_grad_(False)
    print(f"[VGGT][{device}] Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model


def load_vggt_omega_model(model_path: str, device: str = "cuda"):
    """Load the VGGT-Omega model."""
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
        # Try downloading from HuggingFace.
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
    Extract frames from a video, mirroring the logic used during Qwen3-VL training.

    Returns:
        frames: numpy array (S, H, W, 3) in [0, 255] uint8, or None if the video cannot be read.
    """
    try:
        import decord
        decord.bridge.set_bridge("native")

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        duration = total_frames / video_fps

        # Compute the sampling count (aligned with training: nframes = min(duration * fps, max_frames)).
        n_frames = min(int(duration * fps), max_frames)
        n_frames = max(n_frames, 4)  # at least 4 frames

        # Evenly spaced sampling.
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # (S, H, W, 3)

        return frames

    except Exception as e:
        print(f"  Error reading {video_path}: {e}")
        return None


def preprocess_for_vggt(frames: np.ndarray, img_size: int = 518) -> torch.Tensor:
    """
    Preprocess frames for VGGT input (518px, patch=14).

    Args:
        frames: (S, H, W, 3) uint8 [0, 255].

    Returns:
        tensor: (1, S, 3, img_size, img_size) float32 [0, 1].
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
    Preprocess frames for VGGT-Omega input (512px, patch=16).

    VGGT-Omega uses the "balanced" resize mode:
      - Resize the shortest side to img_size while keeping the aspect ratio.
      - Then center-crop to img_size x img_size.

    Args:
        frames: (S, H, W, 3) uint8 [0, 255].

    Returns:
        tensor: (1, S, 3, img_size, img_size) float32 [0, 1].
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    processed = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i])
        # Resize shortest side to img_size, then center crop.
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
    Extract camera token features (and optionally decoded pose) for a single video.

    Args:
        model: VGGT or VGGT-Omega model.
        teacher_type: "vggt" or "vggt_omega".
        save_pose: whether to also run the Camera Head to obtain the decoded 9D pose.

    Returns:
        dict with:
          "camera_features": (S, 2048) float16 -- camera token from the aggregator's last layer.
          "pose_enc":        (S, 9) float32   -- decoded 9D pose [T(3) + Quat(4) + FoV(2)]
                             (only present when save_pose=True).
        or None if the video failed to read.
    """
    # 1. Sample frames.
    frames = extract_frames_from_video(video_path, fps=fps, max_frames=max_frames)
    if frames is None:
        return None

    # 2. Preprocess (different models have different sizes).
    if teacher_type == "vggt_omega":
        images = preprocess_for_vggt_omega(frames, img_size=512).to(device)
    else:
        images = preprocess_for_vggt(frames, img_size=518).to(device)

    # 3. Forward.
    result = {}
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        if save_pose:
            # Run the full model.forward() to get pose_enc.
            predictions = model(images)
            if teacher_type == "vggt_omega":
                if "camera_and_register_tokens" in predictions:
                    camera_features = predictions["camera_and_register_tokens"][:, :, 0, :]
                else:
                    output_list, _ = model.aggregator(images)
                    camera_features = output_list[-1][:, :, 0, :]
                pose_enc = predictions.get("pose_enc")  # (1, S, 9)
            else:
                # VGGT: forward returns a predictions dict.
                camera_features = None
                pose_enc = predictions.get("pose_enc")  # (1, S, 9) -- final iteration
                # camera_features must come from the aggregator output.
                # VGGT.forward internally already ran the aggregator but did not expose the intermediate result,
                # so run the aggregator again to grab the camera token.
                output_list, _ = model.aggregator(images)
                camera_features = output_list[-1][:, :, 0, :]
        else:
            # Only run the aggregator (skip the camera head to save compute).
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
    """Search several directories for a matching video file."""
    for vdir in video_dirs:
        path = os.path.join(vdir, f"{video_id}.mp4")
        if os.path.exists(path):
            return path
    return None


# ============================================================
# Multi-GPU worker
# ============================================================

def gpu_worker(gpu_id: int, args, video_id_shard: list):
    """
    A single GPU worker process. Loads the model onto the specified GPU and processes the
    video_id list assigned to it.

    Args:
        gpu_id: GPU index (0-7).
        args: command-line arguments.
        video_id_shard: list of video_ids for this worker.
    """
    device = f"cuda:{gpu_id}"

    # Bind this process to the target GPU.
    torch.cuda.set_device(gpu_id)

    # Load the model.
    if args.teacher == "vggt_omega":
        model = load_vggt_omega_model(args.vggt_model, device)
    else:
        model = load_vggt_model(args.vggt_model, device)

    # Process one video at a time.
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

        # Skip if the cache already exists.
        if args.skip_existing and os.path.exists(output_path):
            skipped += 1
            continue

        # Video path: strictly use the one from the jsonl.
        video_path = getattr(args, '_video_id_to_path', {}).get(video_id)
        if video_path is None or not os.path.exists(video_path):
            failed += 1
            continue

        # Extract features.
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
            # Clean up any leftover GPU memory fragments.
            torch.cuda.empty_cache()

        pbar.set_postfix(ok=success, fail=failed, skip=skipped)

    print(f"[GPU {gpu_id}] Done! Success: {success}, Failed: {failed}, Skipped: {skipped}")
    return success, failed


def gpu_worker_wrapper(args_tuple):
    """mp.spawn entry-point wrapper (unpacks a tuple)."""
    gpu_id, args, video_id_shard = args_tuple
    return gpu_worker(gpu_id, args, video_id_shard)


# ============================================================
# Main flow
# ============================================================

def cmd_extract(args):
    """Main flow for feature extraction (supports multi-GPU parallelism)."""

    # Default model paths.
    if args.vggt_model is None:
        if args.teacher == "vggt":
            args.vggt_model = "facebook/VGGT-1B"
        else:
            args.vggt_model = "facebook/VGGT-Omega"

    # Create output directory.
    os.makedirs(args.output_dir, exist_ok=True)

    # Read training data and collect every video_id together with its path.
    print(f"Reading {args.input_jsonl}...")
    video_ids = set()
    video_id_to_path = {}  # video_id -> absolute path from the jsonl (preferred)
    with open(args.input_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Extract video_id from the "videos" field.
            videos = data.get("videos", [])
            for v in videos:
                if isinstance(v, str):
                    vid = os.path.splitext(os.path.basename(v))[0]
                    video_ids.add(vid)
                    # Keep the original path from the jsonl (takes precedence over video_dirs lookup).
                    if os.path.isabs(v) and vid not in video_id_to_path:
                        video_id_to_path[vid] = v

    # Attach to args so gpu_worker can access it.
    args._video_id_to_path = video_id_to_path
    print(f"  Found {len(video_id_to_path)} videos with absolute paths in jsonl")

    print(f"Total unique videos: {len(video_ids)}")
    if args.save_pose:
        print(f"Mode: save camera_features (2048D) + pose_enc (9D: Translation+Quaternion+FoV)")
    else:
        print(f"Mode: save camera_features (2048D) only")

    # Skip everything already cached.
    if args.skip_existing:
        existing = set(f[:-3] for f in os.listdir(args.output_dir) if f.endswith(".pt"))
        todo = sorted(video_ids - existing)
        print(f"Already cached: {len(existing)}, remaining: {len(todo)}")
    else:
        todo = sorted(video_ids)

    if len(todo) == 0:
        print("All videos already processed! Nothing to do.")
        return

    # Decide how many GPUs to use.
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus <= 0:
        print("ERROR: No CUDA GPUs available!")
        return

    print(f"\n{'='*60}")
    print(f"Parallel extraction with {num_gpus} GPUs")
    print(f"Videos per GPU: ~{len(todo) // num_gpus}")
    print(f"Estimated speedup: {num_gpus}x (from ~{len(todo) * 6 / 3600:.1f}h to ~{len(todo) * 6 / 3600 / num_gpus:.1f}h)")
    print(f"{'='*60}\n")

    # Single-GPU mode: run in the current process (no spawn needed).
    if num_gpus == 1:
        gpu_id = int(args.device.replace("cuda:", "")) if ":" in args.device else 0
        gpu_worker(gpu_id, args, todo)
        _print_summary(args)
        return

    # Multi-GPU mode: shard evenly.
    shards = [[] for _ in range(num_gpus)]
    for i, vid in enumerate(todo):
        shards[i % num_gpus].append(vid)

    for i, shard in enumerate(shards):
        print(f"  GPU {i}: {len(shard)} videos")

    # Spawn worker processes.
    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, args, shards[gpu_id]),
        )
        p.start()
        processes.append(p)
        # Stagger startup to avoid downloading the model weights simultaneously.
        if gpu_id == 0:
            time.sleep(5)

    # Wait for all workers to finish.
    for p in processes:
        p.join()

    _print_summary(args)


def _print_summary(args):
    """Print the final summary."""
    total_cached = len([f for f in os.listdir(args.output_dir) if f.endswith(".pt")])
    print(f"\n{'='*60}")
    print(f"All GPUs finished!")
    print(f"Cache dir: {args.output_dir}")
    print(f"Total cached: {total_cached} files")
    print(f"\nSaved format per .pt file:")
    print(f"  result['camera_features']: (S, 2048) float16 -- Aggregator camera token")
    if args.save_pose:
        print(f"  result['pose_enc']:         (S, 9) float32 -- Decoded pose")
        print(f"    [0:3] = Translation (absolute, world coords)")
        print(f"    [3:7] = Quaternion rotation (wxyz or xyzw)")
        print(f"    [7:9] = Field of View (horizontal, vertical)")
    print(f"\n  Training usage:")
    print(f"    VGGT_CACHE_DIR={args.output_dir} bash camera_movement_sft/train_camdistill.sh qwen3vl-8b")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="VGGT / VGGT-Omega Camera Token feature pre-extraction (multi-GPU)")
    parser.add_argument("--input_jsonl", type=str, required=True, help="training data JSONL")
    parser.add_argument("--output_dir", type=str, required=True, help="output cache directory")
    parser.add_argument("--teacher", type=str, default="vggt", choices=["vggt", "vggt_omega"],
                        help="teacher model: vggt (518px) or vggt_omega (512px)")
    parser.add_argument("--vggt_model", type=str, default=None,
                        help="model path or HF id (default: vggt -> facebook/VGGT-1B)")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="device used in single-GPU mode (only effective when --num_gpus 1)")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="number of GPUs to use (default 8)")
    parser.add_argument("--save_pose", action="store_true", default=False,
                        help="also save the 9D pose decoded by the Camera Head")
    parser.add_argument("--video_dirs", type=str, nargs="+", default=[
        "/group/40059/yyjyu/data/aigc/camera_data/raw_videos_7w_readlfim+5w_anime_12w_overall/videos",
        "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_21k_video_shot_cameraMove/videos",
        "/group/40059/yyjyu/code/cv/camera_captions_processing/data/train_data/camera_realfilm_30w_260401_deduped_final/videos",
    ])
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="skip caches that already exist (supports resume after interruption)")
    args = parser.parse_args()
    cmd_extract(args)


if __name__ == "__main__":
    main()
