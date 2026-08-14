#!/usr/bin/env python3
"""
Single-video camera-movement inference.

Runs a CamSFT / CamDistill / CamInject model on one video and prints the structured
JSON (per-segment camera movement with time spans), using the official system/user
prompt from common.py — the same prompt used for training and evaluation.

Examples:
    # CamSFT (standard model, no plugin)
    python camera_movement_sft/infer_single.py \
        --model ddz16/CamSFT-4B \
        --video /path/to/video.mp4

    # CamDistill (needs the plugin + model_type; camera tokens are generated
    # internally, no online VGGT required)
    python camera_movement_sft/infer_single.py \
        --model ddz16/CamDistill-4B \
        --video /path/to/video.mp4 \
        --variant camdistill

    # CamInject (needs the plugin + model_type + online VGGT)
    VGGT_TEACHER_TYPE=vggt_omega \
    python camera_movement_sft/infer_single.py \
        --model ddz16/CamInject-4B \
        --video /path/to/video.mp4 \
        --variant caminject
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from common import SYSTEM_PROMPT, USER_PROMPT

PLUGIN_PATH = os.path.join(SCRIPT_DIR, "plugins", "camdistill_plugin.py")

# Model-type per variant (must match the plugin registrations in camdistill_plugin.py).
VARIANT_MODEL_TYPE = {
    "camsft": None,                    # standard model, no plugin
    "camdistill": "qwen3_vl_camdistill",
    "caminject": "qwen3_vl_caminject",
}


def build_command(args):
    import json

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<video>{USER_PROMPT}"},
    ]
    val_dataset = json.dumps([{"messages": messages, "videos": [args.video]}], ensure_ascii=False)

    cmd = [
        "swift", "infer",
        "--model", args.model,
        "--infer_backend", "pt",
        "--val_dataset", val_dataset,
        "--temperature", "0",
        "--max_new_tokens", "4096",
    ]
    if args.use_hf:
        cmd += ["--use_hf", "true"]

    model_type = VARIANT_MODEL_TYPE[args.variant]
    if model_type is not None:
        # CamDistill / CamInject: load the plugin and select the custom model_type.
        cmd += ["--model_type", model_type,
                "--external_plugins", PLUGIN_PATH]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Single-video camera-movement inference")
    parser.add_argument("--model", required=True, help="model path or HF id (e.g. ddz16/CamSFT-4B)")
    parser.add_argument("--video", required=True, help="path to the input video")
    parser.add_argument("--variant", choices=list(VARIANT_MODEL_TYPE), default="camsft",
                        help="camsft (default, no plugin) | camdistill | caminject")
    parser.add_argument("--use_hf", action="store_true", default=True,
                        help="load the model from the HuggingFace hub (default: on)")
    args = parser.parse_args()

    # Video processing config — must match training/evaluation exactly.
    os.environ.setdefault("FPS", "5")
    os.environ.setdefault("FPS_MIN_FRAMES", "4")
    os.environ.setdefault("FPS_MAX_FRAMES", "100")
    os.environ.setdefault("VIDEO_MAX_PIXELS", "100352")
    # decord avoids the torchcodec/CUDA version mismatch seen with some setups.
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

    if args.variant == "caminject":
        # CamInject runs VGGT online; mirror the defaults used by the eval script.
        os.environ.setdefault("VGGT_MODE", "online")
        teacher = os.environ.setdefault("VGGT_TEACHER_TYPE", "vggt")
        if not os.environ.get("VGGT_MODEL_PATH"):
            os.environ["VGGT_MODEL_PATH"] = (
                "facebook/VGGT-Omega" if teacher == "vggt_omega" else "facebook/VGGT-1B"
            )
        os.environ.setdefault("VGGT_ONLINE_FPS", os.environ["FPS"])
        os.environ.setdefault("VGGT_ONLINE_MAX_FRAMES", os.environ["FPS_MAX_FRAMES"])
        os.environ.setdefault("CAMINJECT_STRICT_IDS", "1")
        os.environ.setdefault("CAMINJECT_STRICT_CACHE", "1")
        os.environ.setdefault("CAMINJECT_MAX_MISS_RATIO", "0.0")
        os.environ.setdefault("CAMINJECT_MIN_RATIO_SAMPLES", "32")
        os.environ.setdefault("CAMINJECT_LOG_EVERY", "50")
        os.environ["VGGT_CACHE_DIR"] = ""  # always online; cache dir unused

    if args.variant != "camsft" and not os.path.isfile(PLUGIN_PATH):
        print(f"[ERROR] plugin not found: {PLUGIN_PATH}", file=sys.stderr)
        sys.exit(1)

    cmd = build_command(args)
    print(f"[INFO] variant={args.variant}  model={args.model}")
    print(f"[INFO] running: {' '.join(cmd[:6])} ...")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
