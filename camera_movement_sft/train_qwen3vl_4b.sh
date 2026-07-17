#!/bin/bash
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # Default to decord (overridable via env), avoids torchcodec CUDA (libnvrtc.so.13) version mismatch
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/python3.12/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${CONDA_PREFIX:-/data/miniconda3/envs/cm}/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
# Qwen3-VL-4B-Instruct camera-movement training
# Usage: bash camera_movement_sft/train_qwen3vl_4b.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" qwen3vl-4b
