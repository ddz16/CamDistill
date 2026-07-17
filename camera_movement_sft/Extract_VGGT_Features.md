# Extract VGGT Features

Pre-extract camera token features from training videos offline and save them as `.pt` files.
During training, the distillation target is looked up by `video_id` from the cache, avoiding online inference.

Two teacher models are supported:

| Teacher | HF ID | Params | Vision Encoder | Patch | Image Size |
|---|---|---|---|---|---|
| `vggt` | `facebook/VGGT-1B` | 1.26 B | DINOv2 | 14 | 518 px |
| `vggt_omega` | `facebook/VGGT-Omega` | 1.14 B | DINOv3 | 16 | 512 px |

Both emit camera tokens with the same output dimension (**2048**); downstream training pipelines require no changes when switching between the two.

---

## VGGT (default)

```bash
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl camera_movement_sft/train_data/camera_movement_train_diverse_50k.jsonl \
    --output_dir /path/to/vggt_cache/ \
    --teacher vggt \
    --save_pose \
    --num_gpus 8
```

## VGGT-Omega

```bash
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl camera_movement_sft/train_data/camera_movement_train_diverse_50k.jsonl \
    --output_dir /path/to/vggt_omega_cache/ \
    --teacher vggt_omega \
    --save_pose \
    --num_gpus 8
```
---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--input_jsonl` | *(required)* | Training data JSONL; each line must have a `"videos"` field |
| `--output_dir` | *(required)* | Directory where `.pt` cache files are saved |
| `--teacher` | `vggt` | Teacher model: `vggt` or `vggt_omega` |
| `--vggt_model` | auto | Model path or HF id; defaults to `facebook/VGGT-1B` / `facebook/VGGT-Omega` |
| `--fps` | `5` | Frame sampling rate (must match training) |
| `--max_frames` | `100` | Maximum frames per video (must match training) |
| `--num_gpus` | `8` | Number of GPUs for parallel extraction |
| `--device` | `cuda:0` | Device used in single-GPU mode (`--num_gpus 1`) |
| `--skip_existing` | `True` | Skip already-cached videos; supports resuming an interrupted run |
| `--save_pose` | `False` | Also save the 9-D pose decoded by the Camera Head |
| `--video_dirs` | *(see code)* | Local video search directories (fallback when jsonl paths are unavailable) |

---

## Output Format

Each video produces one `.pt` file named `<video_id>.pt`:

```python
result = torch.load("video_id.pt")
result["camera_features"]  # shape: (S, 2048), dtype: float16  — aggregator camera token
result["pose_enc"]         # shape: (S, 9),    dtype: float32  — only when --save_pose
                           #   [0:3] Translation (absolute, world coords)
                           #   [3:7] Quaternion rotation (wxyz)
                           #   [7:9] Field of View (horizontal, vertical)
```

`S` is the number of sampled frames (≤ `--max_frames`).

---

## Using the Cache for Training

Set `VGGT_CACHE_DIR` to the output directory before launching training:

```bash
# CamDistill training with VGGT cache
VGGT_CACHE_DIR=/path/to/vggt_cache bash camera_movement_sft/train_camdistill.sh qwen3vl-8b

# CamDistill training with VGGT-Omega cache
VGGT_CACHE_DIR=/path/to/vggt_omega_cache \
VGGT_TEACHER_TYPE=vggt_omega \
bash camera_movement_sft/train_camdistill.sh qwen3vl-8b
```

---

## Single-GPU Mode

```bash
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl ... \
    --output_dir ... \
    --num_gpus 1 \
    --device cuda:0
```

---

## Notes

- `--fps` and `--max_frames` must match the values used during training exactly.
- Multi-GPU mode spawns one process per GPU and shards the video list evenly.
  Startup is staggered by 5 s to avoid simultaneous model weight downloads.
- Re-running with `--skip_existing` (default) is safe; already-cached files are skipped.
