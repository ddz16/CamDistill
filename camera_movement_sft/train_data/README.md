# Training Data

## Data Availability

Our internal training set is **not publicly released**. It was annotated using the
same labeling guidelines, quality standards, and **JSON schema as our benchmark**
([CamChoreo](https://huggingface.co/datasets/ddz16/CamChoreo)) — human annotators
labeled camera-movement segments (basic-movement type, direction, speed, and special
techniques) for each video clip.

## Train on Your Own Data

Because the training set shares the CamChoreo schema, you can train on your own data
by preparing it in the same annotation format, one JSON object per line:

```json
{"video_id": "xxx", "local_path": "./videos/xxx.mp4", "segments": [...]}
```

`local_path` is resolved relative to the annotation file's directory (so keep the
videos in a `videos/` folder next to it).

Then convert it to the ms-swift SFT format and point `DATASET_PATH` at the result:

```bash
python camera_movement_sft/prepare_train_data.py \
    --input  /path/to/your_annotations.jsonl \
    --output camera_movement_sft/train_data/train_swift.jsonl

DATASET_PATH=camera_movement_sft/train_data/train_swift.jsonl \
    bash camera_movement_sft/train.sh qwen3vl-4b
```
