"""
CamDistill Plugin: ms-swift external plugin entry point.

Loaded via --external_plugins camera_movement_sft/plugins/camdistill_plugin.py.
Registers a custom model loader and distillation loss.

Camera Token injection strategy:
  Uses a modified modeling_qwen3_vl_camdistill.py (copied and patched from transformers)
  that performs standalone camera-token insertion inside forward (one token in front of
  each frame's visual tokens).
"""

import os
import sys
import json
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import List

# Make sure the plugins directory is on sys.path.
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

# ============================================================
# 1. Register the custom loss
# ============================================================
from camdistill_loss import CamDistillLoss  # noqa: F401 (side-effect: registers in loss_map)

# ============================================================
# 2. Register the custom model loader (with CamDistill module)
# ============================================================
from swift.model import ModelMeta, ModelGroup, Model, register_model
from swift.model.models.qwen import Qwen3VLLoader
from swift.model.constant import MLLMModelType
from swift.template.constant import TemplateType

from camdistill_model import CameraTokenModule

# Import the modified model's forward and helper functions (used for monkey-patching).
import types
from modeling_qwen3_vl_camdistill import (
    _inject_camera_into_video_embeds,
    _expand_video_placeholders,
    _expand_inputs_embeds_for_camera,
)


def _resolve_camdistill_extract_layers(vit_depth: int, camdistill_depth: int) -> List[int]:
    """Resolve the ViT layer indices to cache."""
    raw_layers = os.environ.get('CAMDISTILL_EXTRACT_LAYERS', '').strip()
    if not raw_layers:
        extract_layers = [
            int(round((i + 1) * vit_depth / (camdistill_depth + 1))) - 1
            for i in range(camdistill_depth)
        ]
        return [min(l, vit_depth - 1) for l in extract_layers]

    try:
        extract_layers = [int(x.strip()) for x in raw_layers.split(',') if x.strip()]
    except ValueError as e:
        raise RuntimeError(
            '[CamDistill] CAMDISTILL_EXTRACT_LAYERS must be a comma-separated list of integers, '
            f'got: {raw_layers}'
        ) from e

    if not extract_layers:
        raise RuntimeError('[CamDistill] CAMDISTILL_EXTRACT_LAYERS is empty after parsing.')
    if len(extract_layers) != camdistill_depth:
        raise RuntimeError(
            f'[CamDistill] CAMDISTILL_EXTRACT_LAYERS length mismatch: '
            f'len(extract_layers)={len(extract_layers)} vs CAMDISTILL_DEPTH={camdistill_depth}'
        )
    if any(layer < 0 or layer >= vit_depth for layer in extract_layers):
        raise RuntimeError(
            f'[CamDistill] CAMDISTILL_EXTRACT_LAYERS contains out-of-range index. '
            f'vit_depth={vit_depth}, extract_layers={extract_layers}'
        )
    if any(b <= a for a, b in zip(extract_layers, extract_layers[1:])):
        raise RuntimeError(
            f'[CamDistill] CAMDISTILL_EXTRACT_LAYERS must be strictly increasing, '
            f'got: {extract_layers}'
        )
    return extract_layers


def _expand_labels_for_camera(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    video_token_id: int,
    video_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    tokens_per_frame: int = 1,
) -> torch.Tensor:
    """
    Insert -100 (no loss) at every camera-token position (K=tokens_per_frame per frame).
    Fully aligned with _expand_video_placeholders.
    """
    B, seq_len = labels.shape
    K = int(tokens_per_frame)
    T_total = int(video_grid_thw[:, 0].sum().item())
    new_seq_len = seq_len + T_total * K

    frame_token_counts = []
    for t, h, w in video_grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        tpf = (h * w) // (spatial_merge_size ** 2)
        for _ in range(t):
            frame_token_counts.append(tpf)

    new_labels = torch.full((B, new_seq_len), -100, device=labels.device, dtype=labels.dtype)

    for b in range(B):
        vid_mask = (input_ids[b] == video_token_id)
        vid_positions = vid_mask.nonzero(as_tuple=True)[0]
        if vid_positions.numel() == 0:
            new_labels[b, :seq_len] = labels[b]
            continue

        vid_start = vid_positions[0].item()
        vid_end = vid_positions[-1].item() + 1

        # Copy the tokens before the video.
        new_labels[b, :vid_start] = labels[b, :vid_start]

        # Insert K camera placeholders (-100) + the original visual labels, frame by frame.
        src_offset = vid_start
        dst_offset = vid_start
        for n_tokens in frame_token_counts:
            # Camera token positions (K of them): already -100 from the `full` init.
            dst_offset += K
            # Original visual token labels.
            new_labels[b, dst_offset:dst_offset + n_tokens] = labels[b, src_offset:src_offset + n_tokens]
            dst_offset += n_tokens
            src_offset += n_tokens

        # Copy the tokens after the video.
        remaining = seq_len - vid_end
        if remaining > 0:
            new_labels[b, dst_offset:dst_offset + remaining] = labels[b, vid_end:]

    return new_labels


class CamDistillQwen3VLLoader(Qwen3VLLoader):
    """
    CamDistill Loader for Qwen3-VL.

    Loads the model with the modified modeling file, injecting camera tokens directly
    inside forward. Injection is controlled by the model's _camdistill_enabled flag.
    ViT intermediate layers are collected via forward hooks and consumed by the
    CameraTokenModule.
    """

    def get_model(self, model_dir, config, processor, model_kwargs):
        model = super().get_model(model_dir, config, processor, model_kwargs)

        # Read config.
        vit_hidden_dim = config.vision_config.hidden_size    # 1024 or 1152
        vit_num_heads = config.vision_config.num_heads       # 16
        vit_depth = config.vision_config.depth               # 24 or 27
        llm_hidden_dim = config.vision_config.out_hidden_size  # 2560 or 4096

        # CamDistill config.
        camdistill_depth = int(os.environ.get("CAMDISTILL_DEPTH", "6"))

        # Compute hook layers: evenly spaced by default; overridable via CAMDISTILL_EXTRACT_LAYERS.
        extract_layers = _resolve_camdistill_extract_layers(vit_depth, camdistill_depth)

        print(f"[CamDistill] Initializing CameraTokenModule:")
        print(f"  vit_hidden_dim={vit_hidden_dim}, vit_depth={vit_depth}")
        print(f"  llm_hidden_dim={llm_hidden_dim}")
        print(f"  camdistill_depth={camdistill_depth}, extract_layers={extract_layers}")

        # Instantiate CameraTokenModule.
        # cam_dim is pinned to 1024: camera_features = 2 * cam_dim = 2048, matching the VGGT cache
        # (VGGT stream = 1024). Must be explicit; otherwise the default ViT hidden dim (4B=1024
        # happens to give 2048; 8B=1152 -> 2304 would mismatch VGGT and raise).
        camdistill = CameraTokenModule(
            hidden_dim=vit_hidden_dim,
            num_heads=vit_num_heads,
            depth=camdistill_depth,
            llm_hidden_dim=llm_hidden_dim,
            cam_dim=int(os.environ.get("CAMDISTILL_CAM_DIM", "1024")),
        )
        # CamDistill (learn mode) fixes K=1 since CameraTokenModule emits one token per frame.
        camdistill.tokens_per_frame = 1

        # Attach model attributes consumed by the modified forward.
        inner_model = model.model  # Qwen3VLModel
        inner_model._camdistill_module = camdistill
        inner_model._camdistill_enabled = True
        inner_model._camdistill_mode = 'learn'
        inner_model._camdistill_preexpanded_input = True
        inner_model._camdistill_video_ids = []

        # Also keep a top-level reference (used by the loss).
        model.camdistill = camdistill

        # Explicitly restore CamDistill module parameters from checkpoint so that they are not
        # dropped as UNEXPECTED during the main load stage.
        _try_load_camdistill_module_from_checkpoint(model_dir, camdistill)

        # Configure the ViT to cache the required intermediate layer indices (no hooks; caching
        # happens directly inside the ViT forward).
        visual_module = inner_model.visual
        visual_module._camdistill_extract_layers = set(extract_layers)
        visual_module._camdistill_layer_cache = []

        print(f"[CamDistill] ViT intermediate layer caching configured: {extract_layers}")
        print(f"[CamDistill] CameraTokenModule parameters: "
              f"{sum(p.numel() for p in camdistill.parameters()) / 1e6:.1f}M")

        # Swap the inner model's forward for the modified version (with camera-token injection).
        from modeling_qwen3_vl_camdistill import (
            Qwen3VLModel as Qwen3VLModelCamDistill,
            Qwen3VLVisionModel as Qwen3VLVisionModelCamDistill,
        )
        inner_model.forward = types.MethodType(
            Qwen3VLModelCamDistill.forward, inner_model
        )
        inner_model.visual.forward = types.MethodType(
            Qwen3VLVisionModelCamDistill.forward, inner_model.visual
        )
        # Also swap get_vision_position_ids and get_rope_index.
        inner_model.get_vision_position_ids = types.MethodType(
            Qwen3VLModelCamDistill.get_vision_position_ids, inner_model
        )
        inner_model.get_rope_index = types.MethodType(
            Qwen3VLModelCamDistill.get_rope_index, inner_model
        )

        print('[CamDistill] Inner forward patched for camera token injection (mode=learn, preexpanded=1)')

        return model


# ============================================================
# 3. Register models with ms-swift
# ============================================================

# Qwen3-VL family (CamDistill variant).
register_model(
    ModelMeta(
        "qwen3_vl_camdistill",
        [
            ModelGroup([
                Model("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"),
                Model("Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"),
            ], TemplateType.qwen3_vl),
        ],
        CamDistillQwen3VLLoader,
        architectures=["Qwen3VLForConditionalGeneration"],
        is_multimodal=True,
    ),
)

# ============================================================
# 4. CamInject Baseline (full injection implementation)
# ============================================================
from caminject_model import CamInjectAdapter


def _try_load_camdistill_module_from_checkpoint(model_dir: str, camdistill_module: nn.Module) -> None:
    """Manually restore the CamDistill module parameters from a checkpoint.

    Context: plugin modules are attached after the base model has been loaded, so
    `camdistill.*` / `model._camdistill_module.*` in the checkpoint are often flagged as
    UNEXPECTED by the main load flow. Restoring them here ensures inference/evaluation
    uses the trained CamDistill weights.
    """
    if camdistill_module is None:
        return

    target_state = camdistill_module.state_dict()
    target_keys = list(target_state.keys())
    key_prefixes = (
        'camdistill.',
        'model._camdistill_module.',
        '_camdistill_module.',
    )

    loaded = {}
    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    single_path = os.path.join(model_dir, 'model.safetensors')

    try:
        from safetensors import safe_open
    except Exception as e:
        print(f"[CamDistill] WARNING: safetensors not available, skip module restore: {e}")
        return

    try:
        if os.path.isfile(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            weight_map = index_data.get('weight_map', {})

            shard_to_items = {}
            for tgt_key in target_keys:
                src_key = None
                for prefix in key_prefixes:
                    candidate = f'{prefix}{tgt_key}'
                    if candidate in weight_map:
                        src_key = candidate
                        break
                if src_key is None:
                    continue
                shard = weight_map[src_key]
                shard_to_items.setdefault(shard, []).append((src_key, tgt_key))

            for shard, items in shard_to_items.items():
                shard_path = os.path.join(model_dir, shard)
                if not os.path.isfile(shard_path):
                    continue
                with safe_open(shard_path, framework='pt', device='cpu') as f:
                    for src_key, tgt_key in items:
                        loaded[tgt_key] = f.get_tensor(src_key)

        elif os.path.isfile(single_path):
            with safe_open(single_path, framework='pt', device='cpu') as f:
                all_keys = set(f.keys())
                for tgt_key in target_keys:
                    for prefix in key_prefixes:
                        src_key = f'{prefix}{tgt_key}'
                        if src_key in all_keys:
                            loaded[tgt_key] = f.get_tensor(src_key)
                            break
        else:
            return
    except Exception as e:
        print(f"[CamDistill] WARNING: module restore failed while reading checkpoint: {e}")
        return

    if not loaded:
        print('[CamDistill] WARNING: no CamDistill module tensors found in checkpoint index')
        return

    try:
        incompatible = camdistill_module.load_state_dict(loaded, strict=False)
    except Exception as e:
        print(f"[CamDistill] WARNING: module restore failed while loading state_dict: {e}")
        return

    missing = [k for k in target_keys if k not in loaded]
    if missing or incompatible.unexpected_keys:
        print(
            f"[CamDistill] WARNING: module partial restore, loaded={len(loaded)}/{len(target_keys)}, "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
    else:
        print(f"[CamDistill] module restored from checkpoint: {len(loaded)} tensors")


def _try_load_caminject_projector_from_checkpoint(model_dir: str, adapter: nn.Module) -> None:
    """Manually restore CamInject projector parameters from a checkpoint.

    Context: ms-swift loads the base model before the plugin attaches the adapter, so keys
    like `model._camdistill_module.projector.*` / `caminject_adapter.projector.*` in the
    checkpoint may be reported as UNEXPECTED during the main load stage. Restoring them
    explicitly here avoids the projector being randomly initialized.
    """
    projector = getattr(adapter, 'projector', None)
    if projector is None:
        return

    target_state = projector.state_dict()
    target_keys = list(target_state.keys())
    key_prefixes = (
        'model._camdistill_module.projector.',
        'caminject_adapter.projector.',
        '_camdistill_module.projector.',
        'projector.',
    )

    loaded = {}
    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    single_path = os.path.join(model_dir, 'model.safetensors')

    try:
        from safetensors import safe_open
    except Exception as e:
        print(f"[CamInject] WARNING: safetensors not available, skip projector restore: {e}")
        return

    try:
        if os.path.isfile(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            weight_map = index_data.get('weight_map', {})

            shard_to_items = {}
            for tgt_key in target_keys:
                src_key = None
                for prefix in key_prefixes:
                    candidate = f'{prefix}{tgt_key}'
                    if candidate in weight_map:
                        src_key = candidate
                        break
                if src_key is None:
                    continue
                shard = weight_map[src_key]
                shard_to_items.setdefault(shard, []).append((src_key, tgt_key))

            for shard, items in shard_to_items.items():
                shard_path = os.path.join(model_dir, shard)
                if not os.path.isfile(shard_path):
                    continue
                with safe_open(shard_path, framework='pt', device='cpu') as f:
                    for src_key, tgt_key in items:
                        loaded[tgt_key] = f.get_tensor(src_key)

        elif os.path.isfile(single_path):
            with safe_open(single_path, framework='pt', device='cpu') as f:
                all_keys = set(f.keys())
                for tgt_key in target_keys:
                    for prefix in key_prefixes:
                        src_key = f'{prefix}{tgt_key}'
                        if src_key in all_keys:
                            loaded[tgt_key] = f.get_tensor(src_key)
                            break
        else:
            return
    except Exception as e:
        print(f"[CamInject] WARNING: projector restore failed while reading checkpoint: {e}")
        return

    if not loaded:
        return

    try:
        incompatible = projector.load_state_dict(loaded, strict=False)
    except Exception as e:
        print(f"[CamInject] WARNING: projector restore failed while loading state_dict: {e}")
        return

    missing = [k for k in target_keys if k not in loaded]
    if missing or incompatible.unexpected_keys:
        print(
            f"[CamInject] WARNING: projector partial restore, loaded={len(loaded)}/{len(target_keys)}, "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
    else:
        print(f"[CamInject] projector restored from checkpoint: {len(loaded)} tensors")


def _make_caminject_loader(base_loader_cls, model_file_module_name: str):
    """
    Factory: build a CamInject variant of the given base loader.

    CamInject and CamDistill share exactly the same injection logic (modified forward);
    the only difference is the source of camera_embeds:
      - CamDistill: CameraTokenModule(vit_intermediates) -> camera_embeds
      - CamInject:  VGGT_cache -> VGGTProjector -> camera_embeds

    Setting _camdistill_mode='direct' tells the modified forward to skip the
    vit_intermediates check and call CamInjectAdapter directly (which ignores
    vit_intermediates and loads from cache).
    """

    class CamInjectLoader(base_loader_cls):
        def get_model(self, model_dir, config, processor, model_kwargs):
            model = super().get_model(model_dir, config, processor, model_kwargs)

            llm_hidden_dim = config.vision_config.out_hidden_size
            temporal_patch_size = getattr(config.vision_config, 'temporal_patch_size', 2)
            vggt_cache_dir = os.environ.get("VGGT_CACHE_DIR", "")

            if not vggt_cache_dir:
                print("[CamInject] WARNING: VGGT_CACHE_DIR not set! Camera tokens will be zeros.")

            # Instantiate CamInjectAdapter (same interface as CameraTokenModule).
            adapter = CamInjectAdapter(
                llm_dim=llm_hidden_dim,
                vggt_dim=2048,
                temporal_patch_size=temporal_patch_size,
            )
            adapter.set_cache_dir(vggt_cache_dir)

            # Attach to the inner model (same attribute names as CamDistill).
            inner_model = model.model
            inner_model._camdistill_module = adapter
            inner_model._camdistill_enabled = True
            inner_model._camdistill_mode = 'direct'  # key: tells forward it does not need vit_intermediates
            inner_model._camdistill_preexpanded_input = True

            # Also keep a top-level reference.
            model.caminject_adapter = adapter

            # Explicitly restore projector weights so plugin parameters in the checkpoint are not
            # lost after being flagged as UNEXPECTED during the main load stage.
            _try_load_caminject_projector_from_checkpoint(model_dir, adapter)

            # No ViT intermediate cache is needed (CamInject does not use it).
            # inner_model.visual._camdistill_extract_layers is intentionally left unset.

            print(f"[CamInject] CamInjectAdapter initialized:")
            print(f"  llm_dim={llm_hidden_dim}, temporal_patch_size={temporal_patch_size}")
            print(f"  cache_dir={vggt_cache_dir}")
            print(f"  projector params: {sum(p.numel() for p in adapter.projector.parameters()) / 1e6:.1f}M")

            # Monkey-patch forward (use the modified forward that performs camera-token injection).
            from modeling_qwen3_vl_camdistill import (
                Qwen3VLModel as Qwen3VLModelCamDistill,
            )
            inner_model.forward = types.MethodType(
                Qwen3VLModelCamDistill.forward, inner_model
            )
            inner_model.get_vision_position_ids = types.MethodType(
                Qwen3VLModelCamDistill.get_vision_position_ids, inner_model
            )
            inner_model.get_rope_index = types.MethodType(
                Qwen3VLModelCamDistill.get_rope_index, inner_model
            )
            print(f"[CamInject] Inner forward patched for camera token injection (mode=direct)")

            return model

    CamInjectLoader.__name__ = f"CamInject{base_loader_cls.__name__}"
    CamInjectLoader.__qualname__ = CamInjectLoader.__name__
    return CamInjectLoader

# Qwen3-VL CamInject
CamInjectQwen3VLLoader = _make_caminject_loader(Qwen3VLLoader, "modeling_qwen3_vl_camdistill")

register_model(
    ModelMeta(
        "qwen3_vl_caminject",
        [
            ModelGroup([
                Model("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"),
                Model("Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"),
            ], TemplateType.qwen3_vl),
        ],
        CamInjectQwen3VLLoader,
        architectures=["Qwen3VLForConditionalGeneration"],
        is_multimodal=True,
    ),
)


# ============================================================
# 5. Video ID propagation (Template monkey-patch)
# ============================================================
# Goal: let CamInject reliably access video_ids inside model.forward.
# Implementation:
#   1. patch Template.encode: attach `video_ids` to sample-level features.
#   2. patch Template._data_collator_mm_data: forward `video_ids` from the batch to the model.
#
# Notes:
#   - Patching encode alone is not enough; the default collator drops unknown string fields.
#   - With both patches, video_ids travels alongside video_grid_thw into forward kwargs.

_VIDEO_ID_PATCH_APPLIED = False
_CAMINJECT_MODEL_TYPES = {'qwen3_vl_caminject'}
_CAMDISTILL_MODEL_TYPES = {'qwen3_vl_camdistill'}

def _is_caminject_model(template) -> bool:
    model_type = getattr(getattr(template, 'model_meta', None), 'model_type', None)
    if isinstance(model_type, str) and model_type:
        return model_type in _CAMINJECT_MODEL_TYPES
    model_type_env = os.environ.get('MODEL_TYPE', '')
    return model_type_env in _CAMINJECT_MODEL_TYPES

def _is_camdistill_model(template) -> bool:
    model_type = getattr(getattr(template, 'model_meta', None), 'model_type', None)
    if isinstance(model_type, str) and model_type:
        return model_type in _CAMDISTILL_MODEL_TYPES
    model_type_env = os.environ.get('MODEL_TYPE', '')
    return model_type_env in _CAMDISTILL_MODEL_TYPES

def _is_camera_injection_model(template) -> bool:
    return _is_caminject_model(template) or _is_camdistill_model(template)


def _is_qwen_vl_template_for_camera_patch(template) -> bool:
    template_type = getattr(getattr(template, 'template_meta', None), 'template_type', '')
    return template_type in {'qwen3_vl', 'qwen3_5'}


def _expand_single_sample_for_camera(
    sample: dict,
    *,
    video_token_id: int,
    spatial_merge_size: int,
    insert_position: str,
    tokens_per_frame: int = 1,
) -> None:
    """Expand a sample's sequence during encode: insert K=tokens_per_frame camera placeholders
    in front of / behind each frame.

    Note: inference typically has no labels, so labels cannot be used as a precondition for
    expansion.
    """
    if not isinstance(sample, dict):
        return

    input_ids = sample.get('input_ids')
    video_grid_thw = sample.get('video_grid_thw')
    if input_ids is None or video_grid_thw is None:
        return

    if not isinstance(input_ids, list):
        return

    old_len = len(input_ids)
    if old_len == 0:
        return

    if not isinstance(video_grid_thw, torch.Tensor):
        return

    K = int(tokens_per_frame)
    if K != 1:
        raise RuntimeError(
            f"[CamInject] unsupported tokens_per_frame={K}, expected 1"
        )

    t_total = int(video_grid_thw[:, 0].sum().item())
    if t_total <= 0:
        return

    def _parse_1d_field(value, field_name: str, *, list_dtype: torch.dtype):
        if value is None:
            return None, None

        if isinstance(value, list):
            if len(value) != old_len:
                raise RuntimeError(
                    f"[CamInject] {field_name} length mismatch before camera expansion: "
                    f"got={len(value)}, expected={old_len}"
                )
            return torch.tensor(value, dtype=list_dtype), 'list'

        if isinstance(value, torch.Tensor):
            tensor = value
            if tensor.dim() == 2 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.dim() != 1 or tensor.shape[0] != old_len:
                raise RuntimeError(
                    f"[CamInject] {field_name} length mismatch before camera expansion: "
                    f"got={tuple(tensor.shape)}, expected=({old_len},)"
                )
            return tensor, 'tensor'

        raise RuntimeError(
            f"[CamInject] unsupported {field_name} type before camera expansion: {type(value)}"
        )

    input_ids_1d = torch.tensor(input_ids, dtype=torch.long)

    mm_type_tensor, mm_type_container = _parse_1d_field(
        sample.get('mm_token_type_ids'),
        'mm_token_type_ids',
        list_dtype=torch.long,
    )
    attn_tensor, attn_container = _parse_1d_field(
        sample.get('attention_mask'),
        'attention_mask',
        list_dtype=torch.long,
    )
    labels_tensor, labels_container = _parse_1d_field(
        sample.get('labels'),
        'labels',
        list_dtype=torch.long,
    )
    loss_scale_tensor, loss_scale_container = _parse_1d_field(
        sample.get('loss_scale'),
        'loss_scale',
        list_dtype=torch.float32,
    )

    vid_mask = (input_ids_1d == video_token_id)
    if not vid_mask.any():
        raise RuntimeError('[CamInject] video_grid_thw exists but input_ids has no video token placeholders')
    shifted = torch.zeros_like(vid_mask)
    shifted[1:] = vid_mask[:-1]
    segment_count = int((vid_mask & ~shifted).sum().item())
    if segment_count != t_total:
        raise RuntimeError(
            f"[CamInject] segment_count mismatch before camera expansion: "
            f"segment_count={segment_count}, t_total={t_total}. "
            'Expected one contiguous video segment per frame.'
        )

    input_ids_tensor = input_ids_1d.unsqueeze(0)
    mm_type_2d = mm_type_tensor.unsqueeze(0) if mm_type_tensor is not None else None
    attn_2d = attn_tensor.unsqueeze(0) if attn_tensor is not None else None

    new_input_ids, new_mm_type, new_attn, dst_positions_list = _expand_video_placeholders(
        input_ids_tensor,
        mm_token_type_ids=mm_type_2d,
        attention_mask=attn_2d,
        video_token_id=video_token_id,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        insert_position=insert_position,
        tokens_per_frame=K,
    )
    new_input_ids_list = new_input_ids[0].tolist()
    new_len = len(new_input_ids_list)
    expected_len = old_len + t_total * K
    if new_len != expected_len:
        raise RuntimeError(
            f"[CamInject] expanded input_ids length mismatch: got={new_len}, "
            f"expected={expected_len}, t_total={t_total}, tokens_per_frame={K}"
        )

    sample['input_ids'] = new_input_ids_list

    if new_mm_type is not None:
        mm_1d = new_mm_type[0]
        if mm_type_container == 'list':
            sample['mm_token_type_ids'] = mm_1d.tolist()
        else:
            sample['mm_token_type_ids'] = mm_1d

    if new_attn is not None:
        attn_1d = new_attn[0]
        if attn_container == 'list':
            sample['attention_mask'] = attn_1d.tolist()
        else:
            sample['attention_mask'] = attn_1d

    if labels_tensor is None:
        return

    dst_positions = dst_positions_list[0]
    if dst_positions.numel() != old_len:
        raise RuntimeError(
            f"[CamInject] dst_positions length mismatch: got={dst_positions.numel()}, expected={old_len}"
        )

    new_labels = torch.full((new_len,), -100, dtype=labels_tensor.dtype)
    new_labels[dst_positions] = labels_tensor
    if labels_container == 'list':
        sample['labels'] = new_labels.tolist()
    else:
        sample['labels'] = new_labels

    if loss_scale_tensor is not None:
        new_loss_scale_tensor = torch.zeros((new_len,), dtype=loss_scale_tensor.dtype)
        new_loss_scale_tensor[dst_positions] = loss_scale_tensor
        if loss_scale_container == 'list':
            sample['loss_scale'] = new_loss_scale_tensor.tolist()
        else:
            sample['loss_scale'] = new_loss_scale_tensor


def _to_list_of_str(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [x for x in value if isinstance(x, str) and x]
    return []


def _video_path_to_id(video_path: str):
    if not isinstance(video_path, str) or not video_path:
        return None
    video_name = os.path.basename(video_path)
    if not video_name:
        return None
    video_id, _ = os.path.splitext(video_name)
    return video_id or video_name


def _extract_video_paths_from_messages(messages) -> List[str]:
    """Handle video fields found in OpenAI-style message content."""
    video_paths: List[str] = []
    if not isinstance(messages, list):
        return video_paths

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue

            value = None
            item_type = item.get('type')
            if item_type == 'video':
                value = item.get('video')
            elif item_type == 'video_url':
                value = item.get('video_url')

            if isinstance(value, dict):
                value = value.get('url')
            if isinstance(value, str) and value:
                video_paths.append(value)

    return video_paths


def _extract_video_paths_from_inputs(inputs) -> List[str]:
    """Extract video paths from dict / TemplateInputs / StdTemplateInputs
    (preserves order and does not deduplicate across fields)."""
    if isinstance(inputs, dict):
        chosen = inputs.get('chosen')
        if isinstance(chosen, dict):
            chosen_videos = _to_list_of_str(chosen.get('videos'))
            if chosen_videos:
                return chosen_videos

        videos = _to_list_of_str(inputs.get('videos'))
        if videos:
            return videos

        for key in ('chosen_videos', 'video'):
            values = _to_list_of_str(inputs.get(key))
            if values:
                return values

        message_paths = _extract_video_paths_from_messages(inputs.get('messages'))
        if message_paths:
            return message_paths

        if isinstance(chosen, dict):
            chosen_message_paths = _extract_video_paths_from_messages(chosen.get('messages'))
            if chosen_message_paths:
                return chosen_message_paths
        return []

    chosen = getattr(inputs, 'chosen', None)
    if chosen is not None:
        chosen_videos = _to_list_of_str(getattr(chosen, 'videos', None))
        if chosen_videos:
            return chosen_videos
    return _to_list_of_str(getattr(inputs, 'videos', None))


def _extract_video_ids_from_inputs(inputs) -> List[str]:
    """Extract raw video paths from dict / TemplateInputs / StdTemplateInputs and convert them
    into video_ids."""
    video_paths = _extract_video_paths_from_inputs(inputs)

    video_ids: List[str] = []
    for video_path in video_paths:
        video_id = _video_path_to_id(video_path)
        if not video_id:
            continue
        video_ids.append(video_id)
    return video_ids


def _inject_video_ids_to_encoded(result, video_ids: List[str]) -> None:
    if not video_ids or result is None:
        return

    # Keep single-video samples as a str for backwards compatibility; multi-video samples become
    # a list.
    payload = video_ids[0] if len(video_ids) == 1 else list(video_ids)

    if isinstance(result, dict):
        result.setdefault('video_ids', payload)
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, dict):
                item.setdefault('video_ids', payload)


def _inject_video_paths_to_encoded(result, video_paths: List[str]) -> None:
    if not video_paths or result is None:
        return

    payload = video_paths[0] if len(video_paths) == 1 else list(video_paths)

    if isinstance(result, dict):
        result.setdefault('video_paths', payload)
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, dict):
                item.setdefault('video_paths', payload)


def _collect_batch_video_ids(batch: List[dict]) -> List[str]:
    """Flatten batch-level video_ids (str or list[str]) into a single list[str]."""
    batch_video_ids: List[str] = []
    for sample in batch:
        # Only collect video_ids from samples that actually go through the video branch, so the
        # count stays consistent with video_grid_thw.
        has_video_inputs = sample.get('video_grid_thw') is not None or sample.get('pixel_values_videos') is not None
        if not has_video_inputs:
            continue

        video_ids = sample.get('video_ids')
        if isinstance(video_ids, str):
            batch_video_ids.append(video_ids)
        elif isinstance(video_ids, (list, tuple)):
            for vid in video_ids:
                if isinstance(vid, str) and vid:
                    batch_video_ids.append(vid)
    return batch_video_ids


def _collect_batch_video_paths(batch: List[dict]) -> List[str]:
    """Flatten batch-level video_paths (str or list[str]) into a single list[str]."""
    batch_video_paths: List[str] = []
    for sample in batch:
        has_video_inputs = sample.get('video_grid_thw') is not None or sample.get('pixel_values_videos') is not None
        if not has_video_inputs:
            continue

        video_paths = sample.get('video_paths')
        if isinstance(video_paths, str):
            batch_video_paths.append(video_paths)
        elif isinstance(video_paths, (list, tuple)):
            for path in video_paths:
                if isinstance(path, str) and path:
                    batch_video_paths.append(path)
    return batch_video_paths


def _maybe_expand_encoded_for_camera(template, encoded, *, video_ids: List[str]) -> None:
    """
    Perform camera-placeholder expansion during encode so the collator sees the final sequence.
    Enabled only for qwen3_vl / qwen3_5 templates.
    """
    if not _is_qwen_vl_template_for_camera_patch(template):
        return
    if not _is_camera_injection_model(template):
        return

    sample_list = [encoded] if isinstance(encoded, dict) else encoded
    if not isinstance(sample_list, (list, tuple)):
        return

    video_token_id = getattr(template, 'video_token_id', None)
    processor = getattr(template, 'processor', None)
    image_processor = getattr(processor, 'image_processor', None)
    spatial_merge_size = int(getattr(image_processor, 'merge_size', 2) or 2)
    insert_position = os.environ.get('CAMERA_TOKEN_INSERT_POSITION', 'front').strip().lower()
    # Inject exactly one camera token per frame (two VGGT frames are averaged into one).
    tokens_per_frame = 1
    if video_token_id is None:
        return

    for sample in sample_list:
        if not isinstance(sample, dict):
            continue
        _inject_video_ids_to_encoded(sample, video_ids)
        _expand_single_sample_for_camera(
            sample,
            video_token_id=int(video_token_id),
            spatial_merge_size=spatial_merge_size,
            insert_position=insert_position,
            tokens_per_frame=tokens_per_frame,
        )


def _normalize_batch_video_ids(batch_video_ids) -> List[str]:
    if batch_video_ids is None:
        return []
    if isinstance(batch_video_ids, str):
        return [batch_video_ids]
    if isinstance(batch_video_ids, (list, tuple)):
        return [vid for vid in batch_video_ids if isinstance(vid, str) and vid]
    return []


def _normalize_batch_video_paths(batch_video_paths) -> List[str]:
    if batch_video_paths is None:
        return []
    if isinstance(batch_video_paths, str):
        return [batch_video_paths]
    if isinstance(batch_video_paths, (list, tuple)):
        return [path for path in batch_video_paths if isinstance(path, str) and path]
    return []


def _validate_batch_video_meta(
    template,
    batch: List[dict],
    batch_video_ids: List[str],
    batch_video_paths: List[str],
) -> None:
    if not _is_camera_injection_model(template):
        return

    num_videos = 0
    for sample in batch:
        grid = sample.get('video_grid_thw')
        if isinstance(grid, torch.Tensor):
            num_videos += int(grid.shape[0])

    if num_videos <= 0:
        return

    # CamDistill and CamInject cache mode rely on video_id to look up the cache.
    # CamInject online mode instead requires usable video paths.
    is_caminject_online = _is_caminject_model(template) and os.environ.get('VGGT_MODE', 'cache').strip().lower() == 'online'

    if is_caminject_online:
        if len(batch_video_paths) != num_videos:
            raise RuntimeError(
                f'[CamDistill/CamInject] online mode requires video_paths per video: '
                f'len(video_paths)={len(batch_video_paths)} vs num_videos={num_videos}. '
                'Check dataset video path fields and Template.encode patch.'
            )
        return

    if not batch_video_ids:
        raise RuntimeError(
            '[CamDistill/CamInject] batch contains video inputs but no video_ids were collected. '
            'Check dataset video path fields and Template.encode patch.'
        )

    if len(batch_video_ids) != num_videos:
        raise RuntimeError(
            f'[CamDistill/CamInject] batch video_ids count mismatch: '
            f'len(video_ids)={len(batch_video_ids)} vs num_videos={num_videos}. '
            'Check data path extraction and Template.encode patch.'
        )


def _lookup_camera_module(template, model):
    base_model = template.get_base_model(model)
    inner_model = getattr(base_model, 'model', None)
    if inner_model is None:
        return None
    return getattr(inner_model, '_camdistill_module', None)


def apply_video_id_patch():
    """
    Monkey-patch ms-swift Template:
      - encode: inject sample-level `video_ids`.
      - _data_collator_mm_data: forward `video_ids` to model.forward kwargs.
    """
    global _VIDEO_ID_PATCH_APPLIED
    if _VIDEO_ID_PATCH_APPLIED:
        return
    _VIDEO_ID_PATCH_APPLIED = True

    try:
        from swift.template import Template

        _original_encode = Template.encode
        _original_mm_collator = Template._data_collator_mm_data
        _original_forward_context = Template.forward_context
        _original_prepare_generate_kwargs = Template.prepare_generate_kwargs

        def _patched_encode(self, inputs, *args, **kwargs):
            # Extract video paths before encode; otherwise later preprocess may replace them with
            # tensor/list.
            video_paths = _extract_video_paths_from_inputs(inputs)
            video_ids = _extract_video_ids_from_inputs(inputs)
            result = _original_encode(self, inputs, *args, **kwargs)
            _inject_video_ids_to_encoded(result, video_ids)
            _inject_video_paths_to_encoded(result, video_paths)

            # Camera-token injection models: perform camera placeholder expansion during encode.
            if _is_camera_injection_model(self):
                _maybe_expand_encoded_for_camera(self, result, video_ids=video_ids)
            return result

        def _patched_mm_collator(self, batch: List[dict]):
            res = _original_mm_collator(self, batch)

            # First drop the raw fields possibly forwarded by the base collator so they never
            # reach model kwargs.
            res.pop('video_ids', None)
            res.pop('video_paths', None)

            if not _is_camera_injection_model(self):
                return res

            batch_video_ids = _collect_batch_video_ids(batch)
            batch_video_paths = _collect_batch_video_paths(batch)
            _validate_batch_video_meta(self, batch, batch_video_ids, batch_video_paths)
            if batch_video_ids:
                res['video_ids'] = batch_video_ids
            if batch_video_paths:
                res['video_paths'] = batch_video_paths
            return res

        def _patched_prepare_generate_kwargs(self, generate_kwargs, *, model=None):
            generate_kwargs = _original_prepare_generate_kwargs(self, generate_kwargs, model=model)

            # Uniformly remove helper fields before generate() so that HF's unused-model_kwargs
            # check does not raise.
            batch_video_ids = _normalize_batch_video_ids(generate_kwargs.pop('video_ids', None))
            batch_video_paths = _normalize_batch_video_paths(generate_kwargs.pop('video_paths', None))

            if not _is_camera_injection_model(self):
                return generate_kwargs

            # The generation path does not go through forward_context; write the metadata into
            # the adapter here and remove it from generate kwargs to keep HF's unused-kwargs
            # check happy.
            if not batch_video_ids and not batch_video_paths:
                return generate_kwargs

            adapter = _lookup_camera_module(self, model)
            if adapter is None:
                raise RuntimeError(
                    '[CamDistill/CamInject] camera module missing while generation batch carries video metadata.'
                )

            adapter._pending_video_ids = batch_video_ids
            if hasattr(adapter, '_pending_video_paths'):
                adapter._pending_video_paths = batch_video_paths

            return generate_kwargs

        @contextmanager
        def _patched_forward_context(self, model, inputs):
            # Move the video_ids forwarded by the collator into the camera module, avoiding the
            # need for an outer model.forward wrapper.
            if not _is_camera_injection_model(self):
                # Non camera-injection models also need to drop the helper fields so
                # forward/generate kwargs validation does not raise.
                inputs.pop('video_ids', None)
                inputs.pop('video_paths', None)
                with _original_forward_context(self, model, inputs):
                    yield
                return

            # Must pop from inputs to avoid generate()'s model_kwargs validation reporting
            # "unused model_kwargs: video_ids/video_paths".
            batch_video_ids = _normalize_batch_video_ids(inputs.pop('video_ids', None))
            batch_video_paths = _normalize_batch_video_paths(inputs.pop('video_paths', None))
            adapter = _lookup_camera_module(self, model)
            if adapter is None:
                if batch_video_ids or batch_video_paths:
                    raise RuntimeError('[CamDistill/CamInject] camera module missing while video metadata exists in batch.')
                with _original_forward_context(self, model, inputs):
                    yield
                return

            prev_pending = getattr(adapter, '_pending_video_ids', None)
            prev_pending_paths = getattr(adapter, '_pending_video_paths', None)
            adapter._pending_video_ids = batch_video_ids
            if hasattr(adapter, '_pending_video_paths'):
                adapter._pending_video_paths = batch_video_paths
            with _original_forward_context(self, model, inputs):
                try:
                    yield
                finally:
                    adapter._pending_video_ids = [] if prev_pending is None else prev_pending
                    if hasattr(adapter, '_pending_video_paths'):
                        adapter._pending_video_paths = [] if prev_pending_paths is None else prev_pending_paths

        Template.encode = _patched_encode
        Template._data_collator_mm_data = _patched_mm_collator
        Template.forward_context = _patched_forward_context
        Template.prepare_generate_kwargs = _patched_prepare_generate_kwargs
        print('[CamDistill/CamInject] Template encode/collator patched to propagate video_ids')

    except (ImportError, AttributeError) as e:
        print(f"[CamDistill/CamInject] WARNING: Could not patch Template for video_ids: {e}")
        print('  video_ids may be missing in forward; CamInject cache loading can degrade to zeros.')


# Automatically apply the patch when the plugin is loaded.
apply_video_id_patch()


print("[CamDistill] Plugin loaded. Model types:")
print("  - qwen3_vl_camdistill   (camera token distillation)")
print("  - qwen3_vl_caminject    (direct VGGT injection)")
