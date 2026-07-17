"""
CamDistill Plugin: ms-swift 外部插件入口

通过 --external_plugins camera_movement_sft/plugins/camdistill_plugin.py 加载。
注册自定义模型 loader 和蒸馏 loss。

Camera Token 注入方式:
  使用修改版的 modeling_qwen3_vl_camdistill.py (从 transformers 复制后修改),
  在 forward 中直接完成 camera token 的独立插入 (每帧 visual tokens 前插 1 个)。
"""

import os
import sys
import json
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import List

# 确保 plugins 目录在 path 中
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

# ============================================================
# 1. 注册自定义 Loss
# ============================================================
from camdistill_loss import CamDistillLoss  # noqa: F401 (side-effect: registers in loss_map)
import camera_loss_scale  # noqa: F401 (side-effect: registers loss_scale 'camera_value')

# ============================================================
# 2. 注册自定义 Model Loader (带 CamDistill 模块)
# ============================================================
from swift.model import ModelMeta, ModelGroup, Model, register_model
from swift.model.models.qwen import Qwen3VLLoader
from swift.model.constant import MLLMModelType
from swift.template.constant import TemplateType

from camdistill_model import CameraTokenModule

# 导入修改版模型的 forward 和辅助函数 (用于 monkey-patch)
import types
from modeling_qwen3_vl_camdistill import (
    _inject_camera_into_video_embeds,
    _expand_video_placeholders,
    _expand_inputs_embeds_for_camera,
)


def _resolve_camdistill_extract_layers(vit_depth: int, camdistill_depth: int) -> List[int]:
    """解析要缓存的 ViT 层索引。"""
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
    在 labels 中为每帧的 K=tokens_per_frame 个 camera token 位置插入 -100 (不计算 loss)。
    与 _expand_video_placeholders 完全对齐。
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

        # 复制 video 之前
        new_labels[b, :vid_start] = labels[b, :vid_start]

        # 逐帧插入 K 个 -100 (camera) + 原始 visual labels
        src_offset = vid_start
        dst_offset = vid_start
        for n_tokens in frame_token_counts:
            # Camera token 位置 (K 个): -100 (已由 full 初始化)
            dst_offset += K
            # 原始 visual token labels
            new_labels[b, dst_offset:dst_offset + n_tokens] = labels[b, src_offset:src_offset + n_tokens]
            dst_offset += n_tokens
            src_offset += n_tokens

        # 复制 video 之后
        remaining = seq_len - vid_end
        if remaining > 0:
            new_labels[b, dst_offset:dst_offset + remaining] = labels[b, vid_end:]

    return new_labels


class CamDistillQwen3VLLoader(Qwen3VLLoader):
    """
    CamDistill Loader for Qwen3-VL.

    使用修改版 modeling 文件加载模型, 在 forward 中直接注入 camera tokens。
    模型内部通过 _camdistill_enabled 标志控制是否注入。
    ViT 中间层通过 forward hooks 收集, 供 CameraTokenModule 使用。
    """

    def get_model(self, model_dir, config, processor, model_kwargs):
        model = super().get_model(model_dir, config, processor, model_kwargs)

        # 读取配置
        vit_hidden_dim = config.vision_config.hidden_size    # 1024 or 1152
        vit_num_heads = config.vision_config.num_heads       # 16
        vit_depth = config.vision_config.depth               # 24 or 27
        llm_hidden_dim = config.vision_config.out_hidden_size  # 2560 or 4096

        # CamDistill 配置
        camdistill_depth = int(os.environ.get("CAMDISTILL_DEPTH", "6"))

        # 计算 hook 层: 默认均匀取层，也支持显式指定 CAMDISTILL_EXTRACT_LAYERS
        extract_layers = _resolve_camdistill_extract_layers(vit_depth, camdistill_depth)

        print(f"[CamDistill] Initializing CameraTokenModule:")
        print(f"  vit_hidden_dim={vit_hidden_dim}, vit_depth={vit_depth}")
        print(f"  llm_hidden_dim={llm_hidden_dim}")
        print(f"  camdistill_depth={camdistill_depth}, extract_layers={extract_layers}")

        # 创建 CameraTokenModule
        # cam_dim 固定 1024: camera_features=2*cam_dim=2048, 与 VGGT 缓存对齐(VGGT stream=1024)。
        # 必须显式指定, 否则默认取 ViT hidden(4B=1024 恰好=2048; 8B=1152→2304 与 VGGT 不匹配报错)。
        camdistill = CameraTokenModule(
            hidden_dim=vit_hidden_dim,
            num_heads=vit_num_heads,
            depth=camdistill_depth,
            llm_hidden_dim=llm_hidden_dim,
            cam_dim=int(os.environ.get("CAMDISTILL_CAM_DIM", "1024")),
        )
        # CamDistill (learn) 强制 K=1, 因为 CameraTokenModule 一帧只输出 1 个 token
        camdistill.tokens_per_frame = 1

        # 设置模型属性 (供修改版 forward 使用)
        inner_model = model.model  # Qwen3VLModel
        inner_model._camdistill_module = camdistill
        inner_model._camdistill_enabled = True
        inner_model._camdistill_mode = 'learn'
        inner_model._camdistill_preexpanded_input = True
        inner_model._camdistill_video_ids = []

        # 也在顶层设置引用 (供 loss 使用)
        model.camdistill = camdistill

        # 从 checkpoint 显式恢复 CamDistill 模块参数，避免被主加载阶段当作 UNEXPECTED 丢弃。
        _try_load_camdistill_module_from_checkpoint(model_dir, camdistill)

        # 在 ViT 上设置要缓存的中间层索引 (不用 hooks, 直接在 ViT forward 中缓存)
        visual_module = inner_model.visual
        visual_module._camdistill_extract_layers = set(extract_layers)
        visual_module._camdistill_layer_cache = []

        print(f"[CamDistill] ViT intermediate layer caching configured: {extract_layers}")
        print(f"[CamDistill] CameraTokenModule parameters: "
              f"{sum(p.numel() for p in camdistill.parameters()) / 1e6:.1f}M")

        # 替换内层模型的 forward 为修改版 (含 camera token 注入逻辑)
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
        # 同时替换 get_vision_position_ids 和 get_rope_index
        inner_model.get_vision_position_ids = types.MethodType(
            Qwen3VLModelCamDistill.get_vision_position_ids, inner_model
        )
        inner_model.get_rope_index = types.MethodType(
            Qwen3VLModelCamDistill.get_rope_index, inner_model
        )

        print('[CamDistill] Inner forward patched for camera token injection (mode=learn, preexpanded=1)')

        return model


# ============================================================
# 3. 注册模型到 ms-swift
# ============================================================

# Qwen3-VL 系列 (CamDistill 版)
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

# Qwen3.5 系列 (CamDistill 版)
try:
    from swift.model.models.qwen import Qwen3_5Loader

    class CamDistillQwen35Loader(Qwen3_5Loader):
        """Qwen3.5 版本的 CamDistill Loader"""

        def get_model(self, model_dir, config, processor, model_kwargs):
            model = super().get_model(model_dir, config, processor, model_kwargs)

            vit_hidden_dim = config.vision_config.hidden_size
            vit_num_heads = config.vision_config.num_heads
            vit_depth = config.vision_config.depth
            llm_hidden_dim = config.vision_config.out_hidden_size

            camdistill_depth = int(os.environ.get("CAMDISTILL_DEPTH", "6"))
            extract_layers = _resolve_camdistill_extract_layers(vit_depth, camdistill_depth)

            print(f"[CamDistill-Qwen3.5] Initializing:")
            print(f"  vit_hidden_dim={vit_hidden_dim}, llm_hidden_dim={llm_hidden_dim}")

            # cam_dim 固定 1024 -> camera_features=2048 对齐 VGGT (同上, 避免 ViT hidden≠1024 时维度不匹配)
            camdistill = CameraTokenModule(
                hidden_dim=vit_hidden_dim,
                num_heads=vit_num_heads,
                depth=camdistill_depth,
                llm_hidden_dim=llm_hidden_dim,
                cam_dim=int(os.environ.get("CAMDISTILL_CAM_DIM", "1024")),
            )
            camdistill.tokens_per_frame = 1  # CamDistill 强制 K=1

            # Qwen3.5 的内部结构: model.model = Qwen3_5Model
            inner_model = model.model
            inner_model._camdistill_module = camdistill
            inner_model._camdistill_enabled = True
            inner_model._camdistill_mode = 'learn'
            inner_model._camdistill_preexpanded_input = True
            inner_model._camdistill_video_ids = []
            model.camdistill = camdistill

            # 从 checkpoint 显式恢复 CamDistill 模块参数，避免被主加载阶段当作 UNEXPECTED 丢弃。
            _try_load_camdistill_module_from_checkpoint(model_dir, camdistill)

            # 在 ViT 上设置缓存层
            visual_module = inner_model.visual
            visual_module._camdistill_extract_layers = set(extract_layers)
            visual_module._camdistill_layer_cache = []

            print(f"[CamDistill-Qwen3.5] Parameters: "
                  f"{sum(p.numel() for p in camdistill.parameters()) / 1e6:.1f}M")

            # 替换 forward 为修改版
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
            inner_model.get_vision_position_ids = types.MethodType(
                Qwen3VLModelCamDistill.get_vision_position_ids, inner_model
            )
            inner_model.get_rope_index = types.MethodType(
                Qwen3VLModelCamDistill.get_rope_index, inner_model
            )
            print(f"[CamDistill-Qwen3.5] Forward patched")
            return model

    register_model(
        ModelMeta(
            "qwen3_5_camdistill",
            [
                ModelGroup([
                    Model("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B"),
                    Model("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B"),
                ], TemplateType.qwen3_5),
            ],
            CamDistillQwen35Loader,
            architectures=["Qwen3_5ForConditionalGeneration"],
            is_multimodal=True,
        ),
    )
    print("[CamDistill] Qwen3.5 CamDistill models registered.")
except ImportError:
    print("[CamDistill] WARNING: Qwen3_5Loader not found, Qwen3.5 CamDistill not available.")


# ============================================================
# 4. VGGT-Direct Baseline (完整注入实现)
# ============================================================
from vggt_direct_model import VGGTDirectCameraAdapter


def _try_load_camdistill_module_from_checkpoint(model_dir: str, camdistill_module: nn.Module) -> None:
    """从 checkpoint 手动恢复 CamDistill 模块参数。

    背景: 插件模块在 base model 加载后才挂接，checkpoint 中 `camdistill.*` / `model._camdistill_module.*`
    常被主加载流程标记为 UNEXPECTED。这里主动恢复，确保推理评测使用训练好的 CamDistill 权重。
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


def _try_load_vggt_projector_from_checkpoint(model_dir: str, adapter: nn.Module) -> None:
    """从 checkpoint 手动恢复 VGGT-Direct projector 参数。

    背景: ms-swift 先加载基础模型再由插件挂接 adapter，导致 checkpoint 里的
    `model._camdistill_module.projector.*` / `vggt_direct_adapter.projector.*`
    在主加载阶段可能被报告为 UNEXPECTED。这里做一次显式恢复，避免 projector 随机初始化。
    """
    projector = getattr(adapter, 'projector', None)
    if projector is None:
        return

    target_state = projector.state_dict()
    target_keys = list(target_state.keys())
    key_prefixes = (
        'model._camdistill_module.projector.',
        'vggt_direct_adapter.projector.',
        '_camdistill_module.projector.',
        'projector.',
    )

    loaded = {}
    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    single_path = os.path.join(model_dir, 'model.safetensors')

    try:
        from safetensors import safe_open
    except Exception as e:
        print(f"[VGGT-Direct] WARNING: safetensors not available, skip projector restore: {e}")
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
        print(f"[VGGT-Direct] WARNING: projector restore failed while reading checkpoint: {e}")
        return

    if not loaded:
        return

    try:
        incompatible = projector.load_state_dict(loaded, strict=False)
    except Exception as e:
        print(f"[VGGT-Direct] WARNING: projector restore failed while loading state_dict: {e}")
        return

    missing = [k for k in target_keys if k not in loaded]
    if missing or incompatible.unexpected_keys:
        print(
            f"[VGGT-Direct] WARNING: projector partial restore, loaded={len(loaded)}/{len(target_keys)}, "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
    else:
        print(f"[VGGT-Direct] projector restored from checkpoint: {len(loaded)} tensors")


def _make_vggt_direct_loader(base_loader_cls, model_file_module_name: str):
    """
    工厂函数: 为指定的 base loader 创建 VGGT-Direct 版本的 Loader。

    VGGT-Direct 与 CamDistill 共享完全相同的注入逻辑 (modified forward),
    唯一区别是 camera_embeds 来源:
      - CamDistill: CameraTokenModule(vit_intermediates) -> camera_embeds
      - VGGT-Direct: VGGT_cache -> VGGTProjector -> camera_embeds

    通过设置 _camdistill_mode='direct', modified forward 会跳过 vit_intermediates 检查,
    直接调用 VGGTDirectCameraAdapter (它忽略 vit_intermediates, 从 cache 加载)。
    """

    class VGGTDirectLoader(base_loader_cls):
        def get_model(self, model_dir, config, processor, model_kwargs):
            model = super().get_model(model_dir, config, processor, model_kwargs)

            llm_hidden_dim = config.vision_config.out_hidden_size
            temporal_patch_size = getattr(config.vision_config, 'temporal_patch_size', 2)
            vggt_cache_dir = os.environ.get("VGGT_CACHE_DIR", "")

            if not vggt_cache_dir:
                print("[VGGT-Direct] WARNING: VGGT_CACHE_DIR not set! Camera tokens will be zeros.")

            # 创建 VGGTDirectCameraAdapter (与 CameraTokenModule 相同接口)
            adapter = VGGTDirectCameraAdapter(
                llm_dim=llm_hidden_dim,
                vggt_dim=2048,
                temporal_patch_size=temporal_patch_size,
            )
            adapter.set_cache_dir(vggt_cache_dir)

            # 设置到 inner model 上 (与 CamDistill 相同的属性名)
            inner_model = model.model
            inner_model._camdistill_module = adapter
            inner_model._camdistill_enabled = True
            inner_model._camdistill_mode = 'direct'  # 关键: 告诉 forward 不需要 vit_intermediates
            inner_model._camdistill_preexpanded_input = True

            # 也在顶层设置引用
            model.vggt_direct_adapter = adapter

            # 显式恢复 projector 权重，避免 checkpoint 中插件参数在主加载阶段被当作 UNEXPECTED 后丢失。
            _try_load_vggt_projector_from_checkpoint(model_dir, adapter)

            # 不需要设置 ViT 中间层缓存 (VGGT-Direct 不用)
            # inner_model.visual._camdistill_extract_layers 不设置

            print(f"[VGGT-Direct] VGGTDirectCameraAdapter initialized:")
            print(f"  llm_dim={llm_hidden_dim}, temporal_patch_size={temporal_patch_size}")
            print(f"  cache_dir={vggt_cache_dir}")
            print(f"  projector params: {sum(p.numel() for p in adapter.projector.parameters()) / 1e6:.1f}M")

            # Monkey-patch forward (使用修改版的 forward 进行 camera token 注入)
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
            print(f"[VGGT-Direct] Inner forward patched for camera token injection (mode=direct)")

            return model

    VGGTDirectLoader.__name__ = f"VGGTDirect{base_loader_cls.__name__}"
    VGGTDirectLoader.__qualname__ = VGGTDirectLoader.__name__
    return VGGTDirectLoader


# Qwen3-VL VGGT-Direct
VGGTDirectQwen3VLLoader = _make_vggt_direct_loader(Qwen3VLLoader, "modeling_qwen3_vl_camdistill")

register_model(
    ModelMeta(
        "qwen3_vl_vggt_direct",
        [
            ModelGroup([
                Model("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"),
                Model("Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"),
            ], TemplateType.qwen3_vl),
        ],
        VGGTDirectQwen3VLLoader,
        architectures=["Qwen3VLForConditionalGeneration"],
        is_multimodal=True,
    ),
)

# Qwen3.5 VGGT-Direct
try:
    VGGTDirectQwen35Loader = _make_vggt_direct_loader(Qwen3_5Loader, "modeling_qwen3_vl_camdistill")

    register_model(
        ModelMeta(
            "qwen3_5_vggt_direct",
            [
                ModelGroup([
                    Model("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B"),
                    Model("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B"),
                ], TemplateType.qwen3_5),
            ],
            VGGTDirectQwen35Loader,
            architectures=["Qwen3_5ForConditionalGeneration"],
            is_multimodal=True,
        ),
    )
    print("[VGGT-Direct] Qwen3.5 VGGT-Direct models registered.")
except NameError:
    pass


# ============================================================
# 5. Video ID 注入机制 (Template monkey-patch)
# ============================================================
# 目标: 让 VGGT-Direct 在 model.forward 中稳定拿到 video_ids。
# 实现:
#   1. patch Template.encode: 在样本级 features 增加 `video_ids`
#   2. patch Template._data_collator_mm_data: 将 batch 内的 `video_ids` 显式透传到模型
#
# 说明:
#   - 仅 patch encode 不够，默认 collator 会丢弃未知字符串字段。
#   - 双 patch 后，video_ids 会和 video_grid_thw 一起进入 forward kwargs。

_VIDEO_ID_PATCH_APPLIED = False
_VGGT_DIRECT_MODEL_TYPES = {'qwen3_vl_vggt_direct', 'qwen3_5_vggt_direct'}
_CAMDISTILL_MODEL_TYPES = {'qwen3_vl_camdistill', 'qwen3_5_camdistill'}


def _is_vggt_direct_model(template) -> bool:
    model_type = getattr(getattr(template, 'model_meta', None), 'model_type', None)
    if isinstance(model_type, str) and model_type:
        return model_type in _VGGT_DIRECT_MODEL_TYPES
    model_type_env = os.environ.get('MODEL_TYPE', '')
    return model_type_env in _VGGT_DIRECT_MODEL_TYPES


def _is_camdistill_model(template) -> bool:
    model_type = getattr(getattr(template, 'model_meta', None), 'model_type', None)
    if isinstance(model_type, str) and model_type:
        return model_type in _CAMDISTILL_MODEL_TYPES
    model_type_env = os.environ.get('MODEL_TYPE', '')
    return model_type_env in _CAMDISTILL_MODEL_TYPES


def _is_camera_injection_model(template) -> bool:
    return _is_vggt_direct_model(template) or _is_camdistill_model(template)


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
    """在 encode 阶段扩展 sample 序列: 每帧前插或后插 K=tokens_per_frame 个 camera placeholder。

    注意: 推理阶段通常没有 labels，因此不能把 labels 作为扩展前置条件。
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
            f"[VGGT-Direct] unsupported tokens_per_frame={K}, expected 1"
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
                    f"[VGGT-Direct] {field_name} length mismatch before camera expansion: "
                    f"got={len(value)}, expected={old_len}"
                )
            return torch.tensor(value, dtype=list_dtype), 'list'

        if isinstance(value, torch.Tensor):
            tensor = value
            if tensor.dim() == 2 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.dim() != 1 or tensor.shape[0] != old_len:
                raise RuntimeError(
                    f"[VGGT-Direct] {field_name} length mismatch before camera expansion: "
                    f"got={tuple(tensor.shape)}, expected=({old_len},)"
                )
            return tensor, 'tensor'

        raise RuntimeError(
            f"[VGGT-Direct] unsupported {field_name} type before camera expansion: {type(value)}"
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
        raise RuntimeError('[VGGT-Direct] video_grid_thw exists but input_ids has no video token placeholders')
    shifted = torch.zeros_like(vid_mask)
    shifted[1:] = vid_mask[:-1]
    segment_count = int((vid_mask & ~shifted).sum().item())
    if segment_count != t_total:
        raise RuntimeError(
            f"[VGGT-Direct] segment_count mismatch before camera expansion: "
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
            f"[VGGT-Direct] expanded input_ids length mismatch: got={new_len}, "
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
            f"[VGGT-Direct] dst_positions length mismatch: got={dst_positions.numel()}, expected={old_len}"
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
    """兼容 OpenAI message content 格式里的视频字段。"""
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
    """从 dict / TemplateInputs / StdTemplateInputs 中提取视频路径（保持顺序，不跨字段去重）。"""
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
    """从 dict / TemplateInputs / StdTemplateInputs 中提取原始视频路径并转成 video_id。"""
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

    # 单视频样本保持 str，兼容旧逻辑；多视频样本保存 list。
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
    """将 batch 里的 video_ids (str 或 list[str]) 展平成 list[str]。"""
    batch_video_ids: List[str] = []
    for sample in batch:
        # 只为真正参与视频分支的样本收集 video_ids，避免与 video_grid_thw 计数不一致。
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
    """将 batch 里的 video_paths (str 或 list[str]) 展平成 list[str]。"""
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
    在 encode 阶段完成 camera placeholder 扩展，保证 collator 看到的是最终序列。
    仅对 qwen3_vl / qwen3_5 模板启用。
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
    # 每帧固定注入 1 个 camera token (VGGT 两帧平均成 1 个)
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

    # CamDistill 和 VGGT-Direct cache 模式: 强依赖 video_id 查 cache。
    # VGGT-Direct online 模式: 需要可用的视频路径。
    is_vggt_online = _is_vggt_direct_model(template) and os.environ.get('VGGT_MODE', 'cache').strip().lower() == 'online'

    if is_vggt_online:
        if len(batch_video_paths) != num_videos:
            raise RuntimeError(
                f'[CamDistill/VGGT-Direct] online mode requires video_paths per video: '
                f'len(video_paths)={len(batch_video_paths)} vs num_videos={num_videos}. '
                'Check dataset video path fields and Template.encode patch.'
            )
        return

    if not batch_video_ids:
        raise RuntimeError(
            '[CamDistill/VGGT-Direct] batch contains video inputs but no video_ids were collected. '
            'Check dataset video path fields and Template.encode patch.'
        )

    if len(batch_video_ids) != num_videos:
        raise RuntimeError(
            f'[CamDistill/VGGT-Direct] batch video_ids count mismatch: '
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
      - encode: 注入 sample-level `video_ids`
      - _data_collator_mm_data: 将 `video_ids` 透传到 model.forward kwargs
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
            # 在 encode 前提取视频路径，避免后续 preprocess 把路径替换为 tensor/list。
            video_paths = _extract_video_paths_from_inputs(inputs)
            video_ids = _extract_video_ids_from_inputs(inputs)
            result = _original_encode(self, inputs, *args, **kwargs)
            _inject_video_ids_to_encoded(result, video_ids)
            _inject_video_paths_to_encoded(result, video_paths)

            # Camera-token injection 模型: 在 encode 阶段完成 camera placeholder 扩展
            if _is_camera_injection_model(self):
                _maybe_expand_encoded_for_camera(self, result, video_ids=video_ids)
            return result

        def _patched_mm_collator(self, batch: List[dict]):
            res = _original_mm_collator(self, batch)

            # 先去掉可能由基类 collator 透传进来的原始字段，避免后续进入 model kwargs。
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

            # generate() 前统一移除辅助字段，避免 HF 校验 unused model_kwargs 报错。
            batch_video_ids = _normalize_batch_video_ids(generate_kwargs.pop('video_ids', None))
            batch_video_paths = _normalize_batch_video_paths(generate_kwargs.pop('video_paths', None))

            if not _is_camera_injection_model(self):
                return generate_kwargs

            # 生成路径不会走 forward_context；在这里把 metadata 写入 adapter，
            # 同时从 generate kwargs 移除，避免 HF generate 的 unused kwargs 报错。
            if not batch_video_ids and not batch_video_paths:
                return generate_kwargs

            adapter = _lookup_camera_module(self, model)
            if adapter is None:
                raise RuntimeError(
                    '[CamDistill/VGGT-Direct] camera module missing while generation batch carries video metadata.'
                )

            adapter._pending_video_ids = batch_video_ids
            if hasattr(adapter, '_pending_video_paths'):
                adapter._pending_video_paths = batch_video_paths

            return generate_kwargs

        @contextmanager
        def _patched_forward_context(self, model, inputs):
            # 将 collator 透传的 video_ids 放入 camera 模块，避免外层 model.forward wrapper。
            if not _is_camera_injection_model(self):
                # 非 camera 注入模型也要清掉辅助字段，防止 forward/generate kwargs 校验报错。
                inputs.pop('video_ids', None)
                inputs.pop('video_paths', None)
                with _original_forward_context(self, model, inputs):
                    yield
                return

            # 必须从 inputs 中 pop 掉，避免 generate() 的 model_kwargs 校验报
            # "unused model_kwargs: video_ids/video_paths"。
            batch_video_ids = _normalize_batch_video_ids(inputs.pop('video_ids', None))
            batch_video_paths = _normalize_batch_video_paths(inputs.pop('video_paths', None))
            adapter = _lookup_camera_module(self, model)
            if adapter is None:
                if batch_video_ids or batch_video_paths:
                    raise RuntimeError('[CamDistill/VGGT-Direct] camera module missing while video metadata exists in batch.')
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
        print('[CamDistill/VGGT-Direct] Template encode/collator patched to propagate video_ids')

    except (ImportError, AttributeError) as e:
        print(f"[CamDistill/VGGT-Direct] WARNING: Could not patch Template for video_ids: {e}")
        print('  video_ids may be missing in forward; VGGT-Direct cache loading can degrade to zeros.')


# 自动应用 patch (插件加载时)
apply_video_id_patch()


print("[CamDistill] Plugin loaded. Model types:")
print("  - qwen3_vl_camdistill   (camera token 蒸馏)")
print("  - qwen3_5_camdistill    (camera token 蒸馏)")
print("  - qwen3_vl_vggt_direct  (VGGT 直接注入)")
print("  - qwen3_5_vggt_direct   (VGGT 直接注入)")
