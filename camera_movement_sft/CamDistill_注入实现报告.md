# CamDistill Camera Token 注入实现报告

## 1. 方案概述

将 CameraTokenModule 生成的 camera token 作为独立 token 插入 LLM 输入序列（每帧 visual tokens 前 1 个）。

```
ViT (冻结) → 中间层缓存 → CameraTokenModule (可训练) → camera_embeds
                                                          │
ViT (冻结) → Merger → video_embeds ──────────────────────┘
                                                          │
                                              合并: [CAM, vis_1..P, CAM, vis_{P+1}..2P, ...]
                                                          │
                                              扩展 input_ids/attention_mask/...
                                                          │
                                              position_ids (camera = 帧中心)
                                                          │
                                              LLM forward → loss
```

## 2. 修改的文件

| 文件 | 行数 | 作用 |
|------|------|------|
| `plugins/modeling_qwen3_vl_camdistill.py` | 1992 | 复制自 transformers Qwen3-VL + CamDistill 修改 |
| `plugins/modeling_qwen3_5_camdistill.py` | 2355 | 复制自 transformers Qwen3.5 + CamDistill 修改 |
| `plugins/camdistill_plugin.py` | 395 | ms-swift 插件入口 |
| `plugins/camdistill_model.py` | ~400 | CameraTokenModule 定义 |
| `plugins/camdistill_loss.py` | ~165 | 蒸馏 Loss |

## 3. 每个模型文件的 5 处修改

### 修改 1: 辅助函数 (文件顶部)

```python
def _inject_camera_into_video_embeds(video_embeds, camera_embeds, video_grid_thw, spatial_merge_size=2):
    """每帧 visual tokens 前面插 1 个 camera embed"""
    # video_embeds: (N_vis, D) → (N_vis + T_total, D)

def _expand_video_placeholders(input_ids, mm_token_type_ids, attention_mask, video_token_id, ...):
    """input_ids 中每帧前面多插 1 个 video_token_id"""
    # seq_len → seq_len + T_total

def _expand_inputs_embeds_for_camera(inputs_embeds, new_input_ids, video_token_id, T_total):
    """inputs_embeds 同步扩展"""
```

### 修改 2: ViT forward 中缓存中间层

```python
# Qwen3VLVisionModel.forward / Qwen3_5VisionModel.forward:
camdistill_layer_cache = []
camdistill_extract_layers = getattr(self, '_camdistill_extract_layers', None)

for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(hidden_states, ...)
    if camdistill_extract_layers is not None and layer_num in camdistill_extract_layers:
        camdistill_layer_cache.append(hidden_states.detach())

if camdistill_extract_layers is not None:
    self._camdistill_layer_cache = camdistill_layer_cache
```

### 修改 3: `get_vision_position_ids` 加 camera 位置

```python
def get_vision_position_ids(self, ..., include_camera_token: bool = False):
    # 原有逻辑...
    if include_camera_token:
        # 每帧 patch positions 前面插 1 个 camera position (帧中心)
        cam_center_h = start_position + llm_grid_h // 2
        cam_center_w = start_position + llm_grid_w // 2
        for t_idx in range(llm_grid_t):
            cam_pos = [[cam_temporal], [cam_center_h], [cam_center_w]]
            new_pos_list.append(cam_pos)
            new_pos_list.append(frame_pos)
        vision_position_ids = torch.cat(new_pos_list, dim=1)
```

### 修改 4: `get_rope_index` 对 video 段传 `include_camera_token=True`

```python
# image == 1, video == 2
else:
    grid_thw = next(grid_iters[modality_type])
    _include_cam = getattr(self, '_camdistill_enabled', False) and modality_type == 2
    vision_position_ids = self.get_vision_position_ids(
        ..., include_camera_token=_include_cam)
```

### 修改 5: `forward` 中注入 camera tokens

```python
if pixel_values_videos is not None:
    video_outputs = self.get_video_features(...)
    video_embeds = torch.cat(video_outputs.pooler_output, dim=0)

    # === CamDistill 注入 ===
    camdistill_module = getattr(self, '_camdistill_module', None)
    vit_intermediates = getattr(self.visual, '_camdistill_layer_cache', [])
    if camdistill_module and vit_intermediates and self._camdistill_enabled:
        camera_embeds, _ = camdistill_module(vit_intermediates, video_grid_thw)
        video_embeds = _inject_camera_into_video_embeds(video_embeds, camera_embeds, ...)
        self.visual._camdistill_layer_cache = []
        # 扩展序列
        input_ids, mm_token_type_ids, attention_mask = _expand_video_placeholders(...)
        inputs_embeds = _expand_inputs_embeds_for_camera(...)

    # 正常的 get_placeholder_mask + masked_scatter (数量匹配)
    _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
```

## 4. Plugin 初始化 (camdistill_plugin.py)

```python
class CamDistillQwen3VLLoader(Qwen3VLLoader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        model = super().get_model(...)

        # 创建 CameraTokenModule
        camdistill = CameraTokenModule(hidden_dim, num_heads, depth, llm_hidden_dim)

        # 设置到模型上
        inner_model._camdistill_module = camdistill
        inner_model._camdistill_enabled = True
        inner_model.visual._camdistill_extract_layers = {3, 7, 11, 15, 19, 23}
        inner_model.visual._camdistill_layer_cache = []

        # Monkey-patch forward 为修改版
        inner_model.forward = types.MethodType(Qwen3VLModelCamDistill.forward, inner_model)
        inner_model.get_vision_position_ids = types.MethodType(...)
        inner_model.get_rope_index = types.MethodType(...)
```

## 5. Qwen3-VL vs Qwen3.5 的差异

| | Qwen3-VL | Qwen3.5 |
|--|--|--|
| ViT 有 deepstack | 是 | 否 (更简单) |
| 视频 tokens 在 input_ids 中 | 连续一整块 | 被 timestamps 分隔成多段 |
| `_expand_video_placeholders` | 整块操作 | 逐段检测连续 video segment 的起始 |
| position_ids 维度 | 3D `(3, B, L)` | 3D `(3, B, L)` — 相同 |
| `get_rope_index` 逻辑 | 相同 | 相同 |

## 6. 已发现并修复的 Bug

| # | Bug | 影响 | 状态 |
|---|-----|------|------|
| 1 | `class Qwen3VLVisionPatchEmbed` 声明缺失 | 文件无法 import | ✅ 已修复 |
| 2 | Qwen3-VL 和 Qwen3.5 视频 tokens 都不连续 (version='v3' 用 timestamps 分隔) | `_expand_video_placeholders` 假设连续块会出错 | ✅ 已重写为逐 token 扫描 |
| 3 | deepstack_video_embeds 长度不匹配 (仅 Qwen3-VL) | 注入 camera tokens 后 video_embeds 变长, 但 deepstack embeds 没扩展 → shape mismatch | ✅ 已修复: 对 deepstack embeds 同步插入零 |

## 7. 剩余潜在问题

| # | 问题 | 风险 | 说明 |
|---|------|------|------|
| 1 | labels 扩展 | 高 | 外层 forward wrapper 中需提前扩展 labels (+T_total 个 -100)，当前 plugin 已有实现但需验证 |
| 2 | `loss_scale` tensor | 中 | ms-swift 可能传 loss_scale，也需要同步扩展 |
| 3 | model.save_pretrained | 低 | 需确认 CameraTokenModule 参数在 state_dict 中 |
| 4 | 推理/生成阶段 | 低 | `past_key_values is None` 条件保证只在 prefill 注入 |

## 8. 如何运行

```bash
# CamDistill 训练 (Qwen3-VL-8B)
CAMDISTILL_DEPTH=6 \
VGGT_CACHE_DIR=/path/to/vggt_cache \
swift sft \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --external_plugins camera_movement_sft/plugins/camdistill_plugin.py \
    --model_type qwen3_vl_camdistill \
    --loss_type camdistill \
    ...
```

## 9. 文件关系图

```
camdistill_plugin.py (入口)
    ├── 注册 CamDistillLoss → camdistill_loss.py
    ├── CamDistillQwen3VLLoader
    │   ├── 创建 CameraTokenModule → camdistill_model.py
    │   ├── 设置 ViT 缓存层 (_camdistill_extract_layers)
    │   ├── Monkey-patch forward → modeling_qwen3_vl_camdistill.py
    │   └── 外层 forward wrapper (labels 扩展)
    ├── CamDistillQwen35Loader
    │   └── 同上 → modeling_qwen3_5_camdistill.py
    ├── CamInjectQwen3VLLoader
    │   ├── 创建 CamInjectAdapter → caminject_model.py
    │   ├── 设置 _camdistill_mode='direct'
    │   ├── Monkey-patch forward (共用注入逻辑)
    │   └── 外层 forward wrapper (video_ids pop + labels 扩展)
    ├── CamInjectQwen35Loader
    │   └── 同上
    └── Template.encode monkey-patch (添加 video_ids 字段)
```

## 10. CamInject 方案说明

CamInject 与 CamDistill **共享完全相同的 camera token 注入逻辑**:
- 使用相同的 `_inject_camera_into_video_embeds`
- 使用相同的 `_expand_video_placeholders`
- 使用相同的 `get_vision_position_ids` (camera 在帧中心)
- 使用相同的 `get_rope_index` (include_camera_token=True)

区别仅在于 camera_embeds 的来源:

| | CamDistill | CamInject |
|---|---|---|
| camera_embeds 来源 | CameraTokenModule(vit_intermediates) | VGGT_cache → VGGTProjector |
| 需要 ViT 中间层缓存 | 是 | 否 |
| `_camdistill_mode` | `'learn'` (默认) | `'direct'` |
| forward 中条件 | 需要 vit_intermediates 非空 | 跳过 vit_intermediates 检查 |
| 帧数对齐 | 不需要 (T 天然一致) | 需要 (S帧→T帧, 2帧平均池化) |
| Loss | SFT + distill | 仅 SFT |
| 推理时 | 不需要 VGGT | 需要 VGGT |

**CamInject 的帧数对齐**:
```
VGGT cache: (S, 2048) — S=原始帧数 (fps×duration, max=100)
模型需要:   (T, 2048) — T=S/temporal_patch_size = S/2

对齐方法:
  - S == T*2: 精确 2帧平均池化 (最常见)
  - S > T: adaptive_avg_pool1d
  - S < T: nearest interpolation (极少见)
```

**Video ID 传递机制**:
```
JSONL sample: {"videos": ["path/to/abc.mp4"], ...}
    ↓ Template.encode (patched)
encoded: {..., "video_ids": "abc"}
    ↓ DataCollator
batch: {..., "video_ids": ["abc", "def"]}
    ↓ model.forward(**batch)
outer wrapper: video_ids = kwargs.pop("video_ids")
    ↓ adapter._pending_video_ids = ["abc", "def"]
    ↓ adapter.forward() → 加载 abc.pt, def.pt → 对齐 → 投影
```
