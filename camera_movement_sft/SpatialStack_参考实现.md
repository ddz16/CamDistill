# SpatialStack 参考实现分析

本文档分析 `/group/40009/dazhaodu/SpatialStack` 项目中将 VGGT 多层 3D 特征通过 DeepStack 方式融入 Qwen2.5-VL / Qwen3.5 的实现。

SpatialStack 是 VG-LLM 的进化版本，核心创新：**不再只用 VGGT 单层输出做后融合，而是把 VGGT 多层特征分别注入 ViT 或 LLM 的不同层** (DeepStack)。

---

## 1. 训练/冻结总览

```
Qwen2_5_VLForConditionalGenerationWithVGGT (or Qwen3_5)
├── visual (Qwen ViT)                          ❄️ 冻结
│   ├── blocks
│   └── merger                                  ❄️ 冻结
├── geometry_encoder (VGGT Aggregator)          ❄️ 冻结 (硬编码)
├── multi_layer_feature_fusion                  🔥 训练 ← DeepStack 核心
│   └── fusion_layers: {layer_idx: MLP/CrossAttn} × N
├── language_feature_fusion (可选)              🔥 训练
├── model (LLM)                                 🔥 训练
└── lm_head                                     🔥 训练
```

| 模块 | 状态 | 控制方式 |
|------|------|---------|
| `visual` (ViT) | ❄️ 冻结 | `tune_mm_vision=False` |
| `visual.merger` | ❄️ 冻结 | `tune_mm_mlp=False` |
| `geometry_encoder` (VGGT) | ❄️ 冻结 | 硬编码 |
| `multi_layer_feature_fusion` (DeepStack 融合模块) | 🔥 训练 | 新增可训练模块 |
| `language_feature_fusion` (LLM 侧融合) | 🔥 训练 | 可选 |
| `model` (LLM) | 🔥 训练 | `tune_mm_llm=True` |
| `lm_head` | 🔥 训练 | 跟随 LLM |

---

## 2. 训练脚本关键配置

**文件**: `scripts/train/train.sh`

```bash
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
GEOMETRY_ENCODER_PATH="facebook/VGGT-1B"
GEOMETRY_ENCODER_TYPE="vggt"

--tune_mm_llm True
--tune_mm_vision False
--tune_mm_mlp False
--use_geometry_encoder true

# DeepStack 核心配置:
FEATURE_FUSION_METHOD="deepstack_language_add"
GEOMETRY_FUSION_LAYERS="0 1 2"            # 注入 LLM 的第 0, 1, 2 层
GEOMETRY_ENCODER_LAYERS="11 17 23"        # 从 VGGT 取第 11, 17, 23 层特征
```

**含义**: 取 VGGT Aggregator 的 layer 11, 17, 23 的输出，分别注入到 LLM 的 layer 0, 1, 2。

---

## 3. VGGT 多层特征提取

### `encode_layers` 方法 (SpatialStack 新增)

与 VG-LLM 只取单层 `[-2]` 不同，SpatialStack 的 VGGTEncoder 支持提取多个层的特征：

```python
class VGGTEncoder(BaseGeometryEncoder):
    def encode_layers(self, images, layer_indices=[11, 17, 23], 
                      spatial_merge_size=2, include_camera_token=False):
        """提取 VGGT Aggregator 的多个指定层特征"""
        self.vggt.eval()
        images = self._apply_reference_frame_transform(images)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
        
        tensor_features = []
        for idx in layer_indices:  # e.g., [11, 17, 23]
            tokens = aggregated_tokens_list[idx][0]  # [S, 1374, 2048]
            
            patch_tokens = tokens[:, patch_start_idx:]  # 去掉 camera+register
            camera_token = tokens[:, 0:1]                # camera token
            
            # 2×2 空间重排 (与 Qwen ViT merger 对齐)
            patch_grid = patch_tokens.reshape(S, H_patch, W_patch, 2048)
            # trim → reshape(S, H/2, 2, W/2, 2, D) → permute → (S, H*W, D)
            patch_tokens = ... # 重排后的 patch tokens
            
            if include_camera_token:
                geo_feature = torch.cat([camera_token, patch_tokens], dim=1)
            else:
                geo_feature = patch_tokens
            
            tensor_features.append(geo_feature)
        
        return tensor_features  # 长度 = len(layer_indices), 每个 (S, N_merged, 2048)
```

**关键细节**:
- `aggregated_tokens_list[idx]` 是 VGGT 第 idx 层 frame+global attention 拼接后的缓存 (2048D)
- `layer_indices=[11, 17, 23]` 对应 VGGT cached_layer_indices `{4, 11, 17, 23}` 中的后三个
- `include_camera_token=False` 时只取 patch tokens (不含 camera/register)
- `include_camera_token=True` 时会把 camera token 拼在 patch tokens 前面
- 已经在内部做了 2×2 空间重排 (spatial_merge)

---

## 4. DeepStack 融合方式

SpatialStack 支持 4 种 DeepStack 融合方式：

| fusion_method | 融合位置 | 做法 |
|---|---|---|
| `deepstack_vision_add` | ViT 中间层 | VGGT 特征投影后 add 到 ViT hidden states |
| `deepstack_vision_cross_attn` | ViT 中间层 | ViT Q, VGGT K/V cross attention |
| **`deepstack_language_add`** (默认) | LLM decoder 层 | VGGT 特征投影后 add 到 LLM 中的视觉 token |
| `deepstack_language_cross_attn` | LLM decoder 层 | LLM 视觉 token Q, VGGT K/V cross attention |

### `multi_layer_feature_fusion` 和 `language_feature_fusion`

这两个其实是**同一个类** `MultiLayerFeatureFusionModule` 的不同实例：

```python
# modeling_qwen2_5_vl.py __init__ 中:

# 1. multi_layer_feature_fusion: 用于 deepstack_vision_* (ViT 侧融合)
#    或 deepstack_language_* 的默认融合
if "deepstack" in fusion_method:
    self.multi_layer_feature_fusion = MultiLayerFeatureFusionModule(config)

# 2. language_feature_fusion: 用于 deepstack_language_* 且有 vision_language_fusion_layers 时
#    (把 ViT 中间层特征也反馈到 LLM)
if "deepstack_language" in fusion_method and vision_language_fusion_layers is not None:
    self.language_feature_fusion = MultiLayerFeatureFusionModule(config)
else:
    # 没有额外 vision-language 融合时，复用 multi_layer_feature_fusion
    self.language_feature_fusion = self.multi_layer_feature_fusion
```

**实际训练脚本中** `VISION_LANGUAGE_FUSION_LAYERS` 为空，所以 `language_feature_fusion = multi_layer_feature_fusion`（同一个对象）。

### `MultiLayerFeatureFusionModule` 的结构

```python
class MultiLayerFeatureFusionModule(nn.Module):
    def __init__(self, config):
        # 核心: 为每个 geometry_fusion_layer 创建一个独立的 fusion_layer
        self.fusion_layers = nn.ModuleDict()
        for layer_num in config.geometry_fusion_layers:  # e.g., [0, 1, 2]
            self.fusion_layers[str(layer_num)] = nn.ModuleList([
                self._build_fusion_layer()  # 每层一个独立的 MLP/CrossAttn
            ])
```

所以 **`multi_layer_feature_fusion` 本质上是一个字典**，key 是 LLM 层号 (0, 1, 2)，value 是对应的 fusion 模块:

```
multi_layer_feature_fusion.fusion_layers = {
    "0": [fusion_layer_for_LLM_layer_0],   ← VGGT layer 11 的特征投影到这里
    "1": [fusion_layer_for_LLM_layer_1],   ← VGGT layer 17 的特征投影到这里
    "2": [fusion_layer_for_LLM_layer_2],   ← VGGT layer 23 的特征投影到这里
}
```

### 每个 fusion_layer 的具体结构 (即 "fusion MLP")

取决于 `fusion_method`:

#### `deepstack_language_add` (默认)

```python
fusion_layer = nn.ModuleDict({
    "geo_ln":  Qwen2RMSNorm(2048, eps=1e-6),      # 对 VGGT 特征做 RMSNorm
    "geo_mlp": nn.Sequential(
        nn.Linear(8192, 4096),   # 2048 * 2×2 = 8192 → 4096
        nn.GELU(),
        nn.Linear(4096, 3584),   # → LLM hidden_size
    )
})
```

前馈: `RMSNorm(geo_feats) → flatten(2×2×2048=8192) → Linear(8192→4096) → GELU → Linear(4096→3584) → add`

#### `deepstack_language_cross_attn`

```python
fusion_layer = nn.ModuleDict({
    "geo_ln":   Qwen2RMSNorm(2048, eps=1e-6),
    "geo_mlp":  nn.Sequential(Linear(8192, 4096), GELU, Linear(4096, 3584)),
    "cam_proj": nn.Sequential(RMSNorm(2048), Linear(2048, 4096), GELU, Linear(4096, 3584)),
    "cross_attn": CrossAttentionBlock(3584, num_heads=8, dropout=0.1),
})
```

前馈: patch tokens 走 `geo_ln + geo_mlp`，camera token 走 `cam_proj`，拼接后做 cross_attn

#### `deepstack_vision_add`

```python
fusion_layer = nn.Sequential(
    Qwen2RMSNorm(2048, eps=1e-6),
    nn.Sequential(
        nn.Linear(2048, 4096),       # 不做 2×2 merge (在 ViT 空间)
        nn.GELU(),
        nn.Linear(4096, 1152),       # → ViT hidden_size
    )
)
```

#### `deepstack_vision_cross_attn`

```python
fusion_layer = nn.ModuleDict({
    "geo_proj": nn.Sequential(LayerNorm(2048), Linear(2048, 1152)),
    "cross_attn": CrossAttentionBlock(1152, num_heads=8, dropout=0.1),
})
```

---

## 5. 在 LLM forward 中的具体注入逻辑

`Qwen2_5_VLModel.forward()` (即 LLM 的 `self.model`) 的 decoder loop 中:

```python
# 1325行: forward 接收 geometry_layer_features 和 fusion_module
def forward(self, ..., geometry_layer_features=None, fusion_module=None, image_mask=None, ...):

    # 正常 decoder layers 循环
    for layer_idx, decoder_layer in enumerate(self.layers):
        # 正常的 attention + MLP
        layer_outputs = decoder_layer(hidden_states, ...)
        hidden_states = layer_outputs[0]

        # ↓↓↓ DeepStack 注入点 ↓↓↓
        if geometry_layer_features is not None and layer_idx in geometry_layer_features:
            # 1. 用 image_mask 找到 hidden_states 中视觉 token 的位置
            vision_token_mask = image_mask[..., 0]  # [batch, seq_len]
            vision_tokens = hidden_states[vision_token_mask]  # [N_vis_tokens, LLM_dim]

            # 2. 调用 fusion_module (= multi_layer_feature_fusion)
            fused = fusion_module(
                vision_tokens,                          # LLM 中的视觉 token
                geometry_layer_features[layer_idx],     # VGGT 对应层特征
                layer_idx,                              # 层号
                vis_pos_embed_per_image,                # 可选 2D pos embed
                geo_pos_embed_per_image,                # 可选 2D pos embed
            )

            # 3. 写回 hidden_states (只修改视觉 token 位置)
            hidden_states = hidden_states.clone()
            hidden_states[vision_token_mask] = fused
```

**关键**: 只修改 LLM hidden states 中**视觉 token 对应位置**的值，文本 token 不受影响。

---

## 5. 零初始化 (Zero Init)

```python
def reset_residual_branches_to_noop(self) -> None:
    """对 deepstack_language_add 的 MLP 最后一层做零初始化"""
    if self.config.fusion_method == "deepstack_language_add":
        for fusion_layer in self.fusion_layers.values():
            self._zero_init_last_linear(fusion_layer["geo_mlp"])

@staticmethod
def _zero_init_last_linear(module):
    """将 MLP 最后一个 Linear 的 weight 和 bias 初始化为 0"""
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return
```

**设计意图**: 训练开始时，geo_mlp 输出全零 → add 不改变原始 LLM 行为 → 模型从"无几何信息"状态平滑过渡到"有几何信息"状态，避免一开始就破坏预训练的 LLM 表征。

---

## 6. 完整数据流 (deepstack_language_add)

```
输入 images
    │
    ├─── Qwen ViT (冻结) ───→ image_embeds [N, 3584]
    │                              │
    │                              ▼ 注入 LLM 的 <image> token 位置
    │                              │
    ├─── VGGT Aggregator (冻结):
    │    ├─ layer 11 → features_11 [S, N_merged, 2048]
    │    ├─ layer 17 → features_17 [S, N_merged, 2048]
    │    └─ layer 23 → features_23 [S, N_merged, 2048]
    │
    ▼
LLM Forward:
    Layer 0: hidden_states += fusion_layer_0(features_11)   ← DeepStack 注入
    Layer 1: hidden_states += fusion_layer_1(features_17)   ← DeepStack 注入
    Layer 2: hidden_states += fusion_layer_2(features_23)   ← DeepStack 注入
    Layer 3: normal LLM forward
    Layer 4: normal LLM forward
    ...
    Layer N: → logits
```

**注入时机**: 在 LLM decoder layer 的 forward 过程中，在指定层的 attention+MLP 之后，将 VGGT 特征 add 到视觉 token 对应的位置。

---

## 7. 与 VG-LLM 的关键区别

| | VG-LLM | SpatialStack |
|--|--------|-------------|
| **VGGT 层数** | 单层 (`[-2]`) | **多层** (`[11, 17, 23]`) |
| **融合位置** | ViT 输出之后 (post-merger) | **LLM 层内** (deepstack) |
| **融合方式** | 单次 add | **分层逐层 add/cross_attn** |
| **零初始化** | 无 | **有** (geo_mlp 最后 Linear 零初始) |
| **Camera Token** | 不用 | **可选** (`include_camera_token`) |
| **支持模型** | Qwen2.5-VL 仅 | Qwen2.5-VL + **Qwen3.5** |
| **fusion_layers 数** | 1 个融合模块 | **3 个独立模块** (每层一个) |

---

## 8. 配置参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `feature_fusion_method` | `deepstack_language_add` | 融合方式 |
| `geometry_fusion_layers` | `[0, 1, 2]` | VGGT 特征注入 LLM 的哪些层 |
| `geometry_encoder_layers` | `[11, 17, 23]` | 从 VGGT 取哪些层的输出 |
| `include_camera_token` | `False` | 是否包含 camera token |
| `pos_encoding_type` | `"none"` | 位置编码: none/rope2d/sincos2d |
| `vision_language_fusion_layers` | `None` | LLM 层特征反馈到视觉 (高级) |
| `geometry_merger_type` | `"mlp"` | Merger 类型 |

**层级映射**:
```
geometry_encoder_layers=[11, 17, 23]  ←→  geometry_fusion_layers=[0, 1, 2]

VGGT layer 11 的特征  →  注入到 LLM layer 0
VGGT layer 17 的特征  →  注入到 LLM layer 1
VGGT layer 23 的特征  →  注入到 LLM layer 2
```

---

## 9. 融合模块参数量估算

每个 `deepstack_language_add` 的 fusion_layer:
- `geo_ln`: RMSNorm(2048) = 2048 params
- `geo_mlp`: Linear(8192→4096) + Linear(4096→3584) = 8192×4096 + 4096×3584 ≈ **48M**

总共 3 层: **~144M** 可训练参数 (仅融合模块)

---

## 10. 文件路径

```
/group/40009/dazhaodu/SpatialStack/
├── src/qwen_vl/
│   ├── train/
│   │   ├── train_qwen.py              # 训练入口, set_model 冻结策略
│   │   └── argument.py                # 参数定义 (含 geometry_fusion_layers 等)
│   ├── model/
│   │   ├── modeling_qwen2_5_vl.py     # 主模型 WithVGGT 版本
│   │   ├── modeling_qwen3_5.py        # Qwen3.5 版本
│   │   ├── feature_fusion.py          # MultiLayerFeatureFusionModule (DeepStack 核心)
│   │   ├── position_utils.py          # 位置编码工具
│   │   └── geometry_encoders/
│   │       └── vggt_encoder.py        # encode_layers 多层提取
│   └── data/
│       └── data_qwen.py
└── scripts/
    └── train/train.sh                  # 默认 deepstack_language_add
```
