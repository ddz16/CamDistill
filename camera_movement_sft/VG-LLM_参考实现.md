# VG-LLM 参考实现分析

本文档分析 `/group/40009/dazhaodu/VG-LLM` 项目中将 VGGT 3D 几何特征融合到 Qwen2.5-VL 的实现方式，供我们的 CamDistill 方案参考。

---

## 1. 整体架构

```
Qwen2_5_VLForConditionalGenerationWithVGGT
├── visual (Qwen2.5-VL ViT)                    ❄️ 冻结
│   ├── patch_embed
│   ├── blocks × 32
│   └── merger (2×2 空间合并 + MLP)             ❄️ 冻结
├── geometry_encoder (VGGT Aggregator)          ❄️ 冻结 (硬编码, 始终冻结)
├── geometry_merger (GeometryFeatureMerger)     🔥 训练
│   └── RMSNorm + MLP (8192 → 4096 → 3584)
├── feature_fusion (FeatureFusionModule)        🔥 训练 (add 模式下无参数)
├── model (Qwen2.5 LLM layers)                 🔥 训练
│   ├── embed_tokens
│   ├── layers × N
│   └── norm
└── lm_head                                     🔥 训练
```

---

## 2. 训练/冻结配置

**源码**: `src/qwen_vl/train/train_qwen.py` 中的 `set_model` 函数:

```python
def set_model(model_args, model):
    # ViT: 冻结
    if model_args.tune_mm_vision:  # False
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    # Merger (ViT → LLM 投影): 冻结
    if model_args.tune_mm_mlp:  # False
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    # LLM: 训练
    if model_args.tune_mm_llm:  # True
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    # VGGT: 始终冻结 (硬编码)
    if model_args.use_geometry_encoder:
        for n, p in model.geometry_encoder.named_parameters():
            p.requires_grad = False
```

**训练脚本** (`scripts/train/train_3d.sh`):

```bash
--tune_mm_llm True          # LLM 训练
--tune_mm_vision False      # ViT 冻结
--tune_mm_mlp False         # Merger 冻结
--use_geometry_encoder True # 启用 VGGT (始终冻结)
--feature_fusion_method "add"
--geometry_encoder_type "vggt"
--geometry_encoder_path "facebook/VGGT-1B"
```

| 模块 | 状态 | 参数控制 |
|------|------|---------|
| `visual` (ViT) | ❄️ 冻结 | `tune_mm_vision=False` |
| `visual.merger` | ❄️ 冻结 | `tune_mm_mlp=False` |
| `geometry_encoder` (VGGT) | ❄️ 冻结 | 硬编码 |
| `geometry_merger` | 🔥 训练 | 默认可训练 |
| `feature_fusion` | 🔥 训练 | add 模式下无参数 |
| `model` (LLM) | 🔥 训练 | `tune_mm_llm=True` |
| `lm_head` | 🔥 训练 | 跟随 `tune_mm_llm` |

---

## 3. VGGT Geometry Encoder

**源码**: `src/qwen_vl/model/geometry_encoders/vggt_encoder.py`

```python
class VGGTEncoder(BaseGeometryEncoder):
    def __init__(self, config):
        super().__init__(config)
        from ..vggt.models.vggt import VGGT
        
        # 只启用 Aggregator, 不需要各种 Head
        self.vggt = VGGT(enable_camera=False, enable_point=False, 
                         enable_depth=False, enable_track=False)
        
        # 始终冻结
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False
        
        self.reference_frame = config.reference_frame
        self.patch_size = 14

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        输入: images [S, 3, H, W] (518×518)
        输出: features [S, P, 2048]  (P = patch 数)
        """
        self.vggt.eval()
        images = self._apply_reference_frame_transform(images)
        
        dtype = torch.bfloat16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
                # 取 倒数第二层, 只取 patch tokens (跳过 camera+register)
                features = aggregated_tokens_list[-2][0, :, patch_start_idx:]
        
        features = self._apply_inverse_reference_frame_transform(features)
        return features  # (S, 1369, 2048)
    
    def get_feature_dim(self) -> int:
        return 2048
```

**关键细节**:
- 取 `aggregated_tokens_list[-2]` (倒数第二层, index=17), 不是最后一层
- 只取 **patch tokens** (`[:, patch_start_idx:]`), 跳过 camera token 和 register tokens
- 输出 shape: `(S, 1369, 2048)` — 每帧 1369 个 patch, 每个 2048 维
- 参考帧支持 "first" 或 "last" (翻转帧序)

> **与我们 CamDistill 的区别**: 我们只用 camera token (每帧 1 个, 2048D); VG-LLM 用全部 patch tokens (每帧 1369 个, 2048D) — 因为它做的是 3D 场景理解(检测/描述), 需要空间密集特征。

---

## 4. GeometryFeatureMerger (关键模块)

**源码**: `src/qwen_vl/model/feature_fusion.py`

```python
class GeometryFeatureMerger(nn.Module):
    """
    将 VGGT 的 3D 特征投影到 LLM 维度, 并做 2×2 空间合并 (与 Qwen2.5-VL merger 对齐)。
    """
    
    def __init__(self, output_dim, hidden_dim, context_dim, 
                 spatial_merge_size=2, merger_type="mlp"):
        super().__init__()
        self.merger_type = merger_type
        self.input_dim = context_dim * (spatial_merge_size ** 2)  # 2048 * 4 = 8192
        self.hidden_dim = hidden_dim    # 4096
        self.output_dim = output_dim    # 3584 (LLM hidden_size)
        self.merge_size = spatial_merge_size  # 2
        
        if merger_type == "mlp":
            self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)  # RMSNorm(2048)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),   # 8192 → 4096
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),  # 4096 → 3584
            )
        elif merger_type == "avg":
            self.mlp = nn.Sequential(
                nn.Linear(context_dim, self.hidden_dim),      # 2048 → 4096
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),  # 4096 → 3584
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x [n_image, H_patch, W_patch, 2048]
             e.g., [S, 37, 37, 2048] (518px / 14 = 37)
        输出: [n_image, H_patch//2, W_patch//2, 3584]
             e.g., [S, 18, 18, 3584]
        """
        n_image, h_patch, w_patch, dim = x.shape
        
        # 对齐到 merge_size 的整数倍
        x = x[:, :h_patch // self.merge_size * self.merge_size, 
                :w_patch // self.merge_size * self.merge_size, :]
        
        # 2×2 空间合并: [N, H/2, 2, W/2, 2, D] → [N, H/2, W/2, 2, 2, D]
        x = x.reshape(n_image, h_patch // self.merge_size, self.merge_size,
                       w_patch // self.merge_size, self.merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        if self.merger_type == "mlp":
            # RMSNorm → 展平 2×2 → MLP
            x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
            # shape: [N * H/2 * W/2, 3584]
        elif self.merger_type == "avg":
            # 2×2 平均池化 → MLP
            x = x.mean(dim=(3, 4))  # [N, H/2, W/2, D]
            x = x.view(-1, dim)
            x = self.mlp(x)
        
        # 恢复空间维度
        x = x.reshape(n_image, h_patch // self.merge_size, 
                       w_patch // self.merge_size, -1)
        return x  # [n_image, H/2, W/2, 3584]
```

### 初始化参数

```python
# modeling_qwen2_5_vl.py 第 1610 行:
self.geometry_merger = GeometryFeatureMerger(
    output_dim=config.hidden_size,          # 3584 (Qwen2.5-VL-7B)
    hidden_dim=4096,                         # geometry_merger_hidden_dim
    context_dim=self.geometry_encoder.get_feature_dim(),  # 2048
    spatial_merge_size=2,
    merger_type="mlp"                        # 默认
)
```

### MLP 模式前馈流程

```
VGGT patch features
[S, 37, 37, 2048]
        │
        ▼ 对齐到偶数 (37→36)
[S, 36, 36, 2048]
        │
        ▼ reshape: 2×2 空间合并
[S, 18, 18, 2, 2, 2048]
        │
        ▼ permute
[S, 18, 18, 2, 2, 2048]
        │
        ▼ RMSNorm(2048) + flatten(2×2×2048=8192)
[S*18*18, 8192]
        │
        ▼ Linear(8192 → 4096) + GELU + Linear(4096 → 3584)
[S*18*18, 3584]
        │
        ▼ reshape
[S, 18, 18, 3584]
```

**设计意图**: 与 Qwen2.5-VL 原生 `merger` (也是 2×2 合并 + MLP) 完全对齐，保证两路特征在空间分辨率和维度上一致，可以直接 element-wise 相加。

---

## 5. FeatureFusionModule (融合层)

**源码**: `src/qwen_vl/model/feature_fusion.py`

```python
class FeatureFusionModule(nn.Module):
    def __init__(self, config: FeatureFusionConfig):
        self.fusion_method = config.fusion_method  # "add"
        self.hidden_size = config.hidden_size      # 3584
    
    def forward(self, features_2d, features_3d):
        if self.fusion_method == "add":
            return features_2d + features_3d
        
        elif self.fusion_method == "concat":
            concat = torch.cat([norm1(features_2d), norm2(features_3d)], dim=-1)
            return self.projection(concat)  # Linear(7168 → 3584)
        
        elif self.fusion_method == "cross_attention":
            # 2D 作 Q, 3D 作 K/V, 带 2D sincos pos embedding
            x = features_2d
            for block in self.cross_attn_blocks:
                x = block(x, features_3d, h_grid, w_grid)
            return x
        
        elif self.fusion_method == "gated":
            gate = sigmoid(Linear(cat(2D, 3D)))
            return gate * features_2d + (1 - gate) * features_3d
        
        elif self.fusion_method == "weighted":
            return (α/(α+β)) * features_2d + (β/(α+β)) * features_3d
```

### CrossAttentionBlock (当 fusion_method="cross_attention" 时)

```python
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size=3584, num_heads=8, dropout=0.1):
        self.norm1_query = LayerNorm(hidden_size)
        self.norm1_key = LayerNorm(hidden_size)
        self.norm1_value = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8,
            dropout=0.1, batch_first=True)
        
        self.mlp = nn.Sequential(
            Linear(3584, 14336), GELU(), Dropout(0.1),
            Linear(14336, 3584), Dropout(0.1))
    
    def forward(self, features_2d, features_3d, h_grid, w_grid):
        query = norm1_query(features_2d) + pos_embed_2d
        key = norm1_key(features_3d) + pos_embed_2d
        value = norm1_value(features_3d)
        
        attn_out = cross_attention(query, key, value)
        x = features_2d + attn_out       # 第一个残差
        x = x + mlp(norm2(x))            # 第二个残差
        return x
```

**配置** (train_3d.sh 中用 `fusion_num_layers=1`):

```python
# 初始化:
fusion_config = FeatureFusionConfig(
    fusion_method="add",        # 默认直接相加
    hidden_size=3584,           # LLM hidden
    num_heads=8,
    dropout=0.1,
    num_layers=1                # cross_attention 时用 1 层
)
self.feature_fusion = FeatureFusionModule(fusion_config)
```

---

## 6. 完整数据流

```python
# modeling_qwen2_5_vl.py 中 forward 方法 (第 1939 行):

def forward(self, ..., pixel_values, geometry_encoder_inputs, ...):
    # 1. Qwen2.5-VL ViT 提取 2D 特征
    image_embeds = self.visual(pixel_values, grid_thw)
    # image_embeds shape: [total_patches_merged, 3584]
    
    # 2. VGGT 提取 3D 几何特征 + 融合
    if use_geometry_encoder and geometry_encoder_inputs is not None:
        image_embeds = self._process_geometry_features(image_embeds, geometry_encoder_inputs)
    
    # 3. 嵌入到 LLM token 序列中
    inputs_embeds = self.model.embed_tokens(input_ids)
    inputs_embeds[image_mask] = image_embeds  # 替换 <image> 位置
    
    # 4. LLM forward
    outputs = self.model(inputs_embeds=inputs_embeds, ...)
    logits = self.lm_head(outputs.last_hidden_state)
```

```python
def _process_geometry_features(self, image_embeds, geometry_encoder_inputs):
    """
    对每个 batch 样本:
    1. VGGT encode → [S, H_patch, W_patch, 2048]
    2. geometry_merger → [S, H/2, W/2, 3584]  (空间合并 + MLP)
    3. flatten → [S * H/2 * W/2, 3584]
    4. feature_fusion(image_embeds, geo_embeds) → fused [N, 3584]
    """
    for bn in range(batch_size):
        if geometry_encoder_inputs[bn].shape[0] > 0:
            n_image, _, height, width = geometry_encoder_inputs[bn].shape
            
            # VGGT encode (冻结, no_grad)
            features = self.geometry_encoder.encode(geometry_encoder_inputs[bn])
            # features: [S, P, 2048], P = 1369 patches
            
            # Reshape to spatial grid
            features = features.reshape(n_image, 
                height // self.geometry_encoder.patch_size,   # 37
                width // self.geometry_encoder.patch_size,    # 37
                -1)  # [S, 37, 37, 2048]
            
            # GeometryFeatureMerger: 2×2合并 + MLP
            features = self.geometry_merger(features)
            # [S, 18, 18, 3584]
    
    # Flatten and fuse
    geo_embeds = geo_features.view(-1, hidden_size)  # [total_merged_patches, 3584]
    image_embeds = self.feature_fusion(image_embeds, geo_embeds)
    
    return image_embeds  # [total_merged_patches, 3584]
```

---

## 7. 融合时序：先投影再 add

**核心结论: 两路特征各自独立投影到 LLM 空间 (3584) 后，再 element-wise add。**

```
Qwen2.5-VL ViT                              VGGT
[pixels]                                     [pixels]
    │                                            │
    ▼ ViT blocks (1152 dim)                     ▼ Aggregator (1024 dim → concat 2048)
    │                                            │
    ▼ Qwen2_5_VLPatchMerger                     ▼ GeometryFeatureMerger
    │  RMSNorm(1152)                             │  RMSNorm(2048)
    │  2×2合并: 4×1152 = 4608                     │  2×2合并: 4×2048 = 8192
    │  Linear(4608 → 4608)                       │  Linear(8192 → 4096)
    │  GELU                                      │  GELU
    │  Linear(4608 → 3584)                       │  Linear(4096 → 3584)
    │                                            │
    ▼ image_embeds [N, 3584]                    ▼ geo_embeds [N, 3584]
    │                                            │
    └────────────── ADD ─────────────────────────┘
                     │
                     ▼
              fused [N, 3584]  ──→  注入 LLM 的 <image> token 位置
```

### 代码调用链

```python
# modeling_qwen2_5_vl.py forward() 中:

# Step 1: Qwen ViT 前馈 → 内部最后一步经过 merger → 输出已在 LLM 空间
image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
# image_embeds shape: [total_merged_patches, 3584] ← 已经在 LLM 空间

# Step 2: VGGT → geometry_merger 投影到 LLM 空间 → 与 image_embeds add
image_embeds = self._process_geometry_features(image_embeds, geometry_encoder_inputs)
```

### 视觉 token 的投影 (`Qwen2_5_VLPatchMerger`)

这是 Qwen2.5-VL 原生的 ViT → LLM 投影模块:

```python
class Qwen2_5_VLPatchMerger(nn.Module):
    """原生 Qwen2.5-VL 的 ViT output → LLM space 投影"""
    def __init__(self, dim=3584, context_dim=1152, spatial_merge_size=2):
        # hidden_size = 1152 * (2*2) = 4608
        self.hidden_size = context_dim * (spatial_merge_size ** 2)  # 4608
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)           # RMSNorm(1152)
        self.mlp = nn.Sequential(
            nn.Linear(4608, 4608),   # 保持维度
            nn.GELU(),
            nn.Linear(4608, 3584),   # 投影到 LLM hidden_size
        )

    def forward(self, x):
        # 输入: x [总patches, 1152] (ViT 输出)
        # 外部已做 2×2 spatial reshape+flatten, 所以这里 view(-1, 4608) 是 4 个 patch 拼起来
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x  # [总merged_patches, 3584]
```

### 3D 特征的投影 (`GeometryFeatureMerger`)

与原生 merger 结构完全对齐:

```python
class GeometryFeatureMerger(nn.Module):
    """VGGT 3D features → LLM space 投影 (模仿原生 merger 结构)"""
    def __init__(self, output_dim=3584, hidden_dim=4096, context_dim=2048, spatial_merge_size=2):
        # input_dim = 2048 * (2*2) = 8192
        self.input_dim = context_dim * (spatial_merge_size ** 2)  # 8192
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)          # RMSNorm(2048)
        self.mlp = nn.Sequential(
            nn.Linear(8192, 4096),   # 降维
            nn.GELU(),
            nn.Linear(4096, 3584),   # 投影到 LLM hidden_size
        )

    def forward(self, x):
        # 输入: x [n_image, H_patch, W_patch, 2048]
        # 2×2 空间合并 + flatten
        x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        return x  # [总merged_patches, 3584]
```

### 两个 Merger 的对比

| | Qwen2_5_VLPatchMerger (原生) | GeometryFeatureMerger (VG-LLM 新增) |
|--|--|--|
| **输入来源** | Qwen2.5 ViT (1152D) | VGGT Aggregator (2048D) |
| **空间合并** | 2×2 | 2×2 (对齐) |
| **合并后维度** | 4×1152 = 4608 | 4×2048 = 8192 |
| **MLP 结构** | 4608 → 4608 → 3584 | 8192 → 4096 → 3584 |
| **归一化** | RMSNorm(1152) | RMSNorm(2048) |
| **输出维度** | 3584 | 3584 (对齐) |
| **训练状态** | ❄️ 冻结 | 🔥 训练 |

**设计逻辑**: 两路投影的输出维度完全一致 (都是 3584)，空间分辨率也一致 (都经过 2×2 合并)，所以可以直接 element-wise add。VG-LLM 的 GeometryFeatureMerger 就是模仿原生 merger 的结构设计的。

---

## 9. 与我们 CamDistill 的对比

| 维度 | VG-LLM | 我们的 CamDistill |
|------|--------|-----------------|
| **任务** | 3D 场景理解 (检测/描述) | 运镜识别 (时序分析) |
| **VGGT 使用方式** | 在线推理 (训练时跑 VGGT) | 离线预提取 cache |
| **提取的特征** | Patch tokens (每帧 1369 个) | Camera token (每帧 1 个) |
| **特征层级** | 倒数第二层 `[-2]` | 最后一层 `[-1]` |
| **特征维度** | 1369 × 2048 / 帧 | 1 × 2048 / 帧 |
| **空间合并** | 2×2 (对齐 Qwen ViT merger) | 无 (camera token 无空间维) |
| **融合方式** | add / cross_attn (在 ViT 输出后) | 注入 LLM 序列 (作为额外 token) |
| **额外可训练参数** | GeometryMerger (~33M) + LLM | CameraTokenModule (~200M) + LLM |
| **推理时需要 VGGT** | 是 | 否 (蒸馏后自主生成) |
| **基础模型** | Qwen2.5-VL-7B | Qwen3-VL-4B/8B, Qwen3.5-4B/9B |

### 关键设计差异

1. **VG-LLM 用密集 patch tokens**: 因为 3D 场景任务需要逐像素的空间信息 (检测框、描述位置)
2. **我们用 camera token**: 因为运镜分析只需要全局几何变化 (相机怎么动), 不需要逐像素空间
3. **VG-LLM 在线跑 VGGT**: 训练时每个 batch 实时跑 VGGT forward → 占 GPU 显存
4. **我们离线 cache**: 预提取 camera token 存磁盘 → 训练时不需要 VGGT → 省显存
5. **VG-LLM 直接 add 融合**: 简单, 但推理时必须带 VGGT
6. **我们蒸馏**: 训练时学 VGGT 的 camera token, 推理时自主生成 → 无 VGGT 依赖

---

## 10. 参考价值

VG-LLM 的实现对我们有以下参考意义:

1. **GeometryFeatureMerger 的 2×2 空间合并**: 如果我们将来要融合 patch-level 3D 特征, 这个对齐方式值得参考
2. **feature_fusion 的多种方案**: add / concat / cross_attention / gated 都已实现可对比
3. **VGGT 始终冻结**: 验证了冻结 VGGT 只训练融合层和 LLM 是可行的
4. **用倒数第二层而非最后一层**: VG-LLM 选择 `[-2]`, 可能倒数第二层包含更多空间细节 (最后一层更抽象)

---

## 11. 文件路径参考

```
/group/40009/dazhaodu/VG-LLM/
├── src/qwen_vl/
│   ├── train/
│   │   ├── train_qwen.py          # 训练主入口, set_model 定义冻结策略
│   │   ├── argument.py            # ModelArguments (tune_mm_*, geometry_* 参数)
│   │   └── trainer.py             # 自定义 Trainer
│   ├── model/
│   │   ├── modeling_qwen2_5_vl.py # 主模型 (含 WithVGGT 版本, _process_geometry_features)
│   │   ├── feature_fusion.py      # FeatureFusionModule + GeometryFeatureMerger
│   │   ├── geometry_encoders/
│   │   │   ├── vggt_encoder.py    # VGGTEncoder (冻结 VGGT, 取 patch tokens)
│   │   │   ├── base.py            # BaseGeometryEncoder
│   │   │   └── factory.py         # create_geometry_encoder
│   │   └── vggt/                  # VGGT 模型代码副本
│   └── data/
│       └── data_qwen.py           # 数据处理
└── scripts/
    └── train/train_3d.sh          # 训练启动脚本
```
