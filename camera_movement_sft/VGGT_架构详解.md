# VGGT & VGGT-Omega 架构详解

本文档对 VGGT (Visual Geometry Grounded Transformer) 和 VGGT-Omega 的模型架构进行代码级别的深入解析，覆盖从输入预处理到最终 Camera Token 输出的完整前馈流程。

---

## 目录

1. [总体对比概览](#1-总体对比概览)
2. [VGGT 完整架构](#2-vggt-完整架构)
   - [2.1 顶层模型结构 (vggt.py)](#21-顶层模型结构)
   - [2.2 Aggregator 核心架构 (aggregator.py)](#22-aggregator-核心架构)
   - [2.3 Patch Embedding 层 (DINOv2 ViT-L/14)](#23-patch-embedding-层)
   - [2.4 特殊 Token 设计](#24-特殊-token-设计)
   - [2.5 交替注意力机制 (Alternating Attention)](#25-交替注意力机制)
   - [2.6 Block 结构 (block.py)](#26-block-结构)
   - [2.7 Attention 模块 (attention.py)](#27-attention-模块)
   - [2.8 2D Rotary Position Embedding (rope.py)](#28-2d-rotary-position-embedding)
   - [2.9 中间层缓存机制](#29-中间层缓存机制)
   - [2.10 Camera Head (camera_head.py)](#210-camera-head)
3. [VGGT-Omega 完整架构](#3-vggt-omega-完整架构)
   - [3.1 顶层模型结构 (vggt_omega.py)](#31-顶层模型结构)
   - [3.2 Aggregator 核心架构](#32-aggregator-核心架构)
   - [3.3 Patch Embedding 层 (DINOv3 ViT-L/16)](#33-patch-embedding-层)
   - [3.4 特殊 Token 设计](#34-特殊-token-设计)
   - [3.5 混合帧间注意力机制](#35-混合帧间注意力机制)
   - [3.6 SelfAttentionBlock (block.py)](#36-selfattentionblock)
   - [3.7 SelfAttention 模块 (attention.py)](#37-selfattention-模块)
   - [3.8 RoPE Position Embedding (rope_position_encoding.py)](#38-rope-position-embedding)
   - [3.9 Camera Head](#39-camera-head)
4. [关键差异深度对比](#4-关键差异深度对比)
5. [CamDistill 中我们使用的部分](#5-camdistill-中我们使用的部分)

---

## 1. 总体对比概览

| 维度 | VGGT | VGGT-Omega |
|------|------|------------|
| **总参数量** | ~1.26B | ~1.14B |
| **Patch Embed** | DINOv2 ViT-L/14 (frozen backbone → feature extractor) | DINOv3 ViT-L/16 (集成到 Aggregator, 非独立 backbone) |
| **输入分辨率** | 518×518 | 可变 (默认 512×512) |
| **Patch Size** | 14×14 → 37×37=1369 patches | 16×16 → 32×32=1024 patches |
| **Embed Dim** | 1024 | 1024 |
| **Aggregator 深度** | 24 层 (frame blocks + global blocks = 48 个 Block) | 24 层 (frame blocks + inter-frame blocks = 48 个 Block) |
| **Num Heads** | 16 | 16 |
| **Register Tokens** | 4 个 | 16 个 |
| **Camera Token** | 1 个/帧 (shape `(1,2,1,1024)`, 首帧/非首帧两种) | 1 个/帧 (shape `(1,2,1,1024)`, 首帧/非首帧两种) |
| **帧间注意力** | 纯 Global (所有 token 参与) | 混合: Global + Register-only (5层只让 camera+register 交互) |
| **RoPE** | `RotaryPositionEmbedding2D`, base=100, 绝对坐标 | `RopePositionEmbedding`, base=100, normalize_coords="max" |
| **QK Norm** | 有 (LayerNorm on q,k) | 有 (LayerNorm on q,k, 通过 `use_qk_norm` 控制) |
| **K Bias Mask** | 无 | 有 (`LinearKMaskedBias`, 对 K 的 bias 置零) |
| **Camera Head** | 4-iteration AdaLN 迭代细化 | 单次通过 4 层 Transformer + MLP |
| **Camera Head 输入** | `aggregated_tokens[-1][:,:,0]` → shape `[B,S,2048]` | `aggregated_tokens[-1][:,:,:patch_start]` → shape `[B,S,17,2048]` |
| **Camera 输出** | 9D: Translation(3) + Quaternion(4) + FoV(2) | 9D: Translation(3) + Quaternion(4) + FoV(2) |
| **init_values (LayerScale)** | 0.01 | 1e-5 |

---

## 2. VGGT 完整架构

### 2.1 顶层模型结构

**文件**: `/group/40009/dazhaodu/vggt/vggt/models/vggt.py`

```python
class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        self.aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)    # 2048
        self.point_head = DPTHead(dim_in=2048, output_dim=4)   # 3D points
        self.depth_head = DPTHead(dim_in=2048, output_dim=2)   # Depth
        self.track_head = TrackHead(dim_in=2048, patch_size=14) # Point tracking
```

**前馈流程**:
```
images [B, S, 3, 518, 518]
    │
    ▼
aggregator(images)  →  (aggregated_tokens_list, patch_start_idx)
    │                      List[Tensor | None], 长度=24
    │                      非None的shape: [B, S, P, 2048]
    ▼
camera_head(aggregated_tokens_list)  →  pose_enc_list (4次迭代)
    │                                    最终输出: [B, S, 9]
    ▼
predictions["pose_enc"] = pose_enc_list[-1]  # shape [B, S, 9]
```

**输入规范**:
- `images`: shape `[B, S, 3, H, W]` 或 `[S, 3, H, W]` (无 batch 时自动加)
- 值域: `[0, 1]` (RGB, 归一化在 Aggregator 内部完成)
- S: 序列帧数（可变长）

---

### 2.2 Aggregator 核心架构

**文件**: `/group/40009/dazhaodu/vggt/vggt/models/aggregator.py`

这是 VGGT 的核心，实现了 **交替注意力** (Alternating Attention) 机制。

```python
class Aggregator(nn.Module):
    def __init__(self,
        img_size=518, patch_size=14, embed_dim=1024,
        depth=24, num_heads=16, mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        patch_embed="dinov2_vitl14_reg",  # 使用 DINOv2 ViT-L/14 作为 patch embed
        aa_order=["frame", "global"],      # 交替注意力顺序
        aa_block_size=1,                   # 每种注意力连续执行的层数
        qk_norm=True,
        rope_freq=100,                     # 2D RoPE 基频
        init_values=0.01,                  # LayerScale 初始值
        cached_layer_indices=(4, 11, 17, 23),  # 缓存的中间层
    ):
```

**关键属性**:

| 属性 | 值 | 说明 |
|------|------|------|
| `self.frame_blocks` | `ModuleList[Block × 24]` | 帧内注意力的 24 个独立 Block |
| `self.global_blocks` | `ModuleList[Block × 24]` | 帧间注意力的 24 个独立 Block |
| `self.camera_token` | `Parameter(1, 2, 1, 1024)` | 相机 Token (2 种变体) |
| `self.register_token` | `Parameter(1, 2, 4, 1024)` | Register Token (2 种变体) |
| `self.patch_start_idx` | `5` (= 1 camera + 4 register) | Patch Token 在序列中的起始位置 |
| `self.rope` | `RotaryPositionEmbedding2D(freq=100)` | 2D 旋转位置编码 |
| `self.depth` | `24` | 总层数 |
| `self.aa_order` | `["frame", "global"]` | 先帧内后帧间 |
| `self.aa_block_size` | `1` | 每次切换处理 1 层 |
| `self.cached_layer_indices` | `{4, 11, 17, 23}` | 缓存的层索引 (+ depth-1=23 自动加入) |

---

### 2.3 Patch Embedding 层

VGGT 使用 **DINOv2 ViT-L/14 with registers** 作为 Patch Embedding：

```python
# aggregator.py 中的构建方式
self.patch_embed = vit_large(
    img_size=518,
    patch_size=14,
    num_register_tokens=4,  # DINOv2 的 register tokens (与 Aggregator 的 register 不同)
    interpolate_antialias=True,
    interpolate_offset=0.0,
    block_chunks=0,
    init_values=1.0,
)
```

**DINOv2 ViT-L/14 参数**:
- embed_dim = 1024
- depth = 24 层
- num_heads = 16
- patch_size = 14
- 输入 518×518 → 37×37 = **1369 patches**

**Patch Embed 输出处理**:
```python
# forward() 中
images = (images - self._resnet_mean) / self._resnet_std  # ImageNet 归一化
images = images.view(B * S, C, H, W)                       # flatten batch & seq
patch_tokens = self.patch_embed(images)                    # DINOv2 前馈

# DINOv2 返回 dict 时取 x_norm_patchtokens
if isinstance(patch_tokens, dict):
    patch_tokens = patch_tokens["x_norm_patchtokens"]
# patch_tokens shape: [B*S, 1369, 1024]
```

> **注意**: DINOv2 backbone 内部有自己的 CLS token 和 register tokens，但 Aggregator 只取 `x_norm_patchtokens`（经过 LayerNorm 后的 patch tokens），不使用 DINOv2 的 CLS/register。Aggregator 有自己独立的 camera_token 和 register_token。

---

### 2.4 特殊 Token 设计

#### Camera Token

```python
# 定义: shape (1, 2, 1, 1024)
# 第1维=1: 广播到 batch
# 第2维=2: 两种变体 (variant 0 = 首帧, variant 1 = 非首帧)
# 第3维=1: 每帧 1 个 camera token
# 第4维=1024: embed_dim
self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
nn.init.normal_(self.camera_token, std=1e-6)  # 极小值初始化
```

#### Register Token

```python
# 定义: shape (1, 2, 4, 1024)
# 第2维=2: 两种变体 (首帧/非首帧)
# 第3维=4: 每帧 4 个 register token
self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
nn.init.normal_(self.register_token, std=1e-6)
```

#### `slice_expand_and_flatten` 函数

这个关键函数将 `(1, 2, X, C)` 形状的 token 分配给多帧序列：

```python
def slice_expand_and_flatten(token_tensor, B, S):
    """
    输入: token_tensor shape (1, 2, X, C)  -- X=1 for camera, X=4 for register
    处理:
      1) variant 0 (index=0) → 分配给首帧 (frame 0)
      2) variant 1 (index=1) → 分配给其余 S-1 帧
      3) expand 到 batch size B
      4) 拼接 → (B, S, X, C)
      5) flatten → (B*S, X, C)
    """
    # query: 首帧 token → shape (B, 1, X, C)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # others: 非首帧 token → shape (B, S-1, X, C)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # 拼接 → (B, S, X, C)
    combined = torch.cat([query, others], dim=1)
    # flatten → (B*S, X, C)
    return combined.view(B * S, *combined.shape[2:])
```

**设计意图**: 首帧作为 "query frame"（参考帧），其 camera token 和 register token 学到的是"锚定"语义；后续帧的 token 学到的是"相对变化"语义。

#### Token 拼接顺序

```python
# 在每帧中，token 的排列顺序是:
# [camera_token(1个), register_token(4个), patch_tokens(1369个)]
tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
# 最终 tokens shape: [B*S, 1374, 1024]  (1 + 4 + 1369 = 1374)
```

`self.patch_start_idx = 5` 表示从第 5 个位置开始才是真正的 patch token。

---

### 2.5 交替注意力机制 (Alternating Attention)

这是 VGGT 的核心创新。Aggregator 有 **两套独立的 Block**:
- `self.frame_blocks[0..23]`: 帧内注意力 (Frame Attention)
- `self.global_blocks[0..23]`: 帧间注意力 (Global Attention)

每个 "层" 由一次 frame attention + 一次 global attention 组成，交替执行 24 次。

#### 执行流程

```python
def forward(self, images):
    B, S, C, H, W = images.shape
    # ... (patch embed, token 拼接) ...
    
    # tokens shape: [B*S, 1374, 1024]
    frame_idx = 0
    global_idx = 0
    output_list = []
    
    for _ in range(self.aa_block_num):  # aa_block_num = depth // aa_block_size = 24
        for attn_type in self.aa_order:  # ["frame", "global"]
            if attn_type == "frame":
                tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                    tokens, B, S, P, C, frame_idx, pos=pos)
            elif attn_type == "global":
                tokens, global_idx, global_intermediates = self._process_global_attention(
                    tokens, B, S, P, C, global_idx, pos=pos)
        
        # 缓存: 拼接 frame 和 global 的中间结果
        for i in range(len(frame_intermediates)):
            layer_idx = len(output_list)
            if layer_idx in self.cached_layer_indices:
                # 拼接: [B, S, P, 2C=2048]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
            else:
                output_list.append(None)
    
    return output_list, self.patch_start_idx
```

#### Frame Attention (帧内)

```python
def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
    """
    帧内注意力: 每帧独立进行 self-attention
    tokens 保持 shape (B*S, P, C) — 每帧独立
    """
    # 确保 shape 正确
    if tokens.shape != (B * S, P, C):
        tokens = tokens.view(B, S, P, C).view(B * S, P, C)
    if pos is not None and pos.shape != (B * S, P, 2):
        pos = pos.view(B, S, P, 2).view(B * S, P, 2)
    
    intermediates = []
    for _ in range(self.aa_block_size):  # 默认=1
        if self.training:
            tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, ...)
        else:
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
        frame_idx += 1
        intermediates.append(tokens.view(B, S, P, C))
    
    return tokens, frame_idx, intermediates
```

**关键点**: 
- 输入 shape `(B*S, 1374, 1024)` — 将所有帧作为独立样本放在 batch 维度
- 每帧内所有 token (camera + register + patches) 互相做 full attention
- camera token 可以看到同帧的所有 patch token
- **不同帧之间在此步完全隔离**

#### Global Attention (帧间)

```python
def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
    """
    帧间注意力: 所有帧的所有 token 一起做 self-attention
    tokens reshape 为 (B, S*P, C) — 所有帧的 token 拉平
    """
    if tokens.shape != (B, S * P, C):
        tokens = tokens.view(B, S, P, C).view(B, S * P, C)
    if pos is not None and pos.shape != (B, S * P, 2):
        pos = pos.view(B, S, P, 2).view(B, S * P, 2)
    
    intermediates = []
    for _ in range(self.aa_block_size):
        if self.training:
            tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, ...)
        else:
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
        global_idx += 1
        intermediates.append(tokens.view(B, S, P, C))
    
    return tokens, global_idx, intermediates
```

**关键点**:
- 输入 shape `(B, S*1374, 1024)` — 所有帧的 token 拼成一个长序列
- **所有帧的所有 token 互相可见** (full attention, 无 mask)
- 第1帧的 camera token 可以看到第2帧的 patches, 反之亦然
- **这是 camera token 学习跨帧几何关系的核心机制**
- 注意力长度: S × 1374 tokens (如 S=10 → 13740 tokens 的 full attention!)

#### 交替模式示意

```
Layer 0:
  [Frame Block 0]: (B*S, 1374, 1024) — 帧内 self-attention
  [Global Block 0]: (B, S*1374, 1024) — 全局 self-attention
  → 缓存 concat: [B, S, 1374, 2048] (如果 layer 0 不在 cached_layer_indices 中则为 None)

Layer 1:
  [Frame Block 1]: 帧内
  [Global Block 1]: 帧间
  → ...

...

Layer 23:
  [Frame Block 23]: 帧内
  [Global Block 23]: 帧间
  → 缓存 concat: [B, S, 1374, 2048] ← Camera Head 使用的最终层
```

---

### 2.6 Block 结构

**文件**: `/group/40009/dazhaodu/vggt/vggt/layers/block.py`

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, proj_bias=True,
                 ffn_bias=True, drop=0.0, attn_drop=0.0, init_values=None,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_class=Attention, ffn_layer=Mlp, qk_norm=False,
                 fused_attn=True, rope=None):
        
        self.norm1 = LayerNorm(dim)                    # Pre-LN
        self.attn = Attention(dim, num_heads, ...)     # Self-Attention
        self.ls1 = LayerScale(dim, init_values=0.01)   # LayerScale
        self.drop_path1 = DropPath(drop_path)
        
        self.norm2 = LayerNorm(dim)                    # Pre-LN
        self.mlp = Mlp(dim, hidden=dim*4, act=GELU)   # FFN
        self.ls2 = LayerScale(dim, init_values=0.01)
        self.drop_path2 = DropPath(drop_path)
    
    def forward(self, x, pos=None):
        # Pre-Norm + Attention + LayerScale + Residual
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), pos=pos)))
        # Pre-Norm + MLP + LayerScale + Residual
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
```

**Block 内的参数** (以 dim=1024, mlp_ratio=4 为例):
- `norm1`: LayerNorm(1024), 参数: 2048 (weight + bias)
- `attn.qkv`: Linear(1024 → 3072), 参数: 1024×3072 + 3072 = 3,148,800
- `attn.q_norm`: LayerNorm(64), 参数: 128 (head_dim=64)
- `attn.k_norm`: LayerNorm(64), 参数: 128
- `attn.proj`: Linear(1024 → 1024), 参数: 1024×1024 + 1024 = 1,049,600
- `ls1`: LayerScale(1024), 参数: 1024
- `norm2`: LayerNorm(1024), 参数: 2048
- `mlp.fc1`: Linear(1024 → 4096), 参数: 4,198,400
- `mlp.fc2`: Linear(4096 → 1024), 参数: 4,195,328
- `ls2`: LayerScale(1024), 参数: 1024
- **每个 Block 总参数**: ~12.6M

**Aggregator 有 24 个 frame blocks + 24 个 global blocks = 48 个 Block**: ~605M

---

### 2.7 Attention 模块

**文件**: `/group/40009/dazhaodu/vggt/vggt/layers/attention.py`

```python
class Attention(nn.Module):
    def __init__(self, dim=1024, num_heads=16, qkv_bias=True, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0, qk_norm=True, fused_attn=True, rope=None):
        self.num_heads = 16
        self.head_dim = 64          # 1024 / 16
        self.scale = 64 ** -0.5     # 0.125
        
        self.qkv = nn.Linear(1024, 3072, bias=True)   # 融合 QKV
        self.q_norm = LayerNorm(64)  # QK Norm (per-head)
        self.k_norm = LayerNorm(64)
        self.proj = nn.Linear(1024, 1024, bias=True)
        self.rope = rope  # RotaryPositionEmbedding2D 实例
    
    def forward(self, x, pos=None):
        B, N, C = x.shape  # e.g., (B*S, 1374, 1024)
        
        # 1. 计算 QKV
        qkv = self.qkv(x)  # (B, N, 3072)
        qkv = qkv.reshape(B, N, 3, 16, 64).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 每个: (B, 16, N, 64)
        
        # 2. QK Normalization
        q, k = self.q_norm(q), self.k_norm(k)
        
        # 3. 应用 RoPE
        if self.rope is not None:
            q = self.rope(q, pos)  # pos: (B, N, 2) — 2D坐标
            k = self.rope(k, pos)
        
        # 4. Scaled Dot-Product Attention (使用 F.scaled_dot_product_attention)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        # output: (B, 16, N, 64)
        
        # 5. 重塑并投影
        x = x.transpose(1, 2).reshape(B, N, 1024)
        x = self.proj(x)
        return x
```

**重要细节**:
- **融合 QKV**: 用一个 `Linear(1024, 3072)` 同时计算 Q, K, V
- **QK Norm**: 对每个 head 的 Q, K 分别做 LayerNorm (head_dim=64)，稳定训练
- **RoPE**: 只对 Q 和 K 施加旋转位置编码，V 不受影响
- **Special Token 的 RoPE**: 通过将 pos 设为 0 来禁用位置编码（见下文）

---

### 2.8 2D Rotary Position Embedding

**文件**: `/group/40009/dazhaodu/vggt/vggt/layers/rope.py`

#### PositionGetter

```python
class PositionGetter:
    """生成 2D 空间坐标网格，为每个 patch 分配 (y, x) 坐标"""
    
    def __call__(self, batch_size, height, width, device):
        """
        返回: (batch_size, height*width, 2)
        例: 37×37 grid → (B*S, 1369, 2)
        坐标值: y ∈ [0, 36], x ∈ [0, 36] (整数坐标)
        """
        y_coords = torch.arange(height, device=device)  # [0, 1, ..., 36]
        x_coords = torch.arange(width, device=device)   # [0, 1, ..., 36]
        positions = torch.cartesian_prod(y_coords, x_coords)  # (1369, 2)
        return positions.view(1, H*W, 2).expand(batch_size, -1, -1)
```

#### Special Token 位置处理

```python
# 在 Aggregator.forward() 中:
pos = self.position_getter(B * S, H // patch_size, W // patch_size, device)
# pos shape: (B*S, 1369, 2), 值: 整数坐标 [0..36]

# 关键: 给 special tokens 设置位置 0 (禁用 RoPE)
pos = pos + 1                    # patch 坐标变成 [1..37]
pos_special = torch.zeros(B * S, self.patch_start_idx, 2)  # (B*S, 5, 2) 全零
pos = torch.cat([pos_special, pos], dim=1)  # (B*S, 1374, 2)
# 最终: special tokens 位置=0, patch tokens 位置=[1..37]
```

**设计意图**: Camera token 和 Register token 没有空间位置含义，设为 0 相当于让 RoPE 对它们不做旋转（cos(0)=1, sin(0)=0 → 不改变原值）。

#### RotaryPositionEmbedding2D

```python
class RotaryPositionEmbedding2D(nn.Module):
    """
    2D RoPE: 将 head_dim=64 分成两半:
      - 前 32 维: 编码 Y 方向位置
      - 后 32 维: 编码 X 方向位置
    """
    def __init__(self, frequency=100.0, scaling_factor=1.0):
        self.base_frequency = 100.0
    
    def forward(self, tokens, positions):
        """
        Args:
            tokens: (B, num_heads, N, head_dim=64)
            positions: (B, N, 2)  -- (y, x) 坐标
        Returns:
            tokens with RoPE applied: same shape
        """
        feature_dim = tokens.size(-1) // 2  # 32
        
        # 计算频率: inv_freq = 1 / (100 ^ (2i/32)), i = 0,1,...,15
        # 对应周期从 1 到 100^1 ≈ 100
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, device, dtype)
        
        # 分割为垂直和水平
        vertical, horizontal = tokens.chunk(2, dim=-1)  # 各 (B, heads, N, 32)
        
        # 分别应用 1D RoPE
        vertical = self._apply_1d_rope(vertical, positions[..., 0], cos_comp, sin_comp)
        horizontal = self._apply_1d_rope(horizontal, positions[..., 1], cos_comp, sin_comp)
        
        return torch.cat((vertical, horizontal), dim=-1)  # (B, heads, N, 64)
    
    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        """
        dim=32: 计算 16 个频率分量
        exponents = [0, 2/32, 4/32, ..., 30/32] = [0, 0.0625, ..., 0.9375]
        inv_freq = 1 / (100^exponents) = [1.0, 0.708, ..., 0.0126]
        
        angles[pos] = pos * inv_freq → 每个位置的相位角
        然后 angles = cat(angles, angles) → (seq_len, 32)
        """
        exponents = torch.arange(0, dim, 2).float() / dim  # (16,)
        inv_freq = 1.0 / (100.0 ** exponents)              # (16,)
        positions = torch.arange(seq_len)                    # (seq_len,)
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # (seq_len, 16)
        angles = torch.cat((angles, angles), dim=-1)          # (seq_len, 32)
        return angles.cos(), angles.sin()
    
    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        """
        经典 RoPE: x' = x * cos(θ) + rotate_half(x) * sin(θ)
        rotate_half([x1, x2]) = [-x2, x1]
        """
        cos = F.embedding(positions, cos_comp)[:, None, :, :]  # (B, 1, N, 32)
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)
```

**频率设计 (base=100)**:
- 16 个频率分量: `inv_freq[i] = 1/100^(2i/32)`, i=0..15
- 最高频率 (i=0): 周期=2π (相邻 patch 差异最大)
- 最低频率 (i=15): 周期=2π×100^(30/32)≈500+ (全局位置信息)
- 对 37×37 的 grid，覆盖充分

---

### 2.9 中间层缓存机制

```python
# cached_layer_indices = {4, 11, 17, 23}
# 每层的 frame 和 global 中间结果拼接后缓存

# 缓存格式: List[Tensor | None], 长度=24
# 非 None 条目 shape: [B, S, P, 2*C] = [B, S, 1374, 2048]
# 拼接方式: torch.cat([frame_output, global_output], dim=-1)

# 最终层 (index=23) 一定被缓存
# Camera Head 使用 aggregated_tokens_list[-1] (index=23)
# DPT Head (depth, point) 使用多层特征 (index 4, 11, 17, 23)
```

**缓存内容的含义**:
- `frame_output[layer_i]`: 经过第 i 个 frame block 后的 token 表示 (帧内信息)
- `global_output[layer_i]`: 经过第 i 个 global block 后的 token 表示 (帧间信息)
- 拼接为 2048 维 → 同时包含帧内和帧间信息

---

### 2.10 Camera Head

**文件**: `/group/40009/dazhaodu/vggt/vggt/heads/camera_head.py`

Camera Head 使用 **4 次 AdaLN 迭代细化** 来预测相机参数。

```python
class CameraHead(nn.Module):
    def __init__(self, dim_in=2048, trunk_depth=4, num_heads=16, mlp_ratio=4, init_values=0.01):
        self.target_dim = 9  # Translation(3) + Quaternion(4) + FoV(2)
        
        # 4 个 Transformer Block 组成的 trunk
        self.trunk = nn.Sequential(*[
            Block(dim=2048, num_heads=16, mlp_ratio=4, init_values=0.01)
            for _ in range(4)
        ])
        
        self.token_norm = LayerNorm(2048)           # 输入 camera token 归一化
        self.trunk_norm = LayerNorm(2048)           # trunk 输出归一化
        self.empty_pose_tokens = Parameter(zeros(1, 1, 9))  # 初始空 pose
        self.embed_pose = Linear(9, 2048)           # pose → token 空间
        
        # AdaLN 调制: SiLU + Linear → 3 × 2048 (shift, scale, gate)
        self.poseLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2048, 3 * 2048)
        )
        self.adaln_norm = LayerNorm(2048, elementwise_affine=False)
        self.pose_branch = Mlp(in=2048, hidden=1024, out=9)  # 最终 pose 预测
```

#### 前馈流程

```python
def forward(self, aggregated_tokens_list, num_iterations=4):
    # 取最后一层的缓存 → [B, S, 1374, 2048]
    tokens = aggregated_tokens_list[-1]
    
    # 提取 camera token (index=0 of each frame)
    pose_tokens = tokens[:, :, 0]      # [B, S, 2048]
    pose_tokens = self.token_norm(pose_tokens)  # LayerNorm
    
    return self.trunk_fn(pose_tokens, num_iterations=4)

def trunk_fn(self, pose_tokens, num_iterations):
    """
    4 次迭代细化:
    iteration 0: 从 empty_pose_tokens 开始
    iteration 1-3: 从上次预测结果 (detached) 开始
    每次迭代: AdaLN调制 → trunk(4层Block) → pose_branch → 累加delta
    """
    B, S, C = pose_tokens.shape  # (B, S, 2048)
    pred_pose_enc = None
    pred_pose_enc_list = []
    
    for _ in range(4):
        # === Step 1: 生成 module_input ===
        if pred_pose_enc is None:
            # 第一次迭代: 使用 learnable empty pose
            module_input = self.embed_pose(
                self.empty_pose_tokens.expand(B, S, -1))  # (B, S, 2048)
        else:
            # 后续迭代: embed 上次预测 (detached, 不传梯度)
            module_input = self.embed_pose(pred_pose_enc.detach())
        
        # === Step 2: AdaLN 调制 ===
        # 从 module_input 计算调制参数
        shift, scale, gate = self.poseLN_modulation(module_input).chunk(3, dim=-1)
        # 每个: (B, S, 2048)
        
        # 对 pose_tokens 做 adaptive norm + modulation
        # modulate(x, shift, scale) = x * (1 + scale) + shift
        pose_tokens_modulated = gate * modulate(
            self.adaln_norm(pose_tokens), shift, scale)
        pose_tokens_modulated = pose_tokens_modulated + pose_tokens  # 残差
        
        # === Step 3: Trunk (4层 Block) ===
        pose_tokens_modulated = self.trunk(pose_tokens_modulated)
        # 输入输出都是 (B, S, 2048)
        # trunk 内的 attention: S 个帧的 camera token 互相 attend
        
        # === Step 4: 预测 delta pose ===
        pred_pose_enc_delta = self.pose_branch(
            self.trunk_norm(pose_tokens_modulated))  # (B, S, 9)
        
        # === Step 5: 累加 ===
        if pred_pose_enc is None:
            pred_pose_enc = pred_pose_enc_delta
        else:
            pred_pose_enc = pred_pose_enc + pred_pose_enc_delta
        
        # === Step 6: 激活函数 ===
        activated_pose = activate_pose(pred_pose_enc,
            trans_act="linear",   # Translation: 不变
            quat_act="linear",    # Quaternion: 不变
            fl_act="relu")        # FoV: ReLU 保正
        pred_pose_enc_list.append(activated_pose)
    
    return pred_pose_enc_list  # 4 个迭代的结果
```

#### `activate_pose` 函数

```python
def activate_pose(pred_pose_enc, trans_act, quat_act, fl_act):
    T = pred_pose_enc[..., :3]    # Translation
    quat = pred_pose_enc[..., 3:7]  # Quaternion rotation
    fl = pred_pose_enc[..., 7:]     # Field of View (2D)
    
    T = base_pose_act(T, "linear")     # 不变
    quat = base_pose_act(quat, "linear") # 不变
    fl = base_pose_act(fl, "relu")       # ReLU 保正
    
    return torch.cat([T, quat, fl], dim=-1)  # (B, S, 9)
```

**Camera Head 输出**: `[B, S, 9]`
- `[..., 0:3]`: 绝对 Translation (3D)
- `[..., 3:7]`: Quaternion Rotation (4D, wxyz 或 xyzw)
- `[..., 7:9]`: Field of View (2D, 水平+垂直 FoV)

---

## 3. VGGT-Omega 完整架构

### 3.1 顶层模型结构

**文件**: `/group/40009/dazhaodu/vggt-omega/vggt_omega/models/vggt_omega.py`

```python
class VGGTOmega(nn.Module):
    def __init__(self, patch_size=16, embed_dim=1024,
                 enable_camera=True, enable_depth=True, enable_alignment=False):
        self.aggregator = Aggregator(patch_size=16, embed_dim=1024)
        self.camera_head = CameraHead(dim_in=2048)
        self.dense_head = DenseHead(dim_in=2048, patch_size=16)  # 深度估计
        self.text_alignment_head = TextAlignmentHead(dim_in=2048)  # 文本对齐 (新增)
```

**前馈流程**:
```python
def forward(self, images):
    # images: [B, S, 3, H, W] 或 [S, 3, H, W]
    if len(images.shape) == 4:
        images = images.unsqueeze(0)
    
    # 自动混合精度 (BF16/FP16)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        aggregated_tokens_list, patch_token_start = self.aggregator(images)
    
    final_tokens = aggregated_tokens_list[-1]  # [B, S, P, 2048]
    
    predictions = {
        # 额外输出: camera + register tokens (前 patch_token_start 个)
        "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
    }
    
    # Camera Head (FP32)
    with torch.autocast(device_type="cuda", enabled=False):
        predictions["pose_enc"] = self.camera_head(
            aggregated_tokens_list, patch_token_start=patch_token_start)
    
    return predictions
```

**关键差异 vs VGGT**:
1. 使用 `torch.autocast` 自动混合精度
2. Camera Head 接收 `patch_token_start` 参数（因为它需要取 camera + register tokens）
3. 额外输出 `camera_and_register_tokens` — 方便下游使用
4. 有 `_warn_if_rope_not_max` 检查 RoPE 归一化模式

---

### 3.2 Aggregator 核心架构

**文件**: `/group/40009/dazhaodu/vggt-omega/vggt_omega/models/aggregator.py`

```python
class Aggregator(nn.Module):
    def __init__(self,
        patch_size=16, embed_dim=1024,
        depth=24, num_heads=16, mlp_ratio=4.0,
        num_register_tokens=16,  # ← 16 个 (VGGT 是 4 个)
        register_attention_block_indices=[2, 6, 9, 14, 20],  # ← 新: 5 层用 register-only
        cached_layer_indices=(4, 11, 17, 23),
    ):
```

**关键属性对比**:

| 属性 | VGGT | VGGT-Omega |
|------|------|------------|
| `patch_size` | 14 | 16 |
| `num_register_tokens` | 4 | **16** |
| `camera_token` shape | `(1,2,1,1024)` | `(1,2,1,1024)` — 相同 |
| `register_token` shape | `(1,2,4,1024)` | `(1,2,16,1024)` |
| `patch_token_start` | 5 (1+4) | **17** (1+16) |
| 帧间注意力 | 统一 global | 混合 global + register-only |
| init_values | 0.01 | **1e-5** |
| Block 类型 | `Block` | `SelfAttentionBlock` |
| mask_k_bias | 无 | **有** (LinearKMaskedBias) |

#### Token 排列

```
每帧 token 序列: [camera(1), register(16), patches(1024)] = 1041 tokens
patch_token_start = 17
总 tokens per frame: 1 + 16 + 1024 = 1041
```

#### `register_attention_block_indices`

```python
# 控制每一层帧间注意力的类型
self.inter_frame_attention_types = ["global"] * 24  # 默认全 global
for idx in [2, 6, 9, 14, 20]:
    self.inter_frame_attention_types[idx] = "register"
# 结果: 5层 register-only + 19层 global
```

---

### 3.3 Patch Embedding 层 (DINOv3 ViT-L/16)

```python
def _build_patch_embed(patch_size=16, embed_dim=1024):
    return DinoVisionTransformer(
        img_size=224,           # 标称值，实际支持可变分辨率
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",  # ← 关键: "max" 归一化
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1e-5,
        norm_layer="layernormbf16",  # LayerNorm with eps=1e-5
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,          # DINOv3 的 storage tokens
        mask_k_bias=True,            # ← K bias 掩码
    )
```

**DINOv3 ViT-L/16 内部结构**:
- embed_dim = 1024
- depth = 24
- num_heads = 16, head_dim = 64
- patch_size = 16
- 内部有 `cls_token` + `storage_tokens(4)` + `mask_token`
- 内置 `RopePositionEmbedding` (与 Aggregator 相同频率)
- 使用 `SelfAttentionBlock`（和 Aggregator 的 block 相同类型）

**输入处理**:
```python
# DinoVisionTransformer.prepare_tokens_with_masks:
x = self.patch_embed(x)  # Conv2D → (B, H, W, embed_dim)
B, H, W, _ = x.shape
x = x.flatten(1, 2)      # → (B, H*W, embed_dim)

# 拼接: [cls_token, storage_tokens(4), patch_tokens]
x = torch.cat([cls_token(B), storage_tokens(B), x], dim=1)
```

**输出**: `{"x_norm_patchtokens": ...}`
- 经过 24 层 Block + LayerNorm 后的 patch tokens
- shape: `(B*S, H/16 * W/16, 1024)`
- 对于 512×512 输入: `(B*S, 1024, 1024)` — 32×32=1024 patches

---

### 3.4 特殊 Token 设计

```python
# Camera Token: 与 VGGT 相同设计
self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
# shape: (1, 2, 1, 1024) — 2 variants (首帧/非首帧)

# Register Token: 16 个 (VGGT 只有 4 个)
self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
# shape: (1, 2, 16, 1024)

# 初始化: std=1e-3 (VGGT 是 std=1e-6, Omega 稍大)
nn.init.normal_(self.camera_token, std=1e-3)
nn.init.normal_(self.register_token, std=1e-3)
```

**`slice_expand_and_flatten`**: 与 VGGT 功能完全相同

```python
def slice_expand_and_flatten(token_tensor, batch_size, num_frames):
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, ...)
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames-1, ...)
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
```

---

### 3.5 混合帧间注意力机制

这是 VGGT-Omega 最大的架构创新。

#### Forward 主循环

```python
def forward(self, images):
    B, S, C, H, W = images.shape
    # ... (normalize, patch embed, token concat) ...
    # tokens shape: [B*S, 1041, 1024]
    
    # 计算 RoPE (一次性)
    patch_grid_size = (H // 16, W // 16)  # e.g., (32, 32)
    rope_sin, rope_cos = self.rope_embed(H=32, W=32)  # [1024, D_head]
    frame_rope = (rope_sin, rope_cos)  # FP32
    
    outputs = []
    for block_idx in range(24):
        # === 步骤 1: Frame Block (帧内) ===
        tokens, frame_tokens = self._run_frame_block(
            tokens, B, S, num_tokens, embed_dim, block_idx, frame_rope)
        
        # === 步骤 2: Inter-Frame Block ===
        # 根据 self.inter_frame_attention_types[block_idx] 选择方式
        tokens = self._run_inter_frame_attention_block(
            tokens, B, S, num_tokens, embed_dim, block_idx,
            self.inter_frame_attention_types[block_idx])  # "global" or "register"
        
        # === 步骤 3: 缓存 ===
        if block_idx in self.cached_layer_indices:
            # 拼接 frame_tokens 和当前 tokens → [B, S, P, 2048]
            outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
        else:
            outputs.append(None)
    
    return outputs, self.patch_token_start  # 17
```

#### Frame Block (帧内注意力)

```python
def _run_frame_block(self, tokens, B, S, num_tokens, embed_dim, block_idx, rope_sincos):
    tokens = tokens.view(B * S, num_tokens, embed_dim)  # (B*S, 1041, 1024)
    tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
    return tokens, tokens.view(B, S, num_tokens, embed_dim)
```

**注意**: RoPE 只应用于 patch tokens (因为 rope_sincos 的长度 = H*W/patch^2 = 1024)。特殊 token (前 17 个) 作为 "prefix" 不被 RoPE 修改 (在 attention.py 的 `apply_rope` 中有 prefix 跳过逻辑)。

#### Global Inter-Frame Block

```python
def _run_inter_frame_attention_block(self, tokens, B, S, ..., attention_type):
    tokens = tokens.view(B, S, num_tokens, embed_dim)
    
    if attention_type == "global":
        # 所有帧的所有 token 做全局 attention
        tokens = tokens.view(B, S * num_tokens, embed_dim)  # (B, S*1041, 1024)
        tokens = self.inter_frame_blocks[block_idx](tokens, None)  # 无 RoPE
        return tokens.view(B, S, num_tokens, embed_dim)
```

与 VGGT 的 Global Attention 相同: 所有帧所有 token 互相可见，**无 RoPE**。

#### Register-Only Inter-Frame Block (新)

```python
    if attention_type == "register":
        # 只让 camera + register tokens 跨帧交互
        # patch tokens 冻结（不参与帧间 attention）
        
        patch_token_start = self.patch_token_start  # 17
        
        # 分离: camera+register vs patch
        camera_and_register = tokens[:, :, :17].reshape(B, S*17, embed_dim)
        # shape: (B, S*17, 1024)
        
        patch_tokens = tokens[:, :, 17:].reshape(B, S*(1041-17), embed_dim)
        # shape: (B, S*1024, 1024) — 不参与这一层的注意力!
        
        # 只对 camera+register 做跨帧 attention
        camera_and_register = self.inter_frame_blocks[block_idx](
            camera_and_register, None)  # 无 RoPE
        
        # 重新拼接
        tokens = torch.cat([camera_and_register, patch_tokens], dim=1)
        # reshape 回 (B, S, 1041, 1024)
        camera_and_register = tokens[:, :S*17].view(B, S, 17, embed_dim)
        patch_tokens = tokens[:, S*17:].view(B, S, 1041-17, embed_dim)
        return torch.cat([camera_and_register, patch_tokens], dim=2)
```

**Register-Only 注意力的设计意图**:
1. **降低计算复杂度**: Global attention 对 S×1041 tokens 做 O(n²) attention 非常昂贵。Register-only 只对 S×17 做 attention → 计算量降为 `(S*17)² / (S*1041)² ≈ 0.026%`
2. **Camera Token 信息瓶颈**: 让 camera token 被迫通过有限的 register tokens 来传递/接收跨帧信息，形成信息瓶颈，促进抽象几何特征的学习
3. **Patch Tokens 不参与**: 在这些层中，patch tokens 不被跨帧注意力更新，保持帧内局部特征

**5 层 register-only 位于**: 第 2, 6, 9, 14, 20 层 (共 24 层中)
- 早期层 (2, 6): 低层特征交互
- 中期层 (9, 14): 中间特征交互
- 后期层 (20): 高层特征交互
- 其余 19 层: 完整 global attention

---

### 3.6 SelfAttentionBlock

**文件**: `/group/40009/dazhaodu/vggt-omega/vggt_omega/models/layers/block.py`

```python
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=1024, num_heads=16, ffn_ratio=4.0,
                 qkv_bias=True, proj_bias=True, ffn_bias=True,
                 init_values=1e-5,   # ← 极小 LayerScale
                 use_qk_norm=True,   # ← QK Norm
                 mask_k_bias=True):  # ← K Bias Mask
        
        self.norm1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, proj_bias,
                                  mask_k_bias=True, use_qk_norm=True)
        self.ls1 = LayerScale(dim, init_values=1e-5)
        
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(in=1024, hidden=4096, act=GELU, bias=True)
        self.ls2 = LayerScale(dim, init_values=1e-5)
    
    def forward(self, x_or_x_list, rope_or_rope_list=None):
        # 统一使用 _forward_list 实现
        if isinstance(x_or_x_list, Tensor):
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        else:
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
    
    def _forward(self, x, rope=None):
        """单 Tensor 的标准前馈"""
        x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
        return x_ffn
```

**与 VGGT Block 的差异**:
1. **`mask_k_bias=True`**: 使用 `LinearKMaskedBias` 代替普通 `nn.Linear`
2. **`use_qk_norm=True`**: QK Norm 通过参数控制
3. **`_forward_list`**: 支持 batch 列表输入（用于 DINOv3 多尺度训练）
4. **LayerScale init=1e-5**: 比 VGGT 的 0.01 小 500 倍，更保守的初始化

---

### 3.7 SelfAttention 模块

**文件**: `/group/40009/dazhaodu/vggt-omega/vggt_omega/models/layers/attention.py`

```python
class SelfAttention(nn.Module):
    def __init__(self, dim=1024, num_heads=16, qkv_bias=True, proj_bias=True,
                 mask_k_bias=False, use_qk_norm=True):
        self.num_heads = 16
        head_dim = 64
        self.scale = 64 ** -0.5
        
        # QK Normalization
        self.use_qk_norm = True
        self.q_norm = LayerNorm(64, eps=1e-5)
        self.k_norm = LayerNorm(64, eps=1e-5)
        
        # 融合 QKV (可能用 LinearKMaskedBias)
        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(1024, 3072, bias=True)
        
        self.proj = nn.Linear(1024, 1024, bias=True)
```

#### LinearKMaskedBias

```python
class LinearKMaskedBias(nn.Linear):
    """对 K 部分的 bias 进行掩码（置零）"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features  # 3072
        # bias_mask: 对 Q部分=1, K部分=0, V部分=1
        # 即: [1,1,...,1, 0,0,...,0, 1,1,...,1] (每段1024)
        # 初始化为 nan，在 init_weights_vit 中设置:
        #   mask[o//3 : 2*o//3] = 0  (K 段)
        #   其余 = 1
        self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))
    
    def forward(self, input):
        masked_bias = self.bias * self.bias_mask  # K 的 bias 被清零
        return F.linear(input, self.weight, masked_bias)
```

**设计意图**: 在 `init_weights_vit` 中:
```python
module.bias_mask.fill_(1)                    # 全1
module.bias_mask[o // 3 : 2 * o // 3].fill_(0)  # K 段清零
```
结果: **Q 和 V 有 bias, K 没有 bias**。这是 DINOv3 的设计选择，避免 K bias 对 softmax 的偏移效应。

#### RoPE 应用 (带 prefix 跳过)

```python
def apply_rope(self, q, k, rope):
    """
    关键: 只对 patch tokens 应用 RoPE，跳过 special tokens (prefix)
    """
    sin, cos = rope
    rope_dtype = sin.dtype
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)
    
    N = q.shape[-2]           # 总 token 数, e.g., 1041
    prefix = N - sin.shape[-2]  # special tokens 数, e.g., 1041 - 1024 = 17
    assert prefix >= 0
    
    # 分离 prefix (special tokens) 和 patch tokens
    q_prefix = q[:, :, :prefix, :]        # (B, heads, 17, 64) — 不旋转
    q_patch = q[:, :, prefix:, :]         # (B, heads, 1024, 64)
    q_patch = rope_apply(q_patch, sin, cos)  # 应用 RoPE
    q = torch.cat((q_prefix, q_patch), dim=-2)  # 重新拼接
    
    # K 同理
    k_prefix = k[:, :, :prefix, :]
    k_patch = k[:, :, prefix:, :]
    k_patch = rope_apply(k_patch, sin, cos)
    k = torch.cat((k_prefix, k_patch), dim=-2)
    
    return q.to(q_dtype), k.to(k_dtype)
```

**关键设计**: Camera token (位置0) 和 Register tokens (位置1-16) 不被 RoPE 修改，只有 patch tokens (位置17+) 有位置编码。

#### `rope_apply` 函数

```python
def rope_rotate_half(x):
    """[-x2, x1] 形式的旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def rope_apply(x, sin, cos):
    """x' = x * cos + rotate_half(x) * sin"""
    return (x * cos) + (rope_rotate_half(x) * sin)
```

#### `compute_attention` 核心

```python
def compute_attention(self, qkv, attn_bias=None, rope=None):
    B, N, _ = qkv.shape
    C = self.qkv.in_features  # 1024
    
    qkv = qkv.reshape(B, N, 3, 16, 64)
    q, k, v = torch.unbind(qkv, 2)  # 各 (B, N, 16, 64)
    q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  # (B, 16, N, 64)
    
    # QK Norm
    if self.use_qk_norm:
        q = self.q_norm(q)  # LayerNorm per head
        k = self.k_norm(k)
    
    # RoPE
    if rope is not None:
        q, k = self.apply_rope(q, k, rope)
    
    # SDPA
    x = F.scaled_dot_product_attention(q, k, v)  # (B, 16, N, 64)
    x = x.transpose(1, 2).reshape(B, N, C)       # (B, N, 1024)
    return x
```

---

### 3.8 RoPE Position Embedding

**文件**: `/group/40009/dazhaodu/vggt-omega/vggt_omega/models/layers/rope_position_encoding.py`

```python
class RopePositionEmbedding(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, base=100,
                 normalize_coords="max", dtype=torch.float32):
        # D_head = 1024 // 16 = 64
        # periods buffer: shape (D_head // 4,) = (16,)
        # 每个 period 值: base^(2i / (D_head//2)) = 100^(2i/32)
        self.register_buffer("periods", torch.empty(16))
    
    def forward(self, *, H, W):
        """
        返回: (sin, cos), 各 shape [H*W, D_head] = [1024, 64]
        """
        # 坐标归一化 (normalize_coords="max")
        max_HW = max(H, W)  # e.g., 32
        coords_h = torch.arange(0.5, H) / max_HW   # [0.5/32, 1.5/32, ..., 31.5/32]
        coords_w = torch.arange(0.5, W) / max_HW
        coords = meshgrid(coords_h, coords_w)       # (H*W, 2)
        coords = 2 * coords - 1                     # 映射到 [-1, +1]
        
        # 计算角度
        # periods shape: (16,)
        # angles = 2π * coords / periods → (H*W, 2, 16)
        angles = 2 * pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)   # (H*W, 32) — y和x交错
        angles = angles.tile(2)          # (H*W, 64) — 重复一份 (用于 rope_rotate_half)
        
        return (angles.sin(), angles.cos())  # 各 (H*W, 64)
```

**关键差异 vs VGGT RoPE**:

| 特性 | VGGT | VGGT-Omega |
|------|------|------------|
| 坐标范围 | 整数 [0, 36] | 归一化 [-1, +1] |
| 归一化方式 | 无 (原始坐标) | `normalize_coords="max"` (除以 max(H,W)) |
| 频率计算 | `1/base^(2i/dim)` | `2π/periods`, periods=`base^(2i/(D_head//2))` |
| 应用方式 | 分前后半 (vertical/horizontal) | 交错 (y,x 拼接后 tile) |
| 偏移 | `pos+1` 给 patch, 0 给 special | prefix 跳过 (无 RoPE) |
| 支持可变分辨率 | 是 (插值) | 是 (归一化到 [-1,1]) |

**"max" 归一化的优势**: 无论输入图像尺寸如何变化，坐标始终在 [-1, +1] 范围内，保持频率特性一致。

---

### 3.9 Camera Head

**文件**: `/group/40009/dazhaodu/vggt-omega/vggt_omega/models/heads/camera_head.py`

```python
class CameraHead(nn.Module):
    """Camera head used by the released VGGT-Omega checkpoints."""
    
    def __init__(self, dim_in=2048):
        self.token_norm = LayerNorm(2048, eps=1e-5)
        
        # 4 个 SelfAttentionBlock 作为 trunk
        self.trunk = nn.ModuleList([
            SelfAttentionBlock(
                dim=2048, num_heads=16, ffn_ratio=4.0,
                qkv_bias=True, proj_bias=True, ffn_bias=True,
                init_values=1e-5,
                use_qk_norm=False,  # ← Camera Head 内不用 QK Norm
                mask_k_bias=True,
            )
            for _ in range(4)
        ])
        
        self.trunk_norm = LayerNorm(2048, eps=1e-5)
        
        # 最终 pose 预测 MLP
        self.camera_branch = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.GELU(),
            nn.Linear(1024, 9, bias=True),
        )
```

#### 前馈流程 (单次通过, 无迭代)

```python
def forward(self, aggregated_tokens_list, patch_token_start):
    tokens = aggregated_tokens_list[-1]  # [B, S, 1041, 2048]
    B, S, num_tokens, _ = tokens.shape
    
    # 转 FP32
    if tokens.dtype != torch.float32:
        tokens = tokens.float()
    
    # 取 camera + register tokens (前 17 个)
    camera_and_register = tokens[:, :, :patch_token_start]  # [B, S, 17, 2048]
    camera_and_register = self.token_norm(camera_and_register)
    
    # Reshape: 所有帧的 camera+register 做全局 attention
    camera_and_register = camera_and_register.reshape(B, S * 17, 2048)
    # shape: (B, S*17, 2048)
    
    # 通过 4 个 SelfAttentionBlock (无 RoPE)
    for block in self.trunk:
        camera_and_register = block(camera_and_register, None)
    # shape: (B, S*17, 2048)
    
    # 取回每帧的 camera token (index 0)
    camera_and_register = camera_and_register.reshape(B, S, 17, 2048)
    camera_tokens = self.trunk_norm(camera_and_register[:, :, 0])
    # shape: (B, S, 2048)
    
    # 预测 pose
    raw_pose = self.camera_branch(camera_tokens)  # (B, S, 9)
    return _apply_camera_activation(raw_pose)

def _apply_camera_activation(raw_camera):
    translation = raw_camera[..., :3]           # 不变
    quaternion = raw_camera[..., 3:7]           # 不变
    fov = F.relu(raw_camera[..., 7:]) + 0.01   # ReLU + min_val
    return torch.cat([translation, quaternion, fov], dim=-1)
```

**关键差异 vs VGGT Camera Head**:

| 特性 | VGGT Camera Head | VGGT-Omega Camera Head |
|------|-----------------|----------------------|
| **输入** | 只取 camera token: `tokens[:,:,0]` → (B,S,2048) | 取 camera+register: `tokens[:,:,:17]` → (B,S,17,2048) |
| **迭代次数** | 4 次迭代细化 (AdaLN) | **单次通过** (无迭代) |
| **调制方式** | AdaLN (shift/scale/gate) | 无调制 |
| **Trunk 输入** | (B, S, 2048) — 各帧 camera token 做 attention | (B, S*17, 2048) — 所有帧的 camera+register 做 attention |
| **Trunk 结构** | Sequential(Block×4) | ModuleList(SelfAttentionBlock×4) |
| **最终预测** | Mlp(2048→1024→9) + 累加 delta | Sequential(Linear→GELU→Linear)(2048→1024→9) |
| **FoV 激活** | ReLU | ReLU + 0.01 (避免零值) |
| **QK Norm** | Block 内有 QK Norm | 显式 `use_qk_norm=False` |

---

## 4. 关键差异深度对比

### 4.1 注意力模式对比

```
VGGT (每层):
┌─────────────────────┐     ┌─────────────────────────────────┐
│  Frame Attention     │     │  Global Attention                │
│  (B*S, 1374, 1024)  │ ──> │  (B, S*1374, 1024)              │
│  帧内 self-attn      │     │  所有token全局 self-attn         │
└─────────────────────┘     └─────────────────────────────────┘

VGGT-Omega (每层, 以 register 层为例):
┌─────────────────────┐     ┌───────────────────────────────────────┐
│  Frame Attention     │     │  Register-Only Inter-Frame Attention   │
│  (B*S, 1041, 1024)  │ ──> │  只对 (B, S*17, 1024) 做 self-attn    │
│  帧内 self-attn      │     │  Patch tokens 冻结不更新              │
└─────────────────────┘     └───────────────────────────────────────┘

VGGT-Omega (每层, global 层):
┌─────────────────────┐     ┌─────────────────────────────────┐
│  Frame Attention     │     │  Global Inter-Frame Attention    │
│  (B*S, 1041, 1024)  │ ──> │  (B, S*1041, 1024)              │
│  帧内 self-attn      │     │  所有token全局 self-attn         │
└─────────────────────┘     └─────────────────────────────────┘
```

### 4.2 缓存输出对比

两者的缓存格式完全兼容:

```
VGGT:
  output_list[layer_idx] = cat([frame_output, global_output], dim=-1)
  shape: [B, S, 1374, 2048]
  camera token 位置: [:, :, 0, :] → [B, S, 2048]

VGGT-Omega:
  outputs[block_idx] = cat([frame_tokens, inter_frame_tokens], dim=-1)
  shape: [B, S, 1041, 2048]
  camera token 位置: [:, :, 0, :] → [B, S, 2048]
```

### 4.3 计算复杂度对比

假设 S=10 帧, 512×512 输入:

| 操作 | VGGT | VGGT-Omega |
|------|------|------------|
| Frame Attn tokens | 10×1374 = 13740 | 10×1041 = 10410 |
| Global Attn tokens | 13740 全序列 | 19层×10410 + 5层×170 |
| Patch Embed | DINOv2-L/14: 24层×1369 | DINOv3-L/16: 24层×1024 |
| 总 Attention FLOPs (Global) | O(13740²) ×24 层 | O(10410²) ×19 + O(170²) ×5 |

VGGT-Omega 的 register-only 层大幅减少计算 (170² vs 10410² ≈ 3750× 加速)。

### 4.4 RoPE 在帧间注意力中的使用

| | VGGT | VGGT-Omega |
|--|------|------------|
| Frame Attn | 有 RoPE (2D 位置) | 有 RoPE (2D 位置) |
| Global/Inter-Frame Attn | **有 RoPE** (但 pos 被 reshape) | **无 RoPE** (传 None) |

VGGT 在 global attention 中仍然使用 RoPE (将 pos reshape 为 (B, S*P, 2))，这意味着不同帧的相同位置 patch 有相同的位置编码。

VGGT-Omega 在帧间注意力中 **不使用 RoPE**，让模型自己学习跨帧对应关系。

---

## 5. CamDistill 中我们使用的部分

### 5.1 蒸馏目标

我们蒸馏的是 **Aggregator 最终层输出中的 Camera Token**:

```python
# VGGT:
aggregated_tokens_list[-1][:, :, 0, :]  # [B, S, 2048]
# 这是 frame_output 和 global_output 在 camera token 位置的 concat

# VGGT-Omega:
aggregated_tokens_list[-1][:, :, 0, :]  # [B, S, 2048]
# 相同: frame 和 inter-frame 输出的 concat
```

**两者的输出格式完全兼容**: 都是 2048 维 (1024 frame + 1024 global/inter-frame)。

### 5.2 为什么不用 Camera Head

- Camera Head 是一个独立的 prediction head，输出 9D pose
- 我们需要的是 **丰富的几何特征表示** (2048D)，而非最终的 9D 预测
- 2048D 的 camera token 包含了模型对多帧几何关系的全部理解
- Camera Head 只是这些知识的一个"解码器"

### 5.3 Offline 特征提取

```python
# vggt_feature_extractor.py 中:
# 对于 VGGT:
model = VGGT(img_size=518, patch_size=14)
aggregated_tokens_list, _ = model.aggregator(images)
camera_features = aggregated_tokens_list[-1][:, :, 0, :]  # [1, S, 2048]

# 对于 VGGT-Omega:
model = VGGTOmega(patch_size=16)
aggregated_tokens_list, patch_token_start = model.aggregator(images)
camera_features = aggregated_tokens_list[-1][:, :, 0, :]  # [1, S, 2048]
```

### 5.4 预处理差异

| | VGGT | VGGT-Omega |
|--|------|------------|
| 输入尺寸 | 518×518 (固定) | 512×512 (或可变) |
| 归一化 | ImageNet mean/std | ImageNet mean/std |
| Patch Grid | 37×37 = 1369 | 32×32 = 1024 |

### 5.5 蒸馏对齐

在我们的 CamDistill 中:
- 学生 (Qwen3-VL ViT) camera token 维度: 1024 (通过 CameraTokenModule 生成)
- 教师 (VGGT/Omega) camera token 维度: 2048
- 对齐投影: `align_proj = Linear(1024, 2048)` 在 `CameraTokenProjector` 中

```python
# camdistill_loss.py 中:
student_camera_token = student_output  # [B, S, 1024]
student_aligned = align_proj(student_camera_token)  # [B, S, 2048]
teacher_camera_token = load_from_cache(...)  # [B, S, 2048]
loss = 1 - cosine_similarity(student_aligned, teacher_camera_token)
```

---

## 附录: 参数量估算

### VGGT (~1.26B)

| 组件 | 估算参数 |
|------|---------|
| DINOv2 ViT-L/14 (Patch Embed) | ~304M |
| Frame Blocks ×24 | ~302M |
| Global Blocks ×24 | ~302M |
| Camera Token + Register Token | ~10K |
| Camera Head | ~100M |
| DPT Head (depth) | ~50M |
| DPT Head (point) | ~50M |
| Track Head | ~50M |
| **总计** | **~1.26B** |

### VGGT-Omega (~1.14B)

| 组件 | 估算参数 |
|------|---------|
| DINOv3 ViT-L/16 (Patch Embed) | ~304M |
| Frame Blocks ×24 | ~302M |
| Inter-Frame Blocks ×24 | ~302M |
| Camera Token + Register Token | ~33K |
| Camera Head | ~135M |
| Dense Head | ~100M |
| **总计** | **~1.14B** |

---

## 附录: 完整前馈流程图

### VGGT

```
输入: images [B, S, 3, 518, 518]
  │
  ├─ ImageNet Normalize: (images - mean) / std
  │
  ├─ Reshape: [B*S, 3, 518, 518]
  │
  ├─ DINOv2 ViT-L/14: → patch_tokens [B*S, 1369, 1024]
  │
  ├─ slice_expand_and_flatten(camera_token):  [B*S, 1, 1024]
  ├─ slice_expand_and_flatten(register_token): [B*S, 4, 1024]
  │
  ├─ Concat: tokens [B*S, 1374, 1024]
  │
  ├─ PositionGetter: pos [B*S, 1374, 2]  (special=0, patch=1..37)
  │
  ├─ 交替注意力 ×24 层:
  │   ├─ Frame Block[i]: (B*S, 1374, 1024) → (B*S, 1374, 1024)
  │   │   └─ 帧内 self-attention + RoPE + LayerScale + MLP
  │   ├─ Global Block[i]: (B, S*1374, 1024) → (B, S*1374, 1024)
  │   │   └─ 全局 self-attention + RoPE + LayerScale + MLP
  │   └─ 缓存: cat([frame, global], dim=-1) → [B, S, 1374, 2048]
  │
  ├─ Camera Head:
  │   ├─ 取 tokens[-1][:,:,0] → [B, S, 2048] (camera token)
  │   ├─ token_norm
  │   ├─ 4次迭代:
  │   │   ├─ embed_pose(prev_or_empty) → modulation
  │   │   ├─ AdaLN(pose_tokens, shift, scale, gate)
  │   │   ├─ trunk(4层Block): (B, S, 2048) → (B, S, 2048)
  │   │   ├─ pose_branch: (B, S, 2048) → (B, S, 9)
  │   │   └─ accumulate: pred_pose += delta
  │   └─ activate_pose: T(3) + Quat(4) + ReLU(FoV)(2)
  │
  └─ 输出: pose_enc [B, S, 9]
```

### VGGT-Omega

```
输入: images [B, S, 3, H, W]  (e.g., 512×512)
  │
  ├─ ImageNet Normalize: (images - mean) / std
  │
  ├─ Reshape: [B*S, 3, H, W]
  │
  ├─ DINOv3 ViT-L/16: → patch_tokens [B*S, H/16*W/16, 1024]
  │                     e.g., [B*S, 1024, 1024]
  │
  ├─ slice_expand_and_flatten(camera_token):  [B*S, 1, 1024]
  ├─ slice_expand_and_flatten(register_token): [B*S, 16, 1024]
  │
  ├─ Concat: tokens [B*S, 1041, 1024]
  │
  ├─ RoPE 预计算: rope_embed(H=32, W=32) → (sin, cos) 各 [1024, 64]
  │
  ├─ 交替注意力 ×24 层:
  │   ├─ Frame Block[i]: (B*S, 1041, 1024) + RoPE (skip prefix 17)
  │   │
  │   ├─ Inter-Frame Block[i]:
  │   │   ├─ 如果 attention_type == "global":
  │   │   │   └─ (B, S*1041, 1024) self-attention, 无RoPE
  │   │   ├─ 如果 attention_type == "register":
  │   │   │   ├─ 分离: camera+register → (B, S*17, 1024)
  │   │   │   ├─ 冻结: patch_tokens (不参与)
  │   │   │   ├─ self-attention 只对 camera+register
  │   │   │   └─ 重新拼接
  │   │   └─ (层2,6,9,14,20 用 register; 其余用 global)
  │   │
  │   └─ 缓存: cat([frame_out, inter_frame_out], dim=-1) → [B, S, 1041, 2048]
  │
  ├─ Camera Head:
  │   ├─ 取 tokens[-1][:,:,:17] → [B, S, 17, 2048]
  │   ├─ token_norm
  │   ├─ Reshape: [B, S*17, 2048]
  │   ├─ 4个 SelfAttentionBlock (无RoPE, 无QK-Norm)
  │   ├─ Reshape: [B, S, 17, 2048]
  │   ├─ 取 camera token: [:,:,0] → [B, S, 2048]
  │   ├─ trunk_norm
  │   ├─ camera_branch: Linear(2048→1024) + GELU + Linear(1024→9)
  │   └─ activate: T(3) + Quat(4) + ReLU+0.01(FoV)(2)
  │
  └─ 输出: pose_enc [B, S, 9]
```

---

*文档版本: v1.0*
*生成日期: 2026-05-21*
*基于源码路径:*
- *VGGT: `/group/40009/dazhaodu/vggt/`*
- *VGGT-Omega: `/group/40009/dazhaodu/vggt-omega/`*
