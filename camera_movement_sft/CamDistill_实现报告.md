# CamDistill 实现方案：交替注意力 Camera Token 蒸馏 — 完整技术报告

> 本报告基于对 Qwen3-VL 模型源码（transformers 5.8.1）、ms-swift 4.3.0 训练框架、VGGT 源码的深入调研，给出在当前框架下实现 Camera Token 蒸馏的完整实现方案。

---

## 一、核心架构理解

### 1.1 Qwen3-VL Vision Encoder 架构

| 属性 | 值 |
|------|-----|
| 类名 | `Qwen3VLVisionModel` |
| 层数 | 27 层 `Qwen3VLVisionBlock` |
| Hidden dim | 1152 |
| Attention heads | 16 (head_dim=72) |
| Patch size | 16×16 (temporal_patch_size=2) |
| Spatial merge size | 2 (merger 将 2×2=4 个相邻 patch 合并为 1) |
| 输出维度 | 3584 (LLM hidden_size for 8B) / 2560 (for 4B) |
| DeepStack | 从 ViT 第 8/16/24 层提取中间特征注入 LLM 前 3 层 |
| Position encoding | ViT 内部: **2D RoPE** + **learned absolute pos embedding** (双线性插值) |
| Attention 输入形状 | **扁平化**: `(total_patches_across_all_images, 1152)` 无 batch 维 |
| 序列隔离 | 通过 `cu_seqlens` (flash_attn) 或按图片分块处理 (sdpa/eager) |

**关键源码细节**：
- Q/K/V 是融合的: `self.qkv = nn.Linear(1152, 1152*3, bias=True)`
- 输出投影: `self.proj = nn.Linear(1152, 1152)`
- 每层结构: `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`

### 1.2 Visual Token 的 Merge 全流程

```
原始视频 (7.5s, 336×336, FPS=5) → 37帧

Step 1: Temporal Grouping
  37帧 → 18 temporal groups (temporal_patch_size=2, 每2帧合并)

Step 2: Patch Embedding (Conv3d)
  Conv3d(3, 1152, kernel=[2,16,16], stride=[2,16,16])
  每 temporal group: 21×21 = 441 patches
  ViT 输入: (18 × 441, 1152) = (7938, 1152)

Step 3: ViT 27 Layers
  输出: (7938, 1152) — 维度不变

Step 4: Merger (spatial 2×2 合并)
  ViT 输出 tokens 按特殊顺序排列:
    for block_row in range(H//2):    # merged 行
      for block_col in range(W//2):  # merged 列  
        for intra_row in range(2):   # 块内行
          for intra_col in range(2): # 块内列
  
  .view(-1, 4608):  连续 4 个 token 拼接 → 4×1152 = 4608 维
  linear_fc1(4608, 4608) → GELU → linear_fc2(4608, 3584)
  
  输出: (7938/4, 3584) = (1984, 3584)

Step 5: 进入 LLM
  每 temporal group 有 (21/2)×(21/2) ≈ 110 个 visual tokens
  视频共 18 × 110 = 1984 个 visual tokens (3584维) 进 LLM
```

### 1.3 ViT 中的位置编码

**1. Absolute Position Embedding（加法，patch embedding 之后）：**
```python
pos_embeds = self.fast_pos_embed_interpolate(grid_thw)  # (total_patches, 1152)
# 从 48×48 的 learned embedding 双线性插值到实际分辨率
hidden_states = hidden_states + pos_embeds
```

**2. Rotary Position Embedding（在 Attention 中，应用于 Q/K）：**
```python
rotary_pos_emb = self.rot_pos_emb(grid_thw)  # (total_patches, head_dim//2=36)
# 每个 patch 根据 (row, col) 在 48×48 频率表中查表
# Attention 中: Q, K = apply_rotary_pos_emb_vision(Q, K, cos, sin)
```

**Camera Token 在 VGGT 中的位置编码处理：**
```python
# VGGT 源码 (aggregator.py line 224-232):
pos = self.position_getter(B*S, H//patch_size, W//patch_size, device)  # patch positions
pos = pos + 1  # 所有 patch 位置 +1
pos_special = torch.zeros(B*S, self.patch_start_idx, 2)  # Camera+Register 位置全零
pos = torch.cat([pos_special, pos], dim=1)
# Camera Token RoPE position = (0, 0) → cos=1, sin=0 → 不旋转 → "位置无关"
```

### 1.4 VGGT Camera Token 架构

| 属性 | 值 |
|------|-----|
| 类名 | `Aggregator` |
| Camera Token shape | `(1, 2, 1, 1024)` — 2个变体: 首帧/其他帧 |
| Backbone | DINOv2 ViT-L (1024 dim, patch=14, img=518) |
| 层数 | 24 frame_blocks + 24 global_blocks |
| 注意力方式 | QKV 融合: `self.qkv = nn.Linear(dim, dim*3)` + QK Norm + RoPE |
| 输出维度 | 2048 (frame 中间输出 1024 + global 中间输出 1024 拼接) |
| 帧间交互 | Global Self-Attention: 所有帧所有 tokens 互相 attend |
| Pose 输出 | `(B, S, 9)` = Translation(3) + Quaternion(4) + FoV(2) |
| 初始化 | `nn.init.normal_(self.camera_token, std=1e-6)` 接近零 |

### 1.5 VGGT-Omega：升级版 Teacher

VGGT-Omega 是 VGGT 的更强版本（源码: `/group/40009/dazhaodu/vggt-omega`），**camera token 输出格式与 VGGT 完全兼容**（2048 维），可作为替代 Teacher。

#### VGGT vs VGGT-Omega 对比

| 属性 | VGGT | VGGT-Omega |
|------|------|-----------|
| 总参数 | 1.26B | 1.14B |
| Backbone | DINOv2 ViT-L/**14** | DINOv3 ViT-L/**16** |
| 图像尺寸 | 518px (固定) | 512px (可配置) |
| Patch size | 14 | **16** |
| Register tokens | 4 | **16** |
| Camera token shape | (1, 2, 1, 1024) | (1, 2, 1, 1024) — **相同** |
| Camera token init | std=1e-6 | std=**1e-3** |
| 交替注意力 | frame + global (每层都做) | frame + **混合** (部分层只做 register-attention) |
| Camera Head | 迭代精炼 (4次 AdaLN) | **单次前向** (4层 SelfAttn on camera+register) |
| QK Norm | 无 | **有** (LayerNorm on Q/K) |
| Camera token 输出维度 | **2048** | **2048** — 完全相同 |
| 模型加载 | `from_pretrained("facebook/VGGT-1B")` | `torch.load("vggt_omega_1b_512.pt")` |
| 额外能力 | — | TextAlignmentHead (语言对齐) |

#### 关键结论

1. **输出完全兼容**: camera token 都是 2048 维，我们的 Projector/Loss 无需任何修改
2. **预处理不同**: VGGT 需要 518px, VGGT-Omega 需要 512px → 预提取脚本分别处理
3. **可无缝切换**: 通过 `--teacher vggt` 或 `--teacher vggt_omega` 选择

#### 使用方式

```bash
# 使用 VGGT 预提取
python plugins/vggt_feature_extractor.py --teacher vggt \
    --vggt_model facebook/VGGT-1B --output_dir /path/to/vggt_cache/ ...

# 使用 VGGT-Omega 预提取
python plugins/vggt_feature_extractor.py --teacher vggt_omega \
    --vggt_model /path/to/vggt_omega_1b_512.pt --output_dir /path/to/omega_cache/ ...
```

两种 cache 格式完全相同（每个 video_id 一个 `.pt` 文件，内容 `(S, 2048)` float16），训练时直接切换 `VGGT_CACHE_DIR` 即可。

**VGGT 中 Camera Token 的交互方式（每层2步）：**

```
Step 1: Frame Self-Attention
  tokens = [camera(1), register(4), patches(1369)]  shape: (B*S, 1374, 1024)
  标准 Self-Attention，无 mask，完全双向
  → Camera 聚合本帧空间信息；Patches 也能看到 Camera

Step 2: Global Self-Attention  
  tokens reshape: (B, S*1374, 1024)  所有帧拼在一起
  标准 Self-Attention，完全双向
  → 不同帧的 Camera Tokens 互相通信，学习帧间几何关系
  → Camera 也能看其他帧的 patches

重复 24 轮
```

### 1.5 ms-swift 训练框架关键接口

| 需求 | ms-swift 提供的机制 |
|------|-------------------|
| 自定义模型加载 | `register_model()` + 自定义 `ModelLoader` + `--external_plugins` |
| 自定义 loss | `BaseLoss` 子类注册到 `loss_map` + `--loss_type` |
| 冻结控制 | `--freeze_vit true` → `freeze_parameters += ['model.visual']` |
| 自定义训练参数 | `--trainable_parameters` 精确控制哪些参数可训练 |
| Teacher 模型 | GKD 已有完整范例，可在 loss 类 `__init__` 中加载 |
| Metrics 日志 | `trainer.custom_metrics[mode]['name'].update(value)` |
| 数据传递 | `pixel_values` 在 collator 中拼接，forward hook 中处理 |

---

## 二、设计方案：交替注意力 Camera Token Module

### 2.1 核心设计原则

| 原则 | 说明 | 原因 |
|------|------|------|
| **ViT 完全冻结** | 不修改 ViT 任何权重和计算 | 保持 visual tokens 不变，LLM 原有能力不受损 |
| **Camera → Visual 单向** | Frame层：Camera attend Visual，反向不行 | ViT 冻结约束的必然结果 |
| **Camera ↔ Camera 双向** | Global层：所有帧Camera互相attend | 学习帧间几何关系的关键（对齐VGGT） |
| **交替注意力** | [Frame CrossAttn → Global SelfAttn] × N | 镜像 VGGT 的核心架构 |
| **推理零VGGT开销** | VGGT 仅用于离线预提取 | 推理时只需 CameraTokenModule |

### 2.2 为什么单向合理？

**VGGT 是双向的（Frame Attention 中 Visual 也能看 Camera）**，但我们必须单向：

| 如果让 Visual attend Camera | 后果 |
|---|---|
| Visual tokens 输出改变 | → Merger 输出不同 |
| LLM 收到不同的视觉表示 | → 与预训练不一致 |
| 模型原有能力受损 | → 不可接受 |

**单向是否有效？**  
是的。原因：
1. VGGT 中 Camera Token 初始化接近零 → 早期 Visual attend Camera 贡献几乎为零
2. Visual 能看到 Camera 的主要价值是信息回流，但在我们的场景中 Camera 的信息最终通过**进入 LLM** 来发挥作用，不需要回流到 ViT
3. **跨帧 Camera Self-Attention 才是学习几何关系的核心**，这一步我们保留了双向

### 2.3 与 VGGT 的精确对应关系

```
VGGT 的一轮:                           我们的一轮:
┌─────────────────────────────┐        ┌─────────────────────────────────┐
│ Frame Self-Attention         │        │ Frame Cross-Attention            │
│ tokens = [cam, reg, patches] │        │ Q = cam_token (可学习)           │
│ 完全双向 Self-Attention      │  ≈     │ K = V = patches (冻结ViT输出)    │
│ (cam ↔ patches)             │        │ (cam → patches 单向)             │
│                              │        │ patches 完全不被修改 ✓           │
└─────────────────────────────┘        └─────────────────────────────────┘
              ↓                                          ↓
┌─────────────────────────────┐        ┌─────────────────────────────────┐
│ Global Self-Attention        │        │ Global Camera Self-Attention     │
│ 所有帧所有token互相attend    │  ≈     │ 所有帧的cam_tokens互相attend     │
│ (cam ↔ cam 跨帧)            │        │ (cam ↔ cam 跨帧, 完全双向)      │
│ (cam ↔ other_patches)       │        │ (不涉及 patches → 计算量大减)   │
└─────────────────────────────┘        └─────────────────────────────────┘
```

**数学等价性论证**：
- Frame 层的 Camera→Visual 单向 cross-attention，等价于在 Self-Attention 中对 Visual→Camera 方向施加 `-inf` mask（即文档中的"非对称 Mask 方案 B"）
- 我们选择用独立的 Cross-Attention 实现而非真的改 attention mask，是因为 Qwen3-VL ViT 的扁平化 + cu_seqlens 结构使得在内部加非对称 mask 工程上极其困难

### 2.4 Q/K/V 来源详解

**Frame Cross-Attention（每帧内部）：**

| 角色 | 来源 | 投影 | 说明 |
|---|---|---|---|
| **Q** | Camera Token (可学习参数经过N层更新后的状态) | 独立可学习 `q_proj` | "Camera 想从 visual 中找什么信息" |
| **K** | 当前帧 Visual Tokens (冻结 ViT 第 i 层输出) | 独立可学习 `k_proj` | "每个 patch 的索引特征" |
| **V** | 当前帧 Visual Tokens (同上) | 独立可学习 `v_proj` | "每个 patch 的内容特征" |

> K/V **不包含 Camera Token 自身**。原因：Camera Token 初始接近零，attend 自己没有信息增益；residual connection 已保留自身信息。

**Global Self-Attention（跨帧）：**

| 角色 | 来源 | 投影 | 说明 |
|---|---|---|---|
| **Q** | 所有帧的 Camera Tokens | 统一 `qkv` 投影 | "第 i 帧想知道其他帧什么" |
| **K** | 所有帧的 Camera Tokens | 统一 `qkv` 投影 | "第 j 帧提供什么索引" |
| **V** | 所有帧的 Camera Tokens | 统一 `qkv` 投影 | "第 j 帧提供什么信息" |

> 标准 Self-Attention，**包含自身**（attend 自己是自然的）。这让每个 Camera Token 能在看到所有其他帧后重新校准自己。

---

## 三、完整模块实现

### 3.1 CameraTokenModule 主体

```python
class CameraTokenModule(nn.Module):
    """
    交替注意力 Camera Token 模块
    结构：[Frame CrossAttn → Global SelfAttn] × depth
    从冻结 ViT 中间层输出聚合空间信息，产出 camera token 用于蒸馏和 LLM 输入
    """
    def __init__(self, hidden_dim=1152, num_heads=16, depth=6,
                 extract_layers=(4, 8, 12, 16, 20, 24)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.extract_layers = extract_layers
        
        # 双 Camera Token（与 VGGT 一致）
        # [:, 0] = 第一帧/首 temporal group 专用（世界坐标系锚点）
        # [:, 1] = 其他帧共享（表达相对变化）
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, hidden_dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        
        # Frame Cross-Attention 层（Camera attend 本帧 Visual）
        self.frame_layers = nn.ModuleList([
            FrameCrossAttentionLayer(hidden_dim, num_heads)
            for _ in range(depth)
        ])
        
        # Global Self-Attention 层（所有帧 Camera 互相 attend）
        self.global_layers = nn.ModuleList([
            GlobalCameraSelfAttentionLayer(hidden_dim, num_heads)
            for _ in range(depth)
        ])
        
        # 用于保存最后输出（供 loss 计算使用）
        self.last_output = None
    
    def prepare_camera_tokens(self, grid_thw):
        """
        为每个 temporal group 分配正确的 camera token 变体
        
        Args:
            grid_thw: (num_videos, 3) — 每个视频的 (T, H, W)
        Returns:
            (total_temporal_groups, 1, hidden_dim)
        """
        cam_tokens_list = []
        for t, h, w in grid_thw.tolist():
            t = int(t)
            if t >= 1:
                # 第一个 temporal group 用 camera_token[:, 0]
                first = self.camera_token[:, 0, :, :]  # (1, 1, D)
                cam_tokens_list.append(first)
            if t > 1:
                # 其余 temporal groups 用 camera_token[:, 1]
                others = self.camera_token[:, 1, :, :].expand(1, t-1, -1).reshape(t-1, 1, -1)
                cam_tokens_list.append(others)
        
        return torch.cat(cam_tokens_list, dim=0)  # (total_temporal_groups, 1, D)
    
    def forward(self, vit_layer_outputs, grid_thw):
        """
        Args:
            vit_layer_outputs: list of 6 tensors, each (total_patches, hidden_dim)
                               从 ViT extract_layers 各层 hook 出的中间特征
            grid_thw: (num_videos, 3) tensor
        
        Returns:
            camera_tokens: (total_temporal_groups, hidden_dim) — 蒸馏 + LLM 用
        """
        # 1. 准备 camera tokens
        cam_tokens = self.prepare_camera_tokens(grid_thw)  # (T_total, 1, D)
        
        # 2. 交替注意力
        for layer_idx in range(self.depth):
            # Step A: Frame Cross-Attention
            vit_features = vit_layer_outputs[layer_idx]  # (total_patches, D)
            cam_tokens = self.frame_layers[layer_idx](cam_tokens, vit_features, grid_thw)
            
            # Step B: Global Camera Self-Attention
            cam_tokens = self.global_layers[layer_idx](cam_tokens, grid_thw)
        
        # 3. 保存输出
        output = cam_tokens.squeeze(1)  # (T_total, D)
        self.last_output = output
        
        return output
```

### 3.2 Frame Cross-Attention Layer

```python
class FrameCrossAttentionLayer(nn.Module):
    """
    Camera Token attend 本帧的 Visual Tokens（单向 Cross-Attention）
    
    Q = camera_token (经过前面层更新后的状态)
    K = V = 本帧 visual tokens (来自冻结 ViT 中间层)
    K/V 不包含 camera token 自身
    """
    def __init__(self, hidden_dim=1152, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q 投影（Camera 专用，可学习）
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # K/V 投影（Visual 专用，可学习，可从 ViT 权重初始化）
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Pre-Norm
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        
        # FFN (Camera token 过 Cross-Attn 后还过一个 FFN)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # LayerScale (与 VGGT 一致, init=0.01 让初始时残差贡献小)
        self.ls_attn = nn.Parameter(torch.ones(hidden_dim) * 0.01)
        self.ls_ffn = nn.Parameter(torch.ones(hidden_dim) * 0.01)
    
    def forward(self, cam_tokens, vit_features, grid_thw):
        """
        Args:
            cam_tokens: (T_total, 1, D) — 所有 temporal groups 的 camera tokens
            vit_features: (total_patches, D) — 冻结 ViT 某层输出
            grid_thw: (num_videos, 3) tensor
        Returns:
            cam_tokens: (T_total, 1, D) — 更新后的 camera tokens
        """
        # 按 temporal group 拆分 visual features
        # 每个 temporal group 有 H*W 个 patches
        frame_sizes = []
        for t, h, w in grid_thw.tolist():
            patches_per_frame = int(h) * int(w)
            for _ in range(int(t)):
                frame_sizes.append(patches_per_frame)
        
        vis_splits = torch.split(vit_features, frame_sizes, dim=0)
        
        # 逐帧 Cross-Attention
        attn_outputs = []
        for i, vis in enumerate(vis_splits):
            # vis: (N_i, D) — 本帧的 visual tokens
            q = self.q_proj(self.norm_q(cam_tokens[i]))  # (1, D)
            k = self.k_proj(self.norm_kv(vis))           # (N_i, D)
            v = self.v_proj(self.norm_kv(vis))           # (N_i, D)
            
            # Multi-head reshape
            q = q.view(1, self.num_heads, self.head_dim)      # (1, H, d)
            k = k.view(-1, self.num_heads, self.head_dim)     # (N, H, d)
            v = v.view(-1, self.num_heads, self.head_dim)     # (N, H, d)
            
            # Scaled Dot-Product Attention
            # q: (1, H, d), k^T: (H, d, N) → attn: (H, 1, N)
            attn_weights = torch.einsum('qhd,nhd->hqn', q, k) * self.scale
            attn_weights = attn_weights.softmax(dim=-1)
            # attn @ v: (H, 1, d) → (1, H*d) = (1, D)
            out = torch.einsum('hqn,nhd->qhd', attn_weights, v)
            out = out.reshape(1, -1)
            out = self.out_proj(out)
            attn_outputs.append(out)
        
        attn_output = torch.stack(attn_outputs, dim=0)  # (T_total, 1, D)
        
        # Residual + LayerScale
        cam_tokens = cam_tokens + self.ls_attn.unsqueeze(0).unsqueeze(0) * attn_output
        
        # FFN + Residual + LayerScale
        ffn_out = self.ffn(self.norm_ffn(cam_tokens))
        cam_tokens = cam_tokens + self.ls_ffn.unsqueeze(0).unsqueeze(0) * ffn_out
        
        return cam_tokens
```

### 3.3 Global Camera Self-Attention Layer

```python
class GlobalCameraSelfAttentionLayer(nn.Module):
    """
    所有帧的 Camera Tokens 互相 attend（完全双向 Self-Attention）
    
    Q = K = V = 所有帧的 camera tokens（包含自身）
    
    这是学习帧间几何关系的关键步骤！
    不同帧的 camera tokens 通过 self-attention 互相传递信息，
    使每个 camera token 不仅知道本帧的空间信息，还知道与其他帧的相对关系。
    """
    def __init__(self, hidden_dim=1152, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 标准 Self-Attention QKV（与 VGGT Global Attention 对应）
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Pre-Norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # LayerScale
        self.ls_attn = nn.Parameter(torch.ones(hidden_dim) * 0.01)
        self.ls_ffn = nn.Parameter(torch.ones(hidden_dim) * 0.01)
    
    def forward(self, cam_tokens, grid_thw):
        """
        Args:
            cam_tokens: (T_total, 1, D) — 所有视频所有帧的 camera tokens
            grid_thw: (num_videos, 3) — 用于按视频隔离 attention
        Returns:
            cam_tokens: (T_total, 1, D) — 跨帧交互后的 camera tokens
        """
        # 按视频隔离 attention（不同视频的 camera tokens 不互相 attend）
        video_sizes = [int(t) for t, h, w in grid_thw.tolist()]
        cam_splits = torch.split(cam_tokens.squeeze(1), video_sizes, dim=0)
        # cam_splits: tuple of (T_i, D)
        
        outputs = []
        for cam_seq in cam_splits:
            S = cam_seq.shape[0]  # 该视频的 temporal group 数
            
            # Self-Attention
            normed = self.norm(cam_seq)  # (S, D)
            qkv = self.qkv(normed).reshape(S, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(1, 2, 0, 3)  # (3, H, S, d)
            q, k, v = qkv.unbind(0)         # each: (H, S, d)
            
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (H, S, S)
            attn_weights = attn_weights.softmax(dim=-1)
            out = (attn_weights @ v)  # (H, S, d)
            out = out.transpose(0, 1).reshape(S, -1)  # (S, D)
            out = self.out_proj(out)
            
            # Residual + LayerScale
            cam_seq = cam_seq + self.ls_attn * out
            
            # FFN
            ffn_out = self.ffn(self.norm_ffn(cam_seq))
            cam_seq = cam_seq + self.ls_ffn * ffn_out
            
            outputs.append(cam_seq)
        
        return torch.cat(outputs, dim=0).unsqueeze(1)  # (T_total, 1, D)
```

---

## 四、Camera Token 投影进入 LLM

### 4.1 Visual Token 的投影（Merger）vs Camera Token 的投影

| | Visual Token | Camera Token |
|---|---|---|
| **来源** | ViT 最后一层输出 | CameraTokenModule 最后一层输出 |
| **原始维度** | 1152 | 1152 |
| **投影方式** | Merger: 4个相邻patch拼接→4608→MLP→3584 | 独立 Projector: 1152→MLP→3584 |
| **为什么不同** | 4个空间相邻patch有冗余，合并压缩是合理的 | 只有1个token代表全帧，没有邻居可合并 |
| **投影是否可学习** | Merger 本身冻结 (`freeze_aligner=true`) | Camera Projector **可训练** |

### 4.2 Camera Token Projector 设计

```python
class CameraTokenProjector(nn.Module):
    """
    将 Camera Token 从 ViT 维度投影到 LLM 维度
    结构模仿 Merger 的 2层MLP，但输入维度适配单个 token
    """
    def __init__(self, in_dim=1152, hidden_dim=2304, out_dim=3584):
        super().__init__()
        # 结构与 Merger 对齐: LayerNorm → Linear → GELU → Linear
        self.norm = nn.LayerNorm(in_dim)
        self.linear_fc1 = nn.Linear(in_dim, hidden_dim)   # 1152 → 2304 (2x expansion)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(hidden_dim, out_dim)  # 2304 → 3584
    
    def forward(self, camera_tokens):
        """
        Args:
            camera_tokens: (T_total, 1152)
        Returns:
            (T_total, 3584) — 与 visual tokens 同维度，可直接进 LLM
        """
        x = self.norm(camera_tokens)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x
```

**参数量**: norm(2.3K) + fc1(2.66M) + fc2(8.26M) ≈ **11M**

**为什么不直接用一个 Linear(1152, 3584)?**
- 单个 Linear 表达力不足（线性变换无法学习非线性特征映射）
- 2层 MLP + GELU 与 Merger 结构一致，符合模型内部的设计惯例
- 中间维度 2304 (= 2×1152) 提供了足够的表达能力

### 4.3 Camera Token 在 LLM 序列中的位置

```
修改前（每个 temporal group）:
<|vision_start|> [vis_1, vis_2, ..., vis_110] <|vision_end|>

修改后:
<|vision_start|> [CAM, vis_1, vis_2, ..., vis_110] <|vision_end|>

其中:
  CAM = camera_proj(camera_token)  维度 3584
  vis_i = merger(4_patches)        维度 3584
```

Camera Token 放在**每帧 visual tokens 之前**的理由：
- LLM 是 causal（自回归），左边的 token 先被 attend 到
- Camera Token 在前，相当于先给 LLM 一个"这帧空间信息摘要"
- LLM 后续处理 visual tokens 时可以参考 camera token 的空间 context

### 4.4 M-RoPE Position Encoding 适配

LLM 使用 4 维 M-RoPE position_ids: `[text_pos, temporal_pos, height_pos, width_pos]`

| Token 类型 | temporal | height | width |
|---|---|---|---|
| text token | 递增序号 | = temporal | = temporal |
| visual token (i,j) | 帧对应的时间位置 | patch 行号 i | patch 列号 j |
| **camera token** | 与该帧相同 | `grid_H // (2 * merge_size)` (中心) | `grid_W // (2 * merge_size)` (中心) |

**直觉**: Camera Token 在空间上编码为帧的中心位置，表示"全帧的全局代表"。

### 4.5 DeepStack 兼容性

Qwen3-VL 的 DeepStack 在 LLM 前3层向 `visual_pos_masks` 标记的位置 additive 注入 ViT 中间层特征。

**Camera Token 位置不加入 `visual_pos_masks`**：
- Camera Token 已经通过独立的 Cross-Attention 从 ViT 聚合了充分的信息
- 再叠加 DeepStack additive features 可能导致信息冗余
- 且 DeepStack features 的维度/语义是为 visual tokens 设计的，不适合直接加到 camera token 上

---

## 五、蒸馏 Loss 设计

### 5.1 VGGT Teacher 离线预提取

```python
# 预提取脚本: plugins/vggt_feature_extractor.py
def extract_vggt_features(video_path, vggt_model, fps=5, max_frames=100):
    """
    对一个视频预提取 VGGT camera token features
    
    输入: video.mp4
    输出: camera_features tensor (S, 2048) — S = temporal groups 数
    """
    # 1. 抽帧（与训练完全一致的参数）
    frames = extract_frames(video_path, fps=fps, max_frames=max_frames)  # (S_raw, 3, H, W)
    
    # 2. VGGT 预处理 (独立于 Qwen3-VL)
    frames_518 = resize(frames, 518)                          # resize to 518×518
    frames_norm = (frames_518 - IMAGENET_MEAN) / IMAGENET_STD  # ImageNet normalize
    frames_input = frames_norm.unsqueeze(0)                    # (1, S, 3, 518, 518)
    
    # 3. VGGT 前向
    with torch.no_grad():
        output_list, patch_start_idx = vggt_model.aggregator(frames_input)
        # output_list[-1]: (1, S, 1374, 2048)
        # Camera token 在 index 0:
        camera_features = output_list[-1][:, :, 0, :]  # (1, S, 2048)
    
    return camera_features.squeeze(0).half()  # (S, 2048) float16
```

**帧数对齐问题**：
- Qwen3-VL 使用 `temporal_patch_size=2`，每2帧合并为1个 temporal group
- VGGT 处理单帧，输出每帧一个 camera token
- **对齐方案**: VGGT 输出 S 帧的 features，训练时按 Qwen3-VL 的 temporal grouping 做 2帧平均池化

```python
# 训练时对齐: VGGT (S帧) → Qwen3-VL (S//2 temporal groups)
vggt_features = cache[video_id]  # (S, 2048)
# 每2帧取平均 → (S//2, 2048)
aligned_features = vggt_features.view(-1, 2, 2048).mean(dim=1)  # (T, 2048)
```

### 5.2 蒸馏 Loss 计算

```python
class CamDistillLoss(BaseLoss):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.vggt_cache = self._load_cache(args.vggt_cache_dir)
        self.align_proj = nn.Linear(1152, 2048).to(trainer.model.device)
        self.lambda_cam = 0.3
    
    def __call__(self, outputs, labels, *, num_items_in_batch=None, 
                 loss_scale=None, trainer=None, **kwargs):
        # ===== 1. 标准 SFT Loss =====
        from swift.trainers import per_token_loss_func
        token_loss = per_token_loss_func(outputs, labels)
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        sft_loss = token_loss.sum() / num_items_in_batch
        
        # ===== 2. Camera Token 蒸馏 Loss =====
        # 获取模型产出的 camera tokens
        camera_pred = trainer.model.camdistill.last_output  # (T_total, 1152)
        
        # 获取 VGGT target (从预提取缓存查表)
        video_ids = trainer._current_video_ids  # 需要在 data collator 中传入
        camera_targets = []
        for vid in video_ids:
            target = self.vggt_cache[vid]  # (T_i, 2048)
            camera_targets.append(target)
        camera_target = torch.cat(camera_targets, dim=0).to(camera_pred.device)
        
        # 维度对齐: Student(1152) → Teacher(2048)
        camera_pred_aligned = self.align_proj(camera_pred)  # (T_total, 2048)
        
        # 余弦相似度 Loss
        pred_norm = F.normalize(camera_pred_aligned, dim=-1)
        target_norm = F.normalize(camera_target, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)  # (T_total,)
        distill_loss = (1 - cos_sim).mean()
        
        # ===== 3. 总 Loss =====
        total_loss = sft_loss + self.lambda_cam * distill_loss
        
        # ===== 4. 日志 =====
        mode = 'train' if trainer.model.training else 'eval'
        trainer.custom_metrics[mode]['sft_loss'].update(sft_loss.detach())
        trainer.custom_metrics[mode]['distill_loss'].update(distill_loss.detach())
        trainer.custom_metrics[mode]['cos_sim'].update(cos_sim.mean().detach())
        
        return total_loss
```

**为什么用余弦相似度而非 MSE**：
- VGGT (DINOv2, 1024维) 和 Qwen3-VL (SigLIP, 1152维) 的特征空间尺度完全不同
- 余弦相似度只关注方向不关注幅度，对跨模型对齐更鲁棒
- Loss 值在 [0, 2] 之间，可解释性好（0=完美对齐, 1=正交, 2=完全相反）

---

## 六、维度流转全图

```
以 Qwen3-VL-8B + 视频 7.5s (FPS=5, 336×336) 为例:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   VISUAL TOKEN 路径（冻结）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原始视频             37帧, 336×336
Temporal grouping    18 groups (2帧/group)
Patch embedding      (18×21×21, 1152) = (7938, 1152)
+ absolute pos       (7938, 1152)
ViT 27 layers        (7938, 1152)  ← hook 出第4/8/12/16/20/24层
Merger (2×2 spatial) (7938/4, 3584) = (1984, 3584)
进入 LLM             每 temporal group 110 个 tokens × 3584

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   CAMERA TOKEN 路径（可训练）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Camera Token 初始    (18, 1, 1152)  ← 从 camera_token param 分配
                     第1个 group 用 variant 0, 其余用 variant 1

Layer 1 - Frame:     Q=(18,1,1152) attend K=V=(441/frame, 1152)  × 18帧
Layer 1 - Global:    Q=K=V=(18, 1152) self-attention 跨帧
Layer 2 - Frame:     同上，用 ViT 第8层输出
Layer 2 - Global:    同上
...
Layer 6 - Frame:     用 ViT 第24层输出
Layer 6 - Global:    最终跨帧 self-attention

Camera Token 输出    (18, 1152)

分支 A: 蒸馏 Loss
  align_proj          (18, 2048) → cosine_loss with VGGT cache

分支 B: 进入 LLM
  camera_proj         (18, 3584) → 插入每帧 visual tokens 之前

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   LLM 输入（可训练）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每 temporal group:   [CAM(3584), vis_1(3584), ..., vis_110(3584)] = 111 tokens
视频总 tokens:       18 × 111 = 1998 tokens (vs 原来 1984)
M-RoPE position:     CAM 的 temporal=帧时间, H/W=帧中心

LLM 36 layers + DeepStack (前3层 additive，跳过 CAM 位置)
→ logits → Cross-Entropy Loss (只在 assistant 回复部分)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   总 LOSS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_total = L_sft + 0.3 × L_distill
        = CrossEntropy(logits, labels) + 0.3 × (1 - cosine_sim)
```

---

## 七、参数量统计

### CamDistill 方案

| 组件 | 计算 | 参数量 |
|------|------|--------|
| `camera_token` | 2 × 1 × 1152 | 2.3K |
| Frame Cross-Attn × 6 | 6 × (q+k+v+out proj + FFN + norms + ls) | ~48M |
| Global Self-Attn × 6 | 6 × (qkv+out proj + FFN + norms + ls) | ~48M |
| `camera_proj` (→LLM) | norm + fc1(1152→2304) + fc2(2304→4096) | ~11M |
| `align_proj` (→VGGT) | Linear(1152, 2048) | ~2.4M |
| **CamDistill 总计** | | **~110M** |
| LLM (可训练) | 36层 Decoder + lm_head | ~8B |
| **全部可训练参数** | CamDistill + LLM | ~8.1B |

### CamInject 方案

| 组件 | 计算 | 参数量 |
|------|------|--------|
| VGGTProjector | norm + fc1(2048→2048) + fc2(2048→4096) | **~12.6M** |
| LLM (可训练) | 36层 Decoder + lm_head | ~8B |
| **全部可训练参数** | Projector + LLM | ~8.01B |

### 各模型 CamDistill 模块实际参数量

| 模型 | ViT dim | LLM dim | CamDistill 参数 | CamInject Projector |
|------|---------|---------|----------------|----------------------|
| Qwen3-VL-4B | 1024 | 2560 | 160.6M | 9.4M |
| Qwen3-VL-8B | 1152 | 4096 | 205.8M | 12.6M |
| Qwen3.5-4B | 1024 | 2560 | 160.6M | 9.4M |
| Qwen3.5-9B | 1152 | 4096 | 205.8M | 12.6M |

---

## 八、ms-swift 集成方案

### 8.1 文件结构

```
camera_movement_sft/
├── plugins/
│   ├── __init__.py
│   ├── camdistill_plugin.py          # 总入口: register_model + register_loss
│   ├── camdistill_model.py           # CameraTokenModule + Projector + ModelLoader
│   ├── camdistill_loss.py            # CamDistillLoss (BaseLoss 子类)
│   └── vggt_feature_extractor.py     # VGGT 离线预提取脚本
├── train_camdistill.sh               # CamDistill 训练启动脚本
├── train.sh                          # 普通 SFT（完全不受影响）
└── ...
```

### 8.2 使用方式

```bash
# 普通 SFT（完全不涉及 CamDistill，零影响）
bash camera_movement_sft/train.sh qwen3vl-8b

# CamDistill SFT
bash camera_movement_sft/train_camdistill.sh qwen3vl-8b
# 内部等价于:
# swift sft \
#     --model Qwen/Qwen3-VL-8B-Instruct \
#     --external_plugins camera_movement_sft/plugins/camdistill_plugin.py \
#     --loss_type camdistill \
#     --freeze_vit true \
#     --freeze_aligner true \
#     --trainable_parameters model.camdistill model.camera_proj \
#     ...
```

### 8.3 ViT 中间输出提取（通过 Forward Hook）

```python
# 在 camdistill_plugin.py 中:
class CamDistillModelLoader(Qwen3VLLoader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        model = super().get_model(model_dir, config, processor, model_kwargs)
        
        vit_dim = config.vision_config.hidden_size   # 1152
        llm_dim = config.hidden_size                 # 3584
        
        # 1. 挂载 CameraTokenModule
        model.camdistill = CameraTokenModule(hidden_dim=vit_dim, depth=6)
        
        # 2. 挂载 Camera Projector (→LLM)
        model.camera_proj = CameraTokenProjector(in_dim=vit_dim, out_dim=llm_dim)
        
        # 3. 注册 ViT forward hooks
        extract_layers = [4, 8, 12, 16, 20, 24]  # ViT 第 4/8/12/16/20/24 层
        model._vit_intermediates = []
        
        for layer_idx in extract_layers:
            def make_hook(idx):
                def hook_fn(module, input, output):
                    model._vit_intermediates.append(output.detach())
                return hook_fn
            model.model.visual.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
        
        return model
```

### 8.4 可训练参数控制

```bash
swift sft \
    --freeze_vit true \                    # ViT 冻结
    --freeze_aligner true \                # Merger 冻结
    --freeze_llm false \                   # LLM 可训练
    --trainable_parameters \
        model.camdistill \                 # Camera Token Module 可训练
        model.camera_proj \                # Camera Projector 可训练
    ...
```

---

## 九、训练配置

### 9.1 训练策略

| 阶段 | 可训练参数 | Loss | Epochs | 学习率 |
|------|-----------|------|--------|--------|
| 阶段1: Camera热身 | CamDistill + camera_proj + align_proj | L_distill only | 1 | 2e-4 |
| 阶段2: 联合训练 | CamDistill + camera_proj + LLM | L_sft + 0.3×L_distill | 2 | LLM:1.5e-5, CamDistill:2e-4 |

或**单阶段简化版**:
- 所有参数一起训练，CamDistill 模块用更大学习率（通过参数分组）

### 9.2 VGGT 预提取配置

```bash
python camera_movement_sft/plugins/vggt_feature_extractor.py \
    --input_jsonl train_data/camera_movement_train_human_all_167k.jsonl \
    --output_dir /path/to/vggt_cache/ \
    --fps 5 --max_frames 100 \
    --vggt_model facebook/VGGT-1B \
    --batch_size 4 --num_workers 8
```

缓存大小估算: 167K videos × avg 18 temporal groups × 2048 × 2bytes (fp16) ≈ **12GB**

---

## 十、风险评估与验证方案

### 10.1 风险点

| 风险 | 影响 | 缓解方案 |
|------|------|---------|
| ViT forward hook 增加显存 | ~10-15% | 用 `.detach()` 切断梯度图；只保存必要层 |
| Camera Token 初始对 LLM 扰动 | 训练早期不稳定 | camera_proj 零初始化 + LayerScale=0.01 |
| VGGT 帧数对齐 | 蒸馏 target 维度不匹配 | 预提取用相同抽帧参数 + 2帧平均池化 |
| Global Self-Attn 内存 | 视频帧很多时 S² attention | 限制 max_frames=100 → S≤50 |
| 40GB VGGT cache | 存储压力 | float16 存储(~12GB)；或按需 lazy load |

### 10.2 验证实验设计

| 实验 | 方案 | 说明 | 目的 |
|------|------|------|------|
| Exp 0 | 普通 SFT | 无 Camera Token | 基线 |
| Exp 1 | CamInject | VGGT 输出直接注入 LLM | 验证 camera token 是否有用 |
| Exp 2 | CamDistill (无 Global) | 仅 Frame Cross-Attn | 验证 Frame-only 蒸馏 |
| Exp 3 | CamDistill (完整) | Frame + Global + 蒸馏 | 验证完整方案 |
| Exp 4 | CamDistill vs Direct | 对比推理时不用/用 VGGT | 蒸馏能否达到 Direct 效果 |

---

## 十一、需要修改/新增的文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `plugins/camdistill_plugin.py` | 新建 | 入口: register_model + register loss |
| `plugins/camdistill_model.py` | 新建 | CameraTokenModule + Projector + Loader |
| `plugins/camdistill_loss.py` | 新建 | BaseLoss 子类，蒸馏 loss |
| `plugins/caminject_model.py` | 新建 | CamInjectAdapter + VGGTProjector |
| `plugins/vggt_feature_extractor.py` | 新建 | VGGT 离线预提取 |
| `train_camdistill.sh` | 新建 | CamDistill 训练脚本 |
| `train_caminject.sh` | 新建 | CamInject 训练脚本 |
| `train.sh` | **不动** | 普通 SFT 完全不受影响 |
| ms-swift 框架代码 | **不动** | 全部通过 external_plugins 注入 |
| transformers 源码 | **不动** | 通过 forward hook 获取中间输出 |

---

## 十二、CamInject Baseline 方案

### 12.1 设计思想

与 CamDistill 蒸馏方案不同，CamInject 直接使用 VGGT 冻结输出的 camera token 作为 LLM 的输入，不需要训练 CameraTokenModule。

**核心区别**:

| | CamDistill (蒸馏) | CamInject (直接注入) |
|---|---|---|
| Camera Token 来源 | 自学习的 CameraTokenModule | VGGT 原始输出（冻结） |
| 推理时需要 VGGT | **否**（模块已学会） | **是**（仍需在线推理） |
| 训练时需要 VGGT | 否（用预提取 cache） | 否（同样用 cache） |
| 额外可训练参数 | ~200M (CameraTokenModule) | **~10M** (仅 Projector) |
| Loss | SFT + 余弦蒸馏 | **仅 SFT** |
| 实现复杂度 | 高 | **低** |
| 核心优势 | 推理时轻量（不需VGGT） | 实现简单，直接利用 VGGT 强大表示 |

### 12.2 架构

```
Video → VGGT (冻结, 1B 参数) → camera_token (2048 dim)
                                        ↓
                               VGGTProjector (可训练, ~10M)
                                        ↓
                               camera_embed (LLM dim: 2560/4096)
                                        ↓
                          插入 LLM 每帧 visual tokens 之前
                                        ↓
                               LLM forward → SFT loss
```

### 12.3 VGGTProjector 实现

```python
class VGGTProjector(nn.Module):
    """VGGT 2048 → LLM dim, 2层MLP"""
    def __init__(self, vggt_dim=2048, hidden_dim=2048, llm_dim=4096):
        self.norm = nn.LayerNorm(vggt_dim)
        self.linear_fc1 = nn.Linear(vggt_dim, hidden_dim)   # 2048 → 2048
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(hidden_dim, llm_dim)    # 2048 → 4096
```

### 12.4 各模型参数量

| 模型 | VGGT→LLM | Projector 参数 |
|------|----------|--------------|
| Qwen3-VL-4B / Qwen3.5-4B | 2048 → 2560 | 9.4M |
| Qwen3-VL-8B / Qwen3.5-9B | 2048 → 4096 | 12.6M |

### 12.5 使用方式

```bash
# 与 CamDistill 共享同一个 VGGT 预提取 cache
VGGT_CACHE_DIR=/path/to/vggt_cache \
bash camera_movement_sft/train_caminject.sh qwen3vl-8b

# 支持所有 4 个模型
VGGT_CACHE_DIR=/path/to/vggt_cache bash camera_movement_sft/train_caminject.sh qwen3vl-4b
VGGT_CACHE_DIR=/path/to/vggt_cache bash camera_movement_sft/train_caminject.sh qwen35-9b
```

### 12.6 适合场景

- **快速验证**: 想看看 VGGT 的 camera token 对运镜识别是否有帮助
- **推理环境不受限**: 如果推理时可以跑 VGGT（有足够 GPU 显存），这是最简单的方案
- **作为对比实验**: 与 CamDistill 蒸馏方案对比，看蒸馏是否能达到直接注入的效果

---

## 十三、三种方案完整对比

| 方案 | 脚本 | model_type | 可训练参数 | Loss | 推理依赖 |
|------|------|-----------|-----------|------|---------|
| 普通 SFT | `train.sh` | `qwen3_vl` | LLM (~8B) | SFT | 无 |
| CamDistill | `train_camdistill.sh` | `qwen3_vl_camdistill` | CamDistill(~200M) + LLM | SFT + Distill | 无 |
| CamInject | `train_caminject.sh` | `qwen3_vl_caminject` | Projector(~10M) + LLM | SFT | VGGT |

### 推荐实验顺序

1. **普通 SFT** → 建立 baseline
2. **CamInject** → 快速验证 camera token 是否有增益（最简单）
3. **CamDistill** → 如果 CamInject 有效，尝试蒸馏去掉推理时的 VGGT 依赖

---

## 十四、实现优先级

1. 实现 VGGT 预提取脚本（验证能跑通 + 确认帧数对齐）
2. 实现 CameraTokenModule（独立单测 forward shape）
3. 实现 CameraTokenProjector（验证输出维度）
4. 实现 ms-swift 插件注册（验证 `--external_plugins` 能加载）
5. 实现蒸馏 Loss（验证 loss 数值正常下降）
6. 实现 Camera Token 注入 LLM（position_ids 适配）
7. 联合训练 + 对比评测

---
