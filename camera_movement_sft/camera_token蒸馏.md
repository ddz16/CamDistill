# CamDistill：基于Camera Token的MLLM空间感知蒸馏技术方案

## 1. 项目背景与核心思想

### 1.1 问题背景

多模态大语言模型（MLLM）在视觉-语言任务上取得了显著进展，但其空间理解能力仍然不足。当前MLLM面临以下核心挑战：

|挑战|具体表现|
|---|---|
|Vision Encoder冻结|预训练的ViT与文本语义对齐良好，但缺乏3D空间感知能力|
|空间信息丢失|研究表明Vision Encoder在预训练阶段已丢失大量空间信息[1]|
|现有方案局限|额外3D编码器（如VGGT）过于heavy；中间层对齐无法恢复丢失信息|

### 1.2 核心思想

本方案提出**CamDistill**（Camera Token Distillation），核心思想是：为每帧图像附加一个可学习的Camera Token，通过蒸馏VGGT的Camera Token特征，将3D空间感知能力注入MLLM，同时保持原有视觉编码器的语义能力不变。

设计哲学遵循三大原则：

1. **轻量注入**：仅增加Camera Token，不修改原始视觉token的计算路径
2. **精准蒸馏**：仅对齐Camera Token特征，而非全部视觉特征
3. **推理零开销**：推理时完全不需要VGGT，仅依赖学习到的Camera Token

## 2. VGGT Camera Token机制详解

### 2.1 VGGT架构概述

VGGT（Visual Geometry Grounded Transformer）是Meta提出的多视角几何基础模型[2]，其核心设计是通过Camera Token聚合几何信息。基于对VGGT源码的分析，其关键机制如下：

**Token组成结构**：每帧包含Patch Tokens（DINOv2冻结提取）+ 1个Camera Token + 4个Register Tokens。

**Camera Token参数定义**：
```python
# 关键：shape = (1, 2, 1, embed_dim)，而非(1, 1, 1, embed_dim)
self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
nn.init.normal_(self.camera_token, std=1e-6)
```

### 2.2 双Camera Token设计

VGGT采用**两个独立的可学习Camera Token**，而非所有帧共享同一个：

|索引|用途|含义|
|---|---|---|
|`camera_token[:, 0, :, :]`|第一帧专用|世界坐标系锚点，位姿固定为Identity|
|`camera_token[:, 1, :, :]`|其他帧共享|表达相对于第一帧的位姿变换|

**分配逻辑**（`slice_expand_and_flatten`函数）：
```python
def slice_expand_and_flatten(token_tensor, B, S):
    # token_tensor: (1, 2, X, C)
    # 第一帧：取index=0的token
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # 其他帧：取index=1的token，复制S-1次
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    return combined.view(B * S, *combined.shape[2:])
```

### 2.3 "可学习常数"的正确理解

论文中"第一帧Camera Token固定为可学习常数"的含义需要准确理解：

|概念|含义|是否变化|
|---|---|---|
|**输入端初始值**|输入Transformer之前的embedding|固定（不依赖输入图像）|
|**输出端特征**|经过Transformer之后的特征|会变化（聚合了图像几何信息）|

**关键澄清**：所有帧的Camera Token都参与完整的前向计算，"常数"指的是初始值与输入图像无关，而非不参与计算。

### 2.4 Alternating Attention机制

VGGT采用24层交替注意力结构：

|注意力类型|作用|Token形状|
|---|---|---|
|Frame-wise Self-Attention|Camera Token与本帧Patch Tokens交互，聚合局部几何特征|`[B*S, P, C]`|
|Global Self-Attention|所有Camera Tokens跨帧交互，解决多视角几何约束|`[B, S*P, C]`|

### 2.5 Register Tokens说明

Register Tokens是DINOv2论文中引入的技术[3]，用于抑制低信息量patch tokens产生的artifact。由于本方案基于Qwen2-VL的SigLIP ViT（未使用register tokens机制），**无需引入Register Tokens**。

## 3. 核心设计挑战

### 3.1 核心矛盾

|目标|要求|矛盾|
|---|---|---|
|Camera Token学习空间信息|需要与Visual Tokens深度交互|交互会导致Visual Tokens改变|
|Visual Tokens保持语义能力|需要冻结或最小化修改|冻结后Camera Token无法有效聚合信息|

### 3.2 投影权重冻结问题

如果完全冻结ViT权重，会导致严重问题：

```
问题链：
ViT权重冻结 → Q/K/V投影矩阵固定 
           → Camera Token用冻结的Q去attend冻结的K/V
           → 投影矩阵是为"visual-to-visual"优化的
           → Camera Token无法学到有用的空间信息
```

**核心洞察**：Camera Token必须拥有自己的可学习投影权重，才能学会"从哪些Visual Token提取什么信息"。

## 4. 架构方案对比

### 4.1 方案A：ViT后Cross-Attention

**设计思路**：完全不修改ViT，Camera Token在ViT输出之后通过Cross-Attention聚合视觉信息。

```python
class CamDistill_CrossAttn(nn.Module):
    def __init__(self, hidden_dim):
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, hidden_dim))
        self.cam_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8)
            for _ in range(4)
        ])
        
    def forward(self, images):
        # SigLIP提取visual tokens（完全冻结）
        with torch.no_grad():
            visual_tokens = self.siglip_vit(images)
        
        cam_tok = self.camera_token.expand(B, 1, -1)
        for cross_attn in self.cam_cross_attn:
            cam_tok = cross_attn(
                query=cam_tok, key=visual_tokens, value=visual_tokens
            )[0] + cam_tok
        
        tokens = torch.cat([cam_tok, visual_tokens], dim=1)
        return tokens
```

|优点|缺点|
|---|---|
|Visual tokens完全不变|Camera token无法深度参与ViT内部计算|
|实现最简单|信息流单向，蒸馏效果可能有限|

### 4.2 方案B：魔改Attention + 非对称Mask

**设计思路**：Camera Token和Visual Tokens一起输入ViT，但通过非对称Mask确保Visual Tokens看不到Camera Token。

```python
def create_asymmetric_mask(seq_len, device):
    """
    非对称Mask设计：
    - Camera Token (位置0) 可以attend所有Visual Tokens
    - Visual Tokens (位置1~N) 看不到Camera Token
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask[1:, 0] = float('-inf')  # Visual看不到Camera
    return mask
```

**Attention矩阵示意**：
```
          Camera  V1    V2    V3   ...
Camera    [  ✓     ✓     ✓     ✓   ...]   ← Camera可以看所有
V1        [  ✗     ✓     ✓     ✓   ...]   ← Visual看不到Camera
V2        [  ✗     ✓     ✓     ✓   ...]
V3        [  ✗     ✓     ✓     ✓   ...]
```

|优点|缺点|
|---|---|
|Camera深度参与ViT计算|需要魔改ViT attention|
|Visual tokens self-attention完全不变|实现复杂度中等|
|与VGGT计算方式最接近||

### 4.3 方案C：层间Adapter

**设计思路**：在ViT层之间插入轻量级Adapter，仅更新Camera Token。

```python
class CameraAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck=64):
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, camera_token, visual_tokens):
        # 只更新camera_token
        cam = self.cross_attn(camera_token, visual_tokens, visual_tokens)[0]
        cam = cam + camera_token
        cam = self.ffn(cam) + cam
        return cam, visual_tokens  # visual_tokens直接pass through
```

|优点|缺点|
|---|---|
|Visual tokens完全不变|Camera与Visual交互深度有限|
|模块化设计，不改ViT内部|需要决定在哪些层插入|

### 4.4 方案D：完全分离双路径

**设计思路**：Camera和Visual走完全独立的attention路径。

```python
class DualPathAttention(nn.Module):
    def __init__(self, original_attn, hidden_dim):
        self.visual_attn = original_attn  # 冻结
        self.camera_attn = nn.MultiheadAttention(hidden_dim, num_heads=8)  # 可学习
        
    def forward(self, x):
        camera_tok = x[:, 0:1, :]
        visual_tok = x[:, 1:, :]
        
        with torch.no_grad():
            visual_out = self.visual_attn(visual_tok)
        camera_out = self.camera_attn(
            query=camera_tok, key=visual_tok, value=visual_tok
        )[0]
        
        return torch.cat([camera_out, visual_out], dim=1)
```

|优点|缺点|
|---|---|
|Camera有完整可学习attention|参数量较大|
|Visual tokens完全不变|Camera与Visual的交互模式与VGGT不同|

### 4.5 方案对比总结

|方案|改ViT|Visual影响|蒸馏效果|新增参数|实现复杂度|推荐度|
|---|---|---|---|---|---|---|
|A:Cross-Attn后置|否|完全不变|一般|~2M|低|⭐⭐⭐|
|B:非对称Mask|是|几乎不变|最好|~30M|中|⭐⭐⭐⭐⭐|
|C:层间Adapter|否|完全不变|中等|~5M|中|⭐⭐⭐⭐|
|D:双路径分离|是|完全不变|好|~60M|高|⭐⭐⭐⭐|

## 5. Camera Token可学习投影方案

针对方案B（非对称Mask），为解决"冻结投影无法学习"的问题，提出四种Camera Token投影实现方式：

### 5.1 方案B1：Camera专用投影（推荐）

为Camera Token配置独立的可学习Q/Out投影矩阵：

```python
class CameraAwareAttention(nn.Module):
    def __init__(self, original_attn, hidden_dim):
        super().__init__()
        self.original_attn = original_attn  # 冻结
        
        # Camera专用投影（可学习）
        self.camera_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.camera_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 从原始权重初始化
        self.camera_q_proj.weight.data = original_attn.q_proj.weight.data.clone()
        self.camera_q_proj.bias.data = original_attn.q_proj.bias.data.clone()
        
    def forward(self, x, mask=None):
        camera_tok = x[:, 0:1, :]
        visual_tok = x[:, 1:, :]
        
        # Visual: 冻结路径
        with torch.no_grad():
            v_q = self.original_attn.q_proj(visual_tok)
            v_k = self.original_attn.k_proj(visual_tok)
            v_v = self.original_attn.v_proj(visual_tok)
        
        # Camera: 可学习Q
        c_q = self.camera_q_proj(camera_tok)
        with torch.no_grad():
            c_k = self.original_attn.k_proj(camera_tok)
            c_v = self.original_attn.v_proj(camera_tok)
        
        # 组合计算
        Q = torch.cat([c_q, v_q], dim=1)
        K = torch.cat([c_k, v_k], dim=1)
        V = torch.cat([c_v, v_v], dim=1)
        
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ V
        
        # 分别处理输出
        camera_out = self.camera_out_proj(out[:, 0:1, :])
        with torch.no_grad():
            visual_out = self.original_attn.out_proj(out[:, 1:, :])
        
        return torch.cat([camera_out, visual_out], dim=1)
```

**新增参数**：每层约2×D² ≈ 2.6M（D=1152），24层共约30M。

### 5.2 方案B2：LoRA风格适配

使用低秩分解，极大减少参数量：

```python
class LoRACameraAttention(nn.Module):
    def __init__(self, original_attn, hidden_dim, rank=16):
        super().__init__()
        self.original_attn = original_attn
        
        # LoRA for Camera Q
        self.lora_q_down = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_q_up = nn.Linear(rank, hidden_dim, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_q_down.weight)
        nn.init.zeros_(self.lora_q_up.weight)
        self.scale = 1.0
        
    def forward(self, x, mask=None):
        camera_tok = x[:, 0:1, :]
        
        # Camera Q = 冻结Q + LoRA delta
        with torch.no_grad():
            c_q_frozen = self.original_attn.q_proj(camera_tok)
        c_q_delta = self.lora_q_up(self.lora_q_down(camera_tok)) * self.scale
        c_q = c_q_frozen + c_q_delta
        # ... 后续计算同上
```

**新增参数**：每层约2×D×r ≈ 37K（rank=16），24层共约0.9M。

### 5.3 方案B3：Post-Attention Adapter

在attention输出后仅对Camera Token加adapter：

```python
class PostAttentionAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck=64):
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_dim)
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        
    def forward(self, x):
        camera_out = x[:, 0:1, :]
        visual_out = x[:, 1:, :]
        camera_out = camera_out + self.adapter(camera_out)
        return torch.cat([camera_out, visual_out], dim=1)
```

**新增参数**：每层约2×D×bottleneck ≈ 150K，24层共约3.6M。

### 5.4 投影方案对比

|方案|新增参数|学习能力|实现复杂度|推荐场景|
|---|---|---|---|---|
|B1:专用投影|~30M|最强|中|参数预算充足|
|B2:LoRA|~0.9M|中等|低|参数预算紧张|
|B3:Post-Adapter|~3.6M|较弱|最低|快速验证|

## 6. 推荐方案完整实现

综合考虑蒸馏效果和实现复杂度，推荐**方案B+B1**（非对称Mask + Camera专用投影）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CamDistillViT(nn.Module):
    """
    CamDistill: 基于Camera Token的MLLM空间感知蒸馏模型
    基于Qwen2-VL的SigLIP ViT实现
    """
    
    def __init__(self, siglip_vit, hidden_dim=1152, num_layers=24):
        super().__init__()
        self.vit = siglip_vit
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 双Camera Token（和VGGT一致）
        # [:, 0] = 第一帧专用, [:, 1] = 其他帧共享
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, hidden_dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        
        # 每层的Camera专用投影
        self.camera_q_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.camera_out_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # 初始化：从原始ViT权重复制
        self._init_camera_projections()
        
        # 冻结ViT
        for param in self.vit.parameters():
            param.requires_grad = False
            
    def _init_camera_projections(self):
        """从原始ViT权重初始化Camera投影"""
        for i, block in enumerate(self.vit.blocks):
            if hasattr(block.attn, 'q_proj'):
                self.camera_q_projs[i].weight.data = block.attn.q_proj.weight.data.clone()
                self.camera_q_projs[i].bias.data = block.attn.q_proj.bias.data.clone()
            if hasattr(block.attn, 'out_proj'):
                self.camera_out_projs[i].weight.data = block.attn.out_proj.weight.data.clone()
                self.camera_out_projs[i].bias.data = block.attn.out_proj.bias.data.clone()
    
    def prepare_camera_tokens(self, batch_size, num_frames):
        """准备Camera Tokens，和VGGT一致的分配逻辑"""
        if num_frames == 1:
            return self.camera_token[:, 0:1, :, :].expand(batch_size, 1, 1, -1)
        else:
            first_frame = self.camera_token[:, 0:1, :, :].expand(batch_size, 1, 1, -1)
            other_frames = self.camera_token[:, 1:2, :, :].expand(batch_size, num_frames-1, 1, -1)
            return torch.cat([first_frame, other_frames], dim=1)
    
    def create_asymmetric_mask(self, seq_len, device):
        """创建非对称Mask：Camera可看Visual，Visual看不到Camera"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask[1:, 0] = float('-inf')
        return mask
    
    def forward(self, images, num_frames=1):
        """
        Args:
            images: [B, C, H, W] 或 [B, S, C, H, W] 多帧
            num_frames: 帧数
        Returns:
            camera_tokens: [B, S, D] 用于蒸馏
            visual_tokens: [B, S, N, D] 用于后续处理
        """
        B = images.shape[0]
        
        # 1. Patch Embedding（冻结）
        with torch.no_grad():
            if len(images.shape) == 5:
                B, S, C, H, W = images.shape
                images = images.view(B * S, C, H, W)
                patch_tokens = self.vit.patch_embed(images)
                patch_tokens = patch_tokens.view(B, S, -1, self.hidden_dim)
            else:
                patch_tokens = self.vit.patch_embed(images)
                patch_tokens = patch_tokens.unsqueeze(1)
                S = 1
        
        N = patch_tokens.shape[2]
        
        # 2. 准备Camera Tokens
        cam_tokens = self.prepare_camera_tokens(B, S)
        cam_tokens = cam_tokens.squeeze(2)
        
        # 3. 合并处理每帧
        all_camera_outs = []
        all_visual_outs = []
        
        for frame_idx in range(S):
            frame_patches = patch_tokens[:, frame_idx, :, :]
            frame_camera = cam_tokens[:, frame_idx:frame_idx+1, :]
            
            tokens = torch.cat([frame_camera, frame_patches], dim=1)
            mask = self.create_asymmetric_mask(N + 1, device=images.device)
            
            # 4. 过ViT层（带Camera专用投影）
            for layer_idx, block in enumerate(self.vit.blocks):
                tokens = self._forward_block_with_camera(
                    block, tokens, mask, layer_idx
                )
            
            all_camera_outs.append(tokens[:, 0:1, :])
            all_visual_outs.append(tokens[:, 1:, :])
        
        camera_out = torch.cat(all_camera_outs, dim=1)
        visual_out = torch.stack(all_visual_outs, dim=1)
        
        return camera_out, visual_out
    
    def _forward_block_with_camera(self, block, x, mask, layer_idx):
        """单层forward，Camera用专用投影"""
        # Attention with Camera专用投影
        attn_out = self._camera_aware_attention(
            block.attn, x, mask, layer_idx
        )
        x = x + attn_out
        
        # FFN（正常，不需要改）
        x = x + block.mlp(block.norm2(x))
        return x
    
    def _camera_aware_attention(self, attn_module, x, mask, layer_idx):
        """Camera-aware attention计算"""
        B, N, D = x.shape
        
        camera_tok = x[:, 0:1, :]
        visual_tok = x[:, 1:, :]
        
        # Visual: 冻结路径
        with torch.no_grad():
            normed_visual = attn_module.norm1(visual_tok) if hasattr(attn_module, 'norm1') else visual_tok
            v_q = attn_module.q_proj(normed_visual)
            v_k = attn_module.k_proj(normed_visual)
            v_v = attn_module.v_proj(normed_visual)
        
        # Camera: 可学习Q投影
        normed_camera = attn_module.norm1(camera_tok) if hasattr(attn_module, 'norm1') else camera_tok
        c_q = self.camera_q_projs[layer_idx](normed_camera)
        with torch.no_grad():
            c_k = attn_module.k_proj(normed_camera)
            c_v = attn_module.v_proj(normed_camera)
        
        # 组合
        Q = torch.cat([c_q, v_q], dim=1)
        K = torch.cat([c_k, v_k], dim=1)
        V = torch.cat([c_v, v_v], dim=1)
        
        # Multi-head attention
        num_heads = attn_module.num_heads
        head_dim = D // num_heads
        Q = Q.view(B, N, num_heads, head_dim).transpose(1, 2)
        K = K.view(B, N, num_heads, head_dim).transpose(1, 2)
        V = V.view(B, N, num_heads, head_dim).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        
        # 分别处理输出
        camera_out = self.camera_out_projs[layer_idx](out[:, 0:1, :])
        with torch.no_grad():
            visual_out = attn_module.out_proj(out[:, 1:, :])
        
        return torch.cat([camera_out, visual_out], dim=1)
```

## 7. Loss函数设计

### 7.1 总体Loss结构

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_1 \mathcal{L}_{cam} + \lambda_2 \mathcal{L}_{preserve} + \lambda_3 \mathcal{L}_{pose}$$

### 7.2 各Loss定义

```python
class CamDistillLoss(nn.Module):
    """CamDistill训练Loss"""
    
    def __init__(self, lambda1=0.5, lambda2=0.1, lambda3=0.1):
        super().__init__()
        self.lambda1 = lambda1  # Camera蒸馏权重
        self.lambda2 = lambda2  # Visual保持权重
        self.lambda3 = lambda3  # 位姿监督权重（可选）
        
    def forward(self, outputs, targets, 
                camera_pred, camera_vggt,
                visual_pred, visual_original,
                pose_pred=None, pose_gt=None):
        """
        Args:
            outputs: 模型预测（用于任务loss）
            targets: 标签
            camera_pred: 预测的camera token [B, S, D]
            camera_vggt: VGGT的camera token [B, S, D]
            visual_pred: 预测的visual tokens
            visual_original: 原始冻结ViT的visual tokens
            pose_pred: 可选的位姿预测
            pose_gt: 可选的位姿GT
        """
        
        # 1. 任务Loss（VQA/Caption等）
        L_task = F.cross_entropy(outputs, targets)
        
        # 2. Camera蒸馏Loss（核心）
        # 使用余弦相似度，比MSE更鲁棒
        L_cam = 1 - F.cosine_similarity(
            camera_pred.flatten(1), 
            camera_vggt.flatten(1), 
            dim=1
        ).mean()
        
        # 3. Visual保持Loss
        # 验证visual tokens没有显著变化（理论上应≈0）
        L_preserve = F.mse_loss(visual_pred, visual_original)
        
        # 4. 可选：位姿监督Loss
        L_pose = 0
        if pose_pred is not None and pose_gt is not None:
            # 第一帧跳过（作为世界坐标系原点）
            L_pose = self._pose_loss(pose_pred[:, 1:], pose_gt[:, 1:])
        
        # 总Loss
        total_loss = (L_task + 
                     self.lambda1 * L_cam + 
                     self.lambda2 * L_preserve +
                     self.lambda3 * L_pose)
        
        return total_loss, {
            'L_task': L_task.item(),
            'L_cam': L_cam.item(),
            'L_preserve': L_preserve.item(),
            'L_pose': L_pose.item() if isinstance(L_pose, torch.Tensor) else L_pose
        }
    
    def _pose_loss(self, pred, gt):
        """位姿Loss：旋转+平移"""
        # pred/gt: [B, S-1, 7] = [四元数4 + 平移3]
        quat_pred, trans_pred = pred[..., :4], pred[..., 4:]
        quat_gt, trans_gt = gt[..., :4], gt[..., 4:]
        
        # 旋转：四元数距离
        L_rot = 1 - (quat_pred * quat_gt).sum(dim=-1).abs().mean()
        # 平移：L2距离
        L_trans = F.mse_loss(trans_pred, trans_gt)
        
        return L_rot + L_trans
```

### 7.3 Loss权重建议

|Loss|权重|说明|
|---|---|---|
|$\mathcal{L}_{task}$|1.0|主任务，基准权重|
|$\mathcal{L}_{cam}$|0.5|Camera蒸馏，核心Loss|
|$\mathcal{L}_{preserve}$|0.1|验证性质，权重可小|
|$\mathcal{L}_{pose}$|0.1|可选，有位姿标注时使用|

## 8. 训练策略与配置

### 8.1 训练阶段

本方案采用**单阶段训练**，无需复杂的多阶段预训练：

|阶段|Epoch|学习率|训练组件|冻结组件|
|---|---|---|---|---|
|统一训练|3-5|2e-4|Camera Token, Camera投影|SigLIP ViT, LLM|

### 8.2 超参数配置

```python
config = {
    # 模型配置
    'base_model': 'Qwen2-VL-7B',
    'hidden_dim': 1152,          # SigLIP ViT维度
    'num_layers': 24,            # ViT层数
    
    # Camera Token
    'camera_token_init_std': 1e-6,
    
    # 训练参数
    'batch_size': 16,
    'epochs': 3,
    'lr_camera': 2e-4,           # Camera相关参数
    'lr_projector': 1e-4,        # Projector（如果训练）
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    
    # Loss权重
    'lambda_cam': 0.5,
    'lambda_preserve': 0.1,
    'lambda_pose': 0.1,
    
    # 数据配比
    'spatial_data_ratio': 0.7,   # 空间相关数据
    'general_vqa_ratio': 0.3,    # 通用VQA数据
}
```

### 8.3 VGGT特征提取

训练前需要预先提取VGGT的Camera Token特征作为蒸馏目标：

```python
def extract_vggt_features(vggt_model, dataloader, save_path):
    """预提取VGGT Camera Token特征"""
    vggt_model.eval()
    all_features = {}
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images']  # [B, S, C, H, W]
            image_ids = batch['image_ids']
            
            # VGGT前向
            outputs, patch_start_idx = vggt_model(images)
            
            # 提取最后一层的camera token
            last_output = outputs[-1]  # [B, S, P, 2C]
            camera_tokens = last_output[:, :, 0, :]  # [B, S, 2C]
            
            for i, img_id in enumerate(image_ids):
                all_features[img_id] = camera_tokens[i].cpu()
    
    torch.save(all_features, save_path)
```

### 8.4 训练数据建议

|数据类型|比例|用途|
|---|---|---|
|用户自有空间数据|50%|核心空间理解能力|
|ScanNet/KITTI|20%|补充场景多样性|
|通用VQA|30%|防止过拟合，保持通用能力|

## 9. 方案总结与对比

### 9.1 本方案与其他方法对比

|方法|修改位置|推理开销|参数增量|能力保持|空间学习能力|
|---|---|---|---|---|---|
|额外3D编码器|模型外部|+100%|+1B|✅完全保持|⭐⭐⭐⭐⭐|
|REPA中间层对齐|ViT中间层|0%|~10M|⚠️可能损失|⭐⭐（信息已丢失）|
|Spatial-MoE|ViT FFN|+5%|~8M|⚠️MoE影响|⭐⭐⭐⭐|
|**CamDistill**|Token层面|+0.1%|~30M|✅完全保持|⭐⭐⭐⭐|

### 9.2 CamDistill内部方案对比

|方案|架构|投影|总参数增量|实现复杂度|推荐度|
|---|---|---|---|---|---|
|A+任意|ViT后Cross-Attn|可学习|~2M|低|⭐⭐⭐|
|B+B1|非对称Mask|专用投影|~30M|中|⭐⭐⭐⭐⭐|
|B+B2|非对称Mask|LoRA|~0.9M|低|⭐⭐⭐⭐|
|C+任意|层间Adapter|可学习|~5M|中|⭐⭐⭐⭐|
|D+任意|双路径分离|独立Attn|~60M|高|⭐⭐⭐⭐|

### 9.3 推荐选择

- **首选**：方案B+B1（非对称Mask + 专用投影），蒸馏效果最好
- **参数紧张**：方案B+B2（非对称Mask + LoRA），仅增加0.9M参数
- **快速验证**：方案A（ViT后Cross-Attn），实现最简单

## 10. 实施路线图

1. **跨帧Camera Attention**：参考VGGT的Global Attention，让不同帧的Camera Token互相交互
2. **多层蒸馏**：不仅对齐最后一层，也对齐中间层的Camera Token
3. **深度监督**：增加辅助的深度估计Head，提供额外监督信号
4. **动态路由**：根据图像内容动态决定Camera Token的注入强度

## 参考文献

[1] arXiv, 2024. REPA: Relative Position Alignment for Vision-Language Pre-training. https://arxiv.org/abs/2412.07679

[2] Meta AI, 2025. VGGT: Visual Geometry Grounded Transformer. https://arxiv.org/abs/2503.11651

[3] arXiv, 2023. Vision Transformers Need Registers. https://arxiv.org/abs/2309.16588

[4] GitHub. VGGT Official Repository. https://github.com/facebookresearch/vggt
