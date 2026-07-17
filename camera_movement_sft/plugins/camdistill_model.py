"""
CamDistill: Camera Token Distillation for Spatial-Aware MLLM

核心模块:
- CameraTokenModule: 交替注意力 Camera Token 模块
- CameraTokenProjector: Camera Token → LLM 维度投影
- FrameCrossAttentionLayer: Camera attend 本帧 Visual (单向)
- GlobalCameraSelfAttentionLayer: 所有帧 Camera 互相 attend (双向)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class FrameCrossAttentionLayer(nn.Module):
    """
    Camera Token attend 本帧的 Visual Tokens（单向 Cross-Attention）

    Q = camera_token (cam_dim, 独立于 visual 维度)
    K = V = 本帧 visual tokens (vis_dim, 来自冻结 ViT 中间层)
    K/V 不包含 camera token 自身

    Q/K/V 都投影到 cam_dim 的 attention 空间; camera 流始终维持在 cam_dim。
    """

    def __init__(self, cam_dim: int = 2048, vis_dim: int = 1024, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = cam_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cam_dim = cam_dim
        self.vis_dim = vis_dim

        # Q 投影（Camera 专用, cam_dim → cam_dim）
        self.q_proj = nn.Linear(cam_dim, cam_dim)
        # K/V 投影（Visual 专用, vis_dim → cam_dim, 把视觉特征映射到 camera 的 attention 空间）
        self.k_proj = nn.Linear(vis_dim, cam_dim)
        self.v_proj = nn.Linear(vis_dim, cam_dim)
        # 输出投影
        self.out_proj = nn.Linear(cam_dim, cam_dim)

        # Pre-Norm（Q 在 cam_dim, K/V 在 vis_dim）
        self.norm_q = nn.LayerNorm(cam_dim)
        self.norm_kv = nn.LayerNorm(vis_dim)

        # QK-norm (对齐 VGGT: 注意力前对 Q/K 各做 LayerNorm(head_dim), 稳定 attention logits)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)

        # FFN（cam_dim）
        self.norm_ffn = nn.LayerNorm(cam_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cam_dim, cam_dim * 4),
            nn.GELU(),
            nn.Linear(cam_dim * 4, cam_dim),
        )

        # LayerScale (init=0.01, 让初始时残差贡献小)
        self.ls_attn = nn.Parameter(torch.ones(cam_dim) * 0.01)
        self.ls_ffn = nn.Parameter(torch.ones(cam_dim) * 0.01)

    def forward(
        self,
        cam_tokens: torch.Tensor,
        vit_features: torch.Tensor,
        frame_sizes: List[int],
    ) -> torch.Tensor:
        """
        Args:
            cam_tokens: (T_total, 1, cam_dim) — 所有 temporal groups 的 camera tokens
            vit_features: (total_patches, vis_dim) — 冻结 ViT 某层输出
            frame_sizes: list[int] — 每帧的 patch 数量

        Returns:
            cam_tokens: (T_total, 1, cam_dim) — 更新后的 camera tokens
        """
        if cam_tokens.numel() == 0:
            return cam_tokens

        t_total = int(cam_tokens.shape[0])
        if len(frame_sizes) != t_total:
            raise RuntimeError(
                f"[CamDistill] frame_sizes length mismatch: len(frame_sizes)={len(frame_sizes)} vs t_total={t_total}"
            )
        if int(sum(frame_sizes)) != int(vit_features.shape[0]):
            raise RuntimeError(
                f"[CamDistill] frame_sizes sum mismatch: sum(frame_sizes)={sum(frame_sizes)} vs "
                f"vit_features.shape[0]={vit_features.shape[0]}"
            )

        # 分组向量化: 先按 frame_size 分组，同组一次 SDPA，避免逐帧 Python 循环。
        frame_size_to_indices = {}
        for idx, size in enumerate(frame_sizes):
            key = int(size)
            frame_size_to_indices.setdefault(key, []).append(idx)

        vis_splits = torch.split(vit_features, frame_sizes, dim=0)
        attn_output = torch.zeros_like(cam_tokens)
        cam_flat = cam_tokens.squeeze(1)  # (T, cam_dim)

        for frame_size, indices in frame_size_to_indices.items():
            if frame_size <= 0:
                raise RuntimeError(f"[CamDistill] invalid frame_size={frame_size}")

            # (G, cam_dim)
            q = self.q_proj(self.norm_q(cam_flat[indices]))
            q = q.view(len(indices), self.num_heads, 1, self.head_dim)  # (G, H, 1, d)
            q = self.q_norm(q)  # QK-norm over head_dim

            # (G, N, vis_dim)
            vis_group = torch.stack([vis_splits[i] for i in indices], dim=0)
            vis_norm = self.norm_kv(vis_group)
            k = self.k_proj(vis_norm).view(len(indices), frame_size, self.num_heads, self.head_dim)
            v = self.v_proj(vis_norm).view(len(indices), frame_size, self.num_heads, self.head_dim)
            k = k.permute(0, 2, 1, 3)  # (G, H, N, d)
            v = v.permute(0, 2, 1, 3)  # (G, H, N, d)
            k = self.k_norm(k)  # QK-norm over head_dim

            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            out = out.permute(0, 2, 1, 3).reshape(len(indices), self.cam_dim)  # (G, cam_dim)
            out = self.out_proj(out).unsqueeze(1)  # (G, 1, cam_dim)

            attn_output[indices] = out

        # Residual + LayerScale
        cam_tokens = cam_tokens + self.ls_attn * attn_output

        # FFN + Residual + LayerScale
        ffn_out = self.ffn(self.norm_ffn(cam_tokens))
        cam_tokens = cam_tokens + self.ls_ffn * ffn_out

        return cam_tokens


class GlobalCameraSelfAttentionLayer(nn.Module):
    """
    所有帧的 Camera Tokens 互相 attend（完全双向 Self-Attention）

    Q = K = V = 所有帧的 camera tokens（包含自身）
    按视频隔离：不同视频的 camera tokens 不互相 attend

    这是学习帧间几何关系的关键步骤。
    """

    def __init__(self, cam_dim: int = 2048, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = cam_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cam_dim = cam_dim

        # 标准 Self-Attention QKV
        self.qkv = nn.Linear(cam_dim, cam_dim * 3)
        self.out_proj = nn.Linear(cam_dim, cam_dim)

        # Pre-Norm
        self.norm = nn.LayerNorm(cam_dim)

        # QK-norm (对齐 VGGT)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)

        # FFN
        self.norm_ffn = nn.LayerNorm(cam_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cam_dim, cam_dim * 4),
            nn.GELU(),
            nn.Linear(cam_dim * 4, cam_dim),
        )

        # LayerScale
        self.ls_attn = nn.Parameter(torch.ones(cam_dim) * 0.01)
        self.ls_ffn = nn.Parameter(torch.ones(cam_dim) * 0.01)

    def forward(
        self,
        cam_tokens: torch.Tensor,
        video_sizes: List[int],
    ) -> torch.Tensor:
        """
        Args:
            cam_tokens: (T_total, 1, D) — 所有视频所有帧的 camera tokens
            video_sizes: list[int] — 每个视频的 temporal group 数

        Returns:
            cam_tokens: (T_total, 1, D)
        """
        cam_flat = cam_tokens.squeeze(1)  # (T_total, D)
        cam_splits = torch.split(cam_flat, video_sizes, dim=0)

        outputs = []
        for cam_seq in cam_splits:
            S = cam_seq.shape[0]

            # Self-Attention (SDPA, 数值更稳更快)
            normed = self.norm(cam_seq)  # (S, D)
            qkv = self.qkv(normed).reshape(S, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(1, 2, 0, 3)  # (3, H, S, d)
            q, k, v = qkv.unbind(0)  # each: (H, S, d)
            q = self.q_norm(q)  # QK-norm over head_dim
            k = self.k_norm(k)

            out = F.scaled_dot_product_attention(q, k, v)  # (H, S, d)
            out = out.transpose(0, 1).reshape(S, self.cam_dim)  # (S, cam_dim)
            out = self.out_proj(out)

            # Residual + LayerScale
            cam_seq = cam_seq + self.ls_attn * out

            # FFN
            ffn_out = self.ffn(self.norm_ffn(cam_seq))
            cam_seq = cam_seq + self.ls_ffn * ffn_out

            outputs.append(cam_seq)

        return torch.cat(outputs, dim=0).unsqueeze(1)  # (T_total, 1, D)


class CameraTokenProjector(nn.Module):
    """
    将 Camera Token 从 ViT 维度投影到 LLM 维度
    结构模仿 Qwen3-VL Merger 的 2层MLP
    """

    def __init__(self, in_dim: int = 1152, hidden_dim: int = 2304, out_dim: int = 3584):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear_fc1 = nn.Linear(in_dim, hidden_dim)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, camera_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_tokens: (T_total, in_dim)
        Returns:
            (T_total, out_dim)
        """
        x = camera_tokens.to(self.linear_fc1.weight.dtype)
        x = self.norm(x)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class CameraTokenModule(nn.Module):
    """
    交替注意力 Camera Token 模块

    结构: [Frame CrossAttn → Global SelfAttn] × depth
    从冻结 ViT 中间层输出聚合空间信息，产出 camera token 用于蒸馏和 LLM 输入。

    结构镜像 VGGT-omega: camera 流在 stream_dim (= ViT hidden, 4B=1024) 上做交替注意力,
    最后一层同时取「帧内 cross-attn 后」和「帧间 self-attn 后」两个 camera 表征, concat 成
    2*stream_dim (=2048) 作为 camera_features —— 与 VGGT 缓存的
    torch.cat([frame_tokens, tokens]) (帧内半 + 帧间半) 逐段对应。

      - camera token / 交替注意力 / 单分支表征都在 stream_dim (=1024)
      - visual K/V 输入也是 stream_dim (ViT, 4B=1024), cross-attn 对称
      - camera_features = concat(帧内分支, 帧间分支) = 2*stream_dim (=2048), 直接对齐 VGGT, 无需额外投影
      - camera_proj: 2*stream_dim → llm_hidden_dim, 产出注入 LLM 的 camera embed

    Args:
        hidden_dim: ViT hidden dimension (K/V 输入 = camera 流维度; 4B=1024)
        num_heads: attention heads
        depth: 交替注意力轮数
        llm_hidden_dim: LLM hidden dimension (4B=2560, 8B=4096)
        cam_dim: camera 流单分支维度; 默认 = hidden_dim。camera_features = 2*cam_dim。
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        depth: int = 6,
        llm_hidden_dim: int = 2560,
        cam_dim: int = None,
    ):
        super().__init__()
        stream_dim = cam_dim if cam_dim is not None else hidden_dim  # 单分支维度 (=1024)
        self.hidden_dim = hidden_dim      # ViT visual dim (K/V)
        self.stream_dim = stream_dim      # camera 流单分支维度
        self.feature_dim = 2 * stream_dim  # concat 后维度 (=2048, 匹配 VGGT)
        self.depth = depth

        # 双 Camera Token（与 VGGT 一致）
        # [:, 0] = 第一个 temporal group 专用（世界坐标系锚点）
        # [:, 1] = 其他 temporal groups 共享（表达相对变化）
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, stream_dim))
        nn.init.normal_(self.camera_token, std=1e-3)  # 对齐 VGGT-omega init

        # Frame Cross-Attention 层 (Q=camera stream_dim, K/V=visual hidden_dim)
        self.frame_layers = nn.ModuleList(
            [FrameCrossAttentionLayer(cam_dim=stream_dim, vis_dim=hidden_dim, num_heads=num_heads)
             for _ in range(depth)]
        )

        # Global Self-Attention 层 (stream_dim)
        self.global_layers = nn.ModuleList(
            [GlobalCameraSelfAttentionLayer(cam_dim=stream_dim, num_heads=num_heads) for _ in range(depth)]
        )

        # Camera Token → LLM 投影 (2*stream_dim → llm_hidden_dim)
        self.camera_proj = CameraTokenProjector(
            in_dim=self.feature_dim,
            hidden_dim=self.feature_dim,
            out_dim=llm_hidden_dim,
        )

        # 用于保存中间输出（供 loss 使用）
        self._last_camera_features = None  # (T_total, 2*stream_dim) 直接用于蒸馏 loss, 对齐 VGGT
        self._last_camera_embeds = None  # (T_total, llm_hidden_dim) for LLM input
        self._last_video_ids = []
        self._last_video_grid_thw = None

        # 合理初始化: Linear 用 Xavier, bias 置零 (优于 PyTorch 默认的 kaiming a=√5)
        self._init_linear_weights()

    def _init_linear_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def prepare_camera_tokens(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        为每个 temporal group 分配正确的 camera token 变体

        Args:
            grid_thw: (num_videos, 3) tensor — (T, H, W) per video

        Returns:
            (T_total, 1, hidden_dim) — 每个 temporal group 一个 camera token
        """
        cam_tokens_list = []
        for t, h, w in grid_thw.tolist():
            t = int(t)
            if t >= 1:
                # 第一个 temporal group 用 variant 0
                first = self.camera_token[:, 0, :, :]  # (1, 1, D)
                cam_tokens_list.append(first.squeeze(0))  # (1, D) → will be (1, 1, D) after stack
            if t > 1:
                # 其余用 variant 1
                others = self.camera_token[:, 1, :, :].expand(1, t - 1, -1)  # (1, t-1, D)
                cam_tokens_list.append(others.squeeze(0))  # (t-1, D)

        # 拼接所有 temporal groups 的 camera tokens
        all_cam = torch.cat(cam_tokens_list, dim=0)  # (T_total, D)
        return all_cam.unsqueeze(1)  # (T_total, 1, D)

    def forward(
        self,
        vit_layer_outputs: List[torch.Tensor],
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vit_layer_outputs: list of depth tensors, each (total_patches, hidden_dim)
                               从 ViT 各层 hook 出的中间特征（已 detach）
            grid_thw: (num_videos, 3) tensor

        Returns:
            camera_embeds: (T_total, llm_hidden_dim) — 投影后, 可直接进 LLM
            camera_features: (T_total, 2*stream_dim) — concat(帧内分支, 帧间分支), 用于蒸馏 loss
        """
        # 计算每帧 patch 数和视频 temporal group 数
        frame_sizes = []
        video_sizes = []
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            patches_per_frame = h * w
            video_sizes.append(t)
            for _ in range(t):
                frame_sizes.append(patches_per_frame)

        # 准备 camera tokens
        cam_tokens = self.prepare_camera_tokens(grid_thw)  # (T_total, 1, stream_dim)

        # 统一 dtype: ViT 中间层可能是 bfloat16, 但模块权重可能是 float32
        target_dtype = self.camera_proj.linear_fc1.weight.dtype
        cam_tokens = cam_tokens.to(target_dtype)

        # 交替注意力; 最后一层单独留下「帧内 cross-attn 后」的 camera 表征作为帧内分支
        cam_after_frame = None
        for layer_idx in range(self.depth):
            # 选择 ViT 中间层输出（如果层数不够就复用最后一层）
            vit_idx = min(layer_idx, len(vit_layer_outputs) - 1)
            vit_features = vit_layer_outputs[vit_idx].to(target_dtype)  # (total_patches, hidden_dim)

            # Step A: Frame Cross-Attention (帧内, 偏空间)
            cam_tokens = self.frame_layers[layer_idx](cam_tokens, vit_features, frame_sizes)
            if layer_idx == self.depth - 1:
                cam_after_frame = cam_tokens  # (T_total, 1, stream_dim) 帧内分支

            # Step B: Global Camera Self-Attention (帧间, 偏时序)
            cam_tokens = self.global_layers[layer_idx](cam_tokens, video_sizes)

        # 双分支 concat, 镜像 VGGT 的 torch.cat([frame_tokens, tokens])
        cam_frame = cam_after_frame.squeeze(1)   # (T_total, stream_dim) 帧内半
        cam_global = cam_tokens.squeeze(1)       # (T_total, stream_dim) 帧间半
        camera_features = torch.cat([cam_frame, cam_global], dim=-1)  # (T_total, 2*stream_dim)

        camera_embeds = self.camera_proj(camera_features)  # (T_total, llm_hidden_dim)

        # 保存供 loss 使用
        self._last_camera_features = camera_features
        self._last_camera_embeds = camera_embeds

        return camera_embeds, camera_features


def inject_camera_tokens_into_embeds(
    image_embeds: torch.Tensor,
    camera_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """
    将 camera_embeds 插入到 image_embeds 中每帧 visual tokens 的最前面。

    修改前 (每帧 T 个 temporal group, 每组 H*W 个 patch):
        image_embeds = [vis_1, vis_2, ..., vis_{H*W}, vis_1, ..., vis_{H*W}, ...]
        总长度 = sum(T_i * H_i * W_i)

    修改后 (每个 temporal group 前面插一个 camera token):
        image_embeds = [CAM_1, vis_1, ..., vis_{H*W}, CAM_2, vis_1, ..., vis_{H*W}, ...]
        总长度 = sum(T_i * (1 + H_i * W_i))

    Args:
        image_embeds: (N_total_vis, llm_dim) — Qwen3-VL visual output (经过 merger)
        camera_embeds: (T_total, llm_dim) — CameraTokenModule 投影后的 camera tokens
        grid_thw: (num_videos, 3) — 每个视频的 (T, H, W)

    Returns:
        new_image_embeds: (N_total_vis + T_total, llm_dim) — 插入 camera token 后
    """
    llm_dim = image_embeds.shape[-1]
    device = image_embeds.device
    dtype = image_embeds.dtype

    results = []
    vis_offset = 0
    cam_offset = 0

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        patches_per_frame = h * w

        for frame_idx in range(t):
            # 取该帧的 camera token
            cam = camera_embeds[cam_offset].unsqueeze(0)  # (1, llm_dim)
            # 取该帧的 visual tokens
            vis = image_embeds[vis_offset: vis_offset + patches_per_frame]  # (H*W, llm_dim)

            # camera token 在前, visual tokens 在后
            results.append(cam)
            results.append(vis)

            vis_offset += patches_per_frame
            cam_offset += 1

    return torch.cat(results, dim=0)  # (N_total_vis + T_total, llm_dim)


def build_camera_position_ids(
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """
    为每帧的 camera token 生成 M-RoPE position_ids。
    Camera token 的位置编码为该帧的空间中心。

    Args:
        grid_thw: (num_videos, 3) — (T, H_grid, W_grid) per video
                  注意: H_grid, W_grid 是 merge 后的 grid 尺寸 (已除以 spatial_merge_size)
        spatial_merge_size: Qwen3-VL 的 spatial merge size (默认2)

    Returns:
        camera_pos_ids: (3, T_total) — [temporal, height, width] for each camera token
                        temporal = 该帧的时间位置
                        height = H_grid // 2 (中心)
                        width = W_grid // 2 (中心)
    """
    temporal_ids = []
    height_ids = []
    width_ids = []

    t_offset = 0
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        center_h = h // 2
        center_w = w // 2

        for frame_idx in range(t):
            temporal_ids.append(t_offset + frame_idx)
            height_ids.append(center_h)
            width_ids.append(center_w)

        t_offset += t

    device = grid_thw.device
    return torch.stack([
        torch.tensor(temporal_ids, device=device),
        torch.tensor(height_ids, device=device),
        torch.tensor(width_ids, device=device),
    ], dim=0)  # (3, T_total)
