"""
CamDistill: Camera Token Distillation for Spatial-Aware MLLM

Core modules:
- CameraTokenModule: alternating-attention Camera Token module
- CameraTokenProjector: project Camera Token to the LLM hidden dimension
- FrameCrossAttentionLayer: camera attends to visual tokens of the current frame (one-way)
- GlobalCameraSelfAttentionLayer: camera tokens across frames attend to each other (bidirectional)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class FrameCrossAttentionLayer(nn.Module):
    """
    Camera Token attends to the current frame's visual tokens (one-way cross-attention).

    Q = camera_token (cam_dim, independent from the visual dimension)
    K = V = visual tokens of the current frame (vis_dim, from a frozen ViT intermediate layer)
    K/V do not include the camera token itself.

    Q/K/V are all projected into cam_dim attention space; the camera stream stays in cam_dim.
    """

    def __init__(self, cam_dim: int = 2048, vis_dim: int = 1024, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = cam_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cam_dim = cam_dim
        self.vis_dim = vis_dim

        # Q projection (camera side, cam_dim -> cam_dim).
        self.q_proj = nn.Linear(cam_dim, cam_dim)
        # K/V projection (visual side, vis_dim -> cam_dim; maps visual features into the camera attention space).
        self.k_proj = nn.Linear(vis_dim, cam_dim)
        self.v_proj = nn.Linear(vis_dim, cam_dim)
        # Output projection.
        self.out_proj = nn.Linear(cam_dim, cam_dim)

        # Pre-Norm (Q in cam_dim, K/V in vis_dim).
        self.norm_q = nn.LayerNorm(cam_dim)
        self.norm_kv = nn.LayerNorm(vis_dim)

        # QK-norm (aligned with VGGT: apply LayerNorm(head_dim) to Q/K before attention
        # to stabilize the attention logits).
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)

        # FFN (cam_dim).
        self.norm_ffn = nn.LayerNorm(cam_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cam_dim, cam_dim * 4),
            nn.GELU(),
            nn.Linear(cam_dim * 4, cam_dim),
        )

        # LayerScale (init=0.01 so the residual contribution starts small).
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
            cam_tokens: (T_total, 1, cam_dim) - camera tokens for all temporal groups.
            vit_features: (total_patches, vis_dim) - the output of a frozen ViT layer.
            frame_sizes: list[int] - number of patches per frame.

        Returns:
            cam_tokens: (T_total, 1, cam_dim) - updated camera tokens.
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

        # Grouped vectorization: bucket by frame_size and run one SDPA per bucket,
        # avoiding per-frame Python loops.
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

        # Residual + LayerScale.
        cam_tokens = cam_tokens + self.ls_attn * attn_output

        # FFN + Residual + LayerScale.
        ffn_out = self.ffn(self.norm_ffn(cam_tokens))
        cam_tokens = cam_tokens + self.ls_ffn * ffn_out

        return cam_tokens


class GlobalCameraSelfAttentionLayer(nn.Module):
    """
    Camera tokens across all frames attend to each other (fully bidirectional self-attention).

    Q = K = V = camera tokens of all frames (including itself).
    Isolation per video: camera tokens from different videos never attend to each other.

    This is the key step for learning inter-frame geometric relationships.
    """

    def __init__(self, cam_dim: int = 2048, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = cam_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cam_dim = cam_dim

        # Standard self-attention QKV.
        self.qkv = nn.Linear(cam_dim, cam_dim * 3)
        self.out_proj = nn.Linear(cam_dim, cam_dim)

        # Pre-Norm.
        self.norm = nn.LayerNorm(cam_dim)

        # QK-norm (aligned with VGGT).
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)

        # FFN.
        self.norm_ffn = nn.LayerNorm(cam_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cam_dim, cam_dim * 4),
            nn.GELU(),
            nn.Linear(cam_dim * 4, cam_dim),
        )

        # LayerScale.
        self.ls_attn = nn.Parameter(torch.ones(cam_dim) * 0.01)
        self.ls_ffn = nn.Parameter(torch.ones(cam_dim) * 0.01)

    def forward(
        self,
        cam_tokens: torch.Tensor,
        video_sizes: List[int],
    ) -> torch.Tensor:
        """
        Args:
            cam_tokens: (T_total, 1, D) - camera tokens for every frame of every video.
            video_sizes: list[int] - number of temporal groups per video.

        Returns:
            cam_tokens: (T_total, 1, D)
        """
        cam_flat = cam_tokens.squeeze(1)  # (T_total, D)
        cam_splits = torch.split(cam_flat, video_sizes, dim=0)

        outputs = []
        for cam_seq in cam_splits:
            S = cam_seq.shape[0]

            # Self-attention (SDPA, numerically more stable and faster).
            normed = self.norm(cam_seq)  # (S, D)
            qkv = self.qkv(normed).reshape(S, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(1, 2, 0, 3)  # (3, H, S, d)
            q, k, v = qkv.unbind(0)  # each: (H, S, d)
            q = self.q_norm(q)  # QK-norm over head_dim
            k = self.k_norm(k)

            out = F.scaled_dot_product_attention(q, k, v)  # (H, S, d)
            out = out.transpose(0, 1).reshape(S, self.cam_dim)  # (S, cam_dim)
            out = self.out_proj(out)

            # Residual + LayerScale.
            cam_seq = cam_seq + self.ls_attn * out

            # FFN.
            ffn_out = self.ffn(self.norm_ffn(cam_seq))
            cam_seq = cam_seq + self.ls_ffn * ffn_out

            outputs.append(cam_seq)

        return torch.cat(outputs, dim=0).unsqueeze(1)  # (T_total, 1, D)


class CameraTokenProjector(nn.Module):
    """
    Project camera tokens from the ViT dimension to the LLM dimension.
    The structure mirrors the two-layer MLP in Qwen3-VL's Merger.
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
    Alternating-attention Camera Token module.

    Structure: [Frame CrossAttn -> Global SelfAttn] x depth.
    Aggregates spatial information from frozen ViT intermediate layers and produces the
    camera token used both for distillation and as an LLM input.

    The structure mirrors VGGT-omega: the camera stream stays on stream_dim (= ViT hidden,
    e.g. 1024 for the 4B ViT) and performs alternating attention. At the last layer, both
    the "post intra-frame cross-attn" and "post inter-frame self-attn" camera representations
    are concatenated into 2*stream_dim (= 2048) as camera_features, matching the segments in
    VGGT's cached torch.cat([frame_tokens, tokens]) (intra-frame half + inter-frame half).

      - camera token / alternating attention / single-branch representation stays at stream_dim (=1024)
      - visual K/V inputs are also stream_dim (ViT, 1024 for 4B); cross-attention is symmetric
      - camera_features = concat(intra-frame branch, inter-frame branch) = 2*stream_dim (=2048),
        directly aligned with VGGT with no extra projection required
      - camera_proj: 2*stream_dim -> llm_hidden_dim, produces the camera embed injected into the LLM

    Args:
        hidden_dim: ViT hidden dimension (K/V input == camera stream dimension; 1024 for 4B).
        num_heads: attention heads.
        depth: number of alternating-attention rounds.
        llm_hidden_dim: LLM hidden dimension (2560 for 4B, 4096 for 8B).
        cam_dim: single-branch camera stream dimension; defaults to hidden_dim.
                 camera_features = 2 * cam_dim.
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
        stream_dim = cam_dim if cam_dim is not None else hidden_dim  # single-branch dim (=1024)
        self.hidden_dim = hidden_dim      # ViT visual dim (K/V)
        self.stream_dim = stream_dim      # single-branch camera stream dim
        self.feature_dim = 2 * stream_dim  # concatenated feature dim (=2048, matches VGGT)
        self.depth = depth

        # Dual camera token (matches VGGT):
        #   [:, 0] = variant for the first temporal group (anchors the world coordinate system)
        #   [:, 1] = variant shared by other temporal groups (expresses relative change)
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, stream_dim))
        nn.init.normal_(self.camera_token, std=1e-3)  # matches VGGT-omega init

        # Frame cross-attention layers (Q = camera stream_dim, K/V = visual hidden_dim).
        self.frame_layers = nn.ModuleList(
            [FrameCrossAttentionLayer(cam_dim=stream_dim, vis_dim=hidden_dim, num_heads=num_heads)
             for _ in range(depth)]
        )

        # Global self-attention layers (stream_dim).
        self.global_layers = nn.ModuleList(
            [GlobalCameraSelfAttentionLayer(cam_dim=stream_dim, num_heads=num_heads) for _ in range(depth)]
        )

        # Camera Token -> LLM projection (2 * stream_dim -> llm_hidden_dim).
        self.camera_proj = CameraTokenProjector(
            in_dim=self.feature_dim,
            hidden_dim=self.feature_dim,
            out_dim=llm_hidden_dim,
        )

        # Intermediate outputs (consumed by the loss).
        self._last_camera_features = None  # (T_total, 2*stream_dim), directly used by distill loss (matches VGGT)
        self._last_camera_embeds = None  # (T_total, llm_hidden_dim) for LLM input
        self._last_video_ids = []
        self._last_video_grid_thw = None

        # Reasonable init: Xavier for Linear, zero bias (better than PyTorch's default kaiming a=sqrt(5)).
        self._init_linear_weights()

    def _init_linear_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def prepare_camera_tokens(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Assign the correct camera-token variant to each temporal group.

        Args:
            grid_thw: (num_videos, 3) tensor - (T, H, W) per video.

        Returns:
            (T_total, 1, hidden_dim) - one camera token per temporal group.
        """
        cam_tokens_list = []
        for t, h, w in grid_thw.tolist():
            t = int(t)
            if t >= 1:
                # The first temporal group uses variant 0.
                first = self.camera_token[:, 0, :, :]  # (1, 1, D)
                cam_tokens_list.append(first.squeeze(0))  # (1, D) -> becomes (1, 1, D) after stacking
            if t > 1:
                # The rest use variant 1.
                others = self.camera_token[:, 1, :, :].expand(1, t - 1, -1)  # (1, t-1, D)
                cam_tokens_list.append(others.squeeze(0))  # (t-1, D)

        # Concatenate camera tokens for all temporal groups.
        all_cam = torch.cat(cam_tokens_list, dim=0)  # (T_total, D)
        return all_cam.unsqueeze(1)  # (T_total, 1, D)

    def forward(
        self,
        vit_layer_outputs: List[torch.Tensor],
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vit_layer_outputs: list of `depth` tensors, each (total_patches, hidden_dim);
                               intermediate features hooked from ViT layers (already detached).
            grid_thw: (num_videos, 3) tensor.

        Returns:
            camera_embeds: (T_total, llm_hidden_dim) - projected, ready to feed into the LLM.
            camera_features: (T_total, 2*stream_dim) - concat(intra-frame branch, inter-frame branch),
                             used by the distillation loss.
        """
        # Compute patches per frame and temporal groups per video.
        frame_sizes = []
        video_sizes = []
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            patches_per_frame = h * w
            video_sizes.append(t)
            for _ in range(t):
                frame_sizes.append(patches_per_frame)

        # Prepare camera tokens.
        cam_tokens = self.prepare_camera_tokens(grid_thw)  # (T_total, 1, stream_dim)

        # Unify dtype: ViT intermediate outputs may be bfloat16 while module weights may be float32.
        target_dtype = self.camera_proj.linear_fc1.weight.dtype
        cam_tokens = cam_tokens.to(target_dtype)

        # Alternating attention; at the last layer keep the "post intra-frame cross-attn" camera
        # representation as the intra-frame branch.
        cam_after_frame = None
        for layer_idx in range(self.depth):
            # Choose the ViT intermediate layer output (reuse the last one if there are not enough).
            vit_idx = min(layer_idx, len(vit_layer_outputs) - 1)
            vit_features = vit_layer_outputs[vit_idx].to(target_dtype)  # (total_patches, hidden_dim)

            # Step A: Frame Cross-Attention (intra-frame, spatially oriented).
            cam_tokens = self.frame_layers[layer_idx](cam_tokens, vit_features, frame_sizes)
            if layer_idx == self.depth - 1:
                cam_after_frame = cam_tokens  # (T_total, 1, stream_dim), intra-frame branch

            # Step B: Global Camera Self-Attention (inter-frame, temporally oriented).
            cam_tokens = self.global_layers[layer_idx](cam_tokens, video_sizes)

        # Dual-branch concat mirrors VGGT's torch.cat([frame_tokens, tokens]).
        cam_frame = cam_after_frame.squeeze(1)   # (T_total, stream_dim), intra-frame half
        cam_global = cam_tokens.squeeze(1)       # (T_total, stream_dim), inter-frame half
        camera_features = torch.cat([cam_frame, cam_global], dim=-1)  # (T_total, 2*stream_dim)

        camera_embeds = self.camera_proj(camera_features)  # (T_total, llm_hidden_dim)

        # Save for use by the loss.
        self._last_camera_features = camera_features
        self._last_camera_embeds = camera_embeds

        return camera_embeds, camera_features


def inject_camera_tokens_into_embeds(
    image_embeds: torch.Tensor,
    camera_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """
    Insert camera_embeds in front of each frame's visual tokens inside image_embeds.

    Before (T temporal groups per frame, H*W patches per group):
        image_embeds = [vis_1, vis_2, ..., vis_{H*W}, vis_1, ..., vis_{H*W}, ...]
        total length = sum(T_i * H_i * W_i)

    After (insert one camera token in front of each temporal group):
        image_embeds = [CAM_1, vis_1, ..., vis_{H*W}, CAM_2, vis_1, ..., vis_{H*W}, ...]
        total length = sum(T_i * (1 + H_i * W_i))

    Args:
        image_embeds: (N_total_vis, llm_dim) - Qwen3-VL visual output (after merger).
        camera_embeds: (T_total, llm_dim) - camera tokens produced by CameraTokenModule.
        grid_thw: (num_videos, 3) - (T, H, W) per video.

    Returns:
        new_image_embeds: (N_total_vis + T_total, llm_dim) - after inserting camera tokens.
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
            # Pick the camera token for this frame.
            cam = camera_embeds[cam_offset].unsqueeze(0)  # (1, llm_dim)
            # Pick the visual tokens for this frame.
            vis = image_embeds[vis_offset: vis_offset + patches_per_frame]  # (H*W, llm_dim)

            # Camera token first, visual tokens after.
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
    Generate M-RoPE position_ids for the camera token of each frame.
    The camera token is placed at the spatial center of its frame.

    Args:
        grid_thw: (num_videos, 3) - (T, H_grid, W_grid) per video.
                  Note: H_grid, W_grid are the post-merge grid sizes (already divided by
                  spatial_merge_size).
        spatial_merge_size: Qwen3-VL's spatial merge size (default 2).

    Returns:
        camera_pos_ids: (3, T_total) - [temporal, height, width] for each camera token.
                        temporal = the frame's temporal position
                        height   = H_grid // 2 (center)
                        width    = W_grid // 2 (center)
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
