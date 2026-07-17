"""
VGGT-Direct: 直接使用预提取的 VGGT Camera Token 注入 LLM

与 CamDistill 蒸馏方案的对比:
- CamDistill: 训练一个 CameraTokenModule 从 ViT 中间输出学习 camera token，用 VGGT 做蒸馏 target
- VGGT-Direct: 直接用 VGGT 的 camera token 输出（预提取），只训练投影层

架构:
  Video → VGGT (离线预提取) → camera_token (2048)
                                    ↓
                              [训练时从 cache 加载]
                                    ↓
                              VGGTProjector (可训练, ~8M)
                                    ↓
                              camera_embed (LLM dim)
                                    ↓
                        插入 LLM visual tokens 每帧前 (独立 token)

可训练参数: VGGTProjector (~8M) + LLM (全参或 LoRA)
Loss: 仅标准 SFT Cross-Entropy (无需蒸馏)

与 CamDistill 共享注入逻辑:
  - 使用相同的 _inject_camera_into_video_embeds
  - 使用相同的 _expand_video_placeholders
  - 使用相同的 position_ids 修改 (camera token 在帧中心)
  区别仅在于 camera_embeds 的来源不同
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from vggt_feature_extractor import (
    extract_features_for_video,
    load_vggt_model,
    load_vggt_omega_model,
)

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = str(value).strip().lower()
    return value in {'1', 'true', 'yes', 'y', 'on'}


class VGGTProjector(nn.Module):
    """
    将 VGGT camera token (2048 dim) 投影到 LLM 维度
    结构: LayerNorm -> Linear -> GELU -> Linear
    """

    def __init__(self, vggt_dim: int = 2048, hidden_dim: int = 2048, llm_dim: int = 4096):
        super().__init__()
        self.norm = nn.LayerNorm(vggt_dim)
        self.linear_fc1 = nn.Linear(vggt_dim, hidden_dim)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(hidden_dim, llm_dim)

    def forward(self, vggt_camera_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vggt_camera_tokens: (T_total, 2048) — VGGT 输出的 camera token
        Returns:
            (T_total, llm_dim) — 投影后可直接进 LLM
        """
        x = vggt_camera_tokens.to(self.linear_fc1.weight.dtype)
        x = self.norm(x)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class VGGTDirectCameraAdapter(nn.Module):
    """
    适配器: 对接 CamDistill 的注入管线，但 camera_embeds 来自 VGGT cache 而非 CameraTokenModule。

    与 CameraTokenModule 相同的接口:
        forward(vit_intermediates, video_grid_thw) -> (camera_embeds, camera_features)

    但 vit_intermediates 被忽略，camera tokens 从预提取 cache 加载。

    帧数对齐:
        VGGT cache 存的是原始帧 (S 帧, fps=5, max=100)
        模型中 video_grid_thw 的 T = ceil(S / temporal_patch_size)
        当 temporal_patch_size=2 时, T = S/2, 需要对 VGGT features 做 2帧平均池化
    """

    def __init__(self, llm_dim: int = 4096, vggt_dim: int = 2048, temporal_patch_size: int = 2):
        super().__init__()
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim
        self.temporal_patch_size = temporal_patch_size

        # 每帧 (即每个 Qwen 时间 patch) 注入 1 个 camera token
        # VGGT (2T, 2048) → 两帧平均 → (T, 2048)
        self.tokens_per_frame = 1

        # 可训练的投影层
        self.projector = VGGTProjector(
            vggt_dim=vggt_dim,
            hidden_dim=vggt_dim,
            llm_dim=llm_dim,
        )

        # Cache 目录
        self._cache_dir = ""

        # 数据来源模式: cache 或 online
        self.mode = os.environ.get('VGGT_MODE', 'cache').strip().lower()
        if self.mode not in {'cache', 'online'}:
            raise RuntimeError(f"[VGGT-Direct] unsupported VGGT_MODE={self.mode}, expected 'cache' or 'online'")

        self.teacher_type = os.environ.get('VGGT_TEACHER_TYPE', 'vggt').strip().lower()
        if self.teacher_type not in {'vggt', 'vggt_omega'}:
            raise RuntimeError(
                f"[VGGT-Direct] unsupported VGGT_TEACHER_TYPE={self.teacher_type}, expected 'vggt' or 'vggt_omega'"
            )
        configured_model_path = os.environ.get('VGGT_MODEL_PATH', '').strip()
        if configured_model_path:
            self.vggt_model_path = configured_model_path
        elif self.teacher_type == 'vggt_omega':
            self.vggt_model_path = 'facebook/VGGT-Omega'
        else:
            self.vggt_model_path = 'facebook/VGGT-1B'
        self.online_fps = int(os.environ.get('VGGT_ONLINE_FPS', os.environ.get('FPS', '5')))
        self.online_max_frames = int(os.environ.get('VGGT_ONLINE_MAX_FRAMES', os.environ.get('FPS_MAX_FRAMES', '100')))

        self.vggt_model = None

        # 每次 forward 前由外部设置的视频路径 (仅 online 模式需要)
        self._pending_video_paths: List[str] = []

        # 失败策略/诊断配置
        self.strict_ids = _env_bool('VGGT_DIRECT_STRICT_IDS', default=False)
        self.strict_cache = _env_bool('VGGT_DIRECT_STRICT_CACHE', default=False)
        self.max_miss_ratio = float(os.environ.get('VGGT_DIRECT_MAX_MISS_RATIO', '1.0'))
        self.min_ratio_samples = int(os.environ.get('VGGT_DIRECT_MIN_RATIO_SAMPLES', '64'))
        self.log_every = int(os.environ.get('VGGT_DIRECT_LOG_EVERY', '50'))

        # 每次 forward 前由外部设置的 video_ids (batch 中的视频 ID 列表)
        self._pending_video_ids: List[str] = []

        # 保存最后的输出 (供 loss 或调试使用)
        self._last_camera_embeds = None
        self._last_camera_features = None

        # 运行统计
        self._forward_step = 0
        self._total_videos = 0
        self._total_cache_hits = 0
        self._total_cache_misses = 0
        self._total_missing_ids = 0

        print(
            f"[VGGT-Direct] adapter init: mode={self.mode}, teacher={self.teacher_type}, "
            f"model={self.vggt_model_path}, online_fps={self.online_fps}, online_max_frames={self.online_max_frames}, "
            f"tokens_per_frame={self.tokens_per_frame}"
        )

    @staticmethod
    def _zeros(t_model: int, dim: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(max(t_model, 0), dim, device=device, dtype=torch.float32)

    @staticmethod
    def _normalize_video_ids(video_ids) -> List[str]:
        if video_ids is None:
            return []
        if isinstance(video_ids, str):
            return [video_ids]
        if isinstance(video_ids, (list, tuple)):
            return [x for x in video_ids if isinstance(x, str) and x]
        return []

    @staticmethod
    def _normalize_video_paths(video_paths) -> List[str]:
        if video_paths is None:
            return []
        if isinstance(video_paths, str):
            return [video_paths]
        if isinstance(video_paths, (list, tuple)):
            return [x for x in video_paths if isinstance(x, str) and x]
        return []

    def _maybe_raise(self, should_raise: bool, message: str):
        if should_raise:
            raise RuntimeError(message)
        print(message)

    def _update_and_log_stats(self, batch_total: int, batch_hits: int, batch_misses: int, batch_missing_ids: int):
        self._total_videos += batch_total
        self._total_cache_hits += batch_hits
        self._total_cache_misses += batch_misses
        self._total_missing_ids += batch_missing_ids

        miss_ratio = self._total_cache_misses / max(self._total_videos, 1)
        if (
            self.max_miss_ratio < 1.0
            and self._total_videos >= self.min_ratio_samples
            and miss_ratio > self.max_miss_ratio
        ):
            raise RuntimeError(
                f"[VGGT-Direct] cache miss ratio too high: {miss_ratio:.3f} "
                f"({self._total_cache_misses}/{self._total_videos}), "
                f"threshold={self.max_miss_ratio:.3f}"
            )

        should_log = self._forward_step <= 5 or (self.log_every > 0 and self._forward_step % self.log_every == 0)
        if should_log:
            hit_ratio = self._total_cache_hits / max(self._total_videos, 1)
            print(
                f"[VGGT-Direct] step={self._forward_step} "
                f"batch_videos={batch_total} batch_hits={batch_hits} "
                f"batch_misses={batch_misses} batch_missing_ids={batch_missing_ids} "
                f"total_hit_ratio={hit_ratio:.3f} total_miss_ratio={miss_ratio:.3f}"
            )

    def set_cache_dir(self, cache_dir: str):
        """设置 cache 目录"""
        self._cache_dir = cache_dir
        if cache_dir and not os.path.isdir(cache_dir):
            print(f"[VGGT-Direct] WARNING: cache dir does not exist: {cache_dir}")

    def _ensure_online_model(self, device: torch.device):
        if self.vggt_model is not None:
            return
        device_str = str(device)
        print(
            f"[VGGT-Direct] loading online teacher model: teacher={self.teacher_type}, "
            f"path={self.vggt_model_path}, device={device_str}"
        )
        if self.teacher_type == 'vggt_omega':
            self.vggt_model = load_vggt_omega_model(self.vggt_model_path, device=device_str)
        else:
            self.vggt_model = load_vggt_model(self.vggt_model_path, device=device_str)

    def _extract_online_feature(self, video_path: str, device: torch.device, t_model: int) -> torch.Tensor:
        """在线提取单个视频 VGGT camera feature, 并对齐到 t_model * tokens_per_frame."""
        if not isinstance(video_path, str) or not video_path:
            raise RuntimeError('[VGGT-Direct] online mode received empty video path')
        if not os.path.exists(video_path):
            raise RuntimeError(f"[VGGT-Direct] online video path does not exist: {video_path}")

        self._ensure_online_model(device)

        result = extract_features_for_video(
            self.vggt_model,
            video_path,
            teacher_type=self.teacher_type,
            fps=self.online_fps,
            max_frames=self.online_max_frames,
            device=str(device),
            save_pose=False,
        )
        if result is None or 'camera_features' not in result:
            raise RuntimeError(f"[VGGT-Direct] online extract failed for video: {video_path}")

        feats = result['camera_features']
        if not isinstance(feats, torch.Tensor):
            raise RuntimeError(f"[VGGT-Direct] online extract returned invalid feature type: {type(feats)}")
        feats = feats.to(device=device, dtype=torch.float32)
        if feats.ndim != 2 or feats.shape[-1] != self.vggt_dim:
            raise RuntimeError(
                f"[VGGT-Direct] online feature shape mismatch: got={tuple(feats.shape)}, expected=(*, {self.vggt_dim})"
            )
        if not torch.isfinite(feats).all():
            raise RuntimeError('[VGGT-Direct] online extracted feature contains NaN/Inf')

        s_vggt = int(feats.shape[0])
        if s_vggt <= 0:
            raise RuntimeError('[VGGT-Direct] online extracted empty feature')

        t_out = t_model * self.tokens_per_frame
        if s_vggt == t_out:
            aligned = feats
        elif s_vggt > t_out:
            ratio = s_vggt // t_out
            if ratio >= 1 and s_vggt == t_out * ratio:
                aligned = feats.view(t_out, ratio, -1).mean(dim=1)
            else:
                aligned = F.adaptive_avg_pool1d(feats.T.unsqueeze(0), t_out).squeeze(0).T
        else:
            aligned = F.interpolate(feats.T.unsqueeze(0), size=t_out, mode='nearest').squeeze(0).T

        return aligned

    def _load_and_align_features(
        self,
        video_ids: List[str],
        video_paths: List[str],
        video_grid_thw: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int, int, int]:
        """
        加载 VGGT cache 并对齐帧数到 video_grid_thw 中的 T。

        Args:
            video_ids: 当前 batch 中的视频 ID 列表
            video_grid_thw: (num_videos, 3) — 每个视频的 (T, H, W) grid
            device: 目标设备

        Returns:
            aligned_features: (T_total, 2048) — 对齐后的 VGGT features
        """
        if video_grid_thw is None or video_grid_thw.numel() == 0:
            empty = torch.zeros(0, self.vggt_dim, device=device, dtype=torch.float32)
            return empty, 0, 0, 0

        video_ids = self._normalize_video_ids(video_ids)
        video_paths = self._normalize_video_paths(video_paths)
        num_videos = int(video_grid_thw.shape[0])
        if self.mode == 'cache':
            if len(video_ids) != num_videos:
                msg = (
                    f"[VGGT-Direct] video_ids count mismatch: len(video_ids)={len(video_ids)} "
                    f"vs num_videos={num_videos}"
                )
                self._maybe_raise(self.strict_ids, msg)
        else:
            if len(video_paths) != num_videos:
                raise RuntimeError(
                    f"[VGGT-Direct] online mode requires video_paths per video: "
                    f"len(video_paths)={len(video_paths)} vs num_videos={num_videos}"
                )

        all_features = []
        batch_hits = 0
        batch_misses = 0
        batch_missing_ids = 0

        for i in range(num_videos):
            t_model = int(video_grid_thw[i, 0].item())  # 模型需要的帧数
            if t_model <= 0:
                raise RuntimeError(f"[VGGT-Direct] invalid t_model={t_model} at index={i}")

            if self.mode == 'online':
                video_path = video_paths[i]
                try:
                    aligned = self._extract_online_feature(video_path, device, t_model)
                    batch_hits += 1
                    all_features.append(aligned)
                except Exception as e:
                    batch_misses += 1
                    msg = f"[VGGT-Direct] online extract failed for idx={i}, path={video_path}: {e}"
                    self._maybe_raise(self.strict_cache, msg)
                    all_features.append(self._zeros(t_model * self.tokens_per_frame, self.vggt_dim, device))
                continue

            # cache 模式: 对应 video_id
            if i < len(video_ids):
                vid = video_ids[i]
            else:
                batch_missing_ids += 1
                msg = (
                    f"[VGGT-Direct] missing video_id for video_index={i}/{num_videos}, "
                    f"using zeros with t_model={t_model}"
                )
                self._maybe_raise(self.strict_ids, msg)
                all_features.append(self._zeros(t_model * self.tokens_per_frame, self.vggt_dim, device))
                continue

            if not self._cache_dir:
                batch_misses += 1
                msg = f"[VGGT-Direct] VGGT_CACHE_DIR is empty, cannot load cache for {vid}"
                self._maybe_raise(self.strict_cache, msg)
                all_features.append(self._zeros(t_model * self.tokens_per_frame, self.vggt_dim, device))
                continue

            # 从 cache 加载
            cache_path = os.path.join(self._cache_dir, f"{vid}.pt")
            if not os.path.exists(cache_path):
                batch_misses += 1
                msg = f"[VGGT-Direct] cache not found for {vid}: {cache_path}"
                self._maybe_raise(self.strict_cache, msg)
                all_features.append(self._zeros(t_model * self.tokens_per_frame, self.vggt_dim, device))
                continue

            try:
                data = torch.load(cache_path, map_location='cpu', weights_only=True)
            except Exception as e:
                batch_misses += 1
                msg = f"[VGGT-Direct] failed to load cache for {vid}: {e}"
                self._maybe_raise(self.strict_cache, msg)
                all_features.append(self._zeros(t_model * self.tokens_per_frame, self.vggt_dim, device))
                continue

            if isinstance(data, dict):
                feats = data.get('camera_features')
            else:
                feats = data

            if feats is None:
                batch_misses += 1
                msg = f"[VGGT-Direct] cache for {vid} missing key `camera_features`"
                self._maybe_raise(self.strict_cache, msg)
                all_features.append(self._zeros(t_model * self.tokens_per_frame, self.vggt_dim, device))
                continue

            if not isinstance(feats, torch.Tensor):
                batch_misses += 1
                msg = f"[VGGT-Direct] cache for {vid} is not a tensor: type={type(feats)}"
                self._maybe_raise(self.strict_cache, msg)
                all_features.append(self._zeros(t_model * self.tokens_per_frame, self.vggt_dim, device))
                continue

            if feats.ndim != 2 or feats.shape[-1] != self.vggt_dim:
                batch_misses += 1
                msg = (
                    f"[VGGT-Direct] cache shape mismatch for {vid}: got={tuple(feats.shape)}, "
                    f"expected=(*, {self.vggt_dim})"
                )
                self._maybe_raise(self.strict_cache, msg)
                all_features.append(self._zeros(t_model * self.tokens_per_frame, self.vggt_dim, device))
                continue

            feats = feats.to(device=device, dtype=torch.float32)

            # 防御性检查: VGGT cache 可能存在 NaN/Inf (来自损坏的视频特征提取)
            if not torch.isfinite(feats).all():
                batch_misses += 1
                msg = f"[VGGT-Direct] cache for {vid} contains NaN/Inf"
                self._maybe_raise(self.strict_cache, msg)
                all_features.append(self._zeros(t_model * self.tokens_per_frame, self.vggt_dim, device))
                continue

            s_vggt = feats.shape[0]  # VGGT 原始帧数

            # 边界检查: t_model 必须 > 0
            if s_vggt <= 0:
                batch_misses += 1
                msg = f"[VGGT-Direct] invalid cache shape for {vid}: s_vggt={s_vggt}, t_model={t_model}"
                self._maybe_raise(self.strict_cache, msg)
                all_features.append(self._zeros(t_model * self.tokens_per_frame, self.vggt_dim, device))
                continue

            # 帧数对齐: VGGT 有 S 帧, 模型需要 t_model * tokens_per_frame 帧
            t_out = t_model * self.tokens_per_frame
            if s_vggt == t_out:
                # 完美对齐 (tokens_per_frame=2 时通常命中)
                aligned = feats
            elif s_vggt > t_out:
                # 下采样: 优先精确平均池化, 否则 adaptive avg
                ratio = s_vggt // t_out
                if ratio >= 1 and s_vggt == t_out * ratio:
                    aligned = feats.view(t_out, ratio, -1).mean(dim=1)
                else:
                    # (S, 2048) -> (2048, S) -> adaptive_avg_pool1d -> (2048, t_out) -> (t_out, 2048)
                    aligned = F.adaptive_avg_pool1d(
                        feats.T.unsqueeze(0), t_out
                    ).squeeze(0).T
            else:
                # s_vggt < t_out: 上采样 (少见)
                aligned = F.interpolate(
                    feats.T.unsqueeze(0), size=t_out, mode='nearest'
                ).squeeze(0).T

            batch_hits += 1
            all_features.append(aligned)

        if not all_features:
            empty = torch.zeros(0, self.vggt_dim, device=device, dtype=torch.float32)
            return empty, batch_hits, batch_misses, batch_missing_ids

        return torch.cat(all_features, dim=0), batch_hits, batch_misses, batch_missing_ids

    def forward(
        self,
        vit_intermediates: list,  # 忽略 (保持与 CameraTokenModule 相同的接口)
        video_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        接口与 CameraTokenModule 完全一致。

        Args:
            vit_intermediates: (被忽略) CamDistill 会传 ViT 中间层, VGGT-Direct 不需要
            video_grid_thw: (num_videos, 3) — 每个视频的 (T, H, W) grid

        Returns:
            camera_embeds: (T_total, llm_dim) — 投影后的 camera embeddings
            camera_features: (T_total, 2048) — 原始 VGGT features (用于 loss 或分析)
        """
        self._forward_step += 1
        device = video_grid_thw.device
        expected_t_total = (
            int(video_grid_thw[:, 0].sum().item()) * self.tokens_per_frame
            if video_grid_thw is not None else 0
        )

        # 防御性对齐: 某些推理路径下，插件新挂接的 adapter 可能未随主模型迁移到目标 GPU。
        proj_device = self.projector.linear_fc1.weight.device
        if proj_device != device:
            self.projector.to(device=device)

        pending_video_ids = self._pending_video_ids
        self._pending_video_ids = []
        pending_video_paths = self._pending_video_paths
        self._pending_video_paths = []

        # 加载 VGGT features 并对齐帧数
        camera_features, batch_hits, batch_misses, batch_missing_ids = self._load_and_align_features(
            pending_video_ids,
            pending_video_paths,
            video_grid_thw,
            device,
        )
        if camera_features.shape[0] != expected_t_total:
            raise RuntimeError(
                f"[VGGT-Direct] aligned feature length mismatch: "
                f"got={camera_features.shape[0]}, expected={expected_t_total}"
            )

        self._update_and_log_stats(
            batch_total=int(video_grid_thw.shape[0]),
            batch_hits=batch_hits,
            batch_misses=batch_misses,
            batch_missing_ids=batch_missing_ids,
        )

        # 投影到 LLM 维度
        camera_embeds = self.projector(camera_features)
        if not torch.isfinite(camera_embeds).all():
            raise RuntimeError('[VGGT-Direct] projector output camera_embeds contains NaN/Inf')

        # 保存输出
        self._last_camera_embeds = camera_embeds
        self._last_camera_features = camera_features

        return camera_embeds, camera_features


class VGGTDirectModule(nn.Module):
    """
    [旧接口, 保持向后兼容]
    VGGT-Direct: 管理 VGGT 模型加载和 camera token 提取

    新代码应使用 VGGTDirectCameraAdapter。
    """

    def __init__(self, llm_dim: int = 4096, mode: str = "cache", vggt_model_path: str = "facebook/VGGT-1B"):
        super().__init__()
        self.mode = mode
        self.llm_dim = llm_dim
        self.vggt_dim = 2048

        # 可训练的投影层
        self.projector = VGGTProjector(
            vggt_dim=self.vggt_dim,
            hidden_dim=self.vggt_dim,
            llm_dim=llm_dim,
        )

        # VGGT 模型（online 模式时加载）
        self.vggt_model = None
        self.vggt_model_path = vggt_model_path

        # Cache
        self._cache_dir = ""

        # 保存最后输出
        self._last_camera_embeds = None

    def set_cache_dir(self, cache_dir: str):
        """设置 cache 目录"""
        self._cache_dir = cache_dir

    def forward(self, vggt_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vggt_features: (T_total, 2048) — VGGT camera token features
        Returns:
            camera_embeds: (T_total, llm_dim)
        """
        camera_embeds = self.projector(vggt_features)
        self._last_camera_embeds = camera_embeds
        return camera_embeds
