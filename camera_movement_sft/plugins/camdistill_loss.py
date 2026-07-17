"""
CamDistill Loss: Camera Token 蒸馏损失函数

结合标准 SFT Cross-Entropy Loss 和 Camera Token 余弦蒸馏 Loss。
通过 --loss_type camdistill 激活。

VGGT 缓存: 每个视频一个 {video_id}.pt 文件, 内容为 dict:
  {"camera_features": Tensor(S, 2048) float16}
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Optional

from swift.loss import BaseLoss, loss_map

class CamDistillLoss(BaseLoss):
    """
    CamDistill 训练 Loss = L_sft + lambda_cam * L_distill

    L_sft:     标准 next-token prediction cross-entropy
    L_distill: Camera Token 与 VGGT target 的余弦相似度 loss (1 - cos_sim)
    """

    def __init__(self, args=None, trainer=None):
        if args is not None and trainer is not None:
            super().__init__(args, trainer)
        else:
            super().__init__()

        # 蒸馏权重 (常数): total_loss = sft_loss + lambda_cam * distill_loss
        self.lambda_cam = float(os.environ.get("CAMDISTILL_LAMBDA", "0.3"))
        # 前几步关闭蒸馏，先让 LM/SFT 稳定（避免初期蒸馏梯度主导导致不收敛）
        self.lambda_warmup_steps = int(os.environ.get("CAMDISTILL_WARMUP_STEPS", "200"))

        # per_half: camera_features 是 [帧内半 ; 帧间半] concat, 按半各算 cosine 再平均 (默认开),
        #   对齐 VGGT 的两段结构, 避免整段归一化时被范数大的一半主导。
        #   置 0 则对整段 (2048) 整体计算。
        self.per_half = os.environ.get("CAMDISTILL_PER_HALF", "1").strip().lower() in {'1', 'true', 'yes', 'on'}
        self.vggt_cache_dir = os.environ.get("VGGT_CACHE_DIR", "")
        self.strict_cache = os.environ.get('CAMDISTILL_STRICT_CACHE', '1').strip().lower() in {'1', 'true', 'yes', 'on'}

    def _metric_once(
        self,
        pred_aligned: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """在给定的 (N, D) 向量对上算一次 cosine loss = 1 - cos_sim."""
        pred_norm = F.normalize(pred_aligned, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        return (1 - cos_sim).mean(), cos_sim.mean()

    def _compute_distill_loss(
        self,
        pred_aligned: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.per_half:
            return self._metric_once(pred_aligned, target)

        # 按半计算: camera_features = [帧内半 ; 帧间半], 各自对齐 VGGT 对应半再平均。
        d = pred_aligned.shape[-1]
        if d % 2 != 0:
            raise RuntimeError(
                f"[CamDistillLoss] per_half 需要偶数维度, got dim={d}. "
                "设 CAMDISTILL_PER_HALF=0 用整段计算。"
            )
        h = d // 2
        loss_a, cos_a = self._metric_once(pred_aligned[..., :h], target[..., :h])
        loss_b, cos_b = self._metric_once(pred_aligned[..., h:], target[..., h:])
        loss = 0.5 * (loss_a + loss_b)
        cos_sim = 0.5 * (cos_a + cos_b)
        return loss, cos_sim

    @staticmethod
    def _infer_video_sizes(video_grid_thw: torch.Tensor) -> List[int]:
        if video_grid_thw is None:
            return []
        return [int(x) for x in video_grid_thw[:, 0].tolist()]

    @staticmethod
    def _align_target_to_t_model(target: torch.Tensor, t_model: int) -> torch.Tensor:
        """将 cache target 的帧数对齐到模型 temporal group 数。"""
        if t_model <= 0:
            raise RuntimeError(f"[CamDistillLoss] invalid t_model={t_model}")
        if target is None or target.numel() == 0:
            raise RuntimeError('[CamDistillLoss] empty camera target tensor')

        s_vggt = int(target.shape[0])
        if s_vggt == t_model:
            return target
        if s_vggt > t_model:
            if s_vggt == t_model * 2:
                return target.view(t_model, 2, -1).mean(dim=1)
            return F.adaptive_avg_pool1d(target.T.unsqueeze(0), t_model).squeeze(0).T
        return F.interpolate(target.T.unsqueeze(0), size=t_model, mode='nearest').squeeze(0).T

    def _load_aligned_teacher_targets(
        self,
        video_ids: List[str],
        video_grid_thw: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not video_ids or video_grid_thw is None:
            return None
        if not self.vggt_cache_dir:
            if self.strict_cache:
                raise RuntimeError('[CamDistillLoss] VGGT_CACHE_DIR is empty while distillation is enabled.')
            return None

        video_sizes = self._infer_video_sizes(video_grid_thw)
        if len(video_ids) != len(video_sizes):
            raise RuntimeError(
                f"[CamDistillLoss] video_ids/video_grid_thw mismatch: "
                f"len(video_ids)={len(video_ids)} vs num_videos={len(video_sizes)}"
            )

        aligned_targets = []
        for vid, t_model in zip(video_ids, video_sizes):
            target = self.get_vggt_target(vid, device)
            if target is None:
                if self.strict_cache:
                    raise RuntimeError(f"[CamDistillLoss] VGGT cache missing for video_id={vid}")
                return None
            target = target.to(dtype=torch.float32)
            if target.ndim != 2 or target.shape[-1] != 2048:
                raise RuntimeError(
                    f"[CamDistillLoss] invalid target shape for {vid}: {tuple(target.shape)}, expected=(*, 2048)"
                )
            if not torch.isfinite(target).all():
                raise RuntimeError(f"[CamDistillLoss] target contains NaN/Inf for {vid}")
            aligned_targets.append(self._align_target_to_t_model(target, t_model))

        if not aligned_targets:
            return None
        return torch.cat(aligned_targets, dim=0)

    def get_vggt_target(self, video_id: str, device: torch.device) -> Optional[torch.Tensor]:
        """
        从缓存加载 VGGT camera token features。
        每次从磁盘读取，不做内存缓存（1 epoch 每个样本只访问一次）。

        返回: Tensor (S, 2048) 或 None
        """
        if not self.vggt_cache_dir:
            return None

        cache_path = os.path.join(self.vggt_cache_dir, f"{video_id}.pt")
        if not os.path.exists(cache_path):
            return None

        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        # 兼容新旧格式
        if isinstance(data, dict):
            features = data["camera_features"]
        else:
            features = data

        return features.to(device)

    def __call__(
        self,
        outputs,
        labels,
        *,
        num_items_in_batch=None,
        loss_scale=None,
        trainer=None,
        **kwargs,
    ):
        """
        计算总 loss = SFT loss + lambda * distillation loss

        Args:
            outputs: model forward 输出 (含 logits)
            labels: target token ids
            trainer: Seq2SeqTrainer 实例
        """
        # ===== 1. 标准 SFT Loss =====
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 归一化用真实 token 数(非零权重和), 保持 loss 量级可比
        if num_items_in_batch is None:
            num_items_in_batch = (shift_labels != -100).sum().clamp(min=1)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        sft_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ) / num_items_in_batch

        # ===== 2. Camera Token 蒸馏 Loss =====
        global_step = int(getattr(getattr(trainer, 'state', None), 'global_step', 0) or 0)
        if self.lambda_warmup_steps > 0 and global_step < self.lambda_warmup_steps:
            lambda_scale = 0.0
        else:
            lambda_scale = self.lambda_cam

        distill_loss = torch.tensor(0.0, device=sft_loss.device)
        has_distill = False
        cos_sim_mean = None

        # camera 模块: CamDistill 挂在 model.camdistill, CamInject 挂在 model.caminject_adapter
        camera_module = None
        if trainer is not None:
            camera_module = getattr(trainer.model, "camdistill", None) or \
                getattr(trainer.model, "caminject_adapter", None)

        if camera_module is not None:
            camdistill = camera_module
            if hasattr(camdistill, "_last_camera_features") and camdistill._last_camera_features is not None:
                camera_features = camdistill._last_camera_features  # (T_total, cam_dim=2048)

                video_ids = getattr(camdistill, "_last_video_ids", None)
                video_grid_thw = getattr(camdistill, "_last_video_grid_thw", None)

                if lambda_scale > 0 and (not video_ids or video_grid_thw is None):
                    if self.strict_cache:
                        raise RuntimeError(
                            '[CamDistillLoss] missing batch metadata (_last_video_ids/_last_video_grid_thw) '
                            'for distillation loss'
                        )

                if lambda_scale > 0 and video_ids and self.vggt_cache_dir:
                    camera_target = self._load_aligned_teacher_targets(
                        list(video_ids),
                        video_grid_thw,
                        camera_features.device,
                    )
                    if camera_target is None:
                        raise RuntimeError('[CamDistillLoss] failed to build aligned camera targets')
                    if camera_target.shape[0] != camera_features.shape[0]:
                        raise RuntimeError(
                            f"[CamDistillLoss] feature/target length mismatch: "
                            f"pred={camera_features.shape[0]} vs target={camera_target.shape[0]}"
                        )

                    # camera_features 现在直接是 cam_dim=2048, 与 VGGT target 同维, 无需再投影。
                    # 蒸馏损失在 float32 下计算更稳定。
                    camera_pred = camera_features.to(torch.float32)
                    camera_target = camera_target.to(device=camera_pred.device, dtype=torch.float32)
                    if camera_pred.shape[-1] != camera_target.shape[-1]:
                        raise RuntimeError(
                            f"[CamDistillLoss] feature/target dim mismatch: "
                            f"pred_dim={camera_pred.shape[-1]} vs target_dim={camera_target.shape[-1]} "
                            "(camera_features 应为 cam_dim=2048, 与 VGGT 一致)"
                        )
                    distill_loss, cos_sim_mean = self._compute_distill_loss(camera_pred, camera_target)
                    has_distill = True

        # ===== 3. 总 Loss =====
        total_loss = sft_loss + lambda_scale * distill_loss

        # ===== 4. 日志 =====
        if trainer is not None and hasattr(trainer, "custom_metrics"):
            mode = "train" if trainer.model.training else "eval"
            trainer.custom_metrics[mode]["sft_loss"].update(sft_loss.detach())
            if has_distill:
                trainer.custom_metrics[mode]["distill_loss"].update(distill_loss.detach())
                if cos_sim_mean is not None:
                    trainer.custom_metrics[mode]["cos_sim"].update(cos_sim_mean.detach())
            trainer.custom_metrics[mode]["distill_lambda"].update(
                torch.tensor(float(lambda_scale), device=sft_loss.device)
            )

        return total_loss

# 注册到 ms-swift loss_map
loss_map["camdistill"] = CamDistillLoss
