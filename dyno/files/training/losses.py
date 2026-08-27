"""
Loss Functions for Robust AIGC Detection.
Implements:
1. Binary Cross-Entropy with Logits (BCE)
2. Pairwise Consistency Loss (BCE + Cosine Feature Alignment + KL Probability Alignment)
3. Dynamic Warmup Scheduling to prevent output probability collapse
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassificationLoss(nn.Module):
    """Standard Binary Cross-Entropy with optional Label Smoothing."""
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return self.bce(logits, targets)


def symmetric_binary_kl_divergence(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Numerically Stable Symmetric KL Divergence between two binary logit predictions.
    Computes in float32 with log_softmax to prevent FP16 log(0) NaN underflows.
    """
    lp = (logits_p / temperature).float()
    lq = (logits_q / temperature).float()

    # Stack binary logits: [logit_real, logit_fake]
    p_logits_2d = torch.stack([-lp, lp], dim=-1)
    q_logits_2d = torch.stack([-lq, lq], dim=-1)

    log_p = F.log_softmax(p_logits_2d, dim=-1)
    log_q = F.log_softmax(q_logits_2d, dim=-1)

    prob_p = F.softmax(p_logits_2d, dim=-1)
    prob_q = F.softmax(q_logits_2d, dim=-1)

    # PyTorch kl_div: input is log-probabilities, target is probabilities
    kl_p_q = F.kl_div(log_q, prob_p, reduction="batchmean")
    kl_q_p = F.kl_div(log_p, prob_q, reduction="batchmean")

    sym_kl = 0.5 * (kl_p_q + kl_q_p)
    return torch.nan_to_num(sym_kl, nan=0.0, posinf=1.0, neginf=0.0)


class ConsistencyLoss(nn.Module):
    """
    Pairwise Clean-Transformed Consistency Objective with Continuous Step-Wise Warmup:
    L_total = L_bce + beta(step) * L_feat + alpha(step) * L_kl
    """
    def __init__(
        self,
        feat_weight: float = 0.2,      # Tempered weight prevents feature collapse
        kl_weight: float = 0.1,        # Tempered weight prevents 0.5 logit saddle point
        kl_temperature: float = 2.0,    # Soften probabilities for stability
        total_warmup_steps: int = 500,  # Continuous step-based warmup
        warmup_epochs: Optional[int] = None,
    ):
        super().__init__()
        self.feat_weight = feat_weight
        self.kl_weight = kl_weight
        self.kl_temperature = kl_temperature
        self.total_warmup_steps = total_warmup_steps if warmup_epochs is None else max(1, warmup_epochs * 100)

    def get_warmup_factor(self, step: int) -> float:
        """Continuous linear warmup factor in [0.0, 1.0] across individual steps."""
        if self.total_warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, float(step) / float(max(1, self.total_warmup_steps))))

    def forward(
        self,
        clean_out: Tuple[torch.Tensor, torch.Tensor],
        dist_out: Tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        global_step: int = 0,
        current_epoch: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits_clean, feats_clean = clean_out
        logits_dist, feats_dist = dist_out

        if current_epoch is not None and global_step == 0:
            global_step = int(current_epoch * 100)

        # 1. Supervised BCE on both views (FP32 clamped)
        l_clean = torch.clamp(logits_clean.float(), min=-15.0, max=15.0)
        l_dist = torch.clamp(logits_dist.float(), min=-15.0, max=15.0)
        t_f32 = targets.float()
        loss_bce_clean = F.binary_cross_entropy_with_logits(l_clean, t_f32)
        loss_bce_dist = F.binary_cross_entropy_with_logits(l_dist, t_f32)
        loss_bce = 0.5 * (loss_bce_clean + loss_bce_dist)

        # 2. Smooth Step Warmup Factor
        warmup = self.get_warmup_factor(global_step)

        # 3. Feature Cosine Alignment (on normalized embeddings)
        norm_clean = F.normalize(feats_clean, p=2, dim=-1)
        norm_dist = F.normalize(feats_dist, p=2, dim=-1)
        loss_feat = (1.0 - (norm_clean * norm_dist).sum(dim=-1)).mean()

        # 4. Symmetrized Probability KL Divergence
        p_clean = torch.sigmoid(l_clean / self.kl_temperature)
        p_dist = torch.sigmoid(l_dist / self.kl_temperature)
        eps = 1e-6
        p_clean = torch.clamp(p_clean, eps, 1.0 - eps)
        p_dist = torch.clamp(p_dist, eps, 1.0 - eps)

        kl_1 = p_clean * torch.log(p_clean / p_dist) + (1.0 - p_clean) * torch.log((1.0 - p_clean) / (1.0 - p_dist))
        kl_2 = p_dist * torch.log(p_dist / p_clean) + (1.0 - p_dist) * torch.log((1.0 - p_dist) / (1.0 - p_clean))
        loss_kl = 0.5 * (kl_1.mean() + kl_2.mean())

        # Total Loss with smooth step-wise scaling
        alpha = self.kl_weight * warmup
        beta = self.feat_weight * warmup
        loss_total = loss_bce + beta * loss_feat + alpha * loss_kl

        return loss_total, {
            "loss_total": loss_total.item(),
            "loss_bce": loss_bce.item(),
            "loss_bce_clean": loss_bce_clean.item(),
            "loss_bce_dist": loss_bce_dist.item(),
            "loss_feat": loss_feat.item(),
            "loss_kl": loss_kl.item(),
            "warmup": warmup,
        }
