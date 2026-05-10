import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt).pow(self.gamma) * bce
        return loss.mean()


class OHEMFocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, top_k_ratio: float = 0.7):
        """Keep only the hardest 70% of per-sample losses."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.top_k_ratio = top_k_ratio

    def forward(self, logits, targets):
        targets = targets.float()
        # Compute per-sample focal loss (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * ((1 - pt) ** self.gamma) * bce

        # Keep only top-K hardest samples
        k = max(1, int(self.top_k_ratio * focal.size(0)))
        topk_loss, _ = torch.topk(focal, k)
        return topk_loss.mean()


def build_loss(loss_name: str, pos_weight=None, alpha: float = 1.0, gamma: float = 2.0):
    loss_name = loss_name.lower().strip()

    if loss_name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)

    if loss_name == "ohem_focal":
        # Note: pos_weight is not currently used in OHEM Focal Loss but can be added if needed
        return OHEMFocalLoss(alpha=alpha, gamma=gamma)

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    raise ValueError(f"Unsupported loss_name='{loss_name}'. Expected 'focal', 'ohem_focal', or 'bce'.")
