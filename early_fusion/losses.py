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



class SoftF1Loss(nn.Module):
    """Differentiable approximation of F1 score as a loss function.
    
    Directly optimizes the metric we evaluate on, resolving the
    cross-entropy vs F1 objective misalignment.
    """
    def __init__(self, beta=1.0, eps=1e-7):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)

        # Soft confusion matrix components
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        beta_sq = self.beta ** 2
        soft_f1 = ((1 + beta_sq) * tp) / ((1 + beta_sq) * tp + beta_sq * fn + fp + self.eps)

        return 1.0 - soft_f1


class CompositeFocalF1Loss(nn.Module):
    """Composite loss: Focal Loss for gradient stability + Soft-F1 for metric alignment.
    
    focal_weight * FocalLoss + f1_weight * SoftF1Loss
    """
    def __init__(self, alpha=0.25, gamma=1.5, pos_weight=None,
                 focal_weight=0.7, f1_weight=0.3, f1_beta=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
        self.soft_f1 = SoftF1Loss(beta=f1_beta)
        self.focal_weight = focal_weight
        self.f1_weight = f1_weight

    def forward(self, logits, targets):
        l_focal = self.focal(logits, targets)
        l_f1 = self.soft_f1(logits, targets)
        return self.focal_weight * l_focal + self.f1_weight * l_f1


def build_loss(loss_name: str, pos_weight=None, alpha: float = 1.0, gamma: float = 2.0):
    loss_name = loss_name.lower().strip()

    if loss_name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)

    if loss_name == "ohem_focal":
        return OHEMFocalLoss(alpha=alpha, gamma=gamma)

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if loss_name == "composite_f1":
        return CompositeFocalF1Loss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)

    raise ValueError(f"Unsupported loss_name='{loss_name}'. Expected 'focal', 'ohem_focal', 'bce', or 'composite_f1'.")
