import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLossWrapper(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        targets = targets.float()
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean",
            pos_weight=self.pos_weight,
        )


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


class HybridBCELoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        pos_weight=None,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce_loss = BCEWithLogitsLossWrapper(pos_weight=pos_weight)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return (self.bce_weight * bce) + (self.focal_weight * focal)
