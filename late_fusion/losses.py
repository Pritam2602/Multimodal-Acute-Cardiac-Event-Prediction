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


def build_loss(loss_name: str, pos_weight=None, alpha: float = 1.0, gamma: float = 2.0):
    loss_name = loss_name.lower().strip()

    if loss_name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    raise ValueError(f"Unsupported loss_name='{loss_name}'. Expected 'focal' or 'bce'.")
