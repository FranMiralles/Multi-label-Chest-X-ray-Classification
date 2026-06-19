import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    For multi-label classification with imbalanced classes.
    Penalizes more the false negatives than the false positives.
    """

    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y       * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss    = los_pos + los_neg

        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt  = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        loss *= one_sided_w

        return -loss.mean()
