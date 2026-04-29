"""Loss functions used for imbalance-aware classifier baselines."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    With gamma=0 and no alpha weighting this is equivalent to cross entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", alpha.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - target_probs) ** self.gamma) * target_log_probs

        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).gather(0, targets)

        return loss.mean()
