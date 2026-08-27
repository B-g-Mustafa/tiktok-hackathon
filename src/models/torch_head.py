"""A trainable classification head that stays attached to the computation
graph -- the LoRA counterpart to `training.head.LinearHead`.

`LinearHead` wraps scikit-learn, which is perfect for the frozen probe (fit on
a static feature matrix, no gradients needed) but structurally cannot be part
of an end-to-end backward pass. This is the same idea -- a single linear layer
over L2-normalized features -- expressed as a `torch.nn.Module` so it trains
jointly with the LoRA adapters in one optimizer step.

Normalization is not cosmetic here either: it is what keeps the head reading
direction rather than activation magnitude, and magnitude is exactly what
degradation (JPEG, blur, noise) attenuates.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["TorchLinearHead"]


class TorchLinearHead(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return self.linear(normalized).squeeze(-1)  # logits

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
