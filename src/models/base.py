"""Detector interface.

`predict.py` is a required deliverable with a fixed output contract, so it is
written against this interface rather than against any particular model. That
lets the contract -- and its tests -- be finished and frozen before the real
detector exists, and lets the real detector be swapped in without touching the
inference script.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from PIL import Image

__all__ = ["Detector", "ConstantDetector"]


@runtime_checkable
class Detector(Protocol):
    """Anything that scores images for being AI-generated.

    `predict_batch` returns one probability per input image, in the same order.
    Each value is P(AI-generated) in [0, 1].
    """

    name: str

    def predict_batch(self, images: Sequence[Image.Image]) -> list[float]:
        ...


class ConstantDetector:
    """A placeholder that returns a fixed score for every image.

    This is not a joke model -- it is the harness that lets the inference
    contract ship on day one, and it doubles as a genuine evaluation baseline:
    a constant predictor has AUROC 0.5 by construction, so any real model that
    fails to beat it in the harness has a bug in the plumbing rather than in
    its weights.
    """

    def __init__(self, score: float = 0.5, name: str = "constant") -> None:
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        self.score = float(score)
        self.name = name

    def predict_batch(self, images: Sequence[Image.Image]) -> list[float]:
        return [self.score] * len(images)
