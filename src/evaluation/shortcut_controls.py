"""Shortcut controls: is the model learning forensics, or reading the label off
a dataset artifact?

Run these BEFORE believing any accuracy number.

A detector can reach ~100% on a benchmark without extracting a single forensic
signal, simply because the two classes differ in some incidental property --
image dimensions, file format, compression history. Such a model scores
beautifully in-distribution and collapses completely on the hidden test set,
which is exactly the failure mode this project exists to avoid.

The central tool here is the *canary*: a deliberately crippled classifier that
sees ONLY the suspect artifact and none of the image content. If the canary
scores well, the split is broken and no amount of model work will fix it.

Concretely, for Community Forensics the canary is decisive: real images are
1024x1024 (FFHQ) while generated images are 512x512, so image dimensions alone
almost perfectly separate the classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.metrics import BinaryMetrics, compute_metrics

__all__ = [
    "CanaryResult",
    "feature_canary",
    "resolution_canary",
    "scale_canary",
    "SHORTCUT_ALARM_AUROC",
]

# A canary above this AUROC means the artifact alone carries most of the label.
# 0.60 is deliberately conservative: even a "weak" shortcut gets amplified by a
# high-capacity model, and anything above chance is worth understanding.
SHORTCUT_ALARM_AUROC = 0.60


@dataclass
class CanaryResult:
    """Outcome of a shortcut probe.

    `metrics` are computed from a feature that contains NO image content, so
    any signal at all is leakage by construction.
    """

    name: str
    feature_description: str
    metrics: BinaryMetrics

    @property
    def auroc(self) -> float:
        return self.metrics.auroc

    @property
    def is_alarming(self) -> bool:
        """True when the artifact alone separates the classes.

        AUROC is symmetric about 0.5 for our purposes: a canary at 0.02 is just
        as leaky as one at 0.98, it merely has the sign flipped.
        """
        if np.isnan(self.auroc):
            return False
        return max(self.auroc, 1.0 - self.auroc) >= SHORTCUT_ALARM_AUROC

    def report(self) -> str:
        effective = max(self.auroc, 1.0 - self.auroc) if not np.isnan(self.auroc) else float("nan")
        verdict = (
            "SHORTCUT DETECTED -- this split is not measuring forensics"
            if self.is_alarming
            else "no strong shortcut detected"
        )
        return (
            f"[{self.name}] {verdict}\n"
            f"  feature : {self.feature_description}\n"
            f"  AUROC   : {self.auroc:.4f} (effective {effective:.4f})\n"
            f"  samples : {self.metrics.n_positive} generated / "
            f"{self.metrics.n_negative} authentic"
        )


def feature_canary(
    values: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    name: str,
    description: str,
) -> CanaryResult:
    """Can a single scalar feature -- containing no image content -- predict
    the label?

    The feature is min-max normalized into [0, 1] purely so it can go through
    the standard metrics path. AUROC is rank-based, so the normalization has no
    effect on the result.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=int).ravel()

    if len(values) != len(labels):
        raise ValueError("values and labels must be the same length")
    if len(labels) == 0:
        raise ValueError("cannot run a canary on an empty sample")

    spread = values.max() - values.min()
    if spread == 0:
        # The feature is constant: it cannot carry any signal at all.
        scores = np.full_like(values, 0.5)
    else:
        scores = (values - values.min()) / spread

    return CanaryResult(
        name=name,
        feature_description=description,
        metrics=compute_metrics(labels, scores),
    )


def resolution_canary(
    widths: np.ndarray | list[int],
    heights: np.ndarray | list[int],
    labels: np.ndarray | list[int],
) -> CanaryResult:
    """Can total image size alone predict the label?

    This is the gate for any pipeline that feeds WHOLE images to the model,
    where total pixel count and aspect ratio are both observable.
    """
    widths = np.asarray(widths, dtype=np.float64).ravel()
    heights = np.asarray(heights, dtype=np.float64).ravel()

    if len(widths) != len(heights):
        raise ValueError("widths and heights must be the same length")

    return feature_canary(
        widths * heights,
        labels,
        name="resolution-only",
        description="image pixel count (width x height); no image content",
    )


def scale_canary(
    min_sides: np.ndarray | list[int], labels: np.ndarray | list[int]
) -> CanaryResult:
    """Can image SCALE alone predict the label, after fixed-size cropping?

    This is the gate that actually matters for our pipeline, and it is stricter
    than it first appears.

    Once we take a fixed NxN native-resolution crop, the model can no longer
    observe total pixel count or aspect ratio -- a 256x256 crop looks identical
    whether it came from a 512x512 or a 512x768 source. What *does* survive
    cropping is scale: the shorter side determines how much of the scene an NxN
    window covers, so a crop from a 1024px image is effectively a 2x zoom
    relative to the same crop from a 512px image.

    Matching `min_side` between the classes is therefore the right invariant
    for a crop-based detector, and this canary is the check that it holds.
    """
    return feature_canary(
        np.asarray(min_sides, dtype=np.float64).ravel(),
        labels,
        name="scale-only",
        description=(
            "image min(width, height); the only size cue that survives "
            "fixed-size native cropping"
        ),
    )
