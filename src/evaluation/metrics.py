"""Metrics and the robustness matrix.

The organizing principle here comes straight from the NTIRE 2026 robust-AIGC
challenge, where one team scored 0.9954 clean AUROC -- statistically tied with
the winner's 0.9978 -- and finished 9th because its robust AUROC collapsed to
0.8302. Clean accuracy is nearly uninformative about deployment performance.

So this module deliberately makes the *worst case* the headline. `summary()`
reports worst-case AUROC before mean AUROC, and before clean AUROC, because
that ordering is the thesis of the whole project.

Operational note: we also report TPR at fixed low FPR. On a platform, flagging
authentic photographs as AI-generated is the expensive error, so "how much
generated content do we catch while touching at most 1% of real content" is the
number that matters far more than balanced accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

__all__ = [
    "BinaryMetrics",
    "compute_metrics",
    "expected_calibration_error",
    "equal_error_rate",
    "best_balanced_accuracy",
    "RobustnessMatrix",
]


# Operating points we care about. Low FPR because false positives on authentic
# images are the costly error for a content platform.
FPR_TARGETS = (0.01, 0.05, 0.10)


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width-binning ECE.

    Measures whether a score of 0.9 actually means "90% likely AI-generated".
    The deliverable is a confidence score, so miscalibration is a real defect,
    not a cosmetic one.
    """
    if len(y_true) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index per sample; clip so a score of exactly 1.0 lands in the last bin
    # rather than one past the end.
    indices = np.clip(np.digitize(y_score, edges[1:-1], right=False), 0, n_bins - 1)

    total = 0.0
    for bin_index in range(n_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(y_score[mask].mean())
        accuracy = float(y_true[mask].mean())
        total += (count / len(y_true)) * abs(confidence - accuracy)

    return total


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Highest TPR achievable without exceeding `target_fpr`."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    allowed = fpr <= target_fpr
    if not allowed.any():
        return 0.0
    return float(tpr[allowed].max())


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """The rate where FPR and FNR cross.

    Threshold-free like AUROC, but expressed as an error rate, which makes the
    gap against `accuracy` legible: a model with EER 0.24 that scores 0.45
    accuracy at threshold 0.5 is not a weak model, it is a *misthresholded*
    one. That distinction is the whole point of tracking both.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    crossing = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[crossing] + fnr[crossing]) / 2.0)


def best_balanced_accuracy(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float]:
    """Best achievable balanced accuracy and the threshold that achieves it.

    Reported alongside accuracy@0.5 specifically to separate "the scores cannot
    separate these classes" from "the scores separate them fine but 0.5 is the
    wrong cut". Under distribution shift the second is common and is fixable
    without retraining -- see `src.calibration`.

    Note this threshold is fitted ON the data being scored, so it is an
    optimistic upper bound, not an honest operating point. Use it as a
    diagnostic ceiling; fit a real threshold on held-out data for deployment.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    balanced = (tpr + (1.0 - fpr)) / 2.0
    best = int(np.argmax(balanced))

    # sklearn prepends an `inf` threshold representing "predict nothing
    # positive". For a degenerate scorer (every score identical) that point can
    # win the argmax, and `inf` then serializes as the literal `Infinity`,
    # which is NOT valid strict JSON and breaks conforming readers of the
    # metrics file. Fall back to the highest real score, which is the same
    # decision boundary in practice.
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):
        finite = thresholds[np.isfinite(thresholds)]
        threshold = float(finite.max()) if finite.size else float(np.max(y_score))

    return float(balanced[best]), threshold


@dataclass(frozen=True)
class BinaryMetrics:
    """Metrics for one (model, transform) cell of the robustness matrix."""

    auroc: float
    average_precision: float
    accuracy: float
    ece: float
    brier: float
    tpr_at_fpr: Mapping[float, float]
    n_positive: int
    n_negative: int
    # Threshold-independent / threshold-tuned companions to `accuracy`, which
    # is fixed at 0.5. Defaulted so older callers constructing BinaryMetrics
    # directly keep working.
    balanced_accuracy: float = float("nan")
    eer: float = float("nan")
    best_balanced_accuracy: float = float("nan")
    best_threshold: float = float("nan")

    @property
    def threshold_gap(self) -> float:
        """How much accuracy is being lost purely to a misplaced threshold.

        Large gap + high AUROC is the signature of a calibration failure under
        distribution shift, not of a weak representation.
        """
        return self.best_balanced_accuracy - self.balanced_accuracy

    def as_row(self) -> dict[str, float]:
        row = {
            "auroc": self.auroc,
            "ap": self.average_precision,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "best_balanced_accuracy": self.best_balanced_accuracy,
            "best_threshold": self.best_threshold,
            "eer": self.eer,
            "ece": self.ece,
            "brier": self.brier,
        }
        for target, value in self.tpr_at_fpr.items():
            row[f"tpr@fpr{target:g}"] = value
        return row


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute all metrics for one set of predictions.

    `y_true` is 1 for AI-generated, 0 for authentic. `y_score` is P(AI-generated).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
        )
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics on an empty set")

    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())

    # AUROC and AP are undefined with a single class present. That happens for
    # real (e.g. a per-generator slice with no reals), so degrade to NaN rather
    # than crashing a long evaluation sweep.
    single_class = n_positive == 0 or n_negative == 0
    auroc = float("nan") if single_class else float(roc_auc_score(y_true, y_score))
    ap = (
        float("nan")
        if single_class
        else float(average_precision_score(y_true, y_score))
    )

    accuracy = float(((y_score >= threshold).astype(int) == y_true).mean())
    brier = float(np.mean((y_score - y_true) ** 2))
    ece = expected_calibration_error(y_true, y_score)

    tpr_at_fpr = {
        target: (
            float("nan") if single_class else _tpr_at_fpr(y_true, y_score, target)
        )
        for target in FPR_TARGETS
    }

    if single_class:
        balanced = eer = best_balanced = best_threshold = float("nan")
    else:
        predicted = (y_score >= threshold).astype(int)
        tpr = float((predicted[y_true == 1] == 1).mean())
        tnr = float((predicted[y_true == 0] == 0).mean())
        balanced = (tpr + tnr) / 2.0
        eer = equal_error_rate(y_true, y_score)
        best_balanced, best_threshold = best_balanced_accuracy(y_true, y_score)

    return BinaryMetrics(
        auroc=auroc,
        average_precision=ap,
        accuracy=accuracy,
        ece=ece,
        brier=brier,
        tpr_at_fpr=tpr_at_fpr,
        n_positive=n_positive,
        n_negative=n_negative,
        balanced_accuracy=balanced,
        eer=eer,
        best_balanced_accuracy=best_balanced,
        best_threshold=best_threshold,
    )


@dataclass
class RobustnessMatrix:
    """Per-transform metrics for a single model, plus the aggregate summary.

    Cells are keyed by transform name (matching `robustness.eval_grid()`), so
    the matrix lines up one-to-one with what the organizers specified.
    """

    model_name: str
    cells: dict[str, BinaryMetrics] = field(default_factory=dict)

    def add(self, transform_name: str, metrics: BinaryMetrics) -> None:
        self.cells[transform_name] = metrics

    # -- aggregates ---------------------------------------------------------

    @property
    def clean_auroc(self) -> float:
        cell = self.cells.get("clean")
        return float("nan") if cell is None else cell.auroc

    def _degraded_items(self) -> list[tuple[str, BinaryMetrics]]:
        """Every cell except the clean baseline, skipping undefined AUROCs."""
        return [
            (name, cell)
            for name, cell in self.cells.items()
            if name != "clean" and not np.isnan(cell.auroc)
        ]

    @property
    def worst_auroc(self) -> float:
        items = self._degraded_items()
        return float("nan") if not items else min(c.auroc for _, c in items)

    @property
    def worst_transform(self) -> str | None:
        items = self._degraded_items()
        if not items:
            return None
        return min(items, key=lambda kv: kv[1].auroc)[0]

    @property
    def mean_auroc(self) -> float:
        items = self._degraded_items()
        return (
            float("nan")
            if not items
            else float(np.mean([c.auroc for _, c in items]))
        )

    @property
    def relative_degradation(self) -> float:
        """Fractional AUROC lost from clean to worst case.

        This is the number that separated 1st from 9th place at NTIRE 2026, so
        we track it explicitly rather than leaving it to be eyeballed.
        """
        clean = self.clean_auroc
        worst = self.worst_auroc
        if np.isnan(clean) or np.isnan(worst) or clean == 0:
            return float("nan")
        return float((clean - worst) / clean)

    def summary(self) -> dict[str, float | str | None]:
        """Headline numbers, worst-case first. The ordering is deliberate."""
        return {
            "model": self.model_name,
            "worst_auroc": self.worst_auroc,
            "worst_transform": self.worst_transform,
            "mean_auroc": self.mean_auroc,
            "clean_auroc": self.clean_auroc,
            "relative_degradation": self.relative_degradation,
        }

    # -- reporting ----------------------------------------------------------

    def to_markdown(self, order: Iterable[str] | None = None) -> str:
        """Render the matrix as a Markdown table for the README / report."""
        names = list(order) if order is not None else list(self.cells)
        names = [n for n in names if n in self.cells]

        header = "| Transform | AUROC | AP | Acc | TPR@1%FPR | ECE |"
        divider = "|---|---:|---:|---:|---:|---:|"
        lines = [header, divider]

        for name in names:
            cell = self.cells[name]
            tpr = cell.tpr_at_fpr.get(0.01, float("nan"))
            label = f"**{name}**" if name == "clean" else name
            lines.append(
                f"| {label} | {cell.auroc:.4f} | {cell.average_precision:.4f} "
                f"| {cell.accuracy:.4f} | {tpr:.4f} | {cell.ece:.4f} |"
            )

        summary = self.summary()
        lines.append("")
        lines.append(
            f"**Worst-case AUROC {summary['worst_auroc']:.4f}** "
            f"(on `{summary['worst_transform']}`) · "
            f"mean {summary['mean_auroc']:.4f} · "
            f"clean {summary['clean_auroc']:.4f} · "
            f"degradation {summary['relative_degradation']:.1%}"
        )
        return "\n".join(lines)
