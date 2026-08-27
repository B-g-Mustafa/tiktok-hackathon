"""Tests for the metrics and robustness matrix.

Every number in the final submission flows through this module, so the tests
pin down the properties we actually rely on: correct handling of degenerate
slices, and an aggregation that genuinely surfaces the worst case rather than
letting a good clean score hide a collapse.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    BinaryMetrics,
    RobustnessMatrix,
    compute_metrics,
    expected_calibration_error,
)


def test_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.01, 0.02, 0.98, 0.99])
    m = compute_metrics(y_true, y_score)
    assert m.auroc == pytest.approx(1.0)
    assert m.average_precision == pytest.approx(1.0)
    assert m.accuracy == pytest.approx(1.0)
    assert m.tpr_at_fpr[0.01] == pytest.approx(1.0)


def test_inverted_predictions_score_zero_auroc():
    """A detector with the sign flipped should be obvious, not average out."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.99, 0.98, 0.02, 0.01])
    assert compute_metrics(y_true, y_score).auroc == pytest.approx(0.0)


def test_random_scores_give_chance_auroc():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=4000)
    y_score = rng.random(4000)
    assert compute_metrics(y_true, y_score).auroc == pytest.approx(0.5, abs=0.05)


def test_single_class_returns_nan_not_crash():
    """Per-generator slices legitimately contain only one class. A long
    evaluation sweep must degrade to NaN rather than dying partway through."""
    m = compute_metrics(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
    assert np.isnan(m.auroc)
    assert np.isnan(m.average_precision)
    assert np.isnan(m.tpr_at_fpr[0.01])
    # Threshold metrics remain well defined.
    assert not np.isnan(m.accuracy)
    assert not np.isnan(m.brier)


def test_class_counts_reported():
    m = compute_metrics(np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.3, 0.9]))
    assert m.n_positive == 1
    assert m.n_negative == 3


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0.5]))


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        compute_metrics(np.array([]), np.array([]))


def test_brier_is_zero_for_exact_predictions():
    y_true = np.array([0, 1, 0, 1])
    assert compute_metrics(y_true, y_true.astype(float)).brier == pytest.approx(0.0)


def test_ece_zero_when_perfectly_calibrated():
    """Half the samples at score 1.0 are positive and half at 0.0 are negative:
    confidence matches accuracy exactly in every occupied bin."""
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(y_true, y_score) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    """Scores of 0.99 on samples that are only 50% positive should register a
    large calibration error."""
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.99, 0.99, 0.99, 0.99])
    assert expected_calibration_error(y_true, y_score) == pytest.approx(0.49, abs=0.02)


def test_ece_handles_score_of_exactly_one():
    """A score of exactly 1.0 must land in the final bin, not overflow it."""
    ece = expected_calibration_error(np.array([1, 1]), np.array([1.0, 1.0]))
    assert ece == pytest.approx(0.0)


def test_tpr_at_fpr_respects_budget():
    """With scores overlapping, TPR at 1% FPR must be strictly below 1."""
    rng = np.random.default_rng(1)
    y_true = np.concatenate([np.zeros(500), np.ones(500)])
    y_score = np.concatenate([rng.normal(0.4, 0.15, 500), rng.normal(0.6, 0.15, 500)])
    y_score = np.clip(y_score, 0, 1)
    m = compute_metrics(y_true, y_score)
    assert 0.0 < m.tpr_at_fpr[0.01] < 1.0
    # A looser FPR budget can never catch less.
    assert m.tpr_at_fpr[0.10] >= m.tpr_at_fpr[0.01]


# ---------------------------------------------------------------------------
# Robustness matrix
# ---------------------------------------------------------------------------


def _cell(auroc: float) -> BinaryMetrics:
    return BinaryMetrics(
        auroc=auroc,
        average_precision=auroc,
        accuracy=auroc,
        ece=0.0,
        brier=0.0,
        tpr_at_fpr={0.01: auroc, 0.05: auroc, 0.10: auroc},
        n_positive=10,
        n_negative=10,
    )


def test_matrix_surfaces_worst_case():
    """The whole point of the project: a near-perfect clean score must not
    hide a collapse under one transform."""
    matrix = RobustnessMatrix("shallow-real-lookalike")
    matrix.add("clean", _cell(0.9954))
    matrix.add("jpeg_q30", _cell(0.8302))
    matrix.add("blur_s2.0", _cell(0.95))

    assert matrix.clean_auroc == pytest.approx(0.9954)
    assert matrix.worst_auroc == pytest.approx(0.8302)
    assert matrix.worst_transform == "jpeg_q30"
    assert matrix.relative_degradation == pytest.approx(0.1660, abs=1e-3)


def test_clean_excluded_from_worst_and_mean():
    """Clean is the baseline, not a degradation. Including it would flatter
    the mean and could even become the reported worst case."""
    matrix = RobustnessMatrix("m")
    matrix.add("clean", _cell(0.50))  # deliberately worse than the transforms
    matrix.add("jpeg_q30", _cell(0.90))
    matrix.add("blur_s2.0", _cell(0.80))

    assert matrix.worst_auroc == pytest.approx(0.80)
    assert matrix.mean_auroc == pytest.approx(0.85)


def test_nan_cells_are_skipped_in_aggregates():
    matrix = RobustnessMatrix("m")
    matrix.add("clean", _cell(0.99))
    matrix.add("jpeg_q30", _cell(0.90))
    matrix.add("broken", _cell(float("nan")))
    assert matrix.worst_auroc == pytest.approx(0.90)
    assert not np.isnan(matrix.mean_auroc)


def test_empty_matrix_degrades_gracefully():
    matrix = RobustnessMatrix("empty")
    assert np.isnan(matrix.worst_auroc)
    assert np.isnan(matrix.clean_auroc)
    assert matrix.worst_transform is None
    assert np.isnan(matrix.relative_degradation)


def test_summary_leads_with_worst_case():
    """Key order encodes the thesis; the report renders it in this order."""
    matrix = RobustnessMatrix("m")
    matrix.add("clean", _cell(0.99))
    matrix.add("jpeg_q30", _cell(0.80))
    keys = list(matrix.summary().keys())
    assert keys.index("worst_auroc") < keys.index("clean_auroc")


def test_to_markdown_renders_all_cells():
    matrix = RobustnessMatrix("m")
    matrix.add("clean", _cell(0.99))
    matrix.add("jpeg_q30", _cell(0.80))
    out = matrix.to_markdown()
    assert "clean" in out and "jpeg_q30" in out
    assert "Worst-case AUROC 0.8000" in out


def test_to_markdown_respects_explicit_order():
    matrix = RobustnessMatrix("m")
    matrix.add("clean", _cell(0.99))
    matrix.add("jpeg_q30", _cell(0.80))
    out = matrix.to_markdown(order=["jpeg_q30", "clean"])
    assert out.index("jpeg_q30") < out.index("**clean**")
