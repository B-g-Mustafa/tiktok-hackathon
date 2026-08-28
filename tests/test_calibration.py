"""Tests for post-hoc score calibration.

The property that matters most here is the NEGATIVE one: calibration must not
change AUROC. If it ever does, the transform has stopped being monotone and
every "calibration improved our numbers" claim built on it becomes a claim
about ranking, which would be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    LogitScaler,
    logits_to_probabilities,
    probabilities_to_logits,
)
from src.evaluation.metrics import compute_metrics


def shifted_scores(n: int = 2000, seed: int = 0):
    """Well-ranked but badly-thresholded scores -- the observed OOD failure.

    Mirrors the real measurement on GenImage: AUROC ~0.84 with accuracy ~0.55,
    because both classes' scores sit below 0.5.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], n // 2)
    logits = np.concatenate(
        [rng.normal(-2.4, 1.0, n // 2), rng.normal(-1.0, 1.0, n // 2)]
    )
    return logits_to_probabilities(logits), labels


# -- the round trip ---------------------------------------------------------


def test_logit_round_trip_is_identity():
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    assert np.allclose(logits_to_probabilities(probabilities_to_logits(p)), p)


def test_saturated_probabilities_do_not_produce_infinities():
    """0.0 and 1.0 are real outputs from a confident model; they must not
    poison the fit with +/-inf."""
    z = probabilities_to_logits(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(z))


def test_sigmoid_is_stable_for_large_negative_logits():
    """The naive 1/(1+exp(-z)) overflows here; the guarded form must not."""
    p = logits_to_probabilities(np.array([-800.0, 800.0]))
    assert np.all(np.isfinite(p))
    assert p[0] == pytest.approx(0.0)
    assert p[1] == pytest.approx(1.0)


# -- the central invariant --------------------------------------------------


def test_calibration_does_not_change_auroc():
    scores, labels = shifted_scores()
    before = compute_metrics(labels, scores)
    after = compute_metrics(
        labels, LogitScaler().fit(scores, labels).transform(scores)
    )
    assert after.auroc == pytest.approx(before.auroc)
    assert after.average_precision == pytest.approx(before.average_precision)


def test_prior_calibration_does_not_change_auroc():
    scores, labels = shifted_scores()
    before = compute_metrics(labels, scores)
    scaler = LogitScaler().fit_to_prior(scores, prior=0.5)
    after = compute_metrics(labels, scaler.transform(scores))
    assert after.auroc == pytest.approx(before.auroc)


# -- the point of the exercise ----------------------------------------------


def test_calibration_recovers_accuracy_on_shifted_scores():
    scores, labels = shifted_scores()
    before = compute_metrics(labels, scores)
    after = compute_metrics(
        labels, LogitScaler().fit(scores, labels).transform(scores)
    )
    assert before.accuracy < 0.65, "fixture should start badly thresholded"
    assert after.accuracy > 0.72
    assert after.ece < before.ece / 2


def test_prior_calibration_needs_no_labels():
    """The label-free path must get most of the supervised gain -- that is what
    makes it usable on a target set nobody has annotated."""
    scores, labels = shifted_scores()
    scaler = LogitScaler().fit_to_prior(scores, prior=0.5)  # labels unused
    after = compute_metrics(labels, scaler.transform(scores))
    assert after.balanced_accuracy > 0.72


def test_prior_calibration_predicts_the_requested_positive_rate():
    scores, _ = shifted_scores()
    scaler = LogitScaler().fit_to_prior(scores, prior=0.3)
    rate = float((scaler.transform(scores) >= 0.5).mean())
    assert rate == pytest.approx(0.3, abs=0.02)


# -- guards -----------------------------------------------------------------


def test_identity_scaler_is_a_no_op():
    scores, _ = shifted_scores()
    assert LogitScaler().is_identity
    assert np.allclose(LogitScaler().transform(scores), scores, atol=1e-9)


def test_supervised_fit_rejects_single_class():
    scores, _ = shifted_scores()
    with pytest.raises(ValueError, match="single-class"):
        LogitScaler().fit(scores, np.zeros(len(scores), dtype=int))


def test_prior_must_be_a_probability():
    scores, _ = shifted_scores()
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="prior"):
            LogitScaler().fit_to_prior(scores, prior=bad)


def test_save_load_round_trip(tmp_path):
    scores, labels = shifted_scores()
    scaler = LogitScaler().fit(scores, labels)
    scaler.save(tmp_path / "calibration.json")

    loaded = LogitScaler.load(tmp_path / "calibration.json")
    assert loaded.a == pytest.approx(scaler.a)
    assert loaded.b == pytest.approx(scaler.b)
    assert np.allclose(loaded.transform(scores), scaler.transform(scores))


def test_load_if_present_falls_back_to_identity(tmp_path):
    """An uncalibrated checkpoint must load as a no-op, not an error -- the
    inference path applies a scaler unconditionally."""
    assert LogitScaler.load_if_present(tmp_path / "nope.json").is_identity


# -- aggregation ------------------------------------------------------------


def test_logit_averaging_preserves_confident_agreement():
    """Probability-space averaging drags a near-unanimous verdict toward the
    middle; logit averaging is what keeps multi-crop inference calibrated."""
    from src.models.siglip_detector import _aggregate

    views = np.array([0.95, 0.95, 0.95, 0.95, 0.05])
    logit_mean = _aggregate(views, LogitScaler())
    assert logit_mean > float(np.mean(views))
