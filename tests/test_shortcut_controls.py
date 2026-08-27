"""Tests for the shortcut canaries and the scale-matching fix.

These encode the central claim of the data pipeline: that `min_side` matching
plus fixed-size native cropping removes the size signal, and that the canary
is sensitive enough to notice when it does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.sampling import (
    LABEL_AUTHENTIC,
    LABEL_GENERATED,
    add_size_columns,
    balance_classes,
    min_side_matched_pool,
)
from src.evaluation.shortcut_controls import (
    SHORTCUT_ALARM_AUROC,
    feature_canary,
    resolution_canary,
    scale_canary,
)


# ---------------------------------------------------------------------------
# Canary mechanics
# ---------------------------------------------------------------------------


def test_canary_detects_perfect_separation():
    """The Community Forensics case: reals 1024px, fakes 512px."""
    labels = [0] * 50 + [1] * 50
    widths = [1024] * 50 + [512] * 50
    result = resolution_canary(widths, widths, labels)
    assert result.is_alarming
    assert max(result.auroc, 1 - result.auroc) == pytest.approx(1.0)


def test_canary_is_blind_when_sizes_match():
    labels = [0] * 50 + [1] * 50
    widths = [512] * 100
    result = resolution_canary(widths, widths, labels)
    assert not result.is_alarming
    assert result.auroc == pytest.approx(0.5)


def test_canary_flags_inverted_separation():
    """AUROC 0.02 is exactly as leaky as 0.98 -- only the sign differs."""
    labels = [0] * 50 + [1] * 50
    widths = [256] * 50 + [2048] * 50
    assert resolution_canary(widths, widths, labels).is_alarming


def test_canary_alarm_threshold_boundary():
    """A weak-but-real signal below the threshold should not alarm."""
    rng = np.random.default_rng(0)
    labels = np.array([0] * 500 + [1] * 500)
    # Heavily overlapping distributions -> AUROC near chance.
    sizes = np.concatenate([rng.normal(512, 200, 500), rng.normal(530, 200, 500)])
    result = resolution_canary(sizes, np.ones_like(sizes), labels)
    assert max(result.auroc, 1 - result.auroc) < SHORTCUT_ALARM_AUROC
    assert not result.is_alarming


def test_scale_canary_ignores_aspect_ratio():
    """A 512x768 real and a 512x512 fake differ in pixel count, but a fixed
    256x256 crop from each is indistinguishable in scale. The scale canary
    must see nothing even though the resolution canary fires."""
    labels = [0] * 50 + [1] * 50
    widths = [512] * 100
    heights = [768] * 50 + [512] * 50

    assert resolution_canary(widths, heights, labels).is_alarming
    min_sides = np.minimum(widths, heights)
    assert scale_canary(min_sides, labels).auroc == pytest.approx(0.5)


def test_feature_canary_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        feature_canary([1, 2, 3], [0, 1], "x", "y")


def test_feature_canary_rejects_empty():
    with pytest.raises(ValueError):
        feature_canary([], [], "x", "y")


def test_feature_canary_handles_constant_feature():
    result = feature_canary([7] * 20, [0] * 10 + [1] * 10, "const", "constant")
    assert result.auroc == pytest.approx(0.5)
    assert not result.is_alarming


def test_canary_report_mentions_verdict():
    labels = [0] * 20 + [1] * 20
    widths = [1024] * 20 + [512] * 20
    text = resolution_canary(widths, widths, labels).report()
    assert "SHORTCUT DETECTED" in text
    assert "AUROC" in text


def test_single_class_canary_is_not_alarming():
    """NaN AUROC must not be reported as a shortcut."""
    result = resolution_canary([512] * 10, [512] * 10, [1] * 10)
    assert np.isnan(result.auroc)
    assert not result.is_alarming


# ---------------------------------------------------------------------------
# The scale-matching fix
# ---------------------------------------------------------------------------


def _skewed_frame() -> pd.DataFrame:
    """Reproduces the real pathology: authentic images at one scale, generated
    at another, with a small overlapping region."""
    rows = []
    for i in range(60):  # authentic, large only
        rows.append(
            {"resolution": [1024, 1024], "label": LABEL_AUTHENTIC,
             "model_name": "FFHQ", "image_name": f"a{i}"}
        )
    for i in range(40):  # authentic, overlapping scale, non-square
        rows.append(
            {"resolution": [512, 768], "label": LABEL_AUTHENTIC,
             "model_name": "LandscapesHQ", "image_name": f"b{i}"}
        )
    for g in range(8):  # generated, mostly at the overlapping scale
        for i in range(20):
            rows.append(
                {"resolution": [512, 512], "label": LABEL_GENERATED,
                 "model_name": f"gen/{g}", "image_name": f"g{g}_{i}"}
            )
    for i in range(10):  # a few generated at the large scale
        rows.append(
            {"resolution": [1024, 1024], "label": LABEL_GENERATED,
             "model_name": "gen/big", "image_name": f"gb{i}"}
        )
    return add_size_columns(pd.DataFrame(rows))


def test_skewed_frame_has_a_scale_shortcut_before_matching():
    frame = _skewed_frame()
    assert scale_canary(frame["min_side"], frame["label"]).is_alarming


def test_min_side_matching_removes_the_scale_shortcut():
    pool = add_size_columns(min_side_matched_pool(_skewed_frame(), min_crop_size=256))
    assert len(pool) > 0
    assert scale_canary(pool["min_side"], pool["label"]).auroc == pytest.approx(0.5)


def test_min_side_matching_balances_within_each_bucket():
    pool = add_size_columns(min_side_matched_pool(_skewed_frame(), min_crop_size=256))
    for _, bucket in pool.groupby("min_side"):
        assert (bucket["label"] == LABEL_AUTHENTIC).sum() == (
            bucket["label"] == LABEL_GENERATED
        ).sum()


def test_min_side_matching_keeps_more_data_than_exact_matching():
    """The whole reason for matching on min_side rather than exact resolution:
    it keeps the non-square reals that exact matching throws away."""
    from src.data.sampling import resolution_matched_pool

    frame = _skewed_frame()
    assert len(min_side_matched_pool(frame, 256)) > len(resolution_matched_pool(frame))


def test_min_side_matching_drops_single_class_buckets():
    """A bucket with only one class cannot be balanced without resampling,
    which would fabricate interpolation artifacts."""
    frame = pd.DataFrame(
        [
            {"resolution": [256, 256], "label": LABEL_GENERATED, "model_name": "g"},
            {"resolution": [512, 512], "label": LABEL_AUTHENTIC, "model_name": "a"},
            {"resolution": [512, 512], "label": LABEL_GENERATED, "model_name": "g2"},
        ]
    )
    pool = add_size_columns(min_side_matched_pool(add_size_columns(frame), 256))
    assert set(pool["min_side"]) == {512}


def test_stratified_balancing_preserves_scale_match():
    """Balancing globally re-introduces a scale gap; stratifying does not.

    This is a regression test for a measured failure: unstratified balancing
    moved the scale canary from 0.500 back up to 0.538 on the real dataset.
    """
    pool = add_size_columns(min_side_matched_pool(_skewed_frame(), min_crop_size=256))

    stratified = add_size_columns(
        balance_classes(pool, n_per_class=40, seed=0, stratify_column="min_side")
    )
    assert scale_canary(
        stratified["min_side"], stratified["label"]
    ).auroc == pytest.approx(0.5, abs=0.02)


def test_stratified_balancing_still_balances_classes():
    pool = min_side_matched_pool(_skewed_frame(), min_crop_size=256)
    out = balance_classes(pool, n_per_class=40, seed=0, stratify_column="min_side")
    n_auth = (out["label"] == LABEL_AUTHENTIC).sum()
    n_gen = (out["label"] == LABEL_GENERATED).sum()
    assert n_auth == n_gen


def test_stratified_balancing_handles_missing_column():
    """An unknown stratify column must fall back to global balancing, not crash."""
    pool = min_side_matched_pool(_skewed_frame(), min_crop_size=256)
    out = balance_classes(pool, n_per_class=20, seed=0, stratify_column="nope")
    assert len(out) > 0
