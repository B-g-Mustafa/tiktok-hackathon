"""Tests for the feature cache and the linear head.

Two behaviours matter enough to pin down:

* The cache refuses a config-hash mismatch. Mixing features extracted under
  different settings yields plausible metrics with no valid interpretation, and
  nothing downstream would reveal the mistake.
* The head L2-normalizes its inputs. Degradation changes activation magnitude
  far more than direction, so an unnormalized head partly learns signal
  strength -- exactly what JPEG and blur destroy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.training.head import FeatureCache, LinearHead, load_cache


def write_cache(path, n=200, dim=32, config_hash="abc123", separable=True):
    rng = np.random.default_rng(0)
    labels = np.array([i % 2 for i in range(n)])
    features = rng.normal(0, 1, (n, dim))
    if separable:
        features[:, 0] += np.where(labels == 1, 2.5, -2.5)

    views = np.array(["clean" if i % 2 == 0 else "jpeg_q30" for i in range(n)])

    np.savez_compressed(
        path,
        features=features.astype(np.float16),
        labels=labels.astype(np.int8),
        view_names=views,
        keys=np.array([f"k{i}" for i in range(n)]),
        generators=np.array(["gen" if y else "real" for y in labels]),
    )
    path.with_suffix(".json").write_text(
        json.dumps({"config_hash": config_hash, "encoder": "test"})
    )
    return path


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_load_cache_roundtrip(tmp_path):
    cache = load_cache(write_cache(tmp_path / "c.npz"))
    assert len(cache) == 200
    assert cache.dim == 32
    assert cache.features.dtype == np.float32  # promoted from stored fp16


def test_cache_hash_mismatch_is_refused(tmp_path):
    path = write_cache(tmp_path / "c.npz", config_hash="abc123")
    with pytest.raises(ValueError, match="config mismatch"):
        load_cache(path, expect_hash="different")


def test_cache_hash_match_is_accepted(tmp_path):
    path = write_cache(tmp_path / "c.npz", config_hash="abc123")
    assert len(load_cache(path, expect_hash="abc123")) == 200


def test_cache_view_selection(tmp_path):
    cache = load_cache(write_cache(tmp_path / "c.npz"))
    clean = cache.clean()
    assert len(clean) == 100
    assert set(clean.view_names) == {"clean"}
    assert sorted(cache.unique_views()) == ["clean", "jpeg_q30"]


def test_cache_view_subset_keeps_alignment(tmp_path):
    """Every parallel array must be filtered together, or labels desync from
    features and the metrics become meaningless."""
    cache = load_cache(write_cache(tmp_path / "c.npz"))
    subset = cache.view("jpeg_q30")
    assert (
        len(subset.features)
        == len(subset.labels)
        == len(subset.keys)
        == len(subset.generators)
    )


def test_cache_missing_view_returns_empty(tmp_path):
    cache = load_cache(write_cache(tmp_path / "c.npz"))
    assert len(cache.view("does_not_exist")) == 0


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


def test_head_learns_a_separable_signal(tmp_path):
    cache = load_cache(write_cache(tmp_path / "c.npz", separable=True))
    head = LinearHead().fit(cache.features, cache.labels)
    from sklearn.metrics import roc_auc_score

    scores = head.predict_proba(cache.features)
    assert roc_auc_score(cache.labels, scores) > 0.95


def test_head_is_near_chance_on_noise(tmp_path):
    """Guards against a head that appears to work through a bug rather than
    through signal."""
    cache = load_cache(write_cache(tmp_path / "c.npz", separable=False))
    head = LinearHead().fit(cache.features, cache.labels)
    from sklearn.metrics import roc_auc_score

    # In-sample on pure noise a linear model overfits somewhat, but held-out
    # random data must be uninformative.
    rng = np.random.default_rng(7)
    held_out = rng.normal(0, 1, (400, cache.dim))
    held_labels = rng.integers(0, 2, 400)
    assert roc_auc_score(
        held_labels, head.predict_proba(held_out)
    ) == pytest.approx(0.5, abs=0.12)


def test_head_normalizes_away_magnitude(tmp_path):
    """Scaling every feature vector must not change its score.

    This is the property that keeps the head stable when degradation attenuates
    activations.
    """
    cache = load_cache(write_cache(tmp_path / "c.npz"))
    head = LinearHead().fit(cache.features, cache.labels)

    original = head.predict_proba(cache.features)
    scaled = head.predict_proba(cache.features * 0.25)
    assert np.allclose(original, scaled, atol=1e-6)


def test_head_handles_zero_feature_row(tmp_path):
    """An all-zero row must not produce NaN via a divide-by-zero norm."""
    cache = load_cache(write_cache(tmp_path / "c.npz"))
    head = LinearHead().fit(cache.features, cache.labels)
    zeros = np.zeros((3, cache.dim), dtype=np.float32)
    assert np.isfinite(head.predict_proba(zeros)).all()


def test_head_rejects_single_class_training(tmp_path):
    cache = load_cache(write_cache(tmp_path / "c.npz"))
    with pytest.raises(ValueError, match="single-class"):
        LinearHead().fit(cache.features, np.ones(len(cache), dtype=int))


def test_head_raises_before_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        LinearHead().predict_proba(np.zeros((2, 8)))


def test_head_parameter_count_is_tiny(tmp_path):
    """The head must be negligible against the 2B budget."""
    cache = load_cache(write_cache(tmp_path / "c.npz"))
    head = LinearHead().fit(cache.features, cache.labels)
    assert head.n_parameters == cache.dim + 1


def test_head_save_and_load_roundtrip(tmp_path):
    cache = load_cache(write_cache(tmp_path / "c.npz"))
    head = LinearHead().fit(cache.features, cache.labels)
    expected = head.predict_proba(cache.features)

    head.save(tmp_path / "head.npz")
    restored = LinearHead.load(tmp_path / "head.npz")
    assert np.allclose(expected, restored.predict_proba(cache.features), atol=1e-6)


def test_head_probabilities_are_in_range(tmp_path):
    cache = load_cache(write_cache(tmp_path / "c.npz"))
    head = LinearHead().fit(cache.features, cache.labels)
    scores = head.predict_proba(cache.features)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
