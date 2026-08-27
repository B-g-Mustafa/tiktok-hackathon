"""Tests for the parameter budget and the crop input policy.

The budget test is a build gate: the competition imposes a hard <2B limit, and
the failure mode is silent (an ensemble grows, or a checkpoint drags in a text
tower nobody looks at). The crop tests pin the input policy, which determines
what evidence the model can see at all.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from src.models.budget import PARAMETER_LIMIT, ParameterBudget
from src.models.encoders import ENCODER_CATALOG, FeatureSpec
from src.transforms.crop import multi_crop_views, native_crop, resized_view


def make_image(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, (height, width, 3), dtype=np.uint8), mode="RGB"
    )


# ---------------------------------------------------------------------------
# Parameter budget
# ---------------------------------------------------------------------------


def test_planned_ensemble_fits_the_limit():
    """The configuration we actually intend to ship."""
    budget = ParameterBudget()
    budget.add(
        "SigLIP2-so400m/378 vision tower",
        ENCODER_CATALOG["siglip2-so400m-378"]["params"],
    )
    budget.add(
        "SigLIP2-L/16-384 vision tower",
        ENCODER_CATALOG["siglip2-large-384"]["params"],
    )
    budget.add("linear head", 9_216 + 1, trainable=True)

    budget.check()
    assert budget.total < PARAMETER_LIMIT
    assert budget.utilization < 0.40


def test_budget_rejects_a_breach():
    budget = ParameterBudget().add("oversized", PARAMETER_LIMIT + 1)
    assert not budget.within_limit
    with pytest.raises(ValueError, match="budget exceeded"):
        budget.check()


def test_budget_boundary_is_exclusive():
    """The rule is 'fewer than 2B', so exactly 2B is a violation."""
    assert not ParameterBudget().add("exact", PARAMETER_LIMIT).within_limit
    assert ParameterBudget().add("just under", PARAMETER_LIMIT - 1).within_limit


def test_loading_the_full_siglip_checkpoint_would_breach_more_than_half():
    """Guards the reasoning behind loading the vision tower only.

    The full so400m checkpoint is 1,136,008,498 params; the vision tower is
    428,225,600. Using the former burns 35% of the total budget on a text
    tower that never runs.
    """
    vision_only = 428_225_600
    full_checkpoint = 1_136_008_498
    dead_weight = full_checkpoint - vision_only

    assert dead_weight / PARAMETER_LIMIT > 0.35
    two_tower_ensemble = ParameterBudget()
    two_tower_ensemble.add("a", full_checkpoint).add("b", full_checkpoint)
    assert not two_tower_ensemble.within_limit


def test_budget_tracks_trainable_versus_frozen():
    budget = ParameterBudget()
    budget.add("frozen encoder", 400_000_000, trainable=False)
    budget.add("head", 5_000, trainable=True)
    assert budget.trainable == 5_000
    assert budget.frozen == 400_000_000
    assert budget.total == 400_005_000


def test_budget_rejects_negative_counts():
    with pytest.raises(ValueError):
        ParameterBudget().add("bad", -1)


def test_budget_markdown_includes_total():
    budget = ParameterBudget().add("enc", 428_225_600)
    assert "428,225,600" in budget.to_markdown()


# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------


def test_feature_spec_output_dim():
    spec = FeatureSpec(
        encoder="e", timm_name="t", feature_dim=1152, image_size=378,
        layers=(24, 25, 26),
    )
    # three layers plus the pooled vector
    assert spec.output_dim == 1152 * 4


def test_feature_spec_hash_changes_with_config():
    """Cache filenames embed this hash; a collision would silently mix
    incompatible features."""
    base = FeatureSpec("e", "t", 1152, 378, (24, 25, 26))
    assert base.config_hash() == FeatureSpec("e", "t", 1152, 378, (24, 25, 26)).config_hash()
    assert base.config_hash() != FeatureSpec("e", "t", 1152, 378, (25, 26)).config_hash()
    assert base.config_hash() != FeatureSpec("e", "t", 1152, 224, (24, 25, 26)).config_hash()
    assert base.config_hash() != FeatureSpec("x", "t", 1152, 378, (24, 25, 26)).config_hash()


# ---------------------------------------------------------------------------
# Crop policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(1000, 800), (400, 400), (256, 256)])
def test_native_crop_returns_exact_size(size):
    result = native_crop(make_image(*size), 256)
    assert result.image.size == (256, 256)


def test_native_crop_takes_true_pixels_not_resampled():
    """The crop must be a literal window into the source. If it were resized,
    the high-frequency evidence would be filtered away."""
    image = make_image(512, 512, seed=1)
    result = native_crop(image, 256, mode="center")

    expected = image.crop((128, 128, 384, 384))
    assert np.array_equal(np.asarray(result.image), np.asarray(expected))


def test_center_crop_is_deterministic():
    image = make_image(600, 400)
    a = native_crop(image, 256, mode="center")
    b = native_crop(image, 256, mode="center")
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))


def test_random_crop_varies_and_is_seedable():
    image = make_image(600, 400)
    outputs = [
        np.asarray(native_crop(image, 128, "random", random.Random(s)).image)
        for s in range(6)
    ]
    assert any(not np.array_equal(outputs[0], o) for o in outputs[1:])

    repeat = np.asarray(native_crop(image, 128, "random", random.Random(3)).image)
    assert np.array_equal(outputs[3], repeat)


def test_undersized_image_is_padded_not_upscaled():
    """Upscaling would fabricate interpolation artifacts -- the very signal a
    forensic detector reads. Padding is reported instead."""
    result = native_crop(make_image(100, 100), 256)
    assert result.image.size == (256, 256)
    assert result.pad_fraction > 0.0
    assert result.source_min_side == 100


def test_large_image_reports_no_padding():
    assert native_crop(make_image(800, 600), 256).pad_fraction == 0.0


@pytest.mark.parametrize("size", [(1, 1), (2, 3), (5, 300), (300, 5)])
def test_crop_survives_degenerate_sizes(size):
    result = native_crop(make_image(*size), 64)
    assert result.image.size == (64, 64)


def test_crop_rejects_invalid_arguments():
    image = make_image(100, 100)
    with pytest.raises(ValueError):
        native_crop(image, 0)
    with pytest.raises(ValueError):
        native_crop(image, 64, mode="stretch")


def test_resized_view_covers_whole_image():
    result = resized_view(make_image(1000, 200), 256)
    assert result.image.size == (256, 256)
    assert result.pad_fraction == 0.0


def test_multi_crop_view_count():
    views = multi_crop_views(make_image(800, 600), 128, n_crops=4, include_resized=True)
    assert len(views) == 5
    assert all(v.image.size == (128, 128) for v in views)


def test_multi_crop_without_resized_view():
    views = multi_crop_views(make_image(800, 600), 128, n_crops=3, include_resized=False)
    assert len(views) == 3


def test_multi_crop_first_view_is_the_center_crop():
    """Averaging is an improvement on the centre crop, so the centre crop must
    still be present and first for single-view fallback."""
    image = make_image(800, 600)
    views = multi_crop_views(image, 128, n_crops=3)
    center = native_crop(image, 128, mode="center")
    assert np.array_equal(np.asarray(views[0].image), np.asarray(center.image))


def test_multi_crop_rejects_zero_crops():
    with pytest.raises(ValueError):
        multi_crop_views(make_image(100, 100), 64, n_crops=0)
