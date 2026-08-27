"""Tests for the robustness transform suite.

The suite is the single source of truth for both training augmentation and the
reported robustness matrix, so a bug here silently corrupts every number in the
submission. These tests focus on the cases that actually bite: degenerate image
sizes, non-square aspect ratios, and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.transforms.robustness import (
    TrainAugment,
    center_crop,
    color_jitter,
    downscale_upscale,
    eval_grid,
    eval_grid_combinations,
    gaussian_blur,
    gaussian_noise,
    jpeg,
)


def make_image(width: int = 64, height: int = 64, seed: int = 0) -> Image.Image:
    """A deterministic noise image. Noise (rather than a flat colour) means
    transforms that low-pass filter actually change the pixels, so tests can
    tell a no-op from a real operation."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


# Sizes chosen to break naive implementations: a 1px image, an image smaller
# than a typical crop, and strongly non-square aspect ratios. Tiny-GenImage
# genuinely contains images as narrow as 34px.
EDGE_SIZES = [(1, 1), (3, 7), (34, 34), (17, 200), (200, 17), (64, 64)]


@pytest.mark.parametrize("size", EDGE_SIZES)
def test_eval_grid_survives_edge_sizes(size):
    """Every transform must return a usable RGB image for any input size."""
    width, height = size
    image = make_image(width, height)

    for transform in eval_grid() + eval_grid_combinations():
        out = transform(image)
        assert isinstance(out, Image.Image), transform.name
        assert out.width >= 1 and out.height >= 1, transform.name
        # Everything downstream assumes RGB; a transform silently returning
        # L or CMYK would blow up much later in the pipeline.
        assert out.convert("RGB").mode == "RGB", transform.name


def test_transform_names_are_unique():
    """Names are used as column keys in the robustness matrix, so collisions
    would silently overwrite results."""
    names = [t.name for t in eval_grid() + eval_grid_combinations()]
    assert len(names) == len(set(names))


def test_clean_transform_is_identity():
    image = make_image()
    grid = {t.name: t for t in eval_grid()}
    assert np.array_equal(np.asarray(grid["clean"](image)), np.asarray(image))


def test_center_crop_dimensions():
    image = make_image(100, 50)
    out = center_crop(image, 0.8)
    assert out.size == (80, 40)


def test_center_crop_never_returns_empty():
    """80% of a 1px image rounds to 0 without a floor; that would crash PIL."""
    out = center_crop(make_image(1, 1), 0.8)
    assert out.size == (1, 1)


def test_downscale_upscale_preserves_size():
    """The transform models thumbnail->redisplay, so the output must be the
    same size as the input; only detail is lost."""
    image = make_image(100, 60)
    for scale in (0.5, 0.25):
        assert downscale_upscale(image, scale).size == image.size


def test_downscale_upscale_actually_destroys_detail():
    """Guards against a silent no-op: a 0.25x round trip through bicubic must
    measurably blur a noise image."""
    image = make_image(64, 64)
    out = downscale_upscale(image, 0.25)
    diff = np.abs(
        np.asarray(out, dtype=np.float32) - np.asarray(image, dtype=np.float32)
    ).mean()
    assert diff > 1.0


def test_jpeg_quality_monotonicity():
    """Lower JPEG quality must distort more. If this fails, the quality
    parameter isn't reaching the encoder."""
    image = make_image(64, 64)
    reference = np.asarray(image, dtype=np.float32)

    errors = []
    for quality in (90, 70, 50, 30):
        out = np.asarray(jpeg(image, quality), dtype=np.float32)
        errors.append(np.abs(out - reference).mean())

    assert errors == sorted(errors), f"distortion not monotonic: {errors}"


def test_jpeg_output_is_decoded_eagerly():
    """jpeg() round-trips through a BytesIO that goes out of scope. If the
    image were left lazy, pixel access would fail later."""
    out = jpeg(make_image(), 50)
    assert out.getpixel((0, 0)) is not None


def test_gaussian_noise_is_seeded():
    image = make_image()
    import random as _random

    a = gaussian_noise(image, 0.05, _random.Random(123))
    b = gaussian_noise(image, 0.05, _random.Random(123))
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_gaussian_noise_changes_image():
    image = make_image()
    import random as _random

    out = gaussian_noise(image, 0.10, _random.Random(0))
    assert not np.array_equal(np.asarray(out), np.asarray(image))


def test_gaussian_noise_stays_in_range():
    """Noise must be clipped, not wrapped. A wrap would turn bright pixels
    black and create an artifact far larger than the signal we detect."""
    image = Image.new("RGB", (32, 32), (250, 250, 250))
    import random as _random

    out = np.asarray(gaussian_noise(image, 0.5, _random.Random(0)))
    assert out.min() >= 0 and out.max() <= 255


def test_color_jitter_identity_at_unit_factors():
    image = make_image()
    out = color_jitter(image, 1.0, 1.0, 1.0)
    assert np.array_equal(np.asarray(out), np.asarray(image))


def test_blur_increases_with_sigma():
    image = make_image(64, 64)
    reference = np.asarray(image, dtype=np.float32)
    errors = [
        np.abs(
            np.asarray(gaussian_blur(image, s), dtype=np.float32) - reference
        ).mean()
        for s in (0.5, 1.0, 2.0)
    ]
    assert errors == sorted(errors), f"blur not monotonic: {errors}"


def test_transforms_do_not_mutate_input():
    """Augmentation runs inside a DataLoader over shared/cached buffers; an
    in-place transform would corrupt other samples."""
    image = make_image()
    before = np.asarray(image).copy()

    for transform in eval_grid() + eval_grid_combinations():
        transform(image)

    assert np.array_equal(np.asarray(image), before)


def test_train_augment_is_reproducible():
    image = make_image()
    a = TrainAugment(seed=42)(image)
    b = TrainAugment(seed=42)(image)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_train_augment_produces_variety():
    """A degenerate augmenter that always returns the input would train a
    non-robust model while looking fine."""
    image = make_image()
    augment = TrainAugment(seed=0)
    outputs = [augment(image) for _ in range(30)]
    changed = sum(
        1 for o in outputs if not np.array_equal(np.asarray(o), np.asarray(image))
    )
    # With severity 0 weighted at 0.15, most samples should be degraded.
    assert changed > 15


@pytest.mark.parametrize("size", EDGE_SIZES)
def test_train_augment_survives_edge_sizes(size):
    width, height = size
    augment = TrainAugment(seed=7)
    image = make_image(width, height)
    for _ in range(20):
        out = augment(image)
        assert out.width >= 1 and out.height >= 1


def test_train_augment_rejects_bad_weights():
    with pytest.raises(ValueError):
        TrainAugment(severity_weights=(0.5, 0.5))
