"""The organizer-specified robustness transform suite.

This module is the SINGLE SOURCE OF TRUTH for the post-processing operations
our detector must survive. It is used in two places:

  * evaluation  -- the exact parameter grid the organizers specified, used to
                   build the robustness matrix.
  * training    -- randomized augmentation drawn from the same families, so the
                   model learns invariance to the operations it will face.

Keeping both on one implementation means a transform can never silently drift
between what we train against and what we report against.

Everything here is pure PIL + numpy on purpose: no torch import, so the suite
stays fast, testable, and usable from any context (including data prep on a
machine without a GPU stack).

Note on the train/eval boundary: `eval_grid()` returns the fixed, named
parameterizations the organizers listed. `TrainAugment` samples *continuously*
from the surrounding ranges instead of snapping to those exact values, so we
never evaluate on the precise parameter settings we trained on.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

__all__ = [
    "Transform",
    "eval_grid",
    "eval_grid_combinations",
    "TrainAugment",
    "jpeg",
    "gaussian_blur",
    "downscale_upscale",
    "gaussian_noise",
    "color_jitter",
    "center_crop",
]


# ---------------------------------------------------------------------------
# Primitive operations
#
# Each takes a PIL RGB image and returns a new PIL RGB image. They are written
# to be safe on tiny images (a 34px-wide image exists in Tiny-GenImage) and to
# never mutate their input.
# ---------------------------------------------------------------------------


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode through JPEG at the given quality.

    Models the social-media / messaging re-encode. We round-trip through an
    in-memory buffer rather than touching disk.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    # load() forces the decode now, so the caller isn't holding a lazy handle
    # onto a BytesIO we're about to drop.
    out = Image.open(buffer)
    out.load()
    return out


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur. Models an out-of-focus capture."""
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def downscale_upscale(image: Image.Image, scale: float) -> Image.Image:
    """Downscale by `scale` then upscale back to the original size.

    Models thumbnail generation followed by redisplay. This is the single most
    destructive transform for high-frequency forensic artifacts, which is
    exactly why it is in the suite.
    """
    width, height = image.size
    small_w = max(1, int(round(width * scale)))
    small_h = max(1, int(round(height * scale)))
    small = image.resize((small_w, small_h), Image.BICUBIC)
    return small.resize((width, height), Image.BICUBIC)


def gaussian_noise(
    image: Image.Image, sigma: float, rng: random.Random | None = None
) -> Image.Image:
    """Additive Gaussian noise. `sigma` is on the [0, 1] intensity scale.

    Models low-light sensor noise.
    """
    seed = None if rng is None else rng.randrange(2**32)
    generator = np.random.default_rng(seed)

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noisy = array + generator.normal(0.0, float(sigma), array.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).round().astype(np.uint8), mode="RGB")


def color_jitter(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> Image.Image:
    """Brightness / contrast / saturation scaling. Models filter apps and
    auto-enhance. Factors are multipliers, so 1.2 == +20%."""
    out = ImageEnhance.Brightness(image).enhance(float(brightness))
    out = ImageEnhance.Contrast(out).enhance(float(contrast))
    out = ImageEnhance.Color(out).enhance(float(saturation))
    return out


def center_crop(image: Image.Image, fraction: float) -> Image.Image:
    """Centre-crop to `fraction` of each side.

    `fraction=0.8` keeps 80% of the width and 80% of the height (64% of the
    area). Models profile-picture cropping and reframing.
    """
    width, height = image.size
    new_w = max(1, int(round(width * fraction)))
    new_h = max(1, int(round(height * fraction)))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


# ---------------------------------------------------------------------------
# The evaluation grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transform:
    """A named, reproducible transform used to build the robustness matrix.

    `family` groups parameterizations (all four JPEG qualities share the "jpeg"
    family) so we can report both per-setting and per-family degradation.
    """

    name: str
    family: str
    fn: Callable[[Image.Image], Image.Image]

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.fn(image)


# Exact settings named in the problem statement. Do not edit these to make
# numbers look better -- they define what we are scored against.
JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER_DELTA = 0.2
CENTER_CROP_FRACTION = 0.8


def eval_grid(noise_seed: int = 0) -> list[Transform]:
    """The organizer transform grid, plus an identity 'clean' baseline.

    `noise_seed` fixes the noise realizations so the matrix is reproducible
    across runs and comparable across models.
    """
    grid: list[Transform] = [
        Transform("clean", "clean", lambda im: im)
    ]

    for quality in JPEG_QUALITIES:
        grid.append(
            Transform(
                f"jpeg_q{quality}",
                "jpeg",
                lambda im, q=quality: jpeg(im, q),
            )
        )

    for sigma in BLUR_SIGMAS:
        grid.append(
            Transform(
                f"blur_s{sigma}",
                "blur",
                lambda im, s=sigma: gaussian_blur(im, s),
            )
        )

    for scale in RESIZE_SCALES:
        grid.append(
            Transform(
                f"resize_{scale}x",
                "resize",
                lambda im, s=scale: downscale_upscale(im, s),
            )
        )

    for index, sigma in enumerate(NOISE_SIGMAS):
        grid.append(
            Transform(
                f"noise_s{sigma}",
                "noise",
                lambda im, s=sigma, i=index: gaussian_noise(
                    im, s, random.Random(noise_seed + i)
                ),
            )
        )

    # The spec says brightness/contrast/saturation +/-20%. We evaluate both
    # directions so a model can't pass by being invariant in one direction only.
    lo = 1.0 - COLOR_JITTER_DELTA
    hi = 1.0 + COLOR_JITTER_DELTA
    grid.append(
        Transform(
            "color_down", "color", lambda im: color_jitter(im, lo, lo, lo)
        )
    )
    grid.append(
        Transform("color_up", "color", lambda im: color_jitter(im, hi, hi, hi))
    )

    grid.append(
        Transform(
            f"crop_{int(CENTER_CROP_FRACTION * 100)}",
            "crop",
            lambda im: center_crop(im, CENTER_CROP_FRACTION),
        )
    )

    return grid


def _compose(*fns: Callable[[Image.Image], Image.Image]):
    def run(image: Image.Image) -> Image.Image:
        for fn in fns:
            image = fn(image)
        return image

    return run


def eval_grid_combinations(noise_seed: int = 0) -> list[Transform]:
    """Realistic multi-step redistribution chains.

    Real images rarely suffer exactly one operation: they get cropped, then
    re-encoded, then thumbnailed. These are diagnostics for the worst case, not
    something we tune against.
    """
    return [
        Transform(
            "jpeg70+resize0.5",
            "combo",
            _compose(lambda im: jpeg(im, 70), lambda im: downscale_upscale(im, 0.5)),
        ),
        Transform(
            "crop80+jpeg50",
            "combo",
            _compose(
                lambda im: center_crop(im, 0.8), lambda im: jpeg(im, 50)
            ),
        ),
        Transform(
            "resize0.5+blur1.0",
            "combo",
            _compose(
                lambda im: downscale_upscale(im, 0.5),
                lambda im: gaussian_blur(im, 1.0),
            ),
        ),
        Transform(
            "jpeg50+noise0.02",
            "combo",
            _compose(
                lambda im: jpeg(im, 50),
                lambda im: gaussian_noise(im, 0.02, random.Random(noise_seed + 100)),
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Training-time augmentation
# ---------------------------------------------------------------------------


class TrainAugment:
    """Randomized degradation drawn from the organizer transform families.

    Two properties matter more than the exact parameter choices:

    1. It is applied IDENTICALLY to both classes. If real and fake images get
       different processing, the processing itself becomes the label and the
       model learns a shortcut instead of a forensic signal.

    2. It samples *continuously* from ranges that bracket the organizer's
       discrete settings, rather than snapping to them. Training on exactly
       {90, 70, 50, 30} would let the model memorize four JPEG quantization
       tables; sampling quality uniformly from [25, 95] forces genuine
       invariance and keeps the eval grid honest.

    `severity` controls how many operations get chained, mirroring the
    "hierarchical augmentation" that separated the top NTIRE 2026 teams:
      0 -> clean, 1 -> one op, 2 -> two ops, 3 -> three ops.
    """

    def __init__(
        self,
        severity_weights: Sequence[float] = (0.15, 0.35, 0.30, 0.20),
        seed: int | None = None,
    ) -> None:
        if len(severity_weights) != 4:
            raise ValueError("severity_weights must have exactly 4 entries")
        self.severity_weights = tuple(severity_weights)
        self._rng = random.Random(seed)

    # -- individual samplers ------------------------------------------------

    def _sample_jpeg(self, image: Image.Image) -> Image.Image:
        return jpeg(image, self._rng.randint(25, 95))

    def _sample_blur(self, image: Image.Image) -> Image.Image:
        return gaussian_blur(image, self._rng.uniform(0.3, 2.2))

    def _sample_resize(self, image: Image.Image) -> Image.Image:
        return downscale_upscale(image, self._rng.uniform(0.22, 0.85))

    def _sample_noise(self, image: Image.Image) -> Image.Image:
        return gaussian_noise(image, self._rng.uniform(0.01, 0.11), self._rng)

    def _sample_color(self, image: Image.Image) -> Image.Image:
        def factor() -> float:
            return self._rng.uniform(0.75, 1.25)

        return color_jitter(image, factor(), factor(), factor())

    def _sample_crop(self, image: Image.Image) -> Image.Image:
        return center_crop(image, self._rng.uniform(0.75, 0.95))

    def _operations(self) -> list[Callable[[Image.Image], Image.Image]]:
        return [
            self._sample_jpeg,
            self._sample_blur,
            self._sample_resize,
            self._sample_noise,
            self._sample_color,
            self._sample_crop,
        ]

    # -- entry point --------------------------------------------------------

    def __call__(self, image: Image.Image) -> Image.Image:
        severity = self._rng.choices((0, 1, 2, 3), weights=self.severity_weights)[0]
        if severity == 0:
            return image

        chosen = self._rng.sample(self._operations(), severity)
        for op in chosen:
            image = op(image)
        return image
