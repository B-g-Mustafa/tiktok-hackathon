"""Input policy: how an arbitrary image becomes a fixed-size model input.

This is a load-bearing decision, not plumbing. Three constraints pull against
each other:

1. **Resizing destroys the evidence.** Downscaling is a low-pass filter, and the
   pixel-level traces of image synthesis live in exactly the high frequencies it
   removes. Detectors overwhelmingly crop rather than resize for this reason.

2. **Cropping alone throws away global structure.** Our semantic branch is
   hypothesised to read structural cues that survive heavy degradation. On a
   4000x3000 photograph a single 378px crop covers about 1% of the frame, which
   discards precisely what that branch depends on.

3. **Whole-image input leaks the label.** The dataset audit showed image size
   alone separates the classes at 0.90 AUROC. A fixed-size crop hides total
   pixel count and aspect ratio; only scale survives, which the split builder
   matches away.

So we use both views and let the ablation decide their weighting:

    native crops  -- forensic detail at true resolution, no interpolation
    resized view  -- whole-frame structure, at the cost of high frequencies

Upscaling is never used to reach the crop size. Interpolating a small image
would fabricate exactly the kind of resampling artifact the detector reads, so
images below the crop size are reflect-padded instead, and the padded fraction
is reported so it can be excluded or modelled.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

__all__ = [
    "CropResult",
    "native_crop",
    "resized_view",
    "multi_crop_views",
]


@dataclass(frozen=True)
class CropResult:
    """A crop plus the provenance needed to interpret it.

    `pad_fraction` is the share of the output that is reflected padding rather
    than real pixels. It is non-zero only for images smaller than the crop, and
    is worth tracking: padding is synthetic content, and a model could in
    principle learn from it.
    """

    image: Image.Image
    pad_fraction: float
    source_min_side: int


def _reflect_pad_to(image: Image.Image, size: int) -> tuple[Image.Image, float]:
    """Pad an undersized image up to `size` by reflection.

    Reflection is chosen over a constant fill because a flat border is a strong,
    perfectly uniform artificial region -- far more distinctive to a forensic
    model than mirrored real content.
    """
    width, height = image.size
    if width >= size and height >= size:
        return image, 0.0

    target_w = max(width, size)
    target_h = max(height, size)

    array = np.asarray(image.convert("RGB"))
    pad_h = target_h - height
    pad_w = target_w - width

    # np.pad's reflect mode cannot pad more than (dimension - 1) at a time, so
    # for very small images we apply it repeatedly.
    while pad_h > 0 or pad_w > 0:
        step_h = min(pad_h, max(array.shape[0] - 1, 0))
        step_w = min(pad_w, max(array.shape[1] - 1, 0))
        if step_h == 0 and step_w == 0:
            # Degenerate (1px) input: reflection is impossible, fall back to edge.
            array = np.pad(
                array,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="edge",
            )
            pad_h = pad_w = 0
            break
        array = np.pad(
            array,
            ((0, step_h), (0, step_w), (0, 0)),
            mode="reflect",
        )
        pad_h -= step_h
        pad_w -= step_w

    padded = Image.fromarray(array, mode="RGB")
    real_pixels = width * height
    total_pixels = padded.size[0] * padded.size[1]
    return padded, 1.0 - (real_pixels / total_pixels)


def native_crop(
    image: Image.Image,
    size: int,
    mode: str = "center",
    rng: random.Random | None = None,
) -> CropResult:
    """Take a `size` x `size` crop at the image's native resolution.

    `mode` is "random" for training (position jitter is free augmentation and
    decorrelates crop position from label) or "center" for deterministic
    inference.
    """
    if size < 1:
        raise ValueError("crop size must be >= 1")
    if mode not in ("center", "random"):
        raise ValueError(f"unknown crop mode: {mode!r}")

    image = image.convert("RGB")
    source_min_side = min(image.size)

    working, pad_fraction = _reflect_pad_to(image, size)
    width, height = working.size

    max_left = width - size
    max_top = height - size

    if mode == "random":
        generator = rng or random
        left = generator.randint(0, max_left) if max_left > 0 else 0
        top = generator.randint(0, max_top) if max_top > 0 else 0
    else:
        left = max_left // 2
        top = max_top // 2

    cropped = working.crop((left, top, left + size, top + size))
    return CropResult(cropped, pad_fraction, source_min_side)


def resized_view(image: Image.Image, size: int) -> CropResult:
    """Whole image squashed to `size` x `size`, ignoring aspect ratio.

    This is the complement to `native_crop`: it keeps global composition at the
    cost of the high-frequency detail cropping preserves. Aspect ratio is
    deliberately not preserved -- letterboxing would introduce flat bars whose
    thickness encodes the original aspect ratio, reintroducing a size cue that
    the split builder worked to remove.
    """
    image = image.convert("RGB")
    return CropResult(
        image.resize((size, size), Image.BICUBIC), 0.0, min(image.size)
    )


def multi_crop_views(
    image: Image.Image,
    size: int,
    n_crops: int = 4,
    include_resized: bool = True,
    rng: random.Random | None = None,
) -> list[CropResult]:
    """Views of one image for inference-time averaging.

    A single centre crop is a high-variance estimate: synthetic artifacts are
    not uniform across an image, and the centre is not privileged. Averaging
    scores over several positions plus the whole-frame view is markedly more
    stable, and it is the cheap answer to arbitrary resolutions and extreme
    aspect ratios -- a panorama gets sampled across its width rather than
    judged on its middle square.
    """
    if n_crops < 1:
        raise ValueError("n_crops must be >= 1")

    views = [native_crop(image, size, mode="center")]

    if n_crops > 1:
        generator = rng or random.Random(0)
        for _ in range(n_crops - 1):
            views.append(native_crop(image, size, mode="random", rng=generator))

    if include_resized:
        views.append(resized_view(image, size))

    return views
