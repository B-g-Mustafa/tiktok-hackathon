"""Safe image loading for inference.

The inference script is handed an arbitrary directory. In the wild that means
truncated downloads, CMYK scans, 16-bit TIFFs, palettised GIFs, EXIF-rotated
phone photos, zero-byte files, and the occasional decompression bomb. Every one
of these must produce either a correct RGB image or a clean skip -- never a
crash that loses the other 9,999 predictions.

`load_image` therefore returns `None` on failure instead of raising, and the
caller reports the failure in the output rather than dying.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageFile, ImageOps

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "LoadResult",
    "load_image",
    "iter_image_paths",
]

logger = logging.getLogger(__name__)

# Pillow refuses to decode partially-truncated JPEGs by default. Real-world
# directories contain interrupted downloads, and a partially-decoded image is
# far more useful than a hard failure.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Pillow's default bomb guard is ~89M pixels and raises a warning-turned-error.
# We raise the ceiling so legitimate large photographs load, while still
# refusing genuinely absurd inputs.
Image.MAX_IMAGE_PIXELS = 512_000_000

SUPPORTED_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)


@dataclass(frozen=True)
class LoadResult:
    """Outcome of a load attempt. Exactly one of `image` / `error` is set."""

    path: Path
    image: Image.Image | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.image is not None


def _to_rgb(image: Image.Image) -> Image.Image:
    """Coerce any mode to RGB without introducing black backgrounds.

    Naively calling `.convert("RGB")` on an image with alpha composites against
    black, which turns transparent PNG regions into large flat black areas --
    a strong artificial signal for a forensic detector. Compositing onto white
    matches how these images are actually displayed.
    """
    if image.mode == "RGB":
        return image

    # Palette images may carry transparency in their palette.
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")

    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background

    # L, I, I;16, F, CMYK, YCbCr all convert directly and unambiguously.
    return image.convert("RGB")


def load_image(path: Path | str) -> LoadResult:
    """Load one image as RGB, applying EXIF orientation.

    Returns a `LoadResult`; never raises for a bad input file.
    """
    path = Path(path)

    try:
        if not path.is_file():
            return LoadResult(path, None, "not a file")
        if path.stat().st_size == 0:
            return LoadResult(path, None, "empty file")

        with Image.open(path) as handle:
            # Phone photos store rotation in EXIF. Without this, portrait shots
            # arrive sideways and every geometric statistic is transposed.
            oriented = ImageOps.exif_transpose(handle)
            # exif_transpose returns None if there is nothing to do on some
            # Pillow versions; fall back to the original handle.
            if oriented is None:
                oriented = handle
            image = _to_rgb(oriented)
            image.load()

        if image.width < 1 or image.height < 1:
            return LoadResult(path, None, "zero-sized image")

        return LoadResult(path, image, None)

    except Exception as exc:  # noqa: BLE001 - any decode failure is a skip
        logger.debug("failed to load %s: %s", path, exc)
        return LoadResult(path, None, f"{type(exc).__name__}: {exc}")


def iter_image_paths(
    directory: Path | str, recursive: bool = True
) -> Iterator[Path]:
    """Yield candidate image paths in deterministic (sorted) order.

    Sorting matters: it makes predictions reproducible run-to-run and makes
    diffing two prediction files meaningful.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        # Skip macOS resource forks and other dotfiles that look like images.
        if path.name.startswith("._"):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
