"""Tests for the required inference contract.

The output schema is fixed by the organizers, so these tests pin it down
exactly. They also cover the failure modes an arbitrary user directory actually
contains -- truncated files, alpha channels, CMYK, extreme aspect ratios -- all
of which must degrade gracefully rather than losing the whole run.

Deliberately dependency-free (no torch): the contract is verified against the
placeholder detector, so it stays green regardless of model work.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.io import iter_image_paths, load_image  # noqa: E402
from src.models.base import ConstantDetector  # noqa: E402
from scripts.predict import run  # noqa: E402


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """A directory covering the formats and pathologies we expect in the wild."""
    directory = tmp_path / "images"
    directory.mkdir()
    rng = np.random.default_rng(0)

    def noise(w: int, h: int) -> Image.Image:
        return Image.fromarray(
            rng.integers(0, 256, (h, w, 3), dtype=np.uint8), mode="RGB"
        )

    noise(64, 64).save(directory / "a.jpg")
    noise(50, 80).save(directory / "b.png")
    noise(32, 32).save(directory / "c.jpeg")
    noise(40, 40).save(directory / "d.webp")

    # Extreme aspect ratio -- panorama crops are common on social platforms.
    noise(800, 20).save(directory / "wide.png")

    # Alpha channel: naive .convert("RGB") composites onto black, creating a
    # large flat artificial region.
    Image.new("RGBA", (32, 32), (255, 0, 0, 0)).save(directory / "alpha.png")

    # CMYK, as produced by print-oriented scanners.
    Image.new("CMYK", (32, 32), (0, 0, 0, 0)).save(directory / "cmyk.jpg")

    # Greyscale.
    Image.new("L", (32, 32), 128).save(directory / "gray.png")

    # Palette image with transparency.
    Image.new("P", (32, 32)).save(directory / "palette.png")

    # Pathologies that must be skipped, not crash.
    (directory / "corrupt.jpg").write_bytes(b"this is not a jpeg")
    (directory / "empty.png").write_bytes(b"")
    (directory / "notes.txt").write_text("ignored: not an image")

    return directory


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loads_all_supported_formats(image_dir: Path):
    for name in ("a.jpg", "b.png", "c.jpeg", "d.webp", "wide.png"):
        result = load_image(image_dir / name)
        assert result.ok, f"{name} failed: {result.error}"
        assert result.image.mode == "RGB"


def test_corrupt_file_returns_error_not_exception(image_dir: Path):
    result = load_image(image_dir / "corrupt.jpg")
    assert not result.ok
    assert result.error


def test_empty_file_returns_error(image_dir: Path):
    result = load_image(image_dir / "empty.png")
    assert not result.ok
    assert "empty" in result.error.lower()


def test_missing_file_returns_error(tmp_path: Path):
    assert not load_image(tmp_path / "nope.jpg").ok


def test_directory_path_returns_error(tmp_path: Path):
    assert not load_image(tmp_path).ok


def test_transparent_alpha_composites_onto_white(image_dir: Path):
    """A fully transparent red PNG must become white, not black.

    Compositing onto black would paint a large uniform dark region into the
    image -- a strong artificial signal a forensic model would happily learn.
    """
    result = load_image(image_dir / "alpha.png")
    assert result.ok
    assert result.image.getpixel((0, 0)) == (255, 255, 255)


def test_cmyk_and_grayscale_and_palette_convert(image_dir: Path):
    for name in ("cmyk.jpg", "gray.png", "palette.png"):
        result = load_image(image_dir / name)
        assert result.ok, f"{name}: {result.error}"
        assert result.image.mode == "RGB"


def test_iter_skips_non_images_and_is_sorted(image_dir: Path):
    paths = list(iter_image_paths(image_dir))
    names = [p.name for p in paths]
    assert "notes.txt" not in names
    assert names == sorted(names)


def test_iter_skips_macos_resource_forks(image_dir: Path):
    (image_dir / "._a.jpg").write_bytes(b"junk")
    assert "._a.jpg" not in [p.name for p in iter_image_paths(image_dir)]


def test_iter_finds_nested_images(image_dir: Path):
    nested = image_dir / "sub" / "deeper"
    nested.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(nested / "nested.png")

    assert "nested.png" in [p.name for p in iter_image_paths(image_dir)]
    assert "nested.png" not in [
        p.name for p in iter_image_paths(image_dir, recursive=False)
    ]


def test_iter_rejects_non_directory(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        list(iter_image_paths(tmp_path / "missing"))


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_output_schema(image_dir: Path):
    records, _ = run(image_dir, ConstantDetector(0.5), 4, True, False)
    assert records
    for record in records:
        assert set(record) == {"image_path", "pred"}
        assert isinstance(record["image_path"], str)
        assert isinstance(record["pred"], float)
        assert 0.0 <= record["pred"] <= 1.0


def test_image_paths_are_absolute(image_dir: Path):
    records, _ = run(image_dir, ConstantDetector(0.5), 4, True, False)
    assert all(Path(r["image_path"]).is_absolute() for r in records)


def test_corrupt_images_are_skipped_but_run_completes(image_dir: Path):
    records, n_failed = run(image_dir, ConstantDetector(0.5), 4, True, False)
    # corrupt.jpg and empty.png
    assert n_failed == 2
    assert records
    assert not any("corrupt.jpg" in r["image_path"] for r in records)


def test_include_failures_emits_null_predictions(image_dir: Path):
    records, _ = run(image_dir, ConstantDetector(0.5), 4, True, True)
    failed = [r for r in records if r["pred"] is None]
    assert len(failed) == 2
    assert all("error" in r for r in failed)


def test_output_is_sorted_and_deterministic(image_dir: Path):
    a, _ = run(image_dir, ConstantDetector(0.5), 3, True, False)
    b, _ = run(image_dir, ConstantDetector(0.5), 7, True, False)
    # Batch size must not affect results or ordering.
    assert a == b
    assert [r["image_path"] for r in a] == sorted(r["image_path"] for r in a)


def test_batching_covers_every_image(image_dir: Path):
    """A partial final batch must still be flushed."""
    expected = len(list(iter_image_paths(image_dir))) - 2  # minus the 2 bad files
    for batch_size in (1, 2, 3, 5, 100):
        records, _ = run(image_dir, ConstantDetector(0.5), batch_size, True, False)
        assert len(records) == expected, f"batch_size={batch_size}"


def test_detector_returning_wrong_count_is_caught(image_dir: Path):
    class Broken:
        name = "broken"

        def predict_batch(self, images):
            return [0.5]  # wrong length

    with pytest.raises(RuntimeError, match="scores for"):
        run(image_dir, Broken(), 4, True, False)


def test_empty_directory_produces_empty_json(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    records, n_failed = run(empty, ConstantDetector(0.5), 4, True, False)
    assert records == []
    assert n_failed == 0


def test_constant_detector_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        ConstantDetector(1.5)


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


def test_cli_end_to_end(image_dir: Path, tmp_path: Path):
    out = tmp_path / "preds.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "predict.py"),
            "--image-dir",
            str(image_dir),
            "--out",
            str(out),
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()

    payload = json.loads(out.read_text())
    assert isinstance(payload, list) and payload
    assert set(payload[0]) == {"image_path", "pred"}


def test_cli_rejects_missing_directory(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "predict.py"),
            "--image-dir",
            str(tmp_path / "does-not-exist"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
