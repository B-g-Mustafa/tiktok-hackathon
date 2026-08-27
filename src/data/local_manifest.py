"""Build a manifest from a plain local directory of real/fake images.

This is the entry point for training on data that ISN'T Community Forensics --
GenImage, or any other real-vs-generated folder tree. The Community Forensics
pipeline (`build_manifest.py` / `build_splits.py`) exists because that dataset
is 260 GB on a remote parquet store and needed shard-aware planning; a local
folder tree needs none of that. This module just walks the tree once, labels
each image by which folder it lives under, and writes the same
`manifest.parquet` schema `LocalImageDataset` and `finetune_lora.py` already
consume -- so everything downstream of materialization is unchanged regardless
of which dataset produced it.

Folder-name matching is deliberately case-insensitive and covers GenImage's
own convention (`ai` / `nature`) alongside more generic names, since GenImage
is the concrete case this exists for:

    <root>/<generator>/train/ai/*.png       -> label 1 (generated)
    <root>/<generator>/train/nature/*.jpg   -> label 0 (authentic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.io import SUPPORTED_EXTENSIONS, load_image

__all__ = [
    "REAL_DIRNAMES",
    "FAKE_DIRNAMES",
    "SKIP_DIRNAMES",
    "LocalManifestStats",
    "build_local_manifest",
]

logger = logging.getLogger(__name__)

# Directory names that mark real vs. generated images. Lowercased before
# comparison, so "AI", "Ai", "ai" all match.
REAL_DIRNAMES = frozenset({"real", "nature", "authentic", "reals"})
FAKE_DIRNAMES = frozenset({"fake", "ai", "generated", "synthetic", "fakes"})

# Directories that are structural (split names) rather than label-bearing --
# ignored when inferring a "generator" tag from the path.
SKIP_DIRNAMES = frozenset(
    {"train", "val", "test", "validation"} | REAL_DIRNAMES | FAKE_DIRNAMES
)


@dataclass
class LocalManifestStats:
    n_authentic: int
    n_generated: int
    n_skipped_unlabeled: int
    n_skipped_unreadable: int
    manifest_path: Path

    def __str__(self) -> str:
        return (
            f"{self.n_authentic:,} authentic + {self.n_generated:,} generated "
            f"-> {self.manifest_path} "
            f"({self.n_skipped_unlabeled:,} unlabeled, "
            f"{self.n_skipped_unreadable:,} unreadable, skipped)"
        )


def _label_from_path(path: Path, root: Path) -> int | None:
    """Look at every directory name between `root` and the file for a real/fake
    marker. Deepest match wins, in case a path contains both (unlikely, but a
    deterministic tiebreak is better than an arbitrary one)."""
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]

    label = None
    for part in parts:
        if part in REAL_DIRNAMES:
            label = 0
        elif part in FAKE_DIRNAMES:
            label = 1
    return label


def _generator_tag(path: Path, root: Path) -> str:
    """Best-effort generator/source name: whatever path components are left
    after removing split and label directories. For GenImage this recovers
    the generator name, e.g. `<root>/BigGAN/train/ai/x.png` -> "BigGAN"."""
    parts = [p for p in path.relative_to(root).parts[:-1] if p.lower() not in SKIP_DIRNAMES]
    return "/".join(parts) if parts else root.name


def build_local_manifest(
    root_dir: Path,
    output_dir: Path,
    limit_per_class: int | None = None,
    extensions: frozenset[str] = SUPPORTED_EXTENSIONS,
    split_filter: str | None = None,
) -> LocalManifestStats:
    """Scan `root_dir` and write `output_dir/manifest.parquet`.

    `split_filter` restricts the scan to files with that exact path component
    (case-insensitive) -- e.g. "train" or "val". This matters specifically for
    GenImage's layout, where train and val live side by side under every
    generator (`<generator>/train/...` and `<generator>/val/...`): without a
    filter, pointing `root_dir` at the dataset root silently merges both splits
    into one manifest, which is not what "give me the train split" means.

    Idempotent in the sense that it always rewrites the manifest from a fresh
    scan (unlike `materialize()`'s network-fetch resume logic, a local
    directory scan is cheap enough to just redo).
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not root_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {root_dir}")

    split_key = split_filter.lower() if split_filter else None

    rows: list[dict] = []
    counts = {0: 0, 1: 0}
    n_unlabeled = 0
    n_unreadable = 0
    n_scanned = 0

    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.name.startswith("._"):
            continue

        if split_key is not None:
            parts = [p.lower() for p in path.relative_to(root_dir).parts[:-1]]
            if split_key not in parts:
                continue

        label = _label_from_path(path, root_dir)
        if label is None:
            n_unlabeled += 1
            continue

        if limit_per_class is not None and counts[label] >= limit_per_class:
            continue

        result = load_image(path)
        if not result.ok:
            n_unreadable += 1
            logger.debug("skipping unreadable %s: %s", path, result.error)
            continue

        rows.append(
            {
                "key": str(path.relative_to(root_dir)),
                "path": str(path),
                "label": label,
                "model_name": _generator_tag(path, root_dir),
                "min_side": min(result.image.size),
            }
        )
        counts[label] += 1
        n_scanned += 1

        if n_scanned % 2000 == 0:
            logger.info(
                "scanned %d images (%d authentic, %d generated)...",
                n_scanned, counts[0], counts[1],
            )

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    if counts[0] == 0 or counts[1] == 0:
        logger.warning(
            "manifest has only one class (authentic=%d, generated=%d) -- "
            "check that your directory names match REAL_DIRNAMES/FAKE_DIRNAMES: "
            "real=%s fake=%s",
            counts[0], counts[1], sorted(REAL_DIRNAMES), sorted(FAKE_DIRNAMES),
        )

    return LocalManifestStats(
        n_authentic=counts[0],
        n_generated=counts[1],
        n_skipped_unlabeled=n_unlabeled,
        n_skipped_unreadable=n_unreadable,
        manifest_path=manifest_path,
    )
