"""A local, on-disk image dataset for LoRA fine-tuning.

Fine-tuning needs gradients through the encoder, which rules out the cached
frozen-feature approach entirely: every epoch has to re-decode and re-augment
raw pixels. Streaming that directly from the remote parquet shards on every
epoch would re-download tens of gigabytes per pass, so the workflow is split in
two:

  1. `materialize()` (below) pulls each selected image ONCE, via the same
     `iter_selected_images` used for feature caching, and writes it to a local
     PNG plus a manifest row. PNG rather than JPEG deliberately -- re-encoding
     through JPEG here would bake a compression pass into every image before
     augmentation ever sees it, and the whole training loop already has to
     handle exactly that shortcut risk (see EXP-002/EXP-003 in the ledger).
  2. `LocalImageDataset` (below) reads that manifest and is a plain
     `torch.utils.data.Dataset`, so a standard multi-worker `DataLoader` handles
     the (CPU-bound) decode-and-augment step in parallel with GPU compute.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from src.data.parquet_images import FetchedImage

__all__ = [
    "MaterializeStats",
    "materialize",
    "LocalImageDataset",
    "collate_list",
    "iter_local_images",
]

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.parquet"


@dataclass
class MaterializeStats:
    n_written: int
    n_failed: int
    output_dir: Path


def materialize(
    repo_id: str,
    selection: pd.DataFrame,
    output_dir: Path,
    skip_existing: bool = True,
    show_progress: bool = True,
    checkpoint_every: int = 500,
) -> MaterializeStats:
    """Download and decode every row in `selection` once, to local PNGs.

    Idempotent: if the manifest already lists a key, that image is skipped, so
    a failed or interrupted run can simply be re-invoked -- interrupting a
    multi-hour download and resuming it later is a normal thing to do here,
    not an edge case.

    `show_progress` drives a `tqdm` bar over images written and cumulative
    bytes on disk. Fetching happens inside the `iter_selected_images`
    generator, so the bar reflects real network+decode progress as images
    actually arrive, not an estimate.
    """
    from src.data.parquet_images import iter_selected_images

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME

    existing_keys: set[str] = set()
    rows: list[dict] = []
    bytes_written = 0
    if skip_existing and manifest_path.exists():
        existing = pd.read_parquet(manifest_path)
        existing_keys = set(existing["key"])
        rows = existing.to_dict("records")
        bytes_written = sum(
            Path(r["path"]).stat().st_size
            for r in rows
            if Path(r["path"]).exists()
        )
        logger.info(
            "resuming: %d images already materialized (%.2f GB)",
            len(existing_keys), bytes_written / 1e9,
        )

    n_written = len(rows)
    n_failed = 0
    n_total = len(selection)

    iterator = iter_selected_images(repo_id, selection)
    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(
            total=n_total,
            initial=n_written,
            desc=f"materializing {output_dir.name}",
            unit="img",
        )
        progress.set_postfix_str(f"{bytes_written / 1e9:.2f} GB")

    try:
        for fetched in iterator:
            if fetched.key in existing_keys:
                continue

            safe_name = fetched.key.replace("/", "_").replace("#", "__")
            path = output_dir / f"{safe_name}.png"

            try:
                fetched.image.save(path, format="PNG")
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to write %s: %s", fetched.key, exc)
                n_failed += 1
                continue

            rows.append(
                {
                    "key": fetched.key,
                    "path": str(path),
                    "label": fetched.label,
                    "model_name": fetched.model_name,
                    "min_side": fetched.min_side,
                }
            )
            existing_keys.add(fetched.key)
            n_written += 1
            bytes_written += path.stat().st_size

            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(f"{bytes_written / 1e9:.2f} GB")

            if n_written % checkpoint_every == 0:
                # Checkpoint the manifest periodically so a crash mid-run does
                # not lose already-materialized work.
                pd.DataFrame(rows).to_parquet(manifest_path, index=False)
                if progress is None:
                    logger.info(
                        "materialized %d/%d images (%.2f GB)",
                        n_written, n_total, bytes_written / 1e9,
                    )
    finally:
        if progress is not None:
            progress.close()

    pd.DataFrame(rows).to_parquet(manifest_path, index=False)
    return MaterializeStats(n_written, n_failed, output_dir)


class LocalImageDataset(Dataset):
    """Reads the manifest `materialize()` produced and yields (PIL crop, label).

    Augmentation happens here, in `__getitem__`, specifically so it runs in
    DataLoader worker processes rather than on the main/GPU-feeding thread.
    """

    def __init__(
        self,
        manifest_dir: Path,
        crop_size: int,
        transform: Callable[[Image.Image], Image.Image] | None = None,
        crop_mode: str = "random",
    ) -> None:
        from src.transforms.crop import native_crop

        self._native_crop = native_crop
        self.manifest = pd.read_parquet(Path(manifest_dir) / MANIFEST_NAME)
        self.crop_size = crop_size
        self.transform = transform
        self.crop_mode = crop_mode

        if self.manifest.empty:
            raise ValueError(f"manifest at {manifest_dir} is empty")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        row = self.manifest.iloc[index]

        with Image.open(row["path"]) as handle:
            image = handle.convert("RGB")
            image.load()

        if self.transform is not None:
            image = self.transform(image)

        crop = self._native_crop(image, self.crop_size, mode=self.crop_mode).image
        return crop, int(row["label"])


def iter_local_images(manifest_dir: Path) -> Iterator[FetchedImage]:
    """Yield `FetchedImage`s from a local manifest -- the local-disk
    counterpart to `parquet_images.iter_selected_images`.

    Reusing the same `FetchedImage` type means `cache_features.py`'s
    extraction loop needs zero changes to source images from a local GenImage
    directory instead of remote Community Forensics shards: only which
    iterator function it calls differs.
    """
    from src.data.parquet_images import FetchedImage

    manifest = pd.read_parquet(Path(manifest_dir) / MANIFEST_NAME)

    for _, row in manifest.iterrows():
        path = row["path"]
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                image.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load %s: %s", path, exc)
            continue

        yield FetchedImage(
            image=image,
            label=int(row["label"]),
            shard=str(row["key"]),
            row_in_shard=0,
            model_name=str(row.get("model_name", "")),
            min_side=int(row.get("min_side", min(image.size))),
        )


def collate_list(batch: list[tuple[Image.Image, int]]) -> tuple[list[Image.Image], list[int]]:
    """Keep images as a plain list rather than stacking into a tensor.

    `LoraEncoder.forward_features` does its own PIL-to-tensor conversion and
    normalization (matching `FrozenEncoder` exactly, for comparability), so the
    collate step only needs to separate images from labels.
    """
    images, labels = zip(*batch)
    return list(images), list(labels)
