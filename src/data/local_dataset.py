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
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
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


def _write_manifest_atomically(rows: list[dict], manifest_path: Path) -> None:
    """Write the manifest so a hard kill mid-write can never corrupt it.

    A multi-hour, 100GB+ download on a shared HPC cluster is far more likely
    to end in a SIGKILL (SLURM hitting a walltime limit, the OOM killer) than
    a clean Ctrl-C -- and a plain `to_parquet(manifest_path)` writes directly
    to the final file, so a kill mid-write can leave a truncated,
    unreadable parquet there. The next run's resume attempt would then crash
    trying to *read* the checkpoint meant to protect it, rather than just
    losing a little progress. Writing to a temp file in the same directory and
    using `os.replace` (an atomic rename on POSIX filesystems) means the
    manifest is always either the old complete version or the new complete
    version, never a partial one in between.
    """
    tmp_path = manifest_path.with_suffix(".parquet.tmp")
    pd.DataFrame(rows).to_parquet(tmp_path, index=False)
    os.replace(tmp_path, manifest_path)


@dataclass
class MaterializeStats:
    n_written: int
    n_failed: int
    output_dir: Path
    failed_shards: list[str] = field(default_factory=list)


def _save_one(fetched, output_dir: Path):
    """Encode and write one image. A free function (not a closure) so it can
    be handed to a thread pool -- Pillow's PNG encoder releases the GIL for
    the actual compression work, same reasoning as `_decode_one`."""
    safe_name = fetched.key.replace("/", "_").replace("#", "__")
    path = output_dir / f"{safe_name}.png"
    try:
        fetched.image.save(path, format="PNG")
        return fetched, path, None
    except Exception as exc:  # noqa: BLE001
        return fetched, None, exc


# Total wall-clock budget for one batch's worth of concurrent PNG writes (not
# per-image). Encoding an already-decoded, already-in-memory image is fast;
# this exists to bound a genuinely stuck write (e.g. a hung network
# filesystem), not to police normal speed.
SAVE_BATCH_TIMEOUT_SECONDS = 60.0


def _save_batch_concurrent(batch: list, output_dir: Path, executor: ThreadPoolExecutor):
    """Save a batch of images concurrently, immune to one bad item blocking
    the rest.

    Uses `as_completed` with a batch-level timeout rather than
    `Executor.map()`, for the same reason `_decode_batch` does -- `map()`
    yields strictly in submission order, so one stuck write (a hung network
    filesystem, say) would block every other already-finished result behind
    it rather than just itself. See `_decode_batch`'s docstring for the
    directly-reproduced demonstration of this.
    """
    futures = {
        executor.submit(_save_one, fetched, output_dir): fetched for fetched in batch
    }
    results = []

    try:
        for future in as_completed(futures, timeout=SAVE_BATCH_TIMEOUT_SECONDS):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append((futures[future], None, exc))
    except TimeoutError:
        # FetchedImage is a plain (non-frozen) dataclass, so it is NOT
        # hashable -- a set of the objects themselves would raise. Track by
        # identity instead, which is also the more correct semantic here: we
        # want "this specific submitted item," not "any item with equal
        # field values."
        done_ids = {id(futures[f]) for f in futures if f.done()}
        for fetched in batch:
            if id(fetched) not in done_ids:
                results.append((fetched, None, TimeoutError("save timed out")))

    return results


def materialize(
    repo_id: str,
    selection: pd.DataFrame,
    output_dir: Path,
    skip_existing: bool = True,
    show_progress: bool = True,
    checkpoint_every: int = 500,
    local_dir: Path | None = None,
    workers: int = 8,
) -> MaterializeStats:
    """Download and decode every row in `selection` once, to local PNGs.

    Idempotent: if the manifest already lists a key, that image is skipped, so
    a failed or interrupted run can simply be re-invoked -- interrupting a
    multi-hour download and resuming it later is a normal thing to do here,
    not an edge case.

    `show_progress` drives a `tqdm` bar over images written and cumulative
    bytes on disk. Fetching happens inside the underlying generator, so the
    bar reflects real progress as images actually arrive, not an estimate.

    `local_dir`, if given, reads shards that are already fully downloaded on
    local disk (e.g. via `scripts/download_full_dataset.py`) instead of
    fetching remotely -- same row-selection logic, no network at all.

    A shard that could not be read (after retries, for the remote path) is
    recorded in the returned `MaterializeStats.failed_shards` rather than
    silently reducing the image count -- so a caller can tell the difference
    between "the plan asked for fewer images" and "some shards were
    unreachable and got skipped."

    `workers` controls concurrency for BOTH halves of the pipeline: decoding
    images (inside the fetch iterator) and encoding+writing them as PNGs
    (here). Both are CPU-bound codec work where Pillow releases the GIL, so
    threads give a real speedup, not just I/O overlap -- measured at 7-8x on
    8-16 workers for 512x512 PNG decode+re-encode. Set to 1 for the old
    fully-sequential behaviour.
    """
    if local_dir is not None:
        from src.data.parquet_images import iter_selected_images_local

        def make_iterator(failed: list[str]):
            return iter_selected_images_local(
                local_dir, selection, failed_shards=failed, workers=workers
            )
    else:
        from src.data.parquet_images import iter_selected_images

        def make_iterator(failed: list[str]):
            return iter_selected_images(
                repo_id, selection, failed_shards=failed, workers=workers
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME

    existing_keys: set[str] = set()
    rows: list[dict] = []
    bytes_written = 0
    if skip_existing and manifest_path.exists():
        try:
            existing = pd.read_parquet(manifest_path)
        except Exception as exc:  # noqa: BLE001
            # Only possible from a checkpoint written before the atomic-write
            # fix, or external corruption -- a fresh run's own checkpoints
            # can no longer land here (see _write_manifest_atomically). Start
            # over rather than crash the whole resume on a corrupt leftover;
            # already-downloaded PNGs on disk just get harmlessly re-fetched.
            logger.warning(
                "manifest at %s is unreadable (%s) -- treating as no prior "
                "progress and starting fresh for this split",
                manifest_path, exc,
            )
            existing = None

        if existing is not None:
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
    failed_shards: list[str] = []

    iterator = make_iterator(failed_shards)
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

    save_executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    # Buffer a batch before submitting so encode+write for many images
    # overlaps concurrently, rather than one save blocking the next fetch.
    batch_size = workers * 4 if save_executor is not None else 1

    def save_batch(batch: list) -> None:
        nonlocal n_written, n_failed, bytes_written

        if save_executor is not None:
            results = _save_batch_concurrent(batch, output_dir, save_executor)
        else:
            results = (_save_one(f, output_dir) for f in batch)

        for fetched, path, exc in results:
            if exc is not None:
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

    try:
        pending: list = []
        for fetched in iterator:
            if fetched.key in existing_keys:
                continue

            pending.append(fetched)
            if len(pending) < batch_size:
                continue

            save_batch(pending)
            pending = []

            if n_written % checkpoint_every < batch_size:
                # Checkpoint the manifest periodically so a crash mid-run does
                # not lose already-materialized work.
                _write_manifest_atomically(rows, manifest_path)
                if progress is None:
                    logger.info(
                        "materialized %d/%d images (%.2f GB)",
                        n_written, n_total, bytes_written / 1e9,
                    )

        if pending:
            # The final, less-than-a-full-batch remainder -- without this,
            # up to `batch_size - 1` fetched images would be silently
            # dropped every run.
            save_batch(pending)
    finally:
        if progress is not None:
            progress.close()
        if save_executor is not None:
            save_executor.shutdown(wait=False, cancel_futures=True)

    _write_manifest_atomically(rows, manifest_path)
    return MaterializeStats(n_written, n_failed, output_dir, failed_shards)


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
