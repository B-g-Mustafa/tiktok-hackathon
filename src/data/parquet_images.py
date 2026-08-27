"""Fetch selected image bytes from parquet shards -- remote or already local.

The manifest tells us exactly which rows we want; `iter_selected_images` gets
just those images from the remote dataset without downloading everything.
`iter_selected_images_local` is the same thing for shards that have already
been fully downloaded (e.g. via `scripts/download_full_dataset.py`) --
identical row-selection logic, but reading local files directly with no
network involved at all, which is both faster and immune to the network
timeouts the remote path has to actively defend against.

Parquet stores data in row groups, and pyarrow can read an individual row group
without reading the whole file. For the remote path that is the difference
between a 260 GB download and a few tens of GB; for the local path it just
means only decoding the rows actually needed.

Rows are yielded in shard order rather than manifest order. Feature extraction
does not care about ordering, and every row carries its own identifiers, so
matching features back to labels is done by key rather than by position.
"""

from __future__ import annotations

import io
import logging
import time
from concurrent.futures import Executor, ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

__all__ = [
    "FetchedImage",
    "iter_selected_images",
    "iter_selected_images_local",
    "IMAGE_COLUMN",
]

logger = logging.getLogger(__name__)

IMAGE_COLUMN = "image_data"

# huggingface_hub's own HTTP layer already retries a read up to 5 times with a
# short capped backoff (its `http_backoff` utility) using whatever timeout
# `HF_HUB_DOWNLOAD_TIMEOUT` is set to -- 10 seconds by default, which is
# aggressive for a multi-megabyte row-group read on a busy or higher-latency
# network. Once THOSE retries are exhausted the exception reaches us. A
# network blip lasting a couple of minutes (common on shared clusters) can
# outlast HF's own ~23s of internal backoff without outlasting a bit more
# patience on our end, so we retry the whole shard a few more times, with a
# much longer pause, before giving up on it for this run.
SHARD_RETRY_ATTEMPTS = 3
SHARD_RETRY_BACKOFF_SECONDS = 20.0


@dataclass
class FetchedImage:
    """One decoded image plus the manifest fields needed to label it."""

    image: Image.Image
    label: int
    shard: str
    row_in_shard: int
    model_name: str
    min_side: int

    @property
    def key(self) -> str:
        """Stable identifier, used to align cached features with labels."""
        return f"{self.shard}#{self.row_in_shard}"


def _row_groups_for(
    parquet_file: pq.ParquetFile, wanted_rows: set[int]
) -> dict[int, list[int]]:
    """Map row-group index -> the wanted row offsets it contains.

    Row groups are contiguous, so we walk them accumulating offsets and keep
    only groups that intersect the selection. Reading a group we do not need
    would pull megabytes of image bytes for nothing.
    """
    groups: dict[int, list[int]] = {}
    start = 0

    for group_index in range(parquet_file.num_row_groups):
        n_rows = parquet_file.metadata.row_group(group_index).num_rows
        end = start + n_rows

        hits = [r for r in wanted_rows if start <= r < end]
        if hits:
            # Offsets relative to the start of this row group.
            groups[group_index] = sorted(r - start for r in hits)

        start = end

    return groups


def _decode_one(raw: bytes | None) -> Image.Image | None:
    """Decode one image's raw bytes to an RGB PIL Image, or None on failure.

    Kept as its own top-level function (not a closure) specifically so it can
    be handed to a thread pool: Pillow's C-level decode and colour-convert
    calls release the GIL for the actual codec work, which is what makes
    threading here a genuine parallel speedup rather than just overhead --
    measured at 7-8x on 8-16 threads for 512x512 PNG decode+re-encode, not a
    marginal gain.
    """
    if raw is None:
        return None
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except Exception:  # noqa: BLE001 - caller logs with row context
        return None


# Total wall-clock budget for one row group's worth of concurrent decodes
# (not per-image). Legitimate decode of an already-in-memory image is
# milliseconds, so this is generous margin for real variance, not a tight
# bound -- it exists to catch a genuinely pathological/corrupt record, not to
# police normal speed.
DECODE_BATCH_TIMEOUT_SECONDS = 60.0


def _decode_batch(
    raws: list[bytes | None], executor: Executor
) -> list[Image.Image | None]:
    """Decode a batch of images concurrently, immune to one bad item blocking
    the rest.

    `Executor.map()` looks like the obvious tool for this and is exactly what
    the first version of this code used -- but `map()` yields results in
    SUBMISSION order, not completion order. If item 0 of 32 hangs (a
    corrupted or pathological record; plausible at hundreds of thousands of
    images pulled from a large real-world download), items 1-31 block behind
    it even though their threads finished instantly -- confirmed directly:
    with one artificially slow item in a batch of 8, every other item's
    result only became observable at the same moment the slow one finished,
    not when each individually completed. With a TRUE hang (not just slow),
    that stalls the entire pipeline forever, which is what this replaces.

    `as_completed` yields futures as they actually finish, and the batch-level
    timeout bounds how long we wait for stragglers -- anything not done by
    then is treated as failed (logged, not retried) rather than blocking
    forever. The underlying thread, if genuinely stuck, cannot be forcibly
    killed (Python threads don't support that) and is abandoned; this trades
    one leaked thread for the pipeline continuing to make progress.
    """
    futures = {executor.submit(_decode_one, raw): i for i, raw in enumerate(raws)}
    decoded: list[Image.Image | None] = [None] * len(raws)

    try:
        for future in as_completed(futures, timeout=DECODE_BATCH_TIMEOUT_SECONDS):
            index = futures[future]
            try:
                decoded[index] = future.result()
            except Exception:  # noqa: BLE001
                decoded[index] = None
    except TimeoutError:
        stuck = sum(1 for f in futures if not f.done())
        logger.warning(
            "%d image(s) in this batch did not finish decoding within %.0fs "
            "-- treating as failed and continuing",
            stuck, DECODE_BATCH_TIMEOUT_SECONDS,
        )

    return decoded


def _read_rows_from_open_parquet(
    parquet_file: pq.ParquetFile,
    shard: str,
    wanted: set[int],
    meta_by_row: dict,
    read_columns: list[str],
    executor: Executor | None = None,
) -> Iterator[FetchedImage]:
    """Shared row-extraction logic for an already-open ParquetFile.

    Used by both the remote and local fetch paths -- opening the file is the
    only thing that differs between them, so that is the only thing that
    should differ in the code.

    `executor`, if given, decodes every image in a row group concurrently
    instead of one at a time. Row-group reads themselves stay sequential
    (pyarrow already reads a group as one batched operation, so there is
    nothing per-image to overlap there) -- it is the per-image Python-level
    decode/convert loop that was previously fully serial and is the target
    here.
    """
    available = [c for c in read_columns if c in parquet_file.schema_arrow.names]
    targets = _row_groups_for(parquet_file, wanted)

    for group_index, offsets in targets.items():
        table = parquet_file.read_row_group(group_index, columns=available)
        image_bytes = table.column(IMAGE_COLUMN).to_pylist()

        # Absolute row index for each offset in this group.
        group_start = sum(
            parquet_file.metadata.row_group(i).num_rows for i in range(group_index)
        )

        raws = [image_bytes[offset] for offset in offsets]
        if executor is not None:
            decoded = _decode_batch(raws, executor)
        else:
            decoded = [_decode_one(raw) for raw in raws]

        for offset, raw, image in zip(offsets, raws, decoded):
            absolute = group_start + offset
            meta = meta_by_row.get(absolute)
            if meta is None:
                continue

            if raw is None:
                logger.warning("null image at %s#%d", shard, absolute)
                continue
            if image is None:
                logger.warning("decode failed at %s#%d", shard, absolute)
                continue

            yield FetchedImage(
                image=image,
                label=int(meta["label"]),
                shard=shard,
                row_in_shard=absolute,
                model_name=str(meta.get("model_name", "")),
                min_side=int(meta.get("min_side", min(image.size))),
            )


def _prepare_selection(
    selection: pd.DataFrame, columns: Sequence[str] | None
) -> list[str]:
    required = {"shard", "row_in_shard", "label"}
    missing = required - set(selection.columns)
    if missing:
        raise ValueError(f"selection is missing columns: {sorted(missing)}")

    read_columns = list(columns or [IMAGE_COLUMN, "label", "model_name"])
    if IMAGE_COLUMN not in read_columns:
        read_columns.append(IMAGE_COLUMN)
    return read_columns


def iter_selected_images(
    repo_id: str,
    selection: pd.DataFrame,
    revision: str = "main",
    columns: Sequence[str] | None = None,
    failed_shards: list[str] | None = None,
    workers: int = 8,
) -> Iterator[FetchedImage]:
    """Yield decoded images for every row in `selection`, fetched remotely.

    `selection` must carry `shard` and `row_in_shard` (both produced by the
    manifest scan) along with `label`.

    `workers` decodes that many images concurrently per row group (see
    `_decode_one` for why threads, not processes, give real speedup here).
    Set to 1 to force the old fully-sequential behaviour.

    A shard that still fails after `SHARD_RETRY_ATTEMPTS` attempts (each with
    a growing pause, on top of whatever retries huggingface_hub's own HTTP
    layer already did) is logged and skipped -- one persistently-unreachable
    shard must not abandon a multi-hour extraction run. If `failed_shards` is
    given, the name of every shard that was ultimately skipped is appended to
    it, so the caller can report exactly what's missing rather than the run
    silently under-delivering.
    """
    from huggingface_hub import HfFileSystem

    read_columns = _prepare_selection(selection, columns)
    filesystem = HfFileSystem()

    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for shard, group in selection.groupby("shard", sort=True):
            wanted = set(int(r) for r in group["row_in_shard"])
            # Manifest fields for these rows, so we never re-derive them from
            # the remote file.
            meta_by_row = group.set_index("row_in_shard").to_dict("index")

            path = f"datasets/{repo_id}@{revision}/{shard}"
            succeeded = False

            for attempt in range(1, SHARD_RETRY_ATTEMPTS + 1):
                try:
                    with filesystem.open(path, "rb") as handle:
                        parquet_file = pq.ParquetFile(handle)
                        yield from _read_rows_from_open_parquet(
                            parquet_file, shard, wanted, meta_by_row,
                            read_columns, executor,
                        )
                    succeeded = True
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt < SHARD_RETRY_ATTEMPTS:
                        wait = SHARD_RETRY_BACKOFF_SECONDS * attempt
                        logger.warning(
                            "shard %s attempt %d/%d failed (%s) -- retrying in %.0fs",
                            shard, attempt, SHARD_RETRY_ATTEMPTS, exc, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            "shard %s failed after %d attempts: %s",
                            shard, SHARD_RETRY_ATTEMPTS, exc,
                        )

            if not succeeded and failed_shards is not None:
                failed_shards.append(shard)
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def iter_selected_images_local(
    local_dir: Path,
    selection: pd.DataFrame,
    columns: Sequence[str] | None = None,
    failed_shards: list[str] | None = None,
    workers: int = 8,
) -> Iterator[FetchedImage]:
    """Yield decoded images for every row in `selection`, reading shards that
    are already fully downloaded on local disk.

    `local_dir` is the directory `scripts/download_full_dataset.py` (or any
    `snapshot_download` call) produced. `selection["shard"]` values are the
    canonical remote-style names (e.g. "data/HFCF_small_0.parquet") produced
    by the manifest scan -- resolved against `local_dir` by filename, so this
    works whether or not `local_dir` mirrors the "data/" subdirectory.

    `workers` decodes that many images concurrently per row group -- see
    `_decode_one`. Set to 1 for the old fully-sequential behaviour. Note this
    parallelizes the CPU-bound decode/convert step only; if the real
    bottleneck is reading `local_dir` itself (e.g. it is actually a network
    filesystem mount rather than local disk), more decode workers will not
    help much -- that would need concurrent shard reads instead.

    No network calls happen here at all, so there is nothing to retry: a
    missing or unreadable local file means the earlier download step didn't
    actually get that shard, which `failed_shards` (if given) surfaces so the
    caller can tell the user which shard(s) need re-downloading.
    """
    read_columns = _prepare_selection(selection, columns)
    local_dir = Path(local_dir)

    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for shard, group in selection.groupby("shard", sort=True):
            wanted = set(int(r) for r in group["row_in_shard"])
            meta_by_row = group.set_index("row_in_shard").to_dict("index")

            candidates = [local_dir / shard, local_dir / Path(shard).name]
            local_path = next((p for p in candidates if p.exists()), None)

            if local_path is None:
                logger.error(
                    "local shard not found: %s (looked for %s)",
                    shard, " and ".join(str(c) for c in candidates),
                )
                if failed_shards is not None:
                    failed_shards.append(shard)
                continue

            try:
                parquet_file = pq.ParquetFile(local_path)
                yield from _read_rows_from_open_parquet(
                    parquet_file, shard, wanted, meta_by_row,
                    read_columns, executor,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("local shard %s failed: %s", shard, exc)
                if failed_shards is not None:
                    failed_shards.append(shard)
                continue
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
