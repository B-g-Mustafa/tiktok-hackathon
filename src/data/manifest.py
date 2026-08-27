"""Build a metadata manifest for a Hugging Face parquet dataset.

Why this exists
---------------
Community Forensics-Small is 260 GB of image bytes wrapped around a few tens of
megabytes of metadata. Because parquet is *columnar*, we can read the metadata
columns over HTTP range requests and never touch the `image_data` column at all
-- the whole dataset's metadata costs ~50-100 MB instead of 260 GB.

That matters for correctness, not just speed. The shards are ordered by source
and architecture rather than shuffled: shard 0 indexes ~8,979 generated against
~1,563 authentic images, with every real coming from FFHQ and every fake from a
latent-diffusion model. Any pipeline that "downloads the first few shards to get
started" trains on a class-imbalanced, single-architecture, single-source subset
that looks perfectly healthy in aggregate statistics.

With a full manifest in hand, sampling becomes a local dataframe operation: we
can balance classes, hold out whole generator families, match resolutions
between classes, and exclude contaminated sources -- all before fetching a
single image.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "METADATA_COLUMNS",
    "ShardScanError",
    "scan_shard",
    "build_manifest",
    "list_parquet_files",
]

# Everything except `image_data` (the multi-MB blob) and `prompt` (long free
# text we do not need for sampling). Keeping this list explicit means a schema
# change surfaces as a clear error rather than a silent 260 GB download.
METADATA_COLUMNS = [
    "image_name",
    "format",
    "resolution",
    "mode",
    "model_name",
    "real_source",
    "subset",
    "split",
    "label",
    "architecture",
]


class ShardScanError(RuntimeError):
    """A shard could not be read after retries."""


def list_parquet_files(repo_id: str, revision: str = "main") -> list[str]:
    """All parquet paths in a dataset repo, in sorted order."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return sorted(f for f in files if f.endswith(".parquet"))


def _available_columns(schema: pa.Schema, requested: Sequence[str]) -> list[str]:
    """Intersect requested columns with what the shard actually has.

    Datasets get re-uploaded with changed schemas; we would rather scan the
    columns that exist than fail the whole run on one missing field.
    """
    present = set(schema.names)
    return [c for c in requested if c in present]


def scan_shard(
    repo_id: str,
    path: str,
    columns: Sequence[str] = METADATA_COLUMNS,
    revision: str = "main",
    retries: int = 3,
    backoff: float = 2.0,
) -> pa.Table:
    """Read only `columns` from one parquet shard.

    Adds a `shard` column and a `row_in_shard` index so a later pass can fetch
    exactly the selected rows without re-scanning.
    """
    from huggingface_hub import HfFileSystem

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            fs = HfFileSystem()
            full_path = f"datasets/{repo_id}@{revision}/{path}"

            with fs.open(full_path, "rb") as handle:
                parquet_file = pq.ParquetFile(handle)
                usable = _available_columns(parquet_file.schema_arrow, columns)
                if not usable:
                    raise ShardScanError(
                        f"{path}: none of the requested columns exist "
                        f"(schema: {parquet_file.schema_arrow.names})"
                    )
                table = parquet_file.read(columns=usable)

            n_rows = table.num_rows
            table = table.append_column(
                "shard", pa.array([path] * n_rows, type=pa.string())
            )
            table = table.append_column(
                "row_in_shard", pa.array(range(n_rows), type=pa.int64())
            )
            return table

        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last_error = exc
            if attempt < retries:
                time.sleep(backoff * attempt)

    raise ShardScanError(f"{path}: failed after {retries} attempts: {last_error}")


@dataclass
class ManifestStats:
    n_rows: int
    n_shards: int
    n_failed: int
    failed_shards: list[str]


def build_manifest(
    repo_id: str,
    output_path: Path,
    columns: Sequence[str] = METADATA_COLUMNS,
    revision: str = "main",
    max_shards: int | None = None,
    workers: int = 8,
    progress: Callable[[str], None] | None = None,
) -> ManifestStats:
    """Scan every shard's metadata and write a single manifest parquet.

    Shards are scanned in parallel because the work is network-bound. A shard
    that fails all retries is recorded and skipped rather than aborting the run
    -- a manifest missing 2 of 186 shards is still vastly better than none, and
    the failures are reported so they can be re-run.
    """
    log = progress or (lambda _msg: None)

    paths = list_parquet_files(repo_id, revision=revision)
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise ShardScanError(f"{repo_id}: no parquet files found")

    log(f"scanning {len(paths)} shards with {workers} workers")

    tables: list[pa.Table] = []
    failed: list[str] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scan_shard, repo_id, path, columns, revision): path
            for path in paths
        }
        for future in as_completed(futures):
            path = futures[future]
            completed += 1
            try:
                tables.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failed.append(path)
                log(f"  [{completed}/{len(paths)}] FAILED {path}: {exc}")
            else:
                if completed % 20 == 0 or completed == len(paths):
                    log(f"  [{completed}/{len(paths)}] scanned")

    if not tables:
        raise ShardScanError(f"{repo_id}: every shard failed to scan")

    # Shards may disagree on column order (or be missing a column); unify_schemas
    # plus promote_options handles both without silently dropping data.
    combined = pa.concat_tables(tables, promote_options="default")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, output_path)

    return ManifestStats(
        n_rows=combined.num_rows,
        n_shards=len(tables),
        n_failed=len(failed),
        failed_shards=failed,
    )
