"""Fetch selected image bytes from remote parquet shards.

The manifest tells us exactly which rows we want; this module gets just those
images without downloading the dataset.

Parquet stores data in row groups, and pyarrow can read an individual row group
over HTTP range requests. Community Forensics-Small is ~1.4 GB per shard, so
reading only the row groups that actually contain our selected rows is the
difference between a 260 GB download and a few tens of GB.

Rows are yielded in shard order rather than manifest order. Feature extraction
does not care about ordering, and every row carries its own identifiers, so
matching features back to labels is done by key rather than by position.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Iterator, Sequence

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

__all__ = ["FetchedImage", "iter_selected_images", "IMAGE_COLUMN"]

logger = logging.getLogger(__name__)

IMAGE_COLUMN = "image_data"


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


def iter_selected_images(
    repo_id: str,
    selection: pd.DataFrame,
    revision: str = "main",
    columns: Sequence[str] | None = None,
) -> Iterator[FetchedImage]:
    """Yield decoded images for every row in `selection`.

    `selection` must carry `shard` and `row_in_shard` (both produced by the
    manifest scan) along with `label`.

    A shard or row that fails to load is logged and skipped: one corrupt record
    must not abandon a multi-hour extraction run.
    """
    from huggingface_hub import HfFileSystem

    required = {"shard", "row_in_shard", "label"}
    missing = required - set(selection.columns)
    if missing:
        raise ValueError(f"selection is missing columns: {sorted(missing)}")

    read_columns = list(columns or [IMAGE_COLUMN, "label", "model_name"])
    if IMAGE_COLUMN not in read_columns:
        read_columns.append(IMAGE_COLUMN)

    filesystem = HfFileSystem()

    for shard, group in selection.groupby("shard", sort=True):
        wanted = set(int(r) for r in group["row_in_shard"])
        # Manifest fields for these rows, so we never re-derive them from the
        # remote file.
        meta_by_row = group.set_index("row_in_shard").to_dict("index")

        path = f"datasets/{repo_id}@{revision}/{shard}"

        try:
            with filesystem.open(path, "rb") as handle:
                parquet_file = pq.ParquetFile(handle)
                available = [
                    c for c in read_columns if c in parquet_file.schema_arrow.names
                ]
                targets = _row_groups_for(parquet_file, wanted)

                for group_index, offsets in targets.items():
                    table = parquet_file.read_row_group(
                        group_index, columns=available
                    )
                    image_bytes = table.column(IMAGE_COLUMN).to_pylist()

                    # Absolute row index for each offset in this group.
                    group_start = sum(
                        parquet_file.metadata.row_group(i).num_rows
                        for i in range(group_index)
                    )

                    for offset in offsets:
                        absolute = group_start + offset
                        meta = meta_by_row.get(absolute)
                        if meta is None:
                            continue

                        raw = image_bytes[offset]
                        if raw is None:
                            logger.warning("null image at %s#%d", shard, absolute)
                            continue

                        try:
                            image = Image.open(io.BytesIO(raw))
                            image.load()
                            image = image.convert("RGB")
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "decode failed at %s#%d: %s", shard, absolute, exc
                            )
                            continue

                        yield FetchedImage(
                            image=image,
                            label=int(meta["label"]),
                            shard=shard,
                            row_in_shard=absolute,
                            model_name=str(meta.get("model_name", "")),
                            min_side=int(meta.get("min_side", min(image.size))),
                        )

        except Exception as exc:  # noqa: BLE001
            logger.error("shard %s failed: %s", shard, exc)
            continue
