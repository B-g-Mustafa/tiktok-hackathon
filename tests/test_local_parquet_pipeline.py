"""Tests for the local-disk parquet pipeline: scanning and fetching from
shards that have already been fully downloaded (e.g. via
`scripts/download_full_dataset.py`), with no network involved.

These exercise real pyarrow/parquet I/O against synthetic shards built to
match Community Forensics' actual schema, rather than mocking anything --
the whole point of this path is "does reading a real local parquet file
work," which a mock can't tell you.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from src.data.manifest import ShardScanError, build_manifest_from_local
from src.data.parquet_images import iter_selected_images_local


def _make_shard(path, label: int, n: int, model_names: list[str], seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            buf, format="PNG"
        )
        rows.append(
            {
                "image_data": buf.getvalue(),
                "label": label,
                "model_name": model_names[i % len(model_names)],
                "resolution": [32, 32],
                "format": "PNG",
                "real_source": "N/A" if label == 1 else "FFHQ",
                "architecture": "LatDiff" if label == 1 else "Real",
                "subset": "Manual",
                "split": "train",
                "image_name": f"{i:03d}.png",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def local_shards(tmp_path):
    """A directory shaped like a snapshot_download output: <root>/data/*.parquet."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_shard(data_dir / "HFCF_small_0.parquet", label=1, n=6, model_names=["genA", "genB"])
    _make_shard(data_dir / "HFCF_small_1.parquet", label=0, n=6, model_names=["FFHQ"], seed=1)
    return tmp_path


# ---------------------------------------------------------------------------
# build_manifest_from_local
# ---------------------------------------------------------------------------


def test_local_manifest_shard_names_match_remote_convention(tmp_path, local_shards):
    """A manifest scanned locally must use the SAME shard-name convention
    (e.g. "data/HFCF_small_0.parquet") as a manifest scanned remotely --
    that's what lets the two be interchangeable everywhere downstream."""
    out = tmp_path / "manifest.parquet"
    stats = build_manifest_from_local(local_shards, out)

    assert stats.n_rows == 12
    assert stats.n_shards == 2
    assert stats.n_failed == 0

    manifest = pd.read_parquet(out)
    assert set(manifest["shard"]) == {
        "data/HFCF_small_0.parquet",
        "data/HFCF_small_1.parquet",
    }


def test_local_manifest_row_in_shard_is_zero_indexed_per_shard(tmp_path, local_shards):
    out = tmp_path / "manifest.parquet"
    build_manifest_from_local(local_shards, out)
    manifest = pd.read_parquet(out)

    for shard, group in manifest.groupby("shard"):
        assert sorted(group["row_in_shard"]) == list(range(len(group)))


def test_local_manifest_preserves_labels_and_generators(tmp_path, local_shards):
    out = tmp_path / "manifest.parquet"
    build_manifest_from_local(local_shards, out)
    manifest = pd.read_parquet(out)

    generated = manifest.loc[manifest["label"] == 1]
    authentic = manifest.loc[manifest["label"] == 0]
    assert set(generated["model_name"]) == {"genA", "genB"}
    assert set(authentic["model_name"]) == {"FFHQ"}


def test_local_manifest_raises_on_empty_directory(tmp_path):
    with pytest.raises(ShardScanError, match="no .parquet files"):
        build_manifest_from_local(tmp_path, tmp_path / "out.parquet")


def test_local_manifest_skips_and_reports_a_corrupt_shard(tmp_path, local_shards):
    (local_shards / "data" / "corrupt.parquet").write_bytes(b"not a real parquet file")

    out = tmp_path / "manifest.parquet"
    stats = build_manifest_from_local(local_shards, out)

    assert stats.n_shards == 2  # the two good ones
    assert stats.n_failed == 1
    assert "data/corrupt.parquet" in stats.failed_shards


# ---------------------------------------------------------------------------
# iter_selected_images_local
# ---------------------------------------------------------------------------


def test_fetches_correct_images_and_decodes_them(tmp_path, local_shards):
    out = tmp_path / "manifest.parquet"
    build_manifest_from_local(local_shards, out)
    selection = pd.read_parquet(out)

    fetched = list(iter_selected_images_local(local_shards, selection))
    assert len(fetched) == 12
    for item in fetched:
        assert item.image.size == (32, 32)
        assert item.image.mode == "RGB"

    keys = {f.key for f in fetched}
    assert len(keys) == 12  # all unique


def test_fetches_only_the_requested_subset(tmp_path, local_shards):
    out = tmp_path / "manifest.parquet"
    build_manifest_from_local(local_shards, out)
    selection = pd.read_parquet(out)

    subset = selection[selection["label"] == 1]  # only the generated shard
    fetched = list(iter_selected_images_local(local_shards, subset))
    assert len(fetched) == 6
    assert all(f.label == 1 for f in fetched)


def test_missing_shard_is_reported_not_crashed(tmp_path, local_shards):
    """Simulates an incomplete download: the manifest references a shard
    that was never actually fetched to this directory."""
    out = tmp_path / "manifest.parquet"
    build_manifest_from_local(local_shards, out)
    selection = pd.read_parquet(out)

    (local_shards / "data" / "HFCF_small_1.parquet").unlink()

    failed: list[str] = []
    fetched = list(
        iter_selected_images_local(local_shards, selection, failed_shards=failed)
    )
    assert len(fetched) == 6  # only shard 0's rows
    assert failed == ["data/HFCF_small_1.parquet"]


def test_rejects_selection_missing_required_columns(tmp_path, local_shards):
    bad_selection = pd.DataFrame([{"shard": "x", "label": 1}])  # no row_in_shard
    with pytest.raises(ValueError, match="missing columns"):
        list(iter_selected_images_local(local_shards, bad_selection))


# ---------------------------------------------------------------------------
# materialize(local_dir=...) integration
# ---------------------------------------------------------------------------


def test_materialize_from_local_dir(tmp_path, local_shards):
    from src.data.local_dataset import materialize

    manifest_path = tmp_path / "manifest.parquet"
    build_manifest_from_local(local_shards, manifest_path)
    selection = pd.read_parquet(manifest_path)

    out_dir = tmp_path / "materialized"
    stats = materialize(
        repo_id="unused-for-local",
        selection=selection,
        output_dir=out_dir,
        local_dir=local_shards,
        show_progress=False,
    )

    assert stats.n_written == 12
    assert stats.failed_shards == []

    result_manifest = pd.read_parquet(out_dir / "manifest.parquet")
    assert len(result_manifest) == 12
    for path in result_manifest["path"]:
        assert Path(path).is_file()
