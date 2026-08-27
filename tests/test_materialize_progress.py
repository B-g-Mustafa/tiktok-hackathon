"""Tests for materialize()'s progress-tracking, resume, and checkpointing.

`iter_selected_images` is monkeypatched to a fast, network-free fake so these
run in milliseconds -- the network-fetch machinery itself was already verified
against the real dataset earlier in this project; what's new here is the tqdm
wiring, byte accounting, and resume logic layered on top of it, and those are
exactly what a monkeypatch isolates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import src.data.parquet_images as parquet_images


def make_fake_iter(seed: int = 0):
    """A drop-in replacement for iter_selected_images with no network calls."""
    rng = np.random.default_rng(seed)

    def fake_iter(repo_id, selection):
        for _, row in selection.iterrows():
            image = Image.fromarray(
                rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), mode="RGB"
            )
            yield parquet_images.FetchedImage(
                image=image,
                label=int(row["label"]),
                shard=row["shard"],
                row_in_shard=int(row["row_in_shard"]),
                model_name=row["model_name"],
                min_side=32,
            )

    return fake_iter


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(parquet_images, "iter_selected_images", make_fake_iter())
    # local_dataset.materialize imports iter_selected_images inside the
    # function body (lazy import), so patching the source module is enough --
    # it re-imports the (now patched) name each call.


def make_selection(n: int, n_shards: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "shard": f"shard{i % n_shards}.parquet",
                "row_in_shard": i,
                "label": i % 2,
                "model_name": f"generator{i % 2}",
            }
            for i in range(n)
        ]
    )


def test_materialize_writes_every_row(tmp_path):
    from src.data.local_dataset import materialize

    stats = materialize("fake/repo", make_selection(10), tmp_path / "out")
    assert stats.n_written == 10
    assert stats.n_failed == 0

    manifest = pd.read_parquet(stats.output_dir / "manifest.parquet")
    assert len(manifest) == 10
    assert set(manifest["label"]) == {0, 1}


def test_materialize_manifest_paths_are_real_files(tmp_path):
    from src.data.local_dataset import materialize

    stats = materialize("fake/repo", make_selection(5), tmp_path / "out")
    manifest = pd.read_parquet(stats.output_dir / "manifest.parquet")
    for path in manifest["path"]:
        assert Path(path).is_file()


def test_resume_skips_already_materialized_rows(tmp_path):
    """The central promise: interrupting and re-running must not re-download
    or duplicate anything already on disk."""
    from src.data.local_dataset import materialize

    out = tmp_path / "out"
    selection = make_selection(12)

    first = materialize("fake/repo", selection, out)
    assert first.n_written == 12

    # A second call on the SAME selection must skip everything -- if it
    # re-fetched, n_written would still read 12, so check no duplicate rows
    # were appended to the manifest instead.
    second = materialize("fake/repo", selection, out)
    assert second.n_written == 12

    manifest = pd.read_parquet(out / "manifest.parquet")
    assert len(manifest) == 12  # not 24
    assert manifest["key"].is_unique


def test_resume_after_partial_manifest(tmp_path):
    """Simulates a crash mid-run: manually write a manifest covering only
    some rows, then confirm the rest get picked up on the next call."""
    from src.data.local_dataset import LocalImageDataset  # noqa: F401 (import check)
    from src.data.local_dataset import materialize

    out = tmp_path / "out"
    full_selection = make_selection(10)

    materialize("fake/repo", full_selection.iloc[:4], out)
    assert len(pd.read_parquet(out / "manifest.parquet")) == 4

    materialize("fake/repo", full_selection, out)
    manifest = pd.read_parquet(out / "manifest.parquet")
    assert len(manifest) == 10
    assert manifest["key"].is_unique


def test_checkpoint_every_does_not_lose_rows(tmp_path):
    """Regardless of checkpoint frequency, every row must land in the final
    manifest -- checkpointing is a crash-safety net, not a filter."""
    from src.data.local_dataset import materialize

    stats = materialize(
        "fake/repo", make_selection(17), tmp_path / "out", checkpoint_every=5
    )
    assert stats.n_written == 17
    manifest = pd.read_parquet(stats.output_dir / "manifest.parquet")
    assert len(manifest) == 17


def test_show_progress_false_produces_same_result(tmp_path):
    """The tqdm bar must be cosmetic -- identical output with or without it."""
    from src.data.local_dataset import materialize

    with_bar = materialize(
        "fake/repo", make_selection(6), tmp_path / "a", show_progress=True
    )
    without_bar = materialize(
        "fake/repo", make_selection(6), tmp_path / "b", show_progress=False
    )
    assert with_bar.n_written == without_bar.n_written == 6


def test_empty_selection(tmp_path):
    from src.data.local_dataset import materialize

    stats = materialize("fake/repo", make_selection(0), tmp_path / "out")
    assert stats.n_written == 0
    manifest = pd.read_parquet(stats.output_dir / "manifest.parquet")
    assert manifest.empty
