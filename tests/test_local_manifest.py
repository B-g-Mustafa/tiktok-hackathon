"""Tests for building a manifest from a plain local directory (GenImage etc).

This is the entry point for training on any dataset that isn't Community
Forensics, so the tests focus on exactly the properties a user's directory
tree needs to satisfy for training to work at all: labels come from the right
folder, unlabeled/unreadable files degrade gracefully instead of crashing a
large scan, and the `--split` filter actually isolates train from val rather
than silently merging them (which is what GenImage's layout invites if you
don't filter).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.local_manifest import build_local_manifest


def make_genimage_tree(root, generators=("BigGAN", "ADM"), splits=("train", "val"), n=3):
    """A GenImage-shaped tree: <root>/<generator>/<split>/{ai,nature}/*.png"""
    rng = np.random.default_rng(0)
    for generator in generators:
        for split in splits:
            for cls in ("ai", "nature"):
                d = root / generator / split / cls
                d.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    Image.fromarray(
                        rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                    ).save(d / f"{i:03d}.png")
    return root


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------


def test_genimage_layout_labels_correctly(tmp_path):
    root = make_genimage_tree(tmp_path / "genimage")
    stats = build_local_manifest(root, tmp_path / "out")

    assert stats.n_generated == 2 * 2 * 3  # 2 generators x 2 splits x 3 "ai"
    assert stats.n_authentic == 2 * 2 * 3  # x 3 "nature"

    manifest = pd.read_parquet(stats.manifest_path)
    ai_rows = manifest[manifest["key"].str.contains("/ai/")]
    nature_rows = manifest[manifest["key"].str.contains("/nature/")]
    assert (ai_rows["label"] == 1).all()
    assert (nature_rows["label"] == 0).all()


@pytest.mark.parametrize(
    "real_name,fake_name",
    [("real", "fake"), ("authentic", "synthetic"), ("Nature", "AI")],  # case-insensitive
)
def test_generic_and_case_insensitive_naming(tmp_path, real_name, fake_name):
    root = tmp_path / "data"
    rng = np.random.default_rng(1)
    for name, label in ((real_name, 0), (fake_name, 1)):
        d = root / name
        d.mkdir(parents=True)
        Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(
            d / "img.png"
        )

    stats = build_local_manifest(root, tmp_path / "out")
    assert stats.n_authentic == 1
    assert stats.n_generated == 1


def test_unlabeled_files_are_skipped_not_crashed(tmp_path):
    root = tmp_path / "data"
    (root / "misc").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(root / "misc" / "orphan.png")
    (root / "real").mkdir()
    Image.new("RGB", (16, 16)).save(root / "real" / "r.png")

    stats = build_local_manifest(root, tmp_path / "out")
    assert stats.n_skipped_unlabeled == 1
    assert stats.n_authentic == 1


def test_deepest_label_directory_wins(tmp_path):
    """A pathological path containing both markers (e.g. a 'fake' dataset name
    with a 'real' subfolder) should resolve to the innermost -- most
    specific -- marker, not the outermost."""
    root = tmp_path / "data"
    d = root / "fake_dataset_name" / "real"
    d.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(d / "img.png")

    stats = build_local_manifest(root, tmp_path / "out")
    assert stats.n_authentic == 1
    assert stats.n_generated == 0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_corrupt_file_is_skipped(tmp_path):
    root = tmp_path / "data"
    (root / "real").mkdir(parents=True)
    (root / "real" / "corrupt.png").write_bytes(b"not a real png")
    Image.new("RGB", (16, 16)).save(root / "real" / "good.png")

    stats = build_local_manifest(root, tmp_path / "out")
    assert stats.n_skipped_unreadable == 1
    assert stats.n_authentic == 1


def test_non_image_files_are_ignored(tmp_path):
    root = tmp_path / "data"
    (root / "fake").mkdir(parents=True)
    (root / "fake" / "readme.txt").write_text("not an image")
    Image.new("RGB", (16, 16)).save(root / "fake" / "img.png")

    stats = build_local_manifest(root, tmp_path / "out")
    assert stats.n_generated == 1


def test_missing_root_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        build_local_manifest(tmp_path / "does_not_exist", tmp_path / "out")


def test_single_class_manifest_warns_but_does_not_crash(tmp_path, caplog):
    root = tmp_path / "data"
    (root / "real").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(root / "real" / "img.png")

    stats = build_local_manifest(root, tmp_path / "out")
    assert stats.n_authentic == 1 and stats.n_generated == 0


def test_limit_per_class_caps_output(tmp_path):
    root = tmp_path / "data"
    rng = np.random.default_rng(2)
    for name in ("real", "fake"):
        d = root / name
        d.mkdir(parents=True)
        for i in range(10):
            Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(
                d / f"{i}.png"
            )

    stats = build_local_manifest(root, tmp_path / "out", limit_per_class=3)
    assert stats.n_authentic == 3
    assert stats.n_generated == 3


# ---------------------------------------------------------------------------
# --split filter -- the GenImage-specific behavior
# ---------------------------------------------------------------------------


def test_split_filter_isolates_train_from_val(tmp_path):
    root = make_genimage_tree(tmp_path / "genimage", n=2)

    train_stats = build_local_manifest(root, tmp_path / "train_out", split_filter="train")
    val_stats = build_local_manifest(root, tmp_path / "val_out", split_filter="val")

    train_manifest = pd.read_parquet(train_stats.manifest_path)
    val_manifest = pd.read_parquet(val_stats.manifest_path)

    assert not train_manifest["key"].str.contains("/val/").any()
    assert not val_manifest["key"].str.contains("/train/").any()
    # No overlap between the two manifests.
    assert set(train_manifest["key"]).isdisjoint(set(val_manifest["key"]))


def test_split_filter_is_case_insensitive(tmp_path):
    root = tmp_path / "data"
    (root / "gen" / "TRAIN" / "real").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(root / "gen" / "TRAIN" / "real" / "img.png")

    stats = build_local_manifest(root, tmp_path / "out", split_filter="train")
    assert stats.n_authentic == 1


def test_no_split_filter_merges_everything(tmp_path):
    """Without --split, pointing at the dataset root legitimately merges train
    and val -- this documents that behavior rather than silently surprising
    a caller who forgot the flag."""
    root = make_genimage_tree(tmp_path / "genimage", n=2)
    stats = build_local_manifest(root, tmp_path / "out")
    manifest = pd.read_parquet(stats.manifest_path)
    assert manifest["key"].str.contains("/train/").any()
    assert manifest["key"].str.contains("/val/").any()


# ---------------------------------------------------------------------------
# Generator tag recovery
# ---------------------------------------------------------------------------


def test_generator_tag_ignores_extract_script_output_folder(tmp_path):
    """scripts/extract_genimage.py extracts into <category>/extracted/, adding
    a path segment between the generator name and train/val. The generator
    tag must still resolve to just the generator, not "ADM/extracted"."""
    root = tmp_path / "genimage"
    d = root / "ADM" / "extracted" / "train" / "ai"
    d.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(d / "x.png")

    stats = build_local_manifest(root, tmp_path / "out")
    manifest = pd.read_parquet(stats.manifest_path)
    assert manifest.iloc[0]["model_name"] == "ADM"


def test_generator_tag_recovered_from_genimage_layout(tmp_path):
    root = make_genimage_tree(tmp_path / "genimage", generators=("Midjourney",), n=1)
    stats = build_local_manifest(root, tmp_path / "out")
    manifest = pd.read_parquet(stats.manifest_path)
    assert set(manifest["model_name"]) == {"Midjourney"}


def test_min_side_recorded(tmp_path):
    root = tmp_path / "data"
    (root / "real").mkdir(parents=True)
    Image.new("RGB", (100, 40)).save(root / "real" / "img.png")

    stats = build_local_manifest(root, tmp_path / "out")
    manifest = pd.read_parquet(stats.manifest_path)
    assert manifest.iloc[0]["min_side"] == 40
