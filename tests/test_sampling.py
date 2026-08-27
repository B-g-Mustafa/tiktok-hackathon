"""Tests for split construction.

A bug here does not crash anything -- it produces a split that looks fine and
quietly reports an inflated score. So these tests assert the properties we are
actually relying on: no generator crosses the train/test boundary, contaminated
sources are gone, and the resolution-matched pool really does carry zero size
signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.sampling import (
    LABEL_AUTHENTIC,
    LABEL_GENERATED,
    add_size_columns,
    balance_classes,
    exclude_contaminated_sources,
    generator_disjoint_split,
    native_crop_pool,
    resolution_matched_pool,
    summarize,
)
from src.evaluation.shortcut_controls import resolution_canary


def make_frame() -> pd.DataFrame:
    """A miniature manifest reproducing the real dataset's pathologies:
    authentic images large and varied, generated images uniformly 512px."""
    rows = []

    # Authentic: FFHQ at 1024, COCO at assorted sizes, LandscapesHQ at 1024.
    for i in range(20):
        rows.append(
            {
                "image_name": f"ffhq_{i}.png",
                "resolution": [1024, 1024],
                "label": LABEL_AUTHENTIC,
                "model_name": "FFHQ",
                "format": "PNG",
                "architecture": "Real",
            }
        )
    for i in range(15):
        rows.append(
            {
                "image_name": f"coco_{i}.jpg",
                "resolution": [640, 480],
                "label": LABEL_AUTHENTIC,
                "model_name": "COCO",
                "format": "JPEG",
                "architecture": "Real",
            }
        )
    for i in range(10):
        rows.append(
            {
                "image_name": f"lhq_{i}.png",
                "resolution": [512, 512],
                "label": LABEL_AUTHENTIC,
                "model_name": "LandscapesHQ",
                "format": "PNG",
                "architecture": "Real",
            }
        )

    # Generated: 6 generators, all at 512, plus a few tiny 128px ones.
    for gen in range(6):
        for i in range(10):
            rows.append(
                {
                    "image_name": f"gen{gen}_{i}.png",
                    "resolution": [512, 512],
                    "label": LABEL_GENERATED,
                    "model_name": f"generator/{gen}",
                    "format": "PNG",
                    "architecture": "LatDiff",
                }
            )
    for i in range(5):
        rows.append(
            {
                "image_name": f"tiny_{i}.png",
                "resolution": [128, 128],
                "label": LABEL_GENERATED,
                "model_name": "generator/tiny",
                "format": "PNG",
                "architecture": "GAN",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Size columns
# ---------------------------------------------------------------------------


def test_add_size_columns():
    frame = add_size_columns(make_frame())
    assert {"width", "height", "min_side"} <= set(frame.columns)
    ffhq = frame.loc[frame["model_name"] == "FFHQ"].iloc[0]
    assert (ffhq["width"], ffhq["height"], ffhq["min_side"]) == (1024, 1024, 1024)


def test_add_size_columns_handles_malformed_resolution():
    frame = pd.DataFrame(
        [{"resolution": None, "label": 0}, {"resolution": [64], "label": 1}]
    )
    out = add_size_columns(frame)
    assert (out["min_side"] == 0).all()


def test_add_size_columns_does_not_mutate_input():
    frame = make_frame()
    add_size_columns(frame)
    assert "width" not in frame.columns


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------


def test_coco_authentic_images_are_excluded():
    """The organizer benchmark uses COCO val2017 as its authentic class."""
    out = exclude_contaminated_sources(make_frame())
    remaining = out.loc[out["label"] == LABEL_AUTHENTIC, "model_name"]
    assert "COCO" not in set(remaining)
    assert "FFHQ" in set(remaining)


def test_exclusion_is_case_insensitive_and_substring():
    frame = pd.DataFrame(
        [
            {"label": LABEL_AUTHENTIC, "model_name": "coco_val2017"},
            {"label": LABEL_AUTHENTIC, "model_name": "RAISE-1k"},
            {"label": LABEL_AUTHENTIC, "model_name": "FFHQ"},
        ]
    )
    out = exclude_contaminated_sources(frame)
    assert set(out["model_name"]) == {"FFHQ"}


def test_exclusion_keeps_generated_images():
    """A generated image is not unsafe merely for being COCO-prompted."""
    frame = pd.DataFrame(
        [{"label": LABEL_GENERATED, "model_name": "some/coco-finetune"}]
    )
    assert len(exclude_contaminated_sources(frame)) == 1


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------


def test_native_crop_pool_drops_undersized_images():
    """Upscaling to reach the crop size would fabricate interpolation
    artifacts -- exactly the signal the detector reads."""
    pool = native_crop_pool(make_frame(), crop_size=256)
    assert (pool["min_side"] >= 256).all()
    assert "generator/tiny" not in set(pool["model_name"])


def test_native_crop_pool_keeps_most_data():
    frame = make_frame()
    pool = native_crop_pool(frame, crop_size=256)
    assert len(pool) == len(frame) - 5  # only the 5 tiny images go


def test_resolution_matched_pool_has_zero_size_signal():
    """The whole point of the control: the canary must be blind on it."""
    pool = resolution_matched_pool(make_frame())
    assert len(pool) > 0

    pool = add_size_columns(pool)
    canary = resolution_canary(pool["width"], pool["height"], pool["label"])
    assert canary.auroc == pytest.approx(0.5, abs=1e-9)
    assert not canary.is_alarming


def test_resolution_matched_pool_is_balanced_within_each_resolution():
    pool = add_size_columns(resolution_matched_pool(make_frame()))
    for _, group in pool.groupby(["width", "height"]):
        assert (group["label"] == LABEL_AUTHENTIC).sum() == (
            group["label"] == LABEL_GENERATED
        ).sum()


def test_resolution_matched_pool_empty_when_no_overlap():
    frame = pd.DataFrame(
        [
            {"resolution": [1024, 1024], "label": LABEL_AUTHENTIC, "model_name": "a"},
            {"resolution": [512, 512], "label": LABEL_GENERATED, "model_name": "b"},
        ]
    )
    assert len(resolution_matched_pool(frame)) == 0


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------


def test_balance_classes_is_exactly_balanced():
    out = balance_classes(make_frame(), n_per_class=15, seed=0)
    assert (out["label"] == LABEL_AUTHENTIC).sum() == 15
    assert (out["label"] == LABEL_GENERATED).sum() == 15


def test_balance_classes_caps_at_available():
    out = balance_classes(make_frame(), n_per_class=10_000, seed=0)
    counts = out["label"].value_counts()
    assert counts[LABEL_AUTHENTIC] == counts[LABEL_GENERATED]


def test_balance_classes_maximizes_generator_coverage():
    """Round-robin sampling must touch every generator before repeating any.

    Uniform sampling would over-represent prolific generators; generator
    diversity is the property that drives unseen-generator generalization.
    """
    out = balance_classes(make_frame(), n_per_class=6, seed=0)
    generated = out.loc[out["label"] == LABEL_GENERATED]
    assert generated["model_name"].nunique() == 6


def test_balance_classes_is_reproducible():
    a = balance_classes(make_frame(), n_per_class=12, seed=7)
    b = balance_classes(make_frame(), n_per_class=12, seed=7)
    assert list(a["image_name"]) == list(b["image_name"])


def test_balance_classes_empty_input():
    empty = make_frame().iloc[0:0]
    assert len(balance_classes(empty)) == 0


# ---------------------------------------------------------------------------
# Generator-disjoint split
# ---------------------------------------------------------------------------


def test_no_generator_appears_in_both_halves():
    """The property the whole split exists to guarantee."""
    train, test = generator_disjoint_split(make_frame(), holdout_fraction=0.34, seed=0)

    train_gens = set(train.loc[train["label"] == LABEL_GENERATED, "model_name"])
    test_gens = set(test.loc[test["label"] == LABEL_GENERATED, "model_name"])

    assert train_gens and test_gens
    assert train_gens.isdisjoint(test_gens)


def test_split_preserves_all_rows():
    frame = make_frame()
    train, test = generator_disjoint_split(frame, holdout_fraction=0.3, seed=1)
    assert len(train) + len(test) == len(frame)


def test_split_puts_both_classes_on_each_side():
    train, test = generator_disjoint_split(make_frame(), holdout_fraction=0.34, seed=3)
    for part in (train, test):
        assert (part["label"] == LABEL_AUTHENTIC).sum() > 0
        assert (part["label"] == LABEL_GENERATED).sum() > 0


def test_split_is_reproducible():
    a, _ = generator_disjoint_split(make_frame(), seed=5)
    b, _ = generator_disjoint_split(make_frame(), seed=5)
    assert sorted(a["image_name"]) == sorted(b["image_name"])


def test_split_rejects_bad_fraction():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            generator_disjoint_split(make_frame(), holdout_fraction=bad)


def test_split_requires_multiple_generators():
    frame = pd.DataFrame(
        [
            {"label": LABEL_GENERATED, "model_name": "only-one", "image_name": "x"},
            {"label": LABEL_AUTHENTIC, "model_name": "FFHQ", "image_name": "y"},
        ]
    )
    with pytest.raises(ValueError, match="at least 2 generators"):
        generator_disjoint_split(frame)


def test_summarize_reports_counts():
    summary = summarize(make_frame(), "full")
    assert summary.n_authentic == 45
    assert summary.n_generated == 65
    assert summary.n_generators == 7
    assert "full" in str(summary)


# ---------------------------------------------------------------------------
# Shard planning
#
# Each Community Forensics shard is a single ~4.1 GB parquet row group, so
# fetching one image downloads the whole shard. Choosing shards well is the
# difference between a 66 GB download and a 763 GB one.
# ---------------------------------------------------------------------------


def _sharded_frame() -> pd.DataFrame:
    """Generators clustered by shard, as they are in the real dataset."""
    rows = []
    # Three generated shards. A and B hold the SAME single generator; C holds
    # three distinct ones. Ranking shards independently by generator count
    # would pick A and B (or duplicates); greedy coverage must prefer C.
    for shard, gens, n in (
        ("gen_a.parquet", ["g/solo"], 100),
        ("gen_b.parquet", ["g/solo"], 90),
        ("gen_c.parquet", ["g/x", "g/y", "g/z"], 60),
    ):
        for i in range(n):
            rows.append(
                {
                    "shard": shard,
                    "row_in_shard": i,
                    "label": LABEL_GENERATED,
                    "model_name": gens[i % len(gens)],
                    "resolution": [512, 512],
                    "image_name": f"{shard}-{i}",
                }
            )
    for shard in ("real_a.parquet", "real_b.parquet"):
        for i in range(80):
            rows.append(
                {
                    "shard": shard,
                    "row_in_shard": i,
                    "label": LABEL_AUTHENTIC,
                    "model_name": "LandscapesHQ",
                    "resolution": [512, 768],
                    "image_name": f"{shard}-{i}",
                }
            )
    return add_size_columns(pd.DataFrame(rows))


def test_plan_shards_prefers_generator_diversity_over_size():
    """Greedy marginal coverage must pick the diverse shard, not the two big
    shards that share one generator."""
    from src.data.sampling import plan_shards

    plan = plan_shards(_sharded_frame(), n_shards_per_class=2, min_side=512)
    assert "gen_c.parquet" in plan.shards
    assert plan.n_generators == 4  # g/solo + g/x + g/y + g/z


def test_plan_shards_reports_download_cost():
    from src.data.sampling import SHARD_SIZE_GB, plan_shards

    plan = plan_shards(_sharded_frame(), n_shards_per_class=2, min_side=512)
    assert len(plan.shards) == 4  # 2 generated + 2 authentic
    assert plan.estimated_gb == pytest.approx(4 * SHARD_SIZE_GB)


def test_plan_shards_respects_min_side_filter():
    from src.data.sampling import plan_shards

    plan = plan_shards(_sharded_frame(), n_shards_per_class=2, min_side=1024)
    assert plan.shards == [] and plan.n_images == 0


def test_plan_shards_counts_balanced_images_only():
    """The smaller class caps how many usable pairs a plan yields."""
    from src.data.sampling import plan_shards

    plan = plan_shards(_sharded_frame(), n_shards_per_class=1, min_side=512)
    # Greedy picks the DIVERSE generated shard (gen_c: 60 images, 3 generators)
    # over the larger single-generator one, so the generated side caps the
    # pairing: 60 generated vs 80 authentic -> 60 pairs = 120 images.
    assert plan.shards[0] == "gen_c.parquet"
    assert plan.n_images == 120


def test_restrict_to_shards_filters_rows():
    from src.data.sampling import restrict_to_shards

    frame = _sharded_frame()
    out = restrict_to_shards(frame, ["gen_c.parquet"])
    assert set(out["shard"]) == {"gen_c.parquet"}


def test_shard_selection_keeps_scale_matched():
    """A plan restricted to one min_side must leave no scale signal."""
    from src.data.sampling import min_side_matched_pool, plan_shards, restrict_to_shards
    from src.evaluation.shortcut_controls import scale_canary

    frame = _sharded_frame()
    plan = plan_shards(frame, n_shards_per_class=2, min_side=512)
    pool = add_size_columns(
        min_side_matched_pool(restrict_to_shards(frame, plan.shards), 256)
    )
    assert scale_canary(pool["min_side"], pool["label"]).auroc == pytest.approx(0.5)
