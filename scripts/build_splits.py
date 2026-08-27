#!/usr/bin/env python3
"""Build training and evaluation splits, and prove they are not trivially
separable.

Two constraints shape this script, both discovered by auditing the data rather
than assumed:

**Shortcuts.** Image size alone separates the classes at 0.90 AUROC across the
full dataset. Every stage below reports a canary so the effect of each
mitigation is visible, and the run FAILS if the final training split is still
separable by scale.

**Download economics.** Each shard is a single ~4.1 GB parquet row group, and
the dataset is cleanly partitioned (93 shards all-generated, 92 all-authentic).
Fetching one image downloads its whole shard, so a selection spread across all
186 shards costs ~763 GB to materialise. We therefore choose whole shards --
using greedy generator coverage so concentrating the download does not
concentrate the generators -- and sample within them.

Usage:
    python scripts/build_splits.py
    python scripts/build_splits.py --shards-per-class 8    # more data, ~66 GB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.data.sampling import (  # noqa: E402
    LABEL_AUTHENTIC,
    LABEL_GENERATED,
    add_content_column,
    add_size_columns,
    balance_classes,
    content_matched_pool,
    generator_disjoint_split,
    load_manifest,
    min_side_matched_pool,
    plan_shards,
    restrict_to_shards,
    summarize,
)
from src.evaluation.shortcut_controls import (  # noqa: E402
    resolution_canary,
    scale_canary,
)

DEFAULT_MANIFEST = Path("artifacts/manifests/community_forensics_small.parquet")
DEFAULT_OUTPUT = Path("artifacts/splits")

CANARY_FAIL_THRESHOLD = 0.60

# Training operates at the scale where both classes are plentiful; Control D at
# the scale where content-matched faces exist.
TRAIN_MIN_SIDE = 512
CONTROL_MIN_SIDE = 1024


def canary_of(frame: pd.DataFrame, name: str) -> tuple[float, float]:
    """Report both size canaries for one stage.

    `resolution` is the gate for a whole-image pipeline; `scale` is the gate for
    ours, because fixed-size native cropping hides everything but the shorter
    side.
    """
    if len(frame) == 0:
        print(f"  {name:<36} (empty)")
        return float("nan"), float("nan")

    frame = add_size_columns(frame)
    usable = frame.loc[frame["min_side"] > 0]

    if (usable["label"] == LABEL_AUTHENTIC).sum() == 0 or (
        usable["label"] == LABEL_GENERATED
    ).sum() == 0:
        print(f"  {name:<36} n={len(frame):>7,}  (single class -- undefined)")
        return float("nan"), float("nan")

    res = resolution_canary(usable["width"], usable["height"], usable["label"])
    scale = scale_canary(usable["min_side"], usable["label"])

    res_eff = max(res.auroc, 1.0 - res.auroc)
    scale_eff = max(scale.auroc, 1.0 - scale.auroc)
    flag = "  <-- SCALE SHORTCUT" if scale.is_alarming else ""
    print(
        f"  {name:<36} n={len(frame):>7,}  "
        f"resolution={res_eff:.4f}  scale={scale_eff:.4f}{flag}"
    )
    return res_eff, scale_eff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--shards-per-class",
        type=int,
        default=5,
        help="Whole shards to download per class for training (~4.1 GB each).",
    )
    parser.add_argument(
        "--control-shards-per-class",
        type=int,
        default=3,
        help="Whole shards per class for the content-matched control.",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}")
        print("Run: python scripts/build_manifest.py")
        return 2

    print("=" * 78)
    print("BUILD SPLITS")
    print("=" * 78)

    manifest = add_content_column(add_size_columns(load_manifest(args.manifest)))
    print(f"\nmanifest: {len(manifest):,} rows")

    # -- shard planning -----------------------------------------------------
    train_plan = plan_shards(
        manifest, n_shards_per_class=args.shards_per_class, min_side=TRAIN_MIN_SIDE
    )
    control_plan = plan_shards(
        manifest.loc[manifest["content"] == "face"],
        n_shards_per_class=args.control_shards_per_class,
        min_side=CONTROL_MIN_SIDE,
    )

    print("\nDOWNLOAD PLAN (each shard is one ~4.1 GB parquet row group)")
    print(f"  train   (min_side={TRAIN_MIN_SIDE}): {train_plan}")
    print(f"  controlD(min_side={CONTROL_MIN_SIDE}): {control_plan}")
    total_gb = train_plan.estimated_gb + control_plan.estimated_gb
    print(f"  TOTAL to download: ~{total_gb:.1f} GB "
          f"({len(train_plan.shards) + len(control_plan.shards)} shards of 186)")

    # -- training pool ------------------------------------------------------
    print("\nSize canaries at each mitigation stage")
    print("  resolution = width x height    (matters only for whole-image input)")
    print("  scale      = min(width,height) (the only cue surviving our crops)")

    canaries = {"raw_manifest": canary_of(manifest, "raw manifest")}

    pool = restrict_to_shards(manifest, train_plan.shards)
    canaries["after_shard_selection"] = canary_of(pool, "after shard selection")

    pool = min_side_matched_pool(pool, min_crop_size=args.crop_size)
    canaries["after_scale_matching"] = canary_of(pool, "after min_side matching")

    balanced = balance_classes(
        pool, n_per_class=None, seed=args.seed, stratify_column="min_side"
    )
    canaries["after_balancing"] = canary_of(balanced, "after stratified balancing")

    if balanced.empty:
        print("\nERROR: balanced pool is empty.")
        return 2

    train, cross_gen = generator_disjoint_split(
        balanced, holdout_fraction=args.holdout_fraction, seed=args.seed
    )
    canaries["train"] = canary_of(train, "TRAIN split")
    canaries["cross_generator"] = canary_of(cross_gen, "CROSS-GENERATOR split")

    # -- Control D ----------------------------------------------------------
    control_frame = restrict_to_shards(manifest, control_plan.shards)
    content_control = content_matched_pool(
        control_frame, min_crop_size=args.crop_size
    )
    canaries["content_matched_control"] = canary_of(
        content_control, "CONTENT-MATCHED control (D)"
    )

    # -- report -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SPLIT COMPOSITION")
    print("=" * 78)
    parts = (
        (train, "train"),
        (cross_gen, "cross_generator"),
        (content_control, "content_matched_control"),
    )
    for part, name in parts:
        print(f"  {summarize(part, name)}")

    if len(content_control):
        print("\n  Control D composition (content x scale, classes balanced)")
        for (content, min_side), count in (
            content_control.groupby(["content", "min_side"]).size().sort_index().items()
        ):
            generators = content_control.loc[
                (content_control["content"] == content)
                & (content_control["label"] == LABEL_GENERATED),
                "model_name",
            ].unique()
            print(
                f"    {content:<7} min_side={min_side:<6} {count:>7,} images  "
                f"vs {sorted(generators)}"
            )

    train_gens = set(train.loc[train["label"] == LABEL_GENERATED, "model_name"])
    test_gens = set(cross_gen.loc[cross_gen["label"] == LABEL_GENERATED, "model_name"])
    overlap = train_gens & test_gens
    print(
        f"\n  generator overlap train vs cross-generator: {len(overlap)}"
        + ("  OK" if not overlap else "  <-- LEAK")
    )

    # -- persist ------------------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)
    for part, name in parts:
        path = args.output / f"{name}.parquet"
        part.to_parquet(path, index=False)
        print(f"  wrote {len(part):>7,} rows -> {path}")

    (args.output / "report.json").write_text(
        json.dumps(
            {
                "canaries": {k: list(v) for k, v in canaries.items()},
                "train_shards": train_plan.shards,
                "control_shards": control_plan.shards,
                "estimated_download_gb": round(total_gb, 1),
                "n_train": len(train),
                "n_cross_generator": len(cross_gen),
                "n_content_control": len(content_control),
                "generator_overlap": len(overlap),
                "crop_size": args.crop_size,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    # -- gate ---------------------------------------------------------------
    print("\n" + "=" * 78)
    _, train_scale = canaries["train"]
    if train_scale == train_scale and train_scale >= CANARY_FAIL_THRESHOLD:
        print(
            f"FAIL: training split is still separable by scale "
            f"({train_scale:.4f}).\n"
            f"      Training on it would measure scale, not forensics."
        )
        return 1

    print(
        f"PASS: training split scale canary is {train_scale:.4f} "
        f"(threshold {CANARY_FAIL_THRESHOLD}).\n"
        f"      Size no longer carries the label under fixed-size native crops.\n"
        f"      Content bias is measured separately by Control D."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
