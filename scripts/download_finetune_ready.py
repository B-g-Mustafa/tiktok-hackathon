#!/usr/bin/env python3
"""Download a disk-budgeted, balanced, scale-matched slice of a Community
Forensics dataset, materialized directly into the manifest format
`finetune_lora.py` reads. One command, no separate prep step after.

Why this and not a naive "download N images" script
-----------------------------------------------------
Community Forensics-Small has a resolution shortcut severe enough that a
classifier reading nothing but image dimensions gets ~0.90 AUROC on the raw
dataset (see experiments/LEDGER.md, EXP-000/EXP-002/EXP-003). Naively grabbing
the first N balanced images would reproduce that shortcut. This script reuses
the exact fix already built and tested for the full pipeline:

  1. drop contaminated real sources (COCO train2017 / RAISE)
  2. pick whole SHARDS via greedy generator-coverage (each shard is one
     ~4.1 GB parquet row group -- fetching any row downloads the whole shard,
     so shard choice IS the download-size lever)
  3. match real vs. generated on `min_side` (the only size cue that survives
     the fixed-size native crops the model actually sees)
  4. balance classes WITHIN each size bucket (stratified, not global -- global
     balancing was measured to silently reopen the shortcut, see EXP-003)
  5. split so no generator appears in both train and the held-out slice

Usage
-----
    python scripts/download_finetune_ready.py --budget-gb 100

    # Preview the plan (which shards, how many images/generators) without
    # downloading anything
    python scripts/download_finetune_ready.py --budget-gb 100 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.local_dataset import materialize  # noqa: E402
from src.data.manifest import build_manifest  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.data.sampling import (  # noqa: E402
    SHARD_SIZE_GB,
    add_size_columns,
    balance_classes,
    exclude_contaminated_sources,
    generator_disjoint_split,
    load_manifest,
    min_side_matched_pool,
    plan_shards,
    restrict_to_shards,
    summarize,
)

logger = logging.getLogger("download_finetune_ready")

DEFAULT_REPO = "OwensLab/CommunityForensics-Small"
DEFAULT_MANIFEST = Path("artifacts/manifests/community_forensics_small.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--budget-gb", type=float, default=100.0,
        help="Approximate total shard bytes to fetch (default 100 GB). Real "
             "disk usage after PNG re-encoding is close to but not exactly "
             "this -- treat it as a budget, not a guarantee.",
    )
    parser.add_argument("--min-side", type=int, default=512,
                        help="Scale-match bucket; must match the crop size "
                             "you intend to fine-tune with.")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Held out by GENERATOR (not by row) from the "
                             "same downloaded shards -- free, no extra fetch.")
    parser.add_argument("--output-name", default="cf_100gb",
                        help="Writes artifacts/images/<name>_train and "
                             "artifacts/images/<name>_val.")
    parser.add_argument("--images-dir", type=Path, default=Path("artifacts/images"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap images per class AFTER shard selection -- does NOT reduce "
             "which/how many shards get downloaded, since shard choice is "
             "what --budget-gb controls (each shard is one ~4.1GB unit; "
             "fetching any row from it fetches the whole shard). For a fast "
             "smoke test, shrink --budget-gb too (e.g. --budget-gb 10), not "
             "just --limit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the shard plan and split sizes; download nothing.",
    )
    args = parser.parse_args()

    configure_logging()

    # -- manifest: reuse if present, else scan (metadata only, seconds) -----
    if args.manifest.exists():
        logger.info("using existing manifest: %s", args.manifest)
    else:
        logger.info(
            "no manifest at %s -- scanning %s (metadata only, no image "
            "bytes, ~50MB total)...", args.manifest, args.repo,
        )
        stats = build_manifest(args.repo, args.manifest, progress=logger.info)
        logger.info("wrote %d rows from %d shards", stats.n_rows, stats.n_shards)

    frame = add_size_columns(load_manifest(args.manifest))
    frame = exclude_contaminated_sources(frame)

    # -- pick shards for the budget ------------------------------------------
    n_shards_per_class = max(1, int(args.budget_gb / SHARD_SIZE_GB / 2))
    plan = plan_shards(
        frame, n_shards_per_class=n_shards_per_class, min_side=args.min_side
    )

    print("\n" + "=" * 72)
    print("SHARD PLAN")
    print("=" * 72)
    print(f"  budget       : {args.budget_gb:.1f} GB requested")
    print(f"  shards       : {len(plan.shards)} "
          f"({n_shards_per_class} per class x 2)")
    print(f"  estimated    : {plan.estimated_gb:.1f} GB of shard bytes to fetch")
    print(f"  usable images: {plan.n_images:,} (balanced pairs at min_side="
          f"{args.min_side})")
    print(f"  generators   : {plan.n_generators:,}")

    if plan.n_images == 0:
        logger.error("plan produced zero images -- try a larger --budget-gb")
        return 2

    # -- disk space sanity check ---------------------------------------------
    free_bytes = shutil.disk_usage(args.images_dir.parent if args.images_dir.exists()
                                    else Path(".")).free
    needed_bytes = plan.estimated_gb * 1e9
    print(f"  free disk    : {free_bytes / 1e9:.1f} GB "
          f"(need roughly {plan.estimated_gb:.1f} GB)")
    if free_bytes < needed_bytes * 1.1:
        logger.warning(
            "  free space is close to the estimated download size -- "
            "consider a smaller --budget-gb"
        )

    # -- scale-match + balance + split ---------------------------------------
    pool = restrict_to_shards(frame, plan.shards)
    pool = min_side_matched_pool(pool, min_crop_size=args.min_side)
    balanced = balance_classes(
        pool, n_per_class=args.limit, seed=args.seed, stratify_column="min_side"
    )

    if balanced.empty:
        logger.error("balanced pool is empty after scale-matching")
        return 2

    train_df, val_df = generator_disjoint_split(
        balanced, holdout_fraction=args.val_fraction, seed=args.seed
    )

    print("\n" + "=" * 72)
    print("SPLITS (no generator appears in both)")
    print("=" * 72)
    print(f"  {summarize(train_df, 'train')}")
    print(f"  {summarize(val_df, 'val')}")

    if args.dry_run:
        print("\n[dry-run] downloading nothing. Drop --dry-run to fetch.")
        return 0

    # -- download + materialize, straight into finetune-ready manifests -----
    train_dir = args.images_dir / f"{args.output_name}_train"
    val_dir = args.images_dir / f"{args.output_name}_val"

    print("\n" + "=" * 72)
    print("DOWNLOADING (safe to Ctrl-C and re-run -- resumes from where it left off)")
    print("=" * 72)

    train_stats = materialize(args.repo, train_df, train_dir)
    val_stats = materialize(args.repo, val_df, val_dir)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"  train: {train_stats.n_written:,} images -> {train_stats.output_dir}")
    print(f"  val  : {val_stats.n_written:,} images -> {val_stats.output_dir}")
    if train_stats.n_failed or val_stats.n_failed:
        print(f"  ({train_stats.n_failed + val_stats.n_failed} images failed "
              f"to decode and were skipped)")

    print("\nReady to fine-tune -- no separate prep step needed:")
    print(f"  python scripts/finetune_lora.py --images-dir {args.images_dir} \\")
    print(f"      --train-split {args.output_name}_train --epochs 3 --lora-rank 8")
    print("\nAnd for phase 1 (fast sanity check) or the robustness matrix:")
    print(f"  python scripts/cache_features.py --local-manifest {train_dir} "
          f"--mode train --n-views 8")
    print(f"  python scripts/cache_features.py --local-manifest {val_dir} "
          f"--mode eval")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
