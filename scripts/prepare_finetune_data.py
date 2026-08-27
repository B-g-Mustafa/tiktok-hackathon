#!/usr/bin/env python3
"""Turn the manifest from extract_dataset.py into fine-tune-ready data:
balanced real/fake images, matched in scale so the model can't just learn
image size, split so no generator leaks between train and val.

Usage:
    python scripts/prepare_finetune_data.py --data-dir /path/to/downloaded/data --manifest artifacts/manifest.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.local_dataset import materialize  # noqa: E402
from src.data.sampling import (  # noqa: E402
    add_size_columns,
    balance_classes,
    exclude_contaminated_sources,
    generator_disjoint_split,
    load_manifest,
    min_side_matched_pool,
    summarize,
)

MIN_SIDE = 512  # matched scale for both classes; must match the crop size used to fine-tune


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Same directory used in extract_dataset.py.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/images"))
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap images per class. Default: use everything available.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: no manifest at {args.manifest} -- run extract_dataset.py first")
        return 2

    frame = add_size_columns(load_manifest(args.manifest))
    frame = exclude_contaminated_sources(frame)
    pool = min_side_matched_pool(frame, min_crop_size=MIN_SIDE)
    balanced = balance_classes(
        pool, n_per_class=args.max_per_class, seed=args.seed, stratify_column="min_side"
    )

    if balanced.empty:
        print("ERROR: nothing left after balancing -- check --manifest and --data-dir")
        return 2

    train_df, val_df = generator_disjoint_split(
        balanced, holdout_fraction=args.val_fraction, seed=args.seed
    )
    print(summarize(train_df, "train"))
    print(summarize(val_df, "val"))

    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"

    train_stats = materialize("local", train_df, train_dir, local_dir=args.data_dir)
    val_stats = materialize("local", val_df, val_dir, local_dir=args.data_dir)

    print(f"\ntrain: {train_stats.n_written:,} images -> {train_dir}")
    print(f"val  : {val_stats.n_written:,} images -> {val_dir}")

    print(
        f"\nNext:\n"
        f"  python scripts/finetune_lora.py --images-dir {args.output_dir} "
        f"--train-split train --epochs 3 --lora-rank 8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
