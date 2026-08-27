#!/usr/bin/env python3
"""Turn the manifest from extract_dataset.py into fine-tune-ready data:
balanced real/fake images, matched in scale so the model can't just learn
image size, split so no generator leaks between train and val.

Usage:
    python scripts/prepare_finetune_data.py --data-dir /path/to/downloaded/data --manifest artifacts/manifest.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
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
from src.logging_utils import configure_logging  # noqa: E402

MIN_SIDE = 512  # matched scale for both classes; must match the crop size used to fine-tune

logger = logging.getLogger(__name__)


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
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel decode/encode threads. Set to 1 for "
                             "the old fully-sequential behaviour.")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Also write progress to this file (useful under "
                             "sbatch, where stdout is buffered and doesn't "
                             "update live).")
    args = parser.parse_args()

    configure_logging(log_file=args.log_file)

    if not args.manifest.exists():
        logger.error("no manifest at %s -- run extract_dataset.py first", args.manifest)
        return 2

    frame = add_size_columns(load_manifest(args.manifest))
    frame = exclude_contaminated_sources(frame)
    pool = min_side_matched_pool(frame, min_crop_size=MIN_SIDE)
    balanced = balance_classes(
        pool, n_per_class=args.max_per_class, seed=args.seed, stratify_column="min_side"
    )

    if balanced.empty:
        logger.error("nothing left after balancing -- check --manifest and --data-dir")
        return 2

    train_df, val_df = generator_disjoint_split(
        balanced, holdout_fraction=args.val_fraction, seed=args.seed
    )
    logger.info(summarize(train_df, "train"))
    logger.info(summarize(val_df, "val"))

    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"

    train_stats = materialize(
        "local", train_df, train_dir, local_dir=args.data_dir, workers=args.workers
    )
    val_stats = materialize(
        "local", val_df, val_dir, local_dir=args.data_dir, workers=args.workers
    )

    logger.info("train: %s images -> %s", f"{train_stats.n_written:,}", train_dir)
    logger.info("val  : %s images -> %s", f"{val_stats.n_written:,}", val_dir)

    logger.info(
        "Next:\n  python scripts/finetune_lora.py --images-dir %s "
        "--train-split train --epochs 3 --lora-rank 8",
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    exit_code = main()

    # Force the process to actually terminate here, rather than the normal
    # `raise SystemExit`. A genuinely stuck decode/save thread (a corrupted
    # record that never returns -- Python cannot forcibly kill a thread) is
    # non-daemon by default, so a clean interpreter shutdown would wait to
    # join it even after every real piece of work above is done and printed.
    # By this point all output has been produced and every file has been
    # written, so there is nothing left to lose by exiting immediately.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
