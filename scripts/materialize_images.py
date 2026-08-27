#!/usr/bin/env python3
"""Download and decode a split's images to local PNGs, once.

Run this before `scripts/finetune_lora.py`. It exists as a separate step
because fine-tuning needs many epochs over the same images, and re-streaming
from remote parquet shards every epoch would re-download tens of gigabytes per
pass. This script pays that cost exactly once; training after it is pure local
disk I/O.

Idempotent -- safe to re-run after an interruption; already-materialized
images are skipped.

Usage:
    python scripts/materialize_images.py --split train
    python scripts/materialize_images.py --split cross_generator
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.data.local_dataset import materialize  # noqa: E402

logger = logging.getLogger("materialize_images")

DEFAULT_REPO = "OwensLab/CommunityForensics-Small"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/images"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    split_path = args.splits_dir / f"{args.split}.parquet"
    if not split_path.exists():
        logger.error("split not found: %s (run scripts/build_splits.py)", split_path)
        return 2

    selection = pd.read_parquet(split_path)
    if args.limit:
        selection = selection.head(args.limit)

    out_dir = args.output_dir / args.split
    logger.info("materializing %d rows -> %s", len(selection), out_dir)

    stats = materialize(args.repo, selection, out_dir)
    logger.info(
        "done: %d written, %d failed -> %s",
        stats.n_written, stats.n_failed, stats.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
