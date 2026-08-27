#!/usr/bin/env python3
"""Turn a local real/fake image directory (e.g. GenImage) into the manifest
format the training scripts consume.

Works with GenImage's native layout (`<generator>/train/ai|nature/...`) or any
directory tree where real/fake images live under folders named one of:

    real:  real, nature, authentic, reals
    fake:  fake, ai, generated, synthetic, fakes

Usage:
    python scripts/prepare_local_dataset.py \
        --root /data/genimage --split train --output-name genimage_train

    python scripts/prepare_local_dataset.py \
        --root /data/genimage --split val --output-name genimage_val
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.local_manifest import build_local_manifest  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger("prepare_local_dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Directory to scan (e.g. a GenImage 'train' or 'val' folder).",
    )
    parser.add_argument(
        "--output-name", required=True,
        help="Name for this dataset split; images/<name>/manifest.parquet is written.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/images"))
    parser.add_argument(
        "--limit-per-class", type=int, default=None,
        help="Cap images per class (GenImage's train split alone is >1M images "
             "across generators; this keeps a first run fast).",
    )
    parser.add_argument(
        "--split", default=None,
        help="Restrict to this path component, e.g. 'train' or 'val'. "
             "GenImage keeps both splits side by side under every generator "
             "(<generator>/train/... and <generator>/val/...), so pointing "
             "--root at the dataset root WITHOUT --split merges them.",
    )
    args = parser.parse_args()

    configure_logging(format="%(asctime)s %(levelname)s: %(message)s")

    output_dir = args.output_dir / args.output_name
    logger.info("scanning %s -> %s", args.root, output_dir)

    stats = build_local_manifest(
        args.root,
        output_dir,
        limit_per_class=args.limit_per_class,
        split_filter=args.split,
    )
    logger.info("done: %s", stats)

    if stats.n_authentic == 0 or stats.n_generated == 0:
        logger.error(
            "one class is empty -- this split cannot be used for training/eval"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
