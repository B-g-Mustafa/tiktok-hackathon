#!/usr/bin/env python3
"""Scan your manually-downloaded CommunityForensics-Small parquet files into
a manifest (label, generator, resolution per image). No network, no image
decoding -- just reads metadata so the next step knows what's available.

Usage:
    python scripts/extract_dataset.py --data-dir /path/to/downloaded/data --output artifacts/manifest.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import build_manifest_from_local  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing your downloaded .parquet files.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifest.parquet"))
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Also write progress to this file (useful under "
                             "sbatch, where stdout is buffered and doesn't "
                             "update live).")
    args = parser.parse_args()

    configure_logging(log_file=args.log_file)

    if not args.data_dir.is_dir():
        logger.error("not a directory: %s", args.data_dir)
        return 2

    stats = build_manifest_from_local(args.data_dir, args.output, progress=logger.info)

    logger.info("%s rows from %s shards -> %s", f"{stats.n_rows:,}", stats.n_shards, args.output)
    if stats.n_failed:
        logger.warning("%s shard(s) failed: %s", stats.n_failed, stats.failed_shards)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
