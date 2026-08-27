#!/usr/bin/env python3
"""Scan your manually-downloaded CommunityForensics-Small parquet files into
a manifest (label, generator, resolution per image). No network, no image
decoding -- just reads metadata so the next step knows what's available.

Usage:
    python scripts/extract_dataset.py --data-dir /path/to/downloaded/data --output artifacts/manifest.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import build_manifest_from_local  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing your downloaded .parquet files.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifest.parquet"))
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"ERROR: not a directory: {args.data_dir}")
        return 2

    stats = build_manifest_from_local(args.data_dir, args.output, progress=print)

    print(f"\n{stats.n_rows:,} rows from {stats.n_shards} shards -> {args.output}")
    if stats.n_failed:
        print(f"WARNING: {stats.n_failed} shard(s) failed: {stats.failed_shards}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
