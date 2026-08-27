#!/usr/bin/env python3
"""Build the dataset metadata manifest.

Reads only metadata columns from every parquet shard over HTTP range requests --
never the image bytes. Scanning all 186 shards of Community Forensics-Small
costs tens of megabytes instead of 260 GB.

The manifest is the foundation for every later decision: class balancing,
generator-disjoint splits, resolution matching, and contamination filtering all
become local dataframe operations once it exists.

Usage:

    python scripts/build_manifest.py
    python scripts/build_manifest.py --max-shards 4      # quick smoke test
    python scripts/build_manifest.py --workers 16
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from src.data.manifest import ShardScanError, build_manifest  # noqa: E402

DEFAULT_REPO = "OwensLab/CommunityForensics-Small"
DEFAULT_OUTPUT = Path("artifacts/manifests/community_forensics_small.parquet")


def summarize(path: Path) -> None:
    """Print the composition the manifest reveals.

    This is the output that determines the sampling policy, so it is printed
    rather than buried in the file.
    """
    table = pq.read_table(path)
    columns = set(table.column_names)

    print("\n" + "=" * 72)
    print("MANIFEST SUMMARY")
    print("=" * 72)
    print(f"rows   : {table.num_rows:,}")
    print(f"columns: {sorted(columns)}")

    def counts(column: str) -> Counter:
        if column not in columns:
            return Counter()
        return Counter(table.column(column).to_pylist())

    labels = counts("label")
    if labels:
        total = sum(labels.values())
        print("\nclass balance")
        for value, count in sorted(labels.items(), key=lambda kv: str(kv[0])):
            name = {0: "authentic", 1: "generated"}.get(value, str(value))
            print(f"  {name:<10} {count:>8,}  ({count / total:5.1%})")

    for column, title, top in (
        ("architecture", "architectures", 10),
        ("real_source", "real sources", 10),
        ("format", "formats", 10),
    ):
        values = counts(column)
        if values:
            print(f"\n{title}")
            for value, count in values.most_common(top):
                print(f"  {str(value)[:52]:<54} {count:>8,}")

    models = counts("model_name")
    if models:
        print(f"\ngenerators / sources: {len(models):,} distinct")
        for value, count in models.most_common(5):
            print(f"  {str(value)[:52]:<54} {count:>8,}")

    # Resolution by class is the shortcut we are hunting; show it explicitly.
    if {"resolution", "label"} <= columns:
        resolutions = table.column("resolution").to_pylist()
        label_values = table.column("label").to_pylist()
        by_class: dict[int, Counter] = {}
        for resolution, label in zip(resolutions, label_values):
            if not resolution:
                continue
            by_class.setdefault(label, Counter())[tuple(resolution)] += 1

        print("\nresolution by class  (the shortcut EXP-000 found)")
        for label in sorted(by_class):
            name = {0: "authentic", 1: "generated"}.get(label, str(label))
            top = by_class[label].most_common(5)
            print(f"  {name:<10} {dict(top)}")

        overlap = set(by_class.get(0, {})) & set(by_class.get(1, {}))
        print(f"\n  shared resolutions between classes: {len(overlap)}")
        if overlap:
            print(f"  usable for resolution-matched sampling: {sorted(overlap)[:8]}")
        else:
            print(
                "  WARNING: zero overlap -- classes are perfectly separable by size.\n"
                "  Resolution matching must resample one class onto the other."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Scan only the first N shards (smoke test).",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print("=" * 72)
    print(f"MANIFEST SCAN  --  {args.repo}")
    print("=" * 72)
    print("Reading metadata columns only; image bytes are never fetched.")

    try:
        stats = build_manifest(
            repo_id=args.repo,
            output_path=args.output,
            revision=args.revision,
            max_shards=args.max_shards,
            workers=args.workers,
            progress=print,
        )
    except ShardScanError as exc:
        print(f"\nERROR: {exc}")
        return 2

    print(f"\nwrote {stats.n_rows:,} rows from {stats.n_shards} shards -> {args.output}")
    if stats.n_failed:
        print(f"WARNING: {stats.n_failed} shard(s) failed: {stats.failed_shards}")

    summarize(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
