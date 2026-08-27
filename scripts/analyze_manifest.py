#!/usr/bin/env python3
"""Audit the manifest and quantify every shortcut before designing the split.

The manifest covers the whole dataset, so unlike the datasets-server sample in
EXP-000 the numbers here are exact rather than indicative. Three questions
decide the entire sampling policy:

  1. Resolution -- how strong is the size shortcut across all 556K rows, and at
     which resolutions do both classes actually coexist?
  2. Format     -- do the classes differ in container (PNG vs JPEG)? The first
     shards were 100% PNG, but the full dataset contains both.
  3. Source     -- which real sources and generator architectures are available
     to hold out, and which must be excluded for contamination.

Usage:
    python scripts/analyze_manifest.py
    python scripts/analyze_manifest.py --json artifacts/reports/manifest_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from src.evaluation.shortcut_controls import resolution_canary  # noqa: E402

DEFAULT_MANIFEST = Path("artifacts/manifests/community_forensics_small.parquet")

LABEL_NAMES = {0: "authentic", 1: "generated"}


def cross_tab(rows: dict[str, list], key: str, label_key: str = "label") -> dict:
    """Count `key` values split by class label."""
    table: dict = defaultdict(Counter)
    for value, label in zip(rows[key], rows[label_key]):
        if isinstance(value, list):
            value = tuple(value)
        table[label][value] += 1
    return table


def print_cross_tab(table: dict, title: str, top: int = 12) -> None:
    print(f"\n{title}")
    all_values: Counter = Counter()
    for counter in table.values():
        all_values.update(counter)

    width = 44
    header = f"  {'value':<{width}}" + "".join(
        f"{LABEL_NAMES.get(lbl, lbl):>12}" for lbl in sorted(table)
    )
    print(header)
    print("  " + "-" * (width + 12 * len(table)))

    for value, _ in all_values.most_common(top):
        line = f"  {str(value)[:width - 2]:<{width}}"
        for label in sorted(table):
            line += f"{table[label].get(value, 0):>12,}"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}")
        print("Run: python scripts/build_manifest.py")
        return 2

    table = pq.read_table(args.manifest)
    rows = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows

    print("=" * 78)
    print(f"MANIFEST AUDIT  --  {n:,} rows")
    print("=" * 78)

    labels = Counter(rows["label"])
    for label, count in sorted(labels.items()):
        print(f"  {LABEL_NAMES.get(label, label):<12} {count:>9,}  ({count / n:5.1%})")

    report: dict = {"n_rows": n, "class_balance": {str(k): v for k, v in labels.items()}}

    # -- 1. resolution ------------------------------------------------------
    res_tab = cross_tab(rows, "resolution")
    print_cross_tab(res_tab, "RESOLUTION x CLASS", top=12)

    authentic_res = set(res_tab.get(0, {}))
    generated_res = set(res_tab.get(1, {}))
    shared = authentic_res & generated_res

    print(f"\n  distinct resolutions : authentic={len(authentic_res):,} "
          f"generated={len(generated_res):,} shared={len(shared):,}")

    # The canary, now over every row rather than a 600-row sample.
    widths = [r[0] for r in rows["resolution"] if r and len(r) >= 2]
    heights = [r[1] for r in rows["resolution"] if r and len(r) >= 2]
    canary_labels = [
        lbl for r, lbl in zip(rows["resolution"], rows["label"]) if r and len(r) >= 2
    ]
    canary = resolution_canary(widths, heights, canary_labels)
    print("\n  FULL-DATASET CANARY (resolution only, no image content)")
    print("  " + canary.report().replace("\n", "\n  "))
    report["canary_full"] = {"auroc": canary.auroc, "alarming": canary.is_alarming}

    # Resolution-matched pool: the only rows where size carries no signal.
    print("\n  RESOLUTION-MATCHED POOL (both classes present at the same size)")
    matched_total = 0
    matched: list[dict] = []
    for resolution in sorted(
        shared, key=lambda r: -(res_tab[0][r] + res_tab[1][r])
    )[:10]:
        n_auth = res_tab[0][resolution]
        n_gen = res_tab[1][resolution]
        usable = 2 * min(n_auth, n_gen)  # balanced pairs
        matched_total += usable
        matched.append(
            {"resolution": list(resolution), "authentic": n_auth,
             "generated": n_gen, "balanced": usable}
        )
        print(
            f"    {str(resolution):<16} authentic={n_auth:>8,}  "
            f"generated={n_gen:>8,}  -> balanced usable={usable:>8,}"
        )
    print(f"\n    total balanced, resolution-matched images: {matched_total:,}")
    report["resolution_matched"] = matched
    report["resolution_matched_total"] = matched_total

    # -- 2. format ----------------------------------------------------------
    fmt_tab = cross_tab(rows, "format")
    print_cross_tab(fmt_tab, "FORMAT x CLASS")

    def fmt_share(label: int, fmt: str) -> float:
        counter = fmt_tab.get(label, Counter())
        total = sum(counter.values())
        return counter.get(fmt, 0) / total if total else 0.0

    png_gap = abs(fmt_share(0, "PNG") - fmt_share(1, "PNG"))
    print(f"\n  PNG share: authentic={fmt_share(0, 'PNG'):.1%} "
          f"generated={fmt_share(1, 'PNG'):.1%}  (gap {png_gap:.1%})")
    if png_gap > 0.15:
        print("  WARNING: container format differs by class -- a format shortcut "
              "exists.\n           Force a common re-encode policy for both classes.")
    report["format_gap_png"] = png_gap

    # -- 3. sources / architectures ----------------------------------------
    print_cross_tab(cross_tab(rows, "architecture"), "ARCHITECTURE x CLASS", top=8)

    real_sources = Counter(
        name for name, label in zip(rows["model_name"], rows["label"]) if label == 0
    )
    print("\nREAL SOURCES (label=0)")
    for name, count in real_sources.most_common(10):
        flag = ""
        if name and "coco" in str(name).lower():
            flag = "  <- EXCLUDE: organizer benchmark uses COCO val2017"
        if name and "raise" in str(name).lower():
            flag = "  <- EXCLUDE: authors forbid training on RAISE"
        print(f"  {str(name)[:44]:<46}{count:>10,}{flag}")
    report["real_sources"] = dict(real_sources.most_common(20))

    generators = Counter(
        name for name, label in zip(rows["model_name"], rows["label"]) if label == 1
    )
    print(f"\nGENERATORS (label=1): {len(generators):,} distinct")
    print(f"  median images per generator: "
          f"{sorted(generators.values())[len(generators) // 2]}")
    report["n_generators"] = len(generators)

    # -- verdict ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SAMPLING POLICY IMPLIED BY THIS AUDIT")
    print("=" * 78)
    print(
        f"  1. Resolution-match: draw only from the {matched_total:,} images where\n"
        f"     both classes exist at the same size. Kills the canary by construction.\n"
        f"  2. Exclude every COCO-sourced real (organizer benchmark is COCO val2017).\n"
        f"  3. Hold out whole generators AND whole architecture families for the\n"
        f"     cross-generator split -- {len(generators):,} generators available.\n"
        f"  4. Re-encode both classes identically (PNG gap is {png_gap:.1%})."
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport -> {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
