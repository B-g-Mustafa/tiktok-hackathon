#!/usr/bin/env python3
"""Audit a dataset for label-leaking artifacts before training on it.

This is the gate the whole project passes through first. It downloads only
METADATA from the Hugging Face datasets-server -- no image bytes -- so auditing
a 260GB dataset costs seconds and a few hundred kilobytes.

Usage:

    python scripts/run_canary.py
    python scripts/run_canary.py --dataset OwensLab/CommunityForensics-Small
    python scripts/run_canary.py --limit 1000 --json audit.json

Exit code is 1 when a shortcut is detected, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hf_metadata import DatasetsServerError, iter_rows  # noqa: E402
from src.evaluation.shortcut_controls import resolution_canary  # noqa: E402

DEFAULT_DATASET = "OwensLab/CommunityForensics-Small"

# Columns we audit. `resolution` is the one that matters most; `format` and
# `model_name` provide the context needed to explain *why* a shortcut exists.
AUDIT_COLUMNS = ["resolution", "format", "model_name", "label"]


def collect(dataset: str, label: int, limit: int, offset: int) -> list[dict]:
    """Fetch metadata rows for one class."""
    return list(
        iter_rows(
            dataset,
            AUDIT_COLUMNS,
            where=f'"label"={label}',
            limit=limit,
            start_offset=offset,
        )
    )


def describe(rows: list[dict], name: str) -> dict:
    """Summarize the composition of one class."""
    resolutions = Counter(
        tuple(r["resolution"]) for r in rows if r.get("resolution")
    )
    formats = Counter(r.get("format") for r in rows)
    sources = Counter(r.get("model_name") for r in rows)

    print(f"\n{name} (n={len(rows)})")
    print(f"  resolutions : {dict(resolutions.most_common(5))}")
    print(f"  formats     : {dict(formats)}")
    print(f"  top sources : {[s for s, _ in sources.most_common(4)]}")

    return {
        "n": len(rows),
        "resolutions": {str(k): v for k, v in resolutions.most_common(10)},
        "formats": {str(k): v for k, v in formats.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--limit",
        type=int,
        default=400,
        help="Rows to sample per class (default: 400).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Row offset, to sample a different region of the dataset.",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Write the audit report here."
    )
    args = parser.parse_args()

    print("=" * 72)
    print(f"SHORTCUT AUDIT  --  {args.dataset}")
    print("=" * 72)
    print("Downloading metadata only (no image bytes).")

    try:
        real_rows = collect(args.dataset, 0, args.limit, args.offset)
        fake_rows = collect(args.dataset, 1, args.limit, args.offset)
    except DatasetsServerError as exc:
        print(f"\nERROR: could not read dataset metadata.\n  {exc}")
        return 2

    if not real_rows or not fake_rows:
        print("\nERROR: one of the classes returned no rows; cannot audit.")
        return 2

    report = {
        "dataset": args.dataset,
        "authentic": describe(real_rows, "AUTHENTIC (label=0)"),
        "generated": describe(fake_rows, "GENERATED  (label=1)"),
    }

    widths, heights, labels = [], [], []
    for rows, label in ((real_rows, 0), (fake_rows, 1)):
        for row in rows:
            resolution = row.get("resolution")
            if not resolution or len(resolution) < 2:
                continue
            widths.append(resolution[0])
            heights.append(resolution[1])
            labels.append(label)

    result = resolution_canary(widths, heights, labels)

    print("\n" + "=" * 72)
    print("CANARY: resolution-only classifier")
    print("=" * 72)
    print(result.report())

    report["canary"] = {
        "name": result.name,
        "auroc": result.auroc,
        "is_alarming": result.is_alarming,
    }

    print()
    if result.is_alarming:
        print(
            "VERDICT: image dimensions alone carry the label.\n"
            "         Training on whole images would learn resolution, not\n"
            "         forensics, and collapse on the hidden test set.\n"
            "         Mitigation: feed fixed-size crops so the model cannot\n"
            "         observe image dimensions at all."
        )
    else:
        print("VERDICT: no strong resolution shortcut in this sample.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.json}")

    return 1 if result.is_alarming else 0


if __name__ == "__main__":
    raise SystemExit(main())
