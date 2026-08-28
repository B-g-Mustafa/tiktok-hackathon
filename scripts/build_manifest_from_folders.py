#!/usr/bin/env python3
"""Build a manifest.parquet for an already-extracted real/fake image tree.

`down_image_data.py` (and similar folder-based extractors) encode the label in
the directory name rather than writing a manifest.parquet. This backfills one
from the folder structure.

It writes the same 5-column schema `materialize()` produces -- `key`, `path`,
`label`, `model_name`, `min_side` -- plus `format` and `dataset`, so the output
is usable by every downstream consumer rather than only by the metrics step:

  * `predict.py` / `dyno/files/infer.py` metrics need `path` + `label`
  * `iter_local_images` (and so `cache_features.py --local-manifest`) needs `key`
  * per-generator evaluation needs `model_name`
  * the scale and format shortcut canaries need `min_side` and `format`
  * mixing several datasets needs `dataset` to stratify and hold out by source

`model_name` is recovered from the path: whatever directory components remain
after stripping split names (`train`/`val`/...) and label names (`ai`/`nature`/
...). For GenImage's `genimage/<generator>/ai/x.jpg` that recovers the
generator; for a flat tree it falls back to the root's name.

Reading `min_side`/`format` means opening every file, which dominates runtime
on a large tree. `--fast` skips it and writes only what the path alone gives.

Usage:
    python scripts/build_manifest_from_folders.py --data-dir /path/to/sid_set/val
    python scripts/build_manifest_from_folders.py --data-dir .../genimage --dataset genimage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Folder name -> label. Covers every vocabulary down_image_data.py emits:
# 0_real/1_fake (sid, mock), nature/ai (genimage, community_*), real/fake.
LABEL_BY_DIRNAME = {
    "0_real": 0, "real": 0, "nature": 0, "reals": 0, "authentic": 0,
    "1_fake": 1, "fake": 1, "ai": 1, "fakes": 1, "generated": 1, "synthetic": 1,
}

# Structural directories that carry no generator identity, so they must not
# leak into the recovered `model_name`.
SKIP_DIRNAMES = frozenset(
    {"train", "val", "test", "validation", "extracted", "images"}
) | set(LABEL_BY_DIRNAME)


def label_from_path(path: Path, root: Path) -> int | None:
    """Deepest label directory wins, so `<gen>/ai/train/...` still resolves."""
    label = None
    for part in path.relative_to(root).parts[:-1]:
        if part.lower() in LABEL_BY_DIRNAME:
            label = LABEL_BY_DIRNAME[part.lower()]
    return label


def generator_from_path(path: Path, root: Path) -> str:
    """Whatever path components survive stripping split and label directories."""
    parts = [
        p for p in path.relative_to(root).parts[:-1]
        if p.lower() not in SKIP_DIRNAMES
    ]
    return "/".join(parts) if parts else root.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing labeled subfolders "
                             "(0_real/1_fake, nature/ai, ...), searched recursively.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: <data-dir>/manifest.parquet")
    parser.add_argument("--dataset", default=None,
                        help="Provenance tag for this tree (e.g. 'genimage', "
                             "'sid'). Needed to hold out or stratify by source "
                             "when several datasets are mixed. "
                             "Default: the --data-dir folder name.")
    parser.add_argument("--fast", action="store_true",
                        help="Skip opening images; omits min_side/format (and "
                             "with them the scale and format canaries).")
    args = parser.parse_args()

    output = args.output or (args.data_dir / "manifest.parquet")
    dataset = args.dataset or args.data_dir.name

    rows: list[dict] = []
    n_unlabeled = 0
    n_unreadable = 0

    for path in sorted(args.data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.name.startswith("._"):  # macOS resource forks
            continue

        label = label_from_path(path, args.data_dir)
        if label is None:
            n_unlabeled += 1
            continue

        row = {
            # Relative path is stable under a moved/renamed root, and matches
            # the `key` convention iter_local_images expects.
            "key": str(path.relative_to(args.data_dir)),
            "path": str(path.resolve()),
            "label": label,
            "model_name": generator_from_path(path, args.data_dir),
            "dataset": dataset,
        }

        if not args.fast:
            try:
                with Image.open(path) as handle:
                    row["min_side"] = int(min(handle.size))
                    row["format"] = str(handle.format or "UNKNOWN")
            except Exception as exc:  # noqa: BLE001
                n_unreadable += 1
                print(f"WARNING: unreadable, skipping {path}: {exc}")
                continue

        rows.append(row)

    if not rows:
        print(f"ERROR: no labeled images found under {args.data_dir}")
        return 2

    manifest = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output, index=False)

    n_real = int((manifest["label"] == 0).sum())
    n_fake = int((manifest["label"] == 1).sum())
    print(f"{len(manifest):,} images ({n_real:,} real / {n_fake:,} fake) -> {output}")
    print(f"  dataset    : {dataset}")
    print(f"  generators : {manifest['model_name'].nunique()} "
          f"({', '.join(sorted(manifest['model_name'].unique())[:6])}"
          f"{', ...' if manifest['model_name'].nunique() > 6 else ''})")
    if "format" in manifest.columns:
        counts = manifest.groupby("label")["format"].value_counts().to_dict()
        print(f"  formats    : {counts}")
    if n_unlabeled:
        print(f"  {n_unlabeled:,} files skipped (no label folder in their path)")
    if n_unreadable:
        print(f"  {n_unreadable:,} files skipped (unreadable)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
