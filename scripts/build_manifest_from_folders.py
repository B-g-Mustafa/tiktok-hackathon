#!/usr/bin/env python3
"""Build a manifest.parquet for an already-extracted 0_real/1_fake image tree.

`down_image_data.py --dataset extract_sid` (and similar folder-based
extractors) encode the label in the directory name rather than writing a
manifest.parquet next to the images. `predict.py`/`dyno/files/infer.py` both
score a plain directory fine either way, but their automatic AUROC/AP step
needs a manifest.parquet with `path`/`label` columns to know the ground
truth -- this backfills exactly that, from folder names already on disk.

Usage:
    python scripts/build_manifest_from_folders.py --data-dir /path/to/sid_set/val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Folder-name prefix -> label. "0_real"/"1_fake" is what down_image_data.py
# writes; "real"/"fake" (no prefix) is a plausible alternative someone might
# use by hand, so both are accepted.
LABEL_BY_PREFIX = {"0_real": 0, "1_fake": 1, "real": 0, "fake": 1, "nature": 0, "ai": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing labeled subfolders "
                             "(e.g. 0_real/, 1_fake/), searched recursively.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: <data-dir>/manifest.parquet")
    args = parser.parse_args()

    output = args.output or (args.data_dir / "manifest.parquet")

    rows = []
    for path in sorted(args.data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        label = None
        for parent in path.relative_to(args.data_dir).parts[:-1]:
            if parent.lower() in LABEL_BY_PREFIX:
                label = LABEL_BY_PREFIX[parent.lower()]
                break
        if label is None:
            print(f"WARNING: no recognised label folder for {path} -- skipping")
            continue

        rows.append({"path": str(path.resolve()), "label": label})

    if not rows:
        print(f"ERROR: no labeled images found under {args.data_dir}")
        return 2

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(output, index=False)

    n_real = int((manifest["label"] == 0).sum())
    n_fake = int((manifest["label"] == 1).sum())
    print(f"{len(manifest):,} images ({n_real:,} real / {n_fake:,} fake) -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
