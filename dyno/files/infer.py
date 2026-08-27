#!/usr/bin/env python3
"""Run a dyno (DINO-backbone) detector checkpoint over a directory of images.

training/utils.py's save_checkpoint embeds the exact model config used for
that run (backbone, LoRA, dual-stream, mlp dims -- see training/train.py's
`state = {..., "config": config}`), so this rebuilds the architecture from
ckpt["config"] via the same build_detector() the training script uses,
instead of requiring a separate --config YAML that could silently mismatch
(e.g. loading a LoRA/dual-stream checkpoint into a plain backbone -- which
`load_checkpoint`'s strict=False would not error on, just under-load).

Note: evaluation/evaluate.py's fuller suite (bias audit, WildFake/GenImage
OOD) needs a `datasets.sid` module that wasn't included in the files you
shared, so this script only covers plain directory -> predictions inference,
same contract as the main repo's scripts/predict.py.

Preprocessing (resize + ImageNet normalization) is standard for
DINOv2/DINOv3 checkpoints, but the original `datasets/sid.py` that actually
defined the training-time transform wasn't shared either -- if predictions
look off, that mismatch is the first thing to check.

Usage:
    python dyno/files/infer.py --checkpoint /path/to/best_model.pt --image-dir DIR --out preds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The main repo (two levels up: dyno/files -> dyno -> tiktok-hackathon), so
# the robustness transforms can be reused as-is rather than reimplemented --
# src/transforms/robustness.py is the single source of truth for them.
MAIN_REPO_ROOT = PROJECT_ROOT.parent.parent
if str(MAIN_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score  # noqa: E402

from models import build_detector  # noqa: E402
from training.utils import load_checkpoint  # noqa: E402
from src.transforms.robustness import eval_grid  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEGRADATIONS = {t.name: t for t in eval_grid()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("preds_dyno.json"))
    parser.add_argument("--image-size", type=int, default=None,
                         help="Override input resolution. Default: from the checkpoint's config.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--degrade", default="clean", choices=sorted(DEGRADATIONS),
        help="Apply one robustness transform (from the eval grid) to every "
             "image before scoring, to test the real inference path against "
             "degraded input. Default: 'clean' (no-op).",
    )
    parser.add_argument("--limit", type=int, default=None,
                         help="Score at most this many images (smoke test).")
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="manifest.parquet with a 'path'/'label' column (the output of "
             "prepare_finetune_data.py) to compute AUROC/AP/accuracy against, "
             "in addition to raw predictions. Default: auto-detect "
             "'manifest.parquet' inside --image-dir; pass an explicit path, "
             "or a nonexistent one, to skip metrics entirely.",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config")
    if config is None:
        raise SystemExit(
            f"{args.checkpoint} has no embedded 'config' -- pass a checkpoint "
            f"saved by training/utils.py:save_checkpoint (best_model.pt or "
            f"checkpoint_latest.pt from a training run)."
        )
    print(f"loaded config from checkpoint (epoch {ckpt.get('epoch')}, "
          f"best_auc {ckpt.get('best_auc')})")

    model = build_detector(config)  # handles LoRA / dual-stream internally, from config
    load_checkpoint(args.checkpoint, model)
    model = model.to(device).eval()

    image_size = args.image_size or config.get("data", {}).get("image_size", 384)
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    paths = sorted(
        p for p in args.image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no images found under {args.image_dir}", file=sys.stderr)

    records: list[dict] = []
    batch: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    def flush() -> None:
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().tolist()
        for path, prob in zip(batch_paths, probs):
            records.append({"image_path": str(path.resolve()), "pred": round(float(prob), 6)})
        batch.clear()
        batch_paths.clear()

    degrade = DEGRADATIONS[args.degrade]

    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        batch.append(transform(degrade(image)))
        batch_paths.append(path)
        if len(batch) >= args.batch_size:
            flush()
    flush()

    records.sort(key=lambda r: r["image_path"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))
    print(f"scored {len(records)} image(s) -> {args.out}")

    # -- metrics, only if a labeled manifest is available --------------------
    manifest_path = args.manifest or (args.image_dir / "manifest.parquet")
    if not manifest_path.exists():
        return 0

    manifest = pd.read_parquet(manifest_path)
    label_by_path = {
        str(Path(row.path).resolve()): int(row.label) for row in manifest.itertuples()
    }
    y_true, y_score = [], []
    for record in records:
        label = label_by_path.get(record["image_path"])
        if label is not None:
            y_true.append(label)
            y_score.append(record["pred"])

    if len(y_true) < len(records):
        print(
            f"WARNING: only {len(y_true)}/{len(records)} predictions matched "
            f"a row in {manifest_path} -- metrics below are computed on the "
            f"matched subset only",
            file=sys.stderr,
        )

    if len(set(y_true)) < 2:
        print("not enough labeled/matched images with both classes -- skipping metrics",
              file=sys.stderr)
        return 0

    metrics = {
        "n_matched": len(y_true),
        "auroc": round(float(roc_auc_score(y_true, y_score)), 6),
        "ap": round(float(average_precision_score(y_true, y_score)), 6),
        "accuracy": round(
            float(accuracy_score(y_true, [s >= 0.5 for s in y_score])), 6
        ),
    }
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"metrics ({metrics['n_matched']} images): "
          f"AUROC={metrics['auroc']:.4f}  AP={metrics['ap']:.4f}  "
          f"acc={metrics['accuracy']:.4f}  -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
