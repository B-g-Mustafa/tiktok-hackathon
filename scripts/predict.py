#!/usr/bin/env python3
"""Score every image in a directory for being AI-generated.

This is the required deliverable. Contract:

    python scripts/predict.py --image-dir DIR --out preds.json

produces a JSON array:

    [
      {"image_path": "/abs/path/img1.jpg", "pred": 0.9371},
      {"image_path": "/abs/path/img2.png", "pred": 0.0832}
    ]

where `pred` is the probability that the image is AI-generated.

That score is calibrated only if the checkpoint carries a `calibration.json`
(written by `scripts/calibrate.py`); otherwise it is the model's raw output,
which under dataset shift is systematically biased and should not be read as a
true probability. `--model siglip2` logs which case applies at startup.

Design notes
------------
* Unreadable images never abort the run. They are reported on stderr and, with
  `--include-failures`, emitted with a null prediction so the output still has
  one row per input file.
* The model is injected, so this script is testable (and shippable) before the
  real detector exists. `--model constant` runs the placeholder baseline.
* Output order is deterministic (sorted by path), making runs diffable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score  # noqa: E402

from src.data.io import iter_image_paths, load_image  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.models.base import ConstantDetector, Detector  # noqa: E402
from src.transforms.robustness import eval_grid  # noqa: E402

logger = logging.getLogger("predict")

# Same named transforms used to build the offline robustness matrix
# (src/transforms/robustness.py is the single source of truth) -- lets
# --degrade simulate one of them on the actual inference path, rather than
# only ever scoring clean images here and trusting the offline matrix.
DEGRADATIONS = {t.name: t for t in eval_grid()}


def build_detector(spec: str, checkpoint: Path | None) -> Detector:
    """Resolve a --model value to a Detector.

    Kept deliberately small: the real detector is added here as one extra
    branch once it exists, with no other change to this script.
    """
    if spec == "constant":
        return ConstantDetector(score=0.5)

    if spec == "siglip2":
        # Imported lazily so the torch stack is only required when actually
        # running a neural model -- the contract tests stay dependency-free.
        from src.models.siglip_detector import load_siglip_detector

        if checkpoint is None:
            raise SystemExit("--checkpoint is required for --model siglip2")
        return load_siglip_detector(checkpoint)

    raise SystemExit(f"unknown model: {spec!r}")


def run(
    image_dir: Path,
    detector: Detector,
    batch_size: int,
    recursive: bool,
    include_failures: bool,
    degrade: str = "clean",
) -> tuple[list[dict], int]:
    """Score every image. Returns (records, n_failed)."""
    paths = list(iter_image_paths(image_dir, recursive=recursive))
    if not paths:
        logger.warning("no supported image files found under %s", image_dir)

    records: list[dict] = []
    n_failed = 0

    transform = DEGRADATIONS[degrade]

    batch_images = []
    batch_paths = []

    def flush() -> None:
        if not batch_images:
            return
        scores = detector.predict_batch(batch_images)
        if len(scores) != len(batch_images):
            raise RuntimeError(
                f"detector returned {len(scores)} scores for "
                f"{len(batch_images)} images"
            )
        for path, score in zip(batch_paths, scores):
            records.append(
                {"image_path": str(path.resolve()), "pred": round(float(score), 6)}
            )
        batch_images.clear()
        batch_paths.clear()

    for path in paths:
        result = load_image(path)

        if not result.ok:
            n_failed += 1
            logger.warning("skipping %s (%s)", path, result.error)
            if include_failures:
                records.append(
                    {
                        "image_path": str(path.resolve()),
                        "pred": None,
                        "error": result.error,
                    }
                )
            continue

        batch_images.append(transform(result.image))
        batch_paths.append(path)

        if len(batch_images) >= batch_size:
            flush()

    flush()

    # One image may have been appended before a later failure, so sort at the
    # end to guarantee deterministic ordering regardless of batching.
    records.sort(key=lambda r: r["image_path"])
    return records, n_failed


def write_metrics_report(records: list[dict], image_dir: Path, out_path: Path,
                          manifest_path: Path | None) -> None:
    """AUROC/AP/accuracy against a labeled manifest.parquet, if one is
    available -- same contract as dyno/files/infer.py's metrics step, so the
    two models' reports are directly comparable. Silently does nothing if no
    manifest can be found (predict.py's core contract is label-free scoring;
    this is a bonus when ground truth happens to be sitting right there,
    e.g. from prepare_finetune_data.py's or
    scripts/build_manifest_from_folders.py's output)."""
    manifest_path = manifest_path or (image_dir / "manifest.parquet")
    if not manifest_path.exists():
        return

    manifest = pd.read_parquet(manifest_path)
    label_by_path = {
        str(Path(row.path).resolve()): int(row.label) for row in manifest.itertuples()
    }

    y_true, y_score = [], []
    for record in records:
        label = label_by_path.get(record["image_path"])
        if label is not None and record.get("pred") is not None:
            y_true.append(label)
            y_score.append(record["pred"])

    if len(y_true) < len(records):
        logger.warning(
            "only %d/%d predictions matched a row in %s -- metrics below are "
            "computed on the matched subset only",
            len(y_true), len(records), manifest_path,
        )

    if len(set(y_true)) < 2:
        logger.warning(
            "not enough labeled/matched images with both classes -- skipping metrics"
        )
        return

    detailed = compute_metrics(np.asarray(y_true), np.asarray(y_score))
    metrics = {
        "n_matched": len(y_true),
        "auroc": round(float(roc_auc_score(y_true, y_score)), 6),
        "ap": round(float(average_precision_score(y_true, y_score)), 6),
        "accuracy": round(
            float(accuracy_score(y_true, [s >= 0.5 for s in y_score])), 6
        ),
        "balanced_accuracy": round(detailed.balanced_accuracy, 6),
        "best_balanced_accuracy": round(detailed.best_balanced_accuracy, 6),
        "best_threshold": round(detailed.best_threshold, 6),
        "eer": round(detailed.eer, 6),
        "ece": round(detailed.ece, 6),
    }
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(
        "metrics (%d images): AUROC=%.4f  AP=%.4f  acc=%.4f  bal_acc=%.4f  "
        "EER=%.4f  ECE=%.4f  -> %s",
        metrics["n_matched"], metrics["auroc"], metrics["ap"],
        metrics["accuracy"], metrics["balanced_accuracy"],
        metrics["eer"], metrics["ece"], metrics_path,
    )

    # The single most actionable diagnostic: a large gap here means the scores
    # rank fine and the THRESHOLD is wrong, which scripts/calibrate.py fixes
    # without retraining. Without this line the low accuracy reads as a weak
    # model and invites an expensive, unnecessary retrain.
    if detailed.threshold_gap > 0.05:
        logger.warning(
            "balanced accuracy would be %.4f at threshold %.4f instead of "
            "%.4f at 0.5 (+%.4f) -- this looks like a calibration failure, "
            "not a weak model. Fit a correction with:\n"
            "  python scripts/calibrate.py --predictions %s --manifest %s "
            "--checkpoint <ckpt>",
            detailed.best_balanced_accuracy, detailed.best_threshold,
            detailed.balanced_accuracy, detailed.threshold_gap,
            out_path, manifest_path,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir", type=Path, required=True, help="Directory of images to score."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("preds.json"), help="Output JSON path."
    )
    parser.add_argument(
        "--model",
        default="constant",
        help="Detector to use: 'constant' (baseline) or 'siglip2'.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the top level of --image-dir.",
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Emit unreadable images with pred=null instead of omitting them.",
    )
    parser.add_argument(
        "--degrade", default="clean", choices=sorted(DEGRADATIONS),
        help="Apply one robustness transform (from the eval grid) to every "
             "image before scoring, to test the real inference path against "
             "degraded input instead of only the offline robustness matrix. "
             "Default: 'clean' (no-op).",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="manifest.parquet with a 'path'/'label' column to compute "
             "AUROC/AP/accuracy against, in addition to raw predictions. "
             "Default: auto-detect 'manifest.parquet' inside --image-dir; "
             "pass a nonexistent path to skip metrics entirely.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    configure_logging(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.image_dir.is_dir():
        print(f"ERROR: not a directory: {args.image_dir}", file=sys.stderr)
        return 2

    detector = build_detector(args.model, args.checkpoint)
    logger.info("detector: %s  degrade: %s", detector.name, args.degrade)

    scaler = getattr(detector, "scaler", None)
    if scaler is not None:
        logger.info(
            "scores are %s",
            "UNCALIBRATED raw model output"
            if scaler.is_identity
            else f"calibrated ({scaler})",
        )

    records, n_failed = run(
        image_dir=args.image_dir,
        detector=detector,
        batch_size=args.batch_size,
        recursive=not args.no_recursive,
        include_failures=args.include_failures,
        degrade=args.degrade,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))

    scored = sum(1 for r in records if r.get("pred") is not None)
    logger.info("scored %d image(s); %d unreadable -> %s", scored, n_failed, args.out)

    write_metrics_report(records, args.image_dir, args.out, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
