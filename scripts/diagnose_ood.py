#!/usr/bin/env python3
"""Explain WHY a model underperforms on an out-of-domain dataset.

Run this before changing anything. An OOD AUROC of 0.84 can mean at least
three different things, each with a different fix, and the aggregate number
cannot distinguish them:

  1. The threshold is misplaced (ranking is fine)   -> scripts/calibrate.py, minutes, no GPU
  2. A few generators fail catastrophically         -> targeted data, not a full retrain
  3. The eval set has a shortcut                    -> the number itself is untrustworthy

This reads an existing predictions file plus its labeled manifest -- no model,
no images, no GPU -- and reports all three.

Usage:
    python scripts/diagnose_ood.py --predictions preds.json --manifest DIR/manifest.parquet
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

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.shortcut_controls import (  # noqa: E402
    format_canary,
    scale_canary,
)
from src.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger("diagnose")

# Below this many images a per-generator AUROC is too noisy to act on.
MIN_ROWS_PER_GENERATOR = 20


def load_joined(predictions: Path, manifest: Path) -> pd.DataFrame:
    """Join predictions onto the manifest, keyed by resolved path."""
    records = json.loads(predictions.read_text())
    scores = {
        str(Path(r["image_path"]).resolve()): float(r["pred"])
        for r in records
        if r.get("pred") is not None
    }

    frame = pd.read_parquet(manifest)
    frame["_resolved"] = frame["path"].map(lambda p: str(Path(p).resolve()))
    frame["score"] = frame["_resolved"].map(scores)

    matched = frame.dropna(subset=["score"]).copy()
    if matched.empty:
        raise SystemExit(
            f"no predictions in {predictions} matched a row in {manifest}"
        )
    if len(matched) < len(frame):
        logger.warning(
            "%d/%d manifest rows had no prediction", len(matched), len(frame)
        )
    return matched


def report_overall(frame: pd.DataFrame) -> None:
    m = compute_metrics(frame["label"].to_numpy(), frame["score"].to_numpy())

    logger.info("=" * 72)
    logger.info("OVERALL  (%d images: %d generated / %d authentic)",
                len(frame), m.n_positive, m.n_negative)
    logger.info("=" * 72)
    logger.info("  AUROC              %.4f   <- ranking quality; calibration cannot change this",
                m.auroc)
    logger.info("  AP                 %.4f", m.average_precision)
    logger.info("  EER                %.4f", m.eer)
    logger.info("")
    logger.info("  accuracy @0.5      %.4f", m.accuracy)
    logger.info("  balanced acc @0.5  %.4f", m.balanced_accuracy)
    logger.info("  best balanced acc  %.4f  (at threshold %.4f)",
                m.best_balanced_accuracy, m.best_threshold)
    logger.info("  ECE                %.4f", m.ece)
    logger.info("")

    if m.threshold_gap > 0.05:
        logger.info("  VERDICT: CALIBRATION FAILURE.")
        logger.info("    %.1f accuracy points are being lost to a misplaced threshold,",
                    100 * m.threshold_gap)
        logger.info("    not to a weak representation. Fix without retraining:")
        logger.info("      python scripts/calibrate.py --predictions ... --manifest ... --checkpoint ...")
    else:
        logger.info("  VERDICT: threshold is roughly right; the gap is genuine")
        logger.info("    ranking weakness. Calibration will not help much here.")
    logger.info("")


def report_score_shift(frame: pd.DataFrame) -> None:
    """Which direction is the model biased, and by how much?"""
    real = frame.loc[frame["label"] == 0, "score"]
    fake = frame.loc[frame["label"] == 1, "score"]

    logger.info("-" * 72)
    logger.info("SCORE DISTRIBUTION")
    logger.info("-" * 72)
    logger.info("  %-10s %8s %8s %8s %8s", "class", "mean", "median", "p10", "p90")
    for name, values in (("authentic", real), ("generated", fake)):
        logger.info("  %-10s %8.4f %8.4f %8.4f %8.4f", name, values.mean(),
                    values.median(), values.quantile(0.1), values.quantile(0.9))

    called_fake = float((frame["score"] >= 0.5).mean())
    actually_fake = float((frame["label"] == 1).mean())
    logger.info("")
    logger.info("  predicted generated: %.1f%%   actually generated: %.1f%%",
                100 * called_fake, 100 * actually_fake)
    if abs(called_fake - actually_fake) > 0.1:
        direction = "AUTHENTIC" if called_fake < actually_fake else "GENERATED"
        logger.info("  -> model is systematically biased toward calling images %s",
                    direction)
    logger.info("")


def report_per_generator(frame: pd.DataFrame) -> None:
    """Per-generator AUROC against the shared authentic pool.

    Holding the authentic pool constant across generators is what makes these
    numbers comparable -- each generator is scored against the same reals.
    """
    if "model_name" not in frame.columns:
        logger.info("(no `model_name` column -- rebuild the manifest with "
                    "scripts/build_manifest_from_folders.py for a per-generator "
                    "breakdown)")
        return

    authentic = frame[frame["label"] == 0]
    generated = frame[frame["label"] == 1]
    if authentic.empty or generated.empty:
        return

    logger.info("-" * 72)
    logger.info("PER-GENERATOR  (each vs the full authentic pool of %d images)",
                len(authentic))
    logger.info("-" * 72)

    results = []
    for name, group in generated.groupby("model_name"):
        if len(group) < MIN_ROWS_PER_GENERATOR:
            continue
        subset = pd.concat([authentic, group])
        m = compute_metrics(subset["label"].to_numpy(), subset["score"].to_numpy())
        results.append((name, len(group), m.auroc, m.best_balanced_accuracy))

    if not results:
        logger.info("  (no generator had >= %d images)", MIN_ROWS_PER_GENERATOR)
        return

    results.sort(key=lambda r: r[2])
    logger.info("  %-34s %7s %8s %10s", "generator", "n", "AUROC", "best_bal")
    for name, n, auroc, best in results:
        logger.info("  %-34s %7d %8.4f %10.4f", name[:34], n, auroc, best)

    aurocs = [r[2] for r in results]
    logger.info("")
    logger.info("  spread: worst %.4f / median %.4f / best %.4f",
                min(aurocs), float(np.median(aurocs)), max(aurocs))
    if max(aurocs) - min(aurocs) > 0.15:
        logger.info("  -> FAILURE IS CONCENTRATED. Adding data resembling the")
        logger.info("     worst generators will pay off far more than a uniform")
        logger.info("     data increase.")
    else:
        logger.info("  -> failure is UNIFORM across generators; this is a general")
        logger.info("     representation gap, not a few bad generators.")
    logger.info("")


def report_canaries(frame: pd.DataFrame) -> None:
    """Is the eval set itself measuring forensics, or an artifact?"""
    logger.info("-" * 72)
    logger.info("SHORTCUT CANARIES  (on the EVAL SET -- do these numbers mean anything?)")
    logger.info("-" * 72)

    labels = frame["label"].to_numpy()
    ran = False

    if "format" in frame.columns:
        logger.info(format_canary(frame["format"].to_numpy(), labels).report())
        logger.info("")
        ran = True
    if "min_side" in frame.columns:
        logger.info(scale_canary(frame["min_side"].to_numpy(), labels).report())
        logger.info("")
        ran = True

    if not ran:
        logger.info("  (manifest has no `format`/`min_side` columns -- rebuild "
                    "with scripts/build_manifest_from_folders.py without --fast)")
        logger.info("")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    configure_logging(format="%(message)s", log_file=args.log_file)

    frame = load_joined(args.predictions, args.manifest)

    report_overall(frame)
    report_score_shift(frame)
    report_per_generator(frame)
    report_canaries(frame)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
