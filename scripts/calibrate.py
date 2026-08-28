#!/usr/bin/env python3
"""Fit a calibration for a target dataset and save it into the checkpoint.

Solves the specific failure where a detector ranks well on an unseen dataset
but classifies badly at threshold 0.5 -- e.g. AUROC 0.8441 against 0.5544
accuracy on GenImage. That gap is a misplaced decision boundary, not a weak
representation, and it is fixable without retraining or a GPU.

Runs on an EXISTING predictions file, so it needs no model and no images:

    # 1. score the target set (writes preds.json + preds.metrics.json)
    python scripts/predict.py --model siglip2 --checkpoint CKPT \
        --image-dir DIR --out preds.json

    # 2. fit the correction and save it into the checkpoint
    python scripts/calibrate.py --predictions preds.json --checkpoint CKPT

    # 3. re-run predict.py -- it now picks up calibration.json automatically

By default the fit is evaluated honestly: the predictions are split in half,
the correction is fitted on one half and reported on the other, so the printed
gain is not the optimistic in-sample number.

IMPORTANT: calibration is a monotone transform of the score, so AUROC and AP
are mathematically unchanged. It improves accuracy, balanced accuracy, ECE and
Brier. Report it as a threshold fix, never as a generalization gain.
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

from src.calibration import LogitScaler  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.models.siglip_detector import CALIBRATION_NAME  # noqa: E402

logger = logging.getLogger("calibrate")


def load_scored(predictions: Path, manifest: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Join a predictions JSON against a labeled manifest, returning
    (scores, labels) for the rows present in both."""
    records = json.loads(predictions.read_text())

    if manifest is None:
        raise SystemExit(
            "--manifest is required (a supervised fit needs target labels); "
            "use --prior instead to calibrate without labels"
        )

    frame = pd.read_parquet(manifest)
    label_by_path = {
        str(Path(row.path).resolve()): int(row.label) for row in frame.itertuples()
    }

    scores, labels = [], []
    for record in records:
        # Resolve BOTH sides before joining. predict.py already writes resolved
        # paths, but a predictions file from anywhere else may not, and a
        # symlinked or /var-style path would then silently match nothing --
        # producing "no predictions matched" on data that is perfectly fine.
        key = str(Path(record["image_path"]).resolve())
        label = label_by_path.get(key)
        if label is not None and record.get("pred") is not None:
            scores.append(float(record["pred"]))
            labels.append(label)

    if not scores:
        raise SystemExit(
            f"no predictions in {predictions} matched a row in {manifest} -- "
            f"check that both refer to the same images"
        )
    if len(scores) < len(records):
        logger.warning(
            "%d/%d predictions matched a label", len(scores), len(records)
        )

    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=int)


def _report(title: str, y_true: np.ndarray, y_score: np.ndarray) -> None:
    m = compute_metrics(y_true, y_score)
    logger.info(
        "%-18s AUROC %.4f | acc %.4f | bal_acc %.4f | ECE %.4f | Brier %.4f",
        title, m.auroc, m.accuracy, m.balanced_accuracy, m.ece, m.brier,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="preds.json from predict.py or dyno/files/infer.py.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="manifest.parquet with path/label. Required unless --prior.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help=f"Checkpoint dir to write {CALIBRATION_NAME} into. "
                             f"Omit to fit and report without saving.")
    parser.add_argument("--prior", type=float, default=None,
                        help="Fit WITHOUT labels, assuming this fraction of the "
                             "target set is AI-generated (e.g. 0.5 for a balanced "
                             "set). Ignores --manifest labels for fitting.")
    parser.add_argument("--fit-fraction", type=float, default=0.5,
                        help="Fraction held out for fitting; the rest is used to "
                             "report an honest out-of-sample gain. 1.0 fits on "
                             "everything and reports in-sample (optimistic).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    configure_logging(format="%(message)s", log_file=args.log_file)

    scores, labels = load_scored(args.predictions, args.manifest)
    logger.info("%d scored images (%d generated / %d authentic)",
                len(labels), int((labels == 1).sum()), int((labels == 0).sum()))

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(scores))
    n_fit = len(scores) if args.fit_fraction >= 1.0 else int(len(scores) * args.fit_fraction)
    if n_fit < 2:
        raise SystemExit(f"too few samples to fit ({n_fit})")
    fit_idx = order[:n_fit]
    eval_idx = order[n_fit:] if n_fit < len(scores) else order

    scaler = LogitScaler()
    if args.prior is not None:
        scaler.fit_to_prior(scores[fit_idx], prior=args.prior)
    else:
        scaler.fit(scores[fit_idx], labels[fit_idx])

    in_sample = n_fit >= len(scores)
    logger.info("")
    logger.info("fitted: %s", scaler)
    logger.info("evaluated on %d held-out images%s",
                len(eval_idx), " (IN-SAMPLE, optimistic)" if in_sample else "")
    logger.info("")
    _report("before", labels[eval_idx], scores[eval_idx])
    _report("after", labels[eval_idx], scaler.transform(scores[eval_idx]))
    logger.info("")
    logger.info("AUROC/AP are unchanged by construction -- calibration is a "
                "monotone transform. Only the threshold-dependent numbers move.")

    if args.checkpoint is not None:
        out = args.checkpoint / CALIBRATION_NAME
        scaler.save(out)
        logger.info("")
        logger.info("saved -> %s (predict.py will now apply it automatically)", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
