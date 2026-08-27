#!/usr/bin/env python3
"""Score every image in a directory for being AI-generated.

This is the required deliverable. Contract:

    python scripts/predict.py --image-dir DIR --out preds.json

produces a JSON array:

    [
      {"image_path": "/abs/path/img1.jpg", "pred": 0.9371},
      {"image_path": "/abs/path/img2.png", "pred": 0.0832}
    ]

where `pred` is the calibrated probability that the image is AI-generated.

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

from src.data.io import iter_image_paths, load_image  # noqa: E402
from src.models.base import ConstantDetector, Detector  # noqa: E402

logger = logging.getLogger("predict")


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
) -> tuple[list[dict], int]:
    """Score every image. Returns (records, n_failed)."""
    paths = list(iter_image_paths(image_dir, recursive=recursive))
    if not paths:
        logger.warning("no supported image files found under %s", image_dir)

    records: list[dict] = []
    n_failed = 0

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

        batch_images.append(result.image)
        batch_paths.append(path)

        if len(batch_images) >= batch_size:
            flush()

    flush()

    # One image may have been appended before a later failure, so sort at the
    # end to guarantee deterministic ordering regardless of batching.
    records.sort(key=lambda r: r["image_path"])
    return records, n_failed


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
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.image_dir.is_dir():
        print(f"ERROR: not a directory: {args.image_dir}", file=sys.stderr)
        return 2

    detector = build_detector(args.model, args.checkpoint)
    logger.info("detector: %s", detector.name)

    records, n_failed = run(
        image_dir=args.image_dir,
        detector=detector,
        batch_size=args.batch_size,
        recursive=not args.no_recursive,
        include_failures=args.include_failures,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))

    scored = sum(1 for r in records if r.get("pred") is not None)
    logger.info("scored %d image(s); %d unreadable -> %s", scored, n_failed, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
