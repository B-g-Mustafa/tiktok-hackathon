#!/usr/bin/env python3
"""Extract and cache frozen-encoder features for a split.

Run this on the GPU box. It is the only expensive step in the pipeline; once it
completes, training a head and running the full robustness matrix takes minutes,
which is what makes a wide ablation sweep affordable.

Two decisions matter for throughput and for correctness:

**Decode once, augment K times.** PNG decode dominates the CPU budget (roughly
8-15 ms at 512px, ~40 ms at 1024px), and the GPU will otherwise sit idle waiting
for it. Each image is decoded a single time and all K augmented views are
derived from that one decode, amortizing the cost K-fold.

**Views span severity, and go beyond the organizers' settings.** One clean view
plus K-1 degraded ones, sampled continuously from ranges that bracket the
specified transforms. Training at exactly {90, 70, 50, 30} JPEG would let the
head memorize four quantization tables; sampling quality across a wider range
puts the evaluation settings in the *interior* of the training distribution
instead of at its edge.

Every cache file is stamped with the encoder's config hash and is refused on
load if it does not match, because silently mixing feature sets produced under
different settings is invisible in the metrics.

Usage:
    python scripts/cache_features.py --split train --n-views 8
    python scripts/cache_features.py --split content_matched_control --n-views 1
"""

from __future__ import annotations

import argparse
import json
import logging
import itertools
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.parquet_images import iter_selected_images  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.models.encoders import ENCODER_CATALOG, FrozenEncoder  # noqa: E402
from src.transforms.crop import native_crop  # noqa: E402
from src.transforms.robustness import (  # noqa: E402
    TrainAugment,
    eval_grid,
    eval_grid_combinations,
)

logger = logging.getLogger("cache_features")

DEFAULT_REPO = "OwensLab/CommunityForensics-Small"


def build_train_views(
    image,
    n_views: int,
    crop_size: int,
    augment: TrainAugment,
    rng: random.Random,
) -> list[tuple[str, object]]:
    """One clean view plus (n_views - 1) degraded views of the same decode.

    Crop position is jittered independently per view so crop location stays
    decorrelated from both label and severity.
    """
    views = [("clean", native_crop(image, crop_size, mode="random", rng=rng).image)]

    for i in range(max(0, n_views - 1)):
        degraded = augment(image)
        views.append(
            (f"aug{i}", native_crop(degraded, crop_size, mode="random", rng=rng).image)
        )

    return views


def build_eval_views(image, crop_size: int, transforms) -> list[tuple[str, object]]:
    """One deterministic view per transform in the robustness grid.

    Transform first, crop second -- the same order a real redistribution
    pipeline applies them, and the only order under which a whole-image
    operation like 0.25x downscaling means anything. Cropping first would make
    "resize" operate on an already-cropped patch and understate its damage.

    Centre cropping keeps this reproducible so matrix cells are comparable
    across models.
    """
    views = []
    for transform in transforms:
        transformed = transform(image)
        views.append(
            (transform.name, native_crop(transformed, crop_size, mode="center").image)
        )
    return views


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train",
                        help="Split name under --splits-dir (without .parquet).")
    parser.add_argument("--splits-dir", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--encoder", default="siglip2-so400m-378",
                        choices=sorted(ENCODER_CATALOG))
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument(
        "--adapter", type=Path, default=None,
        help="Path to a LoRA checkpoint (from finetune_lora.py). If given, "
             "features are extracted with the fine-tuned encoder instead of "
             "the frozen one -- e.g. to run the robustness matrix against a "
             "fine-tuned model with the exact same evaluation code path.",
    )
    parser.add_argument("--mode", default="train", choices=("train", "eval"),
                        help="train: random augmented views. "
                             "eval: one deterministic view per robustness transform.")
    parser.add_argument("--n-views", type=int, default=8,
                        help="[train mode] views per image (1 clean + N-1 degraded).")
    parser.add_argument("--combinations", action="store_true",
                        help="[eval mode] also include multi-step transform chains.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many images (smoke test).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--local-manifest", type=Path, default=None,
        help="Directory containing a manifest.parquet (from "
             "prepare_local_dataset.py or materialize_images.py). If given, "
             "images are read from local disk instead of remote HF shards -- "
             "use this to run phase 1 against GenImage or any other local "
             "real/fake directory.",
    )
    args = parser.parse_args()

    configure_logging(format="%(asctime)s %(levelname)s: %(message)s")

    if args.local_manifest is not None:
        from src.data.local_dataset import iter_local_images

        manifest_path = args.local_manifest / "manifest.parquet"
        if not manifest_path.exists():
            logger.error(
                "no manifest at %s (run scripts/prepare_local_dataset.py first)",
                manifest_path,
            )
            return 2
        n_rows = len(pd.read_parquet(manifest_path))
        if args.limit:
            n_rows = min(n_rows, args.limit)
        image_source = iter_local_images(args.local_manifest)
        logger.info("local manifest %s: %d rows", args.local_manifest, n_rows)
    else:
        split_path = args.splits_dir / f"{args.split}.parquet"
        if not split_path.exists():
            logger.error(
                "split not found: %s (run scripts/build_splits.py)", split_path
            )
            return 2

        selection = pd.read_parquet(split_path)
        if args.limit:
            selection = selection.head(args.limit)  # avoids fetching unneeded shards
        n_rows = len(selection)
        image_source = iter_selected_images(args.repo, selection)
        logger.info("split %s: %d rows", args.split, n_rows)

    # `selection.head()` above already bounds the remote case (and does so
    # without fetching shards we don't need); islice is the uniform backstop
    # that also covers the local-manifest case, where nothing pre-bounds it.
    if args.limit:
        image_source = itertools.islice(image_source, args.limit)

    if args.adapter is not None:
        from src.models.lora_encoder import LoraEncoder

        adapter_meta = json.loads((args.adapter / "meta.json").read_text())
        encoder = LoraEncoder(
            encoder=adapter_meta["encoder"],
            n_layers=adapter_meta["n_layers"],
            lora_rank=adapter_meta["lora_rank"],
            lora_alpha=adapter_meta["lora_alpha"],
            device=args.device,
        )
        encoder.load_adapter(args.adapter)
        encoder.eval()
        logger.info("loaded LoRA adapter from %s", args.adapter)
    else:
        encoder = FrozenEncoder(
            encoder=args.encoder, n_layers=args.n_layers, device=args.device
        )

    spec = encoder.spec
    crop_size = spec.image_size

    logger.info(
        "encoder %s (%s params) on %s | crop %d | feature dim %d | hash %s",
        spec.encoder, f"{encoder.n_parameters:,}", encoder.device,
        crop_size, spec.output_dim, spec.config_hash(),
    )

    augment = TrainAugment(seed=args.seed)
    rng = random.Random(args.seed)

    if args.mode == "eval":
        transforms = eval_grid()
        if args.combinations:
            transforms = transforms + eval_grid_combinations()
        logger.info("eval mode: %d transforms per image", len(transforms))
        make_views = lambda im: build_eval_views(im, crop_size, transforms)  # noqa: E731
        views_per_image = len(transforms)
    else:
        make_views = lambda im: build_train_views(  # noqa: E731
            im, args.n_views, crop_size, augment, rng
        )
        views_per_image = args.n_views

    features: list[np.ndarray] = []
    labels: list[int] = []
    view_names: list[str] = []
    keys: list[str] = []
    generators: list[str] = []

    pending_images: list = []
    pending_meta: list[tuple[int, str, str, str]] = []
    n_images = 0
    started = time.time()

    def flush() -> None:
        if not pending_images:
            return
        batch = encoder.extract(pending_images)
        features.append(batch)
        for label, view, key, generator in pending_meta:
            labels.append(label)
            view_names.append(view)
            keys.append(key)
            generators.append(generator)
        pending_images.clear()
        pending_meta.clear()

    for fetched in image_source:
        for view_name, view in make_views(fetched.image):
            pending_images.append(view)
            pending_meta.append(
                (fetched.label, view_name, fetched.key, fetched.model_name)
            )
            if len(pending_images) >= args.batch_size:
                flush()

        n_images += 1
        if n_images % 200 == 0:
            rate = n_images / max(time.time() - started, 1e-6)
            logger.info(
                "%d/%d images (%.1f img/s, %d feature rows)",
                n_images, n_rows, rate, sum(len(f) for f in features),
            )

    flush()

    if not features:
        logger.error("no features extracted; nothing written")
        return 1

    matrix = np.concatenate(features, axis=0).astype(np.float16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "eval" if args.mode == "eval" else f"v{args.n_views}"
    split_name = (
        args.local_manifest.name if args.local_manifest is not None else args.split
    )
    # A LoRA-tuned encoder and the frozen one share the same `spec.encoder`
    # name (e.g. "siglip2-so400m-378") -- only `config_hash()` differs, since
    # LoraFeatureSpec folds lora_rank/alpha into it. That's enough to avoid a
    # silent cache-mismatch load, but it means a glob like `split__encoder__*`
    # would match BOTH files with no visible way to tell them apart. Marking
    # LoRA runs explicitly in the filename keeps `ls` and shell globs honest.
    encoder_tag = f"{spec.encoder}-lora" if args.adapter is not None else spec.encoder
    stem = f"{split_name}__{encoder_tag}__{spec.config_hash()}__{suffix}"
    out_path = args.output_dir / f"{stem}.npz"

    np.savez_compressed(
        out_path,
        features=matrix,
        labels=np.asarray(labels, dtype=np.int8),
        view_names=np.asarray(view_names),
        keys=np.asarray(keys),
        generators=np.asarray(generators),
    )

    meta = {
        "split": args.split,
        "encoder": args.encoder,
        "timm_name": spec.timm_name,
        "config_hash": spec.config_hash(),
        "encoder_parameters": encoder.n_parameters,
        "feature_dim": spec.output_dim,
        "layers": list(spec.layers),
        "crop_size": crop_size,
        "mode": args.mode,
        "views_per_image": views_per_image,
        "n_images": n_images,
        "n_rows": int(matrix.shape[0]),
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    logger.info(
        "wrote %s  shape=%s  (%d images x %d views) in %.1f min",
        out_path, matrix.shape, n_images, views_per_image,
        (time.time() - started) / 60,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
