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
import itertools
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeoutError
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

# Wall-clock budget for one image's view-building (crop/JPEG/blur/etc. -- all
# CPU-bound PIL work). Legitimate view-building is milliseconds; this exists
# to catch a genuinely pathological image, not to police normal speed.
VIEW_BATCH_TIMEOUT_SECONDS = 60.0


def build_train_views(
    image,
    n_views: int,
    crop_size: int,
    seed: int,
) -> list[tuple[str, object]]:
    """One clean view plus (n_views - 1) degraded views of the same decode.

    Crop position is jittered independently per view so crop location stays
    decorrelated from both label and severity.

    Builds its own `TrainAugment`/`random.Random` from `seed` rather than
    taking shared ones, so this can run concurrently across images (view
    building is submitted to a thread pool below) without threads racing on
    the same `random.Random` instance -- reproducible per-image regardless of
    which thread actually executes it, since `seed` is derived from the
    image's position in the source, not wall-clock/thread-scheduling order.
    """
    augment = TrainAugment(seed=seed)
    rng = random.Random(seed)

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


def _seed_for_key(base_seed: int, key: str) -> int:
    """Deterministic per-image seed, stable regardless of processing order.

    Train-mode view building used to draw from ONE shared `random.Random`
    walked sequentially across every image -- fine single-threaded, but
    unsafe once view building runs concurrently (see below), since threads
    would race on that shared state. Hashing the image's own stable `key`
    keeps every image's augmentation reproducible under a given `--seed`
    independent of which thread processes it or whether this run is a resume
    that skips earlier images -- the same key always maps to the same seed.
    """
    import hashlib

    digest = hashlib.sha256(f"{base_seed}:{key}".encode()).hexdigest()
    return int(digest[:8], 16)


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
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel decode + view-building threads. Set to "
                             "1 for the old fully-sequential behaviour.")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                         help="Write the cache to disk every N images, so an "
                              "interrupted run can resume instead of "
                              "restarting from scratch. 0 disables.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many images (smoke test).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Also write progress to this file (useful under "
                             "sbatch/qsub, where stdout is buffered and "
                             "doesn't update live).")
    parser.add_argument(
        "--local-manifest", type=Path, default=None,
        help="Directory containing a manifest.parquet (from "
             "prepare_finetune_data.py). If given, images are read from "
             "local disk instead of remote HF shards.",
    )
    args = parser.parse_args()

    configure_logging(format="%(asctime)s %(levelname)s: %(message)s", log_file=args.log_file)

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

    if args.mode == "eval":
        transforms = eval_grid()
        if args.combinations:
            transforms = transforms + eval_grid_combinations()
        logger.info("eval mode: %d transforms per image", len(transforms))
        make_views = lambda im, seed: build_eval_views(im, crop_size, transforms)  # noqa: E731
        views_per_image = len(transforms)
    else:
        make_views = lambda im, seed: build_train_views(  # noqa: E731
            im, args.n_views, crop_size, seed
        )
        views_per_image = args.n_views

    # -- output path (computed up front: resume needs to know it before the
    # -- data source is even built, to filter out already-cached images) ----
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

    # -- resume: load whatever this exact (encoder, mode, config) combo has
    # -- already cached, and skip those images entirely rather than re-fetch
    # -- + re-encode them ----------------------------------------------------
    features: list[np.ndarray] = []
    labels: list[int] = []
    view_names: list[str] = []
    keys: list[str] = []
    generators: list[str] = []
    done_image_keys: set[str] = set()

    if out_path.exists():
        try:
            with np.load(out_path) as loaded:
                features = [loaded["features"]]
                labels = list(loaded["labels"])
                view_names = list(loaded["view_names"])
                keys = list(loaded["keys"])
                generators = list(loaded["generators"])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cache at %s is unreadable (%s) -- treating as no prior "
                "progress and starting fresh",
                out_path, exc,
            )
            features, labels, view_names, keys, generators = [], [], [], [], []
        else:
            # Every row sharing an image's `key` was written together (see
            # the flush()/checkpoint invariant below), so the SET of keys
            # already present is exactly the set of fully-cached images --
            # never a partial one that could be silently under-counted.
            done_image_keys = set(keys)
            logger.info(
                "resuming %s: %d images already cached (%d feature rows)",
                out_path, len(done_image_keys), len(keys),
            )

    # -- build the (filtered) data source ------------------------------------
    if args.local_manifest is not None:
        from src.data.local_dataset import iter_local_images

        manifest_path = args.local_manifest / "manifest.parquet"
        if not manifest_path.exists():
            logger.error("no manifest at %s (run scripts/prepare_finetune_data.py first)",
                         manifest_path)
            return 2
        full = pd.read_parquet(manifest_path)
        if args.limit:
            full = full.head(args.limit)
        n_rows_total = len(full)
        row_keys = full["key"].astype(str) + "#0"  # matches FetchedImage.key for local rows
        remaining = full[~row_keys.isin(done_image_keys)]
        image_source = iter_local_images(
            args.local_manifest, manifest=remaining, workers=args.workers
        )
        logger.info(
            "local manifest %s: %d rows (%d remaining)",
            args.local_manifest, n_rows_total, len(remaining),
        )
    else:
        split_path = args.splits_dir / f"{args.split}.parquet"
        if not split_path.exists():
            logger.error(
                "split not found: %s (run scripts/build_splits.py)", split_path
            )
            return 2

        full = pd.read_parquet(split_path)
        if args.limit:
            full = full.head(args.limit)  # avoids fetching unneeded shards
        n_rows_total = len(full)
        row_keys = full["shard"].astype(str) + "#" + full["row_in_shard"].astype(str)
        remaining = full[~row_keys.isin(done_image_keys)]
        image_source = iter_selected_images(args.repo, remaining, workers=args.workers)
        logger.info(
            "split %s: %d rows (%d remaining)",
            args.split, n_rows_total, len(remaining),
        )

    pending_images: list = []
    pending_meta: list[tuple[int, str, str, str]] = []
    n_images = len(done_image_keys)
    n_images_this_run = 0
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

    def save_checkpoint() -> None:
        if not keys:
            return
        matrix = np.concatenate(features, axis=0).astype(np.float16)
        # `np.savez_compressed` silently appends ".npz" to any path that
        # doesn't already end with it, so the tmp name must itself end in
        # ".npz" -- ".npz.tmp" would actually get written to
        # "....npz.tmp.npz", and the os.replace below would then fail
        # (confirmed: this was a real bug caught by an end-to-end test).
        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            features=matrix,
            labels=np.asarray(labels, dtype=np.int8),
            view_names=np.asarray(view_names),
            keys=np.asarray(keys),
            generators=np.asarray(generators),
        )
        os.replace(tmp_path, out_path)
        # Collapse `features` to the one array just written, so the next
        # checkpoint's concatenate doesn't redo work for rows already merged.
        features.clear()
        features.append(matrix)

    def process_one(fetched, views) -> None:
        nonlocal n_images, n_images_this_run

        for view_name, view in views:
            pending_images.append(view)
            pending_meta.append(
                (fetched.label, view_name, fetched.key, fetched.model_name)
            )
        # Flushed only here, at the image boundary -- never mid-image -- so a
        # checkpoint right after can never catch one image's views
        # half-written (which would make it wrongly look "done" on resume and
        # permanently lose the other half).
        if len(pending_images) >= args.batch_size:
            flush()

        n_images += 1
        n_images_this_run += 1
        if n_images_this_run % 200 == 0:
            rate = n_images_this_run / max(time.time() - started, 1e-6)
            logger.info(
                "%d/%d images (%.1f img/s, %d feature rows)",
                n_images, n_rows_total, rate, sum(len(f) for f in features) or len(keys),
            )

        if args.checkpoint_every and n_images_this_run % args.checkpoint_every == 0:
            flush()
            save_checkpoint()
            logger.info("checkpoint: %d images cached -> %s", n_images, out_path)

    executor = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    chunk_size = args.workers * 4 if executor is not None else 1
    source_iter = iter(image_source)

    try:
        while True:
            chunk = list(itertools.islice(source_iter, chunk_size))
            if not chunk:
                break

            if executor is None:
                for fetched in chunk:
                    views = make_views(fetched.image, _seed_for_key(args.seed, fetched.key))
                    process_one(fetched, views)
                continue

            # Submit the whole chunk concurrently and collect as each
            # finishes (not submit-then-immediately-block one at a time,
            # which would give zero real overlap) -- same as_completed +
            # batch-timeout pattern as parquet_images._decode_batch, and the
            # same id()-based tracking as local_dataset._save_batch_concurrent
            # since FetchedImage is a plain (unhashable) dataclass.
            futures = {
                executor.submit(
                    make_views, fetched.image, _seed_for_key(args.seed, fetched.key)
                ): fetched
                for fetched in chunk
            }
            results: dict[int, tuple] = {}
            try:
                for future in as_completed(futures, timeout=VIEW_BATCH_TIMEOUT_SECONDS):
                    fetched = futures[future]
                    try:
                        results[id(fetched)] = (fetched, future.result())
                    except Exception:  # noqa: BLE001
                        logger.warning("view-building failed for %s", fetched.key)
            except FutureTimeoutError:
                stuck = sum(1 for f in futures if not f.done())
                logger.warning(
                    "%d image(s) in this batch did not finish view-building "
                    "within %.0fs -- skipping",
                    stuck, VIEW_BATCH_TIMEOUT_SECONDS,
                )

            # Process in the chunk's original order for stable, deterministic
            # checkpoint contents run-to-run.
            for fetched in chunk:
                hit = results.get(id(fetched))
                if hit is not None:
                    process_one(*hit)

        flush()
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    if not keys:
        logger.error("no features extracted; nothing written")
        return 1

    save_checkpoint()
    matrix = features[0]

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
        "wrote %s  shape=%s  (%d images x %d views, %d new this run) in %.1f min",
        out_path, matrix.shape, n_images, views_per_image, n_images_this_run,
        (time.time() - started) / 60,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
