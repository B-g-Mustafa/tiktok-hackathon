#!/usr/bin/env python3
"""LoRA fine-tune the vision encoder + head end-to-end.

This is the step up from the frozen linear probe (`train_and_evaluate.py`),
and the one that actually competes on absolute accuracy: full fine-tuning is
what NTIRE 2026's top teams did, but full fine-tuning also tends to memorize
the artifacts of the generators it trained on, which is exactly the failure
mode an unseen-generator hidden test set is designed to punish. LoRA adapts a
low-rank subspace of every attention/MLP projection instead of the full
weight matrix, which keeps the update small enough that the representation is
perturbed rather than overwritten -- closer in spirit to the frozen probe than
to training from scratch, while still letting the model actually learn from
labels.

Run `scripts/materialize_images.py` for the relevant splits first; this script
is pure local I/O plus GPU compute.

Usage:
    python scripts/materialize_images.py --split train
    python scripts/materialize_images.py --split cross_generator
    python scripts/finetune_lora.py --epochs 3 --lora-rank 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.local_dataset import LocalImageDataset, collate_list  # noqa: E402
from src.models.budget import ParameterBudget  # noqa: E402
from src.models.encoders import ENCODER_CATALOG  # noqa: E402
from src.models.lora_encoder import DEFAULT_LORA_TARGETS, LoraEncoder  # noqa: E402
from src.models.torch_head import TorchLinearHead  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.transforms.robustness import TrainAugment  # noqa: E402

logger = logging.getLogger("finetune_lora")


@torch.no_grad()
def evaluate(encoder: LoraEncoder, head: TorchLinearHead, loader: DataLoader) -> float:
    """Held-out AUROC. Used for monitoring during training, NOT the final
    robustness claim -- that comes from train_and_evaluate.py's transform grid
    run against the real cross_generator / content_matched_control caches.
    """
    encoder.eval()
    head.eval()

    all_scores, all_labels = [], []
    for images, labels in loader:
        features = encoder.forward_features(images)
        logits = head(features)
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(np.array(labels))

    encoder.train()
    head.train()

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("artifacts/images"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Held out from the train manifest for monitoring.")
    parser.add_argument("--encoder", default="siglip2-so400m-378",
                        choices=sorted(ENCODER_CATALOG))
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/lora"))
    args = parser.parse_args()

    configure_logging(format="%(asctime)s %(levelname)s: %(message)s")
    torch.manual_seed(args.seed)

    train_dir = args.images_dir / args.train_split
    if not (train_dir / "manifest.parquet").exists():
        logger.error(
            "no materialized images at %s -- run scripts/materialize_images.py "
            "--split %s first", train_dir, args.train_split,
        )
        return 2

    encoder = LoraEncoder(
        encoder=args.encoder,
        n_layers=args.n_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=args.device,
    )
    head = TorchLinearHead(encoder.spec.output_dim).to(encoder.device)

    budget = ParameterBudget()
    budget.add(
        f"{args.encoder} vision tower (frozen base)",
        ENCODER_CATALOG[args.encoder]["params"],
    )
    budget.add(
        f"LoRA adapters (rank {args.lora_rank})",
        encoder.n_trainable_parameters,
        trainable=True,
    )
    budget.add("linear head", head.n_parameters, trainable=True)
    budget.check()

    print("=" * 78)
    print("LORA FINE-TUNE")
    print("=" * 78)
    print(budget.to_markdown())
    print(f"\ndevice: {encoder.device}")

    # -- data -----------------------------------------------------------
    augment = TrainAugment(seed=args.seed)
    full_train = LocalImageDataset(
        train_dir, crop_size=encoder.image_size, transform=augment, crop_mode="random"
    )

    n_val = max(1, int(len(full_train) * args.val_fraction))
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=generator
    )
    # Validation must not see training-time augmentation severity variance;
    # rebuild the val subset over a clean (unaugmented, centre-cropped) view of
    # the same underlying manifest rows so the monitoring number is stable.
    val_dataset = LocalImageDataset(
        train_dir, crop_size=encoder.image_size, transform=None, crop_mode="center"
    )
    val_subset = torch.utils.data.Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_list, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_list,
    )
    print(f"train: {n_train:,} images  |  held-out monitor: {n_val:,} images")

    # -- optimizer --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        list(encoder.trainable_parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # -- train --------------------------------------------------------------
    best_val_auroc = -1.0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            features = encoder.forward_features(images)
            logits = head(features)
            targets = torch.tensor(
                labels, dtype=torch.float32, device=logits.device
            )

            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach())
            n_batches += 1

        val_auroc = evaluate(encoder, head, val_loader)
        elapsed = (time.time() - started) / 60
        logger.info(
            "epoch %d/%d  loss=%.4f  val_auroc=%.4f  (%.1f min elapsed)",
            epoch, args.epochs, running_loss / max(n_batches, 1), val_auroc, elapsed,
        )

        if val_auroc == val_auroc and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            args.output.mkdir(parents=True, exist_ok=True)
            encoder.save_adapter(args.output)
            torch.save(head.state_dict(), args.output / "head.pt")
            logger.info("  new best (val_auroc=%.4f) -> %s", val_auroc, args.output)

    meta = {
        "encoder": args.encoder,
        "n_layers": args.n_layers,
        "layers": list(encoder.layers),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": list(DEFAULT_LORA_TARGETS),
        "config_hash": encoder.spec.config_hash(),
        "best_val_auroc": best_val_auroc,
        "epochs": args.epochs,
        "n_train_images": n_train,
        "trainable_parameters": encoder.n_trainable_parameters + head.n_parameters,
        "total_parameters": budget.total,
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 78)
    print(f"best held-out AUROC: {best_val_auroc:.4f}")
    print(f"checkpoint -> {args.output}")
    print(
        "\nThis number is a training-time sanity check, not the robustness "
        "claim. Run scripts/train_and_evaluate.py-style transform-grid "
        "evaluation against cross_generator / content_matched_control caches "
        "for the real headline numbers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
