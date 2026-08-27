"""
Unified Distributed (DDP) / Single-GPU Training Engine for Robust AIGC Detector.
Supports:
- Frozen DINOv2/v3 baseline
- Heavy distortion augmentations
- Pairwise consistency training with warmup
- LoRA PEFT adaptation
- Full fine-tuning with differential learning rates
- Multi-GPU PyTorch DDP across 4x RTX 3090
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from models import (
    build_detector,
    apply_lora_to_dino,
    ConsistencyDetectorWrapper,
    DualStreamDetector,
)
from datasets import get_sid_dataloaders
from training.losses import ConsistencyLoss, BinaryClassificationLoss
from training.utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Robust AIGC Detector")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size per GPU")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation on validation set")
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation loader and compute AUROC, Accuracy, AP."""
    model.eval()
    local_preds = []
    local_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)

            local_preds.append(probs)
            local_targets.append(targets)

    if len(local_preds) == 0:
        return {"val_auc": 0.5, "val_ap": 0.5, "val_acc": 0.5}

    local_preds = torch.cat(local_preds, dim=0)
    local_targets = torch.cat(local_targets, dim=0)

    # Gather full validation tensors across all DDP ranks
    if dist.is_available() and dist.is_initialized():
        gathered_preds = [torch.zeros_like(local_preds) for _ in range(dist.get_world_size())]
        gathered_targets = [torch.zeros_like(local_targets) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_preds, local_preds)
        dist.all_gather(gathered_targets, local_targets)

        y_pred = torch.cat(gathered_preds, dim=0).cpu().numpy()
        y_true = torch.cat(gathered_targets, dim=0).cpu().numpy()
    else:
        y_pred = local_preds.cpu().numpy()
        y_true = local_targets.cpu().numpy()

    # Compute metrics safely
    try:
        auc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    try:
        ap = float(average_precision_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        ap = 0.5

    binary_preds = (y_pred >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, binary_preds))

    return {
        "val_auc": auc,
        "val_ap": ap,
        "val_acc": acc,
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    is_consistency: bool,
    start_step: int = 0,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
) -> Tuple[Dict[str, float], int]:
    """Execute one training epoch with step-wise tracking."""
    model.train()
    loss_meter = AverageMeter("Loss")
    bce_meter = AverageMeter("BCE")
    feat_meter = AverageMeter("Feat")
    kl_meter = AverageMeter("KL")

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main_process())
    optimizer.zero_grad()
    current_step = start_step

    for step, batch in enumerate(pbar):
        targets = batch["label"].to(device, non_blocking=True)

        with autocast("cuda", enabled=(device.type == "cuda")):
            if is_consistency:
                clean_imgs = batch["clean_image"].to(device, non_blocking=True)
                dist_imgs = batch["distorted_image"].to(device, non_blocking=True)
                
                # Dual-view forward with continuous step warmup
                clean_out, dist_out = model(clean_imgs, dist_imgs)
                loss, loss_dict = criterion(clean_out, dist_out, targets, global_step=current_step)
            else:
                imgs = batch["image"].to(device, non_blocking=True)
                logits = model(imgs)
                loss = criterion(logits, targets)
                loss_dict = {"loss_total": loss.item(), "loss_bce": loss.item()}

            loss = torch.nan_to_num(loss / grad_accum_steps, nan=0.0)

        # Backward
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            current_step += 1

            if scheduler is not None:
                scheduler.step()

        # Update meters
        loss_meter.update(loss_dict.get("loss_total", loss.item()) * grad_accum_steps, targets.size(0))
        if "loss_bce" in loss_dict:
            bce_meter.update(loss_dict["loss_bce"], targets.size(0))
        if "loss_feat" in loss_dict:
            feat_meter.update(loss_dict["loss_feat"], targets.size(0))
        if "loss_kl" in loss_dict:
            kl_meter.update(loss_dict["loss_kl"], targets.size(0))

        if is_main_process() and (step + 1) % 10 == 0:
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "bce": f"{bce_meter.avg:.4f}",
                "feat": f"{feat_meter.avg:.4f}",
                "kl": f"{kl_meter.avg:.4f}",
            })

    return {
        "train_loss": loss_meter.avg,
        "train_bce": bce_meter.avg,
        "train_feat": feat_meter.avg,
        "train_kl": kl_meter.avg,
    }, current_step


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    # Overrides
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size

    # Setup distributed / GPU
    rank, local_rank, world_size = setup_distributed()
    is_dist = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    set_seed(config.get("experiment", {}).get("seed", 42) + rank)

    output_dir = Path(config.get("experiment", {}).get("output_dir", "./outputs"))
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save active config
        with open(output_dir / "config_run.yaml", "w") as f:
            yaml.dump(config, f)
        print(f"=== Starting Experiment: {config.get('experiment', {}).get('name', 'experiment')} ===")
        print(f"World size: {world_size} GPUs | Output dir: {output_dir}")

    # Build Model (build_detector handles LoRA, Intermediate Layers, and DualStream automatically)
    base_detector = build_detector(config)
    if is_main_process():
        if config.get("model", {}).get("lora", {}).get("enabled", False):
            print("LoRA adapters applied to foundation backbone.")
        if config.get("model", {}).get("type") == "dual_stream" or config.get("model", {}).get("dual_stream", {}).get("enabled", False):
            print("Dual-Stream Spatial (DINO) + Signal (SRM/FFT) architecture initialized.")

    # Consistency Wrapper
    is_consistency = config.get("transforms", {}).get("mode", "clean") == "consistency"
    if is_consistency:
        model = ConsistencyDetectorWrapper(base_detector)
    else:
        model = base_detector

    model = model.to(device)

    # Distributed wrapper
    if is_dist:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            static_graph=True,
        )

    # DataLoaders
    data_cfg = config.get("data", {})
    tf_cfg = config.get("transforms", {})
    train_loader, val_loader = get_sid_dataloaders(
        train_dir=data_cfg.get("train_data_dir", "./data/sid_set/train"),
        val_dir=data_cfg.get("val_data_dir", "./data/sid_set/val"),
        image_size=data_cfg.get("image_size", 384),
        mode=tf_cfg.get("mode", "clean"),
        distortion_prob=tf_cfg.get("distortion_prob", 1.0),
        use_compound=tf_cfg.get("use_compound", True),
        batch_size=data_cfg.get("batch_size", 32),
        eval_batch_size=data_cfg.get("eval_batch_size", 64),
        num_workers=data_cfg.get("num_workers", 4),
        max_shards=data_cfg.get("max_train_shards", None),
        is_distributed=is_dist,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    if is_main_process():
        print(f"--> Found {n_train} training samples and {n_val} validation samples.")
        if n_train == 0:
            train_path = data_cfg.get("train_data_dir", "./data/sid_set/train")
            raise FileNotFoundError(
                f"\n[ERROR] No image samples found in training directory: '{train_path}'!\n"
                f"Please verify that the dataset exists and contains real/fake subdirectories.\n"
                f"To download or prepare the dataset, run:\n"
                f"  python datasets/download_datasets.py --dataset sid --output_dir ./data\n"
            )

    if args.eval_only:
        eval_model = model.module if is_dist else model
        if hasattr(eval_model, "detector"):
            eval_model = eval_model.detector
        metrics = evaluate(eval_model, val_loader, device)
        if is_main_process():
            print(f"Validation Results: AUROC = {metrics['val_auc']:.4f} | AP = {metrics['val_ap']:.4f} | Acc = {metrics['val_acc']:.4f}")
        cleanup_distributed()
        return

    # Loss, Optimizer, Scheduler
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 10)
    
    if is_consistency:
        loss_cfg = config.get("loss", {}).get("consistency", {})
        criterion = ConsistencyLoss(
            feat_weight=loss_cfg.get("feat_weight", 0.2),
            kl_weight=loss_cfg.get("kl_weight", 0.1),
            kl_temperature=loss_cfg.get("kl_temperature", 2.0),
            total_warmup_steps=loss_cfg.get("total_warmup_steps", 500),
        )
    else:
        criterion = BinaryClassificationLoss(label_smoothing=config.get("loss", {}).get("label_smoothing", 0.0))

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, epochs=epochs, steps_per_epoch=len(train_loader))
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_auc = 0.0
    start_epoch = 0
    global_step = 0

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc = ckpt.get("best_auc", 0.0)
        global_step = start_epoch * len(train_loader)
        if is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Training Loop
    for epoch in range(start_epoch, epochs):
        if is_dist and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_metrics, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            is_consistency=is_consistency,
            start_step=global_step,
            grad_clip=train_cfg.get("gradient_clip", 1.0),
            grad_accum_steps=train_cfg.get("gradient_accumulation_steps", 1),
        )

        # Validation
        eval_model = model.module if is_dist else model
        if hasattr(eval_model, "detector"):
            eval_model = eval_model.detector

        val_metrics = evaluate(eval_model, val_loader, device)

        if is_main_process():
            val_auc = val_metrics["val_auc"]
            val_acc = val_metrics["val_acc"]
            is_best = val_auc > best_auc
            if is_best:
                best_auc = val_auc

            print(
                f"[Epoch {epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val AUROC: {val_auc:.4f} (Best: {best_auc:.4f}) | "
                f"Val Acc: {val_acc:.4f}"
            )

            # Save state
            state = {
                "epoch": epoch,
                "model_state_dict": (model.module if is_dist else model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "config": config,
            }
            save_checkpoint(state, is_best=is_best, output_dir=output_dir)

    if is_main_process():
        print(f"=== Training Complete! Best Validation AUROC: {best_auc:.4f} ===")

    cleanup_distributed()


if __name__ == "__main__":
    main()
