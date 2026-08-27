"""
Training Utilities, Distributed Helpers, and Optimizers.
"""

import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, LinearLR


def set_seed(seed: int = 42):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize Distributed Data Parallel (DDP) environment.
    Returns (rank, local_rank, world_size).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", init_method="env://")
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """Tear down DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return True if current process is rank 0 or non-distributed."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build optimizer with differential learning rates for foundation backbone, LoRA, SRM frequency branch, and head.
    """
    opt_cfg = config.get("training", {}).get("optimizer", {})
    backbone_lr = opt_cfg.get("backbone_lr", 0.0)
    lora_lr = opt_cfg.get("lora_lr", 2.0e-5)
    freq_lr = opt_cfg.get("freq_lr", 1.0e-4)
    head_lr = opt_cfg.get("head_lr", 5.0e-4)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)

    # Separate trainable parameters
    lora_params = []
    freq_params = []
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora" in name.lower():
            lora_params.append(param)
        elif "freq" in name.lower() or "srm" in name.lower():
            freq_params.append(param)
        elif "head" in name.lower() or "fusion" in name.lower() or "norm" in name.lower():
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params and backbone_lr > 0:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay})
    if freq_params:
        param_groups.append({"params": freq_params, "lr": freq_lr, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": 1e-3})

    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": head_lr}]

    optimizer_name = opt_cfg.get("name", "adamw").lower()
    if optimizer_name == "adamw":
        return AdamW(param_groups)
    elif optimizer_name == "adam":
        return Adam(param_groups)
    elif optimizer_name == "sgd":
        return SGD(param_groups, momentum=0.9)
    else:
        return AdamW(param_groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    epochs: int,
    steps_per_epoch: int,
) -> Any:
    """Build Cosine Annealing Learning Rate Scheduler with Warmup."""
    sched_cfg = config.get("training", {}).get("scheduler", {})
    warmup_epochs = sched_cfg.get("warmup_epochs", 1)
    min_lr = sched_cfg.get("min_lr", 1.0e-6)

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if warmup_steps > 0:
        warmup_sched = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=min_lr
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps]
        )
    else:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    output_dir: Union[str, Path],
    filename: str = "checkpoint_latest.pt"
):
    """Save model checkpoint and best model weights."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    ckpt_file = out_path / filename
    torch.save(state, str(ckpt_file))

    if is_best:
        best_file = out_path / "best_model.pt"
        torch.save(state, str(best_file))


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load model and optimizer state from checkpoint with recursive prefix resolution."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    
    # Extract state dict
    state_dict = ckpt.get("model_state_dict", ckpt)
    new_state_dict = {}
    
    for k, v in state_dict.items():
        key = k
        # Recursively strip module. (DDP) and detector. (Consistency Wrapper) prefixes
        while key.startswith("module.") or key.startswith("detector."):
            if key.startswith("module."):
                key = key[7:]
            elif key.startswith("detector."):
                key = key[9:]
        new_state_dict[key] = v

    # If model itself is wrapped in ConsistencyDetectorWrapper, adjust keys if needed
    model_keys = set(model.state_dict().keys())
    if any(k.startswith("detector.") for k in model_keys) and not any(k.startswith("detector.") for k in new_state_dict.keys()):
        new_state_dict = {f"detector.{k}": v for k, v in new_state_dict.items()}

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    matched = len(new_state_dict) - len(unexpected)
    print(f"--> Checkpoint loaded successfully! Matched: {matched} weights (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    return ckpt
