"""Training package for Robust AIGC Detector."""

from .losses import ConsistencyLoss, BinaryClassificationLoss
from .utils import (
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

__all__ = [
    "ConsistencyLoss",
    "BinaryClassificationLoss",
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "get_rank",
    "build_optimizer",
    "build_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "AverageMeter",
    "set_seed",
]
