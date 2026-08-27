"""
LoRA Adapter Integration for DINO ViT Backbone.
Allows parameter-efficient fine-tuning (PEFT) of ~300M parameter foundation ViT.
"""

from typing import List, Optional
import math
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class LoRALinear(nn.Module):
    """Custom LoRA Linear Layer for non-PEFT fallback."""
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze base linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Reset parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling


def apply_lora_to_dino(
    detector: nn.Module,
    r: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA to DINO ViT backbone attention projection layers.
    """
    if target_modules is None:
        target_modules = ["query", "value", "q_proj", "v_proj", "qkv"]

    # First freeze backbone
    detector.freeze_backbone()

    # Try PEFT library first if model is Hugging Face
    if PEFT_AVAILABLE and hasattr(detector.backbone, "config"):
        try:
            peft_cfg = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=dropout,
                bias="none",
            )
            detector.backbone = get_peft_model(detector.backbone, peft_cfg)
            detector.freeze_backbone_flag = False
            return detector
        except Exception:
            pass

    # Custom recursive LoRA injection
    to_replace = []
    for name, module in list(detector.backbone.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                if any(t in child_name.lower() or t in name.lower() for t in target_modules):
                    to_replace.append((module, child_name, child))

    for module, child_name, child in to_replace:
        setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))

    detector.freeze_backbone_flag = False
    return detector
