"""
Consistency Detector Wrapper for Dual-Stream Forward Passes.
Optimizes training by combining (x_clean, x_distorted) into a single batch forward pass.
"""

from typing import Tuple, Dict, Any, Union, Optional, List
import torch
import torch.nn as nn
from .dino_detector import DINODetector


class ConsistencyDetectorWrapper(nn.Module):
    """
    Consistency Wrapper for DINODetector.
    Efficiently processes (x_clean, x_distorted) in a unified batched forward pass.
    """
    def __init__(self, detector: DINODetector):
        super().__init__()
        self.detector = detector

    def forward(
        self,
        x_clean: torch.Tensor,
        x_distorted: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        If x_distorted is None, performs standard single forward pass.
        If x_distorted is provided, executes batched dual-view forward pass.
        """
        if x_distorted is None:
            return self.detector(x_clean)

        batch_size = x_clean.size(0)
        # Combine clean and distorted along batch axis for single high-speed forward pass
        x_combined = torch.cat([x_clean, x_distorted], dim=0)

        logits_combined, feats_combined = self.detector(x_combined, return_features=True)

        logits_clean = logits_combined[:batch_size]
        logits_dist = logits_combined[batch_size:]

        feats_clean = feats_combined[:batch_size]
        feats_dist = feats_combined[batch_size:]

        return (logits_clean, feats_clean), (logits_dist, feats_dist)
