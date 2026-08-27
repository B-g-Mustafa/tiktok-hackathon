"""
Spatial Rich Model (SRM) & High-Pass Frequency Forensic Branch.
Implements:
1. Steganographic SRM High-Pass Residual Filter Kernels (non-trainable)
2. Shifted 2D-FFT Log-Magnitude Spectrum Extraction
3. Modern Residual Frequency Stem (~3.5M parameters)
4. Dual-Stream Forensic Detector (Spatial DINOv2 + Signal SRM/FFT)
"""

from typing import Tuple, List, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino_detector import DINODetector, MLPHead


def get_srm_filters() -> torch.Tensor:
    """
    Construct standard Spatial Rich Models (SRM) high-pass residual filter bank.
    Includes:
    - 1st order edge difference filter
    - 2nd order Laplacian / cross difference filter
    - 3x3 / 5x5 sub-band residual filter
    Returns tensor of shape (3, 1, 5, 5).
    """
    filter1 = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32) / 4.0

    filter2 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32) / 12.0

    filter3 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32) / 2.0

    filters = np.stack([filter1, filter2, filter3], axis=0)[:, np.newaxis, :, :]
    return torch.from_numpy(filters)


class ResidualBlock2d(nn.Module):
    """Lightweight 2D Residual Block for Frequency Feature Extraction."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(8, channels), channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(8, channels), channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act2(out + residual)


class FrequencyAblationBranch(nn.Module):
    """
    Forensic Signal & Frequency Stream:
    Extracts high-pass SRM residual maps and 2D-FFT magnitude spectrum,
    then processes them via a lightweight residual CNN to produce a 256-d forensic signal vector.
    """
    def __init__(self, out_features: int = 256):
        super().__init__()
        srm_kernel = get_srm_filters()  # (3, 1, 5, 5)
        self.srm_conv = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.srm_conv.weight = nn.Parameter(srm_kernel, requires_grad=False)

        # 4 Input channels: 3 SRM residual sub-bands + 1 shifted 2D-FFT log-magnitude spectrum
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResidualBlock2d(64),
            nn.MaxPool2d(2, 2),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResidualBlock2d(128),
            nn.MaxPool2d(2, 2),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(128, out_features, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_features),
            nn.GELU(),
            ResidualBlock2d(out_features),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.norm = nn.LayerNorm(out_features)

    def extract_fft_spectrum(self, gray: torch.Tensor) -> torch.Tensor:
        """Compute shifted 2D-FFT log-magnitude spectrum in FP32 for numerical stability."""
        fft = torch.fft.fft2(gray.float())
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift) + 1e-6
        log_mag = torch.log(torch.clamp(magnitude, min=1e-6, max=1e6))
        
        # Robust min-max per sample
        min_v = log_mag.amin(dim=(-2, -1), keepdim=True)
        max_v = log_mag.amax(dim=(-2, -1), keepdim=True)
        norm_mag = (log_mag - min_v) / (max_v - min_v + 1e-6)
        return torch.nan_to_num(norm_mag, nan=0.0, posinf=1.0, neginf=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert RGB to Grayscale in FP32 for forensic noise extraction
        with torch.cuda.amp.autocast(enabled=False):
            x_f32 = x.float()
            gray = 0.299 * x_f32[:, 0:1] + 0.587 * x_f32[:, 1:2] + 0.114 * x_f32[:, 2:3]

            # 1. SRM High-Pass residuals
            srm_residuals = self.srm_conv(gray)
            srm_residuals = torch.clamp(srm_residuals, min=-10.0, max=10.0)

            # 2. 2D-FFT log-magnitude spectrum
            fft_map = self.extract_fft_spectrum(gray)

            # 3. Concatenate (B, 4, H, W)
            freq_input = torch.cat([srm_residuals, fft_map], dim=1)

        # Encode via Residual CNN stem
        x_stem = self.stem(freq_input)
        x_s2 = self.stage2(x_stem)
        x_s3 = self.stage3(x_s2)
        out = self.norm(x_s3)
        return torch.nan_to_num(out, nan=0.0)


class DualStreamDetector(nn.Module):
    """
    Dual-Stream Forensic Detector combining:
    1. Spatial Stream: DINOv2 Vision Transformer with Multi-Layer Token Harvesting & LoRA
    2. Signal Stream: SRM High-Pass Residuals & 2D-FFT Spectrum ConvNet (~3.5M params)
    3. Fused Classification Head: [z_vit, z_freq] -> MLP -> P(AI)
    """
    def __init__(
        self,
        dino_detector: DINODetector,
        freq_dim: int = 256,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dino = dino_detector
        self.freq_branch = FrequencyAblationBranch(out_features=freq_dim)

        dino_feat_dim = self.dino.norm.normalized_shape[0]
        total_dim = dino_feat_dim + freq_dim

        self.fusion_head = MLPHead(
            in_features=total_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract concatenated multi-modal embedding [z_vit, z_freq]."""
        dino_feats = self.dino.extract_features(x)
        freq_feats = self.freq_branch(x)
        return torch.cat([dino_feats, freq_feats], dim=-1)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dino_feats = self.dino.extract_features(x)
        freq_feats = self.freq_branch(x)
        combined_feats = torch.cat([dino_feats, freq_feats], dim=-1)
        logit = self.fusion_head(combined_feats)

        if return_features:
            # Return spatial foundation feature for distortion-invariant alignment
            return logit, dino_feats
        return logit

    def freeze_backbone(self):
        """Freeze underlying foundation vision backbone."""
        if hasattr(self.dino, "freeze_backbone"):
            self.dino.freeze_backbone()

    def unfreeze_backbone(self):
        """Unfreeze underlying foundation vision backbone."""
        if hasattr(self.dino, "unfreeze_backbone"):
            self.dino.unfreeze_backbone()

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logit = self.forward(x)
            return torch.sigmoid(logit)
