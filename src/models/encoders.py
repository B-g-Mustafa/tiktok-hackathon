"""Frozen vision encoders and their feature extraction.

Design decisions and why
------------------------

**Frozen, not fine-tuned.** Linear probes on frozen foundation features
generalize to unseen generators markedly better than fine-tuned backbones,
which tend to memorize the artifacts of the generators they were trained on.
Since the hidden evaluation set is explicitly expected to contain unseen
generators, that is the property we need. Freezing also makes feature caching
possible, which turns backbone and head selection into minute-long experiments
instead of hour-long ones.

**Vision tower only.** Loading `google/siglip2-so400m-patch14-384` through a
generic AutoModel pulls in a 707.7M-parameter text tower -- 294.9M of which is
just the 256K-entry vocabulary embedding -- none of which ever runs. Against a
hard 2B budget that is a third of the allowance spent on dead weight. timm
gives us the vision tower alone: 428,225,600 parameters, verified.

**Multi-layer features, not just the pooled output.** The pooled vector is
optimized for semantic retrieval, which is not our task. Mid-to-late block
activations retain texture and local statistics that pooling discards. Caching
several layers costs a few extra kilobytes per image but keeps the option open:
the pooled-only choice is irreversible without re-encoding the entire dataset.

Note that SigLIP ViTs have no CLS token -- `num_prefix_tokens` is 0 and pooling
is attention-based (MAP) -- so the usual "CLS plus patch mean" recipe does not
apply. We use the model's own MAP-pooled output plus mean-pooled patch tokens
from the last few blocks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from PIL import Image

__all__ = [
    "ENCODER_CATALOG",
    "FeatureSpec",
    "FrozenEncoder",
    "resolve_device",
]


# Verified parameter counts (vision tower only, timm, num_classes=0).
# `params` is recorded here so the budget check does not require instantiating
# every candidate model.
ENCODER_CATALOG: dict[str, dict] = {
    "siglip2-so400m-378": {
        "timm_name": "vit_so400m_patch14_siglip_378.v2_webli",
        "params": 428_225_600,
        "feature_dim": 1152,
        "image_size": 378,
        "license": "Apache-2.0",
    },
    "siglip2-large-384": {
        "timm_name": "vit_large_patch16_siglip_384.v2_webli",
        "params": 316_283_904,
        "feature_dim": 1024,
        "image_size": 384,
        "license": "Apache-2.0",
    },
    "siglip2-base-384": {
        "timm_name": "vit_base_patch16_siglip_384.v2_webli",
        "params": 93_176_064,
        "feature_dim": 768,
        "image_size": 384,
        "license": "Apache-2.0",
    },
}


def resolve_device(preference: str | None = None) -> torch.device:
    """Pick a compute device, honouring an explicit preference."""
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class FeatureSpec:
    """Everything that determines what a cached feature vector means.

    `config_hash` is stamped into cache filenames and asserted on load. Silently
    mixing features produced under different settings is the classic way these
    pipelines go quietly wrong, and it is invisible in the metrics.
    """

    encoder: str
    timm_name: str
    feature_dim: int
    image_size: int
    layers: tuple[int, ...]
    include_pooled: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def output_dim(self) -> int:
        """Width of one cached feature vector."""
        return self.feature_dim * (len(self.layers) + (1 if self.include_pooled else 0))

    def config_hash(self) -> str:
        payload = "|".join(
            [
                self.encoder,
                self.timm_name,
                str(self.feature_dim),
                str(self.image_size),
                ",".join(map(str, self.layers)),
                str(self.include_pooled),
                repr(sorted(self.extra.items())),
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class FrozenEncoder:
    """A frozen vision tower that turns images into feature vectors."""

    def __init__(
        self,
        encoder: str = "siglip2-so400m-378",
        n_layers: int = 3,
        device: str | None = None,
        pretrained: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        if encoder not in ENCODER_CATALOG:
            raise ValueError(
                f"unknown encoder {encoder!r}; "
                f"choose from {sorted(ENCODER_CATALOG)}"
            )

        import timm

        entry = ENCODER_CATALOG[encoder]
        self.encoder_name = encoder
        self.device = resolve_device(device)

        # float16 on MPS/CPU is slow or unsupported for some ops; only use
        # reduced precision on CUDA where it is a clear win.
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        self.model = timm.create_model(
            entry["timm_name"], pretrained=pretrained, num_classes=0
        )
        self.model.eval().to(device=self.device, dtype=self.dtype)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        n_blocks = len(self.model.blocks)
        if not 1 <= n_layers <= n_blocks:
            raise ValueError(
                f"n_layers must be between 1 and {n_blocks}, got {n_layers}"
            )
        # The last `n_layers` blocks, shallowest first.
        self.layers = tuple(range(n_blocks - n_layers, n_blocks))

        data_config = timm.data.resolve_model_data_config(self.model)
        self._mean = torch.tensor(data_config["mean"]).view(1, 3, 1, 1)
        self._std = torch.tensor(data_config["std"]).view(1, 3, 1, 1)

        self.spec = FeatureSpec(
            encoder=encoder,
            timm_name=entry["timm_name"],
            feature_dim=entry["feature_dim"],
            image_size=entry["image_size"],
            layers=self.layers,
        )

    # -- introspection ------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def image_size(self) -> int:
        return self.spec.image_size

    # -- feature extraction -------------------------------------------------

    def _to_tensor(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Stack PIL images into a normalized NCHW batch.

        Images must already be at the model's input size -- resizing is the
        caller's decision (see `transforms.crop`), not something to do silently
        here, because the choice between cropping and resizing materially
        changes what the model can detect.
        """
        arrays = []
        for image in images:
            if image.size != (self.image_size, self.image_size):
                raise ValueError(
                    f"expected {self.image_size}x{self.image_size} inputs, "
                    f"got {image.size}; crop or resize before calling"
                )
            arrays.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)

        batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
        batch = (batch - self._mean) / self._std
        return batch.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def extract(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Feature vectors for a batch of images, shape (N, output_dim).

        Returns float32 regardless of compute dtype so downstream numerics
        (and the cache format) stay consistent across devices.
        """
        if not images:
            return np.zeros((0, self.spec.output_dim), dtype=np.float32)

        batch = self._to_tensor(images)

        pooled, intermediates = self.model.forward_intermediates(
            batch,
            indices=list(self.layers),
            return_prefix_tokens=False,
            output_fmt="NLC",
            intermediates_only=False,
        )

        parts: list[torch.Tensor] = []
        if self.spec.include_pooled:
            # `forward_intermediates` returns the final block's tokens, not the
            # attention-pooled vector, so pool explicitly.
            parts.append(pooled.mean(dim=1) if pooled.ndim == 3 else pooled)

        for tokens in intermediates:
            if isinstance(tokens, tuple):  # (patch_tokens, prefix_tokens)
                tokens = tokens[0]
            parts.append(tokens.mean(dim=1))

        features = torch.cat(parts, dim=1)
        return features.float().cpu().numpy()
