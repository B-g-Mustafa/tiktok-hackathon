"""LoRA-adapted vision encoder -- the step up from the frozen linear probe.

Why this exists, and why it's a SEPARATE model rather than a replacement
-------------------------------------------------------------------------

The frozen linear probe (`FrozenEncoder` + `LinearHead`) answers one question
fast: does the pretrained representation already separate real from generated
on our scale/content-matched data, with no risk of the training itself
introducing a shortcut? It trains in seconds because only ~4.6K parameters ever
move. That makes it the right FIRST checkpoint -- a cheap, low-risk way to
validate the data pipeline before spending GPU-hours on anything bigger.

It is not, by itself, a competitive submission. NTIRE 2026's top teams all
fine-tuned. LoRA is the lever for closing that gap while staying inside two
constraints the frozen probe respects for free and fine-tuning has to be
deliberate about:

  * **<2B parameters.** LoRA adds rank-r update matrices next to the frozen
    weights rather than replacing them, so the trainable footprint stays tiny
    (r=8 on the so400m tower is ~1.3% of it) while the frozen backbone still
    counts toward -- and stays comfortably under -- the budget.
  * **Unseen-generator generalization.** Full fine-tuning is what makes
    detectors memorize the artifacts of the generators they trained on, which
    is exactly the failure mode the hidden test set is designed to punish.
    LoRA constrains the update to a low-rank subspace, which is a much smaller
    perturbation of the pretrained representation than unrestricted
    fine-tuning -- closer in spirit to the frozen probe than to training from
    scratch, while still letting the model adapt.

`target_modules` covers every attention and MLP projection in the vision
transformer (`qkv`, `proj`, `fc1`, `fc2`), which is where a ViT's task-specific
adaptation capacity lives; leaving any of them out would silently starve the
model of a whole class of update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from src.models.encoders import ENCODER_CATALOG, FeatureSpec, resolve_device

__all__ = ["LoraEncoder", "DEFAULT_LORA_TARGETS"]

# Every attention/MLP projection in a timm ViT block. Omitting any one of these
# would silently cap what LoRA can adapt -- e.g. targeting only `qkv` leaves the
# MLP (roughly half the block's capacity) entirely frozen.
DEFAULT_LORA_TARGETS = ("qkv", "proj", "fc1", "fc2")


@dataclass(frozen=True)
class LoraFeatureSpec(FeatureSpec):
    """A FeatureSpec that also records the LoRA configuration.

    Cached features from a LoRA-tuned encoder are NOT interchangeable with
    frozen-encoder features or with features from a different LoRA rank --
    the representation itself has changed. `config_hash` folds in `lora_rank`
    for exactly this reason, so a stale cache is refused rather than silently
    mixed in.
    """

    lora_rank: int = 0
    lora_alpha: int = 0

    def config_hash(self) -> str:
        import hashlib

        base = super().config_hash()
        payload = f"{base}|lora_r{self.lora_rank}|lora_a{self.lora_alpha}"
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class LoraEncoder:
    """A vision tower with trainable LoRA adapters on every attention/MLP
    projection, plus a trainable classification head.

    Unlike `FrozenEncoder`, this model has gradients and is meant to be
    trained end-to-end with `scripts/finetune_lora.py` -- it is not a feature
    cache producer.
    """

    def __init__(
        self,
        encoder: str = "siglip2-so400m-378",
        n_layers: int = 3,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS,
        device: str | None = None,
        pretrained: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        if encoder not in ENCODER_CATALOG:
            raise ValueError(
                f"unknown encoder {encoder!r}; choose from {sorted(ENCODER_CATALOG)}"
            )

        import timm
        from peft import LoraConfig, get_peft_model

        entry = ENCODER_CATALOG[encoder]
        self.encoder_name = encoder
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float32

        self._pretrained = pretrained
        base_model = timm.create_model(
            entry["timm_name"], pretrained=pretrained, num_classes=0
        )

        n_blocks = len(base_model.blocks)
        if not 1 <= n_layers <= n_blocks:
            raise ValueError(
                f"n_layers must be between 1 and {n_blocks}, got {n_layers}"
            )
        self.layers = tuple(range(n_blocks - n_layers, n_blocks))

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
        )
        self.peft_model = get_peft_model(base_model, lora_config)
        self.peft_model.to(device=self.device, dtype=self.dtype)

        # `get_peft_model` freezes everything not matched by target_modules,
        # which is exactly what we want -- but assert it, since a silent
        # target_modules typo (e.g. "attn.qkv" instead of "qkv") would leave
        # peft matching nothing and train an accidentally-frozen model.
        n_trainable = sum(
            p.numel() for p in self.peft_model.parameters() if p.requires_grad
        )
        if n_trainable == 0:
            raise RuntimeError(
                "LoRA injection matched zero parameters -- check target_modules "
                f"{target_modules} against the model's module names"
            )

        data_config = timm.data.resolve_model_data_config(base_model)
        self._mean = torch.tensor(data_config["mean"]).view(1, 3, 1, 1)
        self._std = torch.tensor(data_config["std"]).view(1, 3, 1, 1)

        self.spec = LoraFeatureSpec(
            encoder=encoder,
            timm_name=entry["timm_name"],
            feature_dim=entry["feature_dim"],
            image_size=entry["image_size"],
            layers=self.layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    # -- introspection ------------------------------------------------------

    @property
    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)

    @property
    def n_total_parameters(self) -> int:
        return sum(p.numel() for p in self.peft_model.parameters())

    @property
    def n_parameters(self) -> int:
        """Alias matching `FrozenEncoder.n_parameters` so this encoder is a
        drop-in for the feature-caching script's logging/introspection code."""
        return self.n_total_parameters

    @property
    def image_size(self) -> int:
        return self.spec.image_size

    def train(self) -> "LoraEncoder":
        self.peft_model.train()
        return self

    def eval(self) -> "LoraEncoder":
        self.peft_model.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.peft_model.parameters() if p.requires_grad)

    # -- forward --------------------------------------------------------

    def _to_tensor(self, images) -> torch.Tensor:
        import numpy as np

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

    def forward_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Feature vectors WITH gradients, for training.

        Mirrors `FrozenEncoder.extract`'s recipe (pooled output + mean-pooled
        patch tokens from the last N blocks) so a LoRA-tuned checkpoint and a
        frozen checkpoint are directly comparable in the robustness matrix --
        the only thing that should differ is whether gradients flowed.
        """
        batch = self._to_tensor(images)

        base = self.peft_model.base_model.model
        pooled, intermediates = base.forward_intermediates(
            batch,
            indices=list(self.layers),
            return_prefix_tokens=False,
            output_fmt="NLC",
            intermediates_only=False,
        )

        parts = [pooled.mean(dim=1) if pooled.ndim == 3 else pooled]
        for tokens in intermediates:
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            parts.append(tokens.mean(dim=1))

        return torch.cat(parts, dim=1)

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]):
        """Gradient-free feature extraction, matching `FrozenEncoder.extract`'s
        interface exactly.

        This is what lets a fine-tuned checkpoint reuse the same eval-mode
        feature caching and robustness-matrix code the frozen probe uses:
        once training is done, the LoRA-tuned encoder is just another fixed
        feature extractor.
        """
        import numpy as np

        if not images:
            return np.zeros((0, self.spec.output_dim), dtype=np.float32)

        was_training = self.peft_model.training
        self.eval()
        try:
            return self.forward_features(images).float().cpu().numpy()
        finally:
            if was_training:
                self.train()

    def save_adapter(self, path) -> None:
        self.peft_model.save_pretrained(str(path))

    def load_adapter(self, path) -> "LoraEncoder":
        """Replace the current LoRA wrapper with one loaded from a checkpoint.

        peft's `get_peft_model` injects LoRA layers by mutating the base
        model's submodules IN PLACE, so `self.peft_model` never holds a
        pristine, unwrapped copy to fall back to -- attempting to "unwrap" it
        and re-wrap silently produces a second, stacked adapter instead of
        replacing the first (peft warns about this, but keeps going). The
        standard pattern for loading a previously-trained adapter is instead
        to build a completely fresh base model and attach the saved adapter to
        that, which is what this does; whatever LoRA wrapper `__init__` set up
        is discarded.
        """
        import timm
        from peft import PeftModel

        entry = ENCODER_CATALOG[self.encoder_name]
        fresh_base = timm.create_model(
            entry["timm_name"], pretrained=self._pretrained, num_classes=0
        )

        self.peft_model = PeftModel.from_pretrained(fresh_base, str(path))
        self.peft_model.to(device=self.device, dtype=self.dtype)
        return self
