"""Wraps a trained checkpoint (frozen linear probe OR LoRA fine-tune) as a
`Detector` for `scripts/predict.py`.

`load_siglip_detector` autodetects which kind of checkpoint it was handed --
presence of `adapter_config.json` (written by peft's `save_pretrained`) means
LoRA; otherwise it is the frozen probe's `head.npz` -- so `predict.py` and the
demo do not need to know or care which model produced a given checkpoint
directory. That matters concretely: the frozen probe is the fast sanity
baseline, LoRA fine-tuning is the competitive model, and swapping between them
for the demo should be a `--checkpoint` flag, not a code change.

Both detectors use the same multi-crop inference policy: `n_crops` native crops
plus one whole-image resized view, scores averaged in probability space. A
single centre crop is a high-variance estimate of a whole image -- synthetic
artifacts are not spatially uniform, and the image centre is not privileged --
so averaging several views is materially more stable, and it is also the cheap
answer to arbitrary aspect ratios: a panorama gets sampled across its width
instead of judged on one square in the middle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.base import Detector
from src.transforms.crop import multi_crop_views

__all__ = ["load_siglip_detector", "FrozenProbeDetector", "LoraDetector"]


class FrozenProbeDetector:
    """The fast baseline: frozen SigLIP2 + scikit-learn logistic head."""

    name = "siglip2-frozen-probe"

    def __init__(self, encoder, head, n_crops: int = 4) -> None:
        self.encoder = encoder
        self.head = head
        self.n_crops = n_crops

    def predict_batch(self, images: list[Image.Image]) -> list[float]:
        scores = []
        for image in images:
            views = [
                v.image
                for v in multi_crop_views(
                    image, self.encoder.image_size, n_crops=self.n_crops
                )
            ]
            features = self.encoder.extract(views)
            probs = self.head.predict_proba(features)
            scores.append(float(np.mean(probs)))
        return scores


class LoraDetector:
    """The competitive model: LoRA-tuned SigLIP2 + torch linear head."""

    name = "siglip2-lora"

    def __init__(self, encoder, head, n_crops: int = 4) -> None:
        import torch

        self.encoder = encoder
        self.head = head
        self.n_crops = n_crops
        self._torch = torch

    def predict_batch(self, images: list[Image.Image]) -> list[float]:
        scores = []
        for image in images:
            views = [
                v.image
                for v in multi_crop_views(
                    image, self.encoder.image_size, n_crops=self.n_crops
                )
            ]
            features = self.encoder.extract(views)  # (V, dim) numpy, no grad
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(features).to(self.encoder.device)
                logits = self.head(tensor)
                probs = self._torch.sigmoid(logits).cpu().numpy()
            scores.append(float(np.mean(probs)))
        return scores


def load_siglip_detector(
    checkpoint_dir: Path | str, device: str | None = None, n_crops: int = 4
) -> Detector:
    """Load whichever checkpoint type lives at `checkpoint_dir`."""
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no meta.json in {checkpoint_dir} -- not a recognised checkpoint "
            f"(expected output of train_and_evaluate.py or finetune_lora.py)"
        )
    meta = json.loads(meta_path.read_text())

    if (checkpoint_dir / "adapter_config.json").exists():
        import torch

        from src.models.lora_encoder import LoraEncoder
        from src.models.torch_head import TorchLinearHead

        encoder = LoraEncoder(
            encoder=meta["encoder"],
            n_layers=meta["n_layers"],
            lora_rank=meta["lora_rank"],
            lora_alpha=meta["lora_alpha"],
            device=device,
        )
        encoder.load_adapter(checkpoint_dir)
        encoder.eval()

        head = TorchLinearHead(encoder.spec.output_dim)
        head.load_state_dict(
            torch.load(checkpoint_dir / "head.pt", map_location=encoder.device)
        )
        head.to(encoder.device).eval()

        return LoraDetector(encoder, head, n_crops=n_crops)

    from src.models.encoders import FrozenEncoder
    from src.training.head import LinearHead

    encoder = FrozenEncoder(
        encoder=meta["encoder"], n_layers=meta["n_layers"], device=device
    )
    head = LinearHead.load(checkpoint_dir / "head.npz")

    return FrozenProbeDetector(encoder, head, n_crops=n_crops)
