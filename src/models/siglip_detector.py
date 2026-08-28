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

from src.calibration import (
    LogitScaler,
    logits_to_probabilities,
    probabilities_to_logits,
)
from src.models.base import Detector
from src.transforms.crop import multi_crop_views

__all__ = [
    "load_siglip_detector",
    "FrozenProbeDetector",
    "LoraDetector",
    "CALIBRATION_NAME",
]

# Written into the checkpoint directory by scripts/calibrate.py, loaded here if
# present. Absent means "uncalibrated", which resolves to an identity scaler.
CALIBRATION_NAME = "calibration.json"


def _aggregate(view_scores: np.ndarray, scaler: LogitScaler) -> float:
    """Combine per-crop probabilities into one image-level probability.

    Averaged in LOGIT space, not probability space. Averaging probabilities
    pulls every result toward the middle -- five crops at 0.95 and one at 0.05
    average to 0.80, which understates a near-unanimous verdict -- and that
    compression is precisely what wrecks calibration. Logit averaging (the
    geometric mean of the odds) keeps confident agreement confident, and is
    also the form the calibration correction is defined against, so the two
    compose correctly instead of fighting each other.

    Calibration is applied AFTER aggregation: the scaler is fitted on final
    image-level scores, so it must see the same quantity at inference.
    """
    mean_logit = float(np.mean(probabilities_to_logits(view_scores)))
    return float(logits_to_probabilities(scaler.transform_logits(mean_logit)))


class FrozenProbeDetector:
    """The fast baseline: frozen SigLIP2 + scikit-learn logistic head."""

    name = "siglip2-frozen-probe"

    def __init__(
        self, encoder, head, n_crops: int = 4, scaler: LogitScaler | None = None
    ) -> None:
        self.encoder = encoder
        self.head = head
        self.n_crops = n_crops
        self.scaler = scaler or LogitScaler()

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
            scores.append(_aggregate(probs, self.scaler))
        return scores


class LoraDetector:
    """The competitive model: LoRA-tuned SigLIP2 + torch linear head."""

    name = "siglip2-lora"

    def __init__(
        self, encoder, head, n_crops: int = 4, scaler: LogitScaler | None = None
    ) -> None:
        import torch

        self.encoder = encoder
        self.head = head
        self.n_crops = n_crops
        self.scaler = scaler or LogitScaler()
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
                # The head emits logits directly, so average them as-is rather
                # than round-tripping through sigmoid and back.
                view_logits = self.head(tensor).cpu().numpy()
            mean_logit = float(np.mean(view_logits))
            scores.append(
                float(logits_to_probabilities(self.scaler.transform_logits(mean_logit)))
            )
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
    # Identity when the checkpoint carries no calibration, so both paths below
    # are written the same way regardless.
    scaler = LogitScaler.load_if_present(checkpoint_dir / CALIBRATION_NAME)

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

        return LoraDetector(encoder, head, n_crops=n_crops, scaler=scaler)

    from src.models.encoders import FrozenEncoder
    from src.training.head import LinearHead

    encoder = FrozenEncoder(
        encoder=meta["encoder"], n_layers=meta["n_layers"], device=device
    )
    head = LinearHead.load(checkpoint_dir / "head.npz")

    return FrozenProbeDetector(encoder, head, n_crops=n_crops, scaler=scaler)
