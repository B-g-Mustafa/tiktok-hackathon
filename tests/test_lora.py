"""Tests for the LoRA fine-tuning path.

These run entirely on CPU with `pretrained=True` (real, small checkpoints
already cached locally from earlier runs) so they exercise the actual
production code path -- deliberately not `pretrained=False`, which was found
during development to make base-model weights RNG-dependent in a way that
defeats save/load comparisons for reasons that have nothing to do with
`load_adapter`'s correctness (two separate random constructions can't stay
seed-aligned once other code runs between them).

The central regression this file guards: an earlier version of `load_adapter`
called `self.peft_model.get_base_model()` to "unwrap" back to a plain model
before re-wrapping with the saved adapter. peft's `get_peft_model` mutates a
model's submodules in place when injecting LoRA layers, so there is no
pristine copy to unwrap back to -- that version silently stacked a second
adapter on top of the first (peft warns, but proceeds), and every prediction
from a loaded checkpoint was wrong.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.lora_encoder import LoraEncoder
from src.models.torch_head import TorchLinearHead

ENCODER = "siglip2-base-384"

# Only the save/load roundtrip tests need pretrained=True (real weights, not
# random init) -- see their docstrings for why. Everything else in this file
# runs offline with pretrained=False.
needs_pretrained_weights = pytest.mark.slow


def make_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, (384, 384, 3), dtype=np.uint8), mode="RGB"
    )


# ---------------------------------------------------------------------------
# Construction and parameter accounting
# ---------------------------------------------------------------------------


def test_lora_targets_match_zero_parameters_would_be_a_silent_bug():
    """A target_modules typo must not silently train an accidentally-frozen
    model. peft itself refuses to match zero modules (raising ValueError from
    inside get_peft_model); our own belt-and-suspenders check in __init__
    (n_trainable == 0 -> RuntimeError) exists as a second line of defence in
    case peft's behaviour ever changes to warn instead of raise."""
    with pytest.raises(ValueError, match="not found in the base model"):
        LoraEncoder(
            ENCODER, n_layers=1, device="cpu", pretrained=False,
            target_modules=("this_matches_nothing",),
        )


def test_trainable_parameters_are_a_small_fraction_of_the_tower():
    encoder = LoraEncoder(ENCODER, n_layers=1, lora_rank=8, device="cpu", pretrained=False)
    assert 0 < encoder.n_trainable_parameters < encoder.n_total_parameters
    assert encoder.n_trainable_parameters / encoder.n_total_parameters < 0.05


def test_higher_rank_means_more_trainable_parameters():
    low = LoraEncoder(ENCODER, n_layers=1, lora_rank=4, device="cpu", pretrained=False)
    high = LoraEncoder(ENCODER, n_layers=1, lora_rank=16, device="cpu", pretrained=False)
    assert high.n_trainable_parameters > low.n_trainable_parameters


def test_config_hash_encodes_lora_rank():
    """Features from rank-4 and rank-8 adapters are not interchangeable --
    the cache-mismatch guard depends on this."""
    a = LoraEncoder(ENCODER, n_layers=1, lora_rank=4, device="cpu", pretrained=False)
    b = LoraEncoder(ENCODER, n_layers=1, lora_rank=8, device="cpu", pretrained=False)
    assert a.spec.config_hash() != b.spec.config_hash()


def test_rejects_invalid_n_layers():
    with pytest.raises(ValueError):
        LoraEncoder(ENCODER, n_layers=0, device="cpu", pretrained=False)
    with pytest.raises(ValueError):
        LoraEncoder(ENCODER, n_layers=999, device="cpu", pretrained=False)


def test_unknown_encoder_rejected():
    with pytest.raises(ValueError):
        LoraEncoder("not-a-real-encoder", device="cpu")


# ---------------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------------


def test_forward_features_shape_and_grad():
    encoder = LoraEncoder(ENCODER, n_layers=2, lora_rank=4, device="cpu", pretrained=False)
    features = encoder.forward_features([make_image(), make_image(1)])
    assert features.shape == (2, encoder.spec.output_dim)
    assert features.requires_grad


def test_gradient_reaches_lora_parameters():
    """B is zero-initialized by LoRA convention, so dLoss/dA is exactly zero
    for one step (the standard LoRA warm-start) -- but dLoss/dB must be
    nonzero immediately, proving the path from loss to adapter is intact."""
    encoder = LoraEncoder(ENCODER, n_layers=1, lora_rank=4, device="cpu", pretrained=False)
    features = encoder.forward_features([make_image()])
    features.sum().backward()

    b_grads = [
        p.grad for n, p in encoder.peft_model.named_parameters()
        if "lora_B" in n and p.grad is not None
    ]
    assert any(g.abs().sum() > 0 for g in b_grads)


def test_frozen_parameters_never_receive_gradients():
    encoder = LoraEncoder(ENCODER, n_layers=1, lora_rank=4, device="cpu", pretrained=False)
    encoder.forward_features([make_image()]).sum().backward()
    for p in encoder.peft_model.parameters():
        if not p.requires_grad:
            assert p.grad is None


def test_wrong_input_size_is_rejected():
    encoder = LoraEncoder(ENCODER, n_layers=1, device="cpu", pretrained=False)
    with pytest.raises(ValueError, match="expected"):
        encoder.forward_features([Image.new("RGB", (100, 100))])


def test_extract_matches_forward_features_numerically():
    """extract() must be a pure no-grad view of forward_features, not a
    different code path that could silently drift."""
    encoder = LoraEncoder(ENCODER, n_layers=1, lora_rank=4, device="cpu", pretrained=False)
    encoder.eval()
    image = make_image()

    with torch.no_grad():
        via_forward = encoder.forward_features([image]).numpy()
    via_extract = encoder.extract([image])
    assert np.allclose(via_forward, via_extract, atol=1e-5)


def test_extract_restores_training_mode():
    encoder = LoraEncoder(ENCODER, n_layers=1, lora_rank=4, device="cpu", pretrained=False)
    encoder.train()
    encoder.extract([make_image()])
    assert encoder.peft_model.training


def test_extract_empty_batch():
    encoder = LoraEncoder(ENCODER, n_layers=1, device="cpu", pretrained=False)
    out = encoder.extract([])
    assert out.shape == (0, encoder.spec.output_dim)


# ---------------------------------------------------------------------------
# Save / load -- the regression test for the double-wrapping bug
# ---------------------------------------------------------------------------


@needs_pretrained_weights
def test_save_and_load_adapter_reproduces_features_exactly(tmp_path):
    """The central regression test. Requires `pretrained=True` (real,
    deterministic base weights) -- see module docstring for why."""
    encoder = LoraEncoder(ENCODER, n_layers=1, lora_rank=4, device="cpu", pretrained=True)

    # Move the LoRA weights away from their zero-initialized start so a
    # roundtrip test is actually meaningful.
    optimizer = torch.optim.SGD(encoder.trainable_parameters(), lr=0.1)
    image = make_image()
    for _ in range(2):
        loss = encoder.forward_features([image]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    before = encoder.extract([image])
    encoder.save_adapter(tmp_path)

    reloaded = LoraEncoder(ENCODER, n_layers=1, lora_rank=4, device="cpu", pretrained=True)
    reloaded.load_adapter(tmp_path)
    reloaded.eval()
    after = reloaded.extract([image])

    assert np.allclose(before, after, atol=1e-4)


@needs_pretrained_weights
def test_loaded_adapter_differs_from_a_fresh_random_one(tmp_path):
    """Guards against a vacuous pass: if load_adapter silently did nothing
    (kept the fresh random init instead of loading), the roundtrip test above
    could pass by coincidence on some inputs. This confirms training actually
    moved the weights somewhere a fresh init wouldn't be."""
    encoder = LoraEncoder(ENCODER, n_layers=1, lora_rank=4, device="cpu", pretrained=True)
    image = make_image()

    optimizer = torch.optim.SGD(encoder.trainable_parameters(), lr=1.0)
    for _ in range(3):
        loss = encoder.forward_features([image]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trained_output = encoder.extract([image])
    encoder.save_adapter(tmp_path)

    fresh = LoraEncoder(ENCODER, n_layers=1, lora_rank=4, device="cpu", pretrained=True)
    fresh_output = fresh.extract([image])

    assert not np.allclose(trained_output, fresh_output, atol=1e-3)


# ---------------------------------------------------------------------------
# TorchLinearHead
# ---------------------------------------------------------------------------


def test_torch_head_output_shape():
    head = TorchLinearHead(in_features=16)
    logits = head(torch.randn(5, 16))
    assert logits.shape == (5,)


def test_torch_head_normalizes_away_magnitude():
    head = TorchLinearHead(in_features=8)
    features = torch.randn(4, 8)
    a = head(features)
    b = head(features * 100.0)
    assert torch.allclose(a, b, atol=1e-4)


def test_torch_head_handles_zero_vector():
    head = TorchLinearHead(in_features=8)
    logits = head(torch.zeros(1, 8))
    assert torch.isfinite(logits).all()


def test_torch_head_parameter_count():
    head = TorchLinearHead(in_features=100)
    assert head.n_parameters == 101  # 100 weights + 1 bias
