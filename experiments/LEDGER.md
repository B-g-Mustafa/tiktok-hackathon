# Experiment Ledger

Every experiment gets a hypothesis before it gets a result. Negative results are
kept — several of them are deliverables in their own right.

---

## EXP-000 — Resolution-only canary on CommunityForensics-Small

**Hypothesis:** the two classes can be separated by image dimensions alone, with
no image content whatsoever.

**Method:** `scripts/run_canary.py`. Pulled 300 metadata rows per class from the
HF datasets-server (no image bytes). Scored each image by pixel count
(width × height) — no training, no fitting. Measured AUROC against the label.

**Result: CONFIRMED, maximally.**

| | |
|---|---|
| Authentic (label=0) | **1024×1024**, 100% PNG, source `FFHQ` |
| Generated (label=1) | **512×512**, 100% PNG, SD-family LoRAs |
| Canary AUROC | **0.0000 (effective 1.0000)** |

A classifier reading nothing but `image.size` separates this dataset perfectly.

**Decision:** naive training on CF-Small is invalid. Whole-image input is banned.
Resolution **matching** is required — note that cropping alone is *not* sufficient
(see EXP-001 reasoning). Re-run this canary against the curated manifest and
require it to fall to ≈0.5 before training on anything.

**Artifact:** `experiments/canary_communityforensics.json`

---

## EXP-001 — Design review corrections (no compute)

A design pressure-test overturned four assumptions in the original plan. Recording
them because each one would have silently corrupted results.

**1. The planned RISK-A mitigation was harmful.**
The plan said "re-encode every image through a common JPEG quality distribution."
Re-encoding a pristine generated image yields *single*-compressed pixels; re-encoding
an already-JPEG real (COCO/VISION) yields *double*-compressed pixels. Double-JPEG
leaves periodic gaps in DCT coefficient histograms and is trivially learnable — so
the "fix" would have replaced a visible container shortcut with a stronger, less
visible compression-history shortcut.
→ Apply a JPEG round-trip to **both** classes with quality randomized *independently
of label*, then verify with a **reals-only format probe** (score only real images,
half saved PNG and half JPEG; if the PNG half reads as "more fake", the shortcut is
quantified with zero label confound).

**2. Cropping does not neutralize the resolution shortcut.**
A 512² crop from a 1024² real is a quarter-area window at 2× effective pixel density;
a 512² crop from a 512² fake is the entire image. Cropping converts an obvious
shortcut into a subtle scale/PPI mismatch that shows up in no metric.
→ Explicit resolution matching before augmentation, with the **same resampling kernel
applied to both classes** whenever either is resized.

**3. Parquet shards are ordered by source, not shuffled.**
Shard 0 of CF-Small indexes 8,979 generated / 1,563 authentic (85/15), with *all*
reals FFHQ and *all* fakes `LatDiff`. Taking "the first K shards" yields a
class-imbalanced, single-architecture, single-real-source training set that looks fine.
→ Metadata-only columnar scan of **all** shards to build a manifest first
(`scripts/build_manifest.py`), then sample from the manifest.

**4. A content/semantic shortcut exists that mimics genuine robustness.**
CF-Small generators in the sampled region are anime/character/pet LoRAs
(`alea31415/onimai-characters`, `doohickey/neopian-diffusion`) while the reals are
100% FFHQ faces. A semantic encoder can learn "anime → fake, photographic face → real",
and that signal survives JPEG-30 beautifully — producing a flat robustness curve for
entirely the wrong reason.
→ **Control D (content-matched subset)** is the only experiment that can falsify the
branch-1 hypothesis. Not optional.

**Also corrected (details in EXP-002/003):**
- **Branch 2 is deprioritized.** 12 of the 14 organizer transforms specifically destroy
  high-frequency residuals; branch 2 can only contribute on the clean column, which
  EXP-000's motivating evidence says is uninformative. Test it with hand-crafted
  residual statistics + logistic regression (~1h, no GPU) rather than building a CNN.
- **Tiny-GenImage has a format shortcut** (reals 100% jpg; fakes ~14% png) → **eval-only,
  never train.**
- **SigLIP2-so400m vision tower = 428,225,600 params** (verified). The *full* checkpoint
  is 1,136,008,498 — the text tower is 707.7M of dead weight against the 2B budget.
  Load `SiglipVisionModel`, never `AutoModel`.
- **PE-Core-G14-448 and DINOv3 are both cut** (budget blowout / gated `manual` approval).

---

## EXP-002 — Full-manifest audit (556,541 rows, metadata only)

**Hypothesis:** the EXP-000 shortcut is an artifact of the small datasets-server
sample and will weaken across the full dataset.

**Method:** `scripts/build_manifest.py` scanned all 186 parquet shards, reading
only metadata columns over HTTP range requests (never `image_data`). Then
`scripts/analyze_manifest.py`.

**Result: REFUTED — the shortcut holds at full scale, and a second one appeared.**

| Finding | Value |
|---|---|
| Rows | 556,541 (50.0% authentic / 50.0% generated) |
| Resolution canary, full dataset | **0.9006** |
| Format shortcut | PNG is 55.0% of authentic vs **97.8%** of generated (42.8% gap) |
| Distinct generators | 4,782 (median 41 images each) |
| Real sources | COCO 118,287 · LandscapesHQ 90,000 · FFHQ 63,000 · VISION 6,809 |

Two things the audit settled that guesswork would have got wrong:

1. **The first 3 shards contain zero authentic images.** Shards are ordered by
   source, so "download the first K shards to get started" yields a training set
   with no negative class at all. The manifest makes this visible in seconds.
2. **CF-Small's COCO reals are exactly 118,287 rows with 118,287 unique IDs —
   the precise size of COCO *train2017*.** COCO val2017 (5,000 images, disjoint
   IDs) is therefore not present, so the organizers' demo benchmark is not
   contaminated by it. Excluding COCO wholesale was over-conservative, but we
   keep the exclusion anyway for a different reason (see EXP-003).

---

## EXP-003 — Neutralizing the size shortcut

**Hypothesis:** filtering contaminated sources plus fixed-size cropping is enough
to remove the size signal.

**Result: REFUTED, then fixed.** Dropping COCO made the shortcut *worse* — COCO
was the main source of varied-resolution reals, and removing it left FFHQ and
LandscapesHQ (both large) against 512px generations.

| Pipeline stage | n | resolution canary | scale canary |
|---|---:|---:|---:|
| raw manifest | 556,541 | 0.9006 | 0.5121 |
| after dropping COCO/RAISE reals | 438,254 | 0.9765 | **0.7451** |
| after `min_side >= 256` | 438,254 | 0.9765 | **0.7451** |
| after **min_side matching** | 195,300 | 0.9203 | **0.5000** |
| after stratified balancing | 60,000 | 0.9204 | **0.5000** |
| **TRAIN split** | 47,567 | 0.9382 | **0.5098** |
| CROSS-GENERATOR split | 12,433 | 0.8552 | 0.5358 |
| resolution-matched control | 17,310 | 0.5000 | 0.5000 |

**The key insight — two different canaries for two different pipelines.**
Total pixel count and aspect ratio are only observable if whole images are fed to
the model. Once we take a fixed NxN *native* crop, a 256² window looks identical
whether it came from a 512×512 or a 512×768 source. The one size cue that
survives cropping is **scale** (`min(width, height)`), because it sets how much
of the scene the window covers. So `min_side` is the right invariant to enforce,
and the resolution canary staying high on the train split is expected and
harmless — the model never sees whole images.

Matching on `min_side` rather than exact resolution matters a lot in practice:
exact matching collapses the pool to 17,310 images and **31 generators**, while
min_side matching keeps 195,300 images and **1,714 generators** — and generator
diversity is the property most strongly linked to unseen-generator performance.

**Second-order effect worth recording:** naive class balancing after matching
pushed the scale canary back from 0.5000 to 0.5382, because round-robin
generator sampling ignored the buckets. Stratifying the balance by `min_side`
restored it to 0.5000. This is covered by a regression test.

**Decision:** train split accepted (scale canary 0.5098 ≈ chance). Splits written
to `artifacts/splits/`, generator overlap between train and cross-generator = 0.

**Open risk — content bias is NOT solved.** The surviving reals are LandscapesHQ
(landscapes) and FFHQ (faces); the generated images are community LoRAs (anime,
pets, characters). A semantic encoder can learn "anime → generated" and that
signal survives JPEG-30 beautifully, producing a flat robustness curve for
entirely the wrong reason. **Control D (content-matched evaluation) is required
before any robustness claim is trustworthy.**

---

## EXP-004 — Backbone selection and parameter budget

**Method:** instantiated candidates via `timm` with `pretrained=False` and counted
parameters directly.

| Backbone (vision tower only) | Params | Feature dim | Licence |
|---|---:|---:|---|
| `vit_so400m_patch14_siglip_378.v2_webli` | **428,225,600** | 1152 | Apache-2.0 |
| `vit_large_patch16_siglip_384.v2_webli` | 316,283,904 | 1024 | Apache-2.0 |
| `vit_base_patch16_siglip_384.v2_webli` | 93,176,064 | 768 | Apache-2.0 |

Planned two-encoder ensemble: **744.5M / 2B (37%)** — ample headroom.

**Notes:**
- SigLIP2 so400m ships at 224 and 378 (not 384); `..._384.webli` is SigLIP**1**.
- SigLIP ViTs have **no CLS token** (`num_prefix_tokens=0`, attention/MAP pooling),
  so the conventional "CLS + patch mean" recipe does not apply. We use the pooled
  output plus mean-pooled patch tokens from the last 3 blocks.
- Loading the full so400m checkpoint (1,136,008,498 params) instead of the vision
  tower would spend 35% of the entire budget on a text tower that never runs.
  Enforced by a test.

---

## EXP-005 — Adding a LoRA fine-tuning tier (course correction)

**Context:** the frozen linear probe (Tier 1) was presented as the primary
model. That was a mistake in framing, not in engineering — a 4,609-parameter
head, however well-validated the pipeline underneath it, is not what wins a
detection challenge. NTIRE 2026's top teams all fine-tuned. The frozen probe's
actual purpose was always to validate the data pipeline cheaply before
committing GPU-hours to something bigger; it should have been presented as
step 1 of 2 from the start.

**Decision:** added a second tier — `LoraEncoder` (peft, rank 8, targeting
`qkv`/`proj`/`fc1`/`fc2` on every block) fine-tuned end-to-end with a torch
linear head via `scripts/finetune_lora.py`. LoRA rather than full fine-tuning,
because full fine-tuning is exactly what makes detectors memorize the artifacts
of the generators they trained on — the failure mode an unseen-generator hidden
test set is designed to punish. LoRA constrains the update to a low-rank
subspace, closer in spirit to the frozen probe than to training from scratch.

**Engineering notes:**
- Fine-tuning needs gradients through the encoder, so cached features don't
  apply. Split the workflow: `materialize_images.py` downloads each selected
  image ONCE to local PNGs (reusing the tested `iter_selected_images`), then
  `finetune_lora.py` trains purely against local disk via a standard
  multi-worker `DataLoader` — avoids re-downloading tens of GB per epoch.
- Verified on CPU with `timm.create_model(..., pretrained=False)`: `peft`'s
  `LoraConfig(target_modules=["qkv","proj","fc1","fc2"])` matches all 12 SigLIP2
  blocks (r=8 → ~1.27M trainable / 1.34% of the base-384 tower), composes
  cleanly with `forward_intermediates` (the multi-layer feature recipe), and
  gradients flow correctly (`lora_B` receives a real gradient immediately;
  `lora_A`'s gradient is exactly zero for the first step only — expected LoRA
  zero-init behavior, not a bug, confirmed by checking `lora_B` directly).
- **Real bug found and fixed:** the first `load_adapter()` implementation called
  `self.peft_model.get_base_model()` to "unwrap" before re-attaching a saved
  adapter. `peft`'s `get_peft_model` injects LoRA layers by mutating the base
  model's submodules *in place* — there is no pristine copy to unwrap back to.
  The old code silently stacked a second adapter on top of the first (`peft`
  warns: "Already found a peft_config attribute... multiple adapters"), and
  every prediction from a loaded checkpoint was wrong. Fixed by rebuilding from
  a fresh `timm.create_model(...)` inside `load_adapter` and attaching the saved
  adapter to that, which is the standard peft pattern for loading a
  previously-trained adapter. Verified with `pretrained=True` (deterministic
  base weights): feature roundtrip after save→reload matches the pre-save
  features exactly (max abs diff 0.0), and a checkpoint loaded from a
  *different* random base-weight seed does NOT match — confirming the test
  actually discriminates rather than passing vacuously.
- Added `--adapter` to `cache_features.py` so a LoRA-fine-tuned checkpoint can
  produce eval-mode features through the exact same code path as the frozen
  probe, meaning `train_and_evaluate.py`'s robustness-matrix logic needs zero
  changes to score either tier.
- `predict.py` autodetects checkpoint type (presence of `adapter_config.json`)
  so switching tiers for a demo is a `--checkpoint` flag, not a code change.

**Open:** not yet run against the real ~24K-image train split (needs the GPU
box). The CPU tests validate mechanism correctness (gradients, save/load,
parameter accounting) on tiny synthetic batches, not detection accuracy.

---

## EXP-006 — Generic local-directory data path (GenImage support)

**Context:** the pipeline through EXP-005 only knew how to read from the remote
Community Forensics HF dataset. The user already has GenImage downloaded
locally (via `scripts/down-genimage-dataset.py`) and asked to run both training
phases against it directly.

**Verified GenImage's real structure** (was assumed, not previously checked):
each generator ships both classes under `train/` and `val/` — `ai/` (generated)
and `nature/` (real, sourced from ImageNet). Confirmed against the GenImage
authors' own repo/README, not just inferred from the CATEGORIES list already in
`down-genimage-dataset.py`.

**Built:** `src/data/local_manifest.py` (`build_local_manifest`) scans an
arbitrary real/fake directory tree — GenImage's `ai`/`nature` convention or
generic `real`/`fake`/`authentic`/`synthetic` names, case-insensitive — and
writes the exact same `manifest.parquet` schema the Community-Forensics-specific
`materialize()` already produces. That schema match is what let every
downstream script (`LocalImageDataset`, `finetune_lora.py`) work unchanged;
only `cache_features.py` needed a new `--local-manifest` flag (swaps the image
source, extraction loop untouched) since it previously always fetched from the
remote HF repo.

**Bug caught by testing end-to-end, not by inspection:** the `--split` filter
promised in an early docstring wasn't actually implemented — the CLI had no
`--split` argument at all. Caught by literally running the documented command
against a fake GenImage-shaped tree (2 generators × train/val × ai/nature) and
finding train+val silently merged into one manifest. This matters specifically
for GenImage, which keeps both splits side-by-side under every generator, so
"point `--root` at the dataset" without a filter is a real footgun, not a
theoretical one. Fixed by adding a real `split_filter` parameter and verifying
it actually isolates train from val with zero overlap.

**Second bug caught the same way:** documented an `--eval-cache` glob
(`*siglip2-lora*`) for comparing frozen-probe vs. LoRA eval caches of the same
split, then ran it and found it matched nothing — the cache filename uses the
base encoder name (`siglip2-so400m-378`) regardless of whether a LoRA adapter
produced it; only the config hash differs. Fixed by tagging LoRA-extracted
filenames with `-lora` explicitly in `cache_features.py`, verified both
filenames land in the same directory now visibly distinct
(`naming_check__siglip2-base-384__<hash>__eval.npz` vs.
`naming_check__siglip2-base-384-lora__<hash>__eval.npz`).

**Full pipeline verified end-to-end on a synthetic GenImage-shaped tree**
(2 generators × train/val × ai/nature, 48 images): `prepare_local_dataset.py`
→ correct 12/12 authentic/generated split with `--split train` and zero
leakage into the val manifest → `cache_features.py --local-manifest` (phase 1)
→ `finetune_lora.py --images-dir ... --train-split ...` (phase 2, trains and
saves a checkpoint) → `predict.py --model siglip2 --checkpoint ...` loads it
and produces valid predictions on unseen images from the same tree.

**Deliverable:** `HOWTO.md` — practical, command-by-command guide answering
exactly what was asked (inference, GenImage's real/fake structure, directory
arrangement, phase 1 alone, phase 2 alone), linked from the top of the main
README. Every command in it was run against real (synthetic) data before being
written down, which is how the two bugs above were caught.
