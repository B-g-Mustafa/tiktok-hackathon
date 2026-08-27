# Robust AIGC Image Detection

Detecting AI-generated images in a way that **survives the post-processing real
images actually undergo** — JPEG re-encoding, blur, downscaling, noise, colour
shifts, cropping.

**Just want to run inference or fine-tune on your own data (e.g. GenImage)?**
Skip to [HOWTO.md](HOWTO.md) — it's the practical step-by-step. Everything
below is the design rationale and evidence behind those steps.

---

## The insight this project is built around

From the **NTIRE 2026 Challenge on Robust AI-Generated Image Detection**, the
closest published analogue to this problem:

| Team | Clean ROC AUC | Robust ROC AUC | Rank |
|---|---:|---:|---:|
| MICV (winner) | 0.9978 | **0.9723** | 1 |
| Shallow Real | 0.9954 | **0.8302** | 9 |

Shallow Real was *statistically tied with the winner on clean data* and finished
9th, because robustness collapsed by 0.165 AUC.

**Clean accuracy is nearly uninformative about deployment performance.** So this
project treats AIGC detection as *forensic evidence aggregation under
distribution shift*, not binary image classification. Worst-case AUROC is the
headline metric; clean accuracy is a diagnostic. The reporting code enforces
that ordering rather than leaving it to discipline.

---

## The finding that shaped everything

Before training anything, we audited the training data. On
`OwensLab/CommunityForensics-Small`:

> **A classifier that sees nothing but `image.size` separates the two classes
> perfectly.** Authentic images are 1024×1024 (FFHQ); generated images are
> 512×512. Across all 556,541 rows, image dimensions alone score **0.90 AUROC**.

Training naively on this data yields a ~99% model that has learned image
dimensions and will collapse on any hidden test set. The audit costs seconds and
downloads no image bytes:

```bash
python scripts/run_canary.py
```

Fixing it was not a one-liner. Two reasonable-looking attempts made it *worse*:

| Pipeline stage | n | scale canary |
|---|---:|---:|
| raw manifest | 556,541 | 0.5121 |
| after dropping COCO/RAISE reals | 438,254 | **0.7451** ← worse |
| after `min_side ≥ 256` | 438,254 | 0.7451 |
| after **min_side matching** | 195,300 | 0.5000 |
| after naive class balancing | 60,000 | **0.5382** ← regressed |
| after *stratified* balancing → **train split** | 23,847 | **0.5000** ✓ |

Dropping COCO backfired because COCO was the main source of varied-resolution
reals. The fix that worked came from distinguishing **two different canaries**:

- **Resolution** (`width × height`) is only observable if you feed *whole images*.
- **Scale** (`min(width, height)`) is the only size cue that survives a
  fixed-size native crop, because it sets how much of a scene the window covers.

Matching on `min_side` rather than exact resolution is what makes this practical:
exact matching leaves 17,310 images and **31 generators**; min_side matching
leaves 195,300 images and **1,714 generators** — and generator diversity is the
property most strongly linked to unseen-generator performance.

---

## Architecture: two tiers, deliberately

There are two checkpoints, not one, and they exist for different reasons.

**Tier 1 — frozen linear probe.** SigLIP2 vision tower, entirely frozen, plus a
4,609-parameter logistic regression head. Trains in seconds once features are
cached. This is a *validation* checkpoint, not the submission: its entire
purpose is to prove — cheaply, before committing GPU-hours — that the data
pipeline is sound and the representation genuinely separates real from
generated on our scale/content-matched splits, with no shortcut inflating the
number.

**Tier 2 — LoRA fine-tune.** The same backbone, now with trainable low-rank
adapters (`peft`, rank 8) on every attention/MLP projection, trained
end-to-end with the head. This is the competitive model. Linear probing alone
is not what wins detection challenges — NTIRE 2026's top teams all fine-tuned
— but *unrestricted* fine-tuning is what makes detectors memorize the specific
generators they trained on, which is exactly the failure mode an
unseen-generator hidden test set punishes. LoRA is the middle path: it adapts
a low-rank subspace of the pretrained weights rather than overwriting them, so
the update is a small, structured perturbation instead of a full rewrite.

```
Tier 1 (validation, seconds to train)
  image ──► native crop ──► FROZEN SigLIP2 tower ──► linear head ──► P(AIGC)
            + resized view    (428,225,600 params)     (4,609 params)

Tier 2 (competitive model, ~1h to fine-tune)
  image ──► native crop ──► SigLIP2 tower + LoRA adapters ──► linear head ──► P(AIGC)
            + resized view    (428.2M frozen + ~1-3M LoRA)     (4,609 params)
```

Three choices apply to both tiers, each with a reason:

**Crop, don't resize.** Downscaling is a low-pass filter and the pixel-level
traces of synthesis live in the frequencies it removes. Cropping also structurally
hides image dimensions, which is our defence against the shortcut above. A
whole-image resized view is carried alongside, because on a 4000×3000 photograph
a single 378px crop covers ~1% of the frame and discards the global structure.

**Frozen encoder, linear head.** Linear probes on frozen foundation features
generalize to unseen generators better than fine-tuned backbones, which memorize
the artifacts of the generators they saw. The hidden test set is explicitly
expected to contain unseen generators. Freezing also enables feature caching,
which turns each ablation from an hour into seconds.

**L2-normalized features.** Degradation attenuates activation *magnitude* far
more than it rotates direction. An unnormalized head partly learns "how strong is
the signal" — exactly the quantity JPEG and blur destroy.

### Parameter budget (limit: <2B)

| Component | Parameters | State |
|---|---:|---|
| SigLIP2-so400m/378 vision tower | 428,225,600 | frozen (both tiers) |
| LoRA adapters (rank 8, tier 2 only) | ~1-3M (exact count printed at train time) | trainable |
| Linear head | 4,609 | trainable |
| **Total (tier 2, worst case)** | **~431.5M** | **~21.6% of limit** |

Enforced by a test, not by a promise. Two traps it guards against:

- Loading `google/siglip2-so400m-patch14-384` via a generic `AutoModel` pulls in
  a **707.7M-parameter text tower** that never executes — 35% of the entire
  budget on dead weight. We load the vision tower via `timm`.
- **PE-Core-G14-448** is 1.88B in the vision tower alone (2.35B with text), and
  **DINOv3** is gated behind manual approval. Both rejected, with reasons
  recorded in `src/models/budget.py`.

---

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

Python 3.12 is pinned for ecosystem stability, not wheel availability.

---

## Reproducing the results

The pipeline is four steps. The first three run on metadata alone and cost
seconds; only step 4 needs a GPU.

```bash
# 1. Audit the dataset for label-leaking artifacts (no image bytes downloaded)
python scripts/run_canary.py

# 2. Scan all 186 parquet shards' metadata (~50 MB, not 763 GB)
python scripts/build_manifest.py

# 3. Build splits; FAILS the build if the split is still trivially separable
python scripts/build_splits.py
```

Step 3 prints a download plan and refuses to proceed on a broken split:

```
DOWNLOAD PLAN (each shard is one ~4.1 GB parquet row group)
  train   (min_side=512):  10 shards ~41.0 GB -> 29,920 images, 391 generators
  controlD(min_side=1024):  6 shards ~24.6 GB -> 10,344 images, 3 generators
  TOTAL to download: ~65.6 GB (16 shards of 186)

PASS: training split scale canary is 0.5000 (threshold 0.6).
```

Then, on the GPU box — **tier 1 first** (fast, validates the pipeline):

```bash
# 4a. Cache features (the only expensive step for tier 1)
python scripts/cache_features.py --split train --mode train --n-views 8
python scripts/cache_features.py --split cross_generator --mode eval
python scripts/cache_features.py --split content_matched_control --mode eval

# 4b. Train the head and produce the robustness matrix (seconds)
python scripts/train_and_evaluate.py \
  --train-cache    artifacts/features/train__*.npz \
  --eval-cache     artifacts/features/cross_generator__*.npz \
  --control-cache  artifacts/features/content_matched_control__*.npz \
  --save-checkpoint artifacts/checkpoints/frozen
```

**Tier 2 — LoRA fine-tune (the competitive model)**, once tier 1 confirms the
pipeline is sound:

```bash
# 5a. Materialize images to local disk once (pure network cost, ~1 pass)
python scripts/materialize_images.py --split train
python scripts/materialize_images.py --split cross_generator
python scripts/materialize_images.py --split content_matched_control

# 5b. Fine-tune (pure local I/O + GPU compute from here on)
python scripts/finetune_lora.py --epochs 3 --lora-rank 8 \
  --output artifacts/checkpoints/lora

# 5c. Extract eval-mode features with the FINE-TUNED encoder, then reuse the
#     exact same robustness-matrix code path as tier 1
python scripts/cache_features.py --split cross_generator --mode eval \
  --adapter artifacts/checkpoints/lora
python scripts/cache_features.py --split content_matched_control --mode eval \
  --adapter artifacts/checkpoints/lora
python scripts/train_and_evaluate.py \
  --train-cache    artifacts/features/train__*.npz \
  --eval-cache     artifacts/features/cross_generator__*siglip2-lora*.npz \
  --control-cache  artifacts/features/content_matched_control__*siglip2-lora*.npz
```

Both checkpoints load through the same inference entry point (`predict.py`
autodetects which kind of checkpoint a directory holds), so switching between
the fast baseline and the competitive model for a demo is a `--checkpoint`
flag, not a code change.

### Why the download is 65.6 GB and not 763 GB

Each Community Forensics shard is a **single ~4.1 GB parquet row group**, so
fetching one image downloads the entire shard. The dataset is also cleanly
partitioned — 93 shards hold only generated images, 92 hold only authentic ones.
A selection spread evenly across all 186 shards costs ~763 GB to materialise.

We instead select **whole shards**, chosen by *greedy generator coverage*: at
each step, take the shard adding the most generators we don't already have.
Ranking shards independently by generator count picks redundant shards holding
the same few generators — greedy coverage is the difference between 1 generator
and 4 for the same budget.

---

## Inference

The required deliverable:

```bash
# placeholder, no model needed -- proves the I/O contract independent of weights
python scripts/predict.py --image-dir /path/to/images --out preds.json

# tier 1 (frozen probe) or tier 2 (LoRA) -- predict.py autodetects which
python scripts/predict.py --image-dir /path/to/images \
  --model siglip2 --checkpoint artifacts/checkpoints/lora --out preds.json
```

```json
[
  {"image_path": "/abs/path/img1.jpg", "pred": 0.9371},
  {"image_path": "/abs/path/img2.png", "pred": 0.0832}
]
```

`pred` is P(AI-generated). Handles jpg/jpeg/png/webp/bmp/tiff at arbitrary
resolution and aspect ratio. Unreadable files never abort the run — they are
reported on stderr, or emitted with `pred: null` under `--include-failures`.

Tested against truncated JPEGs, empty files, CMYK scans, greyscale, palette
images, 800×20 panoramas, and transparent PNGs. (Transparency is composited onto
**white**, not black: the default `.convert("RGB")` paints transparent regions
black, creating a large flat artificial region that a forensic model would
happily learn from.)

---

## Verification

```bash
python -m pytest tests/ -q     # 186 tests
```

The suite is written around the failure modes that are *silent* rather than
loud — a broken split doesn't crash, it just reports an inflated number:

- **Shortcut canaries** — including a regression test for the balancing step
  that measurably pushed the scale canary from 0.500 back to 0.538.
- **Generator disjointness** — no generator may appear in both train and test.
- **Parameter budget** — fails the build above 2B.
- **Transform correctness** — JPEG distortion and blur must be *monotonic* in
  their parameters; a transform that silently no-ops would look fine otherwise.
- **Inference contract** — schema, determinism across batch sizes, corrupt-file
  handling.

---

## Limitations

**Content bias is not solved, and it is the main threat to these results.**
After scale matching, the surviving authentic images are landscapes (LandscapesHQ)
and faces (FFHQ), while the generated images are community LoRAs producing anime,
pets and characters. A semantic encoder can learn "anime ⇒ generated" — and that
rule survives JPEG-30 *beautifully*, producing a flat robustness curve for
entirely the wrong reason.

This is why **Control D** exists: a content-matched evaluation set pairing FFHQ
faces against FFHQ-trained face GANs (StyleGAN2-ADA, StyleGAN3, StyleSwin) at
identical resolution. Both size canaries read exactly 0.5000 on it. If the
detector collapses there, it was reading subject matter, not synthesis artifacts.
That result is reported either way.

**Other honest caveats:**

- **Dataset licensing.** Community Forensics-Small is `cc-by-nc-sa-4.0` —
  non-commercial research only, and `ShareAlike` arguably propagates to derived
  weights. Acceptable for a prototype; it would need revisiting for any
  productionisation. `SID_Set` (CC-BY-4.0) is the permissive fallback.
- **Recent commercial generators are the field-wide weak point.** A 2026
  benchmark of 16 detectors found leading methods score only 18–30% on Flux Dev,
  Firefly v4 and Midjourney v7.
- **Heavy degradation genuinely destroys the evidence.** At 0.25× downscaling and
  JPEG-30 there may be little forensic signal left to read. We report where
  confidence collapses rather than hiding it.
- **Tiny-GenImage is eval-only.** Its authentic images are 100% JPEG while ~14%
  of its generated images are PNG — a container shortcut that makes it unsafe to
  train on.

---

## Repository layout

```
src/
  data/        manifest scan (remote HF + local dirs), sampling & splits, safe image I/O
  transforms/  robustness grid (eval) + augmentation (train); crop policy
  models/      frozen + LoRA encoders, parameter budget, detector interface
  training/    feature cache + linear head (sklearn, for the frozen probe)
  evaluation/  metrics, robustness matrix, shortcut canaries
scripts/
  run_canary.py            audit a dataset for label-leaking artifacts
  build_manifest.py        metadata-only scan of all Community Forensics shards
  analyze_manifest.py      composition audit + sampling policy
  build_splits.py          splits, with a hard gate on the canary
  prepare_local_dataset.py build a manifest from a local real/fake dir (e.g. GenImage)
  cache_features.py        extract + cache features (frozen or a LoRA checkpoint)
  materialize_images.py    pull remote images to local disk once, for LoRA fine-tuning
  finetune_lora.py         phase 2: LoRA fine-tune the backbone end-to-end
  train_and_evaluate.py    phase 1: fit the head, emit the robustness matrix
  predict.py               REQUIRED: image dir -> JSON
experiments/LEDGER.md    every experiment, hypothesis first, negative results kept
HOWTO.md                 practical step-by-step: inference, data prep, both fine-tuning phases
```

`experiments/LEDGER.md` is worth reading — several of its entries are refutations
of this project's own earlier assumptions.
