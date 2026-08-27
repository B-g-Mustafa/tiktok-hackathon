# How-To: Inference, Data, and Fine-Tuning

Practical, step-by-step. For the *why* behind the architecture and the
dataset-shortcut findings, see [README.md](README.md) and
[experiments/LEDGER.md](experiments/LEDGER.md).

---

## 1. Inference

```bash
python scripts/predict.py --image-dir /path/to/images --out preds.json
```

Output — one row per readable image:

```json
[
  {"image_path": "/abs/path/img1.jpg", "pred": 0.9371},
  {"image_path": "/abs/path/img2.png", "pred": 0.0832}
]
```

`pred` is P(AI-generated). Handles jpg/jpeg/png/webp/bmp/tiff, any resolution
or aspect ratio; unreadable files are skipped and logged to stderr (add
`--include-failures` to emit them with `"pred": null"` instead).

**Without `--model`/`--checkpoint`**, this runs the `constant` placeholder
detector (always returns 0.5) — useful to confirm the I/O contract works
before you have any trained weights. Once you have a checkpoint (from either
phase below):

```bash
python scripts/predict.py --image-dir /path/to/images \
  --model siglip2 --checkpoint artifacts/checkpoints/<your-checkpoint-dir> \
  --out preds.json
```

`predict.py` autodetects whether the checkpoint is a phase-1 (frozen probe) or
phase-2 (LoRA) checkpoint — same command either way. Add `--batch-size N` to
control throughput.

---

## 2. Does GenImage have real images?

**Yes.** Each generator in GenImage ships both classes, under `train/` and
`val/`:

```
<generator>/
├── train/
│   ├── ai/       ← generated
│   └── nature/   ← real (sourced from ImageNet)
└── val/
    ├── ai/
    └── nature/
```

This repeats per generator (`ADM/`, `BigGAN/`, `glide/`, `Midjourney/`,
`stable_diffusion_v_1_4/`, `stable_diffusion_v_1_5/`, `VQDM/`, `wukong/`). If
you downloaded via [`scripts/down-genimage-dataset.py`](scripts/down-genimage-dataset.py)
and extracted the zips, you already have both classes — **no separate real-image
dataset is needed.**

One thing worth knowing before you train on it, from
[experiments/LEDGER.md](experiments/LEDGER.md) (EXP-004): GenImage's `nature/`
images are ~100% JPEG while ~14% of its `ai/` images are PNG. That container
difference is a learnable shortcut — a model can score well by reading
compression artifacts instead of synthesis artifacts. Two ways to handle it:

- **Use it for evaluation** (recommended default) — it's a good OOD/cross-dataset
  test set precisely because it wasn't trained on.
- **Use it for training anyway** — fine, but re-encode both classes through a
  common JPEG-quality distribution first so the container difference doesn't
  become the thing the model learns. Not yet automated in this repo; flag it
  if you need it and it can be added to `prepare_local_dataset.py`.

If you want a *training* set that avoids this by construction, the codebase's
default is `OwensLab/CommunityForensics-Small` on Hugging Face (see the main
[README](README.md) for why, and its own shortcut, which the pipeline already
neutralizes). But you don't need it — GenImage alone is enough to run both
phases below.

---

## 2b. Extracting GenImage's split zips (if you hit "bad zipfile offset")

Each GenImage category is downloaded as ONE archive split across many volumes
(`imagenet_ai_0508_adm.z01`, `.z02`, ..., `.zip`). The `.zip` is the *last*
volume — it holds the archive's index, not a self-contained file. Running
plain `unzip` on just that file makes `unzip` seek to offsets that live in the
other volumes it never opened; it lands on garbage and reports it as a
corrupt/zip-bomb-shaped file. That's the exact error you hit — confirmed by
reproducing it locally against a real multi-volume test archive.

Use [`scripts/extract_genimage.py`](scripts/extract_genimage.py) instead —
verified end-to-end (byte-identical output, zero extra disk usage) against a
real split archive before being written:

```bash
# See the plan first — checks volumes are complete and disk space is
# sufficient, extracts nothing
python scripts/extract_genimage.py --root /path/to/gen-image-dataset --dry-run

# Extract everything found under --root (ADM, BIGGAN, Glide, midjourney, ...)
python scripts/extract_genimage.py --root /path/to/gen-image-dataset

# Just one category
python scripts/extract_genimage.py --root /path/to/gen-image-dataset --categories ADM
```

It uses `7z` if available — which reads across the split volumes directly and
extracts with **no intermediate combined copy**, so disk usage never exceeds
(original volumes) + (extracted output), which you'd need regardless. Check
with `which 7z`; if missing and you can't `sudo apt/yum install`, try
`conda install -c conda-forge p7zip` in your user environment, or
`module load p7zip` if your cluster provides one (common on HPC login nodes).

Without `7z`, it falls back to `zip -s 0 file.zip --out combined.zip` +
`unzip`, deleting the combined copy immediately after each category — bounded
to one category's extra disk at a time, never left behind. (A raw
`cat z01 z02 ... zip > combined.zip` looks like it should work but is fragile
in practice — a naive glob easily includes `.zip` twice or in the wrong
position and silently corrupts the join, which is exactly why this script uses
`zip -s0` instead of a one-liner.)

The script also catches a second, unrelated cause of the same error: a
**truncated download** (a missing `.z04` in the middle, say) produces
"bad zipfile offset" too, for a completely different reason. It checks the
volume sequence is complete before attempting extraction and tells you which
part is missing rather than trying and failing confusingly.

Once extracted, each category has an `extracted/train/{ai,nature}` and
`extracted/val/{ai,nature}` layout — feed that straight into step 3 below.

**Reclaiming disk after a verified extraction** (optional, off by default —
this permanently deletes your only local copy of the downloaded zip volumes):

```bash
python scripts/extract_genimage.py --root /path/to/gen-image-dataset \
  --categories ADM --delete-originals-after-verify
```

It only deletes a category's volumes after that category's extraction is both
successful *and* passes a structure check (`train/ai`, `train/nature`,
`val/ai`, `val/nature` all present and non-empty) — never before.

---

## 3. Directory arrangement before fine-tuning

Every training/eval script in this repo (`cache_features.py`, `finetune_lora.py`,
`train_and_evaluate.py`) reads from a **manifest**: a `manifest.parquet` file
listing each image's local path and label, always at:

```
artifacts/images/<split-name>/manifest.parquet
```

You don't create this by hand — `prepare_local_dataset.py` scans a real/fake
directory tree and builds it for you. It already understands GenImage's
`ai`/`nature` convention (and generic `real`/`fake`, `authentic`/`synthetic`
names, case-insensitive) — **you do not need to rearrange GenImage's folders
at all.** Point it straight at your extracted GenImage directory:

```bash
# Build the TRAIN manifest (only train/, not val/ — GenImage keeps both
# under every generator, so --split matters here)
python scripts/prepare_local_dataset.py \
  --root /path/to/genimage \
  --split train \
  --output-name genimage_train

# Build a VAL manifest the same way, for held-out evaluation
python scripts/prepare_local_dataset.py \
  --root /path/to/genimage \
  --split val \
  --output-name genimage_val
```

`--root` should point at the directory that *contains* the generator folders
(`ADM/`, `BigGAN/`, ...) — i.e. one level above them, so the scan can recurse
into each generator's `train/ai`, `train/nature`, etc.

Useful flags:
- `--limit-per-class N` — cap images per class. GenImage's full train split is
  well over a million images across generators; capping keeps a first run to
  tens of minutes instead of hours. (`n_layers`/rank etc. below don't need
  huge data to produce a real, if not fully converged, signal.)
- Without `--split`, everything under `--root` is scanned and train+val get
  merged into one manifest — only do this deliberately.

What this actually does, concretely: it walks the tree, labels every image by
which folder it's under (`ai`/`generated`/`fake` → 1, `nature`/`real`/`authentic`
→ 0), records the generator name (e.g. `BigGAN`) from the path, and writes
everything to `artifacts/images/genimage_train/manifest.parquet`. From here,
phase 1 and phase 2 both just point at `genimage_train` / `genimage_val` by name.

**If your own real/fake data isn't GenImage-shaped** — say, one flat `real/`
folder and one flat `generated/` folder anywhere — that's already directly
supported, no adaptation needed:

```
your_data/
├── real/
│   └── *.jpg
└── generated/
    └── *.png
```

```bash
python scripts/prepare_local_dataset.py --root your_data --output-name my_train
```

---

## 4. Fine-tune phase 1 alone (frozen probe)

This is the fast sanity checkpoint — frozen SigLIP2 backbone, ~4.6K trainable
parameters, trains in seconds once features are cached. Run this first: if it
can't separate the classes at all, something upstream (labels, data) is
broken, and you want to know that before spending an hour on phase 2.

```bash
# Extract features from the train manifest (random augmented views)
python scripts/cache_features.py \
  --local-manifest artifacts/images/genimage_train \
  --mode train --n-views 8

# Extract features from the val manifest (one view per robustness transform)
python scripts/cache_features.py \
  --local-manifest artifacts/images/genimage_val \
  --mode eval

# Fit the head and see the robustness matrix; save a checkpoint
python scripts/train_and_evaluate.py \
  --train-cache artifacts/features/genimage_train__*.npz \
  --eval-cache  artifacts/features/genimage_val__*.npz \
  --save-checkpoint artifacts/checkpoints/phase1
```

`--save-checkpoint` writes `head.npz` + `meta.json` to that directory — that's
what `predict.py --checkpoint artifacts/checkpoints/phase1` loads.

Add `--limit N` to either `cache_features.py` call for a fast smoke test
before committing to the full dataset.

---

## 5. Fine-tune phase 2 (LoRA — the competitive model)

This trains LoRA adapters through the backbone end-to-end (gradients require
raw images, not cached features), so it needs the manifest images to exist
locally — which they already do from step 3, no separate download step:

```bash
python scripts/finetune_lora.py \
  --images-dir artifacts/images \
  --train-split genimage_train \
  --epochs 3 \
  --lora-rank 8 \
  --batch-size 32 \
  --output artifacts/checkpoints/phase2
```

This trains on `genimage_train`, holds out `--val-fraction` (default 10%) of
it for per-epoch monitoring, and saves the best checkpoint (adapter +
head) to `artifacts/checkpoints/phase2`.

For the real robustness-matrix numbers (not just the training-time monitor
AUROC), extract eval features with the *fine-tuned* encoder and run the same
evaluation code phase 1 used:

```bash
python scripts/cache_features.py \
  --local-manifest artifacts/images/genimage_val --mode eval \
  --adapter artifacts/checkpoints/phase2

python scripts/train_and_evaluate.py \
  --train-cache artifacts/features/genimage_train__*.npz \
  --eval-cache  artifacts/features/genimage_val__*-lora__*.npz
```

(`cache_features.py` tags LoRA-extracted files with `-lora` in the filename
specifically so this glob can tell them apart from the phase-1 frozen-probe
cache of the same split — both share the same encoder name otherwise.)

Then run inference with it exactly as in step 1:

```bash
python scripts/predict.py --image-dir /path/to/images \
  --model siglip2 --checkpoint artifacts/checkpoints/phase2 --out preds.json
```

**Useful flags:** `--lora-rank` (higher = more trainable capacity, still tiny
against the 2B budget — rank 8 on the base-384 encoder is ~632K params, ~1.3%
of that tower), `--n-layers` (how many final transformer blocks contribute to
the feature vector), `--num-workers` (CPU workers for the augmentation
DataLoader — raise this if the GPU is starved), `--device` (defaults to
auto-detect CUDA/MPS/CPU).

---

## Quick reference

| Task | Command |
|---|---|
| Inference (no model yet) | `predict.py --image-dir DIR --out preds.json` |
| Inference (trained) | `predict.py --image-dir DIR --model siglip2 --checkpoint CKPT --out preds.json` |
| Build a manifest from local real/fake dirs | `prepare_local_dataset.py --root DIR --output-name NAME [--split train\|val]` |
| Phase 1: cache features | `cache_features.py --local-manifest artifacts/images/NAME --mode train\|eval` |
| Phase 1: train + eval | `train_and_evaluate.py --train-cache ... --eval-cache ... --save-checkpoint DIR` |
| Phase 2: LoRA fine-tune | `finetune_lora.py --images-dir artifacts/images --train-split NAME --output DIR` |
