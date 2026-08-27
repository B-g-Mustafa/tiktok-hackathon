"""Turn the raw manifest into training and evaluation splits that measure
forensics rather than dataset artifacts.

The audit (EXP-000 / EXP-002) established three facts that this module exists to
neutralize:

  * Resolution alone separates the classes at 0.90 AUROC across all 556K rows.
  * Container format differs sharply by class (PNG: 55% of authentic, 98% of
    generated).
  * Every COCO-sourced real is a contamination risk, because the organizers'
    demonstration benchmark uses COCO val2017 as its authentic class.

Two complementary strategies address the resolution shortcut, and we use both:

  NATIVE-CROP (primary, large pool)
      Keep every image whose native size is at least the crop size, then take
      equal-size crops from both classes. No resampling is applied to either
      class, so no interpolation artifact is introduced, and the model never
      observes image dimensions. Residual risk: scene scale still differs
      (a 256px crop of a 1024px portrait covers less of the subject than the
      same crop of a 512px generation).

  RESOLUTION-MATCHED (control, small pool)
      Keep only images whose *native resolution* appears in both classes. Here
      the size signal is zero by construction, so it isolates the residual
      scale effect the native-crop policy cannot rule out. It is small
      (~17K images) and content-skewed toward FFHQ faces, which makes it a poor
      training set but an excellent falsification test.

Training uses the native-crop pool; the resolution-matched pool is held back to
verify that a model trained on the former has not simply learned scale.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "LABEL_AUTHENTIC",
    "LABEL_GENERATED",
    "CONTAMINATED_REAL_SOURCES",
    "load_manifest",
    "exclude_contaminated_sources",
    "add_size_columns",
    "native_crop_pool",
    "min_side_matched_pool",
    "resolution_matched_pool",
    "content_matched_pool",
    "content_category",
    "add_content_column",
    "CONTENT_CATEGORIES",
    "balance_classes",
    "generator_disjoint_split",
    "SplitSummary",
    "ShardPlan",
    "plan_shards",
    "restrict_to_shards",
    "SHARD_SIZE_GB",
]

LABEL_AUTHENTIC = 0
LABEL_GENERATED = 1

# Real-image sources we must never train on.
#   COCO  -- the organizers' demonstration benchmark uses COCO val2017 as its
#            authentic class, so training on any COCO real risks leaking it.
#            The manifest does not distinguish train from val2017, so we drop
#            COCO wholesale rather than guess.
#   RAISE -- the Community Forensics authors explicitly forbid training on it;
#            it is reserved as a held-out real source.
CONTAMINATED_REAL_SOURCES = frozenset({"coco", "raise"})


def load_manifest(path: Path | str) -> pd.DataFrame:
    """Read the manifest parquet into a dataframe."""
    frame = pd.read_parquet(path)
    if "label" not in frame.columns:
        raise ValueError(f"manifest at {path} has no 'label' column")
    return frame


def add_size_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand the `resolution` list column into width/height/min_side.

    `min_side` is what determines crop eligibility: an image can supply an
    NxN crop only if both of its sides are at least N.
    """
    frame = frame.copy()

    resolutions = frame["resolution"].apply(
        lambda r: (int(r[0]), int(r[1]))
        if isinstance(r, (list, tuple, np.ndarray)) and len(r) >= 2
        else (0, 0)
    )
    frame["width"] = [r[0] for r in resolutions]
    frame["height"] = [r[1] for r in resolutions]
    frame["min_side"] = frame[["width", "height"]].min(axis=1)
    return frame


def exclude_contaminated_sources(
    frame: pd.DataFrame, sources: frozenset[str] = CONTAMINATED_REAL_SOURCES
) -> pd.DataFrame:
    """Drop authentic images from sources that would contaminate evaluation.

    Only the authentic class is filtered: a generated image is not made unsafe
    by having been prompted from a COCO caption.
    """
    if "model_name" not in frame.columns:
        return frame

    name = frame["model_name"].astype(str).str.lower()
    is_authentic = frame["label"] == LABEL_AUTHENTIC
    is_contaminated = name.apply(lambda n: any(s in n for s in sources))

    return frame.loc[~(is_authentic & is_contaminated)].copy()


def native_crop_pool(frame: pd.DataFrame, crop_size: int = 256) -> pd.DataFrame:
    """Images large enough to yield a native-resolution crop of `crop_size`.

    Excluding smaller images matters: upscaling them to reach the crop size
    would inject interpolation artifacts, and those artifacts are precisely the
    kind of signal a forensic detector latches onto. Better to drop the image
    than to fabricate evidence.
    """
    if "min_side" not in frame.columns:
        frame = add_size_columns(frame)
    return frame.loc[frame["min_side"] >= crop_size].copy()


# Content categories, used to build the falsification control.
#
# For authentic images the source dataset *is* the content: FFHQ is faces,
# LandscapesHQ is landscapes. For generated images, `real_source` records the
# data the generator was trained on or conditioned by, which is the best
# available proxy for what it depicts -- StyleGAN2-ADA with real_source=ffhq
# produces faces.
CONTENT_CATEGORIES: dict[str, tuple[str, ...]] = {
    "face": ("ffhq", "celeba", "celebhq", "metfaces"),
    "animal": ("afhqv2",),
    "scene": ("coco", "imagenet", "laion", "landscapeshq", "lhq", "vision"),
}


def content_category(source: object) -> str | None:
    """Map a source string to a content category, or None if ambiguous.

    Multi-source strings (e.g. "coco,imagenet") resolve only when every listed
    source agrees; a generator conditioned on both faces and scenes cannot be
    content-matched against either.
    """
    if source is None:
        return None

    text = str(source).strip().lower()
    if not text or text in ("n/a", "none", "nan"):
        return None

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None

    found: set[str] = set()
    for part in parts:
        for category, keys in CONTENT_CATEGORIES.items():
            if any(key in part for key in keys):
                found.add(category)
                break
        else:
            return None  # an unrecognised source makes the whole row ambiguous

    return found.pop() if len(found) == 1 else None


# `real_source` only describes depicted content for the manually curated
# generators. Community LoRAs scraped automatically (subset "Systematic") report
# the corpus their *base model* was trained on -- almost always LAION -- which
# says nothing about what the LoRA itself draws. Treating those as "scene"
# content would silently match anime characters against landscape photographs
# and defeat the purpose of the control.
RELIABLE_CONTENT_SUBSET = "Manual"


def add_content_column(
    frame: pd.DataFrame, reliable_only: bool = True
) -> pd.DataFrame:
    """Label each row with a content category.

    Authentic rows take it from `model_name` (the source dataset *is* the
    content). Generated rows take it from `real_source` (what the generator was
    trained on), but only where that field is trustworthy -- see
    `RELIABLE_CONTENT_SUBSET`.
    """
    frame = frame.copy()

    authentic = frame["label"] == LABEL_AUTHENTIC
    from_model = frame.get("model_name", pd.Series(index=frame.index, dtype=object))
    from_source = frame.get("real_source", pd.Series(index=frame.index, dtype=object))

    generated_content = from_source.apply(content_category)

    if reliable_only and "subset" in frame.columns:
        trustworthy = frame["subset"].astype(str) == RELIABLE_CONTENT_SUBSET
        generated_content = generated_content.where(trustworthy, other=None)

    frame["content"] = np.where(
        authentic, from_model.apply(content_category), generated_content
    )
    return frame


def content_matched_pool(
    frame: pd.DataFrame, min_crop_size: int = 256
) -> pd.DataFrame:
    """Balance classes within each (content category, scale) bucket.

    This builds the falsification control for the whole project.

    The training pool matches scale but *not* content: its authentic images are
    landscapes and faces while its generated images are largely anime, pet and
    character LoRAs. A semantic encoder can exploit that gap directly -- "anime
    implies generated" is a rule that survives JPEG-30 compression perfectly,
    and would show up as an impressively flat robustness curve produced by
    entirely the wrong mechanism.

    Matching content as well as scale removes that escape route. What remains is
    faces against generated faces at identical resolution, so a model that still
    separates them is reading synthesis artifacts rather than subject matter.
    A collapse here falsifies the semantic-branch hypothesis, and is worth
    reporting either way.
    """
    if "content" not in frame.columns:
        frame = add_content_column(frame)
    if "min_side" not in frame.columns:
        frame = add_size_columns(frame)

    frame = frame.loc[
        (frame["min_side"] >= min_crop_size) & frame["content"].notna()
    ]
    if frame.empty:
        return frame.copy()

    kept: list[pd.DataFrame] = []
    for _, bucket in frame.groupby(["content", "min_side"], sort=True):
        authentic = bucket.loc[bucket["label"] == LABEL_AUTHENTIC]
        generated = bucket.loc[bucket["label"] == LABEL_GENERATED]
        n = min(len(authentic), len(generated))
        if n == 0:
            continue
        kept.append(authentic.head(n))
        kept.append(generated.head(n))

    if not kept:
        return frame.iloc[0:0].copy()

    return pd.concat(kept, ignore_index=True)


def min_side_matched_pool(
    frame: pd.DataFrame, min_crop_size: int = 256
) -> pd.DataFrame:
    """Balance the classes within each `min_side` bucket.

    This is the primary training pool, and the criterion is chosen to match
    exactly what a crop-based detector can perceive.

    After taking a fixed NxN native crop, total pixel count and aspect ratio are
    invisible to the model -- a 256x256 window looks the same whether it came
    from a 512x512 or a 512x768 image. The one size cue that survives is
    *scale*: the shorter side sets how much of the scene the window covers.

    So we bucket on `min_side` rather than on exact resolution. Requiring exact
    (width, height) agreement would be needlessly strict: on Community Forensics
    it collapses the usable pool from ~195K images and thousands of generators
    down to ~17K images and 31 generators, discarding precisely the generator
    diversity that drives unseen-generator generalization.

    Buckets containing only one class are dropped, since there is no way to
    balance them without resampling one class and fabricating interpolation
    artifacts.
    """
    if "min_side" not in frame.columns:
        frame = add_size_columns(frame)

    frame = frame.loc[frame["min_side"] >= min_crop_size]
    if frame.empty:
        return frame.copy()

    kept: list[pd.DataFrame] = []
    for _, bucket in frame.groupby("min_side", sort=True):
        authentic = bucket.loc[bucket["label"] == LABEL_AUTHENTIC]
        generated = bucket.loc[bucket["label"] == LABEL_GENERATED]
        n = min(len(authentic), len(generated))
        if n == 0:
            continue
        kept.append(authentic.head(n))
        kept.append(generated.head(n))

    if not kept:
        return frame.iloc[0:0].copy()

    return pd.concat(kept, ignore_index=True)


def resolution_matched_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """Only resolutions present in BOTH classes, balanced within each.

    Produces the strict control set: image dimensions carry exactly zero
    information about the label, by construction.
    """
    if "width" not in frame.columns:
        frame = add_size_columns(frame)

    frame = frame.loc[frame["min_side"] > 0]
    keyed = frame.assign(_res=list(zip(frame["width"], frame["height"])))

    authentic = set(keyed.loc[keyed["label"] == LABEL_AUTHENTIC, "_res"])
    generated = set(keyed.loc[keyed["label"] == LABEL_GENERATED, "_res"])
    shared = authentic & generated

    if not shared:
        return keyed.iloc[0:0].drop(columns="_res")

    kept: list[pd.DataFrame] = []
    for resolution in shared:
        at_res = keyed.loc[keyed["_res"] == resolution]
        n = min(
            (at_res["label"] == LABEL_AUTHENTIC).sum(),
            (at_res["label"] == LABEL_GENERATED).sum(),
        )
        if n == 0:
            continue
        for label in (LABEL_AUTHENTIC, LABEL_GENERATED):
            kept.append(at_res.loc[at_res["label"] == label].head(n))

    if not kept:
        return keyed.iloc[0:0].drop(columns="_res")

    return pd.concat(kept, ignore_index=True).drop(columns="_res")


def balance_classes(
    frame: pd.DataFrame,
    n_per_class: int | None = None,
    seed: int = 0,
    generator_column: str = "model_name",
    stratify_column: str | None = None,
) -> pd.DataFrame:
    """Draw an exactly class-balanced sample.

    Generated images are sampled with per-generator round-robin rather than
    uniformly at random. The dataset is dominated by a few prolific generators,
    so uniform sampling would over-represent them; round-robin maximizes the
    number of distinct generators seen, which is the property the Community
    Forensics results identify as driving unseen-generator generalization.

    `stratify_column` preserves an existing per-bucket balance. Balancing the
    classes globally is not enough: if the pool was matched on `min_side`,
    sampling across buckets freely re-introduces a scale difference between the
    classes (measurably -- it moved our scale canary from 0.500 to 0.538).
    Stratifying keeps the match exact.
    """
    rng = np.random.default_rng(seed)

    if stratify_column is not None and stratify_column in frame.columns:
        groups = list(frame.groupby(stratify_column, sort=True))
        # Allocate the per-class budget across buckets in proportion to what
        # each bucket can actually supply.
        capacities = [
            min(
                int((g["label"] == LABEL_AUTHENTIC).sum()),
                int((g["label"] == LABEL_GENERATED).sum()),
            )
            for _, g in groups
        ]
        total_capacity = sum(capacities)
        if total_capacity == 0:
            return frame.iloc[0:0].copy()

        budget = total_capacity if n_per_class is None else min(
            n_per_class, total_capacity
        )

        parts = []
        for (_, group), capacity in zip(groups, capacities):
            if capacity == 0:
                continue
            share = int(round(budget * capacity / total_capacity))
            share = min(share, capacity)
            if share == 0:
                continue
            parts.append(
                balance_classes(
                    group,
                    n_per_class=share,
                    seed=int(rng.integers(2**31)),
                    generator_column=generator_column,
                    stratify_column=None,
                )
            )

        if not parts:
            return frame.iloc[0:0].copy()

        combined = pd.concat(parts, ignore_index=True)
        return combined.sample(
            frac=1.0, random_state=int(rng.integers(2**31))
        ).reset_index(drop=True)

    authentic = frame.loc[frame["label"] == LABEL_AUTHENTIC]
    generated = frame.loc[frame["label"] == LABEL_GENERATED]

    available = min(len(authentic), len(generated))
    if available == 0:
        return frame.iloc[0:0].copy()

    n = available if n_per_class is None else min(n_per_class, available)

    authentic_sample = authentic.sample(n=n, random_state=int(rng.integers(2**31)))

    if generator_column in generated.columns:
        generated_sample = _round_robin_by_group(
            generated, generator_column, n, rng
        )
    else:
        generated_sample = generated.sample(
            n=n, random_state=int(rng.integers(2**31))
        )

    combined = pd.concat([authentic_sample, generated_sample], ignore_index=True)
    return combined.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(
        drop=True
    )


def _round_robin_by_group(
    frame: pd.DataFrame, column: str, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Take up to `n` rows, cycling across groups to maximize group coverage."""
    shuffled = frame.sample(frac=1.0, random_state=int(rng.integers(2**31)))
    # rank within each group: 0 for each group's first row, 1 for its second...
    rank = shuffled.groupby(column, sort=False).cumcount()
    # Sorting by rank interleaves groups: all the rank-0 rows (one per group)
    # come first, then all rank-1 rows, and so on.
    return shuffled.assign(_rank=rank).sort_values("_rank").head(n).drop(
        columns="_rank"
    )


@dataclass
class ShardPlan:
    """Which shards to download, and what they buy.

    Community Forensics-Small stores each shard as a SINGLE parquet row group of
    roughly 4.1 GB, so there is no such thing as fetching one image: reading any
    row downloads the whole shard. The dataset is also cleanly partitioned --
    93 shards hold only generated images, 92 hold only authentic ones, and just
    one is mixed -- so a balanced sample necessarily spans shards from both
    sides.

    A selection spread thinly across all 186 shards therefore costs ~763 GB to
    materialise. Concentrating the same number of images into a chosen handful
    of shards costs ~4.1 GB each, and the manifest lets us choose which ones so
    that concentration does not cost diversity.
    """

    shards: list[str]
    n_images: int
    n_generators: int
    estimated_gb: float
    note: str = ""

    def __str__(self) -> str:
        return (
            f"{len(self.shards)} shards ~{self.estimated_gb:.1f} GB -> "
            f"{self.n_images:,} images, {self.n_generators:,} generators"
        )


# Measured: 4.10 GB per shard, one row group each.
SHARD_SIZE_GB = 4.1


def _greedy_generator_coverage(
    generated: pd.DataFrame, n_shards: int, generator_column: str
) -> list[str]:
    """Pick shards that maximize the number of DISTINCT generators covered.

    Ranking shards independently by generator count is the obvious approach and
    it is wrong: generators cluster by shard, so the top-ranked shards often
    hold the same few generators and each additional one buys almost nothing.
    Greedy marginal coverage instead asks, at each step, which remaining shard
    adds the most generators we do not already have -- a standard set-cover
    heuristic, and here the difference between 1 generator and 4 for the same
    download budget.

    Ties are broken by image count, so equally diverse shards yield more data.
    """
    by_shard = {
        shard: set(group[generator_column].dropna())
        for shard, group in generated.groupby("shard")
    }
    sizes = generated.groupby("shard").size().to_dict()

    chosen: list[str] = []
    covered: set = set()

    while by_shard and len(chosen) < n_shards:
        best = max(
            by_shard,
            key=lambda s: (len(by_shard[s] - covered), sizes.get(s, 0)),
        )
        # Once nothing new is covered, fall back to pure image count so the
        # remaining budget still buys data rather than nothing.
        if not (by_shard[best] - covered):
            best = max(by_shard, key=lambda s: sizes.get(s, 0))

        chosen.append(best)
        covered |= by_shard.pop(best)

    return chosen


def plan_shards(
    frame: pd.DataFrame,
    n_shards_per_class: int = 5,
    min_side: int | None = 512,
    generator_column: str = "model_name",
) -> ShardPlan:
    """Choose whole shards to download for a balanced, scale-matched sample.

    Generated shards are ranked by how many distinct generators they contribute,
    since generator diversity is the property most strongly tied to
    unseen-generator performance and it is free to optimize here. Authentic
    shards are interchangeable within a source, so they are taken in order.
    """
    if "min_side" not in frame.columns:
        frame = add_size_columns(frame)

    pool = frame if min_side is None else frame.loc[frame["min_side"] == min_side]
    if pool.empty:
        return ShardPlan([], 0, 0, 0.0, f"no rows at min_side={min_side}")

    authentic = pool.loc[pool["label"] == LABEL_AUTHENTIC]
    generated = pool.loc[pool["label"] == LABEL_GENERATED]

    generated_shards = _greedy_generator_coverage(
        generated, n_shards_per_class, generator_column
    )

    authentic_ranked = authentic.groupby("shard").size().sort_values(ascending=False)
    authentic_shards = list(authentic_ranked.head(n_shards_per_class).index)

    chosen = generated_shards + authentic_shards
    selected = pool.loc[pool["shard"].isin(chosen)]

    n_authentic = int((selected["label"] == LABEL_AUTHENTIC).sum())
    n_generated = int((selected["label"] == LABEL_GENERATED).sum())

    return ShardPlan(
        shards=chosen,
        # Balanced usable count -- the smaller class caps the pair count.
        n_images=2 * min(n_authentic, n_generated),
        n_generators=int(
            selected.loc[selected["label"] == LABEL_GENERATED, generator_column]
            .nunique()
        ),
        estimated_gb=len(chosen) * SHARD_SIZE_GB,
        note=f"min_side={min_side}",
    )


def restrict_to_shards(frame: pd.DataFrame, shards: Sequence[str]) -> pd.DataFrame:
    """Keep only rows living in the given shards."""
    return frame.loc[frame["shard"].isin(set(shards))].copy()


@dataclass
class SplitSummary:
    """What a split actually contains, for the experiment ledger."""

    name: str
    n_total: int
    n_authentic: int
    n_generated: int
    n_generators: int
    held_out_generators: int = 0
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.n_total:,} images "
            f"({self.n_authentic:,} authentic / {self.n_generated:,} generated), "
            f"{self.n_generators:,} generators"
        )


def _stable_unit_interval(key: str) -> float:
    """A deterministic pseudo-random value in [0, 1) for `key`.

    Uses SHA-256 rather than Python's builtin `hash()`, which is salted with a
    random seed per process (`PYTHONHASHSEED`) specifically to resist
    hash-flooding attacks -- meaning the same string hashes differently across
    runs unless that salt is fixed. A real hash gives the same value for the
    same key on every run, every process, every machine, which is exactly what
    lets a train/val split stay stable as the underlying pool grows (see
    `generator_disjoint_split`).
    """
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:16], 16) / 0xFFFFFFFFFFFFFFFF


def generator_disjoint_split(
    frame: pd.DataFrame,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    generator_column: str = "model_name",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that no generator appears in both halves.

    This is the split that actually estimates hidden-test performance. A random
    row split would put images from the same generator on both sides, letting
    the model memorize per-generator fingerprints and report a score that will
    not survive contact with an unseen generator.

    Authentic images are split by a stable per-row hash, since they have no
    generator to hold out.

    Membership is decided by a per-item hash rather than shuffling the current
    pool and taking a prefix. That distinction matters in exactly one
    situation, but it is the situation this pipeline is actually used in:
    re-running with a LARGER --budget-gb (more shards -> more generators, more
    authentic rows) must not change which split an already-downloaded image
    belongs to. Shuffling `generators` and slicing the top N% depends on the
    full list's size and order, so adding shards can flip a generator that was
    in train into val and vice versa (measured: ~24% of shared generators
    flipped between a 100GB and a 150GB plan on the real dataset). Hashing each
    generator's identity independently of who else is in the pool this run
    fixes that: a generator's split assignment depends only on its own name and
    `seed`, never on how many other generators happen to be selected alongside
    it.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")

    generated = frame.loc[frame["label"] == LABEL_GENERATED]
    authentic = frame.loc[frame["label"] == LABEL_AUTHENTIC]

    generators = generated[generator_column].dropna().unique()
    if len(generators) < 2:
        raise ValueError(
            f"need at least 2 generators to build a disjoint split, "
            f"found {len(generators)}"
        )

    holdout_generators = {
        g
        for g in generators
        if _stable_unit_interval(f"{seed}:generator:{g}") < holdout_fraction
    }
    in_holdout = generated[generator_column].isin(holdout_generators)

    # Authentic images have no generator identity to hash, so use their own
    # stable row identifier (shard + position within it) instead. Falls back
    # to the dataframe index if those columns aren't present -- e.g. in tests
    # that construct a frame directly rather than via the manifest scan;
    # per-run stability isn't a concern there, only correctness of one call.
    if {"shard", "row_in_shard"}.issubset(authentic.columns):
        row_keys = (
            authentic["shard"].astype(str)
            + "#"
            + authentic["row_in_shard"].astype(str)
        )
    else:
        row_keys = authentic.index.to_series().astype(str)

    authentic_mask = row_keys.apply(
        lambda k: _stable_unit_interval(f"{seed}:row:{k}") < holdout_fraction
    ).to_numpy()

    train = pd.concat(
        [generated.loc[~in_holdout], authentic.loc[~authentic_mask]],
        ignore_index=True,
    )
    test = pd.concat(
        [generated.loc[in_holdout], authentic.loc[authentic_mask]],
        ignore_index=True,
    )

    return train, test


def summarize(
    frame: pd.DataFrame, name: str, generator_column: str = "model_name"
) -> SplitSummary:
    generated = frame.loc[frame["label"] == LABEL_GENERATED]
    n_generators = (
        generated[generator_column].nunique()
        if generator_column in frame.columns
        else 0
    )
    return SplitSummary(
        name=name,
        n_total=len(frame),
        n_authentic=int((frame["label"] == LABEL_AUTHENTIC).sum()),
        n_generated=len(generated),
        n_generators=int(n_generators),
    )
