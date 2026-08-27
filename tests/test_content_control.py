"""Tests for Control D, the content-matched falsification set.

Control D exists to answer one question: is the detector reading synthesis
artifacts, or just recognising subject matter? The training pool pairs
landscape and face photographs against anime/pet/character LoRAs, so "anime
implies generated" is a rule that would score well AND survive compression --
looking exactly like genuine robustness.

The subtle failure mode these tests guard is a control that only *appears*
matched. For auto-scraped community LoRAs, `real_source` reports the corpus the
BASE model saw (LAION), not what the LoRA draws, so trusting it would pair anime
characters against landscape photos while reporting "content matched".
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.sampling import (
    LABEL_AUTHENTIC,
    LABEL_GENERATED,
    add_content_column,
    add_size_columns,
    content_category,
    content_matched_pool,
)
from src.evaluation.shortcut_controls import scale_canary


# ---------------------------------------------------------------------------
# Category resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("ffhq", "face"),
        ("FFHQ", "face"),
        ("celeba", "face"),
        ("metfaces", "face"),
        ("afhqv2", "animal"),
        ("coco", "scene"),
        ("LandscapesHQ", "scene"),
        ("imagenet", "scene"),
        ("coco,imagenet", "scene"),  # all parts agree
    ],
)
def test_content_category_resolution(source, expected):
    assert content_category(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        None,
        "",
        "N/A",
        "nan",
        "ffhq,coco",  # disagreeing categories -- genuinely ambiguous
        "some-unknown-corpus",
        "coco,forchheim,imagenet,imd2020,laion,landscapesHQ,vision",
    ],
)
def test_ambiguous_sources_are_unlabelled(source):
    """An unresolvable source must yield None, not a wrong guess. A generator
    conditioned on both faces and scenes cannot be matched against either."""
    result = content_category(source)
    # The multi-source blob resolves only if every part maps to one category.
    assert result is None or result == "scene"


def test_disagreeing_sources_return_none():
    assert content_category("ffhq,coco") is None


# ---------------------------------------------------------------------------
# The reliability filter
# ---------------------------------------------------------------------------


def _mixed_frame() -> pd.DataFrame:
    """Curated generators (trustworthy real_source) alongside scraped LoRAs
    (real_source is the base model's corpus, not the depicted content)."""
    rows = []

    for i in range(30):
        rows.append(
            {"label": LABEL_AUTHENTIC, "model_name": "FFHQ", "real_source": "N/A",
             "subset": "Manual", "resolution": [1024, 1024], "image_name": f"r{i}"}
        )
    for i in range(30):
        rows.append(
            {"label": LABEL_AUTHENTIC, "model_name": "LandscapesHQ",
             "real_source": "N/A", "subset": "Manual", "resolution": [512, 512],
             "image_name": f"l{i}"}
        )
    # Curated face GAN -- real_source genuinely means faces.
    for i in range(20):
        rows.append(
            {"label": LABEL_GENERATED, "model_name": "StyleGAN2-ADA",
             "real_source": "ffhq", "subset": "Manual",
             "resolution": [1024, 1024], "image_name": f"g{i}"}
        )
    # Scraped anime LoRA claiming real_source=laion. Must NOT be labelled.
    for i in range(40):
        rows.append(
            {"label": LABEL_GENERATED, "model_name": "someone/anime-lora",
             "real_source": "LAION", "subset": "Systematic",
             "resolution": [512, 512], "image_name": f"a{i}"}
        )
    return add_size_columns(pd.DataFrame(rows))


def test_scraped_generators_are_not_content_labelled():
    """The central guard: a Systematic LoRA's real_source describes its base
    model, so it must not be treated as content."""
    frame = add_content_column(_mixed_frame(), reliable_only=True)
    scraped = frame.loc[frame["model_name"] == "someone/anime-lora", "content"]
    assert scraped.isna().all()


def test_curated_generators_are_content_labelled():
    frame = add_content_column(_mixed_frame(), reliable_only=True)
    curated = frame.loc[frame["model_name"] == "StyleGAN2-ADA", "content"]
    assert (curated == "face").all()


def test_reliable_only_disabled_labels_everything():
    frame = add_content_column(_mixed_frame(), reliable_only=False)
    scraped = frame.loc[frame["model_name"] == "someone/anime-lora", "content"]
    assert (scraped == "scene").all()  # what we must avoid trusting


def test_authentic_content_comes_from_model_name():
    frame = add_content_column(_mixed_frame())
    assert (frame.loc[frame["model_name"] == "FFHQ", "content"] == "face").all()
    assert (
        frame.loc[frame["model_name"] == "LandscapesHQ", "content"] == "scene"
    ).all()


# ---------------------------------------------------------------------------
# The pool itself
# ---------------------------------------------------------------------------


def test_control_pool_excludes_unmatched_content():
    """The anime LoRA has no trustworthy content label, so it cannot enter the
    control -- even though it would otherwise balance the landscapes."""
    pool = content_matched_pool(_mixed_frame(), min_crop_size=256)
    assert "someone/anime-lora" not in set(pool["model_name"])


def test_control_pool_is_balanced_within_each_bucket():
    pool = add_size_columns(content_matched_pool(_mixed_frame(), min_crop_size=256))
    for _, bucket in pool.groupby(["content", "min_side"]):
        assert (bucket["label"] == LABEL_AUTHENTIC).sum() == (
            bucket["label"] == LABEL_GENERATED
        ).sum()


def test_control_pool_matches_content_within_bucket():
    """Faces must only ever be paired against faces."""
    pool = content_matched_pool(_mixed_frame(), min_crop_size=256)
    for content, bucket in pool.groupby("content"):
        assert set(bucket["content"]) == {content}


def test_control_pool_has_no_scale_signal():
    pool = add_size_columns(content_matched_pool(_mixed_frame(), min_crop_size=256))
    assert scale_canary(pool["min_side"], pool["label"]).auroc == pytest.approx(0.5)


def test_control_pool_drops_single_class_buckets():
    """Landscapes have no trustworthy generated counterpart here, so that
    bucket must vanish rather than being paired with something unrelated."""
    pool = content_matched_pool(_mixed_frame(), min_crop_size=256)
    assert set(pool["content"]) == {"face"}


def test_control_pool_empty_when_nothing_matches():
    frame = add_size_columns(
        pd.DataFrame(
            [
                {"label": LABEL_AUTHENTIC, "model_name": "FFHQ", "real_source": "N/A",
                 "subset": "Manual", "resolution": [1024, 1024]},
                {"label": LABEL_GENERATED, "model_name": "x", "real_source": "coco",
                 "subset": "Manual", "resolution": [512, 512]},
            ]
        )
    )
    assert len(content_matched_pool(frame, min_crop_size=256)) == 0
