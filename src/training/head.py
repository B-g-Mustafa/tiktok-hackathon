"""The trainable head, and the cached-feature store it learns from.

The head is a logistic regression over frozen encoder features. That is a
deliberate choice rather than a placeholder: linear probes on frozen foundation
features generalize to unseen generators better than fine-tuned backbones, the
whole model is ~4.6K parameters against a 2B budget, and training takes seconds,
which is what makes a wide ablation sweep affordable inside a short deadline.

Feature caches are keyed by a config hash. Loading a cache whose hash does not
match the one being requested raises rather than silently proceeding: mixing
feature sets extracted under different settings produces plausible-looking
numbers with no valid interpretation, and nothing downstream would reveal it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["FeatureCache", "LinearHead", "load_cache"]


@dataclass
class FeatureCache:
    """Cached features plus the metadata needed to interpret them.

    `view_names` records which transform produced each row, which is what lets
    the robustness matrix be computed by grouping rather than by re-extracting.
    """

    features: np.ndarray  # (n_rows, dim)
    labels: np.ndarray  # (n_rows,) 1 = generated
    view_names: np.ndarray  # (n_rows,) transform or augmentation name
    keys: np.ndarray  # (n_rows,) source image identifier
    generators: np.ndarray  # (n_rows,) generator / real source
    meta: dict

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    def view(self, name: str) -> "FeatureCache":
        """Rows produced by one named transform."""
        mask = self.view_names == name
        return self._subset(mask)

    def clean(self) -> "FeatureCache":
        return self.view("clean")

    def _subset(self, mask: np.ndarray) -> "FeatureCache":
        return FeatureCache(
            features=self.features[mask],
            labels=self.labels[mask],
            view_names=self.view_names[mask],
            keys=self.keys[mask],
            generators=self.generators[mask],
            meta=self.meta,
        )

    def unique_views(self) -> list[str]:
        return sorted(set(self.view_names.tolist()))


def load_cache(path: Path | str, expect_hash: str | None = None) -> FeatureCache:
    """Load a feature cache, refusing a config mismatch.

    A silent mismatch is the failure mode this guards: the metrics would look
    entirely normal while comparing features that mean different things.
    """
    path = Path(path)
    payload = np.load(path, allow_pickle=False)

    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if expect_hash is not None:
        actual = meta.get("config_hash")
        if actual != expect_hash:
            raise ValueError(
                f"feature cache config mismatch for {path.name}: "
                f"cache has {actual!r}, expected {expect_hash!r}. "
                f"Re-extract rather than mixing feature sets."
            )

    return FeatureCache(
        features=payload["features"].astype(np.float32),
        labels=payload["labels"].astype(int),
        view_names=payload["view_names"],
        keys=payload["keys"],
        generators=payload["generators"],
        meta=meta,
    )


class LinearHead:
    """Logistic regression over frozen features, with L2-normalized inputs.

    Normalizing each feature vector to unit length matters more than it looks:
    degradation changes activation *magnitude* substantially while preserving
    direction, so an unnormalized head partly learns "how strong is the signal"
    -- which is exactly the quantity JPEG and blur destroy. Normalizing makes
    the head read direction only, and is a large part of why a linear probe
    stays stable under the robustness grid.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 0) -> None:
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._model = None

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        # Guard against an all-zero feature row rather than emitting NaNs.
        return features / np.maximum(norms, 1e-8)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LinearHead":
        from sklearn.linear_model import LogisticRegression

        if len(np.unique(labels)) < 2:
            raise ValueError("cannot fit a head on a single-class training set")

        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self._model.fit(self._normalize(features), labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("head is not fitted")
        return self._model.predict_proba(self._normalize(features))[:, 1]

    @property
    def n_parameters(self) -> int:
        if self._model is None:
            return 0
        return int(self._model.coef_.size + self._model.intercept_.size)

    def save(self, path: Path | str) -> None:
        if self._model is None:
            raise RuntimeError("head is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            coef=self._model.coef_,
            intercept=self._model.intercept_,
            C=self.C,
        )

    @classmethod
    def load(cls, path: Path | str) -> "LinearHead":
        from sklearn.linear_model import LogisticRegression

        payload = np.load(path)
        head = cls(C=float(payload["C"]))
        model = LogisticRegression()
        model.coef_ = payload["coef"]
        model.intercept_ = payload["intercept"]
        model.classes_ = np.array([0, 1])
        head._model = model
        return head
