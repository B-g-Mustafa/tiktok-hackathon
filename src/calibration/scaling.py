"""Post-hoc score calibration for distribution shift.

Why this exists
---------------
A detector trained on one dataset and evaluated on another routinely shows a
large gap between how well it *ranks* images and how well it *classifies*
them. Measured on this project's own checkpoint: AUROC 0.8441 on GenImage
against 0.5544 accuracy at threshold 0.5 -- the scores separate the classes
perfectly respectably, but almost all of them land on one side of 0.5, so the
fixed threshold throws most of that separation away.

This is not a quirk of our model. It is the documented default behaviour of
AIGC detectors under shift: they acquire an implicit prior from training and
carry it into a target distribution where it no longer holds, biasing toward
calling generated images authentic.

The fix is a two-parameter affine correction in LOGIT space:

    z' = a * z + b

`a` is an inverse temperature (sharpens or softens confidence) and `b` shifts
the decision boundary. Both are fitted with the backbone and head completely
frozen, on a small sample from the target distribution. This costs no GPU and
no retraining.

What it does and does not fix
-----------------------------
Calibration is a monotone transform of the score, so it CANNOT change AUROC,
AP, or any other rank-based metric -- those are exactly invariant under it.
It changes accuracy, balanced accuracy, ECE and Brier. Reporting a
calibration gain as a generalization improvement would be wrong, and the
distinction is worth stating explicitly wherever these numbers appear.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["LogitScaler", "probabilities_to_logits", "logits_to_probabilities"]

# Probabilities are clipped this far from {0, 1} before the logit transform.
# A saturated 0.0 or 1.0 maps to infinity, which would poison the fit; the
# clip costs nothing since a score that extreme is already unambiguous.
_EPS = 1e-6


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    """Inverse sigmoid, guarded against saturated inputs."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (avoids overflow for large negative input)."""
    z = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


@dataclass
class LogitScaler:
    """An affine correction `z' = a*z + b` applied in logit space.

    Identity by default (`a=1, b=0`), so an uncalibrated model and a
    calibrated one run through exactly the same code path -- there is no
    separate "calibrated" branch that could drift from the plain one.
    """

    a: float = 1.0
    b: float = 0.0
    method: str = "identity"
    n_fit: int = 0

    # -- fitting ------------------------------------------------------------

    def fit(
        self, scores: np.ndarray, labels: np.ndarray, regularization: float = 1e6
    ) -> "LogitScaler":
        """Fit both parameters by logistic regression on the raw logit.

        This is Platt scaling: a one-feature logistic regression whose learned
        weight and intercept *are* `a` and `b`. Using sklearn rather than a
        hand-rolled optimizer is deliberate -- lbfgs on a 2-parameter convex
        problem is about as robust as this gets, and there is no reason to
        reimplement it.

        `regularization` is an sklearn `C`; the default is large enough to be
        effectively unpenalized, which is what you want when there are two
        parameters and (typically) thousands of samples.
        """
        from sklearn.linear_model import LogisticRegression

        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=int).ravel()

        if scores.shape != labels.shape:
            raise ValueError(
                f"shape mismatch: scores {scores.shape} vs labels {labels.shape}"
            )
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "cannot fit a supervised calibration on a single-class sample; "
                "use fit_to_prior() when target labels are unavailable"
            )

        z = probabilities_to_logits(scores).reshape(-1, 1)
        model = LogisticRegression(C=regularization, max_iter=1000).fit(z, labels)

        self.a = float(model.coef_.ravel()[0])
        self.b = float(model.intercept_.ravel()[0])
        self.method = "platt"
        self.n_fit = int(len(labels))
        return self

    def fit_to_prior(
        self, scores: np.ndarray, prior: float = 0.5
    ) -> "LogitScaler":
        """Fit the SHIFT only, with no labels at all.

        Uses the one piece of information usually available about a target set
        without annotating it: roughly what fraction of it is generated. The
        threshold is moved to the matching quantile of the score distribution,
        so exactly `prior` of the sample is predicted positive.

        Temperature is left at 1.0 on purpose -- without labels there is no
        signal about how sharp the scores should be, and inventing one would
        change ECE without justification. This corrects the bias, which is the
        part that actually costs accuracy under shift.
        """
        if not 0.0 < prior < 1.0:
            raise ValueError(f"prior must be in (0, 1), got {prior}")

        scores = np.asarray(scores, dtype=np.float64).ravel()
        if len(scores) == 0:
            raise ValueError("cannot fit calibration on an empty sample")

        z = probabilities_to_logits(scores)
        # Threshold at the (1 - prior) quantile: everything above it is called
        # generated, which is `prior` of the sample by construction.
        cut = float(np.quantile(z, 1.0 - prior))

        self.a = 1.0
        self.b = -cut
        self.method = "prior"
        self.n_fit = int(len(scores))
        return self

    # -- application --------------------------------------------------------

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply the correction to probabilities, returning probabilities."""
        z = probabilities_to_logits(np.asarray(scores, dtype=np.float64))
        return logits_to_probabilities(self.a * z + self.b)

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply the correction to raw logits, returning logits.

        Preferred when the caller already has logits -- avoids a needless
        sigmoid/inverse-sigmoid round trip and the clipping that comes with it.
        """
        return self.a * np.asarray(logits, dtype=np.float64) + self.b

    @property
    def is_identity(self) -> bool:
        return self.a == 1.0 and self.b == 0.0

    # -- persistence --------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"a": self.a, "b": self.b, "method": self.method, "n_fit": self.n_fit},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path | str) -> "LogitScaler":
        payload = json.loads(Path(path).read_text())
        return cls(
            a=float(payload["a"]),
            b=float(payload["b"]),
            method=str(payload.get("method", "unknown")),
            n_fit=int(payload.get("n_fit", 0)),
        )

    @classmethod
    def load_if_present(cls, path: Path | str) -> "LogitScaler":
        """Load a calibration, or return the identity if none was saved.

        Lets an inference path unconditionally apply a scaler without needing
        to branch on whether the checkpoint happens to carry one.
        """
        path = Path(path)
        return cls.load(path) if path.exists() else cls()

    def __str__(self) -> str:
        if self.is_identity:
            return "LogitScaler(identity -- uncalibrated)"
        return (
            f"LogitScaler(a={self.a:.4f}, b={self.b:.4f}, "
            f"method={self.method}, n_fit={self.n_fit:,})"
        )
