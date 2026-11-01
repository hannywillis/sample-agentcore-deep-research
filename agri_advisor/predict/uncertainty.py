"""Uncertainty quantification (WP2 - "every recommendation carries a confidence
level and a consult-expert signal").

Methods implemented
-------------------
* **Ensemble variance** - spread of per-tree predictions in a Random Forest.
* **Split conformal prediction** - distribution-free prediction intervals with
  finite-sample ``1 - alpha`` coverage, calibrated on a held-out split.
* **Class-probability confidence** - max probability and tree disagreement for
  classifiers.
* **Data completeness** - share of inputs that had to be imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def downgrade(self) -> ConfidenceLevel:
        return {
            ConfidenceLevel.HIGH: ConfidenceLevel.MEDIUM,
            ConfidenceLevel.MEDIUM: ConfidenceLevel.LOW,
            ConfidenceLevel.LOW: ConfidenceLevel.LOW,
        }[self]


_RANK = {ConfidenceLevel.HIGH: 2, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.LOW: 0}


def min_level(*levels: ConfidenceLevel) -> ConfidenceLevel:
    return min(levels, key=lambda lv: _RANK[lv])


@dataclass
class Uncertainty:
    level: ConfidenceLevel
    score: float  # 0..1, higher = more confident
    interval: tuple[float, float] | None = None
    ensemble_std: float | None = None
    coverage: float | None = None  # nominal conformal coverage
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level.value,
            "score": round(self.score, 3),
            "interval": [round(v, 3) for v in self.interval] if self.interval else None,
            "ensemble_std": None
            if self.ensemble_std is None
            else round(self.ensemble_std, 4),
            "coverage": self.coverage,
            "reasons": self.reasons,
        }


class SplitConformalRegressor:
    """Absolute-residual split conformal intervals for any point regressor."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.q_hat: float = float("nan")

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        n = len(residuals)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(residuals, level, method="higher"))
        return self.q_hat

    def interval(self, y_pred: float) -> tuple[float, float]:
        return (max(0.0, y_pred - self.q_hat), y_pred + self.q_hat)


def completeness_penalty(n_imputed: int, n_total: int) -> tuple[float, list[str]]:
    frac = n_imputed / max(n_total, 1)
    reasons = []
    if frac > 0.3:
        reasons.append(f"{frac:.0%} of input data missing (filled with typical values)")
    return frac, reasons


def regression_uncertainty(
    prediction: float,
    tree_predictions: np.ndarray,
    conformal: SplitConformalRegressor | None,
    imputed_fraction: float,
) -> Uncertainty:
    std = float(np.std(tree_predictions))
    interval = conformal.interval(prediction) if conformal else None
    width = (interval[1] - interval[0]) if interval else 4 * std
    rel_width = width / max(abs(prediction), 0.25)
    score = float(np.clip(1 - rel_width / 1.5, 0, 1) * (1 - 0.5 * imputed_fraction))
    level = (
        ConfidenceLevel.HIGH
        if rel_width < 0.5
        else ConfidenceLevel.MEDIUM
        if rel_width < 1.0
        else ConfidenceLevel.LOW
    )
    reasons = []
    if level is ConfidenceLevel.LOW:
        reasons.append("wide prediction interval")
    if imputed_fraction > 0.3:
        level = level.downgrade()
        reasons.append("many inputs missing")
    return Uncertainty(
        level=level,
        score=score,
        interval=interval,
        ensemble_std=std,
        coverage=None if conformal is None else 1 - conformal.alpha,
        reasons=reasons,
    )


def classification_uncertainty(
    proba: np.ndarray, tree_proba: np.ndarray | None, imputed_fraction: float
) -> Uncertainty:
    p_max = float(np.max(proba))
    disagreement = (
        float(np.mean(np.std(tree_proba, axis=0))) if tree_proba is not None else 0.0
    )
    # Normalised margin over chance: 0 = uniform guess, 1 = certain.
    k = len(proba)
    margin = (p_max - 1 / k) / (1 - 1 / k)
    score = float(np.clip(margin * (1 - 0.5 * disagreement), 0, 1))
    score *= 1 - 0.5 * imputed_fraction
    if p_max >= 0.75 and disagreement < 0.3:
        level = ConfidenceLevel.HIGH
    elif p_max >= 0.55:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW
    reasons = []
    if level is ConfidenceLevel.LOW:
        reasons.append("model is unsure between outcomes")
    if imputed_fraction > 0.3:
        level = level.downgrade()
        reasons.append("many inputs missing")
    return Uncertainty(
        level=level, score=score, ensemble_std=disagreement, reasons=reasons
    )
