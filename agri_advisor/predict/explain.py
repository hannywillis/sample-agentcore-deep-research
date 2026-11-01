"""Explainability (XAI) for predictive outputs (WP2, D2.6).

Uses SHAP ``TreeExplainer`` when the optional ``shap`` package is installed,
giving exact per-prediction feature attributions for tree ensembles. Otherwise
falls back to a transparent approximation: global impurity importance x the
standardised deviation of the farm's value from the training mean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from agri_advisor.predict.schema import FEATURE_LABELS, FEATURES

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    feature: str
    label: str
    value: float
    contribution: float  # signed, in model-output units (or relative for fallback)
    imputed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": round(self.value, 3),
            "contribution": round(self.contribution, 4),
            "direction": "increases" if self.contribution > 0 else "decreases",
            "imputed": self.imputed,
        }


def _shap_values(
    model: Any, x: np.ndarray, class_index: int | None
) -> np.ndarray | None:
    try:
        import shap
    except ImportError:
        return None
    try:
        values = shap.TreeExplainer(model).shap_values(x)
    except Exception as exc:  # pragma: no cover - shap/model incompatibility
        logger.info("SHAP failed, using fallback: %s", exc)
        return None
    arr = np.asarray(values[class_index] if isinstance(values, list) else values)
    if arr.ndim == 3:  # (n, features, classes)
        arr = arr[..., class_index if class_index is not None else 0]
    return np.asarray(arr[0])


def explain(
    model: Any,
    x: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    imputed: list[str],
    class_index: int | None = None,
    top_k: int = 5,
) -> tuple[str, list[FeatureContribution]]:
    """Return (method name, top-k contributions sorted by absolute size)."""
    contrib = _shap_values(model, x, class_index)
    method = "shap"
    if contrib is None:
        method = "importance_x_deviation"
        importances = np.asarray(
            getattr(model, "feature_importances_", np.ones(x.shape[1]))
        )
        z = (x[0] - train_mean) / np.where(train_std == 0, 1, train_std)
        contrib = importances * z
    order = np.argsort(-np.abs(contrib))
    out = []
    for i in order:
        feat = FEATURES[i]
        if feat.startswith("crop.") or feat in imputed:
            continue  # crop identity and imputed values are not actionable
        out.append(
            FeatureContribution(
                feature=feat,
                label=FEATURE_LABELS.get(feat, feat),
                value=float(x[0, i]),
                contribution=float(contrib[i]),
                imputed=False,
            )
        )
        if len(out) >= top_k:
            break
    return method, out


def summarise(contributions: list[FeatureContribution], n: int = 3) -> str:
    """Plain-language one-liner, e.g. 'Main factors: rainfall (up), pH (down)'."""
    if not contributions:
        return "Not enough farm data to explain this prediction."
    bits = [
        f"{c.label} ({'raises' if c.contribution > 0 else 'lowers'} it)"
        for c in contributions[:n]
    ]
    return "Main factors: " + ", ".join(bits) + "."
