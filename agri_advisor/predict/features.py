"""Convert a :class:`FarmContext` into the model feature vector."""

from __future__ import annotations

import numpy as np

from agri_advisor.predict.schema import (
    CROP_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    FarmContext,
)


def _lookup(ctx: FarmContext, dotted: str) -> float | None:
    section, name = dotted.split(".", 1)
    if dotted == "farm.planting_delay_days":
        pd, onset = ctx.farm.planting_date, ctx.climate.rainfall_onset_date
        return float((pd - onset).days) if pd and onset else None
    value = getattr(getattr(ctx, section), name, None)
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def context_to_vector(
    ctx: FarmContext, medians: dict[str, float]
) -> tuple[np.ndarray, list[str]]:
    """Return (1 x n_features vector, list of imputed feature names)."""
    row: list[float] = []
    imputed: list[str] = []
    for feat in NUMERIC_FEATURES:
        val = _lookup(ctx, feat)
        if val is None:
            val = medians.get(feat, 0.0)
            imputed.append(feat)
        row.append(val)
    row += [1.0 if f == f"crop.{ctx.farm.crop}" else 0.0 for f in CROP_FEATURES]
    assert len(row) == len(FEATURES)
    return np.asarray([row], dtype=np.float64), imputed
