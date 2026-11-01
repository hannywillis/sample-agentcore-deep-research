"""Transparent rule-based advisors for planting window and soil management.

These are deliberately simple, auditable heuristics that act as baselines
until data-driven models are trained (proposal use cases: planting windows,
soil management). Thresholds live in one place so agronomists can review them.

TODO(WP2): replace with crop-model / ML estimates calibrated per agro-ecological
zone (e.g. onset-of-rains detection from CHIRPS + seasonal forecast skill).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from agri_advisor.predict.schema import FarmContext
from agri_advisor.predict.uncertainty import ConfidenceLevel

# Days after rainfall onset considered "on time", per crop (baseline values).
PLANTING_WINDOW_DAYS = {
    "maize": 14,
    "wheat": 21,
    "sorghum": 14,
    "rice": 21,
    "millet": 10,
}
SOIL_PH_RANGE = (5.5, 7.0)
LOW_ORGANIC_CARBON_PCT = 1.0
LOW_PHOSPHORUS_PPM = 15.0


def planting_window(ctx: FarmContext) -> dict[str, Any]:
    crop = ctx.farm.crop
    onset = ctx.climate.rainfall_onset_date
    window = PLANTING_WINDOW_DAYS.get(crop, 14)
    if onset is None:
        return {
            "task": "planting_window",
            "advice": (
                "Plant when the soil is moist to about 30 cm after the rains start."
            ),
            "window": None,
            "confidence": ConfidenceLevel.LOW.value,
            "reason": "rainfall onset date unknown",
            "method": "rule-based baseline",
        }
    start, end = onset, onset + timedelta(days=window)
    forecast = ctx.climate.forecast_rainfall_mm
    level = ConfidenceLevel.MEDIUM
    note = ""
    if forecast is not None and forecast < 250:
        note = " Seasonal forecast is dry: prefer an early-maturing variety."
        level = ConfidenceLevel.LOW
    return {
        "task": "planting_window",
        "advice": (
            f"Best planting window for {crop}: {start:%d %b} to {end:%d %b}.{note}"
        ),
        "window": [start.isoformat(), end.isoformat()],
        "confidence": level.value,
        "method": "rule-based baseline",
    }


def soil_management(ctx: FarmContext) -> dict[str, Any]:
    s = ctx.soil
    actions: list[str] = []
    if s.ph is not None and s.ph < SOIL_PH_RANGE[0]:
        actions.append("Soil is acidic: apply agricultural lime based on a soil test.")
    if s.ph is not None and s.ph > SOIL_PH_RANGE[1] + 0.5:
        actions.append("Soil is alkaline: seek advice on suitable fertilizers.")
    if (
        s.organic_carbon_pct is not None
        and s.organic_carbon_pct < LOW_ORGANIC_CARBON_PCT
    ):
        actions.append("Low organic matter: add manure/compost and keep crop residues.")
    if s.phosphorus_ppm is not None and s.phosphorus_ppm < LOW_PHOSPHORUS_PPM:
        actions.append("Low phosphorus: use a phosphorus fertilizer at planting.")
    known = sum(v is not None for v in (s.ph, s.organic_carbon_pct, s.phosphorus_ppm))
    return {
        "task": "soil_management",
        "actions": actions
        or (["No soil problem detected from the data given."] if known else []),
        "confidence": (
            ConfidenceLevel.MEDIUM if known >= 2 else ConfidenceLevel.LOW
        ).value,
        "method": "rule-based baseline",
    }
