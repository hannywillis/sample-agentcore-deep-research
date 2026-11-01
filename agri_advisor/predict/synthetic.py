"""SYNTHETIC training data for the scaffold models.

!!! WARNING !!!
These data are generated from hand-written, approximately plausible agronomic
relationships so that the full pipeline (training -> prediction -> uncertainty
-> explanation -> recommendation) can run end-to-end before real data exist.
Predictions from models trained on them are NOT agronomically valid and are
always flagged ``model_status="scaffold-synthetic"`` (which triggers expert
escalation by default).

TODO(WP2, D2.1-D2.5): replace :func:`generate` with loaders for the harmonised
multi-country dataset - CHIRPS/ERA5 climate, iSDAsoil/KALRO soil labs,
Sentinel-2/MODIS NDVI-EVI, and ground-truth farm surveys (D2.3) - then retrain
with ``python -m agri_advisor.predict.train --data <path>``.
"""

from __future__ import annotations

import numpy as np

from agri_advisor.predict.schema import CROP_FEATURES, FEATURES

BASE_YIELD_T_HA = {
    "maize": 2.2,
    "wheat": 2.5,
    "sorghum": 1.2,
    "rice": 3.0,
    "millet": 0.9,
}


def generate(n: int = 3000, seed: int = 42) -> dict[str, np.ndarray]:
    """Return ``{"X", "yield_t_ha", "pest_outbreak", "climate_risk"}``."""
    rng = np.random.default_rng(seed)
    crops = rng.choice(list(BASE_YIELD_T_HA), size=n)
    rain = rng.gamma(6.0, 80.0, n)  # mm/season
    anomaly = rng.normal(0, 25, n)
    temp = rng.normal(23, 3.5, n)
    tmax = temp + rng.normal(7, 2, n)
    dry = rng.gamma(2.0, 6.0, n)
    rh = np.clip(rng.normal(62, 14, n), 15, 100)
    ph = np.clip(rng.normal(5.9, 0.8, n), 4.0, 8.5)
    soc = np.clip(rng.gamma(2.5, 0.6, n), 0.2, 6)
    tn = np.clip(soc * 0.09 + rng.normal(0, 0.02, n), 0.01, 0.6)
    p = np.clip(rng.gamma(2.0, 8.0, n), 1, 120)
    ndvi = np.clip(0.25 + rain / 2500 + rng.normal(0, 0.08, n), 0.05, 0.9)
    ndvi_anom = np.clip(anomaly / 200 + rng.normal(0, 0.05, n), -0.5, 0.5)
    evi = np.clip(ndvi * 0.65 + rng.normal(0, 0.03, n), 0.02, 0.8)
    sm = np.clip(rain / 25 + rng.normal(0, 5, n), 2, 60)
    elev = rng.uniform(0, 2600, n)
    n_fert = np.clip(rng.gamma(1.5, 25, n), 0, 200)
    p_fert = np.clip(rng.gamma(1.3, 12, n), 0, 100)
    manure = rng.random(n) < 0.4
    irrigated = rng.random(n) < 0.12
    seed_imp = rng.random(n) < 0.5
    intercrop = rng.random(n) < 0.35
    ext = rng.random(n) < 0.4
    pests = rng.random(n) < 0.3
    delay = np.clip(rng.normal(10, 12, n), -10, 60)

    crop_onehot = np.array(
        [[c == cf.split(".")[1] for cf in CROP_FEATURES] for c in crops]
    )
    X = np.column_stack(
        [
            rain,
            anomaly,
            temp,
            tmax,
            dry,
            rh,
            ph,
            soc,
            tn,
            p,
            ndvi,
            ndvi_anom,
            evi,
            sm,
            elev,
            n_fert,
            p_fert,
            manure,
            irrigated,
            seed_imp,
            intercrop,
            ext,
            pests,
            delay,
            crop_onehot,
        ]
    ).astype(np.float64)
    assert X.shape[1] == len(FEATURES)

    base = np.array([BASE_YIELD_T_HA[c] for c in crops])
    water = np.clip((rain + 300 * irrigated) / 500, 0.3, 1.25)
    acidity = np.where(ph < 5.5, 1 - 0.25 * (5.5 - ph), 1.0)
    yield_t = (
        base * water * acidity
        + 0.010 * n_fert
        + 0.008 * p_fert
        + 0.35 * seed_imp
        + 0.2 * manure
        + 0.15 * ext
        + 1.5 * (ndvi - 0.45)
        - 0.02 * np.maximum(delay, 0)
        - 0.025 * dry
        - 0.6 * pests * (1 - 0.5 * intercrop)
        + rng.normal(0, 0.35 + 0.1 * base, n)
    )
    yield_t = np.clip(yield_t, 0.1, None)

    susceptible = np.isin(crops, ["maize", "sorghum"])
    logit = (
        -1.6
        + 1.9 * pests
        + 0.06 * (temp - 22)
        + 0.02 * (rh - 60)
        - 0.7 * intercrop
        + 0.015 * delay
        + 0.5 * susceptible
    )
    pest = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    risk_score = (
        -anomaly / 20
        + dry / 8
        + (tmax - 30) / 4
        - 1.5 * irrigated
        - sm / 25
        + rng.normal(0, 0.6, n)
    )
    climate_risk = np.digitize(risk_score, [-0.5, 1.2])  # 0 low, 1 moderate, 2 high
    return {
        "X": X,
        "yield_t_ha": yield_t,
        "pest_outbreak": pest,
        "climate_risk": climate_risk,
    }
