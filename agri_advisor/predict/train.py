"""Train predictive models.

Usage::

    # Retrain on REAL open data (FAOSTAT yields + NASA POWER climate)
    python -m agri_advisor.predict.open_data        # build/refresh the CSV
    python -m agri_advisor.predict.train --open-data [--backend xgboost]

    # Scaffold models on synthetic data (pipeline testing only)
    python -m agri_advisor.predict.train --synthetic

    # Partner data: CSV with one row per farm-season, columns = FEATURES names
    # (e.g. "climate.season_rainfall_mm") + "crop" or "crop.<name>" one-hot +
    # targets "yield_t_ha", "pest_outbreak" (0/1), "climate_risk" (0/1/2);
    # optional "year" column enables temporal validation (--test-start).
    python -m agri_advisor.predict.train --data data/farm_seasons.csv --test-start 2020

Only set ``--status validated`` after hold-out evaluation and expert sign-off.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from agri_advisor.config import get_settings
from agri_advisor.predict import synthetic
from agri_advisor.predict.models import (
    ALL_MODELS,
    OPEN_DATA_STATUS,
    SCAFFOLD_STATUS,
    PestRiskModel,
)
from agri_advisor.predict.open_data import OUT_CSV
from agri_advisor.predict.schema import CROP_FEATURES, FEATURES, NUMERIC_FEATURES
from agri_advisor.predict.service import _TARGETS, train_scaffold_models

OPEN_DATA_NOTES = (
    "National FAOSTAT yields + NASA POWER season climate at one cropping point "
    "per country (20 African countries). Ecological-scale model: farm-level "
    "management, soil and NDVI features were not observed and do not influence "
    "predictions. Not validated against farm data (D2.3/D2.5)."
)


def load_csv(path: Path) -> dict[str, Any]:
    """Load a training CSV into X (FEATURES order), targets, years, used features."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    cols = set(rows[0])
    used = [f for f in NUMERIC_FEATURES if f in cols]
    if not used:
        raise ValueError("CSV contains none of the model feature columns")

    def col(name: str) -> np.ndarray:
        return np.array(
            [float(r[name]) if r.get(name, "") != "" else np.nan for r in rows]
        )

    blocks = []
    for f in NUMERIC_FEATURES:
        if f in cols:
            v = col(f)
            blocks.append(np.where(np.isnan(v), np.nanmedian(v), v))
        else:  # unobserved -> constant, so the model cannot use it
            blocks.append(np.zeros(len(rows)))
    for cf in CROP_FEATURES:
        if cf in cols:
            blocks.append(col(cf))
        else:
            blocks.append(
                np.array([1.0 if r.get("crop") == cf.split(".")[1] else 0.0 for r in rows])
            )
    X = np.column_stack(blocks)
    assert X.shape[1] == len(FEATURES)
    out: dict[str, Any] = {"X": X, "used": used + CROP_FEATURES, "n": len(rows)}
    for target in _TARGETS.values():
        if target in cols:
            out[target] = col(target)
    if "year" in cols:
        out["years"] = col("year").astype(int)
    return out


def train_from_csv(
    path: Path,
    backend: str = "random_forest",
    status: str = OPEN_DATA_STATUS,
    test_start: int | None = None,
    notes: str = "",
    pest_fallback_synthetic: bool = True,
) -> dict[str, Any]:
    s = get_settings()
    data = load_csv(path)
    years = data.get("years")
    if years is not None and test_start is None:
        # last ~20 % of seasons for out-of-time testing / calibration
        test_start = int(np.quantile(years, 0.8))
    prev_idx = FEATURES.index("farm.previous_yield_t_ha")
    report: dict[str, Any] = {}
    for cls in ALL_MODELS:
        target = _TARGETS[cls.task]
        anchored = cls.task == "crop_yield" and "farm.previous_yield_t_ha" in data["used"]
        model = cls(
            backend=backend,
            min_samples_leaf=30 if anchored else 10,
            anchor_feature="farm.previous_yield_t_ha" if anchored else None,
        )
        if target in data:
            y = data[target] if cls.task == "crop_yield" else data[target].astype(int)
            report[cls.task] = model.fit(
                data["X"],
                y,
                status=status,
                data_source=str(path.name),
                alpha=s.conformal_alpha,
                years=years,
                test_start=test_start,
                features_used=data["used"],
                baseline_pred=data["X"][:, prev_idx]
                if anchored
                else None,
                notes=notes,
            )
        elif cls is PestRiskModel and pest_fallback_synthetic:
            syn = synthetic.generate()
            report[cls.task] = model.fit(
                syn["X"],
                syn[target],
                alpha=s.conformal_alpha,
                notes="No open pest-surveillance labels yet - synthetic scaffold.",
            )
        else:
            continue
        model.save(s.model_dir)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train agri_advisor predictive models")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--open-data", action="store_true", help=f"use {OUT_CSV.name}")
    g.add_argument("--data", type=Path)
    p.add_argument(
        "--backend", default="random_forest", choices=["random_forest", "xgboost"]
    )
    p.add_argument("--status", default=None)
    p.add_argument("--test-start", type=int, default=None)
    args = p.parse_args(argv)

    if args.open_data:
        if not OUT_CSV.exists():
            raise SystemExit("Run `python -m agri_advisor.predict.open_data` first")
        report = train_from_csv(
            OUT_CSV, args.backend, args.status or OPEN_DATA_STATUS,
            args.test_start, OPEN_DATA_NOTES,
        )
    elif args.data:
        report = train_from_csv(
            args.data, args.backend, args.status or SCAFFOLD_STATUS, args.test_start
        )
    else:
        report = train_scaffold_models(get_settings())
    print(  # noqa: T201
        json.dumps(
            {
                k: {
                    "status": v["status"],
                    "split": v.get("split"),
                    "n_train": v["n_train"],
                    "n_test": v["n_calibration"],
                    "metrics": v["metrics_holdout"],
                }
                for k, v in report.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
