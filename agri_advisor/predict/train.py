"""Train predictive models.

Usage::

    # Scaffold models on synthetic data (pipeline testing only)
    python -m agri_advisor.predict.train --synthetic

    # Real data: CSV with one row per farm-season, columns = FEATURES names
    # (e.g. "climate.season_rainfall_mm", "crop.maize") + target columns
    # "yield_t_ha", "pest_outbreak" (0/1), "climate_risk" (0/1/2)
    python -m agri_advisor.predict.train --data data/farm_seasons.csv \
        --backend xgboost --status scaffold-synthetic

Only set ``--status validated`` after hold-out evaluation and expert sign-off.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from agri_advisor.config import get_settings
from agri_advisor.predict.models import ALL_MODELS, SCAFFOLD_STATUS
from agri_advisor.predict.schema import FEATURES
from agri_advisor.predict.service import _TARGETS, train_scaffold_models


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    """TODO(D2.1): plug the harmonised multi-country dataset loader in here."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = [f for f in FEATURES if f not in rows[0]]
    if missing:
        raise ValueError(f"CSV is missing feature columns: {missing}")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) if r[name] != "" else np.nan for r in rows])

    X = np.column_stack([col(f) for f in FEATURES])
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    out = {"X": X}
    for target in _TARGETS.values():
        if target in rows[0]:
            out[target] = col(target)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train agri_advisor predictive models")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--data", type=Path)
    p.add_argument(
        "--backend", default="random_forest", choices=["random_forest", "xgboost"]
    )
    p.add_argument("--status", default=SCAFFOLD_STATUS)
    args = p.parse_args(argv)
    s = get_settings()

    if args.synthetic or not args.data:
        report = train_scaffold_models(s)
    else:
        data = _load_csv(args.data)
        report = {}
        for cls in ALL_MODELS:
            target = _TARGETS[cls.task]
            if target not in data:
                continue
            model = cls(backend=args.backend)
            y = data[target] if cls.task == "crop_yield" else data[target].astype(int)
            report[cls.task] = model.fit(
                data["X"],
                y,
                status=args.status,
                data_source=str(args.data),
                alpha=s.conformal_alpha,
            )
            model.save(s.model_dir)
    print(
        json.dumps(  # noqa: T201
            {
                k: {"status": v["status"], "metrics": v["metrics_holdout"]}
                for k, v in report.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
