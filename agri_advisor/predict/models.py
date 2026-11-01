"""Scaffold predictive models (WP2, D2.4 benchmark models).

Each model wraps a tree ensemble (Random Forest by default, XGBoost optional)
and returns a :class:`PredictionResult` carrying the point estimate, an
uncertainty estimate, an explanation and the model's validation status.

TODO(WP2, D2.5): after training on the harmonised dataset
* tune hyper-parameters with spatial/temporal cross-validation (leave-one-
  county-out / leave-one-season-out) to avoid optimistic leakage;
* report RMSE/MAE/R2 (yield) and AUC/Brier (risk) per country and crop;
* set ``status="validated"`` in the saved metadata only after expert sign-off.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from agri_advisor.predict.explain import explain, summarise
from agri_advisor.predict.features import context_to_vector
from agri_advisor.predict.schema import FEATURES, NUMERIC_FEATURES, FarmContext
from agri_advisor.predict.uncertainty import (
    SplitConformalRegressor,
    Uncertainty,
    classification_uncertainty,
    completeness_penalty,
    regression_uncertainty,
)

logger = logging.getLogger(__name__)

SCAFFOLD_STATUS = "scaffold-synthetic"
VALIDATED_STATUS = "validated"


@dataclass
class PredictionResult:
    task: str
    value: float | str
    unit: str
    uncertainty: Uncertainty
    explanation_method: str
    explanation: list[dict[str, Any]]
    explanation_summary: str
    model_name: str
    model_status: str
    model_version: str
    probabilities: dict[str, float] | None = None
    imputed_features: list[str] = field(default_factory=list)

    @property
    def is_validated(self) -> bool:
        return self.model_status == VALIDATED_STATUS

    def headline(self) -> str:
        if isinstance(self.value, float):
            iv = self.uncertainty.interval
            rng = f" (likely {iv[0]:.1f}-{iv[1]:.1f})" if iv else ""
            return f"{self.task.replace('_', ' ')}: {self.value:.1f} {self.unit}{rng}"
        return f"{self.task.replace('_', ' ')}: {self.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "value": round(self.value, 3)
            if isinstance(self.value, float)
            else self.value,
            "unit": self.unit,
            "headline": self.headline(),
            "probabilities": self.probabilities,
            "uncertainty": self.uncertainty.to_dict(),
            "explanation_method": self.explanation_method,
            "explanation": self.explanation,
            "explanation_summary": self.explanation_summary,
            "model_name": self.model_name,
            "model_status": self.model_status,
            "model_version": self.model_version,
            "imputed_features": self.imputed_features,
        }


class AgriModel:
    """Base class: fit / predict / save / load with metadata."""

    task: str = "base"
    unit: str = ""
    is_classifier: bool = False
    class_labels: list[str] = []

    def __init__(self, backend: str = "random_forest", n_estimators: int = 200) -> None:
        self.backend = backend
        self.n_estimators = n_estimators
        self.estimator: Any = None
        self.meta: dict[str, Any] = {}
        self.medians: dict[str, float] = {}
        self.train_mean = np.zeros(len(FEATURES))
        self.train_std = np.ones(len(FEATURES))
        self.conformal: SplitConformalRegressor | None = None

    # ----------------------------------------------------------- estimators
    def _make_estimator(self) -> Any:
        if self.backend == "xgboost":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise RuntimeError(
                    "pip install xgboost to use backend=xgboost"
                ) from exc
            cls = xgb.XGBClassifier if self.is_classifier else xgb.XGBRegressor
            return cls(n_estimators=self.n_estimators, max_depth=5, learning_rate=0.05)
        cls_rf = RandomForestClassifier if self.is_classifier else RandomForestRegressor
        return cls_rf(
            n_estimators=self.n_estimators,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=0,
        )

    def _tree_outputs(self, x: np.ndarray) -> np.ndarray | None:
        trees = getattr(self.estimator, "estimators_", None)
        if trees is None:
            return None
        if self.is_classifier:
            return np.stack([t.predict_proba(x)[0] for t in trees])
        return np.array([t.predict(x)[0] for t in trees])

    # ----------------------------------------------------------------- train
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        status: str = SCAFFOLD_STATUS,
        data_source: str = "synthetic",
        alpha: float = 0.1,
    ) -> dict[str, Any]:
        """Train, calibrate conformal intervals and record hold-out metrics."""
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        self.estimator = self._make_estimator()
        self.estimator.fit(X_tr, y_tr)
        self.medians = {
            f: float(np.median(X_tr[:, i])) for i, f in enumerate(NUMERIC_FEATURES)
        }
        self.train_mean = X_tr.mean(axis=0)
        self.train_std = X_tr.std(axis=0)

        if self.is_classifier:
            proba = self.estimator.predict_proba(X_cal)
            metrics = {
                "auc_ovr": float(
                    roc_auc_score(
                        y_cal,
                        proba[:, 1] if proba.shape[1] == 2 else proba,
                        multi_class="ovr",
                    )
                ),
                "accuracy": float((proba.argmax(axis=1) == y_cal).mean()),
            }
        else:
            pred = self.estimator.predict(X_cal)
            self.conformal = SplitConformalRegressor(alpha)
            self.conformal.calibrate(y_cal, pred)
            metrics = {
                "mae": float(mean_absolute_error(y_cal, pred)),
                "r2": float(r2_score(y_cal, pred)),
                "conformal_q_hat": self.conformal.q_hat,
            }
        self.meta = {
            "task": self.task,
            "backend": self.backend,
            "status": status,
            "data_source": data_source,
            "n_train": int(len(y_tr)),
            "n_calibration": int(len(y_cal)),
            "metrics_holdout": metrics,
            "features": FEATURES,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        }
        return self.meta

    # --------------------------------------------------------------- predict
    def predict(self, ctx: FarmContext) -> PredictionResult:
        if self.estimator is None:
            raise RuntimeError(f"{self.task} model is not trained/loaded")
        x, imputed = context_to_vector(ctx, self.medians)
        frac, _ = completeness_penalty(len(imputed), len(NUMERIC_FEATURES))
        tree_out = self._tree_outputs(x)

        probabilities = None
        class_index = None
        if self.is_classifier:
            proba = self.estimator.predict_proba(x)[0]
            class_index = int(np.argmax(proba))
            value: float | str = self.class_labels[class_index]
            probabilities = {
                lbl: round(float(p), 3)
                for lbl, p in zip(self.class_labels, proba, strict=False)
            }
            unc = classification_uncertainty(proba, tree_out, frac)
        else:
            value = float(self.estimator.predict(x)[0])
            unc = regression_uncertainty(
                value,
                tree_out if tree_out is not None else np.array([value]),
                self.conformal,
                frac,
            )
        method, contribs = explain(
            self.estimator, x, self.train_mean, self.train_std, imputed, class_index
        )
        return PredictionResult(
            task=self.task,
            value=value,
            unit=self.unit,
            uncertainty=unc,
            explanation_method=method,
            explanation=[c.to_dict() for c in contribs],
            explanation_summary=summarise(contribs),
            model_name=f"{self.backend}:{self.task}",
            model_status=self.meta.get("status", SCAFFOLD_STATUS),
            model_version=self.meta.get("version", "unknown"),
            probabilities=probabilities,
            imputed_features=imputed,
        )

    # ----------------------------------------------------------- persistence
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.task}.joblib"
        joblib.dump(
            {
                "estimator": self.estimator,
                "medians": self.medians,
                "train_mean": self.train_mean,
                "train_std": self.train_std,
                "conformal": self.conformal,
                "meta": self.meta,
                "backend": self.backend,
            },
            path,
        )
        (directory / f"{self.task}.meta.json").write_text(
            json.dumps(self.meta, indent=1)
        )
        return path

    @classmethod
    def load(cls, directory: Path) -> AgriModel:
        # Only load model files you produced yourself: joblib uses pickle.
        blob = joblib.load(directory / f"{cls.task}.joblib")
        obj = cls(backend=blob["backend"])
        obj.estimator = blob["estimator"]
        obj.medians = blob["medians"]
        obj.train_mean = blob["train_mean"]
        obj.train_std = blob["train_std"]
        obj.conformal = blob["conformal"]
        obj.meta = blob["meta"]
        return obj

    @classmethod
    def exists(cls, directory: Path) -> bool:
        return (directory / f"{cls.task}.joblib").exists()


class CropYieldModel(AgriModel):
    """Seasonal grain yield (t/ha) for maize, wheat, sorghum, rice, millet.

    TODO(D2.3): target = measured yield from crop cuts / farm surveys;
    features = climate + soil + NDVI/EVI time-series aggregates + management.
    Consider a hybrid CNN-LSTM over NDVI time series once data volume allows.
    """

    task = "crop_yield"
    unit = "t/ha"


class PestRiskModel(AgriModel):
    """Probability of an economically damaging pest/disease outbreak this season.

    TODO(D2.3): labels from surveillance data (e.g. FAW monitoring, PlantVillage,
    KALRO/KEPHIS pest reports); add lagged weather and neighbourhood incidence.
    """

    task = "pest_risk"
    is_classifier = True
    class_labels = ["low", "high"]


class ClimateRiskModel(AgriModel):
    """Seasonal crop water-stress / climate risk class (low, moderate, high).

    TODO(D2.2): derive labels from historical drought indices (SPI/WRSI) and
    yield shocks; feed seasonal forecasts (ICPAC/met services) as inputs.
    """

    task = "climate_risk"
    is_classifier = True
    class_labels = ["low", "moderate", "high"]


ALL_MODELS: list[type[AgriModel]] = [CropYieldModel, PestRiskModel, ClimateRiskModel]
