"""Facade that loads (or bootstraps) all predictive models."""

from __future__ import annotations

import logging
from typing import Any

from agri_advisor.config import Settings, get_settings
from agri_advisor.predict import synthetic
from agri_advisor.predict.advisors import planting_window, soil_management
from agri_advisor.predict.models import ALL_MODELS, AgriModel, PredictionResult
from agri_advisor.predict.schema import FarmContext

logger = logging.getLogger(__name__)

_TARGETS = {
    "crop_yield": "yield_t_ha",
    "pest_risk": "pest_outbreak",
    "climate_risk": "climate_risk",
}


def train_scaffold_models(settings: Settings | None = None) -> dict[str, Any]:
    """Train every model on SYNTHETIC data and save it (clearly flagged)."""
    s = settings or get_settings()
    data = synthetic.generate()
    report = {}
    for cls in ALL_MODELS:
        model = cls()
        report[cls.task] = model.fit(
            data["X"], data[_TARGETS[cls.task]], alpha=s.conformal_alpha
        )
        model.save(s.model_dir)
    return report


class PredictiveService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.models: dict[str, AgriModel] = {}
        self._load()

    def _load(self) -> None:
        d = self.settings.model_dir
        missing = [cls for cls in ALL_MODELS if not cls.exists(d)]
        if missing and self.settings.auto_train_scaffold:
            logger.warning(
                "Models missing in %s - training SCAFFOLD models on synthetic data", d
            )
            train_scaffold_models(self.settings)
        for cls in ALL_MODELS:
            if cls.exists(d):
                self.models[cls.task] = cls.load(d)

    @property
    def available(self) -> bool:
        return bool(self.models)

    def predict(
        self, ctx: FarmContext, tasks: list[str] | None = None
    ) -> dict[str, Any]:
        """Run ML models + rule-based advisors for one farm context."""
        results: dict[str, PredictionResult] = {}
        for task, model in self.models.items():
            if tasks and task not in tasks:
                continue
            results[task] = model.predict(ctx)
        return {
            "models": results,
            "planting_window": planting_window(ctx),
            "soil_management": soil_management(ctx),
        }
