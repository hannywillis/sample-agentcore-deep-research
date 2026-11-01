"""Structured input schema for predictive intelligence (WP2, D2.1 / D2.2).

The schema harmonises the four data families named in the proposal:

* **Climate** - seasonal rainfall, temperature, dry spells, forecasts
  (e.g. CHIRPS, ERA5, national met services)
* **Soil** - pH, organic carbon, N/P/K, texture (e.g. iSDAsoil, KALRO soil labs)
* **Remote sensing** - NDVI/EVI, land-surface temperature, soil moisture
  (e.g. Sentinel-2, MODIS, SMAP)
* **Farm / management** - crop, variety, planting date, inputs, extension access
  (farm surveys, ground-truth plots - D2.3)

Every field is optional: missing values are imputed with training medians and
the fraction of imputed features lowers the confidence attached to a prediction.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Crop = Literal["maize", "wheat", "sorghum", "rice", "millet"]


class Location(BaseModel):
    country: str | None = Field(None, description="ISO-3166 alpha-2, e.g. KE")
    admin1: str | None = Field(None, description="County / state / region")
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    elevation_m: float | None = Field(None, ge=-100, le=6000)
    agroecological_zone: str | None = None


class ClimateData(BaseModel):
    season_rainfall_mm: float | None = Field(None, ge=0)
    rainfall_anomaly_pct: float | None = Field(
        None, description="% deviation from long-term mean"
    )
    mean_temp_c: float | None = None
    max_temp_c: float | None = None
    dry_spell_days: float | None = Field(None, ge=0, description="Longest dry spell")
    relative_humidity_pct: float | None = Field(None, ge=0, le=100)
    forecast_rainfall_mm: float | None = Field(
        None, ge=0, description="Seasonal forecast total for the coming season"
    )
    rainfall_onset_date: date | None = None


class SoilData(BaseModel):
    ph: float | None = Field(None, ge=2, le=11)
    organic_carbon_pct: float | None = Field(None, ge=0, le=20)
    total_nitrogen_pct: float | None = Field(None, ge=0, le=2)
    phosphorus_ppm: float | None = Field(None, ge=0)
    potassium_ppm: float | None = Field(None, ge=0)
    clay_pct: float | None = Field(None, ge=0, le=100)
    sand_pct: float | None = Field(None, ge=0, le=100)


class RemoteSensingData(BaseModel):
    ndvi_mean: float | None = Field(None, ge=-1, le=1)
    ndvi_anomaly: float | None = Field(None, ge=-1, le=1)
    evi_mean: float | None = Field(None, ge=-1, le=1)
    land_surface_temp_c: float | None = None
    soil_moisture_pct: float | None = Field(None, ge=0, le=100)


class FarmData(BaseModel):
    crop: Crop = "maize"
    variety_maturity: Literal["early", "medium", "late"] | None = None
    planting_date: date | None = None
    area_ha: float | None = Field(None, gt=0)
    fertilizer_n_kg_ha: float | None = Field(None, ge=0)
    fertilizer_p_kg_ha: float | None = Field(None, ge=0)
    manure_applied: bool | None = None
    irrigated: bool | None = None
    improved_seed: bool | None = None
    intercropped_legume: bool | None = None
    extension_access: bool | None = None
    previous_yield_t_ha: float | None = Field(None, ge=0)
    pest_sightings: bool | None = None


class FarmContext(BaseModel):
    """Everything known about a farm/season when a farmer asks for advice."""

    location: Location = Field(default_factory=Location)
    climate: ClimateData = Field(default_factory=ClimateData)
    soil: SoilData = Field(default_factory=SoilData)
    remote_sensing: RemoteSensingData = Field(default_factory=RemoteSensingData)
    farm: FarmData = Field(default_factory=FarmData)

    def describe(self) -> str:
        """Compact text rendering used inside LLM prompts."""
        parts: list[str] = []
        for section in ("location", "farm", "climate", "soil", "remote_sensing"):
            values = {
                k: v
                for k, v in getattr(self, section).model_dump().items()
                if v is not None
            }
            if values:
                parts.append(
                    f"{section}: " + ", ".join(f"{k}={v}" for k, v in values.items())
                )
        return "; ".join(parts) or "(no farm context provided)"


# Ordered numeric feature vector used by all scaffold models.
NUMERIC_FEATURES: list[str] = [
    "climate.season_rainfall_mm",
    "climate.rainfall_anomaly_pct",
    "climate.mean_temp_c",
    "climate.max_temp_c",
    "climate.dry_spell_days",
    "climate.relative_humidity_pct",
    "soil.ph",
    "soil.organic_carbon_pct",
    "soil.total_nitrogen_pct",
    "soil.phosphorus_ppm",
    "remote_sensing.ndvi_mean",
    "remote_sensing.ndvi_anomaly",
    "remote_sensing.evi_mean",
    "remote_sensing.soil_moisture_pct",
    "location.elevation_m",
    "farm.fertilizer_n_kg_ha",
    "farm.fertilizer_p_kg_ha",
    "farm.manure_applied",
    "farm.irrigated",
    "farm.improved_seed",
    "farm.intercropped_legume",
    "farm.extension_access",
    "farm.pest_sightings",
    "farm.planting_delay_days",
]
CROP_FEATURES = [f"crop.{c}" for c in ("maize", "wheat", "sorghum", "rice", "millet")]
FEATURES: list[str] = NUMERIC_FEATURES + CROP_FEATURES

FEATURE_LABELS: dict[str, str] = {
    "climate.season_rainfall_mm": "seasonal rainfall",
    "climate.rainfall_anomaly_pct": "rainfall compared with normal",
    "climate.mean_temp_c": "average temperature",
    "climate.max_temp_c": "maximum temperature",
    "climate.dry_spell_days": "length of dry spells",
    "climate.relative_humidity_pct": "air humidity",
    "soil.ph": "soil acidity (pH)",
    "soil.organic_carbon_pct": "soil organic matter",
    "soil.total_nitrogen_pct": "soil nitrogen",
    "soil.phosphorus_ppm": "soil phosphorus",
    "remote_sensing.ndvi_mean": "crop greenness from satellite (NDVI)",
    "remote_sensing.ndvi_anomaly": "crop greenness compared with normal",
    "remote_sensing.evi_mean": "vegetation index (EVI)",
    "remote_sensing.soil_moisture_pct": "soil moisture",
    "location.elevation_m": "altitude",
    "farm.fertilizer_n_kg_ha": "nitrogen fertilizer used",
    "farm.fertilizer_p_kg_ha": "phosphorus fertilizer used",
    "farm.manure_applied": "manure use",
    "farm.irrigated": "irrigation",
    "farm.improved_seed": "certified/improved seed",
    "farm.intercropped_legume": "legume intercropping",
    "farm.extension_access": "access to extension advice",
    "farm.pest_sightings": "pests seen in the field",
    "farm.planting_delay_days": "days planted after rain onset",
}
