"""Build a REAL training dataset from open data sources (WP2, D2.1 / D2.2).

Sources
-------
* **FAOSTAT QCL** (bulk download, CC BY 4.0) - national yields (kg/ha) for
  maize, wheat, sorghum, rice and millet, 1961-latest.
* **NASA POWER** daily agro-climatology API (public, no key) - precipitation
  (CHIRPS-corrected ``PRECTOTCORR``), mean / max temperature and relative
  humidity at a representative cropping point per country, 1981-latest.

Output: one row per (country, crop, harvest year) with the model feature names
from :mod:`agri_advisor.predict.schema` plus targets:

* ``yield_t_ha``      - FAOSTAT national yield (t/ha)
* ``climate_risk``    - 0/1/2 production-shock class: yield vs the mean of the
  previous three years (>= -5 % low, -5 to -15 % moderate, < -15 % high)
* ``year``, ``country`` - used for temporal validation splits

LIMITATIONS (stated in model metadata and README):
* National averages and one climate point per country - an ecological-scale
  model. It captures climate/season and recent-yield signals, NOT farm-level
  management effects (those features are absent and imputed at training).
* Production shocks can also have non-climate causes (conflict, prices, pests).
* Pest-outbreak labels have no open equivalent, so the pest model stays on
  synthetic data until surveillance data (KALRO/KEPHIS, FAW monitoring) arrive.

Usage::

    python -m agri_advisor.predict.open_data            # download + build CSV
    python -m agri_advisor.predict.train --open-data    # retrain models
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import time
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests

from agri_advisor.config import REPO_ROOT

logger = logging.getLogger(__name__)

FAOSTAT_URL = (
    "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_Africa.zip"
)
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
RAW_DIR = REPO_ROOT / "data" / "open_data" / "raw"
OUT_CSV = REPO_ROOT / "data" / "open_data" / "harmonised_country_season.csv"

FAO_ITEMS = {
    "Maize (corn)": "maize",
    "Wheat": "wheat",
    "Sorghum": "sorghum",
    "Rice": "rice",
    "Millet": "millet",
}

# ISO2: (FAOSTAT area code, lat, lon, main-season start month, n months,
#        season starts in the year BEFORE harvest?)
# Points sit in each country's main cereal belt; months follow FAO GIEWS crop
# calendars for the main season (approximate - refine with partners, D2.1).
COUNTRIES: dict[str, tuple[int, float, float, int, int, bool]] = {
    "KE": (114, 0.9, 35.1, 3, 6, False),  # Trans Nzoia / Uasin Gishu long rains
    "UG": (226, 0.7, 33.6, 3, 5, False),
    "TZ": (215, -7.8, 35.7, 12, 5, True),  # southern highlands unimodal
    "ET": (238, 9.0, 37.5, 6, 4, False),  # meher
    "RW": (184, -2.0, 29.8, 3, 4, False),
    "NE": (158, 13.5, 7.1, 6, 4, False),  # Maradi
    "NG": (159, 11.0, 7.7, 5, 5, False),  # Kaduna / Zaria
    "GH": (81, 7.6, -1.9, 4, 4, False),
    "BF": (233, 12.3, -1.5, 6, 4, False),
    "ML": (133, 13.0, -6.5, 6, 4, False),
    "SN": (195, 14.0, -15.5, 7, 4, False),
    "TD": (39, 11.0, 17.0, 6, 4, False),
    "CM": (32, 7.3, 13.5, 4, 6, False),
    "BJ": (53, 9.5, 2.5, 5, 5, False),
    "TG": (217, 9.0, 1.0, 5, 5, False),
    "CI": (107, 8.0, -5.5, 4, 4, False),
    "ZM": (251, -14.5, 28.0, 11, 5, True),
    "MW": (130, -13.5, 33.8, 11, 5, True),
    "ZW": (181, -18.0, 31.0, 11, 5, True),
    "MZ": (144, -15.0, 37.0, 11, 5, True),
}
FIRST_YEAR, BASELINE = 1982, (1991, 2020)


# --------------------------------------------------------------------- FAOSTAT
def load_faostat(raw_dir: Path = RAW_DIR) -> dict[tuple[str, str, int], float]:
    """Return {(iso2, crop, year): yield_t_ha} for the configured countries."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zpath = raw_dir / "faostat_qcl_africa.zip"
    if not zpath.exists():
        logger.info("Downloading FAOSTAT bulk file ...")
        resp = requests.get(FAOSTAT_URL, timeout=300)
        resp.raise_for_status()
        zpath.write_bytes(resp.content)
    code_to_iso = {v[0]: k for k, v in COUNTRIES.items()}
    out: dict[tuple[str, str, int], float] = {}
    with zipfile.ZipFile(zpath) as zf:
        name = next(n for n in zf.namelist() if n.endswith("Africa_NOFLAG.csv"))
        with zf.open(name) as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="latin-1"))
            for row in reader:
                if row["Element"] != "Yield" or row["Item"] not in FAO_ITEMS:
                    continue
                iso = code_to_iso.get(int(row["Area Code"]))
                if not iso:
                    continue
                for key, val in row.items():
                    if key.startswith("Y") and key[1:].isdigit() and val:
                        kg = float(val)
                        if kg > 0:
                            out[(iso, FAO_ITEMS[row["Item"]], int(key[1:]))] = kg / 1000
    return out


# ------------------------------------------------------------------ NASA POWER
def load_power(iso: str, last_year: int, raw_dir: Path = RAW_DIR) -> dict:
    """Daily PRECTOTCORR/T2M/T2M_MAX/RH2M for a country point (cached JSON)."""
    _, lat, lon, *_ = COUNTRIES[iso]
    path = raw_dir / f"power_{iso}_{FIRST_YEAR - 1}_{last_year}.json"
    if not path.exists():
        params = {
            "parameters": "PRECTOTCORR,T2M,T2M_MAX,RH2M",
            "community": "AG",
            "latitude": lat,
            "longitude": lon,
            "start": f"{FIRST_YEAR - 1}0101",
            "end": f"{last_year}1231",
            "format": "JSON",
        }
        for attempt in range(3):
            try:
                resp = requests.get(POWER_URL, params=params, timeout=180)
                resp.raise_for_status()
                path.write_text(resp.text)
                break
            except requests.RequestException as exc:
                logger.warning("POWER %s attempt %d failed: %s", iso, attempt + 1, exc)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"NASA POWER download failed for {iso}")
        time.sleep(1)  # be polite to the public API
    return json.loads(path.read_text())


def _season_days(iso: str, harvest_year: int) -> list[str]:
    *_, start_m, n_months, prev = COUNTRIES[iso]
    y = harvest_year - 1 if prev else harvest_year
    d = date(y, start_m, 1)
    m_end = start_m + n_months
    end = date(y + (m_end - 1) // 12, (m_end - 1) % 12 + 1, 1)
    days = []
    while d < end:
        days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return days


def season_climate(power: dict, iso: str, year: int) -> dict[str, float] | None:
    p = power["properties"]["parameter"]
    days = _season_days(iso, year)
    rain = [p["PRECTOTCORR"].get(d, -999) for d in days]
    if not rain or min(rain) < 0:  # missing / fill values (-999)
        return None
    longest = run = 0
    for r in rain:
        run = run + 1 if r < 1.0 else 0
        longest = max(longest, run)

    def mean(key: str) -> float:
        vals = [p[key][d] for d in days if p[key].get(d, -999) > -900]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "climate.season_rainfall_mm": float(sum(rain)),
        "climate.mean_temp_c": mean("T2M"),
        "climate.max_temp_c": mean("T2M_MAX"),
        "climate.dry_spell_days": float(longest),
        "climate.relative_humidity_pct": mean("RH2M"),
        "location.elevation_m": float(power["geometry"]["coordinates"][2]),
    }


# ----------------------------------------------------------------------- build
def build(out_csv: Path = OUT_CSV, last_year: int | None = None) -> Path:
    yields = load_faostat()
    last_year = last_year or min(max(y for *_, y in yields), date.today().year - 1)
    rows: list[dict] = []
    for iso in COUNTRIES:
        power = load_power(iso, last_year)
        clim = {y: season_climate(power, iso, y) for y in range(FIRST_YEAR, last_year + 1)}
        base = [
            c["climate.season_rainfall_mm"]
            for y, c in clim.items()
            if c and BASELINE[0] <= y <= BASELINE[1]
        ]
        base_mean = float(np.mean(base)) if base else float("nan")
        by_crop: dict[str, dict[int, float]] = defaultdict(dict)
        for (i, crop, y), v in yields.items():
            if i == iso:
                by_crop[crop][y] = v
        for crop, series in by_crop.items():
            for y in range(FIRST_YEAR, last_year + 1):
                prev = [series.get(y - k) for k in (1, 2, 3)]
                if y not in series or clim.get(y) is None or None in prev:
                    continue
                prev_mean = float(np.mean(prev))
                shock = series[y] / prev_mean - 1
                c = dict(clim[y])  # type: ignore[arg-type]
                c["climate.rainfall_anomaly_pct"] = (
                    100 * (c["climate.season_rainfall_mm"] / base_mean - 1)
                )
                rows.append(
                    {
                        "country": iso,
                        "year": y,
                        "crop": crop,
                        **{k: round(v, 3) for k, v in c.items()},
                        "farm.previous_yield_t_ha": round(prev_mean, 4),
                        "yield_t_ha": round(series[y], 4),
                        "climate_risk": 2 if shock < -0.15 else 1 if shock < -0.05 else 0,
                    }
                )
    if not rows:
        raise RuntimeError("No rows built - check downloads")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), out_csv)
    return out_csv


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build open-data training set")
    p.add_argument("--out", type=Path, default=OUT_CSV)
    p.add_argument("--last-year", type=int)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(build(args.out, args.last_year))  # noqa: T201


if __name__ == "__main__":
    main()
