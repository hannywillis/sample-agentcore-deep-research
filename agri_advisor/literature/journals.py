"""Catalogue of open agricultural journals and repositories.

Two layers:

1. ``CURATED`` - hand-picked open-access journals / repositories most relevant
   to African cereal and livestock systems. URLs are public homepages; the
   list is a starting point for curators, not an endorsement.
2. ``doaj_journals()`` - live query of the Directory of Open Access Journals
   (~21k fully OA, peer-reviewed journals) filtered by subject keywords, so the
   catalogue grows with DOAJ rather than being frozen in code.
"""

from __future__ import annotations

from typing import Any

import requests

from agri_advisor.literature.sources import _get

CURATED: list[dict[str, Any]] = [
    # Journals
    {"name": "African Journal of Agricultural Research", "type": "journal",
     "url": "https://academicjournals.org/journal/AJAR", "region": "Africa"},
    {"name": "African Journal of Food, Agriculture, Nutrition and Development (AJFAND)",
     "type": "journal", "url": "https://www.ajfand.net", "region": "Africa"},
    {"name": "African Crop Science Journal", "type": "journal",
     "url": "https://www.ajol.info/index.php/acsj", "region": "Africa"},
    {"name": "Livestock Research for Rural Development", "type": "journal",
     "url": "http://www.lrrd.org", "region": "Global (tropics)"},
    {"name": "CABI Agriculture and Bioscience", "type": "journal",
     "url": "https://cabiagbio.biomedcentral.com", "region": "Global"},
    {"name": "Frontiers in Sustainable Food Systems", "type": "journal",
     "url": "https://www.frontiersin.org/journals/sustainable-food-systems", "region": "Global"},
    {"name": "Agriculture & Food Security", "type": "journal",
     "url": "https://agricultureandfoodsecurity.biomedcentral.com", "region": "Global"},
    {"name": "Pesquisa Agropecuária Brasileira (Embrapa)", "type": "journal",
     "url": "https://seer.sct.embrapa.br/index.php/pab", "region": "Brazil"},
    {"name": "Smart Agricultural Technology", "type": "journal",
     "url": "https://www.sciencedirect.com/journal/smart-agricultural-technology",
     "region": "Global"},
    # Repositories / indexes
    {"name": "African Journals Online (AJOL)", "type": "index",
     "url": "https://www.ajol.info", "region": "Africa"},
    {"name": "CGSpace (CGIAR: ILRI, CIMMYT, IITA, ICRISAT, Alliance)", "type": "repository",
     "url": "https://cgspace.cgiar.org", "region": "Global", "connector": "cgspace"},
    {"name": "FAO AGRIS", "type": "index", "url": "https://agris.fao.org", "region": "Global"},
    {"name": "AgEcon Search", "type": "repository", "url": "https://ageconsearch.umn.edu",
     "region": "Global"},
    {"name": "KALRO Knowledge Repository", "type": "repository",
     "url": "https://kalro.org", "region": "Kenya"},
    {"name": "Embrapa Alice / Infoteca-e", "type": "repository",
     "url": "https://www.alice.cnptia.embrapa.br", "region": "Brazil"},
    {"name": "Directory of Open Access Journals (DOAJ)", "type": "index",
     "url": "https://doaj.org", "region": "Global", "connector": "doaj"},
    {"name": "Europe PMC", "type": "index", "url": "https://europepmc.org",
     "region": "Global", "connector": "europepmc"},
    {"name": "OpenAlex", "type": "index", "url": "https://openalex.org",
     "region": "Global", "connector": "openalex"},
    {"name": "CORE", "type": "index", "url": "https://core.ac.uk", "region": "Global",
     "connector": "core"},
    {"name": "Zenodo", "type": "repository", "url": "https://zenodo.org",
     "region": "Global", "connector": "zenodo"},
    {"name": "Research4Life (AGORA / Hinari / OARE) - free/low-cost access for eligible "
     "institutions", "type": "access-programme", "url": "https://www.research4life.org",
     "region": "Low/middle-income countries"},
]

DEFAULT_SUBJECTS = (
    "agriculture", "agronomy", "animal science", "crop science", "soil science",
    "plant pathology", "food security", "veterinary", "agricultural economics",
)


def doaj_journals(subject: str = "agriculture", limit: int = 50, page: int = 1) -> list[dict]:
    """Fully open-access journals in DOAJ whose metadata match ``subject``."""
    data = _get(
        f"https://doaj.org/api/search/journals/{requests.utils.quote(subject)}",
        {"pageSize": min(limit, 100), "page": page},
    ).json()
    out = []
    for r in data.get("results", []):
        b = r.get("bibjson", {})
        out.append({
            "name": b.get("title"),
            "type": "journal",
            "publisher": (b.get("publisher") or {}).get("name"),
            "country": (b.get("publisher") or {}).get("country"),
            "issn": b.get("eissn") or b.get("pissn"),
            "url": (b.get("ref") or {}).get("journal"),
            "subjects": [s.get("term") for s in b.get("subject", [])][:6],
            "languages": b.get("language", []),
            "apc": bool((b.get("apc") or {}).get("has_apc")),
            "license": [lc.get("type") for lc in b.get("license", [])],
            "source": "doaj",
        })
    return out


def catalogue(subjects: list[str] | None = None, live: bool = True,
              per_subject: int = 25, region: str | None = None) -> dict[str, Any]:
    """Curated list + (optionally) live DOAJ journals for each subject."""
    curated = [c for c in CURATED if not region or region.lower() in c["region"].lower()]
    result: dict[str, Any] = {"curated": curated, "doaj": [], "errors": {}}
    if not live:
        return result
    seen: set[str] = set()
    for s in subjects or list(DEFAULT_SUBJECTS[:4]):
        try:
            for j in doaj_journals(s, per_subject):
                key = j.get("issn") or j.get("name")
                if key and key not in seen:
                    seen.add(key)
                    result["doaj"].append(j)
        except Exception as exc:  # noqa: BLE001
            result["errors"][s] = f"{type(exc).__name__}: {exc}"[:200]
    return result
