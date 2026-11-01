"""Federated literature search: parallel queries, de-duplication, ranking."""

from __future__ import annotations

import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from agri_advisor.literature.models import Paper
from agri_advisor.literature.sources import DEFAULT_SOURCES, SOURCES

logger = logging.getLogger(__name__)

ENV_KEYS = {
    "openalex_api_key": "OPENALEX_API_KEY",
    "core_api_key": "CORE_API_KEY",
    "semantic_scholar_api_key": "SEMANTIC_SCHOLAR_API_KEY",
    "contact_email": "LITERATURE_CONTACT_EMAIL",
    "unpaywall_email": "UNPAYWALL_EMAIL",
}


def env_keys() -> dict[str, str]:
    return {k: os.environ[v] for k, v in ENV_KEYS.items() if os.environ.get(v)}


@dataclass
class SearchResult:
    query: str
    papers: list[Paper]
    source_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_raw: int = 0

    @property
    def n_duplicates(self) -> int:
        return self.n_raw - len(self.papers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "n_raw": self.n_raw,
            "n_unique": len(self.papers),
            "n_duplicates_removed": self.n_duplicates,
            "source_status": self.source_status,
            "papers": [p.to_dict() for p in self.papers],
        }


def _terms(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]


def _relevance(p: Paper, terms: list[str]) -> float:
    title = (p.title or "").lower()
    body = (p.abstract or "").lower() + " " + " ".join(p.keywords).lower()
    if not terms:
        return 0.0
    t_hit = sum(t in title for t in terms) / len(terms)
    b_hit = sum(t in body for t in terms) / len(terms)
    cites = math.log1p(p.cited_by or 0) / 10
    recency = 0.0
    if p.year:
        recency = max(0.0, 1 - (date.today().year - p.year) / 30) * 0.3
    return 2 * t_hit + b_hit + min(cites, 0.8) + recency + 0.2 * len(p.sources)


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Merge records by DOI, then by normalised title."""
    by_key: dict[str, Paper] = {}
    title_index: dict[str, str] = {}
    from agri_advisor.literature.models import title_key

    for p in papers:
        tk = title_key(p.title)
        key = p.key if p.doi else title_index.get(tk, p.key)
        if p.doi and tk in title_index and title_index[tk] in by_key:
            key = title_index[tk]
        if key in by_key:
            by_key[key].merge(p)
        else:
            by_key[key] = p
            if tk:
                title_index.setdefault(tk, key)
    return list(by_key.values())


def federated_search(
    query: str,
    sources: list[str] | None = None,
    limit_per_source: int = 25,
    keys: dict[str, str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    oa_only: bool = False,
    max_results: int | None = None,
) -> SearchResult:
    """Query several sources in parallel and return merged, ranked papers."""
    keys = {**env_keys(), **(keys or {})}
    chosen = [s for s in (sources or DEFAULT_SOURCES) if s in SOURCES]
    status: dict[str, dict[str, Any]] = {}
    raw: list[Paper] = []

    def run(name: str) -> tuple[str, list[Paper] | Exception]:
        try:
            return name, SOURCES[name](
                query, limit_per_source, keys,
                year_from=year_from, year_to=year_to, oa_only=oa_only,
            )
        except Exception as exc:  # noqa: BLE001 - report per-source failure
            return name, exc

    with ThreadPoolExecutor(max_workers=max(1, len(chosen))) as pool:
        for name, res in pool.map(run, chosen):
            if isinstance(res, Exception):
                status[name] = {"ok": False, "error": f"{type(res).__name__}: {res}"[:200]}
                logger.warning("source %s failed: %s", name, res)
            else:
                status[name] = {"ok": True, "n": len(res)}
                raw.extend(res)
    papers = deduplicate(raw)
    if oa_only:
        papers = [p for p in papers if p.is_oa]
    if year_from:
        papers = [p for p in papers if not p.year or p.year >= year_from]
    if year_to:
        papers = [p for p in papers if not p.year or p.year <= year_to]
    terms = _terms(query)
    for p in papers:
        p.score = round(_relevance(p, terms), 4)
    papers.sort(key=lambda p: p.score, reverse=True)
    if max_results:
        papers = papers[:max_results]
    return SearchResult(query=query, papers=papers, source_status=status, n_raw=len(raw))
