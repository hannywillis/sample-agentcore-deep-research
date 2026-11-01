"""Connectors to open scholarly indexes and repositories.

Every connector returns :class:`Paper` records and never raises on network
errors (the federated search reports per-source status instead).

| source            | coverage                                         | key needed |
|-------------------|--------------------------------------------------|------------|
| openalex          | ~250M works, all disciplines, OA locations       | optional   |
| crossref          | DOI registry metadata (all publishers)           | no         |
| doaj              | Directory of Open Access Journals articles       | no         |
| europepmc         | life sciences incl. agriculture, full-text OA    | no         |
| cgspace           | CGIAR repository (ILRI, CIMMYT, IITA, ICRISAT...) | no         |
| zenodo            | datasets, reports, preprints                     | no         |
| arxiv             | preprints (CS/stats/q-bio - AI-for-agriculture)  | no         |
| core              | aggregated OA repository full texts              | yes (free) |
| semantic_scholar  | citation graph + TLDRs                           | optional   |

Use ``keys`` (from the user's / organisation's credential vault, or env vars)
to pass ``openalex_api_key``, ``core_api_key``, ``semantic_scholar_api_key``
and ``contact_email``.
"""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

import requests

from agri_advisor.literature.models import Paper

logger = logging.getLogger(__name__)
TIMEOUT = 25
UA = "agri-advisor-literature/1.0 (CNPq PROAFRICA research platform)"
_TAG = re.compile(r"<[^>]+>")


def _clean(text: Any) -> str | None:
    if not text:
        return None
    text = _TAG.sub(" ", html.unescape(str(text)))  # entities first: "&lt;i&gt;" -> tag
    return re.sub(r"\s+", " ", text).strip() or None


def _get(url: str, params: dict | None = None, headers: dict | None = None) -> requests.Response:
    h = {"User-Agent": UA, **(headers or {})}
    resp = requests.get(url, params=params, headers=h, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp


def _year(value: Any) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(m.group(0)) if m else None


# --------------------------------------------------------------------- OpenAlex
def _openalex_abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return " ".join(w for _, w in pos)


def _openalex_paper(w: dict) -> Paper:
    oa = w.get("open_access") or {}
    loc = w.get("primary_location") or {}
    best = w.get("best_oa_location") or {}
    return Paper(
        title=_clean(w.get("title") or w.get("display_name")) or "(untitled)",
        authors=[a["author"]["display_name"] for a in w.get("authorships", [])][:25],
        year=w.get("publication_year"),
        venue=((loc.get("source") or {}).get("display_name")),
        doi=w.get("doi"),
        abstract=_openalex_abstract(w.get("abstract_inverted_index")),
        url=loc.get("landing_page_url") or w.get("doi"),
        pdf_url=best.get("pdf_url") or oa.get("oa_url"),
        is_oa=oa.get("is_oa"),
        oa_status=oa.get("oa_status"),
        license=best.get("license"),
        cited_by=w.get("cited_by_count"),
        keywords=[k.get("display_name") for k in w.get("keywords", [])][:10],
        countries=sorted(
            {c for a in w.get("authorships", []) for c in a.get("countries", [])}
        ),
        doc_type=w.get("type"),
        sources=["openalex"],
        source_ids={"openalex": str(w.get("id", "")).rsplit("/", 1)[-1]},
    )


def search_openalex(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
    filters = []
    if f.get("year_from"):
        filters.append(f"from_publication_date:{f['year_from']}-01-01")
    if f.get("year_to"):
        filters.append(f"to_publication_date:{f['year_to']}-12-31")
    if f.get("oa_only"):
        filters.append("is_oa:true")
    params: dict[str, Any] = {"search": query, "per-page": min(limit, 200)}
    if filters:
        params["filter"] = ",".join(filters)
    if keys.get("openalex_api_key"):
        params["api_key"] = keys["openalex_api_key"]
    if keys.get("contact_email"):
        params["mailto"] = keys["contact_email"]
    data = _get("https://api.openalex.org/works", params).json()
    return [_openalex_paper(w) for w in data.get("results", [])]


def openalex_related(work_id: str, direction: str, limit: int, keys: dict) -> list[Paper]:
    """Snowballing: ``direction='citing'`` (forward) or ``'references'`` (backward)."""
    params: dict[str, Any] = {"per-page": min(limit, 200)}
    if keys.get("openalex_api_key"):
        params["api_key"] = keys["openalex_api_key"]
    if direction == "citing":
        params["filter"] = f"cites:{work_id}"
    else:
        work = _get(f"https://api.openalex.org/works/{work_id}", params).json()
        refs = [r.rsplit("/", 1)[-1] for r in work.get("referenced_works", [])][:limit]
        if not refs:
            return []
        params["filter"] = "openalex:" + "|".join(refs[:50])
    data = _get("https://api.openalex.org/works", params).json()
    return [_openalex_paper(w) for w in data.get("results", [])]


# --------------------------------------------------------------------- Crossref
def search_crossref(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
    params: dict[str, Any] = {"query": query, "rows": min(limit, 100)}
    flt = []
    if f.get("year_from"):
        flt.append(f"from-pub-date:{f['year_from']}")
    if f.get("year_to"):
        flt.append(f"until-pub-date:{f['year_to']}")
    if flt:
        params["filter"] = ",".join(flt)
    if keys.get("contact_email"):
        params["mailto"] = keys["contact_email"]
    items = _get("https://api.crossref.org/works", params).json()["message"]["items"]
    out = []
    for it in items:
        date = (it.get("issued") or {}).get("date-parts", [[None]])[0][0]
        lic = (it.get("license") or [{}])[0].get("URL")
        out.append(
            Paper(
                title=_clean((it.get("title") or ["(untitled)"])[0]) or "(untitled)",
                authors=[
                    f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                    for a in it.get("author", [])
                ][:25],
                year=date,
                venue=(it.get("container-title") or [None])[0],
                doi=it.get("DOI"),
                abstract=_clean(it.get("abstract")),
                url=it.get("URL"),
                license=lic,
                is_oa=bool(lic and "creativecommons" in lic) or None,
                publisher=it.get("publisher"),
                cited_by=it.get("is-referenced-by-count"),
                doc_type=it.get("type"),
                sources=["crossref"],
            )
        )
    return out


# ------------------------------------------------------------------------- DOAJ
def search_doaj(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
    q = query
    if f.get("year_from") or f.get("year_to"):
        q += f" AND bibjson.year:[{f.get('year_from') or 1900} TO {f.get('year_to') or 2100}]"
    data = _get(
        f"https://doaj.org/api/search/articles/{requests.utils.quote(q)}",
        {"pageSize": min(limit, 100)},
    ).json()
    out = []
    for r in data.get("results", []):
        b = r.get("bibjson", {})
        doi = next((i["id"] for i in b.get("identifier", []) if i.get("type") == "doi"), None)
        links = b.get("link", [])
        url = next((lk["url"] for lk in links if lk.get("type") == "fulltext"), None)
        out.append(
            Paper(
                title=_clean(b.get("title")) or "(untitled)",
                authors=[a.get("name", "") for a in b.get("author", [])][:25],
                year=_year(b.get("year")),
                venue=(b.get("journal") or {}).get("title"),
                doi=doi,
                abstract=_clean(b.get("abstract")),
                url=url,
                is_oa=True,
                oa_status="gold",
                publisher=(b.get("journal") or {}).get("publisher"),
                keywords=b.get("keywords", [])[:10],
                doc_type="article",
                sources=["doaj"],
                source_ids={"doaj": r.get("id", "")},
            )
        )
    return out


# -------------------------------------------------------------------- EuropePMC
def search_europepmc(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
    q = query
    if f.get("year_from") or f.get("year_to"):
        q += f" AND PUB_YEAR:[{f.get('year_from') or 1900} TO {f.get('year_to') or 2100}]"
    if f.get("oa_only"):
        q += " AND OPEN_ACCESS:y"
    data = _get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {"query": q, "format": "json", "pageSize": min(limit, 100), "resultType": "core"},
    ).json()
    out = []
    for r in data.get("resultList", {}).get("result", []):
        pdf = None
        for u in (r.get("fullTextUrlList") or {}).get("fullTextUrl", []):
            if u.get("documentStyle") == "pdf" and u.get("availabilityCode") == "OA":
                pdf = u.get("url")
        out.append(
            Paper(
                title=_clean(r.get("title")) or "(untitled)",
                authors=[a.strip() for a in (r.get("authorString") or "").split(",") if a.strip()][:25],
                year=_year(r.get("pubYear")),
                venue=r.get("journalTitle")
                or ((r.get("journalInfo") or {}).get("journal") or {}).get("title"),
                doi=r.get("doi"),
                abstract=_clean(r.get("abstractText")),
                url=f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}",
                pdf_url=pdf,
                is_oa=r.get("isOpenAccess") == "Y",
                cited_by=r.get("citedByCount"),
                sources=["europepmc"],
                source_ids={"europepmc": f"{r.get('source')}:{r.get('id')}"},
            )
        )
    return out


# ---------------------------------------------------------------------- CGSpace
def search_cgspace(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
    data = _get(
        "https://cgspace.cgiar.org/server/api/discover/search/objects",
        {"query": query, "size": min(limit, 100), "dsoType": "ITEM"},
    ).json()
    objs = data.get("_embedded", {}).get("searchResult", {}).get("_embedded", {}).get("objects", [])
    out = []
    for o in objs:
        item = o.get("_embedded", {}).get("indexableObject", {})
        md = item.get("metadata", {})

        def v(k: str, md: dict = md) -> str | None:
            return md[k][0]["value"] if md.get(k) else None

        def vs(k: str, md: dict = md) -> list[str]:
            return [x["value"] for x in md.get(k, [])]

        year = _year(v("dcterms.issued"))
        if f.get("year_from") and year and year < int(f["year_from"]):
            continue
        if f.get("year_to") and year and year > int(f["year_to"]):
            continue
        rights = (v("dcterms.accessRights") or "").lower()
        is_oa = "open" in rights
        if f.get("oa_only") and not is_oa:
            continue
        out.append(
            Paper(
                title=_clean(v("dc.title")) or "(untitled)",
                authors=vs("dc.contributor.author")[:25],
                year=year,
                venue=v("cg.journal") or v("dcterms.publisher"),
                doi=v("cg.identifier.doi"),
                abstract=_clean(v("dcterms.abstract")),
                url=v("dc.identifier.uri"),
                is_oa=is_oa,
                license=v("dcterms.license"),
                publisher=v("dcterms.publisher"),
                keywords=vs("dcterms.subject")[:10],
                countries=vs("cg.coverage.iso3166-alpha2"),
                doc_type=v("dcterms.type"),
                sources=["cgspace"],
                source_ids={"cgspace": item.get("uuid", "")},
            )
        )
    return out


# ----------------------------------------------------------------------- Zenodo
def search_zenodo(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
    data = _get(
        "https://zenodo.org/api/records", {"q": query, "size": min(limit, 100)}
    ).json()
    out = []
    for h in data.get("hits", {}).get("hits", []):
        m = h.get("metadata", {})
        files = [x for x in h.get("files", []) if str(x.get("key", "")).endswith(".pdf")]
        out.append(
            Paper(
                title=_clean(m.get("title")) or "(untitled)",
                authors=[c.get("name", "") for c in m.get("creators", [])][:25],
                year=_year(m.get("publication_date")),
                venue="Zenodo",
                doi=h.get("doi"),
                abstract=_clean(m.get("description")),
                url=(h.get("links") or {}).get("self_html"),
                pdf_url=(files[0].get("links") or {}).get("self") if files else None,
                is_oa=(m.get("access_right") == "open"),
                license=(m.get("license") or {}).get("id"),
                keywords=m.get("keywords", [])[:10],
                doc_type=(m.get("resource_type") or {}).get("type"),
                sources=["zenodo"],
                source_ids={"zenodo": str(h.get("id"))},
            )
        )
    return out


# ------------------------------------------------------------------------ arXiv
def search_arxiv(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
    resp = _get(
        "https://export.arxiv.org/api/query",
        {"search_query": f"all:{query}", "max_results": min(limit, 100)},
    )
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    out = []
    for e in ET.fromstring(resp.text).findall("a:entry", ns):
        pdf = next(
            (lk.get("href") for lk in e.findall("a:link", ns) if lk.get("title") == "pdf"),
            None,
        )
        doi_el = e.find("x:doi", ns)
        out.append(
            Paper(
                title=_clean(e.findtext("a:title", "", ns)) or "(untitled)",
                authors=[a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:25],
                year=_year(e.findtext("a:published", "", ns)),
                venue="arXiv",
                doi=doi_el.text if doi_el is not None else None,
                abstract=_clean(e.findtext("a:summary", "", ns)),
                url=e.findtext("a:id", "", ns),
                pdf_url=pdf,
                is_oa=True,
                oa_status="green",
                doc_type="preprint",
                sources=["arxiv"],
            )
        )
    return out


# ------------------------------------------------------------------------- CORE
def search_core(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
    if not keys.get("core_api_key"):
        raise PermissionError("core_api_key not configured (free at core.ac.uk)")
    data = _get(
        "https://api.core.ac.uk/v3/search/works/",
        {"q": query, "limit": min(limit, 100)},
        {"Authorization": f"Bearer {keys['core_api_key']}"},
    ).json()
    return [
        Paper(
            title=_clean(r.get("title")) or "(untitled)",
            authors=[a.get("name", "") for a in r.get("authors", [])][:25],
            year=r.get("yearPublished"),
            venue=(r.get("journals") or [{}])[0].get("title") if r.get("journals") else None,
            doi=r.get("doi"),
            abstract=_clean(r.get("abstract")),
            url=(r.get("links") or [{}])[0].get("url") if r.get("links") else None,
            pdf_url=r.get("downloadUrl"),
            is_oa=True,
            sources=["core"],
            source_ids={"core": str(r.get("id"))},
        )
        for r in data.get("results", [])
    ]


# ------------------------------------------------------------- Semantic Scholar
def search_semantic_scholar(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
    params: dict[str, Any] = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,authors,year,venue,externalIds,abstract,url,openAccessPdf,"
        "citationCount,isOpenAccess,tldr",
    }
    if f.get("year_from") or f.get("year_to"):
        params["year"] = f"{f.get('year_from') or ''}-{f.get('year_to') or ''}"
    headers = {"x-api-key": keys["semantic_scholar_api_key"]} if keys.get(
        "semantic_scholar_api_key"
    ) else {}
    data = _get(
        "https://api.semanticscholar.org/graph/v1/paper/search", params, headers
    ).json()
    return [
        Paper(
            title=_clean(r.get("title")) or "(untitled)",
            authors=[a.get("name", "") for a in r.get("authors", [])][:25],
            year=r.get("year"),
            venue=r.get("venue"),
            doi=(r.get("externalIds") or {}).get("DOI"),
            abstract=_clean(r.get("abstract") or (r.get("tldr") or {}).get("text")),
            url=r.get("url"),
            pdf_url=(r.get("openAccessPdf") or {}).get("url"),
            is_oa=r.get("isOpenAccess"),
            cited_by=r.get("citationCount"),
            sources=["semantic_scholar"],
            source_ids={"semantic_scholar": r.get("paperId", "")},
        )
        for r in data.get("data", [])
    ]


SOURCES: dict[str, Callable[..., list[Paper]]] = {
    "openalex": search_openalex,
    "crossref": search_crossref,
    "doaj": search_doaj,
    "europepmc": search_europepmc,
    "cgspace": search_cgspace,
    "zenodo": search_zenodo,
    "arxiv": search_arxiv,
    "core": search_core,
    "semantic_scholar": search_semantic_scholar,
}
DEFAULT_SOURCES = ["openalex", "crossref", "doaj", "europepmc", "cgspace"]
