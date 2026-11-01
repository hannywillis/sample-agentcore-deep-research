"""Common bibliographic record shared by every literature source."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

_WS = re.compile(r"[^a-z0-9]+")


def normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi or None


def title_key(title: str) -> str:
    return _WS.sub("", (title or "").lower())[:120]


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    is_oa: bool | None = None
    oa_status: str | None = None
    license: str | None = None
    publisher: str | None = None
    cited_by: int | None = None
    keywords: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    doc_type: str | None = None
    sources: list[str] = field(default_factory=list)
    source_ids: dict[str, str] = field(default_factory=dict)
    score: float = 0.0

    def __post_init__(self) -> None:
        self.doi = normalise_doi(self.doi)

    @property
    def key(self) -> str:
        """Stable identity used for de-duplication across sources."""
        if self.doi:
            return f"doi:{self.doi}"
        return "t:" + hashlib.sha1(title_key(self.title).encode()).hexdigest()[:16]

    def merge(self, other: Paper) -> None:
        """Fill gaps from a duplicate record found in another source."""
        for name in (
            "abstract", "year", "venue", "doi", "url", "pdf_url", "license",
            "publisher", "oa_status", "doc_type",
        ):
            if not getattr(self, name) and getattr(other, name):
                setattr(self, name, getattr(other, name))
        if other.is_oa:
            self.is_oa = True
        if (other.cited_by or 0) > (self.cited_by or 0):
            self.cited_by = other.cited_by
        if len(other.authors) > len(self.authors):
            self.authors = other.authors
        for lst in ("keywords", "countries", "sources"):
            cur = getattr(self, lst)
            cur.extend(x for x in getattr(other, lst) if x not in cur)
        self.source_ids.update(other.source_ids)

    def citation(self) -> str:
        first = self.authors[0].split(",")[0] if self.authors else "Anon."
        etal = " et al." if len(self.authors) > 1 else ""
        doi = f" https://doi.org/{self.doi}" if self.doi else ""
        return f"{first}{etal} ({self.year or 'n.d.'}). {self.title}. {self.venue or ''}.{doi}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["key"] = self.key
        d["citation"] = self.citation()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Paper:
        fields_ = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**fields_)
