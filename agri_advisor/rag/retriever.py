"""Evidence retrieval with source attribution and governance filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agri_advisor.config import Settings, get_settings
from agri_advisor.governance import can_access
from agri_advisor.rag.embeddings import Embedder, get_embedder
from agri_advisor.rag.vector_store import LocalVectorStore


@dataclass
class Evidence:
    """One retrieved passage plus everything needed to trace it to its source."""

    ref: str  # short citation key used inline, e.g. "S1"
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    citation: str
    source_org: str
    url: str | None
    year: int | None
    access_level: str
    validated_by: str | None
    crops: list[str]
    country: list[str]

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        d = dict(self.__dict__)
        if not include_text:
            d.pop("text")
        d["score"] = round(self.score, 4)
        return d


class KnowledgeRetriever:
    """Query the agricultural knowledge base.

    Scores are cosine similarity plus small boosts for metadata matches on crop
    and country, so that a Kenyan maize question prefers Kenyan maize guidance.
    """

    CROP_BOOST = 0.05
    COUNTRY_BOOST = 0.03
    VALIDATED_BOOST = 0.02

    def __init__(
        self,
        store: LocalVectorStore | None = None,
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        if store is None:
            path = self.settings.vector_store_path
            store = (
                LocalVectorStore.load(path)
                if LocalVectorStore.exists(path)
                else LocalVectorStore(path, self.embedder.name)
            )
        self.store = store
        if self.store.embedder_name and self.store.embedder_name != self.embedder.name:
            raise ValueError(
                f"Vector store was built with '{self.store.embedder_name}' but the "
                f"configured embedder is '{self.embedder.name}'. Re-run ingestion."
            )

    @property
    def is_empty(self) -> bool:
        return not self.store.chunks

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        role: str = "public",
        crop: str | None = None,
        country: str | None = None,
        min_score: float | None = None,
    ) -> list[Evidence]:
        if self.is_empty:
            return []
        k = k or self.settings.top_k
        threshold = self.settings.min_relevance if min_score is None else min_score

        docs = self.store.documents
        allowed = np.array(
            [
                can_access(
                    role, docs[c.doc_id].access_level, docs[c.doc_id].embargo_until
                )
                for c in self.store.chunks
            ]
        )
        qvec = self.embedder.embed([query])[0]
        hits = self.store.search(qvec, k=max(k * 4, 20), candidate_mask=allowed)

        rescored: list[tuple[float, Any, float]] = []
        for chunk, score in hits:
            meta = docs[chunk.doc_id]
            boost = 0.0
            if crop and crop.lower() in meta.crops:
                boost += self.CROP_BOOST
            if country and country.upper() in meta.country:
                boost += self.COUNTRY_BOOST
            if meta.validated_by:
                boost += self.VALIDATED_BOOST
            rescored.append((score + boost, chunk, score))
        rescored.sort(key=lambda t: -t[0])

        evidence: list[Evidence] = []
        for final, chunk, raw in rescored:
            if raw < threshold or len(evidence) >= k:
                continue
            meta = docs[chunk.doc_id]
            evidence.append(
                Evidence(
                    ref=f"S{len(evidence) + 1}",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=meta.title,
                    section=chunk.section,
                    text=chunk.text,
                    score=final,
                    citation=meta.citation(),
                    source_org=meta.source_org,
                    url=meta.url,
                    year=meta.year,
                    access_level=meta.access_level,
                    validated_by=meta.validated_by,
                    crops=meta.crops,
                    country=meta.country,
                )
            )
        return evidence
