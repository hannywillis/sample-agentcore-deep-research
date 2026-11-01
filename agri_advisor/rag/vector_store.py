"""Persistent local vector store (NumPy + JSON).

Deliberately simple and dependency-light so that it can run on a laptop,
inside the AgentCore container, or on an offline-first edge server at a county
extension office. For >100k chunks swap in FAISS / OpenSearch / Bedrock
Knowledge Bases behind the same ``search`` interface.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from agri_advisor.rag.chunking import Chunk
from agri_advisor.rag.documents import DocumentMetadata


class LocalVectorStore:
    VECTORS = "vectors.npy"
    CHUNKS = "chunks.json"
    DOCS = "documents.json"
    MANIFEST = "manifest.json"

    def __init__(self, path: Path, embedder_name: str = "") -> None:
        self.path = Path(path)
        self.embedder_name = embedder_name
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.chunks: list[Chunk] = []
        self.documents: dict[str, DocumentMetadata] = {}

    # ------------------------------------------------------------------ write
    def add(
        self, doc: DocumentMetadata, chunks: list[Chunk], vectors: np.ndarray
    ) -> None:
        """Add (or replace) a document's chunks."""
        self.remove(doc.doc_id)
        self.documents[doc.doc_id] = doc
        if self.vectors.size == 0:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.chunks.extend(chunks)

    def remove(self, doc_id: str) -> None:
        if doc_id not in self.documents:
            return
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = (
            self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        )
        del self.documents[doc_id]

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / self.VECTORS, self.vectors)
        (self.path / self.CHUNKS).write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False)
        )
        (self.path / self.DOCS).write_text(
            json.dumps(
                {k: v.to_dict() for k, v in self.documents.items()},
                ensure_ascii=False,
                indent=1,
            )
        )
        (self.path / self.MANIFEST).write_text(
            json.dumps(
                {
                    "embedder": self.embedder_name,
                    "documents": len(self.documents),
                    "chunks": len(self.chunks),
                },
                indent=1,
            )
        )

    # ------------------------------------------------------------------- read
    @classmethod
    def load(cls, path: Path) -> LocalVectorStore:
        path = Path(path)
        manifest = json.loads((path / cls.MANIFEST).read_text())
        store = cls(path, manifest.get("embedder", ""))
        store.vectors = np.load(path / cls.VECTORS)
        store.chunks = [Chunk(**c) for c in json.loads((path / cls.CHUNKS).read_text())]
        store.documents = {
            k: DocumentMetadata(**v)
            for k, v in json.loads((path / cls.DOCS).read_text()).items()
        }
        return store

    @classmethod
    def exists(cls, path: Path) -> bool:
        return (Path(path) / cls.MANIFEST).exists()

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        candidate_mask: np.ndarray | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Cosine-similarity search (vectors are L2-normalised)."""
        if not self.chunks:
            return []
        scores = self.vectors @ query_vec.astype(np.float32).ravel()
        if candidate_mask is not None:
            scores = np.where(candidate_mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [
            (self.chunks[i], float(scores[i])) for i in order if np.isfinite(scores[i])
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder_name,
            "documents": len(self.documents),
            "chunks": len(self.chunks),
        }
