"""Ingest agricultural documents into the local vector store.

Usage::

    python -m agri_advisor.rag.ingest                       # default KB dir
    python -m agri_advisor.rag.ingest --source path/to/kalro_docs --rebuild
    python -m agri_advisor.rag.ingest --stats
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from agri_advisor.config import Settings, get_settings
from agri_advisor.rag.chunking import chunk_document
from agri_advisor.rag.documents import iter_documents
from agri_advisor.rag.embeddings import get_embedder
from agri_advisor.rag.vector_store import LocalVectorStore

logger = logging.getLogger(__name__)


def ingest_directory(
    source: Path,
    settings: Settings | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Load, chunk, embed and persist every document under ``source``."""
    s = settings or get_settings()
    embedder = get_embedder(s)
    store_path = s.vector_store_path
    if rebuild and store_path.exists():
        for name in (
            LocalVectorStore.VECTORS,
            LocalVectorStore.CHUNKS,
            LocalVectorStore.DOCS,
            LocalVectorStore.MANIFEST,
        ):
            (store_path / name).unlink(missing_ok=True)

    if LocalVectorStore.exists(store_path):
        store = LocalVectorStore.load(store_path)
        if store.embedder_name != embedder.name:
            raise ValueError(
                f"Existing store uses '{store.embedder_name}'; use --rebuild to "
                f"switch to '{embedder.name}'."
            )
    else:
        store = LocalVectorStore(store_path, embedder.name)

    docs = iter_documents(Path(source))
    if not docs:
        raise FileNotFoundError(f"No supported documents found under {source}")

    n_chunks = 0
    for doc in docs:
        chunks = chunk_document(doc, s.chunk_size, s.chunk_overlap)
        if not chunks:
            logger.warning("Skipping empty document %s", doc.metadata.path)
            continue
        # Prefix the title so short chunks still carry document context.
        vectors = embedder.embed(
            [f"{doc.metadata.title}. {c.section}. {c.text}" for c in chunks]
        )
        store.add(doc.metadata, chunks, vectors)
        n_chunks += len(chunks)
        logger.info("Indexed %s (%d chunks)", doc.metadata.doc_id, len(chunks))
    store.save()
    return {
        "ingested_documents": len(docs),
        "ingested_chunks": n_chunks,
        **store.stats(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    s = get_settings()
    if args.stats:
        if not LocalVectorStore.exists(s.vector_store_path):
            raise SystemExit("Vector store is empty - run ingestion first.")
        result = LocalVectorStore.load(s.vector_store_path).stats()
    else:
        result = ingest_directory(args.source or s.knowledge_base_dir, s, args.rebuild)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
