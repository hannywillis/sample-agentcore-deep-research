"""Section-aware text chunking that keeps a pointer back to the source."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agri_advisor.rag.documents import Document

_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    position: int  # ordinal of the chunk within its document


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text into (heading, body) pairs."""
    sections: list[tuple[str, str]] = []
    heading, buf = "Introduction", []
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        end = min(len(text), start + size)
        # Prefer to break on a sentence boundary.
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + step // 2)
    return [p for p in pieces if p]


def chunk_document(doc: Document, size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in _split_sections(doc.text):
        for piece in _window(body, size, overlap):
            pos = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.metadata.doc_id}#{pos}",
                    doc_id=doc.metadata.doc_id,
                    text=piece,
                    section=section,
                    position=pos,
                )
            )
    return chunks
