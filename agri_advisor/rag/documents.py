"""Document loading with provenance metadata (evidence traceability).

Supported inputs
----------------
* ``.md`` / ``.txt`` with an optional YAML front-matter block (``---`` ... ``---``)
* ``.pdf`` (requires ``pypdf``) and ``.docx`` (requires ``python-docx``) with an
  optional sidecar ``<filename>.meta.yaml`` holding the same metadata keys.

Metadata keys (all optional except ``title``; see
``data/knowledge_base/README.md``)::

    title, source_org, authors, year, url, doc_type, country, crops,
    language, access_level, embargo_until, validated_by, license, doc_id
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}
_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class DocumentMetadata:
    doc_id: str
    title: str
    source_org: str = "unknown"
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    url: str | None = None
    doc_type: str = "extension_guide"  # research_article | extension_guide | ...
    country: list[str] = field(default_factory=list)
    crops: list[str] = field(default_factory=list)
    language: str = "en"
    access_level: str = "open"
    embargo_until: str | None = None
    validated_by: str | None = None  # expert / institution that validated it
    license: str | None = None
    path: str = ""

    def citation(self) -> str:
        """Human-readable citation, e.g. ``KALRO (2023). Title.``"""
        who = ", ".join(self.authors) if self.authors else self.source_org
        year = f" ({self.year})" if self.year else ""
        link = f" {self.url}" if self.url else ""
        return f"{who}{year}. {self.title}.{link}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Document:
    metadata: DocumentMetadata
    text: str


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def _make_doc_id(path: Path, meta: dict[str, Any]) -> str:
    if meta.get("doc_id"):
        return str(meta["doc_id"])
    digest = hashlib.sha1(str(path).encode(), usedforsecurity=False).hexdigest()[:8]
    return f"{path.stem}-{digest}"


def _build_metadata(path: Path, meta: dict[str, Any]) -> DocumentMetadata:
    year = meta.get("year")
    return DocumentMetadata(
        doc_id=_make_doc_id(path, meta),
        title=str(meta.get("title") or path.stem.replace("_", " ").title()),
        source_org=str(meta.get("source_org", "unknown")),
        authors=_as_list(meta.get("authors")),
        year=int(year) if year else None,
        url=meta.get("url"),
        doc_type=str(meta.get("doc_type", "extension_guide")),
        country=[c.upper() for c in _as_list(meta.get("country"))],
        crops=[c.lower() for c in _as_list(meta.get("crops"))],
        language=str(meta.get("language", "en")),
        access_level=str(meta.get("access_level", "open")).lower(),
        embargo_until=(
            str(meta["embargo_until"]) if meta.get("embargo_until") else None
        ),
        validated_by=meta.get("validated_by"),
        license=meta.get("license"),
        path=str(path),
    )


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install 'pypdf' to ingest PDF files") from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install 'python-docx' to ingest .docx files") from exc
    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def load_document(path: Path) -> Document:
    """Load one file and its provenance metadata."""
    suffix = path.suffix.lower()
    meta: dict[str, Any] = {}
    if suffix in {".md", ".txt"}:
        raw = path.read_text(encoding="utf-8")
        match = _FRONT_MATTER.match(raw)
        if match:
            meta = yaml.safe_load(match.group(1)) or {}
            raw = raw[match.end() :]
        text = raw
    elif suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    sidecar = path.with_name(path.name + ".meta.yaml")
    if sidecar.exists():
        meta.update(yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {})
    return Document(metadata=_build_metadata(path, meta), text=text.strip())


def iter_documents(root: Path) -> list[Document]:
    """Recursively load every supported document under ``root``."""
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and p.name.lower() != "readme.md"
    )
    return [load_document(p) for p in files]
