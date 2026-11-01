"""Research library and systematic-review workspace.

Supports the workflow researchers actually follow:

1. create a review project with a protocol (question, PICO, inclusion /
   exclusion criteria, sources, search strings);
2. run federated searches into it (every search is logged for the PRISMA
   "identification" box and for reproducibility);
3. de-duplicate, then screen title/abstract and full text
   (include / exclude / maybe + reason), optionally with keyword-based
   prioritisation;
4. snowball citations (forward / backward via OpenAlex);
5. export to BibTeX / RIS (Zotero, Mendeley, EndNote, Rayyan) / CSV;
6. produce PRISMA 2020 flow counts and an evidence table + narrative synthesis
   (LLM if configured, otherwise extractive and clearly labelled);
7. optionally push open-licensed included papers into the RAG knowledge base
   so farmer-facing answers can cite peer-reviewed evidence.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from agri_advisor.accounts.store import _env_path
from agri_advisor.literature.models import Paper
from agri_advisor.literature.search import SearchResult

DECISIONS = ("include", "exclude", "maybe", "unscreened")
STAGES = ("title_abstract", "full_text")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY, owner_id INTEGER NOT NULL, org_id INTEGER,
  shared_with_org INTEGER NOT NULL DEFAULT 0, title TEXT NOT NULL,
  protocol TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS review_searches (
  id INTEGER PRIMARY KEY, review_id INTEGER NOT NULL, ts REAL NOT NULL,
  query TEXT, origin TEXT, sources TEXT, n_raw INTEGER, n_new INTEGER, n_dup INTEGER);
CREATE TABLE IF NOT EXISTS review_papers (
  review_id INTEGER NOT NULL, key TEXT NOT NULL, data TEXT NOT NULL,
  ta_decision TEXT NOT NULL DEFAULT 'unscreened', ta_reason TEXT,
  ft_decision TEXT NOT NULL DEFAULT 'unscreened', ft_reason TEXT,
  reviewer TEXT, fulltext_path TEXT, notes TEXT, added REAL NOT NULL,
  PRIMARY KEY (review_id, key));
"""


class ReviewLibrary:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _env_path("PLATFORM_DB", ".agri_store/platform.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):  # type: ignore[no-untyped-def]
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------ reviews
    def create_review(self, owner_id: int, title: str, protocol: dict[str, Any] | None = None,
                      org_id: int | None = None, shared_with_org: bool = False) -> dict:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO reviews(owner_id,org_id,shared_with_org,title,protocol,created) "
                "VALUES(?,?,?,?,?,?)",
                (owner_id, org_id, int(shared_with_org), title, json.dumps(protocol or {}),
                 time.time()),
            )
        return self.get_review(int(cur.lastrowid), owner_id, org_id)  # type: ignore[return-value]

    def _can_read(self, row: sqlite3.Row, user_id: int, org_id: int | None) -> bool:
        return row["owner_id"] == user_id or bool(
            row["shared_with_org"] and org_id and row["org_id"] == org_id
        )

    def get_review(self, review_id: int, user_id: int, org_id: int | None = None) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM reviews WHERE id=?", (review_id,)).fetchone()
            if not r or not self._can_read(r, user_id, org_id):
                return None
            n = c.execute("SELECT COUNT(*) FROM review_papers WHERE review_id=?",
                          (review_id,)).fetchone()[0]
        return {"id": r["id"], "title": r["title"], "owner_id": r["owner_id"],
                "shared_with_org": bool(r["shared_with_org"]),
                "protocol": json.loads(r["protocol"]), "n_papers": n, "created": r["created"]}

    def list_reviews(self, user_id: int, org_id: int | None = None) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id FROM reviews WHERE owner_id=? OR (shared_with_org=1 AND org_id=?) "
                "ORDER BY id DESC", (user_id, org_id or -1)).fetchall()
        return [self.get_review(r["id"], user_id, org_id) for r in rows]  # type: ignore[misc]

    def delete_review(self, review_id: int, user_id: int) -> bool:
        with self._conn() as c:
            r = c.execute("SELECT owner_id FROM reviews WHERE id=?", (review_id,)).fetchone()
            if not r:
                return False
            if r["owner_id"] != user_id:
                raise PermissionError("only the owner can delete a review")
            c.execute("DELETE FROM review_papers WHERE review_id=?", (review_id,))
            c.execute("DELETE FROM review_searches WHERE review_id=?", (review_id,))
            c.execute("DELETE FROM reviews WHERE id=?", (review_id,))
        return True

    # ------------------------------------------------------------- papers
    def add_papers(self, review_id: int, papers: list[Paper], query: str = "",
                   origin: str = "search", sources: list[str] | None = None,
                   n_raw: int | None = None) -> dict[str, int]:
        """Add records; duplicates of papers already in the review are merged."""
        new = dup = 0
        with self._conn() as c:
            for p in papers:
                row = c.execute("SELECT data FROM review_papers WHERE review_id=? AND key=?",
                                (review_id, p.key)).fetchone()
                if row:
                    existing = Paper.from_dict(json.loads(row["data"]))
                    existing.merge(p)
                    c.execute("UPDATE review_papers SET data=? WHERE review_id=? AND key=?",
                              (json.dumps(existing.to_dict()), review_id, p.key))
                    dup += 1
                else:
                    c.execute("INSERT INTO review_papers(review_id,key,data,added) VALUES(?,?,?,?)",
                              (review_id, p.key, json.dumps(p.to_dict()), time.time()))
                    new += 1
            raw = n_raw if n_raw is not None else len(papers)
            c.execute(
                "INSERT INTO review_searches(review_id,ts,query,origin,sources,n_raw,n_new,n_dup) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (review_id, time.time(), query, origin, json.dumps(sources or []), raw, new,
                 raw - new),
            )
        return {"added": new, "merged_duplicates": dup, "identified": raw}

    def add_search_result(self, review_id: int, result: SearchResult, origin: str = "search") -> dict:
        return self.add_papers(review_id, result.papers, result.query, origin,
                               [s for s, st in result.source_status.items() if st.get("ok")],
                               result.n_raw)

    def papers(self, review_id: int, stage: str | None = None,
               decision: str | None = None) -> list[dict]:
        sql = "SELECT * FROM review_papers WHERE review_id=?"
        args: list[Any] = [review_id]
        if stage and decision:
            col = "ta_decision" if stage == "title_abstract" else "ft_decision"
            sql += f" AND {col}=?"
            args.append(decision)
        with self._conn() as c:
            rows = c.execute(sql + " ORDER BY added", args).fetchall()
        out = []
        for r in rows:
            d = json.loads(r["data"])
            d.update({k: r[k] for k in ("ta_decision", "ta_reason", "ft_decision", "ft_reason",
                                        "reviewer", "fulltext_path", "notes")})
            out.append(d)
        return out

    def included(self, review_id: int) -> list[dict]:
        """Final included set: full-text include, or TA include if FT not screened."""
        return [p for p in self.papers(review_id)
                if p["ft_decision"] == "include"
                or (p["ta_decision"] == "include" and p["ft_decision"] == "unscreened")]

    def screen(self, review_id: int, key: str, decision: str, stage: str = "title_abstract",
               reason: str | None = None, reviewer: str | None = None,
               notes: str | None = None) -> dict:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        if decision == "exclude" and not reason:
            raise ValueError("an exclusion reason is required (PRISMA reporting)")
        pre = "ta" if stage == "title_abstract" else "ft"
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE review_papers SET {pre}_decision=?, {pre}_reason=?, reviewer=?, "
                "notes=COALESCE(?, notes) WHERE review_id=? AND key=?",
                (decision, reason, reviewer, notes, review_id, key),
            )
            if not cur.rowcount:
                raise KeyError(key)
        return {"key": key, "stage": stage, "decision": decision}

    def set_fulltext(self, review_id: int, key: str, path: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE review_papers SET fulltext_path=? WHERE review_id=? AND key=?",
                      (path, review_id, key))

    def prioritise(self, review_id: int, include_terms: list[str],
                   exclude_terms: list[str] | None = None) -> list[dict]:
        """Rank unscreened records by keyword evidence to speed up screening.

        This is a transparent aid, not an automatic decision: the researcher
        still records every include/exclude.
        """
        inc = [t.lower() for t in include_terms]
        exc = [t.lower() for t in exclude_terms or []]
        ranked = []
        for p in self.papers(review_id, "title_abstract", "unscreened"):
            text = f"{p.get('title', '')} {p.get('abstract') or ''}".lower()
            hits = [t for t in inc if t in text]
            neg = [t for t in exc if t in text]
            score = len(hits) / max(1, len(inc)) - 0.5 * len(neg)
            ranked.append({"key": p["key"], "title": p["title"], "score": round(score, 3),
                           "matched": hits, "exclusion_terms": neg,
                           "suggestion": "likely include" if score >= 0.5 and not neg
                           else "likely exclude" if score <= 0 else "check"})
        return sorted(ranked, key=lambda r: r["score"], reverse=True)

    # -------------------------------------------------------------- PRISMA
    def prisma(self, review_id: int) -> dict[str, Any]:
        with self._conn() as c:
            searches = [dict(r) for r in c.execute(
                "SELECT * FROM review_searches WHERE review_id=? ORDER BY ts", (review_id,))]
        papers = self.papers(review_id)
        db = [s for s in searches if s["origin"] != "snowball"]
        other = [s for s in searches if s["origin"] == "snowball"]
        identified_db = sum(s["n_raw"] or 0 for s in db)
        identified_other = sum(s["n_raw"] or 0 for s in other)
        unique = len(papers)
        ta = [p for p in papers if p["ta_decision"] != "unscreened"]
        ta_excl = [p for p in ta if p["ta_decision"] == "exclude"]
        ft_sought = [p for p in papers if p["ta_decision"] in ("include", "maybe")]
        ft = [p for p in ft_sought if p["ft_decision"] != "unscreened"]
        ft_excl = [p for p in ft if p["ft_decision"] == "exclude"]
        reasons: dict[str, int] = {}
        for p in ft_excl:
            reasons[p["ft_reason"] or "unspecified"] = reasons.get(p["ft_reason"] or "unspecified", 0) + 1
        return {
            "identification": {
                "records_from_databases": identified_db,
                "records_from_citation_searching": identified_other,
                "duplicates_removed": identified_db + identified_other - unique,
                "searches": [{"query": s["query"], "origin": s["origin"],
                              "sources": json.loads(s["sources"] or "[]"), "n": s["n_raw"],
                              "date": time.strftime("%Y-%m-%d", time.gmtime(s["ts"]))}
                             for s in searches],
            },
            "screening": {
                "records_screened": len(ta),
                "records_awaiting_screening": unique - len(ta),
                "records_excluded": len(ta_excl),
                "reports_sought": len(ft_sought),
                "reports_with_fulltext": sum(1 for p in ft_sought if p.get("fulltext_path")),
                "reports_assessed": len(ft),
                "reports_excluded": len(ft_excl),
                "exclusion_reasons": reasons,
            },
            "included": {"studies": len(self.included(review_id))},
        }

    # ----------------------------------------------------------- synthesis
    def evidence_table(self, review_id: int, focus_terms: list[str] | None = None) -> list[dict]:
        rows = []
        for p in self.included(review_id):
            rows.append({
                "citation": p.get("citation"),
                "year": p.get("year"),
                "countries": ", ".join(p.get("countries") or []),
                "design": _guess_design(p),
                "key_finding": _key_sentence(p.get("abstract") or "", focus_terms or []),
                "open_access": bool(p.get("is_oa")),
                "doi": p.get("doi"),
            })
        return rows

    def synthesis(self, review_id: int, question: str | None = None, llm: Any = None,
                  focus_terms: list[str] | None = None) -> dict[str, Any]:
        table = self.evidence_table(review_id, focus_terms)
        review = self._raw_review(review_id)
        question = question or (review or {}).get("protocol", {}).get("question") or ""
        if not table:
            return {"mode": "none", "summary": "No included studies yet.", "evidence_table": []}
        if llm is not None:
            evidence = "\n".join(
                f"[{i + 1}] {r['citation']} | design: {r['design']} | finding: {r['key_finding']}"
                for i, r in enumerate(table[:60])
            )
            try:
                summary = llm.complete(
                    "You are a careful systematic reviewer. Synthesise ONLY the evidence "
                    "given, cite studies by [number], state agreement, contradictions, "
                    "evidence gaps and the certainty of evidence. Do not invent findings.",
                    f"Review question: {question}\n\nIncluded studies:\n{evidence}",
                )
                return {"mode": "llm", "summary": summary, "evidence_table": table}
            except Exception as exc:  # noqa: BLE001
                fallback_note = f"LLM failed ({type(exc).__name__}); extractive summary shown."
        else:
            fallback_note = "Extractive summary (no LLM configured) - verify before use."
        designs: dict[str, int] = {}
        for r in table:
            designs[r["design"]] = designs.get(r["design"], 0) + 1
        years = [r["year"] for r in table if r["year"]]
        lines = [
            f"{len(table)} included studies"
            + (f" published {min(years)}-{max(years)}" if years else "")
            + "; designs: " + ", ".join(f"{k} ({v})" for k, v in sorted(designs.items())) + ".",
        ]
        lines += [f"[{i + 1}] {r['key_finding']}" for i, r in enumerate(table[:15]) if r["key_finding"]]
        return {"mode": "extractive", "note": fallback_note, "summary": "\n".join(lines),
                "evidence_table": table}

    def _raw_review(self, review_id: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM reviews WHERE id=?", (review_id,)).fetchone()
        return {"protocol": json.loads(r["protocol"]), "title": r["title"]} if r else None

    # ------------------------------------------------------- knowledge base
    def ingest_to_kb(self, review_id: int, out_dir: Path, only_open: bool = True) -> dict:
        """Write included open-licence papers (metadata + abstract) as KB documents.

        Non-open papers are skipped unless ``only_open=False``, in which case
        they are written with ``access_level: closed`` so governance keeps them
        out of farmer-facing answers.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        written, skipped = [], 0
        for p in self.included(review_id):
            if not p.get("abstract"):
                skipped += 1
                continue
            is_open = bool(p.get("is_oa")) and _open_licence(p.get("license"))
            if only_open and not is_open:
                skipped += 1
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", (p.get("doi") or p["title"]).lower())[:80].strip("-")
            meta = {
                "doc_id": f"lit-{slug}", "title": p["title"],
                "source_org": p.get("venue") or p.get("publisher") or "peer-reviewed literature",
                "authors": "; ".join(p.get("authors") or [])[:500], "year": p.get("year"),
                "url": f"https://doi.org/{p['doi']}" if p.get("doi") else p.get("url"),
                "doc_type": "peer_reviewed_article",
                "country": ", ".join(p.get("countries") or []),
                "language": "en", "access_level": "open" if is_open else "closed",
                "license": p.get("license") or "unknown",
                "validated_by": f"systematic review #{review_id} (pending expert validation)",
            }
            body = (f"# Summary of evidence\n{p['abstract']}\n\n"
                    f"# Citation\n{p.get('citation')}\n")
            path = out_dir / f"{slug}.md"
            path.write_text(f"---\n{yaml.safe_dump(meta, sort_keys=False)}---\n\n{body}",
                            encoding="utf-8")
            written.append(str(path))
        return {"written": len(written), "skipped": skipped, "files": written}

    # -------------------------------------------------------------- export
    def export(self, review_id: int, fmt: str = "bibtex", which: str = "included") -> str:
        papers = self.included(review_id) if which == "included" else self.papers(review_id)
        if fmt == "bibtex":
            return to_bibtex(papers)
        if fmt == "ris":
            return to_ris(papers)
        if fmt == "csv":
            return to_csv(papers)
        raise ValueError("fmt must be bibtex | ris | csv")


# ---------------------------------------------------------------- helpers
_DESIGNS = [
    ("systematic review / meta-analysis", ("meta-analy", "systematic review")),
    ("randomised controlled trial", ("randomi", "rct")),
    ("field / on-farm trial", ("field trial", "on-farm", "field experiment", "plots")),
    ("modelling / simulation", ("model", "simulat", "machine learning", "apsim", "dssat")),
    ("survey / observational", ("survey", "household", "questionnaire", "cross-sectional")),
    ("review", ("review",)),
]


def _guess_design(p: dict) -> str:
    text = f"{p.get('title', '')} {p.get('abstract') or ''}".lower()
    for label, cues in _DESIGNS:
        if any(c in text for c in cues):
            return label
    return "unclassified"


def _key_sentence(abstract: str, terms: list[str]) -> str:
    sents = re.split(r"(?<=[.!?])\s+", abstract.strip())
    if not sents or not sents[0]:
        return ""
    cues = ("result", "found", "increase", "reduc", "significant", "improv", "conclu",
            *[t.lower() for t in terms])
    best = max(sents, key=lambda s: sum(c in s.lower() for c in cues))
    return best[:400]


def _open_licence(lic: str | None) -> bool:
    lic = (lic or "").lower()
    return any(k in lic for k in ("cc-by", "cc by", "creativecommons", "cc0", "public-domain"))


def _bib_key(p: dict, used: set[str]) -> str:
    name = (p.get("authors") or ["anon"])[0]
    surname = name.split(",")[0] if "," in name else (name.split() or ["anon"])[-1]
    first = re.sub(r"[^A-Za-z]", "", surname) or "anon"
    word = next((w for w in re.findall(r"[A-Za-z]{4,}", p.get("title", ""))), "paper")
    key = base = f"{first.lower()}{p.get('year') or 'nd'}{word.lower()}"
    i = 1
    while key in used:
        i += 1
        key = f"{base}{i}"
    used.add(key)
    return key


def _bib_escape(s: Any) -> str:
    return str(s).replace("{", "\\{").replace("}", "\\}")


def to_bibtex(papers: list[dict]) -> str:
    used: set[str] = set()
    out = []
    for p in papers:
        fields = {
            "title": p.get("title"), "author": " and ".join(p.get("authors") or []),
            "year": p.get("year"), "journal": p.get("venue"), "doi": p.get("doi"),
            "url": p.get("url"), "abstract": p.get("abstract"), "publisher": p.get("publisher"),
            "keywords": ", ".join(p.get("keywords") or []),
        }
        body = ",\n".join(f"  {k} = {{{_bib_escape(v)}}}" for k, v in fields.items() if v)
        out.append(f"@article{{{_bib_key(p, used)},\n{body}\n}}")
    return "\n\n".join(out) + "\n"


def to_ris(papers: list[dict]) -> str:
    out = []
    for p in papers:
        lines = ["TY  - JOUR", f"TI  - {p.get('title', '')}"]
        lines += [f"AU  - {a}" for a in p.get("authors") or []]
        for tag, key in (("PY", "year"), ("JO", "venue"), ("DO", "doi"), ("UR", "url"),
                         ("AB", "abstract"), ("PB", "publisher")):
            if p.get(key):
                lines.append(f"{tag}  - {p[key]}")
        lines += [f"KW  - {k}" for k in p.get("keywords") or []]
        lines.append("ER  - ")
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


CSV_COLUMNS = ("key", "title", "authors", "year", "venue", "doi", "url", "is_oa", "license",
               "cited_by", "sources", "ta_decision", "ta_reason", "ft_decision", "ft_reason",
               "abstract")


def to_csv(papers: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for p in papers:
        row = dict(p)
        row["authors"] = "; ".join(p.get("authors") or [])
        row["sources"] = "; ".join(p.get("sources") or [])
        w.writerow(row)
    return buf.getvalue()
