"""Command-line literature tools.

    python -m agri_advisor.literature search "maize yield climate Kenya" --year-from 2015 --oa \
        --export bibtex --out maize.bib
    python -m agri_advisor.literature journals --subject "animal science" --subject agronomy
    python -m agri_advisor.literature access --doi 10.1016/j.fcr.2020.107800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agri_advisor.literature.access import resolve_access
from agri_advisor.literature.journals import catalogue
from agri_advisor.literature.library import to_bibtex, to_csv, to_ris
from agri_advisor.literature.search import federated_search
from agri_advisor.literature.sources import SOURCES


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agri_advisor.literature")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--sources", nargs="*", choices=list(SOURCES))
    s.add_argument("--limit", type=int, default=20, help="per source")
    s.add_argument("--max", type=int, default=50)
    s.add_argument("--year-from", type=int)
    s.add_argument("--year-to", type=int)
    s.add_argument("--oa", action="store_true", help="open access only")
    s.add_argument("--export", choices=["bibtex", "ris", "csv", "json"])
    s.add_argument("--out", type=Path)
    j = sub.add_parser("journals")
    j.add_argument("--subject", action="append")
    j.add_argument("--offline", action="store_true", help="curated list only")
    a = sub.add_parser("access")
    a.add_argument("--doi", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "search":
        res = federated_search(args.query, args.sources, args.limit, None, args.year_from,
                               args.year_to, args.oa, args.max)
        dicts = [p.to_dict() for p in res.papers]
        if args.export:
            text = {"bibtex": to_bibtex, "ris": to_ris, "csv": to_csv}.get(
                args.export, lambda d: json.dumps(d, indent=2))(dicts)
            if args.out:
                args.out.write_text(text, encoding="utf-8")
                print(f"wrote {len(dicts)} records to {args.out}")  # noqa: T201
            else:
                print(text)  # noqa: T201
        else:
            for i, p in enumerate(res.papers, 1):
                oa = "OA " if p.is_oa else "   "
                print(f"{i:3d}. {oa}{p.citation()}  [{', '.join(p.sources)}]")  # noqa: T201
        print(json.dumps({"raw": res.n_raw, "unique": len(res.papers),  # noqa: T201
                          "sources": res.source_status}))
    elif args.cmd == "journals":
        print(json.dumps(catalogue(args.subject, not args.offline), indent=2))  # noqa: T201
    else:
        print(json.dumps([o.to_dict() for o in resolve_access(args.doi)], indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
