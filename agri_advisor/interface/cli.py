"""Command-line entry point for quick testing of every tier.

Examples::

    python -m agri_advisor "How do I control fall armyworm in maize?"
    python -m agri_advisor "Nifanye nini kuhusu viwavijeshi kwenye mahindi?" --tier sms
    python -m agri_advisor "When should I plant maize?" --tier voice \
        --context examples/farm_context_kenya_maize.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agri_advisor.engine.recommendation import TIERS, RecommendationEngine
from agri_advisor.interface.formatters import (
    format_extension,
    format_sms,
    format_ussd_answer,
    format_voice,
)
from agri_advisor.predict.schema import FarmContext


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ask the agricultural advisor")
    p.add_argument("query")
    p.add_argument("--tier", choices=TIERS, default="web")
    p.add_argument("--language", default=None)
    p.add_argument("--role", default=None)
    p.add_argument("--context", type=Path, help="JSON file matching FarmContext")
    args = p.parse_args(argv)

    ctx = (
        FarmContext.model_validate_json(args.context.read_text())
        if args.context
        else None
    )
    rec = RecommendationEngine().recommend(
        args.query, ctx, args.language, args.tier, args.role
    )
    if args.tier == "sms":
        out: object = format_sms(rec)
    elif args.tier == "ussd":
        out = format_ussd_answer(rec)
    elif args.tier == "voice":
        out = format_voice(rec)["text"]
    elif args.tier == "extension":
        out = json.dumps(format_extension(rec), indent=2, ensure_ascii=False)
    else:
        out = (
            rec.advisory_text
            + "\n\n"
            + json.dumps(
                {
                    "confidence_level": rec.confidence_level,
                    "consult_expert": rec.consult_expert,
                    "expert_reasons": rec.expert_reasons,
                    "sources": [
                        f"[{e['ref']}] {e['citation']} - {e['section']}"
                        for e in rec.evidence_sources
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    print(out)  # noqa: T201


if __name__ == "__main__":
    main()
