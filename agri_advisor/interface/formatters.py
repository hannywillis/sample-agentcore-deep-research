"""Tier-specific renderers (WP3 tiered access model).

* Tier 1 ``web``       - full structured JSON (web / Android app)
* Tier 2 ``sms/ussd``  - compact plain text within carrier length limits
* Tier 3 ``voice``     - short numbered sentences (+ SSML) for IVR / TTS
* Tier 4 ``extension`` - detailed view for extension-worker mediated queries
"""

from __future__ import annotations

import re
from typing import Any

from agri_advisor.config import Settings, get_settings
from agri_advisor.engine.recommendation import Recommendation
from agri_advisor.nlp.phrases import phrase

_REF = re.compile(r"\s*\[S\d+(?:,\s*S\d+)*\]")

_VOICE_ORDINALS = {
    "en": ["First", "Second", "Third", "Fourth", "Fifth"],
    "sw": ["Kwanza", "Pili", "Tatu", "Nne", "Tano"],
    "fr": [
        "Premièrement",
        "Deuxièmement",
        "Troisièmement",
        "Quatrièmement",
        "Cinquièmement",
    ],
    "ha": ["Na farko", "Na biyu", "Na uku", "Na hudu", "Na biyar"],
    "pt": ["Primeiro", "Segundo", "Terceiro", "Quarto", "Quinto"],
}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


def _short_source(rec: Recommendation) -> str:
    if not rec.evidence_sources:
        return ""
    src = rec.evidence_sources[0]
    org = str(src.get("source_org", "")).split(" - ")[0][:20]
    year = src.get("year") or ""
    return f"{phrase('sources', rec.language)}: {org} {year}".strip()


def format_web(rec: Recommendation) -> dict[str, Any]:
    return rec.to_dict()


def format_extension(rec: Recommendation) -> dict[str, Any]:
    """Everything in ``web`` plus full evidence passages and model diagnostics."""
    out = rec.to_dict()
    out["for_extension_worker"] = {
        "how_to_use": (
            "Read the advice to the farmer, check it against local conditions, "
            "and record the farmer's decision. Items flagged consult_expert are "
            "queued for agronomist review."
        ),
        "model_explanations": {
            task: pred["explanation_summary"]
            for task, pred in (rec.predicted_outcome or {}).get("models", {}).items()
        },
    }
    return out


def format_sms(rec: Recommendation, settings: Settings | None = None) -> str:
    s = settings or get_settings()
    lang = rec.language
    if rec.advice_points:
        body = " ".join(_REF.sub("", p["text"]).strip() for p in rec.advice_points[:2])
    else:
        body = phrase("no_evidence", lang)
    tail = [f"{phrase('confidence', lang)}: {phrase(rec.confidence_level, lang)}."]
    if rec.consult_expert and rec.advice_points:
        tail.append(phrase("consult_expert", lang))
    src = _short_source(rec)
    if src:
        tail.append(src)
    tail_text = " ".join(tail)
    room = s.sms_max_chars - len(tail_text) - 1
    return f"{_truncate(body, max(room, 40))} {tail_text}"


def format_ussd_answer(rec: Recommendation, settings: Settings | None = None) -> str:
    s = settings or get_settings()
    lang = rec.language
    first = (
        _REF.sub("", rec.advice_points[0]["text"]).strip()
        if rec.advice_points
        else phrase("no_evidence", lang)
    )
    flag = " !" + phrase("consult_expert", lang) if rec.consult_expert else ""
    conf = f" ({phrase('confidence', lang)}: {phrase(rec.confidence_level, lang)})"
    room = s.ussd_max_chars - len(conf) - len(flag)
    return f"{_truncate(first, max(room, 40))}{conf}{flag}"[: s.ussd_max_chars]


def _speakable(text: str) -> str:
    text = _REF.sub("", text)
    text = text.replace("%", " percent").replace("&", " and ")
    text = re.sub(r"(\d+)\s*cm\b", r"\1 centimetres", text)
    text = re.sub(r"(\d+)\s*mm\b", r"\1 millimetres", text)
    text = re.sub(r"(\d+)\s*kg\b", r"\1 kilograms", text)
    text = re.sub(r"t/ha\b", "tonnes per hectare", text)
    return re.sub(r"\s+", " ", text).strip()


def format_voice(
    rec: Recommendation, settings: Settings | None = None
) -> dict[str, Any]:
    s = settings or get_settings()
    lang = rec.language
    ords = _VOICE_ORDINALS.get(lang, _VOICE_ORDINALS["en"])
    items = [_speakable(p["text"]) for p in rec.advice_points[: s.voice_max_items]]
    if not items:
        items = [phrase("no_evidence", lang)]
    lines = [f"{ords[i]}. {t}" for i, t in enumerate(items)]
    lines.append(f"{phrase('confidence', lang)}: {phrase(rec.confidence_level, lang)}.")
    if rec.consult_expert:
        lines.append(phrase("consult_expert", lang))
    if lang != "en" and not rec.translation.get("translated", True):
        lines.append(phrase("translation_unavailable", lang))
    text = "\n".join(lines)
    ssml = (
        f'<speak xml:lang="{lang}">'
        + "".join(f'<p>{_escape(line)}</p><break time="600ms"/>' for line in lines)
        + "</speak>"
    )
    return {
        "query_id": rec.query_id,
        "language": rec.language,
        "numbered_recommendations": items,
        "text": text,
        "ssml": ssml,
        "consult_expert": rec.consult_expert,
        "confidence_level": rec.confidence_level,
    }


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
