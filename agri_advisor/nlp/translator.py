"""Knowledge translation: evidence + predictions -> farmer-friendly advice.

Two modes:
* **LLM mode** (``LLM_PROVIDER=bedrock|openai``): grounded generation with the
  prompts in :mod:`agri_advisor.nlp.prompts`; output is validated so that every
  advice point cites evidence that was actually retrieved.
* **Extractive mode** (no LLM configured, or LLM failure): selects the most
  relevant evidence sentences verbatim. Fully traceable, works offline, but
  only in the language of the source documents.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agri_advisor.nlp.language import tokenize as _tokens
from agri_advisor.nlp.llm import LLMClient
from agri_advisor.nlp.prompts import (
    LANGUAGE_NAMES,
    QUERY_TRANSLATION_PROMPT,
    SYSTEM_PROMPT,
    TIER_STYLE,
    USER_PROMPT,
    render_evidence,
)
from agri_advisor.rag.retriever import Evidence

logger = logging.getLogger(__name__)

_STOP = {
    "the",
    "and",
    "for",
    "with",
    "how",
    "what",
    "when",
    "should",
    "can",
    "my",
    "this",
    "that",
    "are",
    "is",
    "to",
    "of",
    "in",
    "on",
    "do",
    "i",
    "a",
    "it",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TIER_POINTS = {"web": 5, "extension": 7, "sms": 2, "ussd": 1, "voice": 3}


@dataclass
class AdvicePoint:
    text: str
    sources: list[str]


@dataclass
class TranslationResult:
    summary: str
    advice_points: list[AdvicePoint]
    language: str
    mode: str  # "llm" | "extractive"
    insufficient_evidence: bool = False
    self_assessed_confidence: str | None = None
    translated: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "advice_points": [p.__dict__ for p in self.advice_points],
            "language": self.language,
            "mode": self.mode,
            "insufficient_evidence": self.insufficient_evidence,
            "self_assessed_confidence": self.self_assessed_confidence,
            "translated": self.translated,
            "warnings": self.warnings,
        }


def _extract_json(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in LLM output")
    return dict(json.loads(raw[start : end + 1]))


class KnowledgeTranslator:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    # ------------------------------------------------------------ query side
    def to_retrieval_query(self, query: str, language: str) -> str:
        """Translate a non-English query to English for retrieval (LLM mode)."""
        if language == "en" or self.llm is None:
            return query
        try:
            out = self.llm.complete(
                "You are a precise translator.",
                QUERY_TRANSLATION_PROMPT.format(query=query),
            ).strip()
            return out or query
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Query translation failed: %s", exc)
            return query

    # ----------------------------------------------------------- answer side
    def translate(
        self,
        query: str,
        evidence: list[Evidence],
        language: str,
        tier: str = "web",
        context_text: str = "",
        predictions_text: str = "",
        retrieval_query: str | None = None,
    ) -> TranslationResult:
        if self.llm is not None and evidence:
            try:
                return self._llm_translate(
                    query, evidence, language, tier, context_text, predictions_text
                )
            except Exception as exc:
                logger.warning("LLM translation failed, using extractive: %s", exc)
                result = self._extractive(
                    retrieval_query or query, evidence, language, tier
                )
                result.warnings.append(f"llm_failed: {type(exc).__name__}")
                return result
        return self._extractive(retrieval_query or query, evidence, language, tier)

    def _llm_translate(
        self,
        query: str,
        evidence: list[Evidence],
        language: str,
        tier: str,
        context_text: str,
        predictions_text: str,
    ) -> TranslationResult:
        assert self.llm is not None
        lang_name = LANGUAGE_NAMES.get(language, "English")
        system = SYSTEM_PROMPT.format(
            language_name=lang_name, tier_style=TIER_STYLE.get(tier, TIER_STYLE["web"])
        )
        user = USER_PROMPT.format(
            language_name=lang_name,
            query=query,
            context=context_text or "(not provided)",
            predictions=predictions_text or "(none)",
            evidence=render_evidence([e.to_dict() for e in evidence]),
        )
        data = _extract_json(self.llm.complete(system, user))

        valid_refs = {e.ref for e in evidence}
        points: list[AdvicePoint] = []
        warnings: list[str] = []
        for item in data.get("advice_points", []):
            refs = [r for r in item.get("sources", []) if r in valid_refs]
            if not refs:
                # Traceability guard: drop claims that cite nothing we retrieved.
                warnings.append("dropped_uncited_point")
                continue
            points.append(AdvicePoint(str(item.get("text", "")).strip(), refs))
        insufficient = bool(data.get("insufficient_evidence")) or not points
        return TranslationResult(
            summary=str(data.get("summary", "")).strip(),
            advice_points=points[: _TIER_POINTS.get(tier, 5)],
            language=language,
            mode="llm",
            insufficient_evidence=insufficient,
            self_assessed_confidence=data.get("self_assessed_confidence"),
            warnings=warnings,
        )

    def _extractive(
        self, query: str, evidence: list[Evidence], language: str, tier: str
    ) -> TranslationResult:
        if not evidence:
            return TranslationResult(
                summary="",
                advice_points=[],
                language=language,
                mode="extractive",
                insufficient_evidence=True,
                translated=language == "en",
            )
        q_terms = {t for t in _tokens(query) if t not in _STOP and len(t) > 2}
        scored: list[tuple[float, str, str]] = []
        for rank, ev in enumerate(evidence):
            for sent in _SENT_SPLIT.split(ev.text):
                sent = re.sub(r"\s*\([^)]*\)", "", sent).strip()
                if len(sent) < 25 or "illustrative sample" in sent.lower():
                    continue
                # Cap overlap so actionable sentences beat merely descriptive ones.
                overlap = min(len(q_terms & set(_tokens(sent))), 2)
                actionable = (
                    1.5
                    if re.match(
                        r"^(plant|apply|use|keep|dry|store|test|check|walk|rotate|"
                        r"choose|grow|avoid|sort|harvest|top-dress|hand-pick|level)",
                        sent.lower(),
                    )
                    else 0.0
                )
                score = overlap + actionable + ev.score - 0.15 * rank
                scored.append((score, sent, ev.ref))
        scored.sort(key=lambda t: -t[0])
        seen: set[str] = set()
        points: list[AdvicePoint] = []
        for _, sent, ref in scored:
            if sent in seen:
                continue
            seen.add(sent)
            points.append(AdvicePoint(sent, [ref]))
            if len(points) >= _TIER_POINTS.get(tier, 5):
                break
        return TranslationResult(
            summary=points[0].text if points else "",
            advice_points=points,
            language=language,
            mode="extractive",
            insufficient_evidence=not points,
            translated=language == "en",
        )
