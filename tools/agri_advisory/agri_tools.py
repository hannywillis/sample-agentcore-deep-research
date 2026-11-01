"""Strands ``@tool`` wrappers around the agri_advisor platform.

These let the existing AgentCore deep-research agent (Strands) answer
agricultural questions with the evidence-grounded RAG knowledge base (WP1) and
the context-aware recommendation engine (WP1 + WP2). Enable them by setting
``AGRI_MODE=true`` on the agent runtime.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from strands import tool

from agri_advisor.engine.recommendation import RecommendationEngine
from agri_advisor.nlp.language import detect_crop, detect_language, expand_query
from agri_advisor.predict.schema import FarmContext
from agri_advisor.rag.retriever import KnowledgeRetriever


@lru_cache(maxsize=1)
def _retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()


@lru_cache(maxsize=1)
def _engine() -> RecommendationEngine:
    return RecommendationEngine()


@tool
def agri_knowledge_search(
    query: str,
    crop: str | None = None,
    country: str | None = None,
    top_k: int = 5,
) -> str:
    """Search the validated agricultural knowledge base (KALRO and partner docs).

    Returns evidence passages with full source attribution (reference id,
    citation, organisation, year, URL, access level, validator) so every claim
    in the final answer can be traced back to its source document. Only
    ``open`` documents are visible to the agent (public role).

    Args:
        query: Farmer or researcher question (any supported language).
        crop: Optional focus crop (maize, wheat, sorghum, rice, millet).
        country: Optional ISO-2 country code (KE, UG, NE, NG, GH, BR).
        top_k: Maximum number of evidence passages to return.
    """
    lang = detect_language(query)
    retrieval_query = expand_query(query, lang, english_only=lang != "en")
    evidence = _retriever().retrieve(
        retrieval_query,
        k=top_k,
        role="public",
        crop=crop or detect_crop(retrieval_query),
        country=country,
    )
    return json.dumps(
        {"language": lang, "evidence": [e.to_dict() for e in evidence]},
        ensure_ascii=False,
    )


@tool
def agri_recommendation(
    query: str,
    farm_context_json: str | None = None,
    language: str | None = None,
    tier: str = "web",
) -> str:
    """Produce a unified, evidence-grounded farmer recommendation.

    Combines RAG evidence (WP1) with predictive models (WP2: yield, pest and
    climate risk with conformal intervals and SHAP explanations), then applies
    the expert-validation protocol. The result contains advisory text, cited
    evidence sources, predicted outcome, confidence level and a
    ``consult_expert`` flag. Always surface the flag and citations to the user.

    Args:
        query: The farmer's question.
        farm_context_json: Optional JSON matching ``FarmContext`` (location,
            climate, soil, remote_sensing, farm). Enables predictions.
        language: Optional ISO code (en, sw, fr, ha, pt, dje); auto-detected.
        tier: Output tier - web, extension, sms, ussd or voice.
    """
    context = None
    if farm_context_json:
        data: dict[str, Any] = json.loads(farm_context_json)
        context = FarmContext.model_validate(data)
    rec = _engine().recommend(query, context=context, language=language, tier=tier)
    return json.dumps(rec.to_dict(), ensure_ascii=False, default=str)


AGRI_TOOLS = [agri_knowledge_search, agri_recommendation]
