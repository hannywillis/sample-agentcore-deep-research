"""Context-Aware Recommendation Engine (WP2 Task / deliverable D2.7).

Converges the two architectural flows of the proposal:

    Evidence flow : query -> language detection -> RAG retrieval
                    -> knowledge translation -> traceable advice
    Data flow     : climate/soil/satellite/farm data -> ML models
                    -> predictions with uncertainty + explanation

and passes the merged result through the Expert Validation Protocol, so every
recommendation carries: advisory text, evidence sources, predicted outcome,
confidence level, and a consult-expert flag.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agri_advisor.config import Settings, get_settings
from agri_advisor.governance import parse_role
from agri_advisor.nlp.language import detect_crop, detect_language, expand_query
from agri_advisor.nlp.llm import get_llm
from agri_advisor.nlp.phrases import phrase
from agri_advisor.nlp.translator import KnowledgeTranslator, TranslationResult
from agri_advisor.predict.schema import FarmContext
from agri_advisor.predict.service import PredictiveService
from agri_advisor.predict.uncertainty import ConfidenceLevel, min_level
from agri_advisor.rag.retriever import Evidence, KnowledgeRetriever
from agri_advisor.validation.protocol import ExpertValidationProtocol, ValidationResult

logger = logging.getLogger(__name__)

TIERS = ("web", "extension", "sms", "ussd", "voice")


@dataclass
class Recommendation:
    query_id: str
    created_at: str
    query: str
    language: str
    tier: str
    crop: str | None
    country: str | None
    summary: str
    advice_points: list[dict[str, Any]]
    advisory_text: str
    evidence_sources: list[dict[str, Any]]
    predicted_outcome: dict[str, Any] | None
    confidence_level: str
    confidence_score: float
    consult_expert: bool
    expert_reasons: list[str]
    risk_level: str
    validation: dict[str, Any]
    translation: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _evidence_confidence(
    evidence: list[Evidence], insufficient: bool
) -> tuple[ConfidenceLevel, float]:
    if insufficient or not evidence:
        return ConfidenceLevel.LOW, 0.0
    top = evidence[0].score
    support = min(len(evidence), 3) / 3
    score = float(min(1.0, (top / 0.35) * 0.7 + support * 0.3))
    if top >= 0.28 and len(evidence) >= 2:
        return ConfidenceLevel.HIGH, score
    if top >= 0.18:
        return ConfidenceLevel.MEDIUM, score
    return ConfidenceLevel.LOW, score


class RecommendationEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: KnowledgeRetriever | None = None,
        predictor: PredictiveService | None = None,
        translator: KnowledgeTranslator | None = None,
        protocol: ExpertValidationProtocol | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or KnowledgeRetriever(settings=self.settings)
        self.predictor = predictor or PredictiveService(self.settings)
        self.translator = translator or KnowledgeTranslator(get_llm(self.settings))
        self.protocol = protocol or ExpertValidationProtocol(settings=self.settings)

    # ------------------------------------------------------------------ main
    def recommend(
        self,
        query: str,
        context: FarmContext | None = None,
        language: str | None = None,
        tier: str = "web",
        role: str | None = None,
        user_id: str | None = None,
    ) -> Recommendation:
        tier = tier if tier in TIERS else "web"
        role_value = parse_role(role or self.settings.default_access_role).value
        lang = language or detect_language(query, self.settings.default_language)

        # ---- Evidence flow --------------------------------------------------
        retrieval_query = self.translator.to_retrieval_query(query, lang)
        if retrieval_query == query and lang != "en":
            retrieval_query = expand_query(query, lang, english_only=True)
        explicit_crop = (
            context.farm.crop
            if context is not None and "crop" in context.farm.model_fields_set
            else None
        )
        crop = explicit_crop or detect_crop(retrieval_query) or detect_crop(query)
        if context is not None and explicit_crop is None and crop:
            context = context.model_copy(deep=True)
            context.farm.crop = crop  # type: ignore[assignment]
        country = context.location.country if context else None
        evidence = self.retriever.retrieve(
            retrieval_query, role=role_value, crop=crop, country=country
        )

        # ---- Data flow ------------------------------------------------------
        predictions: dict[str, Any] | None = None
        pred_dicts: dict[str, dict[str, Any]] = {}
        if context is not None and self.predictor.available:
            raw = self.predictor.predict(context)
            pred_dicts = {k: v.to_dict() for k, v in raw["models"].items()}
            predictions = {
                "models": pred_dicts,
                "planting_window": raw["planting_window"],
                "soil_management": raw["soil_management"],
            }

        # ---- Knowledge translation -------------------------------------------
        translation = self.translator.translate(
            query,
            evidence,
            lang,
            tier=tier,
            context_text=context.describe() if context else "",
            predictions_text=self._predictions_text(predictions),
            retrieval_query=retrieval_query,
        )

        # ---- Validation & confidence ----------------------------------------
        advice_text = " ".join(p.text for p in translation.advice_points)
        validation = self.protocol.evaluate(
            query=query,
            retrieval_query=retrieval_query,
            evidence_scores=[e.score for e in evidence],
            predictions=pred_dicts,
            advisory_text=advice_text,
            insufficient_evidence=translation.insufficient_evidence,
        )
        if validation.out_of_domain:
            # Never pass off loosely related agricultural text as an answer.
            evidence = []
            translation.advice_points = []
            translation.summary = ""
            translation.insufficient_evidence = True
        ev_level, ev_score = _evidence_confidence(
            evidence, translation.insufficient_evidence
        )
        levels = [ev_level] + [
            ConfidenceLevel(p["uncertainty"]["level"]) for p in pred_dicts.values()
        ]
        overall = min_level(*levels)
        scores = [ev_score] + [p["uncertainty"]["score"] for p in pred_dicts.values()]
        overall_score = round(sum(scores) / len(scores), 3)
        consult = validation.requires_expert or overall is ConfidenceLevel.LOW

        rec = Recommendation(
            query_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            query=query,
            language=lang,
            tier=tier,
            crop=crop,
            country=country,
            summary=translation.summary,
            advice_points=[p.__dict__ for p in translation.advice_points],
            advisory_text="",
            evidence_sources=[
                e.to_dict(include_text=tier in {"web", "extension"}) for e in evidence
            ],
            predicted_outcome=predictions,
            confidence_level=overall.value,
            confidence_score=overall_score,
            consult_expert=consult,
            expert_reasons=validation.reasons,
            risk_level=validation.risk_level.value,
            validation=validation.to_dict(),
            translation={
                "mode": translation.mode,
                "translated": translation.translated,
                "self_assessed_confidence": translation.self_assessed_confidence,
            },
            warnings=translation.warnings,
        )
        rec.advisory_text = self._compose_text(rec, translation, validation)
        if consult:
            rec.validation["review_id"] = self.protocol.enqueue(rec.to_dict())
        self._log_usage(rec, user_id)
        return rec

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _predictions_text(predictions: dict[str, Any] | None) -> str:
        if not predictions:
            return ""
        lines = [
            f"- {p['headline']} | confidence {p['uncertainty']['level']} | "
            f"{p['explanation_summary']} | model status {p['model_status']}"
            for p in predictions["models"].values()
        ]
        lines.append(f"- planting window: {predictions['planting_window']['advice']}")
        lines += [f"- soil: {a}" for a in predictions["soil_management"]["actions"]]
        return "\n".join(lines)

    @staticmethod
    def _compose_text(
        rec: Recommendation, tr: TranslationResult, val: ValidationResult
    ) -> str:
        lang = rec.language
        if not tr.advice_points:
            body = phrase("no_evidence", lang)
        else:
            body = "\n".join(
                f"{i}. {p['text']} [{', '.join(p['sources'])}]"
                for i, p in enumerate(rec.advice_points, 1)
            )
        parts = [body]
        if rec.predicted_outcome:
            heads = [p["headline"] for p in rec.predicted_outcome["models"].values()]
            parts.append(f"{phrase('prediction', lang)}: " + "; ".join(heads))
        parts.append(
            f"{phrase('confidence', lang)}: {phrase(rec.confidence_level, lang)}"
        )
        if rec.consult_expert and tr.advice_points:
            parts.append(phrase("consult_expert", lang))
        if not tr.translated and lang != "en":
            parts.append(phrase("translation_unavailable", lang))
        return "\n".join(parts)

    def _log_usage(self, rec: Recommendation, user_id: str | None) -> None:
        """Pseudonymised usage log for WP3 adoption/impact analysis (D3.4)."""
        try:
            path = self.settings.vector_store_path / "usage_log.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            uid = hashlib.sha256(user_id.encode()).hexdigest()[:16] if user_id else None
            entry = {
                "query_id": rec.query_id,
                "ts": rec.created_at,
                "user": uid,
                "tier": rec.tier,
                "language": rec.language,
                "crop": rec.crop,
                "country": rec.country,
                "confidence": rec.confidence_level,
                "consult_expert": rec.consult_expert,
                "n_sources": len(rec.evidence_sources),
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:  # pragma: no cover - logging must never break advice
            logger.warning("usage log failed: %s", exc)
