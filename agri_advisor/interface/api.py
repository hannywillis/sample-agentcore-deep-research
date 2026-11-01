"""HTTP API exposing every access tier.

Run locally::

    uvicorn agri_advisor.interface.api:app --port 8000

Endpoints
---------
GET  /health                 liveness + knowledge-base / model status
POST /v1/advise              Tier 1 - full JSON (web / Android app)
POST /v1/extension           Tier 4 - extension-worker mediated query
POST /v1/sms                 Tier 2 - SMS webhook (Africa's Talking-style form)
POST /v1/ussd                Tier 2 - USSD session webhook (CON/END protocol)
POST /v1/voice               Tier 3 - numbered, TTS-ready text + SSML
POST /v1/predict             WP2 predictions only (uncertainty + XAI)
POST /v1/expert/verdict      record an expert-validation verdict
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from agri_advisor import __version__
from agri_advisor.config import FOCUS_CROPS, SUPPORTED_LANGUAGES, get_settings
from agri_advisor.engine.recommendation import RecommendationEngine
from agri_advisor.interface.formatters import (
    format_extension,
    format_sms,
    format_ussd_answer,
    format_voice,
    format_web,
)
from agri_advisor.nlp.language import detect_language
from agri_advisor.nlp.phrases import phrase
from agri_advisor.predict.schema import FarmContext

app = FastAPI(
    title="Evidence-Grounded Agricultural Decision Support API",
    version=__version__,
    description=(
        "CNPq PROAFRICA platform: RAG knowledge translation + predictive "
        "intelligence + expert validation, delivered through tiered access."
    ),
)


@lru_cache(maxsize=1)
def get_engine() -> RecommendationEngine:
    return RecommendationEngine(get_settings())


class AdviseRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    language: str | None = Field(None, description=f"One of {SUPPORTED_LANGUAGES}")
    context: FarmContext | None = None
    role: str | None = Field(None, description="public | extension | consortium")
    user_id: str | None = Field(None, description="Pseudonymised before logging")


class VoiceRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="ASR transcript")
    language: str | None = None
    caller_id: str | None = None


class PredictRequest(BaseModel):
    context: FarmContext


class VerdictRequest(BaseModel):
    review_id: str
    validator_id: str
    verdict: str
    scores: dict[str, int] | None = None
    comment: str = ""


def _check_language(lang: str | None) -> None:
    if lang and lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(422, f"language must be one of {SUPPORTED_LANGUAGES}")


@app.get("/health")
def health() -> dict[str, Any]:
    eng = get_engine()
    return {
        "status": "ok",
        "version": __version__,
        "knowledge_base": eng.retriever.store.stats(),
        "models": {k: m.meta.get("status") for k, m in eng.predictor.models.items()},
        "llm": getattr(eng.translator.llm, "name", None) or "none (extractive mode)",
        "languages": SUPPORTED_LANGUAGES,
    }


@app.post("/v1/advise")
def advise(req: AdviseRequest) -> dict[str, Any]:
    _check_language(req.language)
    rec = get_engine().recommend(
        req.query, req.context, req.language, "web", req.role, req.user_id
    )
    return format_web(rec)


@app.post("/v1/extension")
def extension(req: AdviseRequest) -> dict[str, Any]:
    _check_language(req.language)
    rec = get_engine().recommend(
        req.query,
        req.context,
        req.language,
        "extension",
        req.role or "extension",
        req.user_id,
    )
    return format_extension(rec)


@app.post("/v1/sms", response_class=PlainTextResponse)
def sms(
    text: str = Form(...),
    from_: str = Form("", alias="from"),
    language: str | None = Form(None),
) -> str:
    _check_language(language)
    rec = get_engine().recommend(text, None, language, "sms", "public", from_ or None)
    return format_sms(rec)


@app.post("/v1/ussd", response_class=PlainTextResponse)
def ussd(
    sessionId: str = Form(""),  # noqa: N803 - gateway field names
    phoneNumber: str = Form(""),  # noqa: N803
    text: str = Form(""),
    language: str | None = Form(None),
) -> str:
    """Menu: 1 = ask a question, 2 = crop risk tips, 3 = request expert call-back."""
    _check_language(language)
    steps = text.split("*") if text else []
    lang = language or (detect_language(steps[-1]) if len(steps) > 1 else "en")
    eng = get_engine()
    if not steps:
        return f"CON {phrase('ussd_welcome', lang)}"
    if steps[0] == "1":
        if len(steps) == 1:
            return f"CON {phrase('ussd_ask', lang)}"
        question = "*".join(steps[1:])
        rec = eng.recommend(
            question, None, language, "ussd", "public", phoneNumber or None
        )
        return f"END {format_ussd_answer(rec)}"
    if steps[0] == "2":
        if len(steps) == 1:
            return f"CON {phrase('ussd_crop', lang)}"
        try:
            crop = FOCUS_CROPS[int(steps[1]) - 1]
        except (ValueError, IndexError):
            return f"CON {phrase('ussd_crop', lang)}"
        rec = eng.recommend(
            f"main pests diseases and risks for {crop} and how to manage them",
            None,
            language or "en",
            "ussd",
            "public",
            phoneNumber or None,
        )
        return f"END {format_ussd_answer(rec)}"
    if steps[0] == "3":
        eng.protocol.enqueue(
            {"type": "callback_request", "session": sessionId, "phone": phoneNumber}
        )
        return f"END {phrase('ussd_expert', lang)}"
    return f"END {phrase('ussd_welcome', lang)}"


@app.post("/v1/voice")
def voice(req: VoiceRequest) -> dict[str, Any]:
    _check_language(req.language)
    rec = get_engine().recommend(
        req.query, None, req.language, "voice", "public", req.caller_id
    )
    return format_voice(rec)


@app.post("/v1/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    eng = get_engine()
    if not eng.predictor.available:
        raise HTTPException(503, "No predictive models available")
    raw = eng.predictor.predict(req.context)
    return {
        "models": {k: v.to_dict() for k, v in raw["models"].items()},
        "planting_window": raw["planting_window"],
        "soil_management": raw["soil_management"],
    }


@app.post("/v1/expert/verdict")
def expert_verdict(req: VerdictRequest) -> dict[str, Any]:
    try:
        return get_engine().protocol.record_verdict(
            req.review_id, req.validator_id, req.verdict, req.scores, req.comment
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
