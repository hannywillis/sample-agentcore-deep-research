"""End-to-end tests for the agri_advisor decision-support platform."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agri_advisor.config import get_settings
from agri_advisor.governance import AccessLevel, Role, can_access

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("agri")
    mp = pytest.MonkeyPatch()
    mp.setenv("VECTOR_STORE_PATH", str(tmp / "store"))
    mp.setenv("MODEL_DIR", str(tmp / "models"))
    mp.setenv("EXPERT_REVIEW_QUEUE", str(tmp / "store" / "q.jsonl"))
    mp.setenv("LLM_PROVIDER", "none")
    get_settings(refresh=True)
    from agri_advisor.rag.ingest import main as ingest_main

    ingest_main(["--rebuild"])
    yield tmp
    mp.undo()
    get_settings(refresh=True)


def test_governance() -> None:
    assert can_access(Role.PUBLIC, AccessLevel.OPEN)
    assert not can_access(Role.PUBLIC, AccessLevel.RESTRICTED)
    assert can_access(Role.CONSORTIUM, AccessLevel.RESTRICTED)
    assert not can_access(Role.CONSORTIUM, AccessLevel.CLOSED)


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("How do I control fall armyworm in my maize?", "en"),
        ("Ninawezaje kudhibiti viwavi jeshi kwenye mahindi yangu?", "sw"),
        ("Comment lutter contre la chenille légionnaire sur le maïs ?", "fr"),
        ("Yaya zan magance tsutsar masara a gonata?", "ha"),
        ("Como controlar a lagarta-do-cartucho no milho?", "pt"),
    ],
)
def test_language_detection(text: str, lang: str) -> None:
    from agri_advisor.nlp.language import detect_language

    assert detect_language(text) == lang


def test_retrieval_traceability_and_governance(env: Path) -> None:
    from agri_advisor.rag.retriever import KnowledgeRetriever

    r = KnowledgeRetriever()
    ev = r.retrieve("fall armyworm maize control", role="public")
    assert ev and ev[0].citation and ev[0].ref.startswith("S")
    assert all(e.access_level == "open" for e in ev)
    restricted_q = "on-farm trial yield data restricted"
    pub = {e.access_level for e in r.retrieve(restricted_q, role="public", k=10)}
    con = {e.access_level for e in r.retrieve(restricted_q, role="consortium", k=10)}
    assert "restricted" not in pub and "restricted" in con


def test_engine_end_to_end(env: Path) -> None:
    from agri_advisor.engine.recommendation import RecommendationEngine
    from agri_advisor.predict.schema import FarmContext

    ctx = FarmContext.model_validate(
        json.loads((ROOT / "examples/farm_context_kenya_maize.json").read_text())
    )
    rec = RecommendationEngine().recommend(
        "How do I control fall armyworm in maize?", context=ctx
    )
    assert rec.evidence_sources and rec.predicted_outcome
    assert rec.confidence_level in {"low", "medium", "high"}
    assert rec.consult_expert  # scaffold models are never validated
    for pred in rec.predicted_outcome["models"].values():
        assert "uncertainty" in pred and "explanation" in pred

    ood = RecommendationEngine().recommend("What is the bitcoin price today?")
    assert ood.validation["out_of_domain"] and not ood.advice_points


def test_api_tiers(env: Path) -> None:
    from fastapi.testclient import TestClient

    from agri_advisor.interface.api import app

    c = TestClient(app)
    assert c.get("/health").status_code == 200
    r = c.post("/v1/advise", json={"query": "maize fall armyworm control"})
    assert r.status_code == 200 and "evidence_sources" in r.text
    s = get_settings()
    sms = c.post("/v1/sms", data={"text": "mahindi viwavi jeshi", "from": "+2547"})
    assert sms.status_code == 200 and len(sms.text) <= s.sms_max_chars + 50
    menu = c.post("/v1/ussd", data={"sessionId": "1", "phoneNumber": "1", "text": ""})
    assert menu.text.startswith("CON")
    ans = c.post(
        "/v1/ussd",
        data={"sessionId": "1", "phoneNumber": "1", "text": "1*wheat rust"},
    )
    assert ans.text.startswith("END") and len(ans.text) <= s.ussd_max_chars + 4
    v = c.post("/v1/voice", json={"query": "sorghum striga drought"})
    assert v.status_code == 200
