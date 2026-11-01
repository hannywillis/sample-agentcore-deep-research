"""Tests for literature search/reviews, journal-credential vault, and Kenyan
channels. All network access is mocked; gateways run in dry-run mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agri_advisor.config import get_settings
from agri_advisor.literature.models import Paper


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("research")
    mp = pytest.MonkeyPatch()
    mp.setenv("VECTOR_STORE_PATH", str(tmp / "store"))
    mp.setenv("MODEL_DIR", str(tmp / "models"))
    mp.setenv("EXPERT_REVIEW_QUEUE", str(tmp / "store" / "q.jsonl"))
    mp.setenv("LLM_PROVIDER", "none")
    mp.setenv("PLATFORM_DB", str(tmp / "platform.sqlite3"))
    mp.setenv("PRIVATE_LIBRARY_DIR", str(tmp / "library"))
    mp.setenv("CHANNELS_DRY_RUN", "true")
    mp.delenv("CHANNELS_WEBHOOK_SECRET", raising=False)
    mp.delenv("UNPAYWALL_EMAIL", raising=False)
    mp.delenv("LITERATURE_CONTACT_EMAIL", raising=False)
    get_settings(refresh=True)
    from agri_advisor.rag.ingest import main as ingest_main

    ingest_main(["--rebuild"])
    yield tmp
    mp.undo()
    get_settings(refresh=True)


def _paper(title: str, doi: str | None = None, **kw: Any) -> Paper:
    return Paper(title=title, doi=doi, **kw)


# ------------------------------------------------------------------ literature
def test_dedup_and_merge() -> None:
    from agri_advisor.literature.search import deduplicate

    a = _paper("Maize yield in Kenya", "https://doi.org/10.1/ABC", sources=["openalex"],
               cited_by=5)
    b = _paper("Maize Yield in Kenya.", "10.1/abc", sources=["crossref"],
               abstract="We found yields increased.", cited_by=9)
    c = _paper("Maize yield in Kenya", None, sources=["cgspace"], is_oa=True)
    out = deduplicate([a, b, c])
    assert len(out) == 1
    p = out[0]
    assert p.doi == "10.1/abc" and p.abstract and p.cited_by == 9 and p.is_oa
    assert set(p.sources) == {"openalex", "crossref", "cgspace"}


def test_federated_search_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from agri_advisor.literature import search as s

    def ok(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
        return [_paper("Drought tolerant maize Kenya", "10.9/x", year=2020, sources=["a"]),
                _paper("Unrelated rice paper", "10.9/y", year=2001, sources=["a"])]

    def dup(query: str, limit: int, keys: dict, **f: Any) -> list[Paper]:
        return [_paper("Drought tolerant maize Kenya", "10.9/X", sources=["b"], is_oa=True)]

    def boom(*a: Any, **k: Any) -> list[Paper]:
        raise RuntimeError("down")

    monkeypatch.setattr(s, "SOURCES", {"a": ok, "b": dup, "c": boom})
    res = s.federated_search("drought maize Kenya", ["a", "b", "c"], year_from=2010)
    assert res.source_status["c"]["ok"] is False and res.source_status["a"]["ok"]
    assert res.n_raw == 3 and [p.doi for p in res.papers] == ["10.9/x"]
    assert res.papers[0].is_oa


def test_openalex_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    from agri_advisor.literature import sources

    class R:
        def json(self) -> dict:
            return {"results": [{
                "id": "https://openalex.org/W1", "title": "Climate-smart maize",
                "publication_year": 2021, "doi": "https://doi.org/10.5/z",
                "abstract_inverted_index": {"Maize": [0], "yields": [1], "rose": [2]},
                "authorships": [{"author": {"display_name": "A. Otieno"}, "countries": ["KE"]}],
                "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "http://x/pdf"},
                "primary_location": {"source": {"display_name": "Field Crops Res"}},
                "cited_by_count": 3,
            }]}

    monkeypatch.setattr(sources, "_get", lambda *a, **k: R())
    p = sources.search_openalex("maize", 5, {})[0]
    assert p.abstract == "Maize yields rose" and p.countries == ["KE"] and p.doi == "10.5/z"
    assert p.source_ids["openalex"] == "W1" and p.pdf_url == "http://x/pdf"


def test_resolve_access_with_entitlements() -> None:
    from agri_advisor.literature.access import resolve_access

    creds = [
        {"provider": "elsevier", "label": "kalro-elsevier", "data": {"api_key": "k"}},
        {"provider": "ezproxy", "label": "uni-proxy",
         "data": {"prefix": "https://ezproxy.uni.ac.ke/login?url="}},
        {"provider": "openathens", "label": "oa", "data": {"domain": "kalro.org"}},
    ]
    opts = resolve_access("10.1016/j.fcr.2020.1", credentials=creds, known_pdf_url="http://oa/pdf")
    methods = [o.method for o in opts]
    assert methods == ["open_access", "publisher_api", "proxy_link", "openathens"]
    assert opts[2].url == "https://ezproxy.uni.ac.ke/login?url=https://doi.org/10.1016/j.fcr.2020.1"
    assert opts[2].requires_browser_login and not opts[2].downloadable_by_server
    # Elsevier key must not apply to a non-Elsevier DOI
    assert all(o.method != "publisher_api" for o in resolve_access("10.1007/x", credentials=creds))


def test_review_workflow(tmp_path: Path) -> None:
    from agri_advisor.literature.library import ReviewLibrary

    lib = ReviewLibrary(tmp_path / "db.sqlite3")
    rv = lib.create_review(1, "DT maize", {"question": "Does DT maize raise yields?"})
    papers = [
        _paper("Drought tolerant maize field trial in Kenya", "10.1/a", year=2019,
               authors=["Otieno, A.", "Kim, B."], abstract="A field trial. Yields increased 20%.",
               is_oa=True, license="cc-by", venue="J Agri"),
        _paper("Rice blast survey", "10.1/b", year=2015, abstract="Household survey."),
        _paper("DT maize meta-analysis", "10.1/c", year=2022, abstract="A meta-analysis found gains."),
    ]
    assert lib.add_papers(rv["id"], papers, "dt maize", sources=["openalex"])["added"] == 3
    assert lib.add_papers(rv["id"], papers[:1], "dt maize 2")["merged_duplicates"] == 1
    ranked = lib.prioritise(rv["id"], ["maize", "drought"], ["rice"])
    assert ranked[0]["key"] == "doi:10.1/a" and ranked[-1]["suggestion"] == "likely exclude"
    with pytest.raises(ValueError):
        lib.screen(rv["id"], "doi:10.1/b", "exclude")  # reason required
    lib.screen(rv["id"], "doi:10.1/a", "include")
    lib.screen(rv["id"], "doi:10.1/b", "exclude", reason="wrong crop")
    lib.screen(rv["id"], "doi:10.1/c", "include")
    lib.screen(rv["id"], "doi:10.1/c", "exclude", "full_text", reason="no Africa data")
    pr = lib.prisma(rv["id"])
    assert pr["identification"]["records_from_databases"] == 4
    assert pr["identification"]["duplicates_removed"] == 1
    assert pr["screening"]["records_excluded"] == 1
    assert pr["screening"]["reports_excluded"] == 1
    assert pr["screening"]["exclusion_reasons"] == {"no Africa data": 1}
    assert pr["included"]["studies"] == 1
    bib = lib.export(rv["id"], "bibtex")
    assert bib.startswith("@article{otieno2019drought") and "doi = {10.1/a}" in bib
    ris = lib.export(rv["id"], "ris", "all")
    assert ris.count("ER  - ") == 3 and "AU  - Otieno, A." in ris
    assert lib.export(rv["id"], "csv").splitlines()[0].startswith("key,title")
    syn = lib.synthesis(rv["id"])
    assert syn["mode"] == "extractive" and "Yields increased 20%" in syn["summary"]
    assert syn["evidence_table"][0]["design"] == "field / on-farm trial"
    kb = lib.ingest_to_kb(rv["id"], tmp_path / "kb")
    assert kb["written"] == 1
    from agri_advisor.rag.documents import load_document

    doc = load_document(Path(kb["files"][0]))
    assert doc.metadata.access_level == "open" and "Yields increased" in doc.text
    assert lib.get_review(rv["id"], 2) is None  # other users cannot read


# ---------------------------------------------------------------- accounts
def test_account_vault(tmp_path: Path) -> None:
    from agri_advisor.accounts.store import AccountStore

    st = AccountStore(tmp_path / "p.sqlite3")
    st.create_org("KALRO", "kalro", "kalro.org")
    with pytest.raises(ValueError):
        st.register("x@gmail.com", "password123", org_slug="kalro")
    admin = st.register("lib@kalro.org", "password123", org_slug="kalro", org_admin=True)
    member = st.register("sci@kalro.org", "password123", org_slug="kalro")
    outsider = st.register("ext@uni.ac.ke", "password123")
    assert st.authenticate("sci@kalro.org", "wrong") is None
    tok = st.issue_token(member)
    assert st.verify_token(tok).id == member.id
    assert st.verify_token(tok[:-2] + "00") is None and st.verify_token("garbage") is None
    with pytest.raises(PermissionError):
        st.add_credential(member, "ezproxy", {"prefix": "p"}, for_org=True)
    st.add_credential(admin, "ezproxy", {"prefix": "https://ez/login?url=", "password": "s3cretpw"},
                      "KALRO proxy", for_org=True)
    st.add_credential(member, "openalex", {"api_key": "abcdefghij"})
    listed = st.list_credentials(member)
    assert {c["label"] for c in listed} == {"KALRO proxy", "user:openalex"}
    assert all("s3cretpw" not in str(c) and "abcdefghij" not in str(c) for c in listed)
    assert st.credentials_for(member)[0]["scope"] == "user"  # personal first
    assert st.search_keys_for(member) == {"openalex_api_key": "abcdefghij"}
    assert st.credentials_for(outsider) == []
    raw = (tmp_path / "p.sqlite3").read_bytes()
    assert b"s3cretpw" not in raw  # encrypted at rest
    with pytest.raises(PermissionError):
        st.delete_credential(member, listed[1]["id"] if listed[1]["scope"] == "org" else listed[0]["id"])


def test_research_api_flow(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from agri_advisor.interface import research_api
    from agri_advisor.interface.api import app

    c = TestClient(app)
    assert c.get("/v1/auth/me").status_code == 401
    r = c.post("/v1/auth/register", json={"email": "a@kalro.org", "password": "password123"})
    assert r.status_code == 200
    h = {"Authorization": f"Bearer {r.json()['token']}"}
    assert c.post("/v1/auth/register", json={"email": "b@x.org", "password": "password123",
                                             "role": "admin"}).status_code == 422
    assert c.post("/v1/auth/login", json={"email": "a@kalro.org", "password": "bad"}).status_code == 401
    cred = c.post("/v1/credentials", headers=h, json={
        "provider": "ezproxy", "data": {"prefix": "https://ez.kalro.org/login?url="}})
    assert cred.status_code == 200
    acc = c.post("/v1/literature/access", headers=h, json={"doi": "10.1/q"}).json()
    assert any(o["url"].startswith("https://ez.kalro.org/login?url=") for o in acc["options"])

    def fake_search(query: str, *a: Any, **k: Any):  # type: ignore[no-untyped-def]
        from agri_advisor.literature.search import SearchResult

        return SearchResult(query, [_paper("Maize Kenya", "10.2/m", abstract="Yield rose.")],
                            {"openalex": {"ok": True, "n": 1}}, 1)

    monkeypatch.setattr(research_api, "federated_search", fake_search)
    rv = c.post("/v1/reviews", headers=h, json={"title": "Maize review", "question": "q?"}).json()
    s = c.post("/v1/literature/search", headers=h,
               json={"query": "maize", "review_id": rv["id"]}).json()
    assert s["saved"]["added"] == 1
    assert c.post(f"/v1/reviews/{rv['id']}/screen", headers=h,
                  json={"key": "doi:10.2/m", "decision": "include"}).status_code == 200
    assert c.get(f"/v1/reviews/{rv['id']}/prisma", headers=h).json()["included"]["studies"] == 1
    bib = c.get(f"/v1/reviews/{rv['id']}/export?fmt=bibtex", headers=h)
    assert bib.status_code == 200 and "10.2/m" in bib.text
    assert c.get(f"/v1/reviews/{rv['id']}/synthesis?use_llm=false", headers=h).json()["mode"] == "extractive"
    # anonymous callers cannot claim privileged governance roles
    assert c.post("/v1/advise", json={"query": "maize", "role": "consortium"}).status_code == 401
    assert c.post("/v1/advise", headers=h, json={"query": "maize armyworm"}).status_code == 200
    assert c.get("/v1/literature/journals?live=false").json()["curated"]


# ---------------------------------------------------------------- channels
@pytest.mark.parametrize("num,op", [
    ("0722139278", "safaricom"), ("+254110000000", "safaricom"), ("0733000000", "airtel"),
    ("254100123456", "airtel"), ("0771000000", "telkom"), ("0747000000", "faiba"),
])
def test_operator_detection(num: str, op: str) -> None:
    from agri_advisor.channels.kenya import detect_operator, normalise_msisdn

    assert normalise_msisdn(num).startswith("+254")
    assert detect_operator(num)["operator"] == op


def test_operator_network_code_and_invalid() -> None:
    from agri_advisor.channels.kenya import detect_operator, normalise_msisdn

    assert detect_operator("0722000000", "63903") == {"operator": "airtel", "basis": "network_code"}
    assert normalise_msisdn("12345") is None and normalise_msisdn("+25472") is None
    assert detect_operator("+5584999999999")["basis"] == "not_kenyan"


def test_gateways_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from agri_advisor.channels import gateways

    monkeypatch.setenv("CHANNELS_DRY_RUN", "true")
    monkeypatch.setenv("SMS_ROUTE_TELKOM", "http")
    res = gateways.send_message(["0722000000", "0771000000", "bad"], "Hello")
    by_op = {r["meta"].get("operator"): r for r in res if r["gateway"] != "none"}
    assert by_op["safaricom"]["gateway"] == "africastalking" and by_op["safaricom"]["dry_run"]
    assert by_op["telkom"]["gateway"] == "http_telkom"
    assert any(r["error"] == "invalid MSISDN" for r in res)
    g = gateways.GenericHTTPSMS("airtel")
    g.template = '{"msisdn": "{to}", "text": "{message}"}'
    assert g.build_body("254733", 'say "hi"') == {"msisdn": "254733", "text": 'say "hi"'}
    p = gateways.Daraja().stk_payload("0722000000", 10, "REF", "desc", ts="20260101120000")
    assert p["PartyA"] == "254722000000" and p["Timestamp"] == "20260101120000"


def test_africastalking_webhooks(env: Path) -> None:
    from fastapi.testclient import TestClient

    from agri_advisor.interface.api import app

    c = TestClient(app)
    menu = c.post("/v1/channels/at/ussd", data={
        "sessionId": "s1", "serviceCode": "*384*1#", "phoneNumber": "+254722000000",
        "text": "", "networkCode": "63902"})
    assert menu.text.startswith("CON Ushauri")  # Kiswahili default for +254
    ans = c.post("/v1/channels/at/ussd", data={
        "sessionId": "s1", "phoneNumber": "+254733000000", "text": "1*maize armyworm control"})
    assert ans.text.startswith("END")
    sms = c.post("/v1/channels/at/sms", data={"from": "+254771000000", "to": "12345",
                                               "text": "wheat rust"}).json()
    assert sms["operator"]["operator"] == "telkom" and sms["delivery"][0]["dry_run"]
    v = c.post("/v1/channels/at/voice", data={"isActive": "1", "sessionId": "v",
                                               "callerNumber": "+254722000000"})
    assert "<GetDigits" in v.text and "mahindi" in v.text
    v2 = c.post("/v1/channels/at/voice", data={"isActive": "1", "sessionId": "v",
                                                "callerNumber": "+254722000000", "dtmfDigits": "1#"})
    assert v2.text.startswith("<?xml") and "<Say" in v2.text
    assert c.get("/v1/channels/operator?msisdn=0110123456").json()["operator"] == "safaricom"


def test_whatsapp_webhook(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from agri_advisor.interface.api import app

    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "tok")
    c = TestClient(app)
    ok = c.get("/v1/channels/whatsapp", params={"hub.mode": "subscribe",
                                                "hub.verify_token": "tok", "hub.challenge": "42"})
    assert ok.text == "42"
    assert c.get("/v1/channels/whatsapp", params={"hub.mode": "subscribe",
                                                  "hub.verify_token": "x"}).status_code == 403
    payload = {"entry": [{"changes": [{"value": {"messages": [
        {"from": "254722000000", "type": "text", "text": {"body": "maize armyworm"}}]}}]}]}
    r = c.post("/v1/channels/whatsapp", json=payload).json()
    assert r["handled"] == 1 and r["replies"][0]["delivery"][0]["gateway"] == "whatsapp"


# ----------------------------------------------------------- open-data train
def test_train_from_csv_temporal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import csv

    from agri_advisor.predict import train
    from agri_advisor.predict.schema import FEATURES, NUMERIC_FEATURES
    from agri_advisor.predict.synthetic import generate

    syn = generate(400, seed=3)
    path = tmp_path / "d.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([*NUMERIC_FEATURES, "crop", "year", "yield_t_ha", "climate_risk"])
        for i in range(400):
            row = [syn["X"][i, FEATURES.index(f)] for f in NUMERIC_FEATURES]
            w.writerow([*row, "maize", 2000 + i % 20, syn["yield_t_ha"][i], syn["climate_risk"][i]])
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
    get_settings(refresh=True)
    try:
        rep = train.train_from_csv(path, "random_forest", "test", test_start=2016)
    finally:
        monkeypatch.undo()
        get_settings(refresh=True)
    assert set(rep) == {"crop_yield", "climate_risk", "pest_risk"}
    assert rep["crop_yield"]["split"].startswith("temporal")
