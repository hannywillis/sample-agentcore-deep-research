# Evidence-Grounded & Context-Aware AI for Agricultural Decision Support

An open platform that gives smallholder farmers, extension workers and
researchers **traceable, uncertainty-aware agricultural advice**. It is the
reference implementation for the CNPq PROAFRICA proposal *"Evidence-Grounded and
Context-Aware AI for Agricultural Decision Support in African and Brazilian
Smallholder Farming Systems"*.

- **Focus crops:** maize, wheat, sorghum, rice, millet
- **Countries:** Kenya, Uganda, Niger, Nigeria, Ghana, Brazil
- **Languages:** English, Kiswahili, French, Hausa, Portuguese, plus Zarma-Songhay (partial)

The platform is built on the [AgentCore Deep Research](docs/AGENTCORE_TEMPLATE.md)
template, which is kept intact (see the
[underlying template section](#underlying-agentcore-deep-research-template)).

> ⚠️ **Status: research prototype.** The documents in `data/knowledge_base/sample/`
> are *illustrative*, and the predictive models are trained on *synthetic* data.
> Neither is agronomically valid. Replace them with validated KALRO and partner
> content and real field data before any farmer-facing use.

---

## Architecture (three layers, two converging flows)

```
                       ┌──────────── WP1 Agricultural Knowledge Intelligence ───────────┐
 Farmer query ──► language detection ──► RAG retrieval (governance-filtered) ──► knowledge
 (any tier)                                 │ evidence + citations [S1..]         translation
                                            ▼                                        │
                       ┌──────── Context-Aware Recommendation Engine (D2.7) ◄────────┘
 Climate / soil /      │  advisory text · evidence sources · predicted outcome ·
 NDVI-EVI / farm ──► ML models (RF/XGBoost) ──► uncertainty (conformal) + SHAP ─┘
 data       └──────── WP2 Predictive Agricultural Intelligence ────────┘
                                            │
                                 Expert Validation Protocol ──► consult_expert flag / review queue
                                            │
              ┌──────────────── WP3 Tiered access ───────────────────────────────┐
              Web/App (full) · Extension worker · SMS · USSD · Voice/IVR (numbered)
```

| Layer | Package | Proposal deliverables |
|---|---|---|
| Configuration & governance | `agri_advisor/config.py`, `agri_advisor/governance.py`, `.env.example` | D1.1, D1.8 |
| RAG knowledge base | `agri_advisor/rag/`, `data/knowledge_base/` | D1.3, D1.4 |
| NLP knowledge translation & multilingual support | `agri_advisor/nlp/` | D1.5, D1.6 |
| Expert validation | `agri_advisor/validation/`, `config/expert_validation.yaml` | D1.7 |
| Predictive intelligence | `agri_advisor/predict/` | D2.1, D2.4, D2.6 |
| Recommendation engine | `agri_advisor/engine/` | D2.7 |
| Tiered access interface | `agri_advisor/interface/` (FastAPI + CLI) | D2.8, WP3 |
| AgentCore agent integration | `tools/agri_advisory/`, `AGRI_MODE` | — |

A full requirement-by-requirement mapping is in [docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md).

## Modules

### RAG knowledge base (`agri_advisor/rag`)
- `documents.py` loads `.md/.txt` files (YAML front-matter) and `.pdf/.docx` files (sidecar `*.meta.yaml`). The provenance metadata covers organisation, year, URL, crops, countries, language, access level and validator.
- `chunking.py` does heading-aware sliding-window chunking.
- `embeddings.py` offers three embedders: `hashing` (offline default), `sentence-transformers` (multilingual) and `bedrock` (Titan).
- `vector_store.py` is a local NumPy cosine store.
- `retriever.py` returns `Evidence` objects with reference id, citation, source organisation, URL, access level and score. It boosts matching crop, country and validated documents, and it **filters by the caller's governance role** before ranking.
- `ingest.py` is the CLI: `--rebuild`, `--source`, `--stats`.

**Pipeline:** documents → metadata validation → chunks → embeddings → vector
store → query (expanded to English for retrieval) → governance mask → top-k
evidence → knowledge translation with `[S#]` citations → each citation resolves
to its source document.

### NLP knowledge translation (`agri_advisor/nlp`)
- Language detection for en/sw/fr/ha/pt/dje uses marker words, with `langdetect` as a fallback. An agricultural glossary (e.g. *mahindi*, *masara*, *milho* → maize) expands queries.
- The system prompt enforces these rules:
  - use only the evidence provided, and cite every point `[S#]`
  - write in plain farmer language
  - reply in the user's language
  - include pesticide safety, and declare when evidence is insufficient
- Points without a citation are dropped.
- LLM backends: Amazon Bedrock (Converse) or any OpenAI-compatible endpoint. With `LLM_PROVIDER=none`, an **extractive fallback** answers in English and adds a note that translation is unavailable.

### Predictive intelligence (`agri_advisor/predict`)
- `schema.py` holds the Pydantic `FarmContext`: location, climate, soil, remote sensing (NDVI/EVI) and farm data.
- `models.py` contains `CropYieldModel`, `PestRiskModel` and `ClimateRiskModel`:
  - backend is Random Forest or XGBoost
  - hold-out metrics
  - `scaffold-synthetic` vs `validated` status
  - joblib persistence
  - `TODO` markers show where real data slots in
- `uncertainty.py` provides split-conformal prediction intervals, ensemble disagreement and a missing-input penalty, which together give a **confidence level** (low / medium / high).
- `explain.py` gives SHAP TreeExplainer top drivers, with an importance × deviation fallback.
- `advisors.py` holds rule-based planting-window and soil-management baselines.
- `train.py` is the CLI: `--synthetic` or `--data your.csv --backend xgboost --status validated`.

### Expert validation (`agri_advisor/validation`, `config/expert_validation.yaml`)
A recommendation is escalated (`consult_expert=true`, then queued for review)
when any of these is true:
- confidence is below threshold
- evidence is missing or weak
- the query is out of domain
- the query contains high-risk keywords (pesticides/chemicals, health and safety, costly or regulated actions), in multiple languages
- a high-risk predicted outcome appears (e.g. high pest risk)
- the model is not yet validated

Expert verdicts are recorded via `/v1/expert/verdict`, using a two-validator rubric.

### Recommendation engine (`agri_advisor/engine`)
`RecommendationEngine.recommend(query, context, language, tier, role)` returns:
`summary`, `advice_points`, `advisory_text`, `evidence_sources`,
`predicted_outcome`, `confidence_level`, `confidence_score`, `consult_expert`,
`expert_reasons`, `risk_level` and `warnings`. It also writes a pseudonymised
usage log for impact evaluation (WP3).

### Tiered access (`agri_advisor/interface`)
| Tier | Endpoint | Output |
|---|---|---|
| 1 Web / app (full) | `POST /v1/advise` | full JSON recommendation |
| 4 Extension worker | `POST /v1/extension` | detailed briefing including model drivers |
| 2 SMS | `POST /v1/sms` (form `text`, `from`) | plain text ≤ `SMS_MAX_CHARS` (3 SMS) |
| 2 USSD | `POST /v1/ussd` (`sessionId`, `phoneNumber`, `text`) | `CON`/`END` menu, ≤ 182 chars |
| 3 Voice / IVR | `POST /v1/voice` | numbered, speakable script + SSML |
| Predictions only | `POST /v1/predict` | model outputs, intervals, explanations |

The SMS and USSD formats follow the Africa's Talking webhook conventions.

### Model retraining on open data (`predict/open_data.py`)
`python -m agri_advisor.predict.open_data` downloads **FAOSTAT** national crop
yields (bulk QCL Africa file) and **NASA POWER** daily climate for one cropping
point in each of 20 African countries. It then builds
`data/open_data/harmonised_country_season.csv` (3,954 country-crop-season rows,
1982-2024, five focus cereals). `python -m agri_advisor.predict.train --open-data`
retrains on that file with a **temporal split** (train < 2016, test ≥ 2016) and
reports skill against naive baselines:

| Model | Data | Hold-out result (2016-2024) | Naive baseline | Reading |
|---|---|---|---|---|
| Crop yield (residual on last-3-season mean) | open data | MAE 0.200 t/ha, RMSE 0.40, R² 0.89, conformal q̂ 0.43 | persistence MAE 0.200 | Matches persistence only; climate adds little at national scale |
| Climate risk (yield shock class) | open data | AUC 0.59, accuracy 0.728 | majority class 0.729 | Weak skill; do not use for decisions |
| Pest risk | synthetic (no open labels) | AUC 0.71 | - | Scaffold only |

These are **national, ecological-scale** models. They are honest benchmarks,
not farm-level advice. Farm/plot data (D2.3) is needed to go further. Status is
recorded as `trained-open-data-national`, so the validation layer still escalates
to experts.

### Research literature & systematic reviews (`agri_advisor/literature`)
- **Federated open search** over OpenAlex, Crossref, DOAJ, Europe PMC, CGSpace
  (CGIAR), Zenodo, arXiv, CORE (free key) and Semantic Scholar. Sources are
  queried in parallel. Results are de-duplicated by DOI and title, merged
  (abstract, OA links, citations) and ranked, and each source reports its own status.
- **Journal catalogue**: a curated list of OA agricultural journals and repositories
  (AJOL, AJFAND, African Crop Science Journal, CGSpace, AGRIS, AgEcon Search,
  Embrapa, KALRO, Research4Life …) plus a **live DOAJ journal search** by subject.
- **Review workspace** (`library.py`):
  - projects with a protocol (question, PICO, inclusion/exclusion);
  - every search is logged;
  - title/abstract and full-text screening, with exclusion reasons required;
  - keyword prioritisation;
  - forward/backward **snowballing**;
  - **PRISMA 2020** counts;
  - export to **BibTeX / RIS / CSV** (Zotero, Mendeley, EndNote, Rayyan);
  - an evidence table and synthesis: LLM if configured, otherwise extractive and labelled;
  - optional push of open-licence included studies into the RAG knowledge base.
- CLI: `python -m agri_advisor.literature search "drought tolerant maize Kenya" --oa --export bibtex --out dt.bib`
- Agent tools (`AGRI_MODE=true`): `literature_search`, `literature_access`.

### Personal & institutional journal login (`agri_advisor/accounts`)
Users register and log in (`/v1/auth/*`, bearer token). An **organisation admin**
(e.g. the KALRO librarian) registers institutional entitlements once with
`for_org=true`, and every member inherits them. Users can add personal ones too;
personal entries take precedence. Secrets are **Fernet-encrypted at rest** and
only ever returned masked.

| Provider | What you store | What the platform does |
|---|---|---|
| `ezproxy` / `research4life` | proxy prefix, e.g. `https://ezproxy.uni.ac.ke/login?url=` | builds proxied links. The server only downloads if you also store the username/password **and** set `ALLOW_PROXY_SESSION_LOGIN=true` (check your licence first) |
| `openathens` | your OpenAthens domain | builds redirector links that open the publisher after you sign in |
| `link_template` | WAYFless / Shibboleth / link-resolver URL with `{doi}` | builds the link |
| `elsevier`, `springer`, `wiley` | institutional TDM API key / insttoken / token | **server-side full-text download** via the publisher's TDM API |
| `unpaywall`, `openalex`, `core`, `semantic_scholar` | email / API keys | better OA discovery and higher search quotas |

`POST /v1/literature/access {doi}` returns every route this user has to the full
text, open copies first. `POST /v1/literature/fulltext` downloads into the
user's **private** library (`/v1/library/files`). Licensed PDFs are never added to
the shared farmer knowledge base. Single sign-on (OpenAthens/Shibboleth) needs
the user's browser, so those routes return links rather than server downloads.

### Kenyan communication channels (`agri_advisor/channels`)
| Channel | Operators | Endpoint |
|---|---|---|
| USSD (Africa's Talking) | Safaricom, Airtel, Telkom | `POST /v1/channels/at/ussd` |
| SMS in/out (Africa's Talking or direct operator bulk-SMS / SDP contract) | Safaricom, Airtel, Telkom | `POST /v1/channels/at/sms`, `POST /v1/channels/send` |
| Voice / IVR (Africa's Talking XML, DTMF crop menu) | all | `POST /v1/channels/at/voice` |
| WhatsApp Business Cloud API | - | `GET/POST /v1/channels/whatsapp` |
| M-Pesa STK push (Safaricom Daraja, optional premium services) | Safaricom | `POST /v1/channels/mpesa/stk`, `/mpesa/callback` |

- The operator comes from the gateway `networkCode` (63902 Safaricom, 63903 Airtel,
  63907 Telkom) or from the CA numbering-plan prefix. Prefix detection is
  approximate because of number portability.
- Kenyan numbers get Kiswahili menus by default (`KENYA_DEFAULT_LANGUAGE`).
- Per-operator routing: `SMS_ROUTE_<OPERATOR>=africastalking|http`.
- `CHANNELS_DRY_RUN=true` (the default) logs instead of sending.
- `GET /v1/channels/status` shows what is configured.
- **Going live requires**:
  - an Africa's Talking account (API key, sender ID, and a USSD code / shortcode, which AT provisions with the telcos);
  - Daraja keys for M-Pesa;
  - a WhatsApp Business number;
  - a **public HTTPS URL** for the callbacks.


## Quickstart

```bash
pip install -r requirements-agri.txt        # or: pip install -e ".[agri]"
cp .env.example .env                         # optional - defaults work offline

# 1. Build the knowledge base (sample docs, or your own)
python -m agri_advisor.rag.ingest --rebuild

# 2. Train models: open FAOSTAT + NASA POWER data (or --synthetic for offline dev)
python -m agri_advisor.predict.open_data      # downloads + harmonises (cached after first run)
python -m agri_advisor.predict.train --open-data

# 3. Ask from the CLI (any tier / language)
python -m agri_advisor "How do I control fall armyworm in maize?"
python -m agri_advisor "Ninawezaje kudhibiti viwavi jeshi kwenye mahindi?" --tier sms
python -m agri_advisor "Comment lutter contre la rouille du blé ?" --tier voice
python -m agri_advisor "Yield outlook for my maize?" --context examples/farm_context_kenya_maize.json

# 4. Run the API
uvicorn agri_advisor.interface.api:app --port 8000
```

```bash
curl -s localhost:8000/v1/advise -H 'Content-Type: application/json' \
  -d '{"query":"How do I control fall armyworm?","context":'"$(cat examples/farm_context_kenya_maize.json)"'}'
curl -s localhost:8000/v1/sms   -d 'text=mahindi viwavi jeshi&from=+254700000000'
curl -s localhost:8000/v1/ussd  -d 'sessionId=1&phoneNumber=+254700000000&text='        # menu
curl -s localhost:8000/v1/ussd  -d 'sessionId=1&phoneNumber=+254700000000&text=1*wheat rust'
curl -s localhost:8000/v1/voice -H 'Content-Type: application/json' -d '{"query":"sorghum striga","language":"fr"}'
curl -s localhost:8000/v1/extension -H 'Content-Type: application/json' -d '{"query":"maize planting","role":"extension"}'
```

```bash
# Researcher accounts, journal entitlements and reviews
python -m agri_advisor.accounts create-admin --email admin@kalro.org
python -m agri_advisor.accounts create-org --name KALRO --slug kalro --domain kalro.org
TOKEN=$(curl -s localhost:8000/v1/auth/register -H 'Content-Type: application/json' \
  -d '{"email":"you@kalro.org","password":"change-me-now","org":"kalro"}' | jq -r .token)
curl -s localhost:8000/v1/credentials -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"provider":"ezproxy","data":{"prefix":"https://ezproxy.example.ac.ke/login?url="}}'
curl -s localhost:8000/v1/literature/search -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"drought tolerant maize Kenya","year_from":2015}'
curl -s localhost:8000/v1/literature/access -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"doi":"10.1016/j.agsy.2025.104434"}'
# Kenyan channels (dry-run by default)
curl -s localhost:8000/v1/channels/at/ussd -d 'sessionId=1&phoneNumber=+254722000000&networkCode=63902&text='
curl -s 'localhost:8000/v1/channels/operator?msisdn=0733000000'
```

Interactive API docs: `http://localhost:8000/docs`. Run the tests with `pytest tests -q`.

### Using it from the AgentCore agent
Set `AGRI_MODE=true` on the agent runtime. The Strands agent
(`patterns/strands-deep-research`) then gets the local tools
`agri_knowledge_search` and `agri_recommendation`, plus an agricultural prompt
addendum: cite knowledge-base sources, state confidence, surface
`consult_expert`, and answer in the user's language. The Dockerfile already
copies `agri_advisor/`, `config/` and `data/knowledge_base/`.

## Configuration
All settings are environment variables, documented in [`.env.example`](.env.example):
- LLM provider, model and key
- embedding provider and model
- vector store path, chunking, top-k and minimum relevance
- model directory and conformal alpha
- expert-validation config and review queue
- default role and language
- SMS, USSD and voice limits

## Data governance
Documents carry an `access_level`: `open`, `embargoed` (until a date), `restricted` or `closed`. Each caller role sees a different set:

| Role | Can see |
|---|---|
| `public` | open documents only |
| `extension` | open documents only |
| `consortium` | open + embargoed + restricted |
| `data_owner` | everything |

Governance is enforced inside retrieval, so restricted text never reaches the LLM for unauthorised callers. Missing or unknown levels default to `closed`.

## Contributing / data ingestion (KALRO and partners)
See **[data/knowledge_base/README.md](data/knowledge_base/README.md)** for the metadata schema, the governance checklist, PDF/DOCX sidecar files and rebuild commands. In short:
1. Put validated documents under `data/knowledge_base/<org>/<crop>/`. Fill in `source_org`, `year`, `url`, `crops`, `countries`, `language`, `access_level` and `validated_by`.
2. Run `python -m agri_advisor.rag.ingest --rebuild`, then test a few queries.
3. For models: export field data to CSV with the columns in `agri_advisor/predict/schema.py`. Then run `python -m agri_advisor.predict.train --data data.csv --status validated`, but only after expert review.
4. Open a pull request. Use conventional commits (`feat(rag): ...`) and make sure `pytest` and `pre-commit run --all-files` pass.

## Future work
- Ingest the real, validated KALRO and partner corpus (D1.3) and use a multilingual dense embedder.
- Train and validate yield, pest and climate models on real ground-truth, CHIRPS/ERA5, soil and Sentinel-2 data (D2.3–D2.5). Add hybrid deep learning models.
- Production LLM configuration and native-speaker validation of all languages. Build Zarma-Songhay resources.
- Live USSD/SMS gateway (e.g. Africa's Talking), IVR with ASR/TTS, an offline-first Android app, and React frontend integration.
- Reviewer UI for the expert queue. Impact-evaluation (RCT) tooling and dashboards. CDK deployment of the API.

---

## Underlying AgentCore Deep Research template
The original template is unchanged and still fully deployable. It is a Strands agent on Amazon Bedrock AgentCore that iteratively queries 10+ data connectors through AgentCore Gateway and writes cited research reports. It includes Memory, Cognito auth, a React streaming frontend, CDK infrastructure and SFT/RL training. Its full documentation (deployment, data sources, frontend, training) is in **[docs/AGENTCORE_TEMPLATE.md](docs/AGENTCORE_TEMPLATE.md)**.
