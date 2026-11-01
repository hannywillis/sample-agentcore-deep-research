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

## Quickstart

```bash
pip install -r requirements-agri.txt        # or: pip install -e ".[agri]"
cp .env.example .env                         # optional - defaults work offline

# 1. Build the knowledge base (sample docs, or your own)
python -m agri_advisor.rag.ingest --rebuild

# 2. Train scaffold models (SYNTHETIC data - development only; also auto-trained)
python -m agri_advisor.predict.train --synthetic

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

Run the tests with `pytest tests -q`.

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
