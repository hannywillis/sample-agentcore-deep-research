# Gap Analysis - Repository vs. CNPq PROAFRICA Proposal

Proposal: *Evidence-Grounded and Context-Aware AI for Agricultural Decision
Support in African and Brazilian Smallholder Farming Systems*. Deliverable codes
(D1.x / D2.x / D3.x) are taken from the proposal's deliverables list.

## Baseline (before this change)

The repository was the AWS **AgentCore Deep Research** template: a Strands agent
on Bedrock AgentCore Runtime that writes cited research reports. It included
AgentCore Gateway Lambda tools (Tavily, Nova, arXiv, OpenFDA, S3, Bedrock
Knowledge Base, PubMed, ClinicalTrials, FRED, EDGAR, AlphaVantage), AgentCore
Memory, Cognito auth, a React frontend, CDK infrastructure and SFT/RL training
scripts. It had **no** agricultural content, multilingual handling, predictive
models, uncertainty, expert validation, data-governance tiers or
SMS/USSD/voice access, and no unit tests.

## Requirement mapping

| Proposal requirement | Before | Implemented now | Remaining work |
|---|---|---|---|
| D1.4 searchable knowledge repository | Generic Bedrock KB tool (no metadata or governance) | `agri_advisor/rag`: ingestion of md/txt/pdf/docx with provenance metadata, chunking, embeddings (hashing / sentence-transformers / Titan), local vector store, crop/country-aware retriever | Ingest real validated KALRO and partner documents (D1.3); production vector DB |
| D1.5 RAG knowledge translation with traceability | Report citations only | `agri_advisor/nlp/translator.py` + prompts: evidence-only answers, `[S#]` citations mapped to source docs, uncited points dropped, extractive offline fallback | Configure a production LLM; evaluate faithfulness |
| D1.6 language framework (Hausa, Zarma, Kiswahili, French, Portuguese, English) | None | Language detection, agricultural glossary for query expansion, fixed UI phrases (en/sw/fr/ha/pt; Zarma falls back to French), answer-in-user-language prompting | Native-speaker validation; full Zarma-Songhay resources; speech models |
| D1.7 expert validation | None | `config/expert_validation.yaml` + `agri_advisor/validation`: escalation rules (low confidence, weak evidence, out-of-domain, high-risk keywords and outcomes, unvalidated models), review queue, verdict recording | Reviewer UI; inter-rater analysis; calibrating thresholds against expert verdicts |
| D1.8 data governance (open / embargoed / restricted / closed) | None | `agri_advisor/governance.py`: role-based access enforced at retrieval time; unknown values default to closed; pseudonymised usage log | Consent workflows; data-sharing agreements; audit logging |
| D2.1 multi-domain data schema | None | `predict/schema.py`: Pydantic schema for location, climate, soil, NDVI/EVI and farm data | Connectors (CHIRPS, ERA5, iSDAsoil/SoilGrids, Sentinel-2/MODIS) |
| D2.4 benchmark models (RF/XGBoost) | None | Yield and climate-risk models **retrained on open FAOSTAT + NASA POWER data** (3,954 country-crop-seasons, 20 countries, temporal hold-out 2016-2024): yield MAE 0.200 t/ha (= persistence baseline), climate-risk AUC 0.59 (≈ majority class). Pest model still synthetic | Farm/plot ground truth (D2.3), sub-national climate (CHIRPS/ERA5), pest surveillance labels, validation (D2.5) |
| D2.6 uncertainty and explainability | None | Split-conformal prediction intervals, ensemble disagreement, completeness penalty → confidence level; SHAP explanations with a feature-importance fallback | Calibration studies; spatial cross-validation |
| Planting window and soil management | None | Rule-based baselines (`predict/advisors.py`) | Replace with data-driven models |
| D2.7 context-aware recommendation engine | None | `engine/recommendation.py`: combines the evidence flow and the data flow, then applies validation; output contains advisory text, sources, predicted outcome, confidence level and a consult-expert flag | Personalisation; feedback loop |
| D2.8 / WP3 tiered access | Web frontend for research reports only | FastAPI: `/v1/advise` (web/app), `/v1/extension`, `/v1/sms`, `/v1/ussd` (menu), `/v1/voice` (numbered, SSML), `/v1/predict`, `/v1/expert/verdict`; CLI; Strands tools behind `AGRI_MODE` | Kenyan gateways now implemented (`agri_advisor/channels`): Africa's Talking USSD/SMS/Voice covering Safaricom, Airtel and Telkom, direct operator HTTP adapters, WhatsApp Cloud API, Daraja M-Pesa; operator detection; dry-run by default. Remaining: live accounts, shortcodes and public HTTPS host; ASR; Android app |
| D3.4 usage dataset / impact evaluation | None | Pseudonymised JSONL usage log for each query (tier, language, confidence, escalation) | RCT/quasi-experimental tooling; dashboards |

## Researcher tools (added)

| Need | Implemented | Remaining |
|---|---|---|
| Find open literature across journals | Federated search over 9 open indexes/repositories, de-duplication, ranking; DOAJ journal catalogue + curated African/agri OA list | Add AJOL/AGRIS connectors when stable APIs are available |
| Use personal / institutional journal subscriptions | Accounts, organisations, encrypted credential vault; EZproxy, OpenAthens, Research4Life, link templates, Elsevier/Springer/Wiley TDM APIs; private per-user full-text library | Configure real institutional entitlements; SSO (SAML) login to the platform itself |
| Systematic / scoping reviews | Review projects, screening with reasons, prioritisation, snowballing, PRISMA counts, BibTeX/RIS/CSV export, evidence table + synthesis, KB ingestion | Dual-reviewer conflict resolution, risk-of-bias templates, web UI |
