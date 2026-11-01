# Agricultural Knowledge Base - Ingestion Guide (KALRO & partners)

This folder is the source corpus for the WP1 RAG knowledge base. Every file is
chunked, embedded and indexed with its metadata so each answer can be traced
back to the exact source document (evidence traceability).

> `sample/` contains **illustrative** documents only. They are NOT validated
> agronomic guidance and must be replaced by validated KALRO / partner content.

## 1. Folder layout

```
data/knowledge_base/
  sample/                 # illustrative demo docs (remove for production)
  kalro/                  # recommended: one folder per organisation
    maize/ wheat/ sorghum/ rice/ millet/
  inera/ iar/ embrapa/ ...
```

Supported formats: `.md`, `.txt` (metadata in YAML front-matter) and `.pdf`,
`.docx` (metadata in a sidecar file `<name>.<ext>.meta.yaml`).

## 2. Metadata schema

```yaml
---
doc_id: kalro-maize-faw-ipm-2024        # unique, stable id
title: Integrated management of fall armyworm in maize
source_org: KALRO                       # organisation that owns the content
authors: [A. Researcher, B. Scientist]
year: 2024
url: https://www.kalro.org/...          # or DOI / repository link
doc_type: extension_guide               # extension_guide | research_paper | policy | trial_report
crops: [maize]                          # maize | wheat | sorghum | rice | millet
countries: [KE]                         # KE | UG | NE | NG | GH | BR
language: en                            # en | sw | fr | ha | pt | dje
access_level: open                      # open | embargoed | restricted | closed
embargo_until: null                     # YYYY-MM-DD for embargoed content
validated_by: Dr. X (KALRO Kakamega)    # expert who validated the content
validated_on: 2024-05-01
license: CC-BY-4.0
---
```

For a PDF `faw_guide.pdf`, create `faw_guide.pdf.meta.yaml` with the same keys
(no `---` delimiters).

## 3. Data governance (proposal Table 8 tiers)

| access_level | Who can retrieve it | Typical content |
|---|---|---|
| `open` | everyone (farmers, public tiers) | published extension guides |
| `embargoed` | consortium until `embargo_until`, then everyone | pre-publication results |
| `restricted` | consortium + data owners | trial data, partner datasets |
| `closed` | data owners only (never served by the API) | personal / sensitive data |

Unknown or missing values default to the safest level (`closed`). Do not ingest
farmer personal data into this corpus.

## 4. Build / update the index

```bash
pip install -r requirements-agri.txt
python -m agri_advisor.rag.ingest --rebuild          # whole corpus
python -m agri_advisor.rag.ingest --source data/knowledge_base/kalro   # add a folder
python -m agri_advisor.rag.ingest --stats
```

The index lives in `VECTOR_STORE_PATH` (default `.agri_store/`, git-ignored).

## 5. Embeddings

The default `EMBEDDING_PROVIDER=hashing` is offline and dependency-free (good for
tests, weak semantics). For pilots use a multilingual dense model:

```bash
pip install sentence-transformers
export EMBEDDING_PROVIDER=sentence-transformers
export EMBEDDING_MODEL_ID=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python -m agri_advisor.rag.ingest --rebuild   # always rebuild after switching
```

or `EMBEDDING_PROVIDER=bedrock` (Amazon Titan). For AWS deployments the same
documents can also be synced to a Bedrock Knowledge Base used by the existing
`gateway/tools/kb_search` tool.

## 6. Contributor checklist

1. Content is validated by a named domain expert (`validated_by`).
2. Correct `access_level` agreed with the data owner (data-sharing agreement).
3. Crops / countries / language tags are filled in.
4. Run `--rebuild` and test a few questions:
   `python -m agri_advisor "How do I control fall armyworm?" --role public`
5. Commit the documents (not the `.agri_store/` index) via a pull request.
