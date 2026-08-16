# RAG Observability Platform

A full-stack observability platform for RAG pipelines: offline + online evaluation
(the RAG Triad via Ragas), root-cause failure localization (retriever vs. generator
vs. permission-layer), multi-tenant ACL-filtered retrieval, and a CI/CD regression
gate that blocks deploys when faithfulness drops.

## Architecture

```
backend/
  app/
    config.py            # env-driven settings, no hardcoded credentials
    rag/
      llm_providers.py    # OpenAI / DeepSeek / Ollama via one OpenAI-compatible client
      embeddings.py        # local sentence-transformers, no API key needed
      vector_store.py      # Chroma + PGVector backends, real ACL metadata filtering
      pipeline.py           # the multi-tenant RAG pipeline under test
    eval/
      ragas_eval.py         # RAG Triad via Ragas, wired to the real LLM provider
      failure_classifier.py # deterministic root-cause classification
    api/                    # FastAPI routes: /ingest /query /debug/trace /regression
  ci/
    corpus.json             # 40 ACL-tagged synthetic company docs
    golden_dataset.json      # 50 query/role/ground_truth regression pairs
    run_regression.py         # the CI gatekeeper script
  tests/                     # pytest suite (ACL, failure classifier, API routes, CI gate)
frontend/                     # Next.js dashboard: query debugger + CI regression history
.github/workflows/rag-regression.yml   # GitHub Actions: runs the gate on every push/PR
docker-compose.yml
```

## Real

- **LLM providers**: OpenAI, DeepSeek, and Ollama all expose an OpenAI-compatible
  `/v1/chat/completions` API, so `app/rag/llm_providers.py` uses one real `openai`
  SDK client pointed at whichever `base_url` your `.env` selects. There is no mock
  or stub LLM anywhere in the request path.
- **Embeddings**: real `sentence-transformers` model, running locally, no API key.
- **ACL enforcement**: implemented as a real metadata `where` filter in Chroma / a
  real `WHERE acl_role = ANY(...)` clause in Postgres — not a Python post-filter —
  so a bug in the underlying filter logic actually surfaces in test results.
- **Evaluation**: real Ragas 0.4.3 metrics (`Faithfulness`, `AnswerRelevancy`,
  `ContextPrecisionWithoutReference`, `ContextRecall`), scored by making real LLM
  calls through the same provider abstraction the pipeline itself uses.

## Prerequisites

- Python 3.11+
- Node.js 20+
- One of: an OpenAI API key, a DeepSeek API key, or a local [Ollama](https://ollama.com) install
- Optional: Docker + Docker Compose (for the Postgres/pgvector option)

## Quick start (Ollama, local, no API key)

```bash
# 1. Install Ollama and pull a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
ollama serve &   # if not already running as a service

# 2. Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # defaults already point at Ollama/mistral + local Chroma
uvicorn app.main:app --reload

# 3. In another terminal: seed the demo corpus
python scripts/seed_corpus.py

# 4. Frontend
cd ../frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

Open http://localhost:3000 — ask a query as "Engineering", "HR", "Executive", or
"General" and watch the trace: retrieved chunks, Ragas scores, and root-cause
classification.

## Using OpenAI or DeepSeek instead

Edit `backend/.env`:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

or

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat
```

No code changes needed — `get_llm_client()` picks the right `base_url`/key at
runtime based on `LLM_PROVIDER`.

## Switching to PGVector instead of Chroma

```bash
# start Postgres with the pgvector extension
docker run -d --name rag-pg -e POSTGRES_USER=raguser -e POSTGRES_PASSWORD=ragpass \
  -e POSTGRES_DB=rag_observability -p 5432:5432 pgvector/pgvector:pg16
```

Then in `backend/.env`:

```bash
VECTOR_STORE=pgvector
PG_DSN=postgresql://raguser:ragpass@localhost:5432/rag_observability
```

`PGVectorBackend` creates its own table + ivfflat index on first run — no manual
migration needed.

## Running the CI regression gate locally

```bash
cd backend
python ci/run_regression.py                 # full 50-sample run
python ci/run_regression.py --limit 5        # quick smoke test
python ci/run_regression.py --gate 0.80      # override the faithfulness gate
```

This ingests the 40-doc corpus into a fresh vector store, runs all 50 golden
queries through the real pipeline, scores each with Ragas, writes a timestamped
JSON report to `backend/ci/reports/`, and exits non-zero (blocking CI) if mean
faithfulness is below the gate (default 0.85). The Next.js `/eval` page reads
these reports via `/regression/latest` and `/regression/history`.

## Running the test suite

```bash
cd backend
LLM_PROVIDER=ollama pytest tests/ -v
```

23 tests cover: real ACL enforcement against a real Chroma collection, the
failure-classifier's root-cause decision logic across all four outcomes
(pass / retriever / generator / permission — including both permission-leak
and permission-block directions), API route wiring and error handling, and the
CI gate's pass/fail arithmetic + fixture integrity. These do not require a
reachable LLM or embedding-model download.

A live end-to-end run (`ci/run_regression.py`, or `/query` and `/debug/trace`
against a real request) additionally requires: your chosen LLM provider to be
reachable, and — on first run only — an internet connection to
`huggingface.co` so `sentence-transformers` can download the embedding model
weights (~90MB, cached after that).

## Docker Compose (full stack)

```bash
docker compose up --build
docker compose exec ollama ollama pull mistral   # first run only
```

Backend on :8000, frontend on :3000, Postgres/pgvector on :5432, Ollama on :11434.

## CI/CD

`.github/workflows/rag-regression.yml` runs on every push/PR touching `backend/**`:
installs dependencies, installs Ollama + pulls `mistral` (so the gate runs
against a real model without needing an API key secret in fork PRs), runs the
pytest suite, then runs `ci/run_regression.py` and fails the job if the
faithfulness gate isn't met. Swap the Ollama step for an `OPENAI_API_KEY` /
`DEEPSEEK_API_KEY` repo secret + `LLM_PROVIDER=openai` env if you'd rather gate
against a hosted model.
