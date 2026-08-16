# Architecture

This document explains how the RAG Observability Platform is put together, the
reasoning behind the main design decisions, and where the trade-offs are. For
setup/running instructions see [README.md](./README.md).

## System overview

```
                              ┌────────────────────────────────────────┐
                              │              Next.js Frontend           │
                              │  /       Query & Trace Debugger         │
                              │  /eval   CI Regression History          │
                              └───────────────────┬──────────────────────┘
                                                   │ REST (fetch)
                                                   ▼
                              ┌────────────────────────────────────────┐
                              │              FastAPI Backend            │
                              │  POST /ingest        POST /query        │
                              │  POST /debug/trace    GET /regression/* │
                              └───┬────────────┬──────────────┬─────────┘
                                  │            │              │
                    ┌─────────────▼──┐ ┌───────▼────────┐ ┌───▼─────────────┐
                    │  RAG Pipeline   │ │ Ragas Evaluator │ │ Failure         │
                    │  (rag/pipeline) │ │ (eval/ragas_eval)│ │ Classifier      │
                    │                 │ │                 │ │(eval/failure_   │
                    │                 │ │                 │ │ classifier)     │
                    └───┬─────────┬───┘ └────────┬────────┘ └─────────────────┘
                        │         │              │
              ┌──────────▼─┐  ┌────▼──────┐ ┌─────▼────────────┐
              │ Embeddings │  │ Vector    │ │ LLM Provider      │
              │ (local     │  │ Store     │ │ (OpenAI /         │
              │ sentence-  │  │ (Chroma / │ │  DeepSeek /       │
              │ transformer│  │ PGVector) │ │  Ollama — one     │
              │ )          │  │           │ │  OpenAI-compatible│
              └────────────┘  └───────────┘ │  client)          │
                                             └───────────────────┘

                    ┌──────────────────────────────────────┐
                    │     CI/CD Regression Gate             │
                    │  ci/run_regression.py                 │
                    │  reads ci/corpus.json (40 docs) +      │
                    │  ci/golden_dataset.json (50 pairs),    │
                    │  drives the same pipeline+evaluator     │
                    │  above, writes ci/reports/*.json,       │
                    │  exits non-zero if faithfulness < gate  │
                    └──────────────────────────────────────┘
                       triggered by .github/workflows/rag-regression.yml
```

The backend has three logical subsystems sharing the same underlying
pipeline and evaluator code — there is exactly one implementation of
"run a query through RAG" and one implementation of "score a response
with Ragas," used by the live `/query`/`/debug/trace` endpoints, the
manual debugger UI, and the CI gate alike. That's a deliberate choice:
if the CI gate and the live API disagreed on how retrieval or scoring
worked, a passing CI run wouldn't actually tell you anything about
production behavior.

## Component walkthrough

### 1. RAG pipeline under test (`app/rag/`)

| File | Responsibility |
|---|---|
| `embeddings.py` | Local `sentence-transformers` model (`all-MiniLM-L6-v2`, 384-dim), cached in-process. No API key, no network call after the first model download. |
| `llm_providers.py` | One `openai.OpenAI` client, pointed at a different `base_url` depending on `LLM_PROVIDER`. OpenAI, DeepSeek, and Ollama all speak the same `/v1/chat/completions` schema, so there's no per-provider branching in the generation code path. |
| `vector_store.py` | `VectorStoreBackend` protocol with two real implementations: `ChromaBackend` (embedded, local) and `PGVectorBackend` (Postgres + the `vector` extension). Both expose `query()` (ACL-filtered) and `query_unfiltered()` (audit-only). |
| `pipeline.py` | `RAGPipeline.run()` — embeds the query, retrieves with the role's ACL filter applied, *also* retrieves unfiltered (for the permission audit, never used to answer), and calls the LLM. Returns a `QueryTrace` dataclass carrying everything downstream consumers need. |

**Why ACL filtering lives in the store, not in application code:** the
`where`/`WHERE` clause is evaluated by Chroma/Postgres directly, using
the same metadata index the similarity search uses. A Python
post-filter (`retrieve top 20, then drop disallowed ones in a loop`)
would silently degrade recall — if a role is only allowed 2 of the top
20 nearest neighbors, you'd return 2 chunks instead of the requested
`k`, and worse, a bug in the post-filter wouldn't be caught by testing
the store directly. Filtering at the query layer means retrieval
tests exercise the real enforcement mechanism.

**there's a parallel unfiltered query:** to distinguish "the ACL
filter is working as intended, and this role genuinely has no relevant
documents" from "the ACL filter has a bug that's either hiding
documents this role should see, or leaking ones it shouldn't." Both
of those failure modes look identical from the outside (a role gets
an empty or wrong answer) unless you have a second signal to compare
against. `RootCause` in the failure classifier is that comparison.

### 2. Evaluation engine (`app/eval/ragas_eval.py`)

Wraps four Ragas 0.4.3 metrics, each constructed with the *same*
LLM client and embedding model the pipeline itself used to generate the
response being scored:

- **Faithfulness** — is the answer entailed by the retrieved context? (hallucination check)
- **AnswerRelevancy** — does the answer address the query, independent of correctness?
- **ContextPrecisionWithoutReference** — of the retrieved chunks, how many were actually relevant?
- **ContextRecall** — (only computed when a `ground_truth` is supplied) did retrieval find everything relevant?

All four are LLM-judged (Ragas asks the configured LLM to score the
response against the context), which is why evaluation cost scales
with dataset size — 50 golden samples means roughly 50 × 3–4 extra
LLM calls per regression run, on top of the 50 generation calls
themselves.

### 3. Failure localization (`app/eval/failure_classifier.py`)

Pure decision logic, no LLM call of its own — it consumes the Ragas
scores plus the retrieval trace and applies a fixed decision order:

```
permission leak detected?        → PERMISSION_FAILURE  (highest priority)
permission block detected?       → PERMISSION_FAILURE
context_precision < threshold?   → RETRIEVER_FAILURE
faithfulness < threshold?        → GENERATOR_FAILURE
else                              → PASS
```

Permission failures are checked first and short-circuit the rest —
a leaked document makes faithfulness/precision numbers moot (the
response might score "faithful" while faithfully summarizing content
the user should never have seen). This ordering is enforced by
`test_permission_failure_takes_priority_over_generator_failure` in
the test suite.

### 4. API layer (`app/api/`)

Four FastAPI routers, each thin — validation and error-shaping only,
all real logic lives in `rag/` and `eval/`:

- `POST /ingest` — load ACL-tagged documents into the active vector store.
- `POST /query` — the "production" endpoint: ACL-filtered RAG, no evaluation overhead. Returns 400 for an unknown role, 502 if the LLM provider call fails (never a silently-degraded fake answer).
- `POST /debug/trace` — the debugger's endpoint: full pipeline + Ragas scoring + root-cause classification in one call. Short-circuits straight to a permission-failure report if retrieval was empty/leaked, skipping a meaningless Ragas call.
- `GET /regression/latest`, `GET /regression/history` — read-only, serve the JSON reports `ci/run_regression.py` writes to disk.

### 5. CI/CD regression gate (`backend/ci/`)

`run_regression.py` is not a separate reimplementation of scoring
logic — it ingests `corpus.json` into a fresh vector store, then
calls the exact same `RAGPipeline` and `RagasEvaluator` classes the
API uses, once per golden-dataset row. That's what makes "CI failed"
mean something: it's the same code path a real user's query would hit.

The gate itself is one comparison — `avg(faithfulness) >= gate`
(default `0.85`) — deliberately simple so the pass/fail condition is
auditable at a glance in a PR.

## Data model

**Document** (what gets embedded and stored):
```
{ id, text, acl_role: "Engineering"|"HR"|"Executive"|"General", source, metadata }
```

**Role → allowed ACL tiers** (`ROLE_ACCESS` in `pipeline.py`):
```
Engineering → [Engineering, General]
HR          → [HR, General]
Executive   → [Engineering, HR, Executive, General]   (sees everything)
```
This mapping is intentionally hardcoded and centralized in one place
rather than derived per-document, so a permission-model change is a
one-line diff instead of a data migration.

**QueryTrace** (the unit everything downstream consumes):
```
query, role, allowed_roles,
retrieved_chunks[],           # what the user's answer was actually built from
unfiltered_top_chunks[],      # audit-only, ACL bypassed, never shown as "the answer"
answer, context_used[]
```

## Key design decisions and trade-offs

| Decision | Reasoning | Trade-off accepted |
|---|---|---|
| One OpenAI-compatible client for all 3 LLM providers | Zero per-provider branching in generation code; switching providers is a `.env` change | Providers with genuinely different APIs (e.g. Anthropic's native format) would need a real adapter, not just a `base_url` swap |
| ACL filter at the store's metadata-index layer | Filtering and retrieval share one code path — no drift between "what's searched" and "what's allowed" | Every backend (Chroma, PGVector) must implement filtering itself; adding a third backend means reimplementing it again rather than inheriting a generic filter |
| Local embeddings, remote generation | Embedding is high-volume (every query, every doc) and commodity; generation is comparatively low-volume and quality-sensitive — worth paying for a hosted model | Embedding quality is capped by the local model choice (`all-MiniLM-L6-v2`, 384-dim); swapping to a larger local model or a hosted embedding API is a config change but affects the PGVector schema (`vector(384)` is hardcoded — see `_ensure_table`) |
| Failure classifier is pure logic over Ragas scores, not its own LLM call | Deterministic, free, testable without a live LLM (all 6 classifier test cases run with zero network calls) | It's only as good as the two thresholds it's given (`faithfulness_threshold=0.7`, `context_precision_threshold=0.5`) and the Ragas scores feeding it — it can't catch a failure mode Ragas itself doesn't measure |
| CI gate reuses the live pipeline/evaluator classes verbatim | A passing gate is evidence about production behavior, not a parallel fixture that could drift | Regression runs are only as fast/cheap as a real 50-query eval — no shortcut path for a "fast but less meaningful" CI check |
| Synthetic 40-doc / 50-query golden dataset, generated programmatically | Deterministic, versioned, and exercises all 4 roles including the specific "General role must not answer Exec/HR/Eng-only questions" cases | It's a small, hand-authored corpus, not sourced from production traffic — regression coverage is only as representative as this dataset's construction |

## Known limitations

- **Embedding dimension is hardcoded to 384** in `PGVectorBackend._ensure_table()` to match `all-MiniLM-L6-v2`. Swapping `EMBEDDING_MODEL` to a model with a different output dimension requires updating that column definition (and re-embedding existing documents).
- **Role→ACL mapping is a fixed dict**, not stored in the database or configurable per-tenant. Multi-tenant in the sense of "ACL-filtered retrieval," not in the sense of "each customer defines their own role hierarchy."
- **The failure classifier's thresholds are static constants**, not calibrated against a labeled failure dataset. In a real deployment you'd want to tune `faithfulness_threshold`/`context_precision_threshold` against known-good and known-bad examples rather than the defaults used here.
- **No authentication on the API itself** — `role` is passed as a plain request field, not derived from a verified session/token. This is a mock-RBAC demo of ACL-filtered retrieval, not a production auth system.
