# Implementation Plan — Production RAG Knowledge Copilot

> **Last updated:** 2026-05-22 (revised: Claude API removed, Groq-first + open-source citations)  
> **Status:** Ready to build — skeleton complete, no core logic yet  
> Keep this file updated as phases are completed.

---

## What This Project Is

A production-grade Agentic RAG (Retrieval-Augmented Generation) system. Think of it as a private
"ask questions about your documents" engine — like Notion AI or Glean — but built from scratch
with every engineering decision documented and provable with numbers.

The full technical design lives in `ARCHITECTURE.md`. This file is the build checklist.

---

## Cost Philosophy (Revised — Zero Anthropic API)

| Component | Tool | Cost |
|---|---|---|
| Answer generation | Groq free tier (Llama-3.3-70B) | **$0** |
| Citations / source grounding | Prompt engineering + `rag-citation` lib | **$0** |
| Context enrichment + query rewriting | Groq free tier (Llama-3.3-70B) | **$0** |
| Eval judge (RAGAS) | Groq free tier (Llama-3.3-70B) | **$0** |
| Embeddings + reranking | BGE-M3 + BGE-reranker (local CPU) | **$0** |
| All infrastructure | Self-hosted Docker | **$0** |
| **Total — development AND production** | | **$0** |

VESSL AI $47 credit: reserved for batch ingestion > 500 docs (Groq rate limit overflow).  
Never use VESSL for always-on serving — A10 24/7 burns $47 in 2.5 days.

> **Note on Claude API:** The `app/services/generation/claude.py` file will still be written  
> (it's a portfolio piece — showing you know the Citations API exists is valuable).  
> But it is OFF by default. Groq is the primary generator. Claude only activates if  
> `GENERATION_MODEL=claude-sonnet-4-6` is explicitly set in `.env`.

---

## Open-Source Citation Alternatives (Research Summary)

This was the main question before removing Claude. Here is what exists, from best to simplest:

### Option A — Prompt Engineering with `[Source N]` tags (what we'll use as baseline)
Any LLM can produce grounded citations if you format context and instruct carefully:
```
[Source 1]
{chunk_1_text}

[Source 2]
{chunk_2_text}

Answer the question. After every claim, add [N] where N is the source number.
Return JSON: {"answer": "...", "citations": [{"source": 1, "claim": "..."}]}
```
Works with Groq Llama-3.3-70B. Post-process the JSON to build citation objects.  
This is already what `ARCHITECTURE.md` describes as the Groq fallback approach.  
**Source:** Medium — "Anthropic-Style Citations with Any LLM" (2025)

---

### Option B — `rag-citation` Python library (citation verification layer, no LLM needed)
GitHub: `rahulanand1103/rag-citation`  
A drop-in library with two modes:
- **Non-LLM mode** (default): SpaCy NER + SentenceTransformers semantic similarity. No API call. Runs locally.
- **LLM mode**: LiteLLM backend — plug in Groq, OpenAI, or any provider.

What it does: after the LLM generates an answer, it maps every claim back to a source chunk.  
Also has beta hallucination detection — flags entities (dates, money, numbers) not found in source.  
We add this as a **post-generation verification step** on top of the Groq generator.

---

### Option C — LongCite (fine-tuned Llama 3.1 / GLM-4, sentence-level citations natively)
GitHub: `THUDM/LongCite` — MIT license  
Models: `LongCite-llama3.1-8b` and `LongCite-glm4-9b` on HuggingFace  
These models are fine-tuned specifically to produce **sentence-level citations inline** —  
not via prompting, but baked into the model weights. Supports 128K context.

This is the closest open-source equivalent to Claude's Citations API.  
**Trade-off:** Needs GPU. 8B model needs ~16GB VRAM. Runs on VESSL A10 (24GB).  
VESSL cost for a 2-hour demo session: ~$1.52. Fine for occasional demos, not always-on.

---

### Option D — Verbatim RAG (KRLabsOrg/verbatim-rag)
Hallucination-prevention by extracting verbatim text spans from source documents.  
Responses are composed entirely from exact passages — no paraphrasing.  
Uses either LLM-based extractors OR a fine-tuned ModernBERT model (HuggingFace, local CPU).  
Most conservative approach — zero hallucination guarantee but answers read robotically.

---

### Option E — Dokis (provenance middleware, no LLM needed)
GitHub: `Vbj1808/Dokis`  
Sits inline between retrieval and response delivery.  
After generation: splits response into atomic claims, matches each to best chunk via BM25,  
builds a `claim → chunk → URL` provenance map, computes a compliance rate.  
LangChain drop-in. ~42MB. Zero LLM calls needed.  
Good as an **observability / trust layer** rather than primary citation generation.

---

### What We'll Actually Build (Decision)

| Step | Approach | Cost |
|---|---|---|
| Generation | Groq Llama-3.3-70B with `[Source N]` prompt | $0 |
| Citation objects | JSON-structured output from Groq | $0 |
| Citation verification (optional) | `rag-citation` non-LLM mode (SpaCy + SentenceTransformers) | $0 |
| Eval judge | Groq Llama-3.3-70B via RAGAS custom LLM | $0 |
| Premium demo path (optional) | LongCite-8B on VESSL A10 (~$1.52/session) | ~$1.52 if used |

The code architecture keeps Claude as an optional path via `GeneratorRouter`. This means the  
portfolio shows awareness of the Citations API even if it's not the default runtime path.

---

## What's Already Done (Phase 0 — Skeleton)

- [x] `docker-compose.yml` — all infrastructure: Postgres, Qdrant, Redis, Langfuse
- [x] `pyproject.toml` — all Python dependencies pinned
- [x] `.env.example` — every env var documented
- [x] `app/main.py` — FastAPI app + lifespan hooks + CORS
- [x] `app/core/config.py` — Pydantic Settings (reads `.env`)
- [x] `app/core/database.py` — SQLAlchemy async engine + session factory
- [x] `app/api/routes/health.py` — `/health` endpoint working
- [x] `scripts/init_db.sql` — PostgreSQL schema (documents, chunks, eval_runs, golden_questions)
- [x] `.github/workflows/test.yml` + `eval.yml` — CI skeletons

---

## Phase 1 — Ingestion Pipeline

Goal: ingest a PDF → chunks appear in Qdrant and Postgres.

### 1a. Core Infrastructure Singletons

These are shared "utility" modules that every other service imports.

**`app/core/qdrant.py`**
- Creates and holds a single `QdrantClient` instance (re-used across requests — expensive to recreate)
- `ensure_collection_exists()` — creates the "documents" collection with 1024-dim vectors + scalar quantization on startup

**`app/core/redis_client.py`**
- Redis client singleton
- `cache_lookup(query_embedding)` — finds a previously cached answer for a semantically similar query (cosine similarity ≥ 0.95)
- `cache_store(query_embedding, response)` — stores answer with TTL of 1 hour

**`app/core/logging.py`**
- `configure_logging()` + `get_logger()` using structlog — outputs structured JSON logs, not print statements

**`app/core/tracing.py`**
- Langfuse client singleton
- Wraps each pipeline step as a "span" so you can see the full query trace in the Langfuse UI

### 1b. ORM Models

SQLAlchemy models that map Python classes to database tables.

**`app/models/document.py`** — mirrors the `documents` SQL table  
**`app/models/chunk.py`** — mirrors the `chunks` SQL table  
**`app/models/eval_run.py`** — mirrors `eval_runs` and `golden_questions` tables

### 1c. Repositories (Database Access Layer)

These are the only files allowed to talk to Postgres. Everything else calls these functions.

**`app/repositories/document_repo.py`**
- `create_document()`, `get_document()`, `list_documents()`, `delete_document()`, `update_document_status()`

**`app/repositories/chunk_repo.py`**
- `create_chunks()`, `get_chunks_by_doc()`, `delete_chunks_by_doc()`

**`app/repositories/eval_repo.py`**
- `create_eval_run()`, `list_eval_runs()`

### 1d. Ingestion Service Modules

Each file does exactly one job. They are called in sequence by `pipeline.py`.

**`app/services/ingestion/parser.py`**
- Wraps Docling (IBM's PDF parser)
- `parse_document(file_path, file_type) -> str` — returns raw text
- Handles PDF (Docling), .md and .txt (direct read), URL (httpx download + Docling)

**`app/services/ingestion/chunker.py`**
- `SemanticChunker` class using LangChain's `RecursiveCharacterTextSplitter`
- `chunk_document(text, doc_id) -> list[ChunkData]`
- Target: 400 tokens per chunk, 64-token overlap, tables never split mid-row

**`app/services/ingestion/enricher.py`**
- The "Contextual Retrieval" step — Anthropic's published technique that reduces retrieval failures by 67%
- For every chunk: sends the full document + the chunk to Groq, gets back an 80-100 token "context blurb"
- Example blurb: *"This chunk is from Section 3 (Revenue Analysis) which discusses Q3 cloud growth drivers. It relates to the broader earnings report context..."*
- The blurb is prepended to the chunk before embedding — so the embedding captures WHERE in the doc this chunk lives

**`app/services/ingestion/embedder.py`**
- `BGEEmbedder` singleton — loads `BAAI/bge-m3` model once at startup (2.3GB, CPU-only)
- `embed_chunks(chunks) -> list[list[float]]` — batch size 32, returns 1024-dim vectors
- `embed_query(query) -> list[float]` — single query embedding at query time

**`app/services/ingestion/bm25_indexer.py`**
- `BM25Index` class wrapping `rank-bm25`
- BM25 is keyword-based search — finds exact words. Complements semantic search.
- `build(chunks)`, `search(query, top_k) -> list[(chunk_id, score)]`
- Persists to `data/bm25_index.pkl` (pickle) — reloads on server startup

**`app/services/ingestion/pipeline.py`**
- `IngestionPipeline.run(file_path, doc_id, db)` — orchestrates all steps above
- Flow: parse → chunk → enrich → embed → upsert Qdrant → build BM25 → store Postgres

### 1e. Schemas (Pydantic Request/Response Models)

**`app/schemas/ingest.py`** — `IngestRequest`, `IngestResponse`, `JobStatus`  
**`app/schemas/document.py`** — `DocumentRead`, `DocumentList`

### 1f. API Routes

**`app/api/routes/ingest.py`**
- `POST /api/v1/ingest` — accepts file upload OR `{"url": "..."}`, starts background task, returns `job_id`
- `GET /api/v1/ingest/{job_id}` — poll for status (pending/running/done/error)

**`app/api/routes/documents.py`**
- `GET /api/v1/documents` — list all indexed documents
- `DELETE /api/v1/documents/{id}` — delete from Postgres + Qdrant + rebuild BM25

**`app/workers/ingest_worker.py`** — runs the pipeline as a FastAPI `BackgroundTask`

**✅ Phase 1 Checkpoint:** `make ingest` → 5 PDFs indexed → chunks visible in Qdrant dashboard at `http://localhost:6333/dashboard`

---

## Phase 2 — Query Pipeline

Goal: ask a question → get a cited answer with source references.

### 2a. Retrieval Services

**`app/services/retrieval/dense.py`** — `QdrantRetriever.search_dense(query_embedding, top_k=50)`
- Semantic / meaning-based search. Finds chunks that mean the same thing even with different words.

**`app/services/retrieval/sparse.py`** — `BM25Retriever.search_sparse(query, top_k=50)`
- Keyword search. Finds chunks with exact term matches (great for names, IDs, rare terms).

**`app/services/retrieval/fusion.py`** — `ReciprocalRankFusion.fuse(dense_results, sparse_results, k=60)`
- Merges both ranked lists into one. Formula: `score = Σ 1 / (k + rank)` summed across lists.
- Why: a chunk ranked #3 in dense + #5 in sparse is more trustworthy than #1 in only one.

**`app/services/retrieval/reranker.py`** — `BGEReranker.rerank(query, chunks, top_k=5)`
- Cross-encoder: looks at (query, chunk) together, not separately. More accurate than embedding similarity.
- Model: `BAAI/bge-reranker-v2-m3` (local CPU, ~300ms for 50 pairs). Reduces 50 → 5 final chunks.

### 2b. Generation Services

**`app/services/generation/claude.py`**
- `ClaudeGenerator.generate_with_citations(query, chunks, stream=False)`
- Uses Anthropic's Citations API (`citations=True`) — server-verified source pointers, not hallucinated
- Returns `GenerationResult(answer, citations, model, tokens_in, tokens_out, cost_usd)`

**`app/services/generation/groq.py`**
- `GroqGenerator.generate_with_citations(query, chunks)` — manual citation via prompt engineering
- Used as fallback if Claude API is unavailable

**`app/services/generation/router.py`**
- Tries Claude first, falls back to Groq on `anthropic.APIError`
- Can be forced to Groq via `GENERATION_MODEL=llama-3.3-70b-versatile` env var

### 2c. LangGraph Agent

The agent is a state machine that can retry retrieval if the first attempt wasn't good enough.

**`app/services/agent/state.py`** — `RAGState` TypedDict:
```python
{ query, rewritten_queries, retrieved_chunks, retrieval_attempt, is_sufficient, answer, citations, trace_id }
```

**`app/services/agent/nodes.py`** — one function per graph node:
- `query_rewriter_node` — Groq classifies query as FACTUAL/SPARSE/COMPLEX; applies HyDE for SPARSE
- `hybrid_retriever_node` — runs dense + sparse in parallel, applies RRF
- `reranker_node` — BGE cross-encoder → top-5 chunks
- `sufficiency_checker_node` — are 5 chunks enough? If not and `attempt < 2`, retry with rewritten query
- `generate_node` — calls GeneratorRouter, writes Langfuse span, returns answer

**`app/services/agent/graph.py`** — compiles and runs the LangGraph state machine

### 2d. Query Schemas + API

**`app/schemas/query.py`** — `QueryRequest`, `QueryResponse`, `Citation`, `ScoredChunk`

**`app/api/routes/query.py`**
- `POST /api/v1/query` → full pipeline → `QueryResponse` with citations and `trace_url`
- `GET /api/v1/query/stream` → SSE stream of tokens, then citations event, then done event

**✅ Phase 2 Checkpoint:** `curl -X POST /api/v1/query -d '{"query":"..."}' ` → cited JSON answer + `trace_url` linking to Langfuse

---

## Phase 3 — Observability

Goal: every query leaves a full trace you can show in an interview.

**Langfuse tracing** — complete `app/core/tracing.py`:
- Each pipeline step (cache_lookup, query_rewriter, dense_search, sparse_search, rrf_fusion, reranker, generator, cache_write) is a span
- Metadata per span: model, tokens_in, tokens_out, cost_usd, latency_ms

**Prometheus metrics** — `GET /metrics` endpoint:
- `rag_query_total`, `rag_query_latency_ms`, `rag_cache_hit_total`, `rag_cost_usd_total`, `rag_faithfulness_score`, ...

**Add to docker-compose.yml:**
- Prometheus service (scrapes `/metrics` every 15s)
- Grafana service (loads `infra/grafana/dashboards/rag_dashboard.json`)

**✅ Phase 3 Checkpoint:** Ask a question → open `http://localhost:3000` → see full span tree with latencies and token counts

---

## Phase 4 — Evaluation Harness

Goal: `make eval` prints RAGAS scores you can quote in interviews.

**`evaluation/golden_dataset.json`** — Start with 50 Q&A pairs:
- 20 factual (specific numbers, dates, names)
- 15 analytical (trends, comparisons)
- 10 multi-hop (require combining 2+ chunks)
- 5 adversarial (answer NOT in docs — system should say "I don't know")

**`evaluation/runner.py`** (complete):
- Runs each question through the full query pipeline
- RAGAS metrics: `faithfulness`, `context_precision`, `context_recall`, `answer_relevancy`
- LLM judge: `claude-haiku-4-5-20251001` (~$0.001/question)
- Stores result in `eval_runs` Postgres table
- Fails CI if faithfulness < 0.90 or context_precision < 0.80

**`scripts/benchmark.py`** — compares 3 variants:
1. Dense-only (no BM25, no rerank)
2. Hybrid BM25+dense + RRF (no rerank)
3. Full pipeline (hybrid + cross-encoder rerank + contextual retrieval)

Expected progression (from Anthropic's published benchmarks):
- Dense-only: context_precision ~0.61
- +Hybrid RRF: ~0.72
- +Reranker: ~0.80
- +Contextual Retrieval: ~0.88

**`GET /eval/run`** + **`GET /eval/history`** API endpoints

**✅ Phase 4 Checkpoint:** `make eval` → RAGAS table printed to terminal + stored in Postgres

---

## Phase 5 — Polish

**`Makefile`:**
```
make up        → docker compose up -d
make down      → docker compose down
make serve     → uvicorn app.main:app --reload --port 8000
make ingest    → python scripts/ingest_sample.py
make eval      → python evaluation/runner.py
make benchmark → python scripts/benchmark.py
make test      → pytest tests/ -v
```

**Unit tests** (`tests/unit/`):
- `test_chunker.py`, `test_fusion.py`, `test_reranker.py`

**Integration tests** (`tests/integration/`):
- `test_ingestion_pipeline.py`, `test_query_pipeline.py`, `test_api_endpoints.py`

**README.md** — Update with actual measured RAGAS numbers, architecture diagram, benchmark comparison table, one-command setup

**✅ Phase 5 Checkpoint:** `docker compose up` → demo ready. Can demo to any engineer in < 5 minutes.

---

## Files to Create (ordered by dependency)

```
# Core infrastructure
app/models/__init__.py
app/models/document.py
app/models/chunk.py
app/models/eval_run.py
app/core/logging.py
app/core/qdrant.py
app/core/redis_client.py
app/core/tracing.py

# Database access layer
app/repositories/document_repo.py
app/repositories/chunk_repo.py
app/repositories/eval_repo.py

# Pydantic schemas
app/schemas/ingest.py
app/schemas/query.py
app/schemas/document.py

# Ingestion pipeline
app/services/ingestion/parser.py
app/services/ingestion/chunker.py
app/services/ingestion/enricher.py
app/services/ingestion/embedder.py
app/services/ingestion/bm25_indexer.py
app/services/ingestion/pipeline.py
app/workers/ingest_worker.py

# Ingestion API
app/api/routes/ingest.py
app/api/routes/documents.py

# Query pipeline
app/services/retrieval/dense.py
app/services/retrieval/sparse.py
app/services/retrieval/fusion.py
app/services/retrieval/reranker.py
app/services/generation/claude.py
app/services/generation/groq.py
app/services/generation/router.py
app/services/agent/state.py
app/services/agent/nodes.py
app/services/agent/graph.py
app/api/routes/query.py

# Evaluation
evaluation/runner.py
evaluation/metrics.py
evaluation/report.py

# Scripts + tooling
scripts/ingest_sample.py
scripts/benchmark.py
Makefile
```

---

## How to Resume This Plan

If you lose context, just read this file and `ARCHITECTURE.md`. Together they tell you:
- What the system does (ARCHITECTURE.md)
- What's been built and what's next (this file — update checkboxes as you go)
- Exact file list with responsibilities (above)
