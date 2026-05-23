# Production-Grade Agentic RAG System

A fully production-ready Retrieval-Augmented Generation system for querying SEC 10-K filings (Apple, Google, Meta, Microsoft, NVIDIA). Built over 4 phases with real-world engineering patterns: Contextual Retrieval, hybrid dense+sparse search, LangGraph agent, inline citations, SSE streaming, full observability stack, and a CI-gated RAGAS evaluation harness.

## Numbers

| Metric | Value |
|---|---|
| Documents ingested | 10 SEC 10-K filings (FY2024 + FY2025) |
| Total chunks | 12,927 |
| Qdrant vectors | 12,927 (BGE-M3 1024-dim) |
| RAGAS Faithfulness | **0.474** (Qwen2.5-14B judge) |
| RAGAS Context Recall | **0.277** |
| RAGAS Factual Correctness | **0.299** |
| Golden questions | 50 (factual / analytical / multi-hop / adversarial) |
| CI gate | faithfulness ≥ 0.40 (Qwen-calibrated) — **PASSED** |
| Eval runtime | ~14 min (VESSL A100 judge, 50 questions) |

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              FastAPI (uvicorn)               │
                    │  POST /api/v1/ingest   POST /api/v1/query   │
                    │  GET  /api/v1/stream   GET  /metrics        │
                    └────────────┬──────────────────┬─────────────┘
                                 │                  │
              ┌──────────────────▼───┐     ┌────────▼──────────────────────┐
              │  Ingestion Pipeline  │     │     LangGraph Query Agent     │
              │                      │     │                               │
              │  1. Docling parse    │     │  retrieve → grade → rerank    │
              │  2. Semantic chunk   │     │  → reflect → generate         │
              │  3. Context enrich   │     │       (5 nodes)               │
              │     (VESSL/Groq LLM) │     └────────┬──────────────────────┘
              │  4. BGE-M3 embed     │              │
              │  5. Qdrant upsert    │     ┌────────▼──────────────────────┐
              │  6. BM25 rebuild     │     │         Hybrid Retrieval      │
              │  7. Postgres persist │     │                               │
              └──────────────────────┘     │  Dense: BGE-M3 (Qdrant ANN)  │
                                           │  Sparse: BM25 (keyword)      │
              ┌───────────────────────┐    │  Fusion: Reciprocal Rank     │
              │  Observability Stack  │    │  Rerank: BGE-reranker-v2-m3  │
              │                       │    │  Top-K: 8 chunks              │
              │  Langfuse (traces)    │    └────────┬──────────────────────┘
              │  Prometheus (metrics) │             │
              │  Grafana (dashboard)  │    ┌────────▼──────────────────────┐
              │  SSE (streaming)      │    │  Generation + Citations       │
              └───────────────────────┘    │  VESSL Qwen2.5-14B (primary) │
                                           │  Groq llama-3.1-8b (fallback)│
              ┌───────────────────────┐    │  Inline [Source N] citations  │
              │  Evaluation Harness   │    │  Semantic cache (Redis)       │
              │                       │    └───────────────────────────────┘
              │  50 golden questions  │
              │  RAGAS (3 metrics)    │
              │  Checkpoint + resume  │
              │  CI gate (GitHub CI)  │
              └───────────────────────┘
```

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Orchestration | LangGraph |
| Embeddings | `BAAI/bge-m3` (1024-dim, local) |
| Reranker | `BAAI/bge-reranker-v2-m3` (local, top-8) |
| Vector DB | Qdrant (self-hosted Docker) |
| Keyword search | BM25 (rank-bm25, self-healing) |
| LLM (generation) | VESSL Qwen2.5-14B (primary) / Groq (fallback) |
| LLM (enrichment) | VESSL A100 + vLLM + Qwen2.5-14B |
| LLM (eval judge) | VESSL Qwen2.5-14B (no TPD limits) |
| Caching | Redis (semantic similarity ≥ 0.95) |
| Tracing | Langfuse (self-hosted) |
| Metrics | Prometheus + Grafana |
| Streaming | Server-Sent Events |
| Evaluation | RAGAS 0.2 (faithfulness, context recall, factual correctness) |
| CI | GitHub Actions |
| Database | PostgreSQL + SQLAlchemy (async) |
| Document parsing | Docling (PDF) + BeautifulSoup (HTML) |

## What Makes It Production-Grade

**Contextual Retrieval** — Each chunk gets an LLM-generated 80-100 token context blurb prepended before embedding. Anthropic's technique that reduces retrieval failure by 67% by giving chunks document-level context they'd otherwise lack.

**Hybrid retrieval + RRF fusion** — Dense BGE-M3 vectors and sparse BM25 keyword index queried independently. Results merged with Reciprocal Rank Fusion (rank-weighted, not score-weighted) so neither modality dominates.

**Cross-encoder reranker** — Top-50 RRF candidates reranked by BGE-reranker-v2-m3. Top 8 passed to the generator (increased from 5 after eval showed recall gains on multi-point answers).

**BM25 self-healing** — On startup, if the BM25 pickle is missing (cold machine, volume wipe), the server auto-rebuilds it from Postgres. Postgres is truth; the pickle is a cache.

**Qdrant batch upsert** — Single upsert with 1,800+ large-payload points exceeds gRPC's ~4MB message limit and fails silently. Fixed by batching at 200 points/call.

**Semantic cache** — Redis stores recent query embeddings. Cosine similarity ≥ 0.95 → return cached answer instantly. Reduces LLM calls and P95 latency on repeated queries.

**LangGraph reflection loop** — The query agent detects low-confidence retrievals and re-queries with a reformulated question before generating. 5-node graph: retrieve → grade → rerank → reflect → generate.

**Non-disclosure system prompt** — The generator explicitly refuses to fabricate answers for information companies withhold by policy (unit sales, forward guidance, stock prices) rather than hallucinating from adjacent context. This is detectable at eval time: adversarial faithfulness is the canary.

**Full observability** — Every query produces a Langfuse trace with per-node spans. Prometheus tracks `rag_queries_total`, `rag_query_latency_seconds`, `rag_cache_hits_total`. Grafana dashboard pre-built.

**RAGAS CI gate** — Every PR runs the 50-question golden evaluation via GitHub Actions. Checkpoint-aware: each batch of 10 is saved immediately so the runner resumes after any crash or rate-limit kill without re-scoring completed samples. VESSL-first judge routing eliminates TPD exhaustion.

## Evaluation Summary

Three evaluation iterations were run after the full system was built. All used VESSL Qwen2.5-14B as the judge (no Groq TPD limits):

| Run | Faithfulness | Recall | Factual | Notes |
|---|---|---|---|---|
| Baseline | 0.406 | 0.243 | 0.265 | stale cache flushed, old prompt, top-K=5 |
| **Final** | **0.474** | **0.277** | **0.299** | prompt fix + top-K=8 + clean cache |
| **Delta** | **+16.8%** | **+13.9%** | **+13.0%** | |

**What drove the gains:**

- *Adversarial prompt fix (+faith)* — The generator was answering "how many iPhones were sold?" instead of stating that Apple stopped disclosing unit sales in FY2019. An explicit non-disclosure rule in the system prompt corrected this. Adversarial faithfulness (the hardest category) was the primary driver.
- *Reranker top-K 5 → 8 (+recall)* — Multi-point ground truth answers require data from multiple chunks. Expanding the context window passed to the generator improved recall across all non-adversarial categories without increasing latency meaningfully.
- *Cache flush (+all metrics)* — A prior eval run during Groq rate-limit failures had cached "sources do not contain information" responses for basic factual queries. Those stale entries scored 0 on every metric until flushed.

**Calibration context:** Qwen2.5-14B scores ~50–60% of GPT-4 levels for equivalent system quality. The CI gate is set at `faithfulness ≥ 0.40` accordingly. Production targets with a GPT-4-class judge would be faithfulness ≥ 0.75. The next step for higher absolute scores is switching to Qwen2.5-72B-AWQ on the same A100 (fits at ~36GB VRAM) or Prometheus-2-7B (purpose-built evaluation model).

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env    # fill in GROQ_API_KEY (and VESSL_ENDPOINT for GPU judge)

# 2. Start infrastructure (Postgres, Qdrant, Redis, Langfuse, Prometheus, Grafana)
docker compose up -d

# 3. Install Python dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 4. Start the API
uvicorn app.main:app --reload

# 5. Verify startup
curl http://localhost:8000/health
# → {"status":"ok","uptime_seconds":...}
# Log: startup_complete  qdrant_vectors=12927  bm25_chunks=12927  bm25_ready=True
```

## Ingest a Document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@your_document.pdf"
# → {"doc_id": "...", "status": "processing"}

curl http://localhost:8000/api/v1/documents
# → list of documents with status: processing → ready
```

## Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple total revenue in fiscal year 2024?"}'
```

Response includes `answer`, `citations` (with `filename`, `page_num`, `cited_text`), `from_cache`, and `latency_ms`.

## Stream (SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize Apple revenue trends 2024 vs 2023"}'
```

## Run Evaluation

```bash
# Full 50-question RAGAS evaluation (~14 min with VESSL judge)
python evaluation/runner.py --output results.json --save

# Subset by category
python evaluation/runner.py --category factual --limit 10

# Resume after partial failure (checkpoint-aware — no re-scoring)
python evaluation/runner.py --output results.json --save
```

## Key URLs

| Service | URL |
|---|---|
| API docs (Swagger) | http://localhost:8000/docs |
| Prometheus metrics | http://localhost:8000/metrics |
| Langfuse traces | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin/admin) |
| Qdrant UI | http://localhost:6333/dashboard |

## Project Structure

```
production-rag/
├── app/
│   ├── api/routes/
│   │   ├── ingest.py          POST /api/v1/ingest
│   │   ├── query.py           POST /api/v1/query
│   │   ├── stream.py          POST /api/v1/query/stream (SSE)
│   │   ├── documents.py       GET  /api/v1/documents
│   │   └── eval.py            GET  /api/v1/eval/run + /history
│   ├── core/
│   │   ├── config.py          All settings (reads .env)
│   │   ├── metrics.py         Prometheus counters/histograms
│   │   ├── redis_client.py    Semantic cache
│   │   └── tracing.py         Langfuse integration
│   ├── services/
│   │   ├── ingestion/
│   │   │   ├── pipeline.py    Orchestrates 7-step ingestion
│   │   │   ├── parser.py      Docling + BeautifulSoup
│   │   │   ├── chunker.py     RecursiveCharacterTextSplitter
│   │   │   ├── enricher.py    Contextual Retrieval (LLM blurbs)
│   │   │   ├── embedder.py    BGE-M3 async batching
│   │   │   └── bm25_indexer.py BM25 with self-healing
│   │   ├── retrieval/
│   │   │   ├── dense.py       Qdrant ANN search
│   │   │   ├── sparse.py      BM25 search
│   │   │   ├── fusion.py      RRF merge
│   │   │   └── reranker.py    BGE cross-encoder (top-8)
│   │   ├── generation/
│   │   │   ├── groq_gen.py    VESSL/Groq LLM + citation extraction
│   │   │   ├── longcite.py    LongCite citation format
│   │   │   └── router.py      Generation strategy selector
│   │   └── agent/
│   │       ├── graph.py       LangGraph pipeline definition
│   │       └── nodes.py       5 nodes with Langfuse spans
│   ├── repositories/
│   │   ├── document_repo.py
│   │   ├── chunk_repo.py
│   │   └── eval_repo.py
│   └── workers/
│       └── ingest_worker.py   Background ingestion task
├── evaluation/
│   ├── golden_dataset.json    50 Q&A pairs (AAPL FY2024)
│   ├── runner.py              RAGAS runner, VESSL judge, CI gate, checkpoint
│   └── report.py              Terminal table formatter
├── scripts/
│   ├── benchmark.py           Latency + hit rate benchmark
│   ├── rebuild_indexes.py     Manual BM25 rebuild
│   ├── ingest_sample.py       Batch ingest helper
│   └── download_sec_filings.py SEC EDGAR downloader
├── docs/
│   ├── phase1-explained.md    Ingestion pipeline deep-dive
│   ├── phase2-explained.md    Query pipeline deep-dive
│   ├── phase3-explained.md    Observability deep-dive
│   └── phase4-explained.md    Evaluation + production patterns
├── PROJECT_DEEP_DIVE.md       Full technical walkthrough (interview prep)
├── infra/
│   ├── prometheus/            prometheus.yml
│   └── grafana/               dashboards + datasources
├── .github/workflows/
│   └── eval.yml               CI gate (faithfulness ≥ 0.40, Qwen-calibrated)
├── docker-compose.yml         All infrastructure services
└── pyproject.toml             Dependencies + ruff config
```

## Phases

| Phase | What Was Built |
|---|---|
| Phase 1 | Ingestion pipeline: Docling parse → semantic chunk → Contextual Retrieval enrichment (VESSL A100 + Qwen2.5-14B) → BGE-M3 embed → Qdrant upsert → BM25 rebuild → Postgres persist |
| Phase 2 | Query pipeline: LangGraph 5-node agent, hybrid dense+BM25 retrieval, RRF fusion, BGE cross-encoder reranker (top-8), VESSL/Groq generation, inline citations, Redis semantic cache |
| Phase 3 | Observability: Langfuse per-node traces, Prometheus metrics, Grafana dashboard, SSE token streaming |
| Phase 4 | Evaluation: 50-question RAGAS golden dataset, VESSL judge routing (no TPD limits), checkpoint-aware runner, CI gate, adversarial prompt hardening |
