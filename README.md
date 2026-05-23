# Production-Grade Agentic RAG System

A fully production-ready Retrieval-Augmented Generation system for querying SEC 10-K filings (Apple, Google, Meta, Microsoft, NVIDIA). Built over 4 phases with real-world engineering patterns: Contextual Retrieval, hybrid dense+sparse search, LangGraph agent, inline citations, SSE streaming, full observability stack, and a CI-gated RAGAS evaluation harness.

## Numbers

| Metric | Value |
|---|---|
| Documents ingested | 10 SEC 10-K filings (FY2024 + FY2025) |
| Total chunks | 12,927 |
| Qdrant vectors | 12,927 (BGE-M3 1024-dim) |
| RAGAS Faithfulness | see `results.json` |
| RAGAS Context Recall | see `results.json` |
| RAGAS Factual Correctness | see `results.json` |
| RAGAS Semantic Similarity | see `results.json` |
| Golden questions | 50 (factual / analytical / multi-hop / adversarial) |
| CI gate | faithfulness ≥ 0.90 |

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
              │                       │    └────────┬──────────────────────┘
              │  Langfuse (traces)    │             │
              │  Prometheus (metrics) │    ┌────────▼──────────────────────┐
              │  Grafana (dashboard)  │    │    Groq LLM + Citations       │
              │  SSE (streaming)      │    │    llama-3.1-8b-instant       │
              └───────────────────────┘    │    Inline source citations    │
                                           │    Semantic cache (Redis)     │
              ┌───────────────────────┐    └───────────────────────────────┘
              │  Evaluation Harness   │
              │                       │
              │  50 golden questions  │
              │  RAGAS (4 metrics)    │
              │  CI gate (GitHub CI)  │
              └───────────────────────┘
```

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Orchestration | LangGraph |
| Embeddings | `BAAI/bge-m3` (1024-dim, local) |
| Reranker | `BAAI/bge-reranker-v2-m3` (local) |
| Vector DB | Qdrant (self-hosted Docker) |
| Keyword search | BM25 (rank-bm25, self-healing) |
| LLM (generation) | Groq `llama-3.1-8b-instant` |
| LLM (enrichment) | VESSL A100 + vLLM + Qwen2.5-14B |
| Caching | Redis (semantic similarity cache) |
| Tracing | Langfuse (self-hosted) |
| Metrics | Prometheus + Grafana |
| Streaming | Server-Sent Events |
| Evaluation | RAGAS 0.2 |
| CI | GitHub Actions |
| Database | PostgreSQL + SQLAlchemy (async) |
| Document parsing | Docling (PDF) + BeautifulSoup (HTML) |

## What Makes It Production-Grade

**Contextual Retrieval** — Each chunk gets an LLM-generated 80-100 token context blurb prepended before embedding. This is Anthropic's technique that reduces retrieval failure by 67% by giving chunks document-level context they'd otherwise lack.

**Hybrid retrieval + RRF fusion** — Dense BGE-M3 vectors and sparse BM25 keyword index are queried independently. Results merged with Reciprocal Rank Fusion (rank-weighted, not score-weighted) so neither modality dominates.

**Cross-encoder reranker** — Top-20 RRF candidates reranked by BGE-reranker-v2-m3. The bi-encoder retrieves fast; the cross-encoder re-scores accurately. Industry standard pattern.

**BM25 self-healing** — On startup, if the BM25 pickle is missing (cold machine, volume wipe), the server auto-rebuilds it from Postgres. Postgres is truth; the pickle is just a cache.

**Qdrant batch upsert** — Single upsert with 1,800+ large-payload points exceeds gRPC's ~4MB message limit and fails silently (empty exception string). Fixed by batching at 200 points/call.

**Semantic cache** — Redis stores recent query embeddings. Cosine similarity ≥ 0.95 → return cached answer instantly. Reduces LLM calls and P95 latency significantly on repeated queries.

**LangGraph reflection loop** — The query agent can detect low-confidence retrievals and re-query with a reformulated question before generating. 5-node graph: retrieve → grade → rerank → reflect → generate.

**Full observability** — Every query produces a Langfuse trace with per-node spans and latency. Prometheus tracks `rag_queries_total`, `rag_query_latency_seconds`, `rag_cache_hits_total` with per-route labels. Grafana dashboard pre-built.

**RAGAS CI gate** — Every PR runs the 50-question golden evaluation via GitHub Actions. faithfulness < 0.90 → exit 1 → PR blocked. Prevents regressions from shipping.

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env    # fill in GROQ_API_KEY

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
```

## Ingest a Document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@your_document.pdf"
# → {"doc_id": "...", "status": "processing"}

# Check status
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
# Full 50-question RAGAS evaluation (takes ~10 min)
python evaluation/runner.py --output results.json --save

# Quick smoke test (5 factual questions)
python evaluation/runner.py --limit 5 --category factual

# Benchmark: latency + hit rate by category
python scripts/benchmark.py --output bench.json
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
│   │   │   ├── bm25_indexer.py BM25 with self-healing
│   │   │   └── pipeline.py
│   │   ├── retrieval/
│   │   │   ├── dense.py       Qdrant ANN search
│   │   │   ├── sparse.py      BM25 search
│   │   │   ├── fusion.py      RRF merge
│   │   │   └── reranker.py    BGE cross-encoder
│   │   ├── generation/
│   │   │   ├── groq_gen.py    Groq LLM + citation extraction
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
│   ├── runner.py              RAGAS runner + CI gate
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
├── infra/
│   ├── prometheus/            prometheus.yml
│   └── grafana/               dashboards + datasources
├── .github/workflows/
│   └── eval.yml               CI gate (faithfulness ≥ 0.90)
├── docker-compose.yml         All infrastructure services
└── pyproject.toml             Dependencies + ruff config
```

## Phases

| Phase | What Was Built |
|---|---|
| Phase 1 | Ingestion pipeline: Docling parse → semantic chunk → Contextual Retrieval enrichment (VESSL A100 + vLLM) → BGE-M3 embed → Qdrant upsert → BM25 rebuild → Postgres persist |
| Phase 2 | Query pipeline: LangGraph 5-node agent, hybrid dense+BM25 retrieval, RRF fusion, BGE cross-encoder reranker, Groq generation, inline citations, Redis semantic cache |
| Phase 3 | Observability: Langfuse per-node traces, Prometheus metrics, Grafana dashboard, SSE token streaming |
| Phase 4 | Evaluation: 50-question RAGAS golden dataset, CI gate (faithfulness ≥ 0.90), BM25 self-healing, Qdrant batch upsert fix, benchmark script |
