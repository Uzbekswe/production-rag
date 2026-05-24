# Agentic RAG for SEC 10-K Filings

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![RAGAS Faithfulness](https://img.shields.io/badge/RAGAS%20faithfulness-0.474-brightgreen)

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/39542c78-2a8f-44b0-b20a-5278afb8a5a5" />


A production-grade Retrieval-Augmented Generation system that answers questions about SEC annual reports using hybrid search, a LangGraph reasoning agent, and a fully automated RAGAS evaluation pipeline.

Built in 4 phases to showcase real-world ML engineering through instrumentation, evaluation, failure analysis, and evidence-driven iteration.


## Table of Contents

- [The Problem](#the-problem)
- [Results](#results)
- [Architecture](#architecture)
- [Key Engineering Decisions](#key-engineering-decisions)
- [What I Would Improve Next](#what-i-would-improve-next)
- [Stack](#stack)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Run Evaluation](#run-evaluation)
- [Service URLs](#service-urls)
- [Project Structure](#project-structure)
- [Phases](#phases)

---

## The Problem

SEC 10-K filings are dense, structured, and full of intentional gaps. Apple's 2024 annual report is 88 pages. Answering *"What was Apple's gross margin improvement between FY2023 and FY2024, and how did it compare to the Services segment margin trend?"* requires:

- Finding the right numbers across multiple sections of the document
- Understanding that "gross margin %" and "gross margin dollars" are different things
- Knowing that Apple stopped disclosing iPhone unit sales in FY2019 (so some questions have correct non-answers)
- Correctly citing the source so the answer is auditable

Naive RAG — embed, chunk, nearest-neighbour retrieval, generate — fails on all of these. This project builds the infrastructure to handle them properly.

**Corpus:** 10 SEC 10-K filings — Apple, Google, Meta, Microsoft, NVIDIA — FY2024 and FY2025.  
**Scale:** 12,927 chunks, 12,927 Qdrant vectors, 3 LLMs across ingestion / generation / evaluation.

---

## Results

All three RAGAS metrics improved between baseline and final run. Every improvement is traced to a specific, diagnosable root cause — not tuning hyperparameters until numbers looked better.

| Metric | Baseline | Final | Change | Root cause fixed |
|---|---|---|---|---|
| **Faithfulness** | 0.406 | **0.474** | +16.8% | adversarial prompt hallucination |
| **Context Recall** | 0.243 | **0.277** | +13.9% | reranker top-K too aggressive |
| **Factual Correctness** | 0.265 | **0.299** | +13.0% | stale "I don't know" answers in Redis cache |
| Samples scored | 45 / 50 | **50 / 50** | +5 | NaN detection fix in runner |

**CI gate: PASSED** — `faithfulness ≥ 0.40` (Qwen-calibrated; GPT-4 equivalent ≈ `faithfulness ≥ 0.75`).

*Judge model: Qwen2.5-14B-Instruct on VESSL A100 (no API rate limits). Three metrics: faithfulness, context recall, factual correctness. 50 golden questions across factual / analytical / multi-hop / adversarial categories.*

### Per-category breakdown (final run)

| Category | N | Faithfulness | Context Recall | Factual Correctness |
|---|---|---|---|---|
| factual | 18 | 0.426 | 0.352 | 0.266 |
| analytical | 13 | 0.499 | 0.164 | 0.285 |
| multi_hop | 12 | 0.429 | 0.278 | 0.257 |
| adversarial | 7 | 0.143 | 0.048 | 0.239 |

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              FastAPI (uvicorn)              │
                    │  POST /api/v1/ingest   POST /api/v1/query   │
                    │  GET  /api/v1/stream   GET  /metrics        │
                    └────────────┬──────────────────┬─────────────┘
                                 │                  │
       ┌─────────────────────────▼────┐    ┌────────▼──────────────────────────┐
       │      Ingestion Pipeline      │    │       LangGraph Query Agent       │
       │                              │    │                                   │
       │  1. Docling PDF parse        │    │  ┌─────────────────────────────┐  │
       │  2. Semantic chunking        │    │  │  retrieve (dense + sparse)  │  │
       │  3. Contextual Retrieval     │    │  │        ↓                    │  │
       │     (LLM context blurb       │    │  │  grade (sufficiency check)  │  │
       │      per chunk, VESSL A100)  │    │  │        ↓                    │  │
       │  4. BGE-M3 embed             │    │  │  rerank (cross-encoder)     │  │
       │  5. Qdrant upsert (batched)  │    │  │        ↓                    │  │
       │  6. BM25 rebuild             │    │  │  reflect (rewrite query?)   │  │
       │  7. Postgres persist         │    │  │        ↓                    │  │
       └──────────────────────────────┘    │  │  generate + cite            │  │
                                           │  └─────────────────────────────┘  │
       ┌──────────────────────────────┐    └────────┬──────────────────────────┘
       │      Observability Stack     │             │
       │                              │    ┌────────▼──────────────────────────┐
       │  Langfuse  (per-node traces) │    │      Hybrid Retrieval Engine      │
       │  Prometheus (metrics)        │    │                                   │
       │  Grafana   (dashboard)       │    │  Dense:  BGE-M3 → Qdrant ANN      │
       │  SSE       (token streaming) │    │  Sparse: BM25 keyword index       │
       └──────────────────────────────┘    │  Fusion: Reciprocal Rank (k=60)   │
                                           │  Rerank: BGE-reranker-v2-m3       │
       ┌──────────────────────────────┐    │  Top-K:  8 chunks to generator    │
       │      Evaluation Harness      │    └────────┬──────────────────────────┘
       │                              │             │
       │  50 golden Q&A pairs         │    ┌────────▼──────────────────────────┐
       │  RAGAS (3 LLM-based metrics) │    │    Generation + Citations         │
       │  VESSL judge (no TPD limits) │    │                                   │
       │  Checkpoint-aware runner     │    │  VESSL Qwen2.5-14B (primary)      │
       │  CI gate (GitHub Actions)    │    │  Groq llama-3.1-8b (fallback)     │
       └──────────────────────────────┘    │  Redis semantic cache (≥0.95 cos) │
                                           └───────────────────────────────────┘
```

**Data flow (query path):**
1. Query arrives at FastAPI → semantic cache check (Redis, cosine sim ≥ 0.95 → return instantly)
2. LangGraph agent starts: dense retrieval (Qdrant ANN) + sparse retrieval (BM25) run in parallel
3. RRF fusion merges ranked lists from both modalities without score normalization
4. BGE cross-encoder reranks top-50 candidates → top-8 passed to the generator
5. Sufficiency check: if top chunks don't cover the query, agent rewrites the query and retries (max 2)
6. Qwen2.5-14B (using VESSL AI Cloud) or Groq API generates a JSON response with inline `[Source N]` citations
7. Citation metadata (filename, page, verbatim excerpt) mapped back to chunk records and returned
8. Langfuse captures the full trace; Prometheus increments latency histogram

---

## Key Engineering Decisions

### 1. Contextual Retrieval over naive chunking

The standard approach — split document into fixed-size chunks, embed, retrieve — fails because chunks lose their document context. A chunk saying *"Revenue increased 12% year-over-year"* tells you nothing about which company, which segment, or which year without surrounding context.

**Decision:** Before embedding each chunk, an LLM (Qwen2.5-14B on VESSL) generates an 80-100 token summary of where that chunk sits in the document: *"This chunk describes Apple's iPhone revenue for FY2024, from the Product Net Sales table in the MD&A section."* That summary is prepended to the chunk text before embedding. The document-aware context is now baked into every vector.

This is Anthropic's Contextual Retrieval pattern. Their published results show a 67% reduction in retrieval failures. The tradeoff is ingestion cost: one LLM call per chunk, which at 12,927 chunks requires a GPU endpoint rather than a rate-limited API.

---

### 2. Reciprocal Rank Fusion over score-normalized fusion

Combining dense (cosine similarity) and sparse (BM25) scores requires a merge strategy. Score normalization is the obvious choice — normalize both to [0,1], take a weighted average. The problem: the score distributions of BGE-M3 and BM25 are incomparable. A cosine similarity of 0.82 and a BM25 score of 14.3 tell you nothing about their relative quality.

**Decision:** RRF merges by *rank* rather than score. `score(d) = Σ 1/(k + rank_i(d))` for each retrieval system i, where k=60 dampens sensitivity to top-rank differences. This works because ranks are always comparable across retrieval systems, regardless of how scores are distributed.

RRF was introduced by Cormack et al. (2009) as a parameter-free fusion method that consistently outperforms score-based fusion without requiring calibration.

---

### 3. Cross-encoder reranker as a deliberate two-pass system

Bi-encoders (BGE-M3) embed query and document *independently*. This is what makes ANN retrieval fast — you precompute document embeddings offline. But it's an approximation: the model never sees the query and document together.

Cross-encoders read query and document jointly, which is dramatically more accurate but requires one forward pass per candidate — too slow for retrieval over 12,000 vectors.

**Decision:** Use both. BGE-M3 retrieves the top-50 candidates fast. BGE-reranker-v2-m3 (568M parameter cross-encoder) rescores those 50 with full query-document attention. The top-8 survivors go to the generator. This is the industry-standard pattern: cheap retrieval, expensive reranking on a small candidate set.

Reranker runs in a thread pool to avoid blocking the FastAPI async event loop.

---

### 4. LangGraph for conditional retry instead of a linear chain

A linear chain (retrieve → rerank → generate) has no mechanism for handling low-confidence retrievals. If the top chunks don't actually answer the question, the generator hallucinates or refuses — and the system just returns that.

**Decision:** The query pipeline is a 5-node LangGraph graph with a conditional retry edge. After retrieval and reranking, a sufficiency checker grades the chunks against the query. If confidence is low, the agent rewrites the query (via an LLM call) and re-retrieves. Maximum 2 retries. The retry branch is a first-class node in the graph, not a special case in application code.

This makes the retry logic explicit, testable, and traceable (each node gets its own Langfuse span with latency).

---

### 5. VESSL A100 for LLM inference instead of API-only

Groq's free tier limits: `llama-3.3-70b-versatile` has 100K tokens/day (rolling 24-hour window), `llama-3.1-8b-instant` has 500K. A 50-question RAGAS evaluation uses ~270K tokens for both generation and judging. Three evaluation runs exhausted both models' daily quotas before a single complete run finished.

**Decision:** Deploy Qwen2.5-14B-Instruct on a VESSL A100 SXM using vLLM's OpenAI-compatible endpoint. Both the generation service and the RAGAS runner use VESSL-first routing: if `VESSL_ENDPOINT` + `VESSL_TOKEN` are set, use VESSL; otherwise fall back to Groq. No TPD limits. The A100 costs ~$1.55/hr — a full 50-question eval run costs roughly $0.40 in GPU time.

The same pattern is used in three places: `enricher.py`, `groq_gen.py`, and `evaluation/runner.py`. All follow the same priority: VESSL → Groq.

---

### 6. Checkpoint-aware evaluation runner

A RAGAS run over 50 questions with a 14B judge takes ~15 minutes and ~270K tokens. RAGAS calls the judge LLM 3 times per sample (once per metric), so 50 samples = 150 LLM calls. If the runner crashes at sample 40 — rate limit, network error, killed process — you lose everything.

**Decision:** After each batch of 10 samples, append scores to a JSONL checkpoint file. On restart, load the checkpoint and skip already-scored indices. Maximum work lost: 10 samples. The checkpoint is deleted automatically on a successful full run.

This also caught a subtle bug during development: RAGAS returns `float('nan')` for failed metric calls, not `None`. An `is not None` check silently treats NaN as a valid score, causing the runner to report "10 scored, 0 failed" even when 70% of samples had no real score. Fixed by normalizing `float('nan') → None` at the point of extraction.

---

### 7. Adversarial system prompt hardening

Financial 10-K filings contain deliberate omissions. Apple stopped reporting iPhone, iPad, and Mac unit sales after FY2018. No 10-K includes forward-looking revenue guidance or daily stock prices. Without explicit instruction, the LLM finds adjacent numerical context and constructs a plausible-sounding but wrong answer.

**Decision:** Add an explicit non-disclosure rule to the system prompt listing the specific categories of information that companies withhold by policy. For these queries, the model responds: *"This information is not disclosed in the company's 10-K annual filing."* This is domain knowledge that only surfaces through adversarial evaluation — the adversarial faithfulness score of 0.143 in the baseline was the signal.

Faithfulness improved 16.8% overall. Adversarial handling was the largest single contributor.

---

## What I Would Improve Next

These are genuine limitations, not theoretical concerns. Each one has a specific eval signal that points to it.

### 1. Query decomposition for analytical questions

Context recall for analytical questions is 0.164 — the lowest of any category. The reason: analytical questions like *"What was Apple's capital return activity and how did it compare to R&D investment?"* need 4–5 distinct data points from different sections. A single retrieval step can't collect all of them. The fix is query decomposition: break the question into atomic sub-queries, retrieve independently, compose the answer. Requires a planning step before retrieval.

### 2. Expand the golden dataset beyond Apple

All 50 current golden questions are AAPL-focused. A production evaluation set would include cross-company comparisons (*"Which company had the highest R&D-to-revenue ratio among FAANG in FY2024?"*), temporal questions (*"How did NVIDIA's data center revenue growth compare in FY2024 vs FY2025?"*), and true multi-hop questions that require linking filings from different companies. 50 questions is also statistically thin — the margin of error at this sample size means a 3-question swing can look meaningful when it's noise.

### 3. Stronger judge model for calibrated evaluation

Qwen2.5-14B is directionally valid (run-to-run comparisons are meaningful) but scores ~50–60% of GPT-4 for equivalent system quality. The CI gate is calibrated accordingly (`faithfulness ≥ 0.40`), but this makes it hard to compare against published benchmarks. The right upgrade path, in order of effort: Qwen2.5-72B-AWQ on the same A100 (fits at ~36GB VRAM), then Prometheus-2-7B (7B model fine-tuned specifically to match GPT-4 judgment quality).

### 4. Re-ranking diversity to reduce chunk redundancy

The top-8 chunks passed to the generator can be highly redundant — the same table cited five different ways, or the same paragraph appearing in the context window and in a footnote. Maximal Marginal Relevance (MMR) or similar diversity-aware reranking would ensure the 8 chunks cover different aspects of the answer, improving recall for multi-point questions without increasing the context window.

### 5. Domain-adapted embeddings

BGE-M3 is a strong general-purpose multilingual embedding model, but it was not trained on SEC filings. Financial terminology has specific connotations: *"provision"* means something very different in accounting vs everyday usage. Fine-tuning on (query, relevant-chunk) pairs mined from this corpus — using synthetic hard negatives from the same filings — would improve retrieval precision for domain-specific vocabulary.

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI + uvicorn | async-native, OpenAPI docs built in |
| Orchestration | LangGraph | conditional retry edges, per-node tracing |
| Embeddings | `BAAI/bge-m3` (1024-dim, local) | top multilingual bi-encoder, no API dependency |
| Reranker | `BAAI/bge-reranker-v2-m3` (local) | best open-source cross-encoder at this size |
| Vector DB | Qdrant (Docker) | supports payload filtering, fast ANN |
| Keyword search | BM25 (rank-bm25) | complementary to dense; exact-match recall |
| LLM (generation) | VESSL Qwen2.5-14B / Groq (fallback) | no TPD limits on VESSL; Groq as cheap fallback |
| LLM (eval judge) | VESSL Qwen2.5-14B | same endpoint; RAGAS-compatible |
| Caching | Redis (semantic similarity) | cosine ≥ 0.95 → instant cache hit |
| Tracing | Langfuse (self-hosted) | per-node spans, latency breakdown |
| Metrics | Prometheus + Grafana | query counts, latency histograms, cache hit rate |
| Streaming | Server-Sent Events | token-by-token streaming without WebSocket complexity |
| Evaluation | RAGAS 0.2 | faithfulness, context recall, factual correctness |
| CI | GitHub Actions | gate on faithfulness regression |
| Database | PostgreSQL + SQLAlchemy (async) | source of truth for chunks; BM25 rebuilds from it |
| Parsing | Docling (PDF) + BeautifulSoup (HTML) | structure-aware PDF; handles SEC EDGAR HTML filings |

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/Uzbekswe/production-rag.git && cd production-rag
cp .env.example .env
# Fill in GROQ_API_KEY (required). Add VESSL_ENDPOINT + VESSL_TOKEN to avoid Groq TPD limits.

# 2. Start infrastructure
docker compose up -d
# Starts: Postgres, Qdrant, Redis, Langfuse, Prometheus, Grafana

# 3. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 4. Download SEC filings (optional — corpus already ingested into Docker volumes)
python scripts/download_sec_filings.py

# 5. Start the API
uvicorn app.main:app --reload
# Startup log: startup_complete  qdrant_vectors=12927  bm25_chunks=12927  bm25_ready=True
```

> **No data ingestion needed.** The 12,927 chunks are stored in Docker volumes (Qdrant + Postgres). `docker compose up -d` restores the full corpus. Re-ingest only if you add new documents.

---

## API Reference

All endpoints are documented interactively at `http://localhost:8000/docs`.

### `POST /api/v1/query`

Answer a question against the 10-K corpus.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple gross margin percentage in fiscal year 2024?"}'
```

```json
{
  "answer": "Apple's gross margin was 46.2% in FY2024, up from 44.1% in FY2023 [Source 1]. The improvement was driven primarily by Services segment margin expansion [Source 2].",
  "citations": [
    {"source_id": 1, "filename": "AAPL_10K_2024.pdf", "page_num": 31, "cited_text": "..."},
    {"source_id": 2, "filename": "AAPL_10K_2024.pdf", "page_num": 33, "cited_text": "..."}
  ],
  "from_cache": false,
  "latency_ms": 1843
}
```

### `GET /api/v1/stream`

Token-by-token SSE streaming for the same query schema.

```bash
curl -N "http://localhost:8000/api/v1/stream?query=What+was+Apple+revenue+2024"
```

### `POST /api/v1/ingest`

Ingest a new PDF or HTML filing into the corpus. Background task — returns immediately with a job ID.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@TSLA_10K_2024.pdf"
```

### `GET /api/v1/documents`

List all ingested documents with chunk count and ingestion timestamp.

### `GET /api/v1/eval/history`

Retrieve stored evaluation run history from Postgres (all RAGAS metric sets, CI gate result per run).

### `GET /metrics`

Prometheus metrics endpoint: query counts, latency histograms, cache hit rate.

### `GET /health`

```json
{
  "status": "startup_complete",
  "qdrant_vectors": 12927,
  "bm25_chunks": 12927,
  "bm25_ready": true
}
```

---

## Run Evaluation

```bash
# Full 50-question RAGAS eval (~14 min with VESSL judge, ~$0.40 GPU cost)
python evaluation/runner.py --output results.json --save

# Resume after any failure — checkpoint-aware, skips already-scored samples
python evaluation/runner.py --output results.json --save

# Category deep-dive
python evaluation/runner.py --category adversarial
python evaluation/runner.py --category analytical --limit 5

# Print results table from an existing run
python evaluation/report.py results.json
```

**Judge routing:** set `VESSL_ENDPOINT` + `VESSL_TOKEN` in `.env` to use Qwen2.5-14B on VESSL (recommended). Falls back to Groq if unset — subject to 100K tokens/day limit, which a full run can exhaust.

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| API (Swagger) | http://localhost:8000/docs | — |
| Prometheus metrics | http://localhost:8000/metrics | — |
| Langfuse traces | http://localhost:3000 | from `.env` |
| Grafana dashboard | http://localhost:3001 | admin / admin |
| Qdrant UI | http://localhost:6333/dashboard | — |

---

## Project Structure

```
production-rag/
├── app/
│   ├── api/routes/          # ingest, query, stream (SSE), documents, eval
│   ├── core/                # config, metrics (Prometheus), Redis cache, Langfuse
│   ├── services/
│   │   ├── ingestion/       # pipeline, parser, chunker, enricher, embedder, BM25
│   │   ├── retrieval/       # dense (Qdrant), sparse (BM25), RRF fusion, reranker
│   │   ├── generation/      # VESSL/Groq generator, citation extraction, router
│   │   └── agent/           # LangGraph graph + 5 nodes with Langfuse spans
│   ├── models/              # SQLAlchemy ORM models
│   ├── repositories/        # Postgres: documents, chunks, eval_runs
│   └── workers/             # background ingestion task
├── evaluation/
│   ├── golden_dataset.json  # 50 ground-truth Q&A pairs
│   ├── runner.py            # RAGAS runner: VESSL judge, checkpoints, CI gate
│   └── report.py            # terminal report formatter
├── scripts/
│   ├── benchmark.py         # latency + hit rate benchmark
│   ├── download_sec_filings.py  # SEC EDGAR downloader
│   ├── ingest_sample.py     # batch ingest helper
│   └── rebuild_indexes.py   # manual BM25 + Qdrant rebuild
├── docs/                    # deep-dive per phase (ingestion, query, observability, eval)
├── PROJECT_DEEP_DIVE.md     # full technical walkthrough + interview prep
├── STATUS.md                # final eval scores, ingestion summary
├── infra/                   # Prometheus config, Grafana dashboards
├── .github/workflows/       # eval CI gate (eval.yml), test runner (test.yml)
├── .env.example             # environment variable template
├── docker-compose.yml       # all infrastructure services
└── pyproject.toml           # dependencies + ruff config
```

---

## Phases

| Phase | What Was Built |
|---|---|
| **Phase 1** | Ingestion: Docling parse → semantic chunk → Contextual Retrieval enrichment (VESSL A100, one LLM call per chunk) → BGE-M3 embed → Qdrant upsert (batched at 200 to stay under gRPC payload limit) → BM25 rebuild → Postgres |
| **Phase 2** | Query: LangGraph 5-node agent, hybrid dense+BM25, RRF fusion, BGE cross-encoder rerank (top-8), Qwen/Groq generation, inline citations, Redis semantic cache |
| **Phase 3** | Observability: Langfuse per-node traces, Prometheus query counters and latency histograms, Grafana dashboard, SSE token streaming |
| **Phase 4** | Evaluation: 50-question golden dataset, VESSL-routed RAGAS runner (no TPD limits), checkpoint-aware batching, NaN score detection, adversarial prompt hardening, CI gate on faithfulness regression |
