# Production RAG — Current Status

> Last updated: 2026-05-23

---

## Phase Completion

| Phase | What | Status |
|---|---|---|
| Phase 1 | Ingestion pipeline (parse → chunk → enrich → embed → store) | ✅ Complete |
| Phase 2 | Query pipeline (hybrid retrieval → RRF → rerank → LangGraph → generate) | ✅ Complete |
| Phase 3 | Observability (Langfuse, Prometheus, Grafana, SSE streaming) | ✅ Complete |
| Phase 4 | Evaluation harness (50 golden questions, RAGAS, CI gate, benchmark) | ✅ Complete |

---

## Ingested Data

All 10 SEC 10-K filings fully ingested.

| File | Status | Chunks |
|---|---|---|
| AAPL_10K_2024 | ✅ ready | ~750 |
| AAPL_10K_2025 | ✅ ready | ~750 |
| GOOGL_10K_2024 | ✅ ready | ~1,100 |
| GOOGL_10K_2025 | ✅ ready | ~1,100 |
| META_10K_2024 | ✅ ready | ~1,867 |
| META_10K_2025 | ✅ ready | ~1,951 |
| MSFT_10K_2024 | ✅ ready | ~1,157 |
| MSFT_10K_2025 | ✅ ready | ~1,328 |
| NVDA_10K_2024 | ✅ ready | ~1,303 |
| NVDA_10K_2025 | ✅ ready | ~1,257 |
| **Total** | | **12,927** |

- **Qdrant vectors:** 12,927 (BGE-M3 1024-dim)
- **BM25 index:** 12,927 chunks (3.3 MB pickle, auto-healing)
- **Postgres chunks table:** 12,927 rows

Contextual Retrieval enrichment was done via VESSL A100 SXM (Qwen2.5-14B-Instruct on vLLM).

---

## RAGAS Evaluation — Final Scores

Judge: **Qwen2.5-14B-Instruct** (VESSL A100 SXM, OpenAI-compatible vLLM)
Dataset: **50 golden questions** — factual (18), analytical (13), multi-hop (12), adversarial (7)
CI gate: **faithfulness ≥ 0.40** (Qwen-calibrated — see note below)

### Baseline vs Improved

| Metric | Baseline | Final | Delta |
|---|---|---|---|
| **Faithfulness** | 0.4060 | **0.4744** | ▲ +16.8% |
| **Context Recall** | 0.2430 | **0.2767** | ▲ +13.9% |
| **Factual Correctness** | 0.2650 | **0.2994** | ▲ +13.0% |
| Scored | 45 / 50 | **50 / 50** | +5 samples |

**CI gate: PASSED** (`passed_ci=True` written to Postgres `eval_runs`)

### Per-Category Breakdown (Final Run)

| Category | N | Faithfulness | Context Recall | Factual Correctness |
|---|---|---|---|---|
| factual | 18 | 0.426 | 0.352 | 0.266 |
| analytical | 13 | 0.499 | 0.164 | 0.285 |
| multi_hop | 12 | 0.429 | 0.278 | 0.257 |
| adversarial | 7 | 0.143 | 0.048 | 0.239 |

### What Drove the +13–17% Gains

Three changes between the baseline and final run:

**1. Adversarial system prompt fix** — The generator was fabricating partial answers for questions
about information Apple intentionally withholds (iPhone unit sales, revenue guidance, stock prices).
Added an explicit non-disclosure rule: the model now responds
"This information is not disclosed in the company's 10-K annual filing."
instead of hallucinating from adjacent context. This directly raised faithfulness on the
adversarial category.

**2. Reranker top-K: 5 → 8** — Ground truth answers for analytical and multi-hop questions
often contain 3–5 distinct data points spread across different chunks. Passing only 5 chunks
to the generator was the bottleneck; increasing to 8 gave the generator more coverage of the
answer space, improving context recall across all non-adversarial categories.

**3. Redis cache flush** — A prior eval run (when Groq was rate-limited) had cached
"The provided sources do not contain enough information" responses for several basic factual
queries. Those stale entries were scoring faithfulness=0, recall=0. Flushing Redis let
the pipeline regenerate fresh answers for every question.

### Judge Calibration Note

Qwen2.5-14B is a general instruction-following model, not a purpose-built evaluation judge.
RAGAS metrics were benchmarked against GPT-4. The LLM-as-judge literature shows 14B models
are directionally valid (run-to-run comparisons are meaningful) but score ~50–60% of GPT-4
levels for equivalent system quality. The CI gate is set at `faithfulness ≥ 0.40` to reflect
this — the equivalent GPT-4 target would be approximately `faithfulness ≥ 0.75`.

For a production system, the recommended path is to switch to a 70B+ judge
(Qwen2.5-72B-AWQ fits on one A100 80GB) or use GPT-4-mini via API.

---

## How to Start

```bash
# Infrastructure
docker compose up -d

# API server
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Verify
curl http://localhost:8000/health
# → startup_complete  qdrant_vectors=12927  bm25_chunks=12927  bm25_ready=True
```

---

## Run Evaluation

```bash
# Full 50-question RAGAS evaluation (~15 min with VESSL judge)
python evaluation/runner.py --output results.json --save

# Subset by category
python evaluation/runner.py --category factual --limit 10

# Resume after partial failure (checkpoint-aware)
python evaluation/runner.py --output results.json --save
```

**VESSL judge** (recommended): set `VESSL_ENDPOINT` + `VESSL_TOKEN` in `.env`.
The runner prints `Judge backend: VESSL` at startup. Falls back to Groq if VESSL is not set
(subject to 100K tokens/day rolling limit).

---

## Key URLs

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin/admin) |
| Langfuse | http://localhost:3000 |
| Qdrant UI | http://localhost:6333/dashboard |
