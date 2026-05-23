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

Enrichment was done via VESSL A100 SXM (Qwen2.5-14B-Instruct on vLLM). VESSL workspace stopped after ingestion.

---

## RAGAS Evaluation

Run `python evaluation/runner.py --output results.json` to see latest scores.
See `results.json` for the most recent run.

CI gate: faithfulness ≥ 0.90 → exit 0 (pass).

---

## How to Start

```bash
# Infrastructure
docker compose up -d

# API server
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Verify
curl http://localhost:8000/health
```

Startup log confirms health:
```
startup_complete  qdrant_vectors=12927  bm25_chunks=12927  bm25_ready=True
```

---

## Key URLs

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin/admin) |
| Langfuse | http://localhost:3000 |
| Qdrant UI | http://localhost:6333/dashboard |
