# Production-Grade Agentic RAG Knowledge Copilot

Upload PDFs and query them with cited, verifiable answers. Built with Anthropic's Contextual Retrieval + Citations API, hybrid BM25+BGE-M3 retrieval, and a LangGraph reflection loop.

## Stack

| Layer | Tool |
|---|---|
| Answer generation | `claude-sonnet-4-6` + Citations API |
| Contextual chunking | Groq `llama-3.3-70b-versatile` (free) |
| Embeddings | `BAAI/bge-m3` local CPU |
| Reranker | `BAAI/bge-reranker-v2-m3` local CPU |
| Vector DB | Qdrant (self-hosted) |
| Orchestration | LangGraph |
| Observability | Langfuse (self-hosted) |
| Evaluation | RAGAS + DeepEval + GitHub Actions CI |

## Quick Start

```bash
# 1. Copy and fill env
cp .env.example .env

# 2. Start infrastructure
docker compose up -d

# 3. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 4. Run the API
uvicorn app.main:app --reload

# 5. Check health
curl http://localhost:8000/health
```

## Development

```bash
# Lint
ruff check app tests

# Test
pytest tests/unit -v

# Eval gate
python evaluation/runner.py
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.
