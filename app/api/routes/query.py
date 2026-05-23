import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.core.metrics import rag_chunks_retrieved, rag_queries_total, rag_query_latency_seconds
from app.core.redis_client import cache_lookup, cache_store, get_redis_client
from app.core.tracing import create_trace
from app.schemas.query import Citation, QueryRequest, QueryResponse
from app.services.agent.graph import rag_graph
from app.services.agent.state import RAGState
from app.services.ingestion.embedder import BGEEmbedder

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)


@router.post("", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """
    Full RAG query pipeline:
      1. Semantic cache check  — return instantly if we've seen a similar query
      2. LangGraph agent        — rewrite → hybrid retrieve → rerank → sufficiency → generate
      3. Cache write            — store result for future similar queries
    """
    query_id = str(uuid4())
    start = time.monotonic()
    logger.info("query_received", query_id=query_id, query=payload.query[:100])

    # ── Step 1: Langfuse trace (wraps the entire request) ────────────────────
    trace = create_trace(
        name="rag_query",
        trace_id=query_id,   # makes trace.id == query_id == state["trace_id"]
        session_id=query_id,
        input={"query": payload.query, "top_k": payload.top_k},
    )

    # ── Step 2: Semantic cache check ─────────────────────────────────────────
    redis = get_redis_client()
    embedder = BGEEmbedder.get()

    import asyncio
    loop = asyncio.get_event_loop()
    query_embedding = await loop.run_in_executor(None, embedder.embed_query, payload.query)

    cached = await cache_lookup(redis, query_embedding)
    if cached:
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info("cache_hit", query_id=query_id, latency_ms=latency_ms)
        try:
            # Re-hydrate Citation objects from stored dicts
            cached["citations"] = [Citation(**c) for c in cached.get("citations", [])]
            rag_queries_total.labels(from_cache="true").inc()
            rag_query_latency_seconds.observe(latency_ms / 1000)
            return QueryResponse(
                **{k: v for k, v in cached.items() if k not in ("query_id", "from_cache", "latency_ms")},
                query_id=query_id,
                from_cache=True,
                latency_ms=latency_ms,
            )
        except Exception:
            pass  # malformed cache entry — fall through to pipeline

    # ── Step 3: Run LangGraph agent ──────────────────────────────────────────
    initial_state: RAGState = {
        "query": payload.query,
        "rewritten_queries": [payload.query],
        "retrieved_chunks": [],
        "retrieval_attempt": 0,
        "is_sufficient": False,
        "answer": "",
        "citations": [],
        "trace_id": query_id,
        "from_cache": False,
        "latency_ms": 0,
        "model_used": "",
    }

    try:
        result_state: RAGState = await rag_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("pipeline_error", query_id=query_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    # ── Step 4: Build response ───────────────────────────────────────────────
    latency_ms = int((time.monotonic() - start) * 1000)
    citations = [Citation(**c) for c in result_state.get("citations", [])]

    response = QueryResponse(
        query_id=query_id,
        answer=result_state.get("answer", ""),
        citations=citations,
        retrieval_method="hybrid_rrf_rerank",
        latency_ms=latency_ms,
        model_used=result_state.get("model_used", settings.generation_model),
        from_cache=False,
        trace_url=f"{settings.langfuse_host}/trace/{query_id}",
    )

    # ── Step 5: Record Prometheus metrics ────────────────────────────────────
    rag_queries_total.labels(from_cache="false").inc()
    rag_query_latency_seconds.observe(latency_ms / 1000)
    rag_chunks_retrieved.observe(len(citations))

    # ── Step 6: Write to semantic cache ──────────────────────────────────────
    try:
        cacheable = response.model_dump()
        await cache_store(redis, query_embedding, cacheable, ttl=settings.cache_ttl_seconds)
    except Exception as e:
        logger.warning("cache_write_failed", query_id=query_id, error=str(e))

    trace.update(
        output={"answer": response.answer[:200], "citations": len(citations), "latency_ms": latency_ms}
    )
    logger.info(
        "query_done",
        query_id=query_id,
        latency_ms=latency_ms,
        citations=len(citations),
        model=response.model_used,
        from_cache=False,
    )
    return response
