"""
LangGraph node implementations for the RAG query pipeline.

Each function receives a RAGState dict and returns an updated RAGState dict.
LangGraph merges the returned dict into the current state — only returned keys
are updated, unmentioned keys stay unchanged.

Node execution order (defined in graph.py):
  query_rewriter → hybrid_retriever → reranker → sufficiency_checker
    ↑                                                      |
    └──────── (retry if not sufficient) ───────────────────┘
                                                           ↓
                                                       generate
"""

import asyncio

from groq import AsyncGroq

from app.core.config import settings
from app.core.logging import get_logger
from app.core.tracing import get_langfuse, span as lf_span
from app.schemas.query import ScoredChunk
from app.services.agent.state import RAGState
from app.services.generation import router as generation_router
from app.services.retrieval.dense import QdrantRetriever
from app.services.retrieval.fusion import ReciprocalRankFusion
from app.services.retrieval.reranker import BGEReranker
from app.services.retrieval.sparse import BM25Retriever

logger = get_logger(__name__)

_dense = QdrantRetriever()
_sparse = BM25Retriever()
_fusion = ReciprocalRankFusion()

REWRITE_SYSTEM = """\
You are a search query optimizer for financial document retrieval.
Given a user question, rewrite it to be more specific and retrieval-friendly.
Focus on key entities (company names, ticker symbols, fiscal years, metrics).
Return ONLY the rewritten query — no explanation, no preamble."""


async def query_rewriter_node(state: RAGState) -> dict:
    """
    Rewrite the current query to improve retrieval quality.

    On the first attempt (retrieval_attempt=0): expand the original query with
    financial context to improve recall.
    On retry (retrieval_attempt>0): the sufficiency checker already ran; we know
    the previous retrieval missed something. The rewriter gets a second chance to
    rephrase with different terminology.
    """
    current_query = state["rewritten_queries"][-1] if state["rewritten_queries"] else state["query"]
    attempt = state["retrieval_attempt"]

    trace = get_langfuse().trace(id=state["trace_id"])

    with lf_span(trace, "query_rewriter",
                 input={"query": current_query, "attempt": attempt}) as s:
        try:
            client = AsyncGroq(api_key=settings.groq_api_key)
            resp = await client.chat.completions.create(
                model=settings.context_enrichment_model,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": f"Original query: {current_query}"},
                ],
                max_tokens=100,
                temperature=0.3,
            )
            rewritten = resp.choices[0].message.content.strip()
            usage = resp.usage
            s.update(output={
                "rewritten": rewritten,
                "model": settings.context_enrichment_model,
                "tokens_in": usage.prompt_tokens if usage else 0,
                "tokens_out": usage.completion_tokens if usage else 0,
            })
        except Exception as e:
            logger.warning("query_rewrite_failed", error=str(e), attempt=attempt)
            rewritten = current_query
            s.update(output={"rewritten": rewritten, "error": str(e)})

    logger.info(
        "query_rewritten",
        attempt=attempt,
        original=current_query[:80],
        rewritten=rewritten[:80],
    )

    return {
        "rewritten_queries": [*state["rewritten_queries"], rewritten],
        "retrieval_attempt": attempt + 1,
    }


async def hybrid_retriever_node(state: RAGState) -> dict:
    """
    Run dense and sparse retrieval in parallel, then fuse with RRF.

    asyncio.gather runs both searches concurrently — Qdrant ANN and BM25 are
    independent operations and can overlap. Total latency ≈ max(dense, sparse)
    instead of sum(dense + sparse).
    """
    query = state["rewritten_queries"][-1]
    trace = get_langfuse().trace(id=state["trace_id"])

    with lf_span(trace, "hybrid_retriever",
                 input={"query": query, "top_k": settings.retrieval_top_k}) as s:
        dense_results, sparse_results = await asyncio.gather(
            _dense.search(query, top_k=settings.retrieval_top_k),
            _sparse.search(query, top_k=settings.retrieval_top_k),
        )
        fused = _fusion.fuse(dense_results, sparse_results, k=settings.rrf_k)
        s.update(output={
            "dense": len(dense_results),
            "sparse": len(sparse_results),
            "fused": len(fused),
        })

    logger.info(
        "hybrid_retrieval_done",
        query=query[:60],
        dense=len(dense_results),
        sparse=len(sparse_results),
        fused=len(fused),
    )

    return {"retrieved_chunks": [c.model_dump() for c in fused]}


async def reranker_node(state: RAGState) -> dict:
    """
    Cross-encoder rerank: top-50 fused chunks → top-5 most relevant.

    The reranker sees (query, chunk) jointly — unlike the bi-encoder similarity
    used in dense retrieval. It can reason about whether the passage specifically
    answers this specific question, not just whether they share semantic space.
    """
    query = state["rewritten_queries"][-1]
    chunks = [ScoredChunk(**c) for c in state["retrieved_chunks"]]
    trace = get_langfuse().trace(id=state["trace_id"])

    if not chunks:
        return {"retrieved_chunks": []}

    with lf_span(trace, "reranker",
                 input={"chunks_in": len(chunks), "top_k": settings.rerank_top_k}) as s:
        reranker = BGEReranker.get()
        reranked = await reranker.rerank_async(
            query, chunks, top_k=settings.rerank_top_k
        )
        s.update(output={
            "chunks_out": len(reranked),
            "top_score": round(reranked[0].score, 3) if reranked else 0,
        })

    logger.info(
        "rerank_done",
        input=len(chunks),
        output=len(reranked),
        top_score=reranked[0].score if reranked else None,
    )

    return {"retrieved_chunks": [c.model_dump() for c in reranked]}


async def sufficiency_checker_node(state: RAGState) -> dict:
    """
    Heuristic sufficiency check: do we have enough good chunks to generate from?

    Sufficient if: ≥3 chunks AND average rerank score ≥ 0.2
    Insufficient if: fewer chunks or very low scores (retrieval likely failed)

    Why a heuristic instead of an LLM judge (as in architecture.md):
    An LLM call here adds 1-2 seconds and burns Groq tokens on EVERY query —
    even when retrieval obviously worked. The heuristic catches the real problem
    cases (empty results, zero-score chunks) at zero cost. An LLM judge would
    be the Phase 4 upgrade for adversarial query handling.

    The retry loop (in graph.py) allows up to max_agent_retries attempts.
    """
    chunks = state["retrieved_chunks"]
    trace = get_langfuse().trace(id=state["trace_id"])

    if not chunks:
        with lf_span(trace, "sufficiency_checker",
                     input={"chunk_count": 0, "avg_score": 0.0}) as s:
            s.update(output={"is_sufficient": False, "reason": "no_chunks"})
        logger.info("sufficiency_check", result="insufficient", reason="no_chunks")
        return {"is_sufficient": False}

    avg_score = sum(c["score"] for c in chunks) / len(chunks)
    is_sufficient = len(chunks) >= 3 and avg_score >= 0.2

    with lf_span(trace, "sufficiency_checker",
                 input={"chunk_count": len(chunks), "avg_score": round(avg_score, 3)}) as s:
        s.update(output={
            "is_sufficient": is_sufficient,
            "attempt": state["retrieval_attempt"],
        })

    logger.info(
        "sufficiency_check",
        result="sufficient" if is_sufficient else "insufficient",
        chunk_count=len(chunks),
        avg_score=round(avg_score, 3),
        attempt=state["retrieval_attempt"],
    )

    return {"is_sufficient": is_sufficient}


async def generate_node(state: RAGState) -> dict:
    """
    Generate the answer with inline [Source N] citations.

    Calls the generation router which picks Groq (default, free) or
    LongCite (demo, VESSL) based on settings.longcite_endpoint.

    Even if retrieval was marked insufficient, we generate anyway — the
    model is instructed to say "sources don't contain enough information"
    rather than hallucinate. Graceful degradation > hard failure.
    """
    query = state["rewritten_queries"][-1]
    chunks = [ScoredChunk(**c) for c in state["retrieved_chunks"]]
    trace = get_langfuse().trace(id=state["trace_id"])

    with lf_span(trace, "generate",
                 input={"query": query, "chunks": len(chunks)}) as s:
        result = await generation_router.generate(query, chunks)
        s.update(output={
            "answer_len": len(result.answer),
            "citations": len(result.citations),
            "model": result.model_used,
        })

    logger.info(
        "generation_done",
        model=result.model_used,
        answer_len=len(result.answer),
        citations=len(result.citations),
    )

    return {
        "answer": result.answer,
        "citations": [c.model_dump() for c in result.citations],
        "model_used": result.model_used,
    }
