import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.query import QueryRequest, ScoredChunk
from app.services.ingestion.embedder import BGEEmbedder
from app.services.retrieval.dense import QdrantRetriever
from app.services.retrieval.fusion import ReciprocalRankFusion
from app.services.retrieval.reranker import BGEReranker
from app.services.retrieval.sparse import BM25Retriever
from app.services.generation.groq_gen import GroqGenerator

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)

_dense = QdrantRetriever()
_sparse = BM25Retriever()
_fusion = ReciprocalRankFusion()
_groq = GroqGenerator()


@router.post("/stream")
async def query_stream(payload: QueryRequest) -> StreamingResponse:
    """
    Streaming variant of POST /query.

    Runs the same hybrid retrieval + rerank pipeline, then streams
    generation tokens as Server-Sent Events instead of waiting for
    the complete answer.

    SSE event format:
      data: {"type": "token",   "content": "Apple"}\n\n
      data: {"type": "token",   "content": "'s gross"}\n\n
      ...
      data: {"type": "done",    "citations": [...], "model_used": "..."}\n\n

    Why retrieval is called directly (not through LangGraph):
    LangGraph's ainvoke() waits for the full graph to complete before
    returning. Streaming requires calling Groq with stream=True after
    retrieval finishes — we call the same singleton services directly,
    skipping the graph wrapper for just this endpoint.
    """

    async def event_generator():
        query = payload.query
        logger.info("stream_query_start", query=query[:100])

        try:
            # ── Embed query ───────────────────────────────────────────────
            loop = asyncio.get_event_loop()
            embedder = BGEEmbedder.get()
            await loop.run_in_executor(None, embedder.embed_query, query)

            # ── Hybrid retrieval ──────────────────────────────────────────
            dense_results, sparse_results = await asyncio.gather(
                _dense.search(query, top_k=settings.retrieval_top_k),
                _sparse.search(query, top_k=settings.retrieval_top_k),
            )
            fused = _fusion.fuse(dense_results, sparse_results, k=settings.rrf_k)

            # ── Rerank ────────────────────────────────────────────────────
            chunks_raw = [
                ScoredChunk(**c) if isinstance(c, dict) else c for c in fused
            ]
            reranker = BGEReranker.get()
            chunks = await reranker.rerank_async(
                query, chunks_raw, top_k=settings.rerank_top_k
            )

            logger.info(
                "stream_retrieval_done",
                dense=len(dense_results),
                sparse=len(sparse_results),
                reranked=len(chunks),
            )

            # ── Stream generation ─────────────────────────────────────────
            async for event in _groq.generate_stream(query, chunks):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as exc:
            logger.error("stream_error", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
