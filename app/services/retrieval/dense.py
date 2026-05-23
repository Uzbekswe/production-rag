import asyncio

from app.core.config import settings
from app.core.logging import get_logger
from app.core.qdrant import get_qdrant_client
from app.schemas.query import ScoredChunk
from app.services.ingestion.embedder import BGEEmbedder

logger = get_logger(__name__)


class QdrantRetriever:
    """
    Dense (semantic) retrieval: embed the query → ANN search in Qdrant → ScoredChunks.

    BGE-M3 maps the query to a 1024-dim vector. Qdrant's HNSW graph returns the
    50 nearest stored vectors by cosine similarity. "Nearest" here means "most
    semantically similar" — not keyword matching, but meaning matching.

    run_in_executor: embed_query is synchronous and CPU-bound. We offload it to
    a thread pool so the async event loop stays free during the ~50ms encoding.
    """

    async def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or settings.retrieval_top_k

        # Embed query in a thread pool (sync model, async context)
        loop = asyncio.get_event_loop()
        embedder = BGEEmbedder.get()
        query_vec = await loop.run_in_executor(None, embedder.embed_query, query)

        qdrant = get_qdrant_client()
        results = await qdrant.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vec,
            limit=k,
            with_payload=True,
        )

        chunks: list[ScoredChunk] = []
        for hit in results:
            payload = hit.payload or {}
            chunks.append(
                ScoredChunk(
                    chunk_id=str(hit.id),
                    doc_id=payload.get("doc_id", ""),
                    filename=payload.get("filename", "unknown"),
                    raw_text=payload.get("raw_text", ""),
                    full_text=payload.get("full_text", ""),
                    page_num=payload.get("page_num"),
                    char_start=payload.get("char_start"),
                    char_end=payload.get("char_end"),
                    score=float(hit.score),
                )
            )

        logger.info("dense_search_done", query_len=len(query), results=len(chunks))
        return chunks
