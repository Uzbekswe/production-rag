from app.core.config import settings
from app.core.logging import get_logger
from app.core.qdrant import get_qdrant_client
from app.schemas.query import ScoredChunk
from app.services.ingestion.bm25_indexer import BM25Index

logger = get_logger(__name__)


class BM25Retriever:
    """
    Sparse (keyword) retrieval: BM25 scoring → batch Qdrant payload fetch → ScoredChunks.

    BM25 gives us chunk IDs (which are Qdrant point IDs) with keyword scores.
    We then fetch the full payloads from Qdrant in a single batch call — avoiding
    N individual Postgres queries. All the metadata we need (raw_text, filename,
    page_num, etc.) is already in the Qdrant payload from ingest time.

    Why not just return BM25 results directly?
    BM25Index stores only (qdrant_id, score). To build a ScoredChunk with raw_text,
    filename, page_num etc., we need the payload — Qdrant is the fastest source.
    """

    async def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or settings.retrieval_top_k

        bm25_results = BM25Index.get().search(query, top_k=k)
        if not bm25_results:
            logger.info("sparse_search_done", query_len=len(query), results=0, reason="no_bm25_index")
            return []

        # Build a score map before the order changes
        score_map: dict[str, float] = {cid: score for cid, score in bm25_results}
        ids = list(score_map.keys())

        # Batch fetch payloads from Qdrant (one round-trip for all IDs)
        qdrant = get_qdrant_client()
        points = await qdrant.retrieve(
            collection_name=settings.qdrant_collection,
            ids=ids,
            with_payload=True,
        )

        chunks: list[ScoredChunk] = []
        for point in points:
            payload = point.payload or {}
            chunks.append(
                ScoredChunk(
                    chunk_id=str(point.id),
                    doc_id=payload.get("doc_id", ""),
                    filename=payload.get("filename", "unknown"),
                    raw_text=payload.get("raw_text", ""),
                    full_text=payload.get("full_text", ""),
                    page_num=payload.get("page_num"),
                    char_start=payload.get("char_start"),
                    char_end=payload.get("char_end"),
                    score=score_map.get(str(point.id), 0.0),
                )
            )

        # Sort by BM25 score descending (Qdrant.retrieve doesn't guarantee order)
        chunks.sort(key=lambda c: c.score, reverse=True)

        logger.info("sparse_search_done", query_len=len(query), results=len(chunks))
        return chunks
