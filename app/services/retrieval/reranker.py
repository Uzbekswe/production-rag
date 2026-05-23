import asyncio

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.query import ScoredChunk

logger = get_logger(__name__)


class BGEReranker:
    """
    Cross-encoder reranker: scores (query, chunk) pairs jointly → top-k.

    Why cross-encoder > bi-encoder for the final reranking step:
    - Bi-encoder (BGE-M3): encodes query and chunk *separately*, compares via cosine.
      Fast enough for top-50 ANN search, but loses fine-grained query-chunk interaction.
    - Cross-encoder: concatenates query + chunk, runs them through the model together.
      Sees exactly how this query relates to this passage — much higher accuracy.
      Too slow for full-corpus search, but perfect for reranking top-50 → top-5.

    Singleton: ~1.5GB RAM, loads once. Async wrapper offloads CPU work to thread pool.
    """

    _instance: "BGEReranker | None" = None

    @classmethod
    def get(cls) -> "BGEReranker":
        if cls._instance is None:
            logger.info("reranker_loading", model=settings.reranker_model)
            cls._instance = cls()
            logger.info("reranker_ready", model=settings.reranker_model)
        return cls._instance

    def __init__(self) -> None:
        from FlagEmbedding import FlagReranker
        self._model = FlagReranker(settings.reranker_model, use_fp16=False)

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        k = top_k or settings.rerank_top_k
        if not chunks:
            return []

        pairs = [[query, c.raw_text] for c in chunks]
        # normalize=True maps raw logits → [0, 1] via sigmoid — easier to threshold
        scores: list[float] = self._model.compute_score(pairs, normalize=True)

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        result = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in ranked[:k]
        ]

        logger.info(
            "rerank_done",
            input_count=len(chunks),
            output_count=len(result),
            top_score=result[0].score if result else None,
        )
        return result

    async def rerank_async(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Thread-pool wrapper — keeps the async event loop free during CPU reranking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, chunks, top_k)
