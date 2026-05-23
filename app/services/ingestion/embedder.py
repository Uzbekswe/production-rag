import asyncio

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BGEEmbedder:
    """
    Singleton wrapper around BAAI/bge-m3.

    The model takes ~10 seconds and ~3GB RAM to load. Loading it per-request
    would make the first query 10 seconds slower. Loading once at startup
    (via get() in the lifespan) amortizes that cost across all requests.

    CPU inference on Apple Silicon: ~50ms per batch of 32 chunks.
    Output: 1024-dimensional dense vectors (used for Qdrant ANN search).
    """

    _instance: "BGEEmbedder | None" = None

    @classmethod
    def get(cls) -> "BGEEmbedder":
        if cls._instance is None:
            logger.info("embedder_loading", model=settings.embedding_model)
            cls._instance = cls()
            logger.info("embedder_ready", model=settings.embedding_model)
        return cls._instance

    def __init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel
        # use_fp16=False: Apple Silicon MPS doesn't support fp16 ops uniformly;
        # float32 is safer and the accuracy difference is negligible for retrieval.
        self._model = BGEM3FlagModel(settings.embedding_model, use_fp16=False)

    def embed_chunks(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode a list of texts. Returns one 1024-dim vector per text."""
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [v.tolist() for v in output["dense_vecs"]]

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query string for retrieval."""
        output = self._model.encode(
            [query],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"][0].tolist()

    async def embed_chunks_async(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Async wrapper — runs embedding in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_chunks, texts, batch_size)
