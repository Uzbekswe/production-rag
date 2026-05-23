import pickle
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

INDEX_PATH = Path("data/bm25_index.pkl")


class BM25Index:
    """
    In-memory BM25 keyword index persisted as a pickle file.

    Why BM25 at all when we have semantic search?
    Semantic search is great at "similar meaning" but bad at exact matches.
    A query like "NVDA Q3 2024 revenue" relies on the model knowing that
    "NVDA" = "NVIDIA" and "Q3" = "third quarter" — sometimes it does,
    sometimes it doesn't. BM25 finds the exact tokens every time.
    RRF fusion (Phase 2) combines both signals for the best of both worlds.

    The full index is rebuilt on every ingest because BM25Okapi (from rank-bm25)
    doesn't support incremental updates. At 2,500–10,000 chunks, a full rebuild
    takes under 1 second. At 1M+ chunks you'd switch to Elasticsearch.
    """

    _instance: "BM25Index | None" = None

    @classmethod
    def get(cls) -> "BM25Index":
        if cls._instance is None:
            cls._instance = cls()
            loaded = cls._instance.load()
            if loaded:
                logger.info("bm25_loaded", index_path=str(INDEX_PATH))
            else:
                logger.info("bm25_empty", reason="no index file found")
        return cls._instance

    def __init__(self) -> None:
        self._bm25 = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[dict]) -> None:
        """
        Rebuild index from scratch over all chunks.
        chunks: list of dicts with at minimum {"id": ..., "full_text": ...}
        """
        from rank_bm25 import BM25Okapi

        corpus = [c["full_text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._chunk_ids = [str(c["id"]) for c in chunks]

        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self._chunk_ids}, f)

        logger.info("bm25_built", chunk_count=len(chunks), index_path=str(INDEX_PATH))

    def load(self) -> bool:
        if not INDEX_PATH.exists():
            return False
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]
        return True

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """
        Returns [(chunk_id, score), ...] sorted by score descending.
        Only returns chunks with score > 0 (i.e. at least one query token matched).
        """
        if self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            (self._chunk_ids[i], float(s))
            for i, s in ranked[:top_k]
            if s > 0
        ]

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_ids)
