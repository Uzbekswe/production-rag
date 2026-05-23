from collections import defaultdict

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.query import ScoredChunk

logger = get_logger(__name__)


class ReciprocalRankFusion:
    """
    Merges dense and sparse result lists using Reciprocal Rank Fusion (RRF).

    Formula (Cormack et al., 2009):
        rrf_score(d) = Σ_list  1 / (k + rank_in_list)

    k=60 is the smoothing constant from the original paper. It prevents a document
    ranked #1 in one list from dominating — the denominator is always at least 60,
    so the maximum possible score per list is 1/61 ≈ 0.016.

    Why this works: dense and sparse retrieval are complementary. A chunk appearing
    in the top of BOTH lists is almost certainly relevant — RRF gives it ~2× the
    score of a chunk appearing in only one list. Chunks in neither list score 0.

    No normalization needed: RRF scores are directly comparable across lists because
    they're rank-based (not scale-dependent like raw embedding distances or BM25 TF-IDF).
    """

    def fuse(
        self,
        dense: list[ScoredChunk],
        sparse: list[ScoredChunk],
        k: int | None = None,
    ) -> list[ScoredChunk]:
        rrf_k = k or settings.rrf_k

        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, ScoredChunk] = {}

        for rank, chunk in enumerate(dense):
            rrf_scores[chunk.chunk_id] += 1.0 / (rrf_k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse):
            rrf_scores[chunk.chunk_id] += 1.0 / (rrf_k + rank + 1)
            # Keep the dense version if the chunk appears in both lists
            # (dense payload is identical; we just prefer the version we already have)
            chunk_map.setdefault(chunk.chunk_id, chunk)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused: list[ScoredChunk] = []
        for chunk_id, rrf_score in ranked:
            chunk = chunk_map[chunk_id]
            fused.append(chunk.model_copy(update={"score": rrf_score}))

        logger.info(
            "rrf_fusion_done",
            dense_count=len(dense),
            sparse_count=len(sparse),
            fused_count=len(fused),
            k=rrf_k,
        )
        return fused
