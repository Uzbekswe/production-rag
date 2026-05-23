import json
from functools import lru_cache

import numpy as np
import redis.asyncio as aioredis

from app.core.config import settings

CACHE_KEY_PREFIX = "rag:semantic_cache:"


@lru_cache(maxsize=1)
def get_redis_client() -> aioredis.Redis:
    return aioredis.from_url(
        settings.redis_url,
        db=settings.redis_cache_db,
        encoding="utf-8",
        decode_responses=False,
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


async def cache_lookup(
    redis: aioredis.Redis,
    query_embedding: list[float],
    threshold: float | None = None,
) -> dict | None:
    """
    Semantic cache lookup: finds a cached response whose stored embedding is
    cosine-similar to query_embedding above threshold (default from settings).

    Why cosine similarity instead of exact key match: two phrasings of the same
    question ("Apple revenue FY2024?" vs "How much did Apple earn in 2024?") produce
    similar embeddings. Exact-match caching would miss these — wasting a Groq call.
    At demo scale (<100 cached entries) a linear scan is fast enough.
    """
    if threshold is None:
        threshold = settings.cache_similarity_threshold

    cursor = 0
    pattern = f"{CACHE_KEY_PREFIX}*"

    while True:
        cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=100)
        for key in keys:
            raw = await redis.get(key)
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                stored_embedding = entry["embedding"]
                if _cosine_similarity(query_embedding, stored_embedding) >= threshold:
                    return entry["response"]
            except (json.JSONDecodeError, KeyError):
                continue
        if cursor == 0:
            break

    return None


async def cache_store(
    redis: aioredis.Redis,
    query_embedding: list[float],
    response: dict,
    ttl: int | None = None,
) -> None:
    """
    Store a query embedding + response in the semantic cache.
    Key is a hash of the embedding so identical embeddings overwrite gracefully.
    TTL defaults to settings.cache_ttl_seconds (1 hour).
    """
    if ttl is None:
        ttl = settings.cache_ttl_seconds

    # Use a short hash of the embedding as the key suffix — collision risk is negligible
    key_suffix = hex(hash(tuple(round(v, 4) for v in query_embedding[:8])))[2:]
    key = f"{CACHE_KEY_PREFIX}{key_suffix}"

    payload = json.dumps({"embedding": query_embedding, "response": response})
    await redis.set(key, payload, ex=ttl)
