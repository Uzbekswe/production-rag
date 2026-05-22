from functools import lru_cache

import redis.asyncio as aioredis

from app.core.config import settings


@lru_cache(maxsize=1)
def get_redis_client() -> aioredis.Redis:
    return aioredis.from_url(
        settings.redis_url,
        db=settings.redis_cache_db,
        encoding="utf-8",
        decode_responses=False,
    )
