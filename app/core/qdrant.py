from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import settings

VECTOR_SIZE = 1024  # BGE-M3 output dimension


@lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url)


async def ensure_collection_exists(client: AsyncQdrantClient) -> None:
    collections = await client.get_collections()
    names = {c.name for c in collections.collections}
    if settings.qdrant_collection not in names:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
