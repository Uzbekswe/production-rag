from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

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
            # Scalar quantization: compresses float32 (4 bytes) → int8 (1 byte) per dimension.
            # 4x memory reduction with ~1% accuracy loss — essential for local hardware.
            # quantile=0.99 clips outlier values before quantizing to preserve most info.
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=False,
                )
            ),
            on_disk_payload=True,  # metadata (text, page_num, etc.) stored on disk, not RAM
        )
