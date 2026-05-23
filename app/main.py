from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.routes import documents, eval, health, ingest, query, stream
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.logging import configure_logging, get_logger
from app.core.qdrant import ensure_collection_exists, get_qdrant_client
from app.core.tracing import flush as flush_traces
from app.repositories.chunk_repo import get_all_chunks_for_bm25
from app.services.ingestion.bm25_indexer import BM25Index

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("startup", env=settings.app_env, model=settings.generation_model)

    qdrant = get_qdrant_client()
    await ensure_collection_exists(qdrant)
    logger.info("qdrant_ready", collection=settings.qdrant_collection)

    # BM25 self-healing: if the pickle is missing (new machine, clean clone,
    # accidental delete) rebuild from Postgres instead of silently degrading
    # to dense-only retrieval. The pickle is a derived cache — Postgres is truth.
    bm25 = BM25Index.get()
    if not bm25.is_ready:
        logger.warning("bm25_missing_rebuilding", reason="index file not found")
        async with AsyncSessionLocal() as db:
            chunks = await get_all_chunks_for_bm25(db)
        if chunks:
            bm25.build(chunks)
            logger.info("bm25_rebuilt_on_startup", chunk_count=len(chunks))
        else:
            logger.info("bm25_skipped_no_data", reason="postgres has no chunks yet")

    # Startup integrity: emit a single log line showing the loaded state of
    # every retrieval component so the first startup lines tell you exactly
    # what the system has available.
    collection_info = await qdrant.get_collection(settings.qdrant_collection)
    logger.info(
        "startup_complete",
        qdrant_vectors=collection_info.points_count or 0,
        bm25_chunks=bm25.chunk_count,
        bm25_ready=bm25.is_ready,
        vessl_endpoint=bool(settings.vessl_endpoint),
    )

    yield

    flush_traces()
    logger.info("shutdown")


app = FastAPI(
    title="Production RAG Knowledge Copilot",
    description="Agentic RAG with Anthropic Contextual Retrieval + Citations API",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    data = generate_latest()
    return Response(
        content=data,
        media_type=CONTENT_TYPE_LATEST,
        headers={"Content-Length": str(len(data))},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(stream.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(eval.router, prefix="/api/v1")
