from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health, ingest, query
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.qdrant import ensure_collection_exists, get_qdrant_client

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("startup", env=settings.app_env, model=settings.generation_model)

    qdrant = get_qdrant_client()
    await ensure_collection_exists(qdrant)
    logger.info("qdrant_ready", collection=settings.qdrant_collection)

    yield

    logger.info("shutdown")


app = FastAPI(
    title="Production RAG Knowledge Copilot",
    description="Agentic RAG with Anthropic Contextual Retrieval + Citations API",
    version="0.1.0",
    lifespan=lifespan,
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
app.include_router(documents.router, prefix="/api/v1")
