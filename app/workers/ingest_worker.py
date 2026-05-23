"""
Background worker for document ingestion.

The API routes return 202 immediately and hand off to these functions via
FastAPI's BackgroundTasks. Because the HTTP response is already sent by the
time these run, they must create their own database session — the request-scoped
session from Depends(get_db) is closed when the response is returned.
"""

from pathlib import Path
from uuid import UUID

import httpx

from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger
from app.services.ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)

UPLOAD_DIR = Path("data/uploads")
_pipeline = IngestionPipeline()


async def run_ingestion_background(
    file_path: Path,
    doc_id: UUID,
    file_type: str,
) -> None:
    """
    Entry point for background ingestion. Creates its own DB session so it
    can run safely after the HTTP response has been sent.
    """
    async with AsyncSessionLocal() as db:
        try:
            await _pipeline.run(file_path, doc_id, file_type, db)
        except Exception as exc:
            # pipeline.run() already marks the doc as failed and commits.
            # Log here for visibility; don't re-raise (background tasks
            # swallow exceptions anyway, but explicit is cleaner).
            logger.error(
                "background_ingestion_error",
                doc_id=str(doc_id),
                error=str(exc),
            )
        finally:
            # Clean up the staging file after ingestion (success or failure)
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass


async def download_and_ingest_url(
    url: str,
    doc_id: UUID,
    file_type: str,
) -> None:
    """
    Downloads a URL to a staging file, then runs the ingestion pipeline.
    Handles both PDF URLs and HTML pages (Docling parses both).
    """
    dest = UPLOAD_DIR / f"{doc_id}.{file_type}"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            resp = await client.get(url, headers={"User-Agent": "ProductionRAG/0.1"})
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        logger.info("url_downloaded", url=url, doc_id=str(doc_id), bytes=len(resp.content))
    except Exception as exc:
        logger.error("url_download_failed", url=url, doc_id=str(doc_id), error=str(exc))
        async with AsyncSessionLocal() as db:
            from app.repositories import document_repo
            await document_repo.update_document_status(
                db, doc_id, "failed", error_msg=f"Download failed: {exc}"
            )
            await db.commit()
        return

    await run_ingestion_background(dest, doc_id, file_type)
