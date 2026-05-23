import os
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.repositories import document_repo
from app.schemas.ingest import IngestResponse, IngestURLRequest, JobStatusResponse
from app.workers.ingest_worker import UPLOAD_DIR, download_and_ingest_url, run_ingestion_background

router = APIRouter(prefix="/ingest", tags=["ingestion"])
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".htm", ".html", ".txt", ".md"}
MAX_FILE_SIZE_MB = 100


@router.post("", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Upload a file for ingestion. Returns immediately with a job_id.
    Poll GET /ingest/{job_id} to track progress.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit",
        )

    file_type = ext.lstrip(".")
    doc_id = uuid4()

    # Stage the file to disk before responding — the background task reads
    # from here and deletes it when done (success or failure).
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{doc_id}{ext}"
    dest.write_bytes(contents)

    doc = await document_repo.create_document(
        db,
        filename=file.filename or dest.name,
        file_type=file_type,
    )
    await db.commit()

    background_tasks.add_task(run_ingestion_background, dest, doc.id, file_type)

    logger.info("ingest_queued", doc_id=str(doc.id), filename=file.filename, bytes=len(contents))
    return IngestResponse(
        job_id=str(doc.id),
        doc_id=str(doc.id),
        message=f"'{file.filename}' queued. Poll /api/v1/ingest/{doc.id} for status.",
    )


@router.post("/url", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_url(
    payload: IngestURLRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """Ingest a document from a URL (PDF link or HTML page). Docling handles both."""
    url_str = str(payload.url)
    ext = Path(url_str.split("?")[0]).suffix.lower()
    file_type = ext.lstrip(".") if ext in ALLOWED_EXTENSIONS else "htm"

    doc = await document_repo.create_document(
        db,
        filename=payload.title or url_str.split("/")[-1] or "document",
        file_type=file_type,
        source_url=url_str,
    )
    await db.commit()

    background_tasks.add_task(download_and_ingest_url, url_str, doc.id, file_type)

    logger.info("ingest_url_queued", doc_id=str(doc.id), url=url_str)
    return IngestResponse(
        job_id=str(doc.id),
        doc_id=str(doc.id),
        message=f"URL queued. Poll /api/v1/ingest/{doc.id} for status.",
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """Poll ingestion job status. job_id == doc_id."""
    try:
        doc_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid job_id format",
        )

    doc = await document_repo.get_document(db, doc_uuid)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return JobStatusResponse(
        job_id=str(doc.id),
        doc_id=str(doc.id),
        status=doc.status,
        chunk_count=doc.chunk_count if doc.chunk_count > 0 else None,
        error_msg=doc.error_msg,
    )
