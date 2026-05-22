import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.schemas.ingest import IngestFileResponse, IngestURLRequest, IngestURLResponse

router = APIRouter(prefix="/ingest", tags=["ingestion"])
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50


@router.post("/file", response_model=IngestFileResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> IngestFileResponse:
    import os

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit",
        )

    document_id = str(uuid.uuid4())
    logger.info("file_ingestion_queued", document_id=document_id, filename=file.filename)

    # TODO: background_tasks.add_task(ingest_pipeline.run, document_id, contents, ext, db)

    return IngestFileResponse(
        document_id=document_id,
        filename=file.filename or "unknown",
        status="queued",
        message="Document queued for processing. Poll /documents/{id} for status.",
    )


@router.post("/url", response_model=IngestURLResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_url(
    payload: IngestURLRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> IngestURLResponse:
    document_id = str(uuid.uuid4())
    url_str = str(payload.url)
    logger.info("url_ingestion_queued", document_id=document_id, url=url_str)

    # TODO: background_tasks.add_task(ingest_pipeline.run_url, document_id, url_str, db)

    return IngestURLResponse(
        document_id=document_id,
        url=url_str,
        status="queued",
        message="URL queued for processing.",
    )
