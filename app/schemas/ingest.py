from pydantic import BaseModel, HttpUrl


class IngestURLRequest(BaseModel):
    url: HttpUrl
    title: str | None = None


class IngestResponse(BaseModel):
    """Returned immediately (202 Accepted) for both file and URL ingestion."""
    job_id: str       # same as doc_id — used to poll status
    doc_id: str
    status: str = "pending"
    message: str


class JobStatusResponse(BaseModel):
    """Returned by GET /ingest/{job_id} — reflects current documents.status."""
    job_id: str
    doc_id: str
    status: str       # pending | processing | ready | failed
    chunk_count: int | None = None
    error_msg: str | None = None
