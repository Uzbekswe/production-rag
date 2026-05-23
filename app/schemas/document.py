from datetime import datetime

from pydantic import BaseModel


class DocumentRead(BaseModel):
    """Single document — returned by GET /documents and DELETE /documents/{id}."""
    id: str
    filename: str
    file_type: str
    status: str
    chunk_count: int
    source_url: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentList(BaseModel):
    """Paginated document list — returned by GET /documents."""
    documents: list[DocumentRead]
    total: int
