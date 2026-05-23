from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document


async def create_document(
    db: AsyncSession,
    *,
    filename: str,
    file_type: str,
    source_url: str | None = None,
) -> Document:
    doc = Document(filename=filename, file_type=file_type, source_url=source_url)
    db.add(doc)
    await db.flush()  # assigns id without committing — caller controls the transaction
    return doc


async def get_document(db: AsyncSession, doc_id: UUID) -> Document | None:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    return result.scalar_one_or_none()


async def list_documents(db: AsyncSession, limit: int = 100) -> list[Document]:
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


async def update_document_status(
    db: AsyncSession,
    doc_id: UUID,
    status: str,
    chunk_count: int | None = None,
    error_msg: str | None = None,
) -> Document | None:
    doc = await get_document(db, doc_id)
    if doc is None:
        return None
    doc.status = status
    doc.updated_at = datetime.utcnow()
    if chunk_count is not None:
        doc.chunk_count = chunk_count
    if error_msg is not None:
        doc.error_msg = error_msg
    await db.flush()
    return doc


async def delete_document(db: AsyncSession, doc_id: UUID) -> bool:
    doc = await get_document(db, doc_id)
    if doc is None:
        return False
    await db.delete(doc)
    await db.flush()
    return True
