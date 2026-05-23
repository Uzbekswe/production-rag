from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.core.qdrant import get_qdrant_client
from app.repositories import chunk_repo, document_repo
from app.schemas.document import DocumentList, DocumentRead
from app.services.ingestion.bm25_indexer import BM25Index

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)


@router.get("", response_model=DocumentList)
async def list_documents(
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> DocumentList:
    docs = await document_repo.list_documents(db, limit=limit)
    return DocumentList(
        documents=[DocumentRead.model_validate(d) for d in docs],
        total=len(docs),
    )


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentRead:
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid document_id",
        )
    doc = await document_repo.get_document(db, doc_uuid)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentRead.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete a document and all its data from Postgres, Qdrant, and the BM25 index.
    Order matters: collect Qdrant IDs first (before cascade deletes the chunk rows),
    then delete from Postgres, then clean up Qdrant, then rebuild BM25.
    """
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid document_id",
        )

    doc = await document_repo.get_document(db, doc_uuid)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Step 1: collect Qdrant point IDs before the rows are gone
    chunks = await chunk_repo.get_chunks_by_doc(db, doc_uuid)
    qdrant_ids = [str(c.qdrant_id) for c in chunks if c.qdrant_id is not None]

    # Step 2: delete from Postgres — FK cascade removes chunk rows automatically
    await document_repo.delete_document(db, doc_uuid)
    await db.commit()
    logger.info("document_deleted_postgres", doc_id=document_id, chunk_count=len(chunks))

    # Step 3: delete vectors from Qdrant
    if qdrant_ids:
        from app.core.config import settings
        qdrant = get_qdrant_client()
        await qdrant.delete(
            collection_name=settings.qdrant_collection,
            points_selector=qdrant_ids,
        )
        logger.info("document_deleted_qdrant", doc_id=document_id, points=len(qdrant_ids))

    # Step 4: rebuild BM25 from remaining chunks
    remaining = await chunk_repo.get_all_chunks(db)
    bm25 = BM25Index.get()
    if remaining:
        bm25.build([
            {"id": c.qdrant_id or c.id, "full_text": c.full_text}
            for c in remaining
        ])
    else:
        bm25._bm25 = None
        bm25._chunk_ids = []

    logger.info("bm25_rebuilt_after_delete", doc_id=document_id, remaining=len(remaining))
