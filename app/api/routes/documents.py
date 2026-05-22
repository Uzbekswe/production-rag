from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.document import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    db: AsyncSession = Depends(get_db),
) -> list[DocumentOut]:
    # TODO: return await document_repo.list_all(db)
    return []


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentOut:
    # TODO: doc = await document_repo.get_by_id(db, document_id)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    # TODO: await document_repo.delete(db, document_id)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
