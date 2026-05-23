from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk


async def create_chunks(db: AsyncSession, chunks: list[dict]) -> list[Chunk]:
    """
    Bulk-insert chunks. Each dict must have the keys that match Chunk columns.
    Uses a single flush so all chunks land in one round-trip to Postgres.
    """
    orm_chunks = [Chunk(**c) for c in chunks]
    db.add_all(orm_chunks)
    await db.flush()
    return orm_chunks


async def get_chunks_by_doc(db: AsyncSession, doc_id: UUID) -> list[Chunk]:
    result = await db.execute(
        select(Chunk)
        .where(Chunk.doc_id == doc_id)
        .order_by(Chunk.chunk_index)
    )
    return list(result.scalars().all())


async def get_all_chunks(db: AsyncSession) -> list[Chunk]:
    """Return all chunks — used by the BM25 indexer to rebuild the full index."""
    result = await db.execute(select(Chunk).order_by(Chunk.doc_id, Chunk.chunk_index))
    return list(result.scalars().all())


async def get_all_chunks_for_bm25(db: AsyncSession) -> list[dict]:
    """
    Fetch only the two columns BM25 needs: id + full_text.
    Skips raw_text, context, and all other columns — meaningfully faster
    than get_all_chunks() when rebuilding at startup on large corpora.
    """
    result = await db.execute(
        select(Chunk.id, Chunk.full_text).order_by(Chunk.doc_id, Chunk.chunk_index)
    )
    return [{"id": str(row.id), "full_text": row.full_text} for row in result]


async def delete_chunks_by_doc(db: AsyncSession, doc_id: UUID) -> int:
    """Delete all chunks for a document. Returns the count of deleted rows."""
    result = await db.execute(
        delete(Chunk).where(Chunk.doc_id == doc_id)
    )
    await db.flush()
    return result.rowcount
