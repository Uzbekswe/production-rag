from pathlib import Path
from uuid import UUID

from qdrant_client.models import PointStruct
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.core.qdrant import get_qdrant_client
from app.repositories import chunk_repo, document_repo
from app.services.ingestion.bm25_indexer import BM25Index
from app.services.ingestion.chunker import SemanticChunker
from app.services.ingestion.embedder import BGEEmbedder
from app.services.ingestion.enricher import ContextualEnricher
from app.services.ingestion.parser import DocumentParser

logger = get_logger(__name__)

_parser = DocumentParser()
_chunker = SemanticChunker()
_enricher = ContextualEnricher()


class IngestionPipeline:
    """
    Orchestrates the 6-step document processing assembly line:

      1. Parse   — Docling extracts text + structure from PDF/HTML/TXT
      2. Chunk   — RecursiveCharacterTextSplitter splits into 400-char chunks
      3. Enrich  — Groq generates 80-100 token context blurb per chunk (Contextual Retrieval)
      4. Embed   — BGE-M3 encodes full_text → 1024-dim vectors
      5. Qdrant  — upsert vectors + payload for ANN search
      6. BM25    — rebuild keyword index over all chunks
      7. Postgres — persist chunk records; mark document as ready

    Steps 1-4 are CPU/IO bound and run sequentially (Groq calls are async and
    rate-limited by the enricher's semaphore). Step 7 commits the transaction.

    If any step fails, the document is marked "failed" in Postgres and the
    exception propagates to the background worker for logging.
    """

    async def run(
        self, file_path: Path, doc_id: UUID, file_type: str, db: AsyncSession
    ) -> dict:
        logger.info("ingestion_start", doc_id=str(doc_id), file=str(file_path))

        doc = await document_repo.get_document(db, doc_id)
        filename = doc.filename if doc else file_path.name

        await document_repo.update_document_status(db, doc_id, "processing")
        await db.commit()

        try:
            # ── Step 1: Parse ────────────────────────────────────────────────
            logger.info("ingestion_step", step="parse", doc_id=str(doc_id))
            parsed = await _parser.parse(file_path, file_type)
            logger.info(
                "parse_done",
                doc_id=str(doc_id),
                chars=len(parsed.text),
                pages=parsed.page_count,
            )

            # ── Step 2: Chunk ────────────────────────────────────────────────
            logger.info("ingestion_step", step="chunk", doc_id=str(doc_id))
            chunks = _chunker.chunk_document(parsed, doc_id)
            logger.info("chunk_done", doc_id=str(doc_id), chunk_count=len(chunks))

            # ── Step 3: Enrich ───────────────────────────────────────────────
            logger.info("ingestion_step", step="enrich", doc_id=str(doc_id))
            chunks = await _enricher.enrich_chunks(parsed.text, chunks)

            # ── Step 4: Embed ────────────────────────────────────────────────
            logger.info("ingestion_step", step="embed", doc_id=str(doc_id))
            embedder = BGEEmbedder.get()
            full_texts = [c.full_text for c in chunks]
            embeddings = await embedder.embed_chunks_async(full_texts)
            logger.info("embed_done", doc_id=str(doc_id), vector_count=len(embeddings))

            # ── Step 5: Upsert to Qdrant ─────────────────────────────────────
            logger.info("ingestion_step", step="qdrant_upsert", doc_id=str(doc_id))
            qdrant = get_qdrant_client()
            # We use sequential UUIDs from the chunk index as Qdrant point IDs.
            # Storing them in chunks.qdrant_id lets us delete specific points
            # when a document is removed without a full collection scan.
            from uuid import uuid4
            qdrant_ids = [uuid4() for _ in chunks]

            points = [
                PointStruct(
                    id=str(qid),
                    vector=embedding,
                    payload={
                        "chunk_index": chunk.chunk_index,
                        "doc_id": str(chunk.doc_id),
                        "filename": filename,
                        "raw_text": chunk.raw_text,
                        "full_text": chunk.full_text,
                        "page_num": chunk.page_num,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                    },
                )
                for chunk, embedding, qid in zip(chunks, embeddings, qdrant_ids)
            ]
            # Batch upsert to stay under Qdrant's gRPC/HTTP payload size limit.
            # A single call with 1800+ large-payload points exceeds the default
            # 4 MB message limit and fails with an empty exception string.
            _BATCH = 200
            for i in range(0, len(points), _BATCH):
                await qdrant.upsert(
                    collection_name=settings.qdrant_collection,
                    points=points[i : i + _BATCH],
                    wait=True,
                )
            logger.info("qdrant_upsert_done", doc_id=str(doc_id), points=len(points))

            # ── Step 6: Rebuild BM25 index ───────────────────────────────────
            logger.info("ingestion_step", step="bm25_rebuild", doc_id=str(doc_id))
            all_existing = await chunk_repo.get_all_chunks(db)
            # Include newly ingested chunks (not yet committed) by merging them in
            new_chunk_dicts = [
                {
                    "id": qid,
                    "full_text": c.full_text,
                }
                for c, qid in zip(chunks, qdrant_ids)
            ]
            existing_chunk_dicts = [
                {"id": c.qdrant_id or c.id, "full_text": c.full_text}
                for c in all_existing
            ]
            BM25Index.get().build(existing_chunk_dicts + new_chunk_dicts)

            # ── Step 7: Persist to Postgres ──────────────────────────────────
            logger.info("ingestion_step", step="postgres_persist", doc_id=str(doc_id))
            chunk_dicts = [
                {
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "raw_text": c.raw_text,
                    "context": c.context,
                    "full_text": c.full_text,
                    "page_num": c.page_num,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "qdrant_id": qid,
                }
                for c, qid in zip(chunks, qdrant_ids)
            ]
            await chunk_repo.create_chunks(db, chunk_dicts)
            await document_repo.update_document_status(
                db, doc_id, "ready", chunk_count=len(chunks)
            )
            await db.commit()

            logger.info(
                "ingestion_complete",
                doc_id=str(doc_id),
                chunk_count=len(chunks),
            )
            return {"status": "ready", "chunk_count": len(chunks)}

        except Exception as exc:
            await db.rollback()
            await document_repo.update_document_status(
                db, doc_id, "failed", error_msg=str(exc)
            )
            await db.commit()
            logger.error("ingestion_failed", doc_id=str(doc_id), error=str(exc))
            raise
