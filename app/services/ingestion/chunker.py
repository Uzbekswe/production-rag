from dataclasses import dataclass, field
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.ingestion.parser import ParsedDocument


@dataclass
class ChunkData:
    doc_id: UUID
    chunk_index: int
    raw_text: str
    page_num: int | None
    char_start: int
    char_end: int
    # Filled in by ContextualEnricher after chunking
    context: str = field(default="")
    full_text: str = field(default="")


class SemanticChunker:
    """
    Splits a ParsedDocument into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter.

    The splitter tries delimiters in order: paragraph break → line break →
    sentence end → word boundary. This means it always cuts at the most
    natural semantic boundary it can find within the size limit, rather
    than slicing mid-sentence.

    chunk_size=400, overlap=64 keeps chunks small enough for BGE-M3's
    512-token limit while the 64-char overlap ensures a sentence split
    between two chunks doesn't lose a key phrase from both.
    """

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            add_start_index=True,  # populates chunk.metadata["start_index"]
        )

    def chunk_document(self, parsed: ParsedDocument, doc_id: UUID) -> list[ChunkData]:
        lc_chunks = self._splitter.create_documents([parsed.text])

        chunks: list[ChunkData] = []
        for i, lc_chunk in enumerate(lc_chunks):
            char_start: int = lc_chunk.metadata.get("start_index", 0)
            char_end: int = char_start + len(lc_chunk.page_content)
            page_num = self._estimate_page(char_start, len(parsed.text), parsed.page_count)

            chunks.append(
                ChunkData(
                    doc_id=doc_id,
                    chunk_index=i,
                    raw_text=lc_chunk.page_content,
                    page_num=page_num,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        return chunks

    @staticmethod
    def _estimate_page(char_start: int, total_chars: int, page_count: int) -> int | None:
        if total_chars == 0 or page_count <= 1:
            return 1 if page_count >= 1 else None
        ratio = char_start / total_chars
        return min(int(ratio * page_count) + 1, page_count)
