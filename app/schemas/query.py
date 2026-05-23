from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = None  # scope retrieval to specific docs


class ScoredChunk(BaseModel):
    """
    A retrieved chunk with its retrieval score. Flows through the entire
    query pipeline: dense → sparse → RRF fusion → reranker → generator.
    Keeping it as a Pydantic model (rather than a plain dict) means every
    pipeline step gets type-checked fields — no KeyError surprises at runtime.
    """
    chunk_id: str
    doc_id: str
    filename: str
    raw_text: str
    full_text: str
    page_num: int | None
    char_start: int | None
    char_end: int | None
    score: float


class Citation(BaseModel):
    """A single source cited in the generated answer."""
    source_id: int        # [Source N] number used inline in the answer text
    chunk_id: str
    filename: str
    page_num: int | None
    cited_text: str       # the verbatim chunk passage the model cited
    score: float


class QueryResponse(BaseModel):
    """Full response from POST /query."""
    query_id: str
    answer: str
    citations: list[Citation]
    retrieval_method: str   # e.g. "hybrid_rrf_rerank"
    latency_ms: int
    model_used: str
    from_cache: bool
    trace_url: str | None   # Langfuse trace link for this query
