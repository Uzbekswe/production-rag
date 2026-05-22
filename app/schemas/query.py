from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = None  # scope to specific docs


class Citation(BaseModel):
    chunk_id: str
    filename: str
    page_num: int | None
    cited_text: str
    score: float


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[Citation]
    retrieval_method: str
    latency_ms: int
    model_used: str
    from_cache: bool
    trace_url: str | None
