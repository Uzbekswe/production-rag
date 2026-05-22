from typing import TypedDict


class RAGState(TypedDict):
    query: str
    rewritten_queries: list[str]
    retrieved_chunks: list[dict]
    retrieval_attempt: int
    is_sufficient: bool
    answer: str
    citations: list[dict]
    trace_id: str
    from_cache: bool
    latency_ms: int
    model_used: str
