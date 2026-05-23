from contextlib import contextmanager
from typing import Any, Generator

from langfuse import Langfuse
from langfuse.client import StatefulSpanClient, StatefulTraceClient

from app.core.config import settings

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return _langfuse


def create_trace(
    name: str,
    trace_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    input: Any = None,
) -> StatefulTraceClient:
    """
    Create a root Langfuse trace for one user query.

    Pass trace_id=query_id so the trace's ID equals the query_id stored in
    RAGState. Nodes then call get_langfuse().trace(id=state["trace_id"]) to
    attach child spans to this exact trace.
    """
    return get_langfuse().trace(
        id=trace_id,
        name=name,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
        input=input,
    )


@contextmanager
def span(
    trace: StatefulTraceClient | StatefulSpanClient,
    name: str,
    metadata: dict[str, Any] | None = None,
    input: Any = None,
) -> Generator[StatefulSpanClient, None, None]:
    """
    Context manager that wraps a pipeline step in a Langfuse span.

    Usage:
        trace = create_trace("rag_query", input={"query": q})
        with span(trace, "dense_search", metadata={"top_k": 50}) as s:
            results = await qdrant.search(...)
            s.update(output={"result_count": len(results)})

    The span automatically ends when the context manager exits, even on exception.
    """
    s = trace.span(name=name, metadata=metadata, input=input)
    try:
        yield s
    finally:
        s.end()


def flush() -> None:
    """Flush all pending Langfuse events. Call during shutdown."""
    get_langfuse().flush()
