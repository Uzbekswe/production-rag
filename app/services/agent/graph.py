from langgraph.graph import END, START, StateGraph

from app.services.agent.state import RAGState

# Node stubs — each will be implemented in its own module (Phase 2+)


async def query_rewriter_node(state: RAGState) -> RAGState:
    """Expand or rewrite query; increment attempt counter."""
    raise NotImplementedError


async def hybrid_retriever_node(state: RAGState) -> RAGState:
    """Run BM25 + BGE-M3 in parallel, fuse via RRF."""
    raise NotImplementedError


async def reranker_node(state: RAGState) -> RAGState:
    """Cross-encoder rerank top-50 → top-5."""
    raise NotImplementedError


async def sufficiency_checker_node(state: RAGState) -> RAGState:
    """LLM judge: are retrieved chunks sufficient to answer?"""
    raise NotImplementedError


async def generate_node(state: RAGState) -> RAGState:
    """Call Claude Citations API and stream answer."""
    raise NotImplementedError


def _should_retry(state: RAGState) -> str:
    if not state["is_sufficient"] and state["retrieval_attempt"] < 2:
        return "query_rewriter"
    return "generate"


def build_rag_graph() -> StateGraph:
    g = StateGraph(RAGState)

    g.add_node("query_rewriter", query_rewriter_node)
    g.add_node("hybrid_retriever", hybrid_retriever_node)
    g.add_node("reranker", reranker_node)
    g.add_node("sufficiency_checker", sufficiency_checker_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "query_rewriter")
    g.add_edge("query_rewriter", "hybrid_retriever")
    g.add_edge("hybrid_retriever", "reranker")
    g.add_edge("reranker", "sufficiency_checker")
    g.add_conditional_edges("sufficiency_checker", _should_retry, {
        "query_rewriter": "query_rewriter",
        "generate": "generate",
    })
    g.add_edge("generate", END)

    return g


rag_graph = build_rag_graph().compile()
