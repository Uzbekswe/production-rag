from langgraph.graph import END, START, StateGraph

from app.services.agent.nodes import (
    generate_node,
    hybrid_retriever_node,
    query_rewriter_node,
    reranker_node,
    sufficiency_checker_node,
)
from app.services.agent.state import RAGState


def _should_retry(state: RAGState) -> str:
    """
    Conditional edge after sufficiency_checker.
    Retry (loop back to query_rewriter) only if:
      - chunks were insufficient, AND
      - we haven't exceeded max_agent_retries
    Otherwise proceed to generation.
    """
    from app.core.config import settings
    if not state["is_sufficient"] and state["retrieval_attempt"] < settings.max_agent_retries:
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
    g.add_conditional_edges(
        "sufficiency_checker",
        _should_retry,
        {"query_rewriter": "query_rewriter", "generate": "generate"},
    )
    g.add_edge("generate", END)

    return g


rag_graph = build_rag_graph().compile()
