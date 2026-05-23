from prometheus_client import Counter, Histogram

rag_queries_total = Counter(
    "rag_queries_total",
    "Total RAG queries handled",
    ["from_cache"],
)

rag_query_latency_seconds = Histogram(
    "rag_query_latency_seconds",
    "End-to-end RAG query latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

rag_chunks_retrieved = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks after reranking passed to generation",
    buckets=[0, 1, 2, 3, 4, 5],
)
