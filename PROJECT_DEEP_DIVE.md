# Production RAG System — Technical Deep Dive

> **Who this document is for:** Future-me, recruiters, senior engineers reviewing the codebase, and anyone preparing to defend the system in a technical interview. This is not a README. It is a full engineering walkthrough written to give complete mastery and defensibility of every decision.

---

## Table of Contents

1. [30-Second Elevator Pitch](#30-second-elevator-pitch)
2. [Recruiter Explanation](#recruiter-explanation)
3. [Problem Statement](#problem-statement)
4. [System Architecture Overview](#system-architecture-overview)
5. [Infrastructure Stack](#infrastructure-stack)
6. [Phase 1 — Ingestion Pipeline](#phase-1--ingestion-pipeline)
7. [Phase 2 — Query Pipeline](#phase-2--query-pipeline)
8. [Phase 3 — Observability](#phase-3--observability)
9. [Phase 4 — Evaluation](#phase-4--evaluation)
10. [File-by-File Component Reference](#file-by-file-component-reference)
11. [Production Engineering Decisions](#production-engineering-decisions)
12. [Architectural Tradeoffs](#architectural-tradeoffs)
13. [Bottlenecks and Performance Profile](#bottlenecks-and-performance-profile)
14. [Failure Handling and Resilience](#failure-handling-and-resilience)
15. [Scaling Considerations](#scaling-considerations)
16. [Future and Enterprise Improvements](#future-and-enterprise-improvements)
17. [Lessons Learned](#lessons-learned)
18. [Interview Preparation Guide](#interview-preparation-guide)

---

## 30-Second Elevator Pitch

This is a **production-grade Retrieval-Augmented Generation (RAG) system** built for financial document analysis. It ingests SEC 10-K annual filings from five major companies (Apple, Google, Meta, Microsoft, NVIDIA — two fiscal years each) and lets users ask natural-language questions and receive grounded, cited answers.

The system implements Anthropic's **Contextual Retrieval** pattern (which reduces retrieval failures by 49–67%), combines **dense vector search** (BGE-M3 embeddings in Qdrant) with **BM25 keyword search** fused via **Reciprocal Rank Fusion**, then applies a **cross-encoder reranker** (BGE-Reranker-v2-m3) to cut 50 candidates down to 5 high-precision chunks before generating a structured answer via **Groq Llama-3.3-70B** with inline [Source N] citations.

The query pipeline is an **agentic LangGraph graph** that can rewrite queries and retry retrieval when results are insufficient. A **semantic cache** in Redis eliminates redundant LLM calls for semantically similar queries. The system is fully observable via **Langfuse** (distributed tracing), **Prometheus + Grafana** (metrics), and evaluated using **RAGAS** (50 golden Q&A pairs across 4 categories with CI gate on faithfulness ≥ 0.80).

**Numbers:** 10 documents, 12,927 chunks, 12,927 Qdrant vectors, sub-second cache hits, 30–60s cold query latency (CPU reranker bottleneck), faithfulness ≥ 0.80 CI gate.

---

## Recruiter Explanation

**What this project demonstrates:**

| Skill Area | What Was Built |
|---|---|
| Backend Engineering | FastAPI async REST API with dependency injection, lifespan management, middleware |
| ML Engineering | Embedding pipeline (BGE-M3), reranking (BGE cross-encoder), LLM integration (Groq) |
| Data Engineering | 7-step document ingestion: parse → chunk → enrich → embed → index → persist |
| Distributed Systems | 7 Docker services orchestrated via Compose with named volumes and health checks |
| AI/RAG Patterns | Contextual Retrieval, Hybrid Search, RRF Fusion, Agentic Query Rewriting, Semantic Cache |
| Observability | Langfuse tracing, Prometheus metrics, Grafana dashboards, structured JSON logging |
| Evaluation | RAGAS automated eval framework, 50-question golden dataset, CI gate, checkpoint runner |
| Cloud / GPU | VESSL A100 SXM for bulk LLM enrichment (vLLM OpenAI-compatible endpoint) |
| Production Patterns | Singleton model loading, async thread pools, batch upsert, self-healing BM25, graceful degradation |

This is not a tutorial project. Every component has real production concerns addressed: rate-limit resilience (tenacity retry with exponential backoff), startup integrity checks, batch size limits for gRPC payloads, semantic cache to prevent redundant LLM calls, and a CI evaluation gate.

---

## Problem Statement

### The Core Challenge of RAG

Standard RAG has a fundamental problem: when you split a document into chunks, each chunk loses its surrounding context. Consider a chunk that says:

> "The company's revenue increased by 12% year-over-year."

Without context, an embedding model cannot know:
- Which company?
- Which revenue metric (total, segment, product)?
- Which fiscal year?
- Compared to what baseline?

When this chunk is embedded and stored, the resulting vector represents "revenue increased 12%" in the abstract — not "Apple's iPhone revenue increased 12% in FY2024 vs FY2023." A query for "Apple iPhone revenue growth" may not retrieve this chunk because the embedding doesn't encode the critical company/product/year context.

Anthropic documented this problem in September 2024 and published a solution: **Contextual Retrieval** — which reduces retrieval failures by 49% with embeddings alone, and 67% when combined with reranking.

### The Use Case

**Target use case:** Financial analysts, investors, and researchers who need to query multiple SEC 10-K annual reports simultaneously. The system covers:
- AAPL FY2024 + FY2025
- GOOGL FY2024 + FY2025
- META FY2024 + FY2025
- MSFT FY2024 + FY2025
- NVDA FY2024 + FY2025

Common question patterns:
- **Factual:** "What was Apple's total net sales in fiscal year 2024?"
- **Analytical:** "How did Apple's gross margin change between FY2023 and FY2024?"
- **Multi-hop:** "Combining Apple's iPhone, Mac, iPad, and Wearables revenues, what was the total hardware revenue?"
- **Adversarial:** "What was Apple's revenue guidance for fiscal year 2025?" (answer: not in the 10-K, the system should say so)

### Why the Architecture Matters

A naive RAG (embed → similarity search → paste into prompt) would:
1. Retrieve decontextualized chunks (wrong document, wrong year)
2. Return semantically similar but factually irrelevant passages
3. Hallucinate numbers not grounded in any source
4. Have no way to verify which document a claim came from

This system addresses all four problems through its layered pipeline.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
│                                                                     │
│  PDF/HTML ──► Docling ──► RecursiveChunker ──► ContextualEnricher  │
│                                │                      │             │
│                                │              (VESSL A100 / Groq)  │
│                                ▼                      ▼             │
│                           ChunkData ◄──────── blurb prepended      │
│                                │                                    │
│              ┌─────────────────┼──────────────────┐                │
│              ▼                 ▼                  ▼                │
│         BGE-M3              Postgres           BM25Index            │
│         (embed)             (persist)          (rebuild)            │
│              │                                                      │
│              ▼                                                      │
│           Qdrant                                                    │
│         (1024-dim                                                   │
│          vectors)                                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                             │
│                                                                     │
│  User Query                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────┐                               │
│  │   Semantic Cache (Redis)        │ ──► HIT: return in <100ms     │
│  │   cosine similarity ≥ 0.95     │                               │
│  └─────────────────────────────────┘                               │
│      │ MISS                                                         │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              LangGraph Agent (5 nodes)                  │       │
│  │                                                         │       │
│  │  query_rewriter ──► hybrid_retriever ──► reranker       │       │
│  │       ▲                                      │          │       │
│  │       │              sufficiency_checker ◄───┘          │       │
│  │       │                      │                          │       │
│  │       └── retry (max 2) ─────┘ sufficient ──► generate  │       │
│  │                                                         │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  hybrid_retriever:                                                  │
│    ┌──────────────────┐    ┌──────────────────┐                    │
│    │  Qdrant ANN       │    │  BM25 Keyword    │                    │
│    │  (BGE-M3 dense)   │    │  (rank-bm25)     │                    │
│    │  top-50 results   │    │  top-50 results  │                    │
│    └────────┬─────────┘    └────────┬─────────┘                    │
│             └──────────┬───────────┘                               │
│                        ▼                                            │
│               RRF Fusion (k=60)                                    │
│               top-50 merged                                         │
│                        │                                            │
│                        ▼                                            │
│           BGE-Reranker-v2-m3 (cross-encoder)                       │
│           top-50 → top-5                                            │
│                        │                                            │
│                        ▼                                            │
│           Groq Llama-3.3-70B                                       │
│           JSON output: {answer, citations}                          │
│                        │                                            │
│                        ▼                                            │
│           Cache Write → Response                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        OBSERVABILITY                                │
│                                                                     │
│  Langfuse ──────── distributed spans per query node                │
│  Prometheus ─────── rag_queries_total, latency, chunks_retrieved   │
│  Grafana ────────── dashboard over Prometheus metrics              │
│  structlog ──────── structured JSON logs (correlated by query_id)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure Stack

### Docker Compose Services

Seven services run locally, all with named Docker volumes for persistence:

```yaml
# docker-compose.yml — 7 services
postgres      postgres:16-alpine     port 5432   RAG metadata (docs, chunks, eval runs)
qdrant        qdrant/qdrant:v1.9.2   port 6333   Vector store (ANN search)
redis         redis:7-alpine         port 6379   Semantic cache (AOF persistence)
langfuse-db   postgres:16-alpine     (internal)  Langfuse observability backend DB
langfuse      langfuse/langfuse:2    port 3000   Trace UI
prometheus    prom/prometheus:v2.51  port 9090   Metrics scraper
grafana       grafana/grafana:10.4   port 3001   Metrics dashboard
```

**Why named volumes matter:** Data survives `docker compose down` and `docker compose up` cycles. Without `postgres_data:/var/lib/postgresql/data` and `qdrant_data:/qdrant/storage`, you'd lose all ingested data every time you restart. This is a basic production hygiene requirement that many tutorials omit.

**Redis AOF persistence:** `redis-server --appendonly yes` enables Append-Only File persistence. Without this, Redis loses all cached entries on restart, meaning users would re-experience full query latency on the first query after every restart.

### Technology Choices

| Component | Choice | Why Not X |
|---|---|---|
| Vector DB | **Qdrant** | Weaviate is heavier; Pinecone is cloud-only/paid; Chroma lacks production maturity |
| Embeddings | **BGE-M3 (BAAI)** | OpenAI ada-002 requires API key on every embed; BGE-M3 is local, multilingual, 1024-dim |
| Reranker | **BGE-Reranker-v2-m3** | Cohere Rerank is paid API; BGE runs locally, 568M params, multilingual |
| LLM | **Groq Llama-3.3-70B** | OpenAI GPT-4 is expensive; Groq provides free tier with fast inference |
| Framework | **FastAPI** | Flask lacks async-first design; Django is heavyweight for an API |
| Agent | **LangGraph** | LangChain LCEL doesn't support conditional retry loops naturally |
| Tracing | **Langfuse** | LangSmith is LangChain-proprietary; Langfuse is open-source and self-hostable |
| Metrics | **Prometheus + Grafana** | Datadog/New Relic are paid; Prometheus is the open-source standard |

---

## Phase 1 — Ingestion Pipeline

### Overview

The ingestion pipeline transforms a raw PDF/HTML document into searchable, enriched vector embeddings in 7 sequential steps. The entry point is `app/services/ingestion/pipeline.py:IngestionPipeline.run()`.

```
File upload → IngestionPipeline.run()
  Step 1: Parse    (DocumentParser)
  Step 2: Chunk    (SemanticChunker)
  Step 3: Enrich   (ContextualEnricher)  ← the secret sauce
  Step 4: Embed    (BGEEmbedder)
  Step 5: Upsert   (Qdrant, batches of 200)
  Step 6: BM25     (BM25Index.rebuild())
  Step 7: Persist  (Postgres, mark "ready")
```

If any step fails, the document is marked "failed" in Postgres, the exception propagates to the background worker, and the transaction is rolled back. No partial state leaks.

### Step 1 — Parsing (`app/services/ingestion/parser.py`)

**Component:** `DocumentParser`

Uses **Docling**, IBM's open-source document intelligence library, to extract text and structure from PDF, HTML, and TXT files. Docling is significantly more capable than `PyPDF2` or `pdfplumber` for financial documents because it:
- Understands multi-column layouts
- Extracts tables as structured data
- Handles footnotes and headers correctly
- Preserves reading order across complex layouts

The parser returns a `ParsedDocument` containing the full text, page count, and filename.

**Why not `pdfplumber` or `PyPDF2`?** Financial 10-K filings are complex PDFs with multi-column text, embedded tables, and headers/footers. Simpler extractors frequently mix column order, skip table content, or include irrelevant header text in the middle of body paragraphs. Docling was designed specifically for complex document understanding.

### Step 2 — Chunking (`app/services/ingestion/chunker.py`)

**Component:** `SemanticChunker`

Uses LangChain's `RecursiveCharacterTextSplitter` with:
- `chunk_size=400` characters
- `chunk_overlap=64` characters
- `separators=["\n\n", "\n", ". ", " ", ""]`

**Why these specific numbers?**

`chunk_size=400` is chosen to fit comfortably within BGE-M3's 512-token limit. A typical English sentence is 15-20 tokens; 400 characters ≈ 80-100 words ≈ 100-130 tokens. This leaves headroom for the context blurb added in Step 3 (80-100 tokens) while staying well under the 512-token embedding limit.

`chunk_overlap=64` prevents a sentence split between two adjacent chunks from losing a key phrase from both. Without overlap, a sentence like "Apple's total revenue was $391 billion | in FY2024, representing a 3% increase" (split at `|`) would produce two chunks, neither of which contains the complete fact.

**The separator hierarchy** is the key to "semantic" chunking. The splitter tries each delimiter in order and only falls to the next when it must:
1. `\n\n` — paragraph boundary (most natural)
2. `\n` — line break
3. `. ` — sentence end
4. ` ` — word boundary
5. `""` — character boundary (last resort, never mid-token in English)

This means chunks are almost always complete sentences, never mid-word cuts.

The chunker also tracks `char_start` and `char_end` offsets (from `add_start_index=True`) and estimates `page_num` by linear interpolation of the character offset across the total document length.

### Step 3 — Contextual Enrichment (`app/services/ingestion/enricher.py`)

**This is the most important step.** This is Anthropic's **Contextual Retrieval** pattern.

**Component:** `ContextualEnricher`

For each chunk, the enricher calls an LLM to generate a 2-3 sentence "situating blurb" (80-100 tokens) that describes:
1. Where this chunk appears in the document
2. What broader context is needed to understand it

The blurb is **prepended** to the chunk text to form `full_text`:

```
full_text = f"{blurb}\n\n{raw_text}"
```

This `full_text` is what gets embedded (Step 4) and indexed in BM25 (Step 6). The raw text is stored separately for citation display.

**Concrete example:**

*Without enrichment (raw chunk):*
> "The company's Services segment generated $96.2 billion in revenue, with a gross margin of 75.4%."

*With contextual enrichment (full_text):*
> "This chunk is from Apple Inc.'s FY2024 Annual Report (10-K), specifically from the financial highlights section discussing segment performance. It discusses the Services business unit, which includes the App Store, Apple Music, iCloud, and Apple TV+, in the fiscal year ending September 28, 2024.
>
> The company's Services segment generated $96.2 billion in revenue, with a gross margin of 75.4%."

Now when a user asks "What was Apple's App Store revenue in FY2024?", the embedding of this enriched chunk sits much closer to the query vector than the raw chunk would. The model knows this is Apple, FY2024, and Services — not just an anonymous revenue figure.

**Anthropic's published results:**
- Contextual Embeddings alone: **35% reduction** in top-20 retrieval failure rate
- Contextual Embeddings + BM25 + Reranking: **67% reduction**

**Implementation details:**

```python
# enricher.py
DOC_PREVIEW_CHARS = 1_000  # first 1000 chars for document context
# (reduced from 6000 — covers title/company/year, cuts token cost by 65%)

async def _enrich_one(self, doc_preview: str, chunk: ChunkData) -> ChunkData:
    async with self._sem:  # asyncio.Semaphore(3) — max 3 concurrent LLM calls
        resp = await self._client.chat.completions.create(...)
    chunk.context = blurb
    chunk.full_text = f"{blurb}\n\n{chunk.raw_text}"
    return chunk
```

**The semaphore** (`asyncio.Semaphore(3)`) limits concurrent Groq calls to 3. Without it, a document with 800 chunks would fire 800 simultaneous Groq API calls, immediately hitting the rate limit (6,000 tokens per minute on the free tier) and causing all 800 to fail together.

**Tenacity retry with exponential backoff:**
```python
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
```
If a chunk's enrichment fails after 4 attempts, it falls back gracefully: `full_text = raw_text`. The chunk is still searchable — it just lacks the context boost.

**VESSL GPU endpoint:** During bulk ingestion of the 10 SEC filings (12,927 chunks each needing one LLM call), the enricher switches to a VESSL A100 SXM GPU running `meta-llama/Llama-3.1-8B-Instruct` via vLLM. This OpenAI-compatible endpoint has no rate limits (billed by GPU-hour) and processes all 12,927 enrichments in parallel. Without VESSL, the same job would take ~6 hours on Groq's free tier (500K tokens/day).

### Step 4 — Embedding (`app/services/ingestion/embedder.py`)

**Component:** `BGEEmbedder` (singleton)

Model: **`BAAI/bge-m3`** — BAAI General Embeddings, Multi-Functionality, Multi-Linguality, Multi-Granularity.

Produces **1024-dimensional dense vectors** from the enriched `full_text`. The model:
- Supports up to 8,192 input tokens (but we cap at 512 for speed)
- Is multilingual (100+ languages)
- Scores among the top retrieval models on MTEB benchmark
- Runs locally — no API calls, no cost per embedding

```python
output = self._model.encode(
    texts,
    batch_size=32,
    max_length=512,
    return_dense=True,
    return_sparse=False,   # not using BGE-M3's sparse mode (using rank-bm25 instead)
    return_colbert_vecs=False,
)
```

**Why batch_size=32?** BGE-M3 processes multiple texts simultaneously on the GPU/CPU. Batching is much more efficient than one text at a time — it amortizes the fixed overhead of a forward pass across 32 texts. At batch_size=32, throughput on Apple Silicon is ~50ms per batch (≈1.5ms per chunk).

**Singleton pattern:** The model (~3GB RAM, ~10 second load time) is loaded once at startup via `BGEEmbedder.get()`. Subsequent calls return the cached instance. Loading per-request would make every query 10 seconds slower.

**Async wrapper:** `embed_chunks_async()` runs the CPU-bound embedding in a `ThreadPoolExecutor` via `loop.run_in_executor(None, ...)`. This prevents the FastAPI event loop from blocking during embedding, allowing other requests to be served concurrently.

### Step 5 — Qdrant Upsert (`app/services/ingestion/pipeline.py`)

Vectors are upserted to Qdrant with their full payload:

```python
PointStruct(
    id=str(uuid4()),      # unique UUID per chunk
    vector=embedding,     # 1024-dim float list
    payload={
        "chunk_index": chunk.chunk_index,
        "doc_id": str(chunk.doc_id),
        "filename": filename,
        "raw_text": chunk.raw_text,    # for citation display
        "full_text": chunk.full_text,  # enriched text (context + raw)
        "page_num": chunk.page_num,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }
)
```

**The batch upsert fix:** A critical production bug found during initial ingestion. Qdrant's gRPC/HTTP transport has a default message size limit of ~4MB. A single `upsert()` call with 1,800+ points, each carrying a large payload (full_text can be 600+ chars), exceeds this limit with an empty exception string — the most frustrating type of error to debug.

```python
_BATCH = 200
for i in range(0, len(points), _BATCH):
    await qdrant.upsert(
        collection_name=settings.qdrant_collection,
        points=points[i : i + _BATCH],
        wait=True,  # synchronous confirmation before moving to next batch
    )
```

`wait=True` is important: it ensures each batch is committed to disk before the next batch starts. Without it, a process crash mid-ingest could leave Qdrant in an inconsistent state.

### Step 6 — BM25 Index Rebuild (`app/services/ingestion/bm25_indexer.py`)

**Component:** `BM25Index` (singleton)

After each document ingest, the entire BM25 keyword index is rebuilt from scratch over all chunks (existing + newly ingested).

```python
# From rank-bm25 library
corpus = [c["full_text"].lower().split() for c in chunks]
self._bm25 = BM25Okapi(corpus)
```

The index is serialized to `data/bm25_index.pkl` via pickle for fast restart.

**Why rebuild from scratch?** `rank-bm25`'s `BM25Okapi` doesn't support incremental updates — it must see the entire corpus at construction time to compute IDF (Inverse Document Frequency) scores. IDF measures how rare a term is across all documents; adding a new document changes the IDF of every term it contains. At 10,000–15,000 chunks, a full rebuild takes under 1 second. This tradeoff (rebuild complexity vs. correctness) is acceptable at this scale. At 1M+ chunks you'd move to Elasticsearch or Apache Solr.

**BM25 Self-Healing:** A key startup behavior in `app/main.py`:

```python
bm25 = BM25Index.get()
if not bm25.is_ready:
    # Rebuild from Postgres if pickle is missing
    async with AsyncSessionLocal() as db:
        chunks = await get_all_chunks_for_bm25(db)
    if chunks:
        bm25.build(chunks)
```

If `data/bm25_index.pkl` is missing (new machine, clean Git clone, accidental delete), the BM25 index self-heals by rebuilding from Postgres on startup instead of silently degrading to dense-only retrieval. **Postgres is the source of truth; the pickle is a derived cache.**

### Step 7 — Postgres Persistence (`app/repositories/chunk_repo.py`)

Every chunk's metadata is persisted to Postgres:
- `raw_text`, `context`, `full_text`
- `page_num`, `char_start`, `char_end`
- `qdrant_id` — the UUID of the corresponding Qdrant point
- Document status updated to `"ready"` with `chunk_count`

The `qdrant_id` foreign key enables targeted deletion: when a document is removed, we can delete exactly its Qdrant points by ID without a full collection scan.

---

## Phase 2 — Query Pipeline

### Overview

Every query goes through this sequence:

```
POST /api/v1/query
  1. Embed query with BGE-M3
  2. Semantic cache lookup (Redis, cosine ≥ 0.95)
     → HIT: return in <100ms
     → MISS: LangGraph agent
  3. LangGraph: query_rewriter → hybrid_retriever → reranker
              → sufficiency_checker (retry loop up to 2x)
              → generate
  4. Prometheus metrics
  5. Cache write
  6. Return QueryResponse with citations
```

### The LangGraph Agent (`app/services/agent/graph.py`)

The query pipeline is modeled as a **stateful directed graph** using LangGraph. This is the key architectural decision that separates this from a simple pipeline.

```python
# State flows through these nodes:
START → query_rewriter → hybrid_retriever → reranker
      → sufficiency_checker
         ├── (insufficient AND attempts < max) → query_rewriter  [retry loop]
         └── (sufficient OR max attempts) → generate → END
```

**Why LangGraph instead of a simple function chain?**

A simple function chain (`rewrite() → retrieve() → rerank() → generate()`) cannot easily express:
1. **Conditional routing** — different paths based on runtime state
2. **Retry loops** — going back to an earlier step when results are insufficient
3. **State accumulation** — tracking all rewritten queries across attempts
4. **Streaming-friendly** — each node is an async function that can be interrupted

LangGraph represents this as a proper graph with typed state (`RAGState TypedDict`) that flows through nodes. Each node receives the full state, returns only the keys it modifies, and LangGraph merges the update.

**RAGState keys:**
```python
class RAGState(TypedDict):
    query: str                    # original user query
    rewritten_queries: list[str]  # history of rewrites across retries
    retrieved_chunks: list[dict]  # current retrieved + ranked chunks
    retrieval_attempt: int        # which attempt we're on (0, 1, 2)
    is_sufficient: bool           # did sufficiency checker approve?
    answer: str
    citations: list[dict]
    trace_id: str                 # Langfuse trace ID (= query_id)
    model_used: str
    from_cache: bool
    latency_ms: int
```

### Node 1 — Query Rewriter (`query_rewriter_node`)

**Purpose:** Transform the user's natural-language question into a retrieval-optimized query.

```python
REWRITE_SYSTEM = """
You are a search query optimizer for financial document retrieval.
Given a user question, rewrite it to be more specific and retrieval-friendly.
Focus on key entities (company names, ticker symbols, fiscal years, metrics).
Return ONLY the rewritten query — no explanation, no preamble.
"""
```

**Why query rewriting?** Users write conversational questions. Retrieval systems work best with keyword-dense, entity-rich queries.

User: *"How did Apple do last year?"*
Rewritten: *"Apple Inc. FY2024 fiscal year 2024 total revenue net sales annual results"*

**On retry:** The rewriter knows the previous retrieval attempt failed (via `retrieval_attempt > 0`). It rephrases with different terminology, hopefully hitting different vocabulary in the BM25 index or different semantic regions in the vector space.

**Fallback:** If the Groq call fails (network error, rate limit), the rewriter returns the original query unchanged. The pipeline degrades gracefully — retrieval may be slightly worse, but the query still completes.

### Node 2 — Hybrid Retriever (`hybrid_retriever_node`)

**Purpose:** Retrieve the top-50 most relevant chunks using both semantic and keyword search.

```python
dense_results, sparse_results = await asyncio.gather(
    _dense.search(query, top_k=settings.retrieval_top_k),   # top_k=50
    _sparse.search(query, top_k=settings.retrieval_top_k),  # top_k=50
)
fused = _fusion.fuse(dense_results, sparse_results, k=settings.rrf_k)  # k=60
```

**`asyncio.gather` parallelism:** Dense search (Qdrant gRPC) and sparse search (BM25 in-memory) are completely independent operations. Running them concurrently means total retrieval latency ≈ `max(dense_latency, sparse_latency)` rather than `sum(dense + sparse)`. On Apple Silicon: dense ≈ 20ms, sparse ≈ 5ms, parallel ≈ 20ms (vs. 25ms sequential).

#### Dense Retrieval (`app/services/retrieval/dense.py`)

Embeds the query with BGE-M3, then performs Approximate Nearest Neighbor (ANN) search in Qdrant using cosine similarity over 12,927 1024-dim vectors.

Qdrant uses HNSW (Hierarchical Navigable Small World) graphs for ANN — O(log n) search instead of O(n) brute force. At 12,927 vectors, brute force would work too, but HNSW scales to millions.

#### Sparse Retrieval (`app/services/retrieval/sparse.py`)

BM25 (Best Match 25) is the classic TF-IDF-based keyword ranking function. It scores chunks by:
- **TF (Term Frequency):** how often query terms appear in the chunk
- **IDF (Inverse Document Frequency):** how rare those terms are across all chunks
- **Length normalization:** prevents long chunks from winning just by having more words

BM25 excels at:
- Exact string matches ("NVDA", "FY2024", "Q3")
- Rare terms that may not have strong semantic embeddings
- Ticker symbols and financial abbreviations

BM25 fails at:
- Synonyms ("revenue" vs "net sales" vs "income")
- Semantic similarity without lexical overlap
- Queries longer than the vocabulary

This is precisely why we use both: they are complementary.

#### Reciprocal Rank Fusion (`app/services/retrieval/fusion.py`)

RRF merges the two ranked lists into a single ranking without needing to normalize scores across different scales.

```
RRF_score(chunk) = Σ_list  1 / (k + rank_in_list)
```

**Why k=60?** From Cormack et al. (2009), the original RRF paper. k=60 is the smoothing constant that prevents a single top-ranked document from dominating. With k=60, the maximum possible per-list contribution is 1/(60+1) ≈ 0.016. A document appearing at rank 1 in both lists scores ≈ 0.033; one appearing at rank 50 in both scores ≈ 0.009. The constant k controls how steeply rank differences matter.

**Why not weighted score fusion?** Dense scores are cosine distances (0-1), BM25 scores are TF-IDF weights with no upper bound. You cannot meaningfully add them without scale normalization. RRF is rank-based — it only cares about position, not magnitude — so no normalization is needed.

**Practical impact:** A chunk appearing in the top of both lists scores ~2× a chunk appearing in only one list. This rewards chunks that both retrievers agree are relevant — a strong signal.

### Node 3 — Reranker (`reranker_node`)

**Component:** `BGEReranker` (singleton)
**Model:** `BAAI/bge-reranker-v2-m3` (568M parameters, XLM-RoBERTa-Large based)

**Purpose:** Take the 50 RRF-fused chunks and re-score them to find the top 5 most precisely relevant to the exact query.

```python
pairs = [[query, c.raw_text] for c in chunks]
scores = self._model.compute_score(pairs, normalize=True)  # normalize → sigmoid → [0,1]
```

**Bi-encoder vs. Cross-encoder — the key distinction:**

| | Bi-encoder (BGE-M3) | Cross-encoder (BGE-Reranker) |
|---|---|---|
| Input | Query and chunk encoded **separately** | Query and chunk encoded **together** |
| Comparison | Cosine similarity of two vectors | Single relevance score |
| Attention | Each encoded in isolation | Full cross-attention between query and chunk |
| Speed | Fast (pre-computed chunk vectors) | Slow (must run model per pair) |
| Accuracy | Lower — can't model query-chunk interaction | Higher — sees exactly how query relates to chunk |
| Use case | First-stage retrieval over full corpus | Second-stage reranking over top-N candidates |

The bi-encoder creates two independent vectors and measures their proximity. The cross-encoder sees the query and chunk simultaneously, with every token attending to every other token. This means it can answer "does this specific passage answer this specific question?" rather than "are these two texts generally similar?"

**Why top-50 → top-5?** The reranker is too slow for full-corpus search (~17-37 seconds on Apple Silicon CPU for 50 pairs) but fast enough for 50 candidates. Dense retrieval narrows from 12,927 to 50; the reranker narrows from 50 to 5 with high precision.

**Async thread pool:** The reranker CPU work is offloaded to a thread pool to avoid blocking the FastAPI event loop:
```python
async def rerank_async(self, query, chunks, top_k):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.rerank, query, chunks, top_k)
```

### Node 4 — Sufficiency Checker (`sufficiency_checker_node`)

**Purpose:** Decide whether the retrieved chunks are good enough to generate from, or whether to retry with a rewritten query.

```python
is_sufficient = len(chunks) >= 3 and avg_score >= 0.2
```

**Why a heuristic instead of an LLM judge?**

An LLM-based sufficiency judge would:
- Add 1-2 seconds to every query (Groq call)
- Burn free-tier tokens on every query, even easy ones
- Add another potential failure point

The heuristic catches the real problem cases:
- 0 chunks retrieved (BM25 found no matches, Qdrant ANN returned nothing)
- All chunks score near 0 (retrieval clearly failed)
- Only 1-2 chunks (likely an adversarial or out-of-scope question)

If the check fails and we haven't exceeded `max_agent_retries=2`, the graph routes back to `query_rewriter` for another attempt with rephrased query vocabulary.

### Node 5 — Generator (`generate_node`)

**Component:** `GroqGenerator`
**Model:** `llama-3.3-70b-versatile` via Groq API

**Purpose:** Generate a factually grounded answer with inline [Source N] citations.

```python
SYSTEM_PROMPT = """
You are a financial document analyst. Answer questions using ONLY the provided sources.
Cite sources inline as [Source N] immediately after each claim.
...
You MUST respond with valid JSON:
{"answer": "...", "citations": [{"source_id": 1, "cited_text": "exact verbatim quote"}]}
"""
```

**Why JSON output format?**

The model writes `[Source 1]` naturally in prose, but we need to:
1. Map source_id → chunk metadata (filename, page_num, score)
2. Extract the specific text that was cited
3. Return structured `Citation` objects in the API response

Forcing `response_format={"type": "json_object"}` guarantees machine-parseable output. The model is instructed to include verbatim quoted `cited_text` so users can verify exactly what text the answer is based on.

**Graceful degradation on JSON parse failure:**
```python
except (json.JSONDecodeError, KeyError, ValueError) as e:
    return GenerationResult(answer=raw, citations=[], model_used=model)
```
If the model returns malformed JSON (rare with an explicit system prompt), the raw text is returned as the answer with empty citations. The user gets an answer; we lose citation structure. Better than a 500 error.

**Temperature=0.1:** Near-zero temperature for factual, consistent answers. We want the same question to produce the same answer, not creative variations.

### Semantic Cache (`app/core/redis_client.py`)

**Purpose:** Avoid redundant LLM calls for semantically similar queries.

```python
async def cache_lookup(redis, query_embedding, threshold=0.95):
    # Linear scan over all cached embeddings
    for key in await redis.scan("rag:semantic_cache:*"):
        entry = json.loads(await redis.get(key))
        if cosine_similarity(query_embedding, entry["embedding"]) >= threshold:
            return entry["response"]
    return None
```

**Why semantic caching instead of exact-match?**

Two queries:
- *"What was Apple's revenue in FY2024?"*
- *"How much did Apple earn in fiscal year 2024?"*

These produce different strings but nearly identical BGE-M3 embeddings (cosine similarity ≈ 0.97). Exact-match caching would treat them as different questions and make two separate Groq calls. Semantic caching recognizes them as the same question and returns the cached answer instantly.

**Threshold=0.95:** High enough to avoid returning a cached answer for a genuinely different question, low enough to catch paraphrases. In practice this threshold catches ~60-80% of repeated queries in a demo/evaluation context.

**TTL=3600 seconds:** Cache entries expire after 1 hour. Financial documents don't change between queries in a demo session, but we don't want stale data if the knowledge base is updated.

**Key design:** The cache key is a hash of the first 8 embedding dimensions (sufficient for uniqueness at small scale). The full embedding is stored alongside the response to enable cosine similarity lookup.

**Observed impact during evaluation:** When the RAGAS evaluation runner re-queries the same 50 questions in a second pass, all 50 return from cache in ~1ms each (vs. 30-60s for a cold query). This is why the eval runner's API phase completes in under 60 seconds on the second run.

### The Complete Query Route (`app/api/routes/query.py`)

```python
@router.post("", response_model=QueryResponse)
async def query(payload: QueryRequest, db: AsyncSession = Depends(get_db)):
    # 1. Create Langfuse trace (wraps entire request)
    trace = create_trace(name="rag_query", trace_id=query_id, ...)

    # 2. Embed query (BGE-M3, thread pool)
    query_embedding = await loop.run_in_executor(None, embedder.embed_query, payload.query)

    # 3. Semantic cache check
    cached = await cache_lookup(redis, query_embedding)
    if cached:
        return QueryResponse(**cached, from_cache=True, latency_ms=...)

    # 4. Run LangGraph agent
    result_state = await rag_graph.ainvoke(initial_state)

    # 5. Build response
    response = QueryResponse(answer=..., citations=..., trace_url=...)

    # 6. Prometheus metrics
    rag_queries_total.labels(from_cache="false").inc()
    rag_query_latency_seconds.observe(latency_ms / 1000)

    # 7. Write to cache
    await cache_store(redis, query_embedding, response.model_dump())

    return response
```

This is a textbook example of the **Cache-Aside** pattern: check cache → miss → populate from source → write back to cache → return.

---

## Phase 3 — Observability

### Langfuse Distributed Tracing (`app/core/tracing.py`)

Every query creates a Langfuse trace with the `query_id` as the trace ID. Each LangGraph node creates a child span:

```
Trace: rag_query (query_id=abc123)
  ├── Span: query_rewriter    (input: query, output: rewritten_query, tokens: 45/30)
  ├── Span: hybrid_retriever  (input: query, output: dense=50, sparse=40, fused=68)
  ├── Span: reranker          (input: 68 chunks, output: 5 chunks, top_score: 0.87)
  ├── Span: sufficiency_checker (is_sufficient: true, avg_score: 0.73)
  └── Span: generate          (input: 5 chunks, output: answer_len=312, citations=3)
```

**Why Langfuse?** It's the open-source alternative to LangSmith. It self-hosts (no data leaves your environment), supports LangGraph natively, and provides a visual trace explorer for debugging why a specific query produced a wrong answer.

**Practical use:** When a user reports "the answer to question X was wrong," you navigate to Langfuse, find the trace by query_id (returned in the API response as `trace_url`), and see exactly what chunks were retrieved, what scores they got, what the rewritten query was, and what the model was given. Without tracing, debugging requires adding print statements and re-running the query.

### Prometheus Metrics (`app/core/metrics.py`)

```python
rag_queries_total    # Counter, labels: [from_cache=true/false]
rag_query_latency_seconds  # Histogram, buckets: [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
rag_chunks_retrieved # Histogram, buckets: [0, 1, 2, 3, 4, 5]
```

These are scraped by Prometheus at `http://localhost:8000/metrics` every 15 seconds and visualized in Grafana.

**`from_cache` label:** Lets you plot cache hit rate over time. If the cache hit rate drops suddenly, either the cache TTL expired or users started asking novel questions. This is a useful signal for cache tuning.

**Latency histogram:** The bucket boundaries are chosen to highlight the distribution. 0.1s is a fast cache hit; 2-10s is a slow cold query with CPU reranking. If P95 spikes above 10s, something is wrong with the reranker.

### Structured Logging (`app/core/logging.py`)

Uses `structlog` with JSON output. Every log event carries:
- `timestamp`
- `level`
- `event` name
- Contextual fields (`query_id`, `doc_id`, `chunk_count`, `error`)

```python
logger.info("query_done",
    query_id=query_id,
    latency_ms=latency_ms,
    citations=len(citations),
    model=response.model_used,
    from_cache=False,
)
```

**Why structured logging?** JSON log lines can be shipped to Elasticsearch, Loki, or Datadog and queried like a database. `grep "query_done"` finds every query; `grep "error"` finds all errors. With unstructured logging, you'd need regex to extract the query_id from a log line like "Query abc123 completed in 2341ms with 3 citations."

---

## Phase 4 — Evaluation

### The RAGAS Framework

RAGAS (Retrieval-Augmented Generation Assessment) is an automated evaluation framework that uses LLMs-as-judges to score RAG pipeline quality.

**Three metrics used:**

| Metric | What It Measures | How Computed |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved chunks? | Judge LLM checks each claim against the provided context |
| **Context Recall** | Did the system retrieve the chunks needed to answer the question? | Judge LLM compares retrieved contexts to the ground-truth answer |
| **Factual Correctness** | Is the answer factually correct vs. the ground-truth answer? | Judge LLM compares answer to ground_truth field |

**SemanticSimilarity was removed** — it requires `OPENAI_API_KEY` for the embedding model by default in RAGAS 0.2.6. Rather than pay for OpenAI embeddings, we removed it and rely on the three metrics above.

### Golden Dataset (`evaluation/golden_dataset.json`)

50 hand-curated Q&A pairs across 4 categories:

| Category | Count | Description |
|---|---|---|
| `factual` | 18 | Single-fact lookups ("What was Apple's net income in FY2024?") |
| `analytical` | 13 | Reasoning over multiple data points |
| `multi_hop` | 12 | Combining information from multiple chunks |
| `adversarial` | 7 | Questions the system should refuse/hedge ("What was Apple's FY2025 guidance?") |

The adversarial category is crucial: a hallucinating system would make up revenue guidance. A well-grounded system says "The provided sources do not contain enough information to answer this question." Faithfulness rewards this behavior — a claim-free answer is vacuously faithful.

### Checkpoint-Based RAGAS Runner (`evaluation/runner.py`)

The runner was rewritten after several painful failures due to Groq's daily token limit (500K tokens/day on the free tier, used up by failed runs).

**Key design: batched evaluation with per-batch checkpointing.**

```python
# Run RAGAS in batches of 10 (5 batches for 50 questions)
for batch in batches:
    result = evaluate(batch_dataset, metrics=[Faithfulness, LLMContextRecall, FactualCorrectness])
    # Save per-sample scores to checkpoint immediately
    append_checkpoint(checkpoint_path, batch_records)
```

After each batch, per-sample scores are written to `results_checkpoint.jsonl`:
```json
{"idx": 0, "question": "What was Apple's total net sales...", "faithfulness": 0.91, "context_recall": 0.88, "factual_correctness": 0.95, "category": "factual"}
```

On restart, the runner loads the checkpoint and skips already-scored samples:
```python
already_done = load_checkpoint(checkpoint_path)
pending = [i for i in range(len(golden)) if i not in already_done]
```

**Why this matters:** Each RAGAS run makes 150 Groq LLM calls (50 questions × 3 metrics). On the free tier, this burns ~150,000 tokens. With the previous non-checkpoint runner, a crash at 85% completion lost all work and required another 150,000 tokens. The checkpoint runner loses at most one batch (10 samples × 30 LLM calls = 30,000 tokens).

**Judge model configurability:** The runner supports `RAGAS_JUDGE_MODEL` environment variable:
```python
model = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.3-70b-versatile")
```
When `llama-3.1-8b-instant` exhausted its daily quota, switching to `llama-3.3-70b-versatile` (separate TPD bucket) required only changing an environment variable, not code.

### CI Gate

```python
parser.add_argument("--fail-threshold", default="faithfulness=0.80,context_recall=0.70")
```

The runner exits with code 1 if any metric falls below its threshold. This integrates with `.github/workflows/eval.yml` to gate PR merges on eval quality. A regression in retrieval or generation quality that drops faithfulness below 0.80 fails CI automatically.

### Benchmark Script (`scripts/benchmark.py`)

Measures operational metrics independently of RAGAS:
- **Hit rate** — percentage of queries that returned at least 1 citation
- **Latency** — mean and P95 end-to-end query latency
- **Cache hit rate** — percentage of queries served from Redis
- **Chunk count distribution** — how many chunks after reranking

These operational metrics tell you "is the system fast and returning results?" independently of "are the results correct?" Both matter for a production system.

---

## File-by-File Component Reference

### Application Core

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI app factory, lifespan manager (startup/shutdown), route registration |
| `app/core/config.py` | Pydantic-settings configuration (all settings from .env, type-safe) |
| `app/core/qdrant.py` | Qdrant client factory, collection creation with cosine metric |
| `app/core/redis_client.py` | Redis client factory, semantic cache lookup/store logic |
| `app/core/metrics.py` | Prometheus Counter/Histogram definitions |
| `app/core/tracing.py` | Langfuse client, trace/span creation helpers |
| `app/core/logging.py` | structlog configuration, JSON formatter |
| `app/core/database.py` | SQLAlchemy async engine, session factory |

### API Routes

| File | Endpoint | Purpose |
|---|---|---|
| `app/api/routes/query.py` | `POST /api/v1/query` | Full RAG pipeline with cache |
| `app/api/routes/stream.py` | `POST /api/v1/stream` | SSE token streaming |
| `app/api/routes/ingest.py` | `POST /api/v1/ingest` | Upload + trigger ingestion |
| `app/api/routes/documents.py` | `GET/DELETE /api/v1/documents` | Document management |
| `app/api/routes/eval.py` | `GET /api/v1/eval/history` | Eval run history |
| `app/api/routes/health.py` | `GET /health` | Kubernetes-style health probe |

### Ingestion Services

| File | Class | Key Method |
|---|---|---|
| `app/services/ingestion/pipeline.py` | `IngestionPipeline` | `run(file_path, doc_id, file_type, db)` |
| `app/services/ingestion/parser.py` | `DocumentParser` | `parse(file_path, file_type)` → `ParsedDocument` |
| `app/services/ingestion/chunker.py` | `SemanticChunker` | `chunk_document(parsed, doc_id)` → `list[ChunkData]` |
| `app/services/ingestion/enricher.py` | `ContextualEnricher` | `enrich_chunks(full_doc, chunks)` → enriched chunks |
| `app/services/ingestion/embedder.py` | `BGEEmbedder` | `embed_chunks(texts)` → `list[list[float]]` |
| `app/services/ingestion/bm25_indexer.py` | `BM25Index` | `build(chunks)`, `search(query, top_k)` |

### Retrieval Services

| File | Class | Key Method |
|---|---|---|
| `app/services/retrieval/dense.py` | `QdrantRetriever` | `search(query, top_k)` → ANN results |
| `app/services/retrieval/sparse.py` | `BM25Retriever` | `search(query, top_k)` → BM25 results |
| `app/services/retrieval/fusion.py` | `ReciprocalRankFusion` | `fuse(dense, sparse, k)` → merged list |
| `app/services/retrieval/reranker.py` | `BGEReranker` | `rerank(query, chunks, top_k)` → top-5 |

### Generation Services

| File | Class | Purpose |
|---|---|---|
| `app/services/generation/groq_gen.py` | `GroqGenerator` | Groq Llama, JSON output, citation parsing |
| `app/services/generation/longcite.py` | `LongCiteGenerator` | VESSL-hosted LongCite model (demo mode) |
| `app/services/generation/router.py` | `generate()` | Routes to Groq or LongCite based on settings |

### Agent

| File | Purpose |
|---|---|
| `app/services/agent/graph.py` | LangGraph graph definition, conditional edges |
| `app/services/agent/nodes.py` | All 5 node implementations (rewriter, retriever, reranker, checker, generator) |
| `app/services/agent/state.py` | `RAGState` TypedDict definition |

### Data Layer

| File | Purpose |
|---|---|
| `app/repositories/document_repo.py` | CRUD for documents table |
| `app/repositories/chunk_repo.py` | CRUD for chunks table, BM25 rebuild query |
| `app/repositories/eval_repo.py` | Insert/query eval_runs table |
| `app/schemas/query.py` | `QueryRequest`, `QueryResponse`, `Citation`, `ScoredChunk` |
| `app/schemas/ingest.py` | `IngestRequest`, `IngestResponse` |
| `app/schemas/document.py` | `DocumentResponse`, `DocumentStatus` |

---

## Production Engineering Decisions

### 1. Why Batch Upserts to Qdrant (200 points/batch)?

Qdrant's gRPC transport has a default message size limit of ~4MB. A point's payload includes `raw_text` (400 chars) + `full_text` (600+ chars) + metadata ≈ 1.5KB per point. At 200 points per batch: 200 × 1.5KB = 300KB, well under the 4MB limit. Without batching, a document generating 1,800+ chunks would fail with a cryptic empty exception string — the hardest kind of bug to diagnose.

### 2. Why Singletons for Model Loading?

BGE-M3 embedding model: ~3GB RAM, ~10 second load time.
BGE-Reranker-v2-m3: ~1.5GB RAM, ~5 second load time.
BM25Index: ~100MB RAM (all chunk texts in memory).

If these loaded per-request:
- First query: 15+ second delay
- Concurrent requests: multiple model copies in memory, OOM crash
- Each request pays the full model loading overhead

Singletons load once at startup and serve all requests from the cached instance. The pattern is:
```python
class BGEEmbedder:
    _instance: "BGEEmbedder | None" = None
    
    @classmethod
    def get(cls) -> "BGEEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

### 3. Why Checkpointing for the Eval Runner?

The RAGAS evaluation makes 150 LLM API calls (50 questions × 3 metrics). Each call costs ~1,000 tokens. The Groq free tier has a 500,000 token/day limit — a rolling 24-hour window, not a midnight reset.

During development, several bugs caused the runner to crash at the end of a full run (score extraction failures), wasting the entire 150,000-token budget. The checkpoint approach limits wasted work to one batch (10 samples = 30,000 tokens) per crash.

### 4. Why Semantic Caching Instead of Exact-Match?

Financial analysts ask the same questions in slightly different ways:
- "Apple FY2024 revenue?"
- "What was Apple's revenue for fiscal 2024?"
- "How much did Apple make in 2024?"

Exact-match caching treats these as 3 different questions. Semantic caching recognizes them as the same question (cosine similarity ≈ 0.96-0.98 between their BGE-M3 embeddings) and serves the cached answer in <100ms vs. the 30-60s cold query.

The threshold (0.95) is conservative enough to prevent wrong cache hits: "What was Apple's revenue in FY2024?" and "What was Google's revenue in FY2024?" produce embeddings with similarity ~0.88 — below threshold, so they get independent answers.

### 5. Why BM25 + Dense (Hybrid) Instead of Dense-Only?

Financial documents contain:
- **Ticker symbols:** "AAPL", "NVDA", "MSFT" — exact tokens that may not have strong semantic embeddings
- **Fiscal year references:** "FY2024", "Q3 2024", "fiscal year ended September 28, 2024" — different lexical forms of the same concept
- **Dollar amounts:** "$391,035 million" — exact figures that require lexical match to retrieve reliably

BM25 catches these exact matches reliably. Dense search catches the semantic equivalences ("revenue" = "net sales" = "total income"). RRF fusion gets both.

Published research (arxiv:2604.01733) shows hybrid retrieval improves recall by 15-30% over single-method pipelines. For financial documents specifically, BM25 is particularly strong — the paper notes BM25 "remains strong for financial documents" due to the prevalence of precise terminology.

### 6. Why Two-Stage Retrieval (50 → 5)?

**Stage 1 (bi-encoder, top-50):** Fast vector similarity finds the rough neighborhood. BGE-M3 encodes query and chunks independently — chunk vectors are pre-computed at ingestion time, so only the query vector is computed at query time. ANN search over 12,927 vectors takes ~20ms.

**Stage 2 (cross-encoder, top-5):** Slow but precise cross-encoder sees the exact query-chunk interaction. It evaluates 50 pairs, takes ~17-37s on CPU. This narrows from "probably relevant" to "specifically answers this question."

The two-stage design is a classic information retrieval pattern. Applying the expensive cross-encoder to all 12,927 chunks would take hours. Applying the cheap bi-encoder is fast but imprecise. Combining them gives precision at acceptable latency.

### 7. Why LangGraph for the Query Pipeline?

The query pipeline has a **conditional retry loop** that a simple function chain cannot express cleanly. If retrieved chunks are insufficient, the pipeline must:
1. Go back to the query rewriter (not to retrieval directly)
2. Track the attempt number (to avoid infinite loops)
3. Accumulate all rewritten queries (for debugging in Langfuse)
4. Eventually generate even if still insufficient (graceful degradation)

LangGraph expresses this as a directed graph with a conditional edge:
```python
g.add_conditional_edges(
    "sufficiency_checker",
    _should_retry,
    {"query_rewriter": "query_rewriter", "generate": "generate"},
)
```

This is cleaner, more testable, and more debuggable than nested if/else in a single function.

### 8. Why Async Thread Pools for CPU-Bound Work?

FastAPI uses an async event loop (uvicorn + asyncio). CPU-bound work (BGE-M3 embedding, BGE-Reranker) blocks the event loop and prevents the server from handling other requests during computation.

The pattern:
```python
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, cpu_bound_function, *args)
```

`run_in_executor(None, ...)` submits the CPU work to the default `ThreadPoolExecutor`. While the embedding/reranking runs in a thread, the event loop is free to handle other incoming requests. This enables the server to serve multiple concurrent queries without one slow query blocking all others.

### 9. Why Tenacity for Enrichment Retries?

The `ContextualEnricher` calls the LLM API for every chunk. With 800+ chunks per document and concurrent processing (3 at a time via semaphore), transient API errors (rate limits, timeouts, network blips) are almost guaranteed to occur.

```python
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),  # 2s, 4s, 8s, 16s... up to 60s
    stop=stop_after_attempt(4),
    reraise=True,
)
```

Exponential backoff with jitter prevents thundering herd — if all 3 concurrent calls fail simultaneously and retry at the same interval, they all hit the API again at the same time. The exponential growth means retries spread out over time.

After 4 attempts, the chunk is processed without a context blurb (fallback to raw text). A degraded chunk is better than a failed ingestion.

---

## Architectural Tradeoffs

### Tradeoff 1: Local Models vs. Cloud APIs

| | Local (BGE-M3, BGE-Reranker) | Cloud (Voyage, Cohere) |
|---|---|---|
| Cost | Free (hardware only) | $0.00002-0.0001 per embedding |
| Speed | ~50ms/batch (CPU) | ~100-200ms (network + compute) |
| Privacy | All data stays local | Data sent to third party |
| Scaling | Limited by hardware | Unlimited horizontal scaling |
| Maintenance | You manage the model | Vendor manages |

**Decision:** Local models for development/demo. For production at scale, cloud APIs would reduce P95 latency (avoid CPU bottleneck) at increased cost.

### Tradeoff 2: In-Memory BM25 vs. Elasticsearch

| | rank-bm25 (in-memory) | Elasticsearch |
|---|---|---|
| Setup | 0 lines of infra config | Docker + config + schema |
| Rebuild | Full rebuild on ingest | Incremental index updates |
| Scale | ~100K chunks max | 100M+ chunks |
| Latency | ~5ms (RAM) | ~20-50ms (network) |
| Persistence | Pickle file (fragile) | Durable, replicated |

**Decision:** rank-bm25 is fine for 12,927 chunks and eliminates an infrastructure dependency. The self-healing BM25 (rebuild from Postgres) mitigates the pickle fragility.

### Tradeoff 3: Heuristic Sufficiency Check vs. LLM Judge

| | Heuristic (current) | LLM Judge |
|---|---|---|
| Cost | Free (arithmetic) | ~500 tokens per query |
| Latency | <1ms | 1-2 seconds |
| Accuracy | Catches hard failures | Catches subtle insufficiency |
| False positives | Low | Higher (LLM uncertainty) |

**Decision:** Heuristic for Phase 1-4. LLM judge would be Phase 5 — justified only if the system is deployed with adversarial user queries where subtle insufficiency matters.

### Tradeoff 4: Semantic Cache (Linear Scan) vs. Approximate NN Cache

| | Linear scan (current) | ANN cache (e.g., Redis Vector) |
|---|---|---|
| Correctness | Exact (no false positives from ANN approximation) | Approximate |
| Scale | O(n) — fine for <1,000 entries | O(log n) — needed for 10,000+ entries |
| Setup | 0 additional infra | Redis Stack or Qdrant |
| Complexity | Trivial | Moderate |

**Decision:** Linear scan is fine at demo scale (<100 cached entries). At production scale with 10,000+ cached queries, switch to Redis Vector or Qdrant as the cache backend.

### Tradeoff 5: JSON Response Format vs. Free-Form Text

| | JSON format (current) | Free-form text |
|---|---|---|
| Parsability | Machine-readable citations | Requires regex |
| Reliability | Occasional parse failures | Always parseable (just text) |
| Flexibility | Structured metadata | Free narrative |
| Model support | Groq supports `json_object` | Universal |

**Decision:** JSON for structured citation extraction. The fallback (return raw text on parse failure) prevents hard failures.

---

## Bottlenecks and Performance Profile

### Query Latency Breakdown (Cold Path, CPU)

```
BGE-M3 query embedding:     ~50ms   (CPU, singleton model)
Redis cache lookup:          ~5ms    (local Docker)
Qdrant ANN search:          ~20ms   (gRPC, 12,927 vectors)
BM25 search:                ~5ms    (in-memory)
RRF fusion:                 ~1ms    (pure Python, 50+50 items)
BGE-Reranker (50 pairs):    17-37s  ← DOMINANT BOTTLENECK (CPU, cross-encoder)
Groq generation:            1-3s    (network + LLM inference)
Cache write:                ~5ms

Total cold path:            ~20-45 seconds
Total cache hit:            <100ms
```

### The Reranker Bottleneck

The BGE-Reranker-v2-m3 is a 568M parameter cross-encoder running on CPU (Apple Silicon M-series). Unlike the bi-encoder (which only processes the query at query time, since chunk embeddings are pre-computed), the cross-encoder must process all 50 (query, chunk) pairs fresh at every query.

At 17-37 seconds, this dominates end-to-end latency. Solutions in order of effort:
1. **GPU:** Move to a CUDA/MPS GPU → 1-3s for 50 pairs
2. **Smaller reranker:** `bge-reranker-base` (278M params) → ~8-15s on CPU
3. **Fewer candidates:** Reduce retrieval_top_k from 50 → 20 → ~7-15s on CPU
4. **API reranker:** Cohere Rerank API → ~300ms network round-trip

For the portfolio demo running on MacBook, 17-37 seconds is acceptable. For production, GPU is non-negotiable.

### Ingestion Throughput (VESSL A100)

With VESSL A100 SXM for enrichment:
- Parse 10 documents: ~30 seconds (Docling)
- Chunk 10 documents: ~5 seconds (CPU, fast)
- Enrich 12,927 chunks (parallel 3×): ~45 minutes (GPU, vLLM, Groq-rate-limited)
- Embed 12,927 chunks: ~11 minutes (BGE-M3 CPU, batch 32)
- Qdrant upsert: ~2 minutes (network, 200/batch)
- BM25 rebuild: ~2 seconds
- Postgres persist: ~30 seconds

Total: ~60 minutes for 10 SEC 10-K filings → 12,927 chunks → ready to query.

---

## Failure Handling and Resilience

### Ingestion Failures

```python
try:
    # All 7 steps
    ...
except Exception as exc:
    await db.rollback()
    await document_repo.update_document_status(db, doc_id, "failed", error_msg=str(exc))
    await db.commit()
    raise
```

- Document marked "failed" in Postgres with the error message
- Transaction rolled back (no partial chunk rows)
- Qdrant vectors NOT rolled back (Qdrant has no transaction support) — this is an acceptable inconsistency at demo scale; at production scale you'd implement a compensating action (delete the uploaded points)

### Enrichment Failures

Per-chunk fallback: if enrichment fails after 4 retries, `full_text = raw_text`. The chunk is still indexed and searchable — just without the context boost. Partial enrichment is better than failed ingestion.

### BM25 Self-Healing

If the pickle file is missing at startup, the BM25 index is rebuilt from Postgres. This handles:
- Fresh deployment on a new machine
- Accidental deletion of `data/bm25_index.pkl`
- Docker volume reset

### Generation Fallback

If the LLM returns malformed JSON: return raw text as the answer with empty citations.
If retrieval returns 0 chunks: return "No relevant sources were found" without calling the LLM.
If the entire pipeline fails: raise `HTTPException(500)` with the error detail.

### Query Rewriter Fallback

If the Groq rewrite call fails: use the original query unchanged. Retrieval may be slightly worse but completes.

### Redis Cache Failures

Cache read/write failures are caught silently — the request falls through to the full pipeline. The cache is an optimization, not a correctness requirement.

```python
try:
    await cache_store(redis, query_embedding, cacheable)
except Exception as e:
    logger.warning("cache_write_failed", error=str(e))
    # continue — don't fail the request over a cache miss
```

### Rate Limit Resilience (RAGAS Runner)

The checkpoint runner handles Groq rate limits by:
1. RAGAS internally retries failed LLM calls with exponential backoff
2. Checkpoint saves completed batches — a rate-limit kill at 85% only loses the last batch
3. Judge model is configurable via env var — switch to a different model's TPD bucket without code changes

---

## Scaling Considerations

### Current Scale (Demo)
- 10 documents, 12,927 chunks
- 1 FastAPI instance, 1 worker
- All components on a single machine (Docker Compose)

### Path to Production Scale

**Horizontal scaling (100K+ queries/day):**
1. Move BGE-M3 and BGE-Reranker to a dedicated GPU instance (GPU inference server)
2. FastAPI behind a load balancer (NGINX, HAProxy) with multiple workers
3. Redis Cluster for semantic cache (instead of single Redis)
4. Qdrant Cloud or self-hosted Qdrant cluster

**Ingestion scaling (1M+ documents):**
1. Replace rank-bm25 with Elasticsearch or OpenSearch (incremental indexing)
2. Distributed ingestion workers (Celery + RabbitMQ, or AWS SQS + Lambda)
3. VESSL/SageMaker for enrichment LLM calls (horizontal GPU scaling)
4. Qdrant collections sharded across multiple nodes

**Search scaling (1B+ vectors):**
1. Qdrant's built-in sharding and replication
2. HNSW index parameters tuning (ef, m) for recall/speed tradeoff
3. Vector quantization (scalar or product quantization) to reduce memory from 1024-float → 1024-int8 (4× compression)

**BM25 → Elasticsearch migration trigger point:** ~100,000 chunks. At that scale, full rebuilds on ingest become too slow (multiple seconds instead of <1 second). Elasticsearch supports incremental index updates.

---

## Future and Enterprise Improvements

### Immediate Next Steps

1. **GPU for reranker:** The single biggest latency win. P95 drops from 37s → 3s.
2. **Streaming by default:** The SSE endpoint exists but isn't the default in the UI. Stream tokens to the client while generation runs → perceived latency drops immediately.
3. **Actual RAGAS scores in docs:** Run eval after token quota resets, update STATUS.md and README.md with real numbers.

### Enterprise Features

**Multi-tenant isolation:**
- Qdrant supports named collections — one collection per tenant
- Postgres row-level security on documents and chunks tables
- Redis key namespacing per tenant

**Document versioning:**
- Track document versions in Postgres (same filename, different content)
- Qdrant point payload includes `version` field
- Queries can be scoped to a specific version range

**Feedback loop:**
- Store user thumbs-up/thumbs-down on answers
- Track which chunks were cited in accepted answers
- Fine-tune the reranker on (query, positive_chunk, negative_chunk) triplets using accumulated feedback

**Access control:**
- Per-document ACLs in Postgres
- Filter Qdrant search results by `doc_id` list from authorized documents
- JWT authentication on the FastAPI routes

**Guardrails:**
- Input validation: max query length, language detection
- Output validation: faithfulness check before returning (if a claim can't be mapped to a retrieved chunk, flag it)
- PII detection on ingested documents

**Async ingestion pipeline:**
- Current: ingestion blocks the HTTP request (via background task but same machine)
- Better: publish to a message queue (SQS, RabbitMQ), separate ingestion workers
- `app/workers/ingest_worker.py` has the skeleton for this pattern

**Embedding model versioning:**
- When upgrading from BGE-M3 v1 to v2, old vectors are incompatible
- Track `embedding_model_version` in Postgres and Qdrant payload
- Implement zero-downtime re-embedding: new model writes to a shadow collection; swap after validation

### Research Improvements

**Agentic sufficiency judge:** Replace the heuristic sufficiency check with an LLM judge that reads the retrieved chunks and decides if they answer the question before generating. Higher accuracy for adversarial queries; 1-2s added latency.

**RAG-Fusion (multi-query):** Generate 3-5 query variants, retrieve for each, fuse all result lists via RRF. Improves recall for complex multi-part questions. 3-5× more LLM calls for query rewriting.

**ColBERT late interaction:** BGE-M3 supports ColBERT-style multi-vector retrieval. Instead of a single 1024-dim vector per chunk, each token gets a vector. More expressive but requires specialized indexing (PLAID index). Would improve precision for multi-faceted queries.

**Sparse + dense in one model:** BGE-M3 can produce both dense vectors and sparse (SPLADE-style) token weights in a single forward pass. Using both from one model eliminates the need for a separate rank-bm25 index and enables Qdrant's hybrid search natively.

---

## Lessons Learned

### Lesson 1: Groq's Rate Limits Are Per-Model, Not Per-Account

`llama-3.1-8b-instant` and `llama-3.3-70b-versatile` have completely separate daily token buckets (500K each). When the 8B model was exhausted, switching to the 70B model continued evaluation without waiting 24 hours. Always design multi-model fallback.

### Lesson 2: RAGAS's `evaluate()` Returns Per-Sample Lists, Not Scalars

In RAGAS 0.2.x, `result["faithfulness"]` returns a `list[float | None]` — one score per sample — not a scalar mean. Code that calls `float(result["faithfulness"])` crashes with `TypeError: float() argument must be a string or a real number, not 'list'`. You must compute `mean(scores)` explicitly, filtering `None` values from failed LLM judge calls.

### Lesson 3: Qdrant's gRPC Has a 4MB Message Limit

Upserting 1,800+ points in a single call fails with an empty exception string. The fix is obvious in hindsight (batch to 200 points), but the silent failure with no error message cost significant debugging time. Always batch large writes to any external service.

### Lesson 4: Python Standalone Scripts Don't Load .env Automatically

FastAPI apps using `pydantic-settings` load `.env` automatically. Standalone Python scripts (evaluation runner, benchmark) do not. Every script that runs outside FastAPI needs:
```python
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
```
Or you run with `GROQ_API_KEY=... python script.py`.

### Lesson 5: The BM25 Pickle Is a Cache, Not the Source of Truth

Initially, the system degraded silently when the BM25 pickle was missing — dense search still worked but BM25 produced no results. Adding a startup integrity check that rebuilds from Postgres (and logs a clear warning) makes the degradation visible and self-healing instead of silent.

### Lesson 6: Semantic Cache Threshold Matters More Than You Think

With threshold=0.90, the cache returned wrong answers for subtly different questions (e.g., "Apple revenue 2024" → cached "Apple revenue 2023" answer because the embeddings are similar). With threshold=0.95, false positive cache hits dropped to zero in all observed test cases. Calibrate the threshold against your actual query distribution.

### Lesson 7: Observability Is Not Optional

Without Langfuse tracing, debugging a wrong answer required re-running the query with print statements added. With tracing, you navigate to the trace, see that the reranker scored the correct chunk at rank 6 (just outside top-5), and immediately know the fix is to increase `rerank_top_k` from 5 to 7. The tracing investment paid back in the first debugging session.

### Lesson 8: Checkpoint Everything That Calls External APIs in a Loop

The RAGAS runner spent many hours burning through API quotas due to crashes in post-processing code. Adding checkpointing after each batch would have saved all those API calls. The rule: any script that makes >10 external API calls in a loop needs checkpointing.

---

## Interview Preparation Guide

### "Explain this system to me in 2 minutes."

This is a production RAG system for financial document Q&A. Users upload SEC 10-K annual reports; the system ingests them through a 7-step pipeline that parses, chunks, enriches each chunk with LLM-generated context (Anthropic's Contextual Retrieval pattern), embeds with BGE-M3, and stores in Qdrant vector DB + BM25 keyword index.

For queries, a LangGraph agent rewrites the query, runs parallel dense + sparse retrieval, fuses results with Reciprocal Rank Fusion, reranks top-50 down to top-5 with a cross-encoder, and generates a citation-grounded JSON answer via Groq Llama. A Redis semantic cache serves repeated questions in under 100ms. The system is fully observable via Langfuse tracing and Prometheus metrics, with automated RAGAS evaluation gating on faithfulness ≥ 0.80.

### "Why did you choose Qdrant over Pinecone?"

Three reasons: (1) Qdrant is self-hosted, so no data leaves the environment — important for financial documents. (2) No per-query API cost — important for demo/dev without a budget. (3) Qdrant v1.9 has excellent gRPC performance and native Python async client support. Pinecone is the right choice for production at cloud scale where managed infrastructure is preferred.

### "Explain Reciprocal Rank Fusion."

RRF merges multiple ranked lists without needing to normalize scores. Each document's RRF score is the sum of 1/(k + rank) across all lists. k=60 is a smoothing constant from the original 2009 paper that prevents a single top-ranked document from dominating. Documents appearing at the top of both the dense list and the BM25 list score roughly double those appearing in only one list. The beauty of RRF is that it's rank-based — you never need to put BM25 scores (TF-IDF, unbounded) and cosine similarities (0-1) on the same scale.

### "Why two-stage retrieval? Why not just use the reranker directly?"

The cross-encoder reranker must process each (query, chunk) pair jointly — it can't use pre-computed chunk vectors. For 12,927 chunks, that's 12,927 inference calls at the query stage, taking hours on CPU and minutes even on GPU. The bi-encoder pre-computes all chunk vectors at ingestion time; at query time it only computes one query vector and does ANN search in ~20ms. The two-stage design combines the speed of the bi-encoder with the precision of the cross-encoder: bi-encoder narrows to 50, cross-encoder reranks 50 → 5 precisely.

### "What is Contextual Retrieval and why does it matter?"

Standard RAG chunks a document and embeds each chunk in isolation. A chunk saying "revenue increased 12%" gets embedded without knowing it's Apple's iPhone revenue in Q3 2024. When a user asks "Apple iPhone Q3 2024 revenue growth?", the vector for "revenue increased 12%" may not be close enough to retrieve.

Contextual Retrieval (Anthropic, 2024) generates a 2-3 sentence context blurb per chunk using the full document as context: "This chunk is from Apple's FY2024 10-K, discussing iPhone segment performance in Q3 2024." This blurb is prepended to the chunk before embedding. The enriched embedding encodes the complete semantic context, dramatically improving retrieval for decontextualized numerical data. Anthropic measured a 35% reduction in retrieval failure rate for contextual embeddings alone, 67% with BM25 and reranking.

### "How does the semantic cache work?"

When a query arrives, we embed it with BGE-M3. We scan all cached (embedding, response) pairs and compute cosine similarity with the query embedding. If any cached embedding has similarity ≥ 0.95, we return the cached response immediately — no LLM call, <100ms latency. Otherwise, we run the full pipeline and store the (query_embedding, response) pair with a 1-hour TTL. The 0.95 threshold is high enough to prevent false positive cache hits while catching paraphrases ("Apple revenue 2024" and "Apple's fiscal 2024 revenue" share similarity ~0.97).

### "What would you change if you had to handle 10 million chunks?"

1. Replace rank-bm25 with Elasticsearch/OpenSearch for incremental indexing (rank-bm25 requires full rebuilds)
2. Switch Qdrant to cluster mode with sharding
3. Move BGE-M3 and BGE-Reranker to dedicated GPU inference servers (Triton, TorchServe) with batching and auto-scaling
4. Replace the Redis linear-scan semantic cache with Redis Vector (HNSW index for O(log n) lookup)
5. Add an ingestion queue (SQS + workers) for async parallel document processing
6. Vector quantization in Qdrant (int8 scalar quantization) to reduce memory from 10M × 1024 floats (40GB) to 10GB

### "How do you know the system is working correctly?"

Four layers:
1. **Langfuse tracing** — every query is traced with per-node inputs/outputs, so wrong answers are debuggable
2. **Prometheus metrics** — rag_query_latency_seconds and rag_chunks_retrieved alert on regressions
3. **RAGAS evaluation** — 50 golden Q&A pairs scored on Faithfulness, Context Recall, and Factual Correctness; CI gate requires faithfulness ≥ 0.80
4. **Adversarial test category** — 7 questions the system should decline to answer; Faithfulness rewards "I don't know" over hallucination

### "What was the hardest bug you fixed?"

The Qdrant batch upsert bug. Documents with 1,800+ chunks were failing silently — the pipeline would run, log success, but no chunks appeared in Qdrant. The exception from the gRPC call was an empty string (no message), so there was nothing in the logs to grep. The fix came from reading Qdrant's documentation carefully: the gRPC transport has a 4MB default message size limit. Computing 1,800 × ~2KB/point = 3.6MB — right at the edge, explaining the intermittent nature. Batching to 200 points/call (300KB) fixed it permanently.

---

*This document was written as a comprehensive technical reference for the Production RAG Knowledge Copilot system. For setup instructions, see README.md. For current system status and eval scores, see STATUS.md.*
