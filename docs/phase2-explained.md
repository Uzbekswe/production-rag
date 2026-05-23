# Phase 2 — Query Pipeline: Everything Explained

> A self-contained study guide. Every concept, design decision, and trade-off from the Phase 2 build — written so you can re-read any section independently and defend every choice in an interview.

---

## TLDR (read this first)

Phase 2 is the other half of the system. Phase 1 got data **in**. Phase 2 gets **answers out**.

A user types a question. The system returns a cited answer in ~1-2 seconds. Here's the full flow:

```
POST /api/v1/query  {"query": "What was Apple gross margin in FY2024?"}
  │
  ├─ Step 0: Embed query → check Redis semantic cache
  │           HIT  ────────────────────────────────────────► return cached answer (~10ms)
  │           MISS ↓
  │
  │  LangGraph Agent (stateful pipeline)
  ├─ Node 1: query_rewriter      Groq rewrites query for better retrieval
  ├─ Node 2: hybrid_retriever    Dense (BGE-M3 + Qdrant) + Sparse (BM25) in parallel
  │                               → RRF fusion → top-50 chunks
  ├─ Node 3: reranker            BGE cross-encoder → top-5 chunks
  ├─ Node 4: sufficiency_checker Is this enough context to answer? (heuristic)
  │           NO + retries left ──────────────────────────► loop back to Node 1
  │           YES ↓
  ├─ Node 5: generate            Groq llama-3.3-70b → answer + [Source N] citations
  │
  ├─ Write answer to Redis cache (for future similar queries)
  └─ Return QueryResponse {answer, citations, trace_url, latency_ms}
```

Files built in Phase 2:

| File | Role |
|---|---|
| `app/services/retrieval/dense.py` | BGE-M3 → Qdrant ANN search → top-50 |
| `app/services/retrieval/sparse.py` | BM25 keyword search → Qdrant payload fetch → top-50 |
| `app/services/retrieval/fusion.py` | RRF merge of dense + sparse results |
| `app/services/retrieval/reranker.py` | BGE cross-encoder singleton → top-5 |
| `app/services/generation/groq_gen.py` | Groq + structured JSON citation prompting |
| `app/services/generation/longcite.py` | LongCite-8B VESSL demo path |
| `app/services/generation/router.py` | Pick Groq vs LongCite via env var |
| `app/services/agent/nodes.py` | All 5 LangGraph node implementations |
| `app/services/agent/graph.py` | Updated: imports real nodes (was stubs) |
| `app/api/routes/query.py` | Complete query route: cache + agent + response |
| `app/services/ingestion/pipeline.py` | Patched: add `filename` to Qdrant payload |

---

## 1. The Big Picture — Why This Architecture?

### Hybrid Search: Dense + Sparse

Every production RAG system uses **hybrid search**. Here's why neither method alone is sufficient:

**Dense search only (BGE-M3 + Qdrant):**
- Great at semantic similarity — "cloud revenue" finds "cloud services income"
- Terrible at exact identifiers — "NVDA Q3 2024" might miss because the model maps "NVDA" inconsistently to "NVIDIA"
- Great at paraphrases, terrible at precision

**Sparse search only (BM25):**
- Great at exact token matches — finds "NVDA", "Q3", "2024" every time
- Terrible at paraphrases — "gross margin" won't find "profitability ratio"
- Great at precision, terrible at recall

**Hybrid (both + RRF):**
- A chunk that appears in both result lists is almost certainly relevant — rewarded with higher score
- A chunk only in one list might be relevant — kept, but lower score
- Best of both worlds: semantic understanding + exact token matching

This is why every serious RAG implementation (Cohere, Weaviate, Qdrant, OpenSearch) has hybrid search as a first-class feature. It's not a nice-to-have — it's table stakes for production.

### The Reranking Stage

Hybrid retrieval gives us top-50 candidates. But 50 chunks is too many to send to an LLM (context window, cost, distraction). We need to pick the best 5.

**Why not just use the top-5 from Qdrant cosine similarity?**
Cosine similarity compares query and chunk as independent vectors. It answers "are these in the same neighborhood of the 1024-dim space?" — a coarse measure.

A cross-encoder answers a different question: "given this exact query AND this exact chunk, how relevant is this passage?" It reads them together. It knows that "Apple's gross margin was 45.9%" is a direct answer to "What was Apple's gross margin?" in a way that pure vector similarity doesn't capture.

**The practical pattern in production:**
```
Stage 1: ANN retrieval     → top-100  (fast, approximate, recall-focused)
Stage 2: Cross-encoder     → top-10   (slow, exact, precision-focused)
Stage 3: Send to LLM       → top-5    (final quality filter)
```

We do this in two stages (RRF top-50 → cross-encoder top-5) because:
- Running cross-encoder on the whole corpus (thousands of chunks) would take minutes
- Running it on 50 candidates takes ~400ms — acceptable latency

### LangGraph: Why an Agent Loop?

Simple RAG is a one-shot pipeline: retrieve → generate. Agentic RAG adds a **retry loop**:

```
retrieve → evaluate quality → if insufficient: rewrite query and retry → generate
```

**When does this matter?**
- Vague queries: "Tell me about Apple" → first retrieval is too broad. Rewriter narrows it.
- Query-document vocabulary mismatch: "profitability" vs "gross margin" — BM25 misses the exact term, rewriter adds synonyms on retry.
- Multi-hop questions: "Compare Apple and Microsoft margins in 2024" — first retrieval focuses on one company; retry adds the other.

**Why LangGraph specifically?**
LangGraph builds a **stateful directed graph** where each node is a function and edges determine the next step. The "conditional edge" after sufficiency check is what enables the retry loop — it's not just a linear pipeline, it's a graph with a cycle.

LangGraph also provides:
- State persistence between nodes (no passing giant function arguments)
- Built-in support for async execution
- Native Langfuse tracing integration
- Easy to add human-in-the-loop approval nodes in production

**Portfolio story:** Being able to say "I built an agentic RAG with a LangGraph retry loop" is significantly more impressive than "I built a RAG pipeline" — it demonstrates understanding of multi-step reasoning systems.

---

## 2. Retrieval Layer

### `retrieval/dense.py` — QdrantRetriever

**What it does:** Embeds the query → finds the 50 most similar vectors in Qdrant.

**How Qdrant ANN works — HNSW:**
Qdrant uses HNSW (Hierarchical Navigable Small World graph). Imagine your 2,500 chunk vectors laid out in a high-dimensional space. HNSW builds a multi-layer graph connecting nearby vectors:
- Top layer: few nodes, long-range connections (fast navigation)
- Bottom layer: all nodes, short-range connections (precise neighborhood)

A search starts at the top layer, greedily navigates toward the query vector, then drills down for precision. This finds the approximate nearest neighbors in O(log n) time instead of O(n) linear scan. At 2,500 chunks the difference is small; at 1M chunks HNSW is the only practical option.

**`run_in_executor` pattern (again):**
`embedder.embed_query()` is synchronous. If called directly in `async def search()`, it blocks the event loop — no other requests are served while it runs. `loop.run_in_executor(None, embedder.embed_query, query)` moves it to a thread pool. The event loop stays free.

**The payload — why `filename` needed a pipeline.py patch:**
Qdrant stores vectors + a JSON "payload" alongside each vector. The original pipeline stored `doc_id`, `raw_text`, `full_text`, `page_num`, `char_start`, `char_end` in the payload — but NOT `filename`. The retriever needs `filename` to build `ScoredChunk` objects (for citations). Options:
1. Look up filename from Postgres for every result (N queries per search) — slow
2. Add filename to the Qdrant payload at ingest time (one extra field) — free

We chose option 2. One-line patch in `pipeline.py`: add `"filename": filename` to the `PointStruct.payload` dict. Now retrieval is self-contained — Qdrant has everything we need.

**Architecture.md alignment:** Exact match — BGE-M3 → Qdrant cosine top-50.

---

### `retrieval/sparse.py` — BM25Retriever

**What it does:** BM25 keyword scoring → top-50 chunk IDs → batch Qdrant fetch for payloads.

**Why fetch payloads from Qdrant (not Postgres)?**
BM25Index stores only `(qdrant_id, bm25_score)` — just enough to return ranked IDs. To build a full `ScoredChunk` (with raw_text, filename, page_num), we need the payload. Two options:
1. Postgres: `SELECT * FROM chunks WHERE qdrant_id IN (...)` — SQL query, cross-process call
2. Qdrant `retrieve()`: one in-memory batch lookup — same process, no network round-trip

We use Qdrant. It's faster, and both retrievers now have the same interface: "give Qdrant some IDs, get back ScoredChunks." Consistency matters for maintainability.

**The sort after retrieve:**
Qdrant's `retrieve()` doesn't guarantee order — it returns points in whatever order they're stored. After fetching, we re-sort by the BM25 score stored in `score_map`. Don't forget this step — unsorted results would feed wrong rankings into RRF.

**Architecture.md alignment:** Exact match — BM25 top-50 sparse retrieval.

---

### `retrieval/fusion.py` — ReciprocalRankFusion

**What it does:** Merges two ranked lists into one using the RRF formula.

**The RRF formula in detail:**

```
For each result list L containing chunk d at rank r:
    rrf_score(d) += 1 / (k + r + 1)

k = 60   (smoothing constant, prevents top-rank dominance)
```

Why `k=60`? Without k, rank 1 = score 1.0, rank 2 = score 0.5 — huge cliff. With k=60:
- Rank 1: 1/61 = 0.0164
- Rank 2: 1/62 = 0.0161
- Rank 10: 1/70 = 0.0143

The differences are much smaller, meaning a chunk at rank 5 in both lists can outrank a chunk at rank 1 in only one list:
- Rank 5 in both: 2 × 1/65 = 0.0308
- Rank 1 in one: 1 × 1/61 = 0.0164

This is the RRF paper's key insight: **agreement between lists is more valuable than dominance in one list.**

**The `chunk_map.setdefault` pattern:**
If a chunk appears in both dense and sparse results, we have two `ScoredChunk` objects with the same content but different original scores. We keep the dense version (via `chunk_map[chunk_id] = chunk` for dense, `chunk_map.setdefault(chunk_id, chunk)` for sparse — setdefault only writes if key doesn't exist). Either version is fine since the RRF score replaces both original scores anyway.

**Production note:** At scale, RRF is sometimes replaced by learned fusion weights trained on click-through data. k=60 is a solid default that performs well without training data.

**Architecture.md alignment:** Exact match — RRF k=60.

---

### `retrieval/reranker.py` — BGEReranker

**What it does:** Cross-encoder reranking — takes top-50, returns top-5.

**Bi-encoder vs Cross-encoder — the core difference:**

| | Bi-encoder (BGE-M3) | Cross-encoder (BGE-reranker) |
|---|---|---|
| How | Encode query → vector. Encode chunk → vector. Compare vectors. | Concatenate [query, chunk] → single forward pass → relevance score |
| Speed | Fast (encode once, compare many) | Slow (one forward pass per pair) |
| Scale | Millions of documents | Hundreds of candidates |
| Accuracy | Good | Excellent |
| Used for | Initial retrieval (top-50) | Reranking (top-50 → top-5) |

The cross-encoder attention layers can see exactly how query tokens relate to chunk tokens. When the query is "What was Apple gross margin?" and the chunk contains "Apple reported a gross margin of 45.9%", the cross-encoder learns that "gross margin" in the query directly matches "gross margin" in the chunk. A bi-encoder just knows they're both in the "financial metrics" neighborhood.

**`normalize=True` in compute_score:**
BGE-reranker's raw output is a logit (unbounded real number). `normalize=True` applies sigmoid to convert it to [0, 1]. This makes scores interpretable: 0.9 = very relevant, 0.1 = probably not relevant. It also makes the sufficiency checker's `avg_score ≥ 0.2` threshold meaningful.

**Singleton + async wrapper:**
Same pattern as BGEEmbedder. Model loads once (~1.5GB RAM). `rerank_async()` offloads to thread pool. The 50-pair computation takes ~400ms on CPU — acceptable for a query pipeline.

**Architecture.md alignment:** Exact match — BGE-reranker-v2-m3, top-5.

---

## 3. Generation Layer

### `generation/groq_gen.py` — GroqGenerator

**What it does:** Builds a numbered-source prompt → calls Groq → parses JSON citations.

**The citation prompting strategy:**

Naive approach (don't do this):
```
"Here are some documents: [paste chunks]. Answer the question. Cite your sources."
```
Problem: the model will write something like "According to the document..." without specifying which document, or will hallucinate citation numbers.

Our approach:
```
[Source 1]
{chunk_1_full_text}

[Source 2]
{chunk_2_full_text}

Question: What was Apple's gross margin in FY2024?
```

With the system prompt forcing JSON output:
```json
{
  "answer": "Apple's gross margin in FY2024 was 45.9% [Source 1], driven by strong services performance [Source 2].",
  "citations": [
    {"source_id": 1, "cited_text": "gross margin of 45.9%"},
    {"source_id": 2, "cited_text": "Services segment revenue grew 13%"}
  ]
}
```

**Why `response_format: {"type": "json_object"}`:**
Groq's JSON mode constrains the model's output to valid JSON. Without it, the model might write valid JSON 95% of the time and malformed JSON 5% of the time (usually on long answers). With JSON mode: 100% valid JSON or an error we can handle.

**`_parse_response` — the citation mapper:**
`source_id` is 1-indexed (matches the `[Source N]` in the prompt). Chunks are 0-indexed in Python. `chunk_idx = source_id - 1`. We look up the original `ScoredChunk` by index to get its `chunk_id`, `filename`, `page_num`, and `score` — the citation carries full provenance.

**Graceful degradation on JSON failure:**
If the model returns malformed JSON (extremely rare with JSON mode, but possible), `json.JSONDecodeError` is caught. We return the raw text as the answer with empty citations instead of raising a 500. The user gets an answer, just without structured citations. Graceful degradation > hard failure.

**`temperature=0.1`:**
Low temperature = more deterministic, less creative. For factual financial Q&A, we want the model to report what the sources say, not generate novel interpretations. 0.1 gives slightly more variation than 0.0 (avoids degenerate repetition) while staying factual.

**Architecture.md alignment:** Exact match — Groq llama-3.3-70b, [Source N] JSON prompting, ADR-004.

---

### `generation/longcite.py` — LongCiteGenerator

**What it is:** A stub generator that calls a VESSL-hosted LongCite-8B inference server.

**LongCite vs Groq prompting — the key difference:**
With Groq: we *prompt engineer* citations into a general-purpose model. We write a system prompt that says "you MUST cite as [Source N]" and parse the output.

With LongCite: citations are *baked into the model weights*. The model was fine-tuned on 30,000+ document-QA pairs where citation spans were annotated at the sentence level. It natively outputs which sentences in the answer came from which part of the source — no prompt engineering required.

**The practical trade-off:**

| | Groq + prompting | LongCite-8B |
|---|---|---|
| Cost | $0 (free tier) | ~$1.52 per 2-hour VESSL session |
| Citation quality | Good (95%+ when JSON mode works) | Better (native, sentence-level) |
| Availability | Always-on | On-demand only |
| Model size | 70B params | 8B params |
| Latency | ~2s (API call) | ~1s (smaller model) |

The router makes switching transparent: set `LONGCITE_ENDPOINT=http://vessl:8000` in .env → LongCite activates. Remove it → back to Groq. No code changes.

**Architecture.md alignment:** Exact match — LongCite-8B demo path, VESSL on-demand, ADR-004.

---

### `generation/router.py` — GeneratorRouter

**What it does:** 4 lines that implement the Strategy pattern.

**The Strategy pattern:**
Define a common interface (`generate(query, chunks) → GenerationResult`). Multiple implementations (Groq, LongCite) implement that interface. The caller never imports a specific implementation — it imports the router. Swapping backends is an environment variable change, not a code change.

This is the same pattern used in production systems everywhere:
- Payment processors (Stripe vs Braintree) behind a unified payment interface
- Storage backends (S3 vs GCS) behind a unified object store interface
- LLM providers (OpenAI vs Anthropic vs Groq) behind a unified completion interface

**Why a module-level `_groq = GroqGenerator()` instance:**
`GroqGenerator` creates an `AsyncGroq` client in `__init__`. Creating it once at module import time means the HTTP connection pool is set up once, not per-request. `LongCiteGenerator` is stateless (just makes HTTP calls) — no need to pre-initialize.

**Architecture.md alignment:** Exact match — dual-path generation, ADR-004.

---

## 4. Agent Layer

### `agent/nodes.py` — The 5 LangGraph Node Functions

**How LangGraph nodes work:**
Each node function receives the full `RAGState` dict and returns a *partial* dict of only the keys it changed. LangGraph merges the returned dict into the current state — unmentioned keys stay unchanged.

```python
# Node returns only what it changed:
async def reranker_node(state: RAGState) -> dict:
    ...
    return {"retrieved_chunks": [c.model_dump() for c in reranked]}
    # query, rewritten_queries, etc. are unchanged — not mentioned, not touched
```

This is elegant: each node is independent and only knows about its own inputs/outputs. If you add a new field to RAGState later, existing nodes don't need to be updated.

---

#### Node 1: `query_rewriter_node`

**Input:** The current query (last in `rewritten_queries`), current `retrieval_attempt`
**Output:** Appends rewritten query to `rewritten_queries`, increments `retrieval_attempt`

**Why rewrite at all?**
Financial documents have specific vocabulary. A user might ask "How profitable was Apple?" but the 10-K says "gross margin" and "operating income" — not "profitable." A rewriter with knowledge of financial terminology can bridge this vocabulary gap before retrieval.

On the **first attempt** (retrieval_attempt=0): the rewriter adds financial context to a possibly vague query.
On a **retry** (retrieval_attempt>0): the sufficiency checker already ran and found the results insufficient. The rewriter gets a second shot with different wording, hopefully covering vocabulary the first attempt missed.

**Failure handling:**
If Groq fails (rate limit, network), `logger.warning` and fall back to the original query — the pipeline continues without crashing. The rewriter is an enhancement; the query still works without it.

---

#### Node 2: `hybrid_retriever_node`

**Input:** Latest rewritten query
**Output:** `retrieved_chunks` list (top-50 after RRF fusion)

**`asyncio.gather` — parallel retrieval:**
```python
dense_results, sparse_results = await asyncio.gather(
    _dense.search(query),
    _sparse.search(query),
)
```

Both searches are independent — Qdrant ANN and BM25 don't share state. `asyncio.gather` runs both coroutines concurrently. Total latency = max(dense_latency, sparse_latency) ≈ 50ms, not sum ≈ 100ms.

In production systems, you might add a third retrieval branch (e.g., knowledge graph lookup, metadata filtering) and include it in the gather — same pattern, free parallelism.

**Storing as dicts in state:**
`ScoredChunk` objects are Pydantic models. LangGraph serializes state between nodes (for checkpointing and distributed execution). Pydantic models aren't directly JSON-serializable in LangGraph's internal format. We call `.model_dump()` to convert each `ScoredChunk` to a plain dict before storing in state. The next node converts back: `ScoredChunk(**c)`.

---

#### Node 3: `reranker_node`

**Input:** `retrieved_chunks` (top-50 dicts)
**Output:** `retrieved_chunks` (top-5 dicts, re-scored by cross-encoder)

Converts dicts back to `ScoredChunk` objects, runs `BGEReranker.rerank_async()` in thread pool, converts back to dicts. The `score` field is overwritten with the cross-encoder's normalized score.

**Why top-5?**
- Context window: 5 × ~400 chars = ~2,000 chars ≈ 500 tokens of context. Leaves ample room in Groq's 32K context window for the system prompt, question, and answer.
- Quality: More chunks = more noise. The reranker ensures these 5 are the best.
- Cost: Fewer input tokens = cheaper (even on free tier, fewer tokens = faster response).

---

#### Node 4: `sufficiency_checker_node`

**Input:** `retrieved_chunks` (top-5), `retrieval_attempt`
**Output:** `is_sufficient` bool

**The heuristic:**
```python
avg_score = sum(c["score"] for c in chunks) / len(chunks)
is_sufficient = len(chunks) >= 3 and avg_score >= 0.2
```

- `len(chunks) >= 3`: If we retrieved fewer than 3 chunks, retrieval almost certainly failed (the corpus has thousands of chunks — we should always get at least 3 matches for any real query).
- `avg_score >= 0.2`: BGE-reranker scores in [0,1]. Score 0.2 is low — it means "marginally relevant." If the average is below this, the reranker is telling us these chunks are poor matches.

**Why not an LLM judge?**
Architecture.md mentions using an LLM to estimate context_recall — "do these chunks contain enough information to answer the question?" This is semantically more accurate but costs 1-2 Groq API seconds + tokens on EVERY query, including the 95% where retrieval obviously worked. The heuristic catches the real failure modes (empty results, garbage retrieval) at zero cost. An LLM judge would be the right upgrade when you have adversarial users deliberately crafting unanswerable queries — a Phase 3 enhancement.

**The retry loop:**
```python
def _should_retry(state: RAGState) -> str:
    if not state["is_sufficient"] and state["retrieval_attempt"] < settings.max_agent_retries:
        return "query_rewriter"
    return "generate"
```

`max_agent_retries=2` means at most 2 full retrieval attempts (configured in `settings`). After that, we generate anyway — better to return a low-confidence answer than to loop indefinitely.

---

#### Node 5: `generate_node`

**Input:** Latest query, `retrieved_chunks` (top-5)
**Output:** `answer`, `citations`, `model_used`

Calls `router.generate()` — single line that hides the Groq vs LongCite decision. Converts dicts back to `ScoredChunk` for the generator, returns `GenerationResult` fields as dicts for state storage.

**The "generate anyway" principle:**
Even when `is_sufficient=False` (retrieval failed, retries exhausted), we still call generate. The Groq system prompt says: "If no source contains the answer, say 'The provided sources do not contain enough information.'" So the model gracefully admits ignorance rather than hallucinating. This is far better than returning a 503 error to the user.

---

### `agent/graph.py` — The Compiled Graph

**What changed:** Removed 5 `raise NotImplementedError` stubs, imported real implementations from `nodes.py`.

**The conditional retry edge:**
```python
g.add_conditional_edges(
    "sufficiency_checker",
    _should_retry,                                            # routing function
    {"query_rewriter": "query_rewriter", "generate": "generate"},  # possible destinations
)
```

LangGraph calls `_should_retry(state)` after `sufficiency_checker` runs. The return value ("query_rewriter" or "generate") maps to the next node via the dict. This is LangGraph's way of encoding decision logic in the graph structure — the routing function is just a Python function.

**`rag_graph = build_rag_graph().compile()`:**
`.compile()` validates the graph structure (no dangling edges, all nodes reachable) and optimizes it for execution. The compiled graph is a module-level singleton — it's created once at import time and reused for every query.

---

## 5. Query Route — `api/routes/query.py`

**The full request lifecycle:**

### Step 1: Langfuse Trace
```python
trace = create_trace("rag_query", session_id=query_id, input={"query": ...})
```
Every query gets a root Langfuse trace. Each LangGraph node adds its own span. In Langfuse UI at `localhost:3000`, you click any query and see the full breakdown: which node took how long, what was the input/output at each step.

### Step 2: Semantic Cache Check
```python
query_embedding = await loop.run_in_executor(None, embedder.embed_query, payload.query)
cached = await cache_lookup(redis, query_embedding)
```

We embed the query first (before the pipeline) because we need the embedding for:
1. Cache lookup (compare against stored embeddings by cosine similarity)
2. Cache write (store this embedding alongside the answer for future lookups)

If cache hit: return in ~10ms. No Groq calls, no Qdrant searches, no reranking. The semantic cache is the single biggest latency optimization in the system.

### Step 3: LangGraph Pipeline
```python
result_state = await rag_graph.ainvoke(initial_state)
```

`ainvoke` is LangGraph's async execution method. It runs all nodes sequentially (or in parallel where the graph allows it), threading state through each node. The entire pipeline — rewrite, retrieve, rerank, generate — happens in this one awaited call.

**The initial state:**
```python
initial_state: RAGState = {
    "query": payload.query,
    "rewritten_queries": [payload.query],  # starts with original query
    "retrieved_chunks": [],
    "retrieval_attempt": 0,
    "is_sufficient": False,
    ...
}
```

We pre-populate `rewritten_queries` with the original query so Node 1 (query_rewriter) can read `state["rewritten_queries"][-1]` consistently — whether it's the first attempt or a retry.

### Step 4: Response Assembly
```python
response = QueryResponse(
    query_id=query_id,
    answer=result_state["answer"],
    citations=[Citation(**c) for c in result_state["citations"]],
    retrieval_method="hybrid_rrf_rerank",
    latency_ms=latency_ms,
    model_used=result_state["model_used"],
    from_cache=False,
    trace_url=f"{settings.langfuse_host}/trace/{query_id}",
)
```

`trace_url` is the direct Langfuse link for this specific query. When demoing to an interviewer, you can run a query, point to the trace URL in the response, open it in the browser, and show the full pipeline execution in real time. That's a powerful demo moment.

### Step 5: Cache Write
```python
await cache_store(redis, query_embedding, response.model_dump(), ttl=settings.cache_ttl_seconds)
```

Store the query embedding + full response. The next similar query will hit this cache and return instantly. TTL = 1 hour (from settings) — stale answers expire automatically.

**Error handling:**
Cache write failure is caught and logged as a warning (not an error). The cache is an optimization, not a critical path — if Redis is down, the user still gets their answer. This is the "fail open" principle for non-critical systems.

---

## 6. What Was Simplified vs Full Production

| Feature | What we built | Full production version |
|---|---|---|
| Query rewriting | Simple Groq rewrite | FACTUAL/SPARSE/COMPLEX classifier + HyDE + query decomposition |
| Sufficiency check | Heuristic (count + score) | Fine-tuned LLM judge scoring context recall |
| BM25 scale | In-memory pickle | Elasticsearch / OpenSearch cluster |
| Semantic cache | Linear scan over all keys | Redis Vector Search (sub-ms ANN lookup) |
| Streaming | Synchronous JSON response | SSE token streaming (Phase 3) |
| Retry count | Fixed `max_agent_retries=2` | Dynamic, based on query complexity |
| Multi-tenancy | Single knowledge base | Per-user / per-org collections in Qdrant |

**What's genuinely production-grade in what we built:**

| Pattern | Where | Why it matters |
|---|---|---|
| Hybrid search (dense + sparse) | retrieval/ | Industry standard — Weaviate, Cohere, Qdrant all use it |
| RRF fusion | fusion.py | Peer-reviewed formula, beats all naive fusion methods |
| Cross-encoder reranking | reranker.py | Standard two-stage retrieval pattern in production |
| Singleton model loading | reranker.py, embedder.py | Amortizes 10-30s load time across all requests |
| Thread pool for CPU-bound work | everywhere | Keeps async event loop responsive |
| Structured JSON citation output | groq_gen.py | Reliable parsing at production scale |
| Semantic caching | query route | Cuts repeated query costs to zero |
| Graceful degradation | enricher, groq_gen, nodes | System degrades without crashing |
| Full distributed tracing | every node | LLMOps best practice, debuggable in production |
| Agentic retry loop | LangGraph graph | Beyond basic RAG — handles retrieval failures |
| Strategy pattern for generators | router.py | Zero-code swap between Groq and LongCite |

---

## 7. Data Flow — End to End

Let's trace a real query through every function call:

```
User: "What was Apple gross margin in FY2024?"

① query_route.py:
   embed_query("What was Apple gross margin in FY2024?") → 1024-dim vector
   cache_lookup(redis, vector) → None (miss)

② graph.ainvoke(initial_state)

③ query_rewriter_node:
   Groq: "What was Apple Inc gross margin percentage fiscal year 2024 annual report?"
   state.rewritten_queries = [original, rewritten]
   state.retrieval_attempt = 1

④ hybrid_retriever_node:
   asyncio.gather(
     dense.search("Apple Inc gross margin...") → [chunk_A(0.91), chunk_B(0.88), ...50 total],
     sparse.search("Apple Inc gross margin...") → [chunk_A(18.2), chunk_C(15.1), ...50 total]
   )
   rrf.fuse(dense, sparse) → [chunk_A(0.033), chunk_B(0.016), chunk_C(0.016), ...merged 60]

⑤ reranker_node:
   BGEReranker.rerank("Apple Inc gross margin...", [chunk_A, ...60]) →
   [chunk_A(0.94), chunk_D(0.87), chunk_B(0.81), chunk_E(0.73), chunk_F(0.65)]

⑥ sufficiency_checker_node:
   len(chunks)=5 ≥ 3 ✓, avg_score=0.80 ≥ 0.2 ✓ → is_sufficient=True

⑦ generate_node:
   GroqGenerator.generate("Apple gross margin FY2024?", [chunk_A, chunk_D, ...]) →
   answer: "Apple's gross margin in FY2024 was 45.9% [Source 1], an improvement from 44.1% in FY2023 [Source 2]."
   citations: [{source_id:1, chunk_id:"abc...", filename:"AAPL_10K_2024.pdf", page_num:41, ...}, ...]

⑧ query_route.py:
   QueryResponse(answer=..., citations=[...], latency_ms=2341, from_cache=False, trace_url="http://localhost:3000/trace/...")
   cache_store(redis, query_vector, response)
   return response
```

---

## 8. Key Concepts Cheat Sheet

| Concept | One-liner |
|---|---|
| Hybrid search | Dense (semantic) + Sparse (keyword) → better than either alone |
| HNSW | Graph-based ANN index. O(log n) search. Used by Qdrant. |
| Bi-encoder | Encodes query and chunk separately → cosine similarity (fast, scalable) |
| Cross-encoder | Encodes query + chunk together → relevance score (slow, accurate) |
| RRF (k=60) | Rank fusion formula. Agreement between lists = higher score. |
| LangGraph | Framework for stateful agent graphs with conditional edges and retry loops |
| `asyncio.gather` | Run multiple async operations concurrently. Latency = max, not sum. |
| `run_in_executor` | Offload sync CPU work to thread pool. Keeps event loop free. |
| JSON mode | `response_format: {"type": "json_object"}` forces valid JSON from Groq |
| Sufficiency checker | Gate before generation. Retry if retrieval quality is too low. |
| Strategy pattern | Common interface, multiple backends. Swap via config, not code. |
| Semantic cache | Cache by embedding similarity (cosine ≥ 0.95), not exact string match |
| Graceful degradation | On failure: return degraded answer, never crash. |
| `model_dump()` | Convert Pydantic model → dict for LangGraph state serialization |
| `trace_url` | Direct Langfuse link in every response. Open it to debug any query. |

---

## 9. How to Run and Verify

```bash
# 1. Start all infrastructure (Postgres, Qdrant, Redis, Langfuse)
docker compose up -d

# 2. Start the FastAPI server
.venv/bin/uvicorn app.main:app --reload

# 3. Ingest the SEC 10-K PDFs (Phase 1 must have run first)
python scripts/ingest_sample.py

# 4. Ask a question — first time (cache miss, full pipeline)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple gross margin in FY2024?"}'

# Expected:
# {
#   "query_id": "...",
#   "answer": "Apple's gross margin was 45.9% in FY2024 [Source 1]...",
#   "citations": [{"source_id": 1, "filename": "AAPL_10K_2024.pdf", "page_num": 41, ...}],
#   "retrieval_method": "hybrid_rrf_rerank",
#   "latency_ms": 2341,
#   "model_used": "llama-3.3-70b-versatile",
#   "from_cache": false,
#   "trace_url": "http://localhost:3000/trace/..."
# }

# 5. Same query again (cache hit — should be ~10ms)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple gross margin in FY2024?"}'
# Expected: "from_cache": true, "latency_ms": ~10

# 6. Open Langfuse to see the trace
open http://localhost:3000
# Click the trace → see query_rewriter, hybrid_retriever, reranker, sufficiency_checker, generate spans
```

---

## 10. What Phase 3 Adds

Phase 3 is **Observability** — making the system measurable and monitorable:

- **Prometheus metrics** — request count, latency percentiles, cache hit rate, model errors
- **Grafana dashboard** — visualize all metrics over time
- **SSE streaming** — stream answer tokens as they're generated (better UX)
- **LLM sufficiency judge** — replace the heuristic sufficiency checker with a trained model
- **Full Langfuse spans** — add `span()` context managers inside every node for per-step timing

The system works without Phase 3. Phase 3 makes it *observable* — you can tell when it's slow, when the cache is cold, when Groq is rate-limiting. Observability is what separates a system you can operate from one you're guessing about.
