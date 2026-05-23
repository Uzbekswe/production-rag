# Phase 1 — Ingestion Pipeline: Everything Explained

> A self-contained study guide. Every concept taught during the Phase 1 build, organized so you can re-read any section independently.

---

## TLDR (read this first)

Phase 1 turns a raw PDF into searchable, citable knowledge stored in three places:
- **Qdrant** — vector database for semantic (meaning-based) search
- **BM25 index** — in-memory keyword search (complementary to semantic)
- **Postgres** — source of truth for document metadata and chunk text

The pipeline has 7 sequential steps:

```
PDF file
  │
  ▼
[1] PARSE      Docling extracts clean Markdown text from the PDF
  │
  ▼
[2] CHUNK      Split into 400-char overlapping pieces
  │
  ▼
[3] ENRICH     Groq LLM writes an 80-100 token "context blurb" per chunk
  │             (Contextual Retrieval — reduces retrieval failure by 67%)
  ▼
[4] EMBED      BGE-M3 converts each enriched chunk → 1024-dim vector
  │
  ▼
[5] QDRANT     Store vectors + metadata in Qdrant for ANN search
  │
  ▼
[6] BM25       Rebuild the keyword index over all chunks (including new ones)
  │
  ▼
[7] POSTGRES   Persist chunk records; mark document as "ready"
```

The API returns **202 Accepted immediately** — the pipeline runs in the background. The user polls `GET /ingest/{job_id}` to check progress.

Files built in Phase 1:

| File | Role |
|---|---|
| `scripts/download_sec_filings.py` | Download 10 SEC 10-K PDFs from EDGAR |
| `app/core/qdrant.py` | Qdrant client + scalar quantization |
| `app/core/redis_client.py` | Redis client + semantic cache helpers |
| `app/core/tracing.py` | Langfuse trace + span context manager |
| `app/models/` | SQLAlchemy ORM: Document, Chunk, EvalRun, GoldenQuestion |
| `app/repositories/` | Only code that talks to Postgres directly |
| `app/schemas/` | Pydantic request/response contracts |
| `app/services/ingestion/` | The 6 pipeline stages |
| `app/workers/ingest_worker.py` | Background task session management |
| `app/api/routes/ingest.py` | POST /ingest, POST /ingest/url, GET /ingest/{job_id} |
| `app/api/routes/documents.py` | GET/DELETE /documents |
| `scripts/ingest_sample.py` | Batch ingest all 10 PDFs in one command |

---

## 1. Data Acquisition — `scripts/download_sec_filings.py`

### What it does
Downloads 10-K annual reports (Apple, Microsoft, NVIDIA, Alphabet, Meta) from SEC EDGAR for fiscal years 2023 and 2024. These are our RAG corpus — 10 PDFs, ~830 pages total.

### Why SEC 10-K filings?
After researching 8+ production RAG portfolio projects on GitHub, this is the industry standard corpus. Reasons:
- **Free and public** — SEC is a US government agency, no copyright issues
- **Rich content** — 80-150 pages with tables, risk factors, financial statements
- **Well-known companies** — any interviewer immediately understands the demo
- **Multi-hop questions** — comparing Apple vs Microsoft naturally tests complex retrieval

### Why `edgartools`?
It's a free Python library (`pip install edgartools`) with a clean API for navigating EDGAR's filing index. No API key needed. The SEC only requires you to identify yourself via a `User-Agent` header — `set_identity()` handles this.

### PDF vs HTM fallback
Many 10-K filings are HTML, not PDF. The script tries PDF first, falls back to the primary HTML document. Docling (our parser) handles both — so either output works.

### Key concept: why data comes first
You cannot test a parser without a document. You cannot test a chunker without parsed text. Writing code without real input produces code that looks correct but breaks on real data. Download the data first, build everything else against it.

---

## 2. Core Infrastructure Upgrades

### `app/core/qdrant.py` — Scalar Quantization

**What changed:** Added `ScalarQuantization(INT8, quantile=0.99)` to the collection creation.

**Why this matters:**
BGE-M3 produces 1024-dimensional float32 vectors. Each float32 = 4 bytes → one chunk's vector = 4 KB.

- 2,500 chunks (our corpus) = 10 MB of raw vectors
- 100,000 chunks (production) = 400 MB of raw vectors just for vectors, plus metadata

Scalar quantization compresses each float32 (4 bytes) to int8 (1 byte) — **4x memory reduction** with only ~1% accuracy loss.

`quantile=0.99` means: before compressing, clip the top 1% of extreme values. This preserves 99% of the information distribution and avoids the extremes distorting the quantization range.

`on_disk_payload=True`: the text payload stored alongside each vector (raw_text, page_num, filename) goes to disk instead of RAM. On a laptop with shared memory, this matters.

### `app/core/redis_client.py` — Semantic Cache

**What changed:** Added `cache_lookup()` and `cache_store()` functions.

**The concept — semantic caching vs exact-key caching:**

Normal (exact-key) cache: "Apple revenue FY2024?" → cache hit only if the *exact same string* is asked again.

Semantic cache: stores the BGE-M3 embedding of the query alongside the answer. On lookup, computes cosine similarity between the new query's embedding and all stored embeddings. If similarity ≥ 0.95, returns the cached answer — even for differently phrased questions.

**Example:**
- "Apple revenue FY2024?" → embedding vector A
- "How much did Apple earn in 2024?" → embedding vector B
- cosine_similarity(A, B) = 0.97 → **cache hit**, same answer returned without calling Groq

This saves Groq API calls during demos when the same question gets rephrased slightly.

**Cosine similarity formula:**
```
similarity = dot(A, B) / (|A| × |B|)
```
Returns 1.0 for identical vectors, 0.0 for completely unrelated, -1.0 for opposite.

At demo scale (<100 cached queries), a linear scan over all keys is fast enough. Production would use Redis Vector Search for sub-millisecond lookup.

### `app/core/tracing.py` — Langfuse Spans

**What changed:** Added `create_trace()` and the `span()` context manager.

**The concept — distributed tracing for RAG:**

When a query returns a wrong answer, you need to know *which pipeline step failed*. Was the retrieval returning irrelevant chunks? Did the reranker drop the right one? Did the LLM ignore the context?

Without tracing: pure guesswork.
With tracing: open Langfuse at `localhost:3000`, click the trace, see exactly this:

```
rag_query [2.7s total]
  ├─ query_rewrite    [120ms]
  ├─ dense_search     [45ms]   → 50 chunks
  ├─ bm25_search      [3ms]    → 50 chunks
  ├─ rrf_fusion       [1ms]    → 50 de-duplicated chunks
  ├─ rerank           [380ms]  → 5 chunks
  └─ generate         [2.1s]   → answer with 3 citations
```

**Usage pattern in every pipeline service:**
```python
trace = create_trace("rag_query", input={"query": q})
with span(trace, "dense_search", metadata={"top_k": 50}) as s:
    results = await qdrant.search(...)
    s.update(output={"result_count": len(results)})
```

The context manager calls `span.end()` automatically, even if an exception is thrown inside the block.

---

## 3. ORM Models — `app/models/`

### What is an ORM?

ORM = Object-Relational Mapper. It maps Python classes to database tables so you never write raw SQL in application code.

**Without ORM (raw SQL):**
```python
await db.execute("SELECT * FROM documents WHERE id = $1", doc_id)
# No type safety. Typo in column name? Runtime error.
```

**With SQLAlchemy ORM:**
```python
doc = await document_repo.get_document(db, doc_id)
doc.chunk_count  # ← typed as int, autocomplete works, typos caught at write time
```

### SQLAlchemy 2.x `Mapped` syntax

Modern SQLAlchemy uses Python type annotations to define columns:

```python
class Document(Base):
    __tablename__ = "documents"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    status: Mapped[str] = mapped_column(default="pending")
    chunk_count: Mapped[int] = mapped_column(default=0)
```

`Mapped[str]` tells both SQLAlchemy and your IDE that this column is a string. `Mapped[str | None]` means it's nullable.

### The Relationship

```python
# In Document:
chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")

# In Chunk:
document: Mapped["Document"] = relationship(back_populates="chunks")
```

`cascade="all, delete-orphan"` means: if you delete a `Document` row, SQLAlchemy automatically deletes all its `Chunk` rows. This matches the `ON DELETE CASCADE` in `init_db.sql` — consistency between ORM and SQL schema is critical.

### The `context` and `full_text` fields on Chunk

These are the heart of Contextual Retrieval:

| Field | Content | Used for |
|---|---|---|
| `raw_text` | Verbatim chunk from PDF | Shown to user as citation |
| `context` | LLM-generated 80-100 token blurb | Prepended before embedding |
| `full_text` | `context + "\n\n" + raw_text` | What gets embedded and stored in Qdrant |

You search against `full_text` (richer signal), but cite `raw_text` (verbatim from document).

### `GoldenQuestion.relevant_chunks` — Postgres ARRAY

Postgres supports array columns natively. `TEXT[]` stores a list of strings in one column — no join table needed. SQLAlchemy maps this with `ARRAY(String)`. Used in Phase 4 for RAGAS evaluation: which chunk IDs should appear in a correct answer?

---

## 4. Repositories — `app/repositories/`

### The Repository Pattern

**Rule: repositories are the only place that talks to Postgres.**

Services don't write SQLAlchemy queries. Routes don't write SQLAlchemy queries. Only repos do.

**Why this rule exists:**
Imagine 10 files each writing their own `select(Document).where(...)`. Now you need to add a soft-delete flag to documents. You have to find and update every query across 10 files.

With repositories: change `document_repo.py` once. Done.

It also makes testing clean — you mock `document_repo.get_document()`, not the database.

### `flush()` vs `commit()` — critical distinction

Every repo function calls `await db.flush()`, NOT `await db.commit()`.

| `flush()` | `commit()` |
|---|---|
| Sends SQL to Postgres | Finalizes the transaction permanently |
| Assigns generated IDs (UUIDs) | Data is visible to other connections |
| Transaction stays open | Transaction closes |
| Caller controls when to commit | No going back |

**Why this matters for the pipeline:**
The ingestion pipeline creates a Document, parses 200 chunks, enriches them, embeds them, upserts to Qdrant — then commits everything at once. If Step 5 (Qdrant) fails after Step 4 (embed), the entire transaction rolls back. No orphaned rows.

If each repo called `commit()` internally, you'd have partial commits scattered through the pipeline — impossible to roll back cleanly.

### Bulk insert in `create_chunks()`

```python
db.add_all(orm_chunks)   # queue all 2500 chunks
await db.flush()          # send all in one batch to Postgres
```

vs inserting one at a time:

```python
for chunk in chunks:
    db.add(chunk)
    await db.flush()   # 2500 separate round-trips!
```

For 2,500 chunks, bulk insert takes ~0.3 seconds. One-by-one takes ~30 seconds.

---

## 5. Pydantic Schemas — `app/schemas/`

### What Pydantic does

Pydantic validates data at system boundaries — where untrusted input enters your system (HTTP requests) or where you produce output (HTTP responses).

```python
class IngestResponse(BaseModel):
    job_id: str
    doc_id: str
    status: str = "pending"
    message: str
```

If your code returns a dict missing `job_id`, Pydantic raises a `ValidationError` immediately. You catch the bug before the client sees a broken response.

### Why separate schema files?

- `ingest.py` — async job lifecycle (submit → poll → done)
- `document.py` — document browsing and management
- `query.py` — the query pipeline (request → chunks → answer)

These three concerns grow independently. Adding streaming to the query doesn't touch ingest schemas.

### `ScoredChunk` — why it's a Pydantic model, not a dict

`ScoredChunk` travels through 4 pipeline stages: dense retriever → BM25 → RRF fusion → reranker → generator.

If it were a `dict`, each stage would guess what keys exist. Typo `"chunk_id"` vs `"chunkid"`? Runtime KeyError. As a Pydantic model, the IDE catches it immediately and every stage has autocomplete.

### `Citation.source_id` — how inline citations work

When Groq generates an answer, it writes:
> "Apple's gross margin was 45.9% [Source 1], driven by services growth [Source 2]."

`source_id` is the integer N in `[Source N]`. The `QueryResponse` includes a list of `Citation` objects indexed by `source_id`. The client uses these to link the inline reference to the full citation: filename, page number, verbatim passage.

### `IngestResponse` — one class for file and URL

The old skeleton had `IngestFileResponse` and `IngestURLResponse` as separate classes. From the client's perspective, both return the same thing: a `job_id` to poll. One class = less code, cleaner contract.

---

## 6. Ingestion Services — `app/services/ingestion/`

### `parser.py` — DocumentParser

**Docling vs PyPDF2:**

| PyPDF2 / pdfminer | Docling (IBM) |
|---|---|
| ~60% table accuracy | ~94% table accuracy |
| Scrambles multi-column text | Preserves reading order |
| Raw character dump | Structured Markdown output |
| No understanding of layout | Full document structure |

For financial 10-K reports packed with tables and multi-column sections, Docling is not optional — it's essential.

**Lazy-load pattern:**
```python
def _get_converter(self):
    if self._converter is None:
        from docling.document_converter import DocumentConverter
        self._converter = DocumentConverter()
    return self._converter
```
Docling loads heavy ML models when first called, not at import time. Server starts in <1 second; Docling initializes when the first document arrives.

**`run_in_executor` — why it's needed:**
FastAPI runs on an async event loop. One coroutine runs at a time; others wait. If you call a slow synchronous function (like Docling parsing a 100-page PDF) directly inside `async def`, you freeze the entire server for 10-30 seconds — zero other requests get served.

`await loop.run_in_executor(None, sync_fn)` offloads the sync work to a thread pool. The event loop stays responsive; other requests are served while the PDF parses in the background thread.

---

### `chunker.py` — SemanticChunker

**Why 400 characters?**
BGE-M3 has a 512-token limit. 400 characters ≈ 80-100 tokens, leaving room for the context blurb (80-100 tokens) when combined into `full_text`. This keeps everything within the embedding model's window.

**RecursiveCharacterTextSplitter — the hierarchy:**
```
Try to split on: "\n\n"  (paragraph break — best)
If too large:    "\n"    (line break)
If too large:    ". "   (sentence end)
If too large:    " "    (word boundary)
If too large:    ""     (character — last resort)
```
A chunk that cuts mid-word is worse than one that's slightly oversized. The splitter always finds the most natural boundary it can.

**64-character overlap:**
```
... end of chunk 1 ...the company reported [overlap starts]
[overlap starts] the company reported significant revenue growth... [start of chunk 2]
```
If a key fact straddles a chunk boundary, the overlap ensures it appears in at least one complete chunk. Without overlap, a sentence split in half would be missing context in both chunks.

**`char_start` / `char_end` offsets:**
`add_start_index=True` injects the character position of each chunk in the original document. These are stored in `chunks.char_start` / `chunks.char_end` and returned in citations — enabling a future frontend to highlight the exact passage.

---

### `enricher.py` — ContextualEnricher (the most important file in Phase 1)

**The problem Contextual Retrieval solves:**

Take this chunk:
> "Revenue increased 34% year-over-year."

Embed it. The vector represents: growth, revenue, year-over-year. But it doesn't know: *whose* revenue? *Which* year? *Which segment?*

A completely different company's similar sentence would produce a nearly identical vector. The wrong document would match.

**The solution — prepend a context blurb:**
```
[context blurb]
This passage is from the Revenue Overview section of Apple's FY2024 10-K 
annual report, discussing Q3 cloud services performance relative to prior year.

[raw chunk]
Revenue increased 34% year-over-year.
```

Now the embedding captures: Apple, FY2024, cloud services, Q3 — the search is specific.

**Anthropic's benchmark:** This technique reduces retrieval failure rate by **67%**. It's the single highest-ROI step in the entire pipeline.

**The Groq rate limit problem and how we solve it:**

Groq free tier: 14,400 tokens/min.
Each enrichment call: ~500 tokens (doc preview + chunk + output).
Rate limit: ~28 calls/min at full speed.

Our solution: `asyncio.Semaphore(3)` limits concurrent calls to 3.

```python
self._sem = asyncio.Semaphore(3)

async def _enrich_one(self, doc_preview, chunk):
    async with self._sem:   # only 3 coroutines can be here at once
        resp = await self._client.chat.completions.create(...)
```

`asyncio.gather(*tasks)` launches all 2,500 coroutines at once — they all exist, but only 3 hold the semaphore at any moment. The other 2,497 are parked, waiting their turn without consuming CPU or memory.

**Tenacity retry — handling 429 Too Many Requests:**

```python
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
)
```

On a 429: wait 2s, retry. If fails again: wait 4s. Then 8s. Then 16s. Max 60s. After 4 attempts: give up and log.

`wait_exponential` is called "exponential backoff" — the wait doubles each retry. This is the standard approach for rate-limited APIs: back off progressively rather than hammering the server with retries.

**Graceful degradation:**

```python
enriched = await asyncio.gather(*tasks, return_exceptions=True)
```

`return_exceptions=True` means if one enrichment fails after all retries, it returns the exception *as a value* instead of crashing the whole gather. The pipeline catches it, logs a warning, and uses `full_text = raw_text` for that chunk. A chunk without a context blurb is still searchable — just slightly less accurate than enriched chunks.

---

### `embedder.py` — BGEEmbedder

**What BGE-M3 does:**
It maps text → a 1024-dimensional vector. Texts with similar meaning produce vectors that are "close" in this 1024-dimensional space (high cosine similarity). Qdrant's HNSW graph answers "find the 50 nearest vectors to this query vector" in milliseconds.

**The Singleton pattern:**
```python
class BGEEmbedder:
    _instance: "BGEEmbedder | None" = None

    @classmethod
    def get(cls) -> "BGEEmbedder":
        if cls._instance is None:
            cls._instance = cls()   # loads model once, ~10 seconds
        return cls._instance
```

Loading BGE-M3 takes ~10 seconds and uses ~3 GB RAM. The singleton ensures this happens exactly once — during the first ingest after server startup. Every subsequent call to `BGEEmbedder.get()` returns the already-loaded model instantly.

**Why `use_fp16=False`:**
fp16 (16-bit float) is faster but Apple Silicon MPS doesn't support all fp16 operations uniformly. float32 is safer and the accuracy difference for retrieval is negligible.

**Async wrapper:**
```python
async def embed_chunks_async(self, texts, batch_size=32):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.embed_chunks, texts, batch_size)
```
Same pattern as the parser — offload CPU-bound work to a thread pool so the event loop stays free.

---

### `bm25_indexer.py` — BM25Index

**Why BM25 if we have semantic search?**

| Semantic (BGE-M3) | Keyword (BM25) |
|---|---|
| "cloud revenue" finds "cloud services income" | "NVDA Q3 2024" finds exact ticker/date |
| Understands synonyms and paraphrases | Finds exact tokens every time |
| Fails on specific identifiers (tickers, codes) | Fails on paraphrases |
| Dense vector, ANN search | Sparse scoring, linear scan |

They're **complementary**. Phase 2 combines both with Reciprocal Rank Fusion (RRF) — chunks appearing in both result lists get double credit.

**BM25 formula (simplified):**
```
score = TF × IDF × length_normalization

TF  = how often the query token appears in this chunk (term frequency)
IDF = how rare the token is across all chunks (inverse document frequency)
     → rare tokens ("NVDA") score higher than common ones ("the")
```

**Full rebuild vs incremental:**
`rank-bm25` doesn't support adding chunks incrementally — you must rebuild from scratch. For 2,500–10,000 chunks this takes <1 second. At 1M+ chunks you'd switch to Elasticsearch or OpenSearch.

**Pickle persistence:**
The BM25 index is saved to `data/bm25_index.pkl`. On server startup, `BM25Index.get()` loads it. If the file doesn't exist (first run), the index is empty — which is fine, it gets built after the first ingest.

---

### `pipeline.py` — IngestionPipeline (the orchestrator)

**The transaction design:**

```python
await document_repo.update_document_status(db, doc_id, "processing")
await db.commit()   # ← early commit so polling returns "processing"

try:
    # ... all 7 steps ...
    await db.commit()   # ← final commit: chunks in Postgres, doc = "ready"

except Exception as exc:
    await db.rollback()  # ← undo everything if any step failed
    await document_repo.update_document_status(db, doc_id, "failed", error_msg=str(exc))
    await db.commit()    # ← commit only the "failed" status
    raise
```

Two commits on success:
1. After marking `"processing"` — so the polling endpoint shows progress
2. After all 7 steps — atomically saves all chunks and marks `"ready"`

**The orphaned Qdrant point trade-off:**
If Step 5 (Qdrant upsert) succeeds but Step 7 (Postgres persist) fails, the rollback removes chunk records from Postgres — but the Qdrant vectors remain. These are "orphan" vectors that consume space and could return as ghost results.

This is an accepted trade-off at portfolio scale. Production-grade solutions use:
- **Saga pattern**: compensating transactions (explicitly delete Qdrant points on rollback)
- **Two-phase commit**: coordinate both systems together

Worth mentioning in interviews — it shows you understand the limitations.

---

## 7. Background Worker — `app/workers/ingest_worker.py`

**The session lifetime problem:**

FastAPI's `Depends(get_db)` gives a session scoped to the HTTP request. That session closes the moment the response is sent.

A `BackgroundTask` runs *after* the response is sent. If it used the request session:
```python
# ❌ WRONG — session is closed by the time background task runs
async def ingest_file(file, background_tasks, db = Depends(get_db)):
    background_tasks.add_task(pipeline.run, file_path, doc_id, file_type, db)  # db is closed!
```

The fix: create a fresh session in the background task:
```python
# ✓ CORRECT — background task owns its own session
async def run_ingestion_background(file_path, doc_id, file_type):
    async with AsyncSessionLocal() as db:   # fresh session, lives until ingestion completes
        await pipeline.run(file_path, doc_id, file_type, db)
```

---

## 8. API Routes

### `ingest.py` — why commit before queuing the background task

```python
doc = await document_repo.create_document(db, ...)
await db.commit()   # ← commit HERE, before queuing

background_tasks.add_task(run_ingestion_background, dest, doc.id, file_type)
```

If you queued the background task *before* committing, the background worker would start, open its own session, query for `doc.id`, and get nothing — because the Document row only exists in the uncommitted request transaction.

The rule: **always commit any rows that the background task will need to read before queuing the task.**

### Why `202 Accepted` instead of `200 OK`

HTTP status codes have specific meanings:
- `200 OK` — work is done, here's the result
- `202 Accepted` — I received your request and queued it; result isn't ready yet

For async jobs, `202` is the correct semantic. The client knows to poll.

### `documents.py` — the delete ordering matters

```
1. Read chunk.qdrant_id from Postgres   ← MUST happen before cascade delete
2. Delete Document (FK cascade removes chunks)
3. commit()
4. Delete Qdrant points using collected IDs
5. Rebuild BM25 from remaining chunks
```

If step 1 and step 2 were swapped, the chunk rows (and their qdrant_ids) would be gone before you could collect them. Those Qdrant vectors would become orphans forever — consuming space and returning ghost results in searches.

---

## Key Concepts Cheat Sheet

| Concept | One-liner |
|---|---|
| ORM | Python class ↔ database table. No raw SQL in application code. |
| Repository Pattern | One file = one entity's database access. Swap DB? Change one file. |
| `flush()` vs `commit()` | flush = send SQL but keep transaction open. commit = finalize permanently. |
| Singleton | Load expensive resource once; reuse everywhere. BGEEmbedder, BM25Index. |
| `run_in_executor` | Run sync CPU work in a thread pool without blocking the async event loop. |
| Scalar Quantization | Float32 → Int8. 4x smaller vectors, ~1% accuracy loss. |
| Contextual Retrieval | Prepend an LLM-generated blurb to each chunk before embedding. 67% fewer retrieval failures. |
| Semaphore | `asyncio.Semaphore(3)` — at most 3 coroutines run concurrently. Rate limit enforcement. |
| Exponential Backoff | Wait 2s, 4s, 8s, 16s... between retries. Standard pattern for rate-limited APIs. |
| Semantic Cache | Cache by embedding similarity, not by exact query string. |
| 202 Accepted | HTTP status for "request queued, poll for result." |
| BM25 | Keyword search. Complementary to semantic search — finds exact identifiers. |
| RRF (coming in Phase 2) | Reciprocal Rank Fusion — merges dense + sparse results. Chunks in both lists score higher. |
| Graceful degradation | If enrichment fails, fall back to raw text. System degrades without crashing. |

---

## What Phase 2 Adds

Phase 1 = getting data IN (ingestion).  
Phase 2 = getting answers OUT (query pipeline).

```
User query
  │
  ▼
[Query Rewriter]   → Groq: rewrite/expand the query for better retrieval
  │
  ▼
[Dense Search]     → BGE-M3 embed query → Qdrant ANN search → top-50 chunks
[BM25 Search]      → tokenize query → BM25 score → top-50 chunks  (parallel)
  │
  ▼
[RRF Fusion]       → merge both lists: chunks in both get double credit
  │
  ▼
[Reranker]         → BGE cross-encoder scores (query, chunk) jointly → top-5
  │
  ▼
[Generator]        → Groq Llama-3.3-70B: answer + [Source N] inline citations
  │
  ▼
QueryResponse      → answer + citations + trace_url
```
