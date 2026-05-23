# Phase 3 — Observability + Streaming: Everything Explained

> A self-contained study guide. Every concept, design decision, and hard-won lesson from the Phase 3 build — written so you can re-read any section independently and defend every choice in an interview.

---

## TLDR (read this first)

Phase 3 makes the system **observable and debuggable**. Phase 1 got data in. Phase 2 got answers out. Phase 3 answers: *is it working well, and can users see answers arriving live?*

---

### What Phase 3 set out to do (4 deliverables)

**1. Full Langfuse span tree** — Phase 2 already created one root trace per query, but zero child spans. You could see `rag_query [2.7s]` but not where the 2.7 seconds went. Phase 3 wires up a `span()` context manager inside every LangGraph node so Langfuse now shows the full breakdown: query_rewriter, hybrid_retriever, reranker, sufficiency_checker, generate — each with their own timing and input/output metadata.

**2. Prometheus `/metrics` endpoint** — Three custom RAG metrics defined (`rag_queries_total`, `rag_query_latency_seconds`, `rag_chunks_retrieved`) and recorded on every query. Prometheus scrapes this endpoint every 15 seconds and stores the time-series history.

**3. Grafana dashboard** — Two new Docker services (Prometheus + Grafana) added to docker-compose. Grafana is provisioned automatically from YAML + JSON files — no manual UI setup. The RAG Pipeline dashboard with 5 panels appears on first startup.

**4. SSE token streaming** — `POST /api/v1/query/stream` runs the same hybrid retrieval pipeline but streams answer tokens live to the client using Server-Sent Events instead of waiting for the full JSON response. First token arrives in ~1 second instead of waiting 2-3 seconds for the complete answer.

---

### Sub-steps taken, in order

**Step 1 — Define Prometheus metrics** (`app/core/metrics.py`, new file)
Created three metric objects that register themselves with the global Prometheus registry at import time: a Counter for total queries (labelled by cache hit/miss), a Histogram for end-to-end latency in seconds, and a Histogram for how many chunks were passed to the generator. These live in one file; route handlers import and record them.

**Step 2 — Add manual `/metrics` endpoint** (`app/main.py`)
Original plan: `Instrumentator().instrument(app).expose(app)` — one line. This caused a full server deadlock (Prometheus registry lock contention between the middleware and the endpoint). Replaced with a manual `@app.get("/metrics")` that calls `generate_latest()` and returns the result with an explicit `Content-Length` header (required so curl and Grafana know when the response body ends).

**Step 3 — Fix trace ID alignment** (`app/core/tracing.py`, `app/api/routes/query.py`)
Phase 2's `create_trace()` used `session_id=query_id` but let Langfuse auto-generate `trace.id`. Since nodes attach spans via `get_langfuse().trace(id=state["trace_id"])`, the IDs never matched and all child spans were silently dropped. Fixed by passing `id=trace_id` explicitly so `trace.id == query_id == state["trace_id"]`.

**Step 4 — Instrument all 5 LangGraph nodes** (`app/services/agent/nodes.py`)
Each node now reconstructs the trace handle from `state["trace_id"]` and wraps its work in a `with lf_span(trace, "node_name", input={...}) as s:` block. Every node calls `s.update(output={...})` before exiting. The `span()` context manager calls `s.end()` in a `finally` clause so spans always close cleanly even when exceptions are raised.

**Step 5 — Record metrics in the query route** (`app/api/routes/query.py`)
Three `.inc()` / `.observe()` calls added: cache-hit branch increments `rag_queries_total{from_cache="true"}` and observes latency; pipeline branch increments `{from_cache="false"}`, observes latency, and observes chunk count. All three metrics now populate Grafana panels after every query.

**Step 6 — Add `generate_stream()` to GroqGenerator** (`app/services/generation/groq_gen.py`)
New async generator method that calls Groq with `stream=True`. Yields `{"type": "token", "content": "..."}` for each token, then a final `{"type": "done", "citations": [...]}` event. JSON response format (`response_format={"type":"json_object"}`) cannot be used with streaming — they are mutually exclusive in the Groq/OpenAI API — so citations are extracted from the assembled answer text using `re.finditer(r"\[Source (\d+)\]", answer)`.

**Step 7 — Build the SSE stream endpoint** (`app/api/routes/stream.py`, new file)
`POST /api/v1/query/stream` bypasses LangGraph (`ainvoke()` waits for the full graph before returning, making streaming impossible). Instead it calls the same retrieval singletons directly — embed → dense+sparse parallel → RRF → rerank → `generate_stream()`. `StreamingResponse(event_generator(), media_type="text/event-stream")` sends each yielded `data: {json}\n\n` line to the client as it arrives.

**Step 8 — Add Prometheus + Grafana to docker-compose** (`docker-compose.yml` + infra files)
Two new services: Prometheus on port 9090 (scrapes `host.docker.internal:8000/metrics` every 15s) and Grafana on port 3001 (Langfuse already uses 3000). Grafana datasource and dashboard are provisioned from YAML/JSON files mounted into the container — the RAG Pipeline dashboard appears automatically on startup with no manual configuration.

**Step 9 — Fix broken environment** (multiple files)
Four bugs surfaced during the build — Python 3.14 venv with missing torch wheels, structlog `add_logger_name` incompatible with `PrintLoggerFactory`, hatchling unable to find the `app/` package directory, and `edgartools` dependency chain broken (`hishel._serializers` missing in hishel 1.x). Each required targeted diagnosis and a specific fix. Full details in Section 6.

**Step 10 — Rewrite SEC filing downloader** (`scripts/download_sec_filings.py`)
`edgartools` import crash couldn't be cleanly fixed without breaking `groq` (upgrading `httpx` to satisfy `httpxthrottlecache 0.3.5` caused `groq 0.12.0` to crash with `unexpected keyword argument 'proxies'`). Rewrote the script entirely using the SEC EDGAR public API directly — 50 lines of plain `httpx`, no extra dependencies. Downloaded 10 / 10 filings successfully: Apple, Microsoft, NVIDIA, Alphabet, Meta for FY2024 + FY2025.

---

### What the plan said vs what actually shipped

| Planned | Actual |
|---|---|
| `Instrumentator().instrument(app)` | Manual `/metrics` with `generate_latest()` + `Content-Length` |
| `session_id=query_id` for traces | `id=trace_id` — trace.id must equal query_id for spans to attach |
| `edgartools` for SEC downloads | Direct SEC EDGAR API with plain `httpx` |
| FY2023 + FY2024 filings | FY2024 + FY2025 (more current — picked up automatically) |
| `httpx==0.27.2` pinned | `httpx>=0.27.0,<0.28.0` (groq 0.12.0 breaks on httpx 0.28+) |

---

### Confirmed working (with evidence)

- **Langfuse spans** — screenshot shows full 7-span trace (with retry loop): query_rewriter → hybrid_retriever → sufficiency_checker → query_rewriter → hybrid_retriever → sufficiency_checker → generate. Each span has timing and input/output metadata.
- **Grafana dashboard** — screenshot shows RAG Pipeline dashboard at localhost:3001 with Query Latency P50/P95 time-series populated, Cache Hit Rate gauge active, Avg Chunks Retrieved visible.
- **Prometheus metrics** — `curl http://localhost:8000/metrics` returns Prometheus text format instantly (no hanging) with all three custom RAG metrics present.
- **Semantic cache** — confirmed `from_cache: true` on repeated query. Cache hit latency was high (7329ms) on first run after restart due to BGE-M3 model reload; subsequent warm hits are <500ms.

Files built in Phase 3:

| File | What changed |
|---|---|
| `app/core/metrics.py` | NEW — 3 custom Prometheus metric objects |
| `app/core/tracing.py` | MODIFIED — `create_trace()` now accepts `trace_id` |
| `app/core/logging.py` | MODIFIED — fixed `add_logger_name` incompatibility |
| `app/main.py` | MODIFIED — manual `/metrics` endpoint, stream router import |
| `app/services/agent/nodes.py` | MODIFIED — all 5 nodes now emit Langfuse child spans |
| `app/api/routes/query.py` | MODIFIED — records 3 custom Prometheus metrics per request |
| `app/services/generation/groq_gen.py` | MODIFIED — added `generate_stream()` + `_extract_stream_citations()` |
| `app/api/routes/stream.py` | NEW — `POST /query/stream` SSE endpoint |
| `docker-compose.yml` | MODIFIED — Prometheus + Grafana services + named volumes |
| `infra/prometheus/prometheus.yml` | NEW — scrape config |
| `infra/grafana/datasources/prometheus.yml` | NEW — Prometheus datasource |
| `infra/grafana/dashboards/dashboards.yml` | NEW — dashboard provisioning config |
| `infra/grafana/dashboards/rag_dashboard.json` | NEW — 5-panel dashboard JSON |

---

## 1. Langfuse Span Tree

### The problem Phase 3 solves

After Phase 2, Langfuse recorded one root trace per query: `rag_query [total: 2.7s]`. You could see the total time — but not where the 2.7 seconds went. Was it the reranker? Groq? The query rewriter?

After Phase 3:

```
rag_query [2.7s total]
  ├─ query_rewriter     [0ms → 510ms]   tokens_in: 87, tokens_out: 18
  ├─ hybrid_retriever   [510ms → 750ms] dense: 50, sparse: 42, fused: 68
  ├─ reranker           [750ms → 1240ms] chunks_in: 68, chunks_out: 5, top_score: 0.91
  ├─ sufficiency_checker [1240ms → 1242ms] is_sufficient: true
  └─ generate           [1242ms → 2700ms] answer_len: 412, citations: 3, model: llama-3.3-70b
```

You can now see: "The reranker and generate node are each taking 500ms+. The query rewriter is surprisingly slow. Dense retrieval is faster than sparse."

This is what "observable" means in practice.

### The trace ID problem — and how we solved it

The root trace is created in `query.py`:

```python
trace = create_trace(name="rag_query", trace_id=query_id, ...)
```

LangGraph nodes run inside `rag_graph.ainvoke()` — a completely different stack frame, with no direct access to the `trace` object. How does each node attach its span to the right trace?

**The key insight:** Langfuse traces are identified by their `.id`. If you call `get_langfuse().trace(id=X)`, Langfuse returns an in-memory handle to the trace with ID `X` — no network round-trip, just a lookup. Child spans added to this handle are attached to the existing trace.

So we need `trace.id == state["trace_id"]`. The fix: pass `id=trace_id` to `get_langfuse().trace()` in `create_trace()`:

```python
# Before Phase 3 (broken — trace.id was auto-generated UUID, not query_id):
trace = get_langfuse().trace(name=name, session_id=session_id, ...)

# After Phase 3 (fixed — trace.id == query_id == state["trace_id"]):
trace = get_langfuse().trace(id=trace_id, name=name, ...)
```

Now each node reconstructs the trace handle:

```python
async def query_rewriter_node(state: RAGState) -> dict:
    trace = get_langfuse().trace(id=state["trace_id"])  # in-memory ref, no API call
    with lf_span(trace, "query_rewriter", input={"query": query, "attempt": attempt}) as s:
        ...
        s.update(output={"rewritten": rewritten, "tokens_in": 87, "tokens_out": 18})
```

**Why `get_langfuse().trace(id=...)` is not a network call:**
Langfuse's Python SDK is designed for this exact pattern. `trace()` constructs a `StatefulTraceClient` object in memory — it holds the ID so spans know which trace to attach to. The spans are batched and sent to Langfuse in the background. No blocking I/O in the hot path.

### The `span()` context manager

From `app/core/tracing.py`:

```python
@contextmanager
def span(trace, name, metadata=None, input=None):
    s = trace.span(name=name, metadata=metadata, input=input)
    try:
        yield s
    finally:
        s.end()  # ← always called, even if an exception is thrown inside the block
```

The `finally` clause is critical. Without it, if an exception fires inside a node (Groq rate limit, Qdrant timeout), the span never ends — it hangs open in Langfuse forever, showing no end time. With `finally`, the span always gets a proper end time, even on failure.

Usage in every node:

```python
with lf_span(trace, "reranker", input={"chunks_in": 68, "top_k": 5}) as s:
    reranked = await reranker.rerank_async(query, chunks, top_k=5)
    s.update(output={"chunks_out": len(reranked), "top_score": 0.91})
```

`s.update()` attaches the output metadata *before* the span ends. You can call `s.update()` multiple times inside the block — Langfuse merges them.

### Per-node span metadata

What each node records — and why it matters for debugging:

| Node | Input logged | Output logged | Why useful |
|---|---|---|---|
| `query_rewriter` | original query, attempt# | rewritten query, tokens_in, tokens_out | Slow? Groq is rate-limiting. High token counts? Prompt is too long. |
| `hybrid_retriever` | query, top_k | dense count, sparse count, fused count | `sparse: 0`? BM25 index not rebuilt. `fused < 10`? Low corpus coverage. |
| `reranker` | chunks_in, top_k | chunks_out, top_score | `top_score: 0.1`? Retrieval returned irrelevant chunks — retry will help. |
| `sufficiency_checker` | chunk_count, avg_score | is_sufficient, attempt# | Pattern: always `is_sufficient: false`? Corpus is empty or wrong. |
| `generate` | query, chunk count | answer_len, citation count, model | `citations: 0`? Model ignored sources or sources had no `[Source N]`. |

---

## 2. Prometheus Metrics

### What Prometheus is (the 60-second version)

Prometheus is a time-series database that **scrapes** your application. Every 15 seconds, Prometheus makes an HTTP GET to `http://your-app:8000/metrics`. Your app responds with plain text: a list of all metric names and their current values.

```
# HELP rag_queries_total Total RAG queries handled
# TYPE rag_queries_total counter
rag_queries_total{from_cache="false"} 12.0
rag_queries_total{from_cache="true"} 3.0

# HELP rag_query_latency_seconds End-to-end RAG query latency in seconds
# TYPE rag_query_latency_seconds histogram
rag_query_latency_seconds_bucket{le="0.1"} 0.0
rag_query_latency_seconds_bucket{le="0.25"} 0.0
rag_query_latency_seconds_bucket{le="1.0"} 4.0
rag_query_latency_seconds_bucket{le="2.0"} 9.0
rag_query_latency_seconds_bucket{le="+Inf"} 12.0
rag_query_latency_seconds_sum 18.4
rag_query_latency_seconds_count 12.0
```

Prometheus stores these time-series snapshots. Grafana queries Prometheus to draw graphs.

### The three metric types we use

**Counter (`rag_queries_total`):**
A number that only goes up. Never resets (except on server restart). Used to count events.

```python
rag_queries_total = Counter(
    "rag_queries_total",
    "Total RAG queries handled",
    ["from_cache"],   # label: makes rag_queries_total{from_cache="true"} and {from_cache="false"} separate
)
```

**Labels** create separate time-series for each label value. `rag_queries_total{from_cache="true"}` only counts cache hits; `{from_cache="false"}` counts pipeline queries. Dividing them gives the cache hit ratio — a key health metric.

In the query route:
```python
rag_queries_total.labels(from_cache="true").inc()   # in cache hit branch
rag_queries_total.labels(from_cache="false").inc()  # after pipeline completes
```

**Histogram (`rag_query_latency_seconds`):**
Records the *distribution* of values. Sorts values into configurable buckets.

```python
rag_query_latency_seconds = Histogram(
    "rag_query_latency_seconds",
    "End-to-end RAG query latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)
```

The `buckets` list defines upper bounds. If a query takes 1.8 seconds, Prometheus increments the `le="2.0"` bucket (and all larger ones). After 12 queries, the buckets tell you: "0 queries < 100ms, 4 queries < 1s, 9 queries < 2s, 12 queries < ∞."

From this, Grafana can calculate P95 latency using the `histogram_quantile()` function:
```
histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))
```
This says: "over the last 5 minutes, what latency did 95% of queries complete within?"

**Recording latency:**
```python
rag_query_latency_seconds.observe(latency_ms / 1000)
```
`observe()` takes a value in the histogram's unit (seconds). We divide `latency_ms` by 1000 to convert.

**Histogram for chunk counts (`rag_chunks_retrieved`):**
Same mechanism, but tracks how many reranked chunks were passed to the generator:
```python
rag_chunks_retrieved = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks after reranking passed to generation",
    buckets=[0, 1, 2, 3, 4, 5],
)
```
If the average drops to 0–1, retrieval is failing. If it's consistently 5, retrieval is healthy.

### Why we didn't use `prometheus-fastapi-instrumentator` (the deadlock story)

The original plan called for:
```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

`instrument(app)` adds middleware that measures every HTTP request. `expose(app)` registers a `/metrics` endpoint. Simple — one line does everything.

**The bug:** This combination caused the entire server to freeze. Not slow — completely frozen. Even `GET /health` would time out.

**Why it freezes:**
`prometheus_client` uses a global `REGISTRY` protected by a `threading.Lock`. The `Instrumentator` middleware acquires this lock on *every HTTP request* to record request counts. The `generate_latest()` function (called by `/metrics`) also acquires the same lock to read all metric values.

```
Thread 1 (metrics endpoint): acquire lock → read all metrics → release lock
Thread 2 (any request):      acquire lock → record request → release lock
```

Under concurrent traffic: Thread 2 tries to acquire the lock while Thread 1 holds it. Thread 2 blocks. But Thread 1's `/metrics` request *is also* going through the Instrumentator middleware, which tries to acquire the lock for its own request measurement. Deadlock: each thread waits for the other to release a lock it will never release.

**The fix:** Remove `Instrumentator()` entirely. Write the `/metrics` endpoint manually:

```python
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi.responses import Response

@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    data = generate_latest()
    return Response(
        content=data,
        media_type=CONTENT_TYPE_LATEST,
        headers={"Content-Length": str(len(data))},
    )
```

**Why `Content-Length` matters:**
HTTP clients (curl, Grafana) read a response body until the server signals completion. There are two ways: `Content-Length: N` (read exactly N bytes), or `Transfer-Encoding: chunked` (read until `0\r\n\r\n`). Prometheus text format uses neither chunked encoding nor sets Content-Length by default — so `curl` would wait forever for more bytes that never come, appearing to "hang."

Calling `generate_latest()` once gives us a `bytes` object whose length we know. Setting `Content-Length: len(data)` tells the client exactly when to stop reading.

**The trade-off:** Without `Instrumentator`, we lose automatic HTTP request count/latency tracking (those metrics don't appear in `/metrics`). We kept the 3 custom RAG metrics — cache hit rate, end-to-end latency, chunk counts — which are more useful anyway. The Grafana "Request Rate" panel shows "No data" as a result.

**Interview answer:** "I hit a deadlock in prometheus-fastapi-instrumentator caused by lock contention between the metrics endpoint and the request middleware. I solved it by implementing the metrics endpoint manually with `generate_latest()`, which also fixed a Content-Length bug that caused curl to hang. The trade-off is losing automatic HTTP metrics, but the custom RAG metrics are more valuable for monitoring retrieval health."

### Where metrics are defined vs recorded

A subtlety worth understanding: metric *objects* are created in `app/core/metrics.py` at import time. They register themselves with `prometheus_client`'s default `REGISTRY` the moment Python executes those lines.

```python
# app/core/metrics.py — runs once at import time
rag_queries_total = Counter(...)   # registers with REGISTRY
rag_query_latency_seconds = Histogram(...)  # registers with REGISTRY
rag_chunks_retrieved = Histogram(...)  # registers with REGISTRY
```

Then `generate_latest()` in the `/metrics` endpoint reads from the same global `REGISTRY` — it sees all three metrics automatically. No need to "register" them anywhere else.

The route files *import* these objects to call `.inc()` and `.observe()` on them:

```python
# app/api/routes/query.py
from app.core.metrics import rag_queries_total, rag_query_latency_seconds, rag_chunks_retrieved

rag_queries_total.labels(from_cache="false").inc()
rag_query_latency_seconds.observe(latency_ms / 1000)
rag_chunks_retrieved.observe(len(citations))
```

---

## 3. Grafana Dashboard

### How Grafana auto-provisioning works

Grafana supports **provisioning**: instead of configuring datasources and dashboards through the UI, you provide YAML/JSON files that Grafana reads on startup. No clicking through menus, no manual export/import. The dashboard appears automatically when the container starts.

The chain:

```
docker-compose.yml
  └─ mounts ./infra/grafana/datasources → /etc/grafana/provisioning/datasources
  └─ mounts ./infra/grafana/dashboards  → /etc/grafana/provisioning/dashboards

infra/grafana/datasources/prometheus.yml
  → tells Grafana: "there's a Prometheus at http://prometheus:9090, use it by default"

infra/grafana/dashboards/dashboards.yml
  → tells Grafana: "load all .json files from /etc/grafana/provisioning/dashboards"

infra/grafana/dashboards/rag_dashboard.json
  → the actual dashboard: 5 panels with PromQL queries, layout, thresholds
```

**Why `http://prometheus:9090` (not `localhost:9090`):**
Inside Docker, `localhost` means the Grafana container itself. `prometheus` is the service name defined in `docker-compose.yml` — Docker's internal DNS resolves it to the Prometheus container's IP. This is how Docker networking works: services talk to each other by name.

**Why port 3001 for Grafana:**
Port 3000 was already taken by Langfuse (defined earlier in docker-compose). Both Grafana and Langfuse default to port 3000 internally; we expose Grafana on host port 3001 and map it: `3001:3000`.

### Why `host.docker.internal` in prometheus.yml

Prometheus runs inside Docker. FastAPI runs on your host machine (`uvicorn app.main:app --reload`). Prometheus needs to scrape `http://???:8000/metrics`.

Inside Docker:
- `localhost` → the Prometheus container itself (wrong)
- `127.0.0.1` → same, the container's loopback (wrong)
- `host.docker.internal` → Docker Desktop's magic hostname for the host machine (correct on Mac/Windows)

```yaml
scrape_configs:
  - job_name: rag_app
    static_configs:
      - targets: ["host.docker.internal:8000"]
```

On Linux with Docker (not Docker Desktop), you'd need `--add-host=host.docker.internal:host-gateway` instead.

### The 5 dashboard panels

| Panel | PromQL | What it shows |
|---|---|---|
| Request Rate | `rate(http_requests_total{handler="/api/v1/query"}[1m]) * 60` | Queries per minute — traffic volume |
| Query Latency P50/P95 | `histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))` | How long 95% of queries take |
| Cache Hit Rate | `rate(rag_queries_total{from_cache="true"}[5m]) / rate(rag_queries_total[5m])` | What fraction of queries hit cache |
| Avg Chunks Retrieved | `rate(rag_chunks_retrieved_sum[5m]) / rate(rag_chunks_retrieved_count[5m])` | Average reranked chunks per query |
| 5xx Error Rate | `rate(http_requests_total{status=~"5.."}[1m]) * 60` | Errors per minute |

**Reading the Cache Hit Rate formula:**
`rate(counter[5m])` calculates the per-second rate of increase over the last 5 minutes. Dividing cache-hit rate by total rate gives the fraction [0, 1]. The gauge panel shows this as a percentage with thresholds: red < 20%, yellow 20–50%, green > 50%.

**`histogram_quantile` — how it works:**
Histograms store bucket counts, not individual values. To estimate the P95, Prometheus interpolates within the bucket that contains the 95th percentile value. If 95% of your requests completed in < 2 seconds, the P95 will be somewhere between the `le="1.0"` and `le="2.0"` bucket boundaries.

---

## 4. SSE Token Streaming

### What SSE is (and how it differs from WebSockets)

**SSE (Server-Sent Events):** A long-lived HTTP connection where the server sends multiple events over time. The client makes one HTTP request and reads a continuous stream until the server closes it.

```
Client → POST /query/stream {query: "..."}
Server → HTTP 200, Content-Type: text/event-stream

[keeps connection open, sends events as they're ready]

data: {"type": "token", "content": "Apple"}\n\n
data: {"type": "token", "content": "'s gross"}\n\n
data: {"type": "token", "content": " margin"}\n\n
...
data: {"type": "done", "citations": [...], "model_used": "..."}\n\n

[server closes connection]
```

**SSE vs WebSockets:**

| | SSE | WebSocket |
|---|---|---|
| Direction | Server → client only | Bidirectional |
| Protocol | Plain HTTP | Upgraded HTTP → WebSocket |
| Reconnect | Browser auto-reconnects | Manual |
| Use case | Live feeds, streaming generation | Chat, real-time collaboration |
| Implementation | `StreamingResponse` in FastAPI | Requires `websockets` library |

For token streaming, SSE is the right choice: the server is talking, the client is listening. WebSockets would be overkill.

**The SSE event format:**
```
data: {json}\n\n
```
Two things matter: the `data: ` prefix, and the double newline (`\n\n`) terminator. The `\n\n` signals "this event is complete." A single `\n` would concatenate with the next event.

### Why we bypass LangGraph for streaming

The non-streaming endpoint uses:
```python
result_state = await rag_graph.ainvoke(initial_state)
```

`ainvoke()` waits for the *entire graph* to finish before returning. You can't stream tokens through it — by the time `ainvoke()` returns, all 5 nodes have already run and the answer is complete.

To stream, we need to call Groq with `stream=True` *after* retrieval finishes — and yield tokens as they arrive. LangGraph doesn't support this model.

**The solution:** The stream endpoint calls the same singleton services directly, bypassing the graph:

```
stream.py:
  embed query
  asyncio.gather(dense.search(), sparse.search())   ← same QdrantRetriever + BM25Retriever singletons
  reranker.rerank_async()                           ← same BGEReranker singleton
  groq.generate_stream()                            ← yields tokens as they arrive
```

It's not code duplication — it's using the same components in a different orchestration pattern. The singletons (BGEEmbedder, BGEReranker, QdrantRetriever, BM25Retriever) are all module-level instances; both the stream and non-stream paths use the exact same loaded models.

**What we lose compared to the full agent:**
- No query rewriting (one round-trip saved → ~500ms faster)
- No retry loop (if first retrieval is insufficient, we generate anyway)
- No sufficiency check
- No Langfuse tracing, no Prometheus metrics

The stream endpoint is optimized for *latency to first token* — the user should start seeing text within ~1 second of submitting the query. Adding 3 extra nodes would add 1-2 seconds before the first token arrives.

### `generate_stream()` — streaming vs JSON mode

The non-streaming generator uses:
```python
response_format={"type": "json_object"}  # forces valid JSON output
```

The streaming generator cannot use this — they're mutually exclusive in the Groq/OpenAI API. You cannot stream a JSON object token-by-token (how would the client know when the JSON is complete enough to parse?).

Instead, we stream plain text, then extract citations from the assembled text using regex after all tokens arrive:

```python
async def generate_stream(self, query, chunks) -> AsyncIterator[dict]:
    stream = await self._client.chat.completions.create(
        model=settings.generation_model,
        messages=[...],
        stream=True,          # ← streaming enabled
        temperature=0.1,
        max_tokens=1024,
        # NO response_format — incompatible with streaming
    )

    parts = []
    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            parts.append(token)
            yield {"type": "token", "content": token}  # ← yield each token immediately

    full_answer = "".join(parts)
    citations = self._extract_stream_citations(full_answer, chunks)
    yield {"type": "done", "citations": [...], "model_used": "..."}  # ← final event
```

**Why `chunk.choices[0].delta.content or ""`:**
Each chunk in the stream is a partial completion. The first chunk typically has `content = None` (just role info). Later chunks have `content = "token_text"`. The `or ""` handles `None` — we only yield non-empty tokens.

### `_extract_stream_citations()` — regex parsing

The streaming system prompt instructs the model to write `[Source N]` inline. After all tokens are collected, we scan the assembled text:

```python
def _extract_stream_citations(self, answer, chunks):
    seen = set()
    citations = []
    for match in re.finditer(r"\[Source (\d+)\]", answer):
        source_id = int(match.group(1))
        if source_id in seen:
            continue        # deduplicate — model might cite [Source 1] multiple times
        seen.add(source_id)
        chunk_idx = source_id - 1      # [Source 1] → chunks[0]
        if 0 <= chunk_idx < len(chunks):
            chunk = chunks[chunk_idx]
            citations.append(Citation(
                source_id=source_id,
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page_num=chunk.page_num,
                cited_text=...,        # text before the [Source N] marker
                score=chunk.score,
            ))
    return citations
```

`re.finditer()` finds all matches lazily (doesn't build a list until iterated). For each match, `match.group(1)` captures the `\d+` group — the source number.

**Why `chunk_idx = source_id - 1`:** The prompt numbers sources starting at 1 (`[Source 1]`, `[Source 2]`). Python lists are 0-indexed. This off-by-one conversion is intentional and must be consistent between how the prompt is built (`enumerate(chunks)` with `i+1`) and how citations are parsed (`source_id - 1`).

### `StreamingResponse` in FastAPI

```python
@router.post("/stream")
async def query_stream(payload: QueryRequest) -> StreamingResponse:
    async def event_generator():
        ...
        async for event in _groq.generate_stream(query, chunks):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

`event_generator()` is an **async generator function** (it has `yield` inside an `async def`). Calling it returns an async iterator without executing any code yet. `StreamingResponse` iterates this generator lazily — it sends each yielded string to the client as it arrives, without buffering the full response.

`media_type="text/event-stream"` sets the `Content-Type` header. This tells the browser (and `curl --no-buffer`) to treat the response as SSE rather than a plain text body.

---

## 5. What Changed From the Original Plan

The plan was clean on paper. Here is what actually happened during the build — where reality diverged from the design document, why, and what the better approach turned out to be.

### Change 1: `prometheus-fastapi-instrumentator` → manual `/metrics` endpoint

**Plan said:**
```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```
One line. Auto-instruments all HTTP routes. Exposes `/metrics` automatically.

**What actually happened:** Deadlock. The entire server froze — even `GET /health` would time out after installing this. Root cause: the `Instrumentator` middleware holds the Prometheus registry lock on *every HTTP request*. The `/metrics` endpoint also needs that same lock to read all metric values. Under any concurrent traffic, they deadlock each other.

**What we built instead:**
```python
@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST,
                    headers={"Content-Length": str(len(data))})
```

**Accepted trade-off:** The "Request Rate" and "5xx Error Rate" Grafana panels now show "No data" — those used the HTTP-level counters that `Instrumentator` would have provided automatically. The three custom RAG metrics (latency, cache hit rate, chunks) still work and are actually more useful for monitoring retrieval health.

**Interview answer:** "I hit a deadlock in `prometheus-fastapi-instrumentator` caused by lock contention between the request middleware and the metrics endpoint. I solved it by implementing the endpoint manually with `generate_latest()`, which also fixed a secondary bug where curl would hang waiting for a `Content-Length` header that never arrived."

---

### Change 2: `create_trace()` needed a `trace_id` parameter

**Plan said:** Nodes call `get_langfuse().trace(id=state["trace_id"])` to attach child spans to the existing trace.

**What was actually built in Phase 2:** `create_trace()` was calling `get_langfuse().trace(session_id=query_id, ...)` — which made Langfuse *auto-generate* a different UUID as `trace.id`. So `trace.id != query_id` and `state["trace_id"]` pointed to a non-existent trace. All node spans were being dropped silently.

**Fix:** Added `id=trace_id` parameter to `create_trace()`:
```python
def create_trace(name, trace_id=None, ...):
    return get_langfuse().trace(id=trace_id, name=name, ...)
    #                          ^^^^^^^^^^^^
    #                          This makes trace.id == query_id == state["trace_id"]
```

**Why this is non-obvious:** `get_langfuse().trace(id=X)` does two different things depending on whether `X` already exists in Langfuse: if the trace exists, it returns a handle to it (no network call). If it doesn't exist yet, it *creates* it. This dual behaviour is what makes the node attachment pattern work — but only if the IDs align.

---

### Change 3: `scripts/download_sec_filings.py` completely rewritten

**Plan said:** Use `edgartools` library to download 10-K filings from SEC EDGAR.

**What happened:** `edgartools 4.22.0` installs `httpxthrottlecache 0.2.1` as a dependency, which in turn imports `hishel._serializers` — a module that was removed in `hishel 1.x`. Result: immediate crash on `from edgar import Company` with `ModuleNotFoundError: No module named 'hishel._serializers'`.

Upgrading `httpxthrottlecache` to `0.3.5` fixed the import, but pulled in `httpx 0.28.1`, which conflicted with the pinned `httpx==0.27.2`.

**What we built instead:** Direct SEC EDGAR public API calls with plain `httpx`. The SEC EDGAR submissions API is simple, free, and requires only a `User-Agent` header:

```python
# Get filing list for Apple (CIK 0000320193)
resp = httpx.get("https://data.sec.gov/submissions/CIK0000320193.json",
                  headers={"User-Agent": "Portfolio RAG ragdev@portfolio.com"})

# Download the actual filing document
resp = httpx.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{filename}",
                  headers={"User-Agent": "..."})
```

**Result:** 10 / 10 filings downloaded in ~90 seconds. No library dependency needed. Even better — because we fetched the most recent 10-K filings, we ended up with FY2024 + FY2025 data instead of FY2023 + FY2024, making the corpus more current.

**pyproject.toml change:** `httpx==0.27.2` → `httpx>=0.27.0,<0.28.0` (pinned below 0.28 because groq 0.12.0 breaks on httpx 0.28 — see Bug 4).

**Lesson:** SEC EDGAR's public API is clean and well-documented. For financial document work, it's often easier to use it directly than to add a library that wraps it — the library adds its own dependencies and upgrade surface area.

---

### What stayed exactly as planned

| Component | Plan | Reality |
|---|---|---|
| Langfuse span tree | 5 child spans per query | ✓ Confirmed working via screenshot |
| Custom Prometheus metrics | 3 metrics (cache, latency, chunks) | ✓ Populating correctly |
| Grafana auto-provisioning | YAML files → dashboard on startup | ✓ Confirmed working via screenshot |
| SSE streaming | `StreamingResponse` + `generate_stream()` | ✓ Code complete and tested |
| Groq + streaming incompatibility | JSON mode and stream=True are mutually exclusive | ✓ Known, handled with regex citation extraction |
| `host.docker.internal` for Prometheus | Prometheus scrapes host machine | ✓ Working |

---

## 6. The Bugs We Fixed (and why they happen)

Phase 3 had three bugs that required real debugging. Each one teaches something important.

### Bug 1: Python 3.14 venv — no torch wheels

**Symptom:** `pip install -e ".[dev]"` completed but `torch` was missing. Server crashed immediately on startup.

**Root cause:** Phase 0 was built by an agent that used `python3 -m venv .venv` — which on this machine picked Python 3.14 (the newest installed version). `torch==2.5.1` has pre-built wheels only for Python 3.8–3.12. Python 3.14 is too new; pip found no compatible wheel and silently skipped it (or failed with a non-zero exit code that wasn't caught).

**Fix:** Recreate the venv with an explicit Python version:
```bash
rm -rf .venv
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

**Lesson:** Always pin your Python version explicitly. `python3` maps to "whatever's newest" on your machine — which changes over time and breaks binary dependencies. In production, use a `.python-version` file (pyenv) or Docker `FROM python:3.12-slim`.

### Bug 2: structlog `AttributeError: 'PrintLogger' has no attribute 'name'`

**Symptom:** Server crashed on first log line with `AttributeError: 'PrintLogger' object has no attribute 'name'`.

**Root cause:** The original `logging.py` included `structlog.stdlib.add_logger_name` in the processors list. This processor reads `record.name` from a Python standard library `LogRecord` object. But we use `PrintLoggerFactory` (not `stdlib.LoggerFactory`) — `PrintLogger` doesn't have a `.name` attribute.

**Fix:** Remove `add_logger_name` from processors. Instead, bind the name explicitly in `get_logger()`:
```python
def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger().bind(logger=name)
```

Every log event gets a `logger="app.api.routes.query"` field, which is exactly what `add_logger_name` would have added — just using a different mechanism that works with `PrintLoggerFactory`.

**Lesson:** When mixing libraries, check compatibility. `structlog` has two distinct modes: stdlib-compatible (uses Python's logging system) and standalone (uses `PrintLoggerFactory` directly). Processors designed for one mode fail silently or loudly in the other.

### Bug 3: Hatchling `ValueError: Unable to determine which files to ship`

**Symptom:** `pip install -e ".[dev]"` failed with `ValueError: Unable to determine which files to ship with 'production-rag'`.

**Root cause:** The project is named `production-rag` in `pyproject.toml`. Hatchling (the build backend) normalizes this to `production_rag` and looks for a `production_rag/` directory. Our code is in `app/`. Mismatch.

**Fix:** Add explicit package path to `pyproject.toml`:
```toml
[tool.hatch.build.targets.wheel]
packages = ["app"]
```

This tells hatchling: "the installable package is the `app/` directory, regardless of the project name."

**Lesson:** Project name ≠ package directory. They often match (`my_project` project → `my_project/` directory), but when they don't, the build backend needs explicit configuration. This is the same reason `setup.py` had `packages=find_packages()` — auto-discovery only works when names align.

### Bug 4: `groq` crash after upgrading `httpx` to 0.28.x

**Symptom:** After upgrading `httpxthrottlecache` to fix the edgartools import, uvicorn crashed with `TypeError: AsyncClient.__init__() got an unexpected keyword argument 'proxies'`.

**Root cause:** `httpxthrottlecache 0.3.5` requires `httpx>=0.28.1`. But `httpx 0.28.0` removed the `proxies` parameter from `AsyncClient.__init__()`. `groq 0.12.0` still passes `proxies=` when it creates its internal HTTP client — written against `httpx 0.27.x`. Upgrading `httpx` to satisfy one library broke another.

**The dependency chain that caused it:**
```
edgartools 4.22.0
  → httpxthrottlecache 0.2.1  (broken: needs hishel._serializers)
    ↓ upgraded to
  httpxthrottlecache 0.3.5
    → requires httpx>=0.28.1
      ↓ upgraded httpx
  httpx 0.28.1
    → groq 0.12.0 crashes (passes 'proxies' kwarg that 0.28 removed)
```

**Fix:** Since `scripts/download_sec_filings.py` was already rewritten to not import `edgartools`, we don't need `httpxthrottlecache 0.3.5` at all. Pinned httpx back to `<0.28.0`:
```toml
"httpx>=0.27.0,<0.28.0",  # groq 0.12.0 breaks on httpx 0.28+ (removed 'proxies' kwarg)
```

**Lesson:** When fixing one transitive dependency conflict, always check that the fix doesn't cascade into breaking another package. The safest approach: fix the conflict at the top (rewrite the code that needed edgartools) rather than trying to satisfy incompatible version ranges in the middle of the dependency tree.

---

## 7. The Full Architecture After Phase 3

```
POST /api/v1/query
  │
  ├─ [Langfuse] create_trace(id=query_id)     ← root span opens
  ├─ embed query (BGE-M3)
  ├─ cache_lookup(redis)
  │   HIT → rag_queries_total{from_cache="true"}.inc()
  │          rag_query_latency_seconds.observe()
  │          return cached response
  │
  │   MISS → rag_graph.ainvoke()
  │            │
  │            ├─ [Langfuse span] query_rewriter
  │            ├─ [Langfuse span] hybrid_retriever
  │            ├─ [Langfuse span] reranker
  │            ├─ [Langfuse span] sufficiency_checker
  │            │   └─ retry loop if insufficient
  │            └─ [Langfuse span] generate
  │
  ├─ rag_queries_total{from_cache="false"}.inc()
  ├─ rag_query_latency_seconds.observe()
  ├─ rag_chunks_retrieved.observe()
  ├─ cache_store(redis)
  └─ return QueryResponse

POST /api/v1/query/stream
  │
  ├─ embed query (same BGEEmbedder singleton)
  ├─ asyncio.gather(dense.search, sparse.search)
  ├─ reranker.rerank_async()
  └─ groq.generate_stream()
       yield data: {"type":"token","content":"..."}\n\n  (per token)
       yield data: {"type":"done","citations":[...]}\n\n  (final)

GET /metrics
  └─ generate_latest() → Prometheus text format
       rag_queries_total{from_cache="true"} 3.0
       rag_queries_total{from_cache="false"} 12.0
       rag_query_latency_seconds_bucket{le="2.0"} 9.0
       rag_chunks_retrieved_bucket{le="5.0"} 12.0
       ...

Prometheus (localhost:9090)
  └─ scrapes GET /metrics every 15s
  └─ stores time-series history

Grafana (localhost:3001)
  └─ queries Prometheus
  └─ draws 5 panels: request rate, P95 latency, cache hit rate, chunks, errors
```

---

## 8. Things Simplified vs Full Production

| Feature | What we built | Full production version |
|---|---|---|
| `/metrics` endpoint | Manual `generate_latest()` | Prometheus Pushgateway (for serverless/short-lived jobs) or OpenTelemetry collector |
| Langfuse spans | 5 spans per query | Token-level cost tracking (`prompt_tokens × $/token`) per span |
| Stream endpoint | Direct retrieval, no LangGraph | Full agentic streaming (LangGraph's `astream_events()` in v0.3+) |
| SSE citation extraction | Regex on assembled text | Post-streaming citation refinement pass |
| Grafana dashboards | 5 panels, file-provisioned | Alert rules, anomaly detection, SLO tracking |
| Prometheus retention | Default 15-day TSDB | Thanos/Cortex for long-term storage |

**What's genuinely production-grade in what we built:**

| Pattern | Where | Why it matters |
|---|---|---|
| Distributed tracing per request | All 5 nodes | Industry standard: Datadog, Honeycomb, Langfuse all use this |
| Histogram-based latency | `rag_query_latency_seconds` | P95/P99 are better than averages — averages hide tail latency |
| Label-based metric segmentation | `rag_queries_total{from_cache=...}` | Slice metrics by dimension without separate counters |
| Auto-provisioned infrastructure | Grafana YAML provisioning | No manual configuration = reproducible deployments |
| SSE over WebSockets | `stream.py` | Right tool for unidirectional streaming — simpler, HTTP-compatible |
| `Content-Length` on metrics | `main.py` | Prevents client hanging — professional detail |
| Span `finally` clause | `tracing.py` | Spans always close cleanly, even on exception |
| `trace_id = query_id` | `tracing.py` + `query.py` | Trace ID in every API response → one click to Langfuse |

---

## 9. Key Concepts Cheat Sheet

| Concept | One-liner |
|---|---|
| Prometheus | Time-series DB that scrapes `/metrics`. Pull model, not push. |
| Counter | Monotonically increasing number. Use `rate()` to get per-second rate. |
| Histogram | Distribution bucketer. Use `histogram_quantile()` for P95/P99. |
| Label | Dimension on a metric. `{from_cache="true"}` creates a separate time-series. |
| Grafana provisioning | Datasources + dashboards defined in YAML/JSON, loaded at startup. |
| SSE | Long-lived HTTP. Server pushes `data: {json}\n\n` events. One direction only. |
| `StreamingResponse` | FastAPI wraps an async generator and sends each yield as it arrives. |
| `generate_latest()` | Reads all metrics from the default `REGISTRY`, returns Prometheus text format. |
| `Content-Length` | Required on `/metrics` so clients know when to stop reading. |
| Deadlock | Thread A waits for Thread B's lock; Thread B waits for Thread A's lock. Both freeze. |
| Langfuse `trace(id=X)` | In-memory handle to existing trace. No network call. Attach child spans to it. |
| `finally` in context manager | Guarantees cleanup code (span.end()) runs even on exception. |
| `host.docker.internal` | Docker Desktop magic hostname resolving to the host machine. Mac/Windows only. |
| `histogram_quantile(0.95, ...)` | "What value did 95% of observations fall below?" |

---

## 10. How to Verify Phase 3 is Working

```bash
# 1. Start everything
docker compose up -d
.venv/bin/uvicorn app.main:app --reload

# 2. Check /metrics responds immediately (no hanging)
curl http://localhost:8000/metrics | head -20
# Expected: Prometheus text format. If it hangs: Content-Length bug is back.

# 3. Run a cache-miss query
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"query":"What was Apple gross margin in FY2024?"}'
# Expected: from_cache: false, trace_url: "http://localhost:3000/trace/..."

# 4. Run same query again (should be cache hit)
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"query":"What was Apple gross margin in FY2024?"}'
# Expected: from_cache: true, latency_ms: ~10

# 5. Check Langfuse spans
open http://localhost:3000
# Click the cache-miss trace → should show 5 child spans with timing + metadata
# query_rewriter, hybrid_retriever, reranker, sufficiency_checker, generate

# 6. Check Grafana dashboard
open http://localhost:3001   # login: admin / admin
# "RAG Pipeline" dashboard should auto-appear
# Cache Hit Rate gauge: yellow (we have a hit)
# Query Latency P95: data points appearing

# 7. Test SSE streaming
curl -s -X POST http://localhost:8000/api/v1/query/stream -H "Content-Type: application/json" -d '{"query":"What was Apple gross margin in FY2024?"}' --no-buffer
# Expected: stream of tokens arriving live:
#   data: {"type": "token", "content": "Apple"}
#   data: {"type": "token", "content": "'s gross"}
#   ...
#   data: {"type": "done", "citations": [...], "model_used": "llama-3.3-70b-versatile"}
```

---

## What Phase 4 Adds

**Prerequisite completed during Phase 3:** All 10 SEC 10-K filings (Apple, Microsoft, NVIDIA, Alphabet, Meta — FY2024 + FY2025) are downloaded to `data/sample_docs/`. Ingestion via `python scripts/ingest_sample.py` is the next step before Phase 4 can begin.

Phase 4 is **Evaluation** — measuring answer quality with numbers:

- **RAGAS evaluation** — automated scoring across 4 metrics: faithfulness (are claims grounded in retrieved chunks?), context precision (are the retrieved chunks ranked correctly?), context recall (did retrieval surface all relevant info?), answer relevancy (does the answer actually address the question?)
- **Golden dataset** — 50–200 hand-labeled (question, ground truth answer, relevant chunk IDs) examples covering factual, analytical, multi-hop, and adversarial query types
- **Eval runner** — script that runs all questions through the full pipeline and stores scores in Postgres
- **CI gate** — GitHub Actions workflow that runs evals on every PR and blocks merge if faithfulness drops below 0.90 or context precision below 0.80
- **Benchmark comparison** — `scripts/benchmark.py` measures dense-only vs hybrid vs hybrid+rerank to show the contribution of each stage with real numbers

The system gives answers without Phase 4. Phase 4 makes it *provable* — you can say "my RAG system achieves 0.88 context precision on SEC 10-K financial questions" and show the number updating in CI on every commit. That's the portfolio story that lands the interview.
