import asyncio

from groq import AsyncGroq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.services.ingestion.chunker import ChunkData

logger = get_logger(__name__)

# First 1000 chars covers the document title, company name, fiscal year — enough
# context for the model to write a useful "where this chunk sits" blurb.
# Reduced from 6000 to cut token usage per call by ~65% (fits Groq free tier).
DOC_PREVIEW_CHARS = 1_000

SYSTEM_PROMPT = "You are a document understanding assistant. Be concise and factual."

USER_TEMPLATE = """\
<document_overview>
{doc_preview}
</document_overview>

<chunk>
{chunk_text}
</chunk>

In 2-3 sentences (80-100 tokens), describe where this chunk appears in the document \
and what broader context is needed to understand it. \
Output only the description — no preamble, no labels."""


def _make_client():
    """
    Returns an OpenAI-compatible async client.

    Priority:
      1. VESSL endpoint (set VESSL_ENDPOINT + VESSL_TOKEN in .env) — no rate limits,
         GPU billed per hour. Use for bulk ingestion.
      2. Groq free tier — rate-limited (500K TPD on llama-3.1-8b-instant).
         Falls back automatically when VESSL_ENDPOINT is not set.
    """
    if settings.vessl_endpoint and settings.vessl_token:
        from openai import AsyncOpenAI
        logger.info("enricher_using_vessl", endpoint=settings.vessl_endpoint)
        return AsyncOpenAI(
            base_url=f"{settings.vessl_endpoint}/v1",
            api_key=settings.vessl_token,
        )
    return AsyncGroq(api_key=settings.groq_api_key)


class ContextualEnricher:
    """
    Implements Anthropic's Contextual Retrieval pattern.

    Uses VESSL (OpenAI-compatible vLLM endpoint) when VESSL_ENDPOINT is set,
    otherwise falls back to Groq free tier (llama-3.1-8b-instant, 500K TPD).

    Semaphore(3) limits concurrency; tenacity handles 429s with exponential
    backoff so the pipeline recovers automatically from rate limit hits.
    """

    def __init__(self) -> None:
        self._client = _make_client()
        self._sem = asyncio.Semaphore(3)

    async def enrich_chunks(
        self, full_doc: str, chunks: list[ChunkData]
    ) -> list[ChunkData]:
        doc_preview = full_doc[:DOC_PREVIEW_CHARS]
        tasks = [self._enrich_one(doc_preview, chunk) for chunk in chunks]
        enriched = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[ChunkData] = []
        for chunk, result in zip(chunks, enriched):
            if isinstance(result, Exception):
                # Enrichment failed after all retries — use raw text only.
                # A chunk with no context blurb is still searchable; it just
                # relies on the raw text embedding instead of the enriched one.
                logger.warning(
                    "enrichment_failed",
                    chunk_index=chunk.chunk_index,
                    error=str(result),
                )
                chunk.context = ""
                chunk.full_text = chunk.raw_text
                results.append(chunk)
            else:
                results.append(result)  # type: ignore[arg-type]

        enriched_count = sum(1 for c in results if c.context)
        logger.info(
            "enrichment_complete",
            total=len(results),
            enriched=enriched_count,
            skipped=len(results) - enriched_count,
        )
        return results

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _enrich_one(self, doc_preview: str, chunk: ChunkData) -> ChunkData:
        model = (
            settings.vessl_model
            if (settings.vessl_endpoint and settings.vessl_token)
            else settings.context_enrichment_model
        )
        async with self._sem:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            doc_preview=doc_preview,
                            chunk_text=chunk.raw_text,
                        ),
                    },
                ],
                max_tokens=150,
                temperature=0.0,
            )
        blurb = resp.choices[0].message.content.strip()
        chunk.context = blurb
        chunk.full_text = f"{blurb}\n\n{chunk.raw_text}"
        return chunk
