import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.query import Citation, ScoredChunk

logger = get_logger(__name__)


def _make_llm_client():
    """
    VESSL-first, Groq-fallback — same priority logic as enricher.py.

    VESSL (VESSL_ENDPOINT + VESSL_TOKEN in .env):
      - OpenAI-compatible vLLM, no TPD limit, GPU billed per hour
    Groq fallback:
      - llama-3.1-8b-instant / GENERATION_MODEL, 500K TPD limit
    """
    if settings.vessl_endpoint and settings.vessl_token:
        from openai import AsyncOpenAI
        logger.info("generator_using_vessl", endpoint=settings.vessl_endpoint)
        return AsyncOpenAI(
            base_url=f"{settings.vessl_endpoint}/v1",
            api_key=settings.vessl_token,
        ), settings.vessl_model
    from groq import AsyncGroq
    return AsyncGroq(api_key=settings.groq_api_key), settings.generation_model

STREAM_SYSTEM_PROMPT = """\
You are a financial document analyst. Answer questions using ONLY the provided sources.
Cite sources inline as [Source N] immediately after each claim.
Be factual, concise, and do not speculate beyond the sources.
If no source contains the answer, say so explicitly."""

SYSTEM_PROMPT = """\
You are a financial document analyst. Answer questions using ONLY the provided sources.
Cite sources inline as [Source N] immediately after each claim.
Be factual, concise, and do not speculate beyond the sources.

You MUST respond with valid JSON in exactly this format:
{"answer": "Your answer here with [Source N] citations inline.",
 "citations": [{"source_id": 1, "cited_text": "exact verbatim quote from source"}]}

Rules:
- Every factual claim must have an inline [Source N] citation.
- cited_text must be a verbatim excerpt from the source (not a paraphrase).
- If no source contains the answer, set answer to "The provided sources do not contain enough information to answer this question." and citations to [].
"""


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation]
    model_used: str


class GroqGenerator:
    """
    Primary generator: Groq Llama-3.3-70B with structured [Source N] citation prompting.

    Why JSON output instead of free-form text:
    The model can write "[Source 1]" anywhere in prose, but we need to extract citations
    programmatically and map them back to chunk metadata (filename, page_num, score).
    Forcing JSON output gives us machine-parseable citations every time.

    Fallback: if the model returns malformed JSON (rare with explicit system prompt),
    we return the raw text as the answer with empty citations rather than crashing.
    """

    def __init__(self) -> None:
        self._client, self._model = _make_llm_client()

    async def generate(
        self, query: str, chunks: list[ScoredChunk]
    ) -> GenerationResult:
        if not chunks:
            return GenerationResult(
                answer="No relevant sources were found in the knowledge base for this query.",
                citations=[],
                model_used=self._model,
            )

        # Build numbered source blocks
        source_blocks = "\n\n".join(
            f"[Source {i + 1}]\n{chunk.full_text}"
            for i, chunk in enumerate(chunks)
        )
        user_message = f"{source_blocks}\n\nQuestion: {query}"

        logger.info(
            "groq_generate_start",
            model=self._model,
            chunks=len(chunks),
            query_len=len(query),
        )

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            temperature=0.1,  # low temperature for factual, consistent answers
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content or ""
        return self._parse_response(raw, chunks, self._model)

    def _parse_response(
        self, raw: str, chunks: list[ScoredChunk], model: str
    ) -> GenerationResult:
        """Parse the JSON response and map source_ids back to ScoredChunk metadata."""
        try:
            data = json.loads(raw)
            answer = data.get("answer", raw)
            raw_citations = data.get("citations", [])

            citations: list[Citation] = []
            for c in raw_citations:
                source_id = int(c.get("source_id", 0))
                # source_id is 1-indexed; chunk list is 0-indexed
                chunk_idx = source_id - 1
                if 0 <= chunk_idx < len(chunks):
                    chunk = chunks[chunk_idx]
                    citations.append(
                        Citation(
                            source_id=source_id,
                            chunk_id=chunk.chunk_id,
                            filename=chunk.filename,
                            page_num=chunk.page_num,
                            cited_text=c.get("cited_text", chunk.raw_text[:200]),
                            score=chunk.score,
                        )
                    )

            logger.info("groq_generate_done", answer_len=len(answer), citations=len(citations))
            return GenerationResult(answer=answer, citations=citations, model_used=model)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Model returned malformed JSON — degrade gracefully
            logger.warning("groq_parse_failed", error=str(e), raw_preview=raw[:200])
            return GenerationResult(answer=raw, citations=[], model_used=model)

    async def generate_stream(
        self, query: str, chunks: list[ScoredChunk]
    ) -> AsyncIterator[dict]:
        """
        Stream generation tokens for the SSE endpoint.

        Yields {"type": "token", "content": "..."} for each token,
        then {"type": "done", "citations": [...], "model_used": "..."}.

        No JSON response_format here — streaming and json_object mode are
        mutually exclusive in the Groq/OpenAI API. Citations are extracted
        from the assembled text via regex after streaming completes.
        """
        if not chunks:
            yield {"type": "token", "content": "No relevant sources were found in the knowledge base for this query."}
            yield {"type": "done", "citations": [], "model_used": self._model}
            return

        source_blocks = "\n\n".join(
            f"[Source {i + 1}]\n{chunk.full_text}"
            for i, chunk in enumerate(chunks)
        )
        user_message = f"{source_blocks}\n\nQuestion: {query}"

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": STREAM_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            stream=True,
            temperature=0.1,
            max_tokens=1024,
        )

        parts: list[str] = []
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                parts.append(token)
                yield {"type": "token", "content": token}

        full_answer = "".join(parts)
        citations = self._extract_stream_citations(full_answer, chunks)
        yield {
            "type": "done",
            "citations": [c.model_dump() for c in citations],
            "model_used": self._model,
        }

    def _extract_stream_citations(
        self, answer: str, chunks: list[ScoredChunk]
    ) -> list[Citation]:
        """
        Parse [Source N] markers from streamed answer text.

        Extracts unique source IDs from inline [Source N] references,
        maps them back to chunk metadata. Returns deduplicated citations
        in the order they first appear in the answer.
        """
        seen: set[int] = set()
        citations: list[Citation] = []
        for match in re.finditer(r"\[Source (\d+)\]", answer):
            source_id = int(match.group(1))
            if source_id in seen:
                continue
            seen.add(source_id)
            chunk_idx = source_id - 1
            if 0 <= chunk_idx < len(chunks):
                chunk = chunks[chunk_idx]
                # Extract a short snippet around the citation as cited_text
                pos = match.start()
                snippet_start = max(0, pos - 100)
                cited_text = answer[snippet_start:pos].strip()[-200:] or chunk.raw_text[:200]
                citations.append(
                    Citation(
                        source_id=source_id,
                        chunk_id=chunk.chunk_id,
                        filename=chunk.filename,
                        page_num=chunk.page_num,
                        cited_text=cited_text,
                        score=chunk.score,
                    )
                )
        return citations
