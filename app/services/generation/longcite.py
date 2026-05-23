"""
LongCite-llama3.1-8B generator — demo path.

LongCite (THUDM, MIT license) is a model fine-tuned specifically for sentence-level
citations. Unlike prompt-engineering citations into a general model, LongCite has
citation behavior baked into its weights — it natively outputs citation spans.

This generator is ONLY active when settings.longcite_endpoint is set (a VESSL A10
instance running the LongCite inference server). Without the env var, the router
falls back to GroqGenerator automatically.

VESSL cost: ~$1.52 per 2-hour on-demand session. Never always-on.

The LongCite inference API follows vLLM's OpenAI-compatible format, so we can reuse
the same chat completion pattern as Groq — just pointed at the VESSL endpoint.
"""

import json

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.query import Citation, ScoredChunk
from app.services.generation.groq_gen import GenerationResult

logger = get_logger(__name__)

# LongCite's output format includes citation spans like:
# {"answer": "Revenue was $89.5B [S1].", "statements": [{"cite": "S1", "text": "..."}]}
# We normalise this to our GenerationResult / Citation schema.


class LongCiteGenerator:
    """
    Calls a VESSL-hosted LongCite-8B inference endpoint (vLLM / OpenAI-compatible).
    Falls back gracefully to returning the raw answer if parsing fails.
    """

    async def generate(
        self, query: str, chunks: list[ScoredChunk]
    ) -> GenerationResult:
        if not settings.longcite_endpoint:
            raise RuntimeError("LONGCITE_ENDPOINT is not configured")

        if not chunks:
            return GenerationResult(
                answer="No relevant sources were found in the knowledge base for this query.",
                citations=[],
                model_used=settings.generation_demo_model,
            )

        source_blocks = "\n\n".join(
            f"[Source {i + 1}]\n{chunk.full_text}"
            for i, chunk in enumerate(chunks)
        )
        user_message = f"{source_blocks}\n\nQuestion: {query}"

        logger.info(
            "longcite_generate_start",
            endpoint=settings.longcite_endpoint,
            chunks=len(chunks),
        )

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.longcite_endpoint}/v1/chat/completions",
                json={
                    "model": settings.generation_demo_model,
                    "messages": [{"role": "user", "content": user_message}],
                    "max_tokens": 1024,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data["choices"][0]["message"]["content"]
        return self._parse_longcite_output(raw, chunks)

    def _parse_longcite_output(
        self, raw: str, chunks: list[ScoredChunk]
    ) -> GenerationResult:
        """
        LongCite returns JSON with answer + statements array.
        We map statement citation refs back to our ScoredChunk list.
        """
        try:
            parsed = json.loads(raw)
            answer = parsed.get("answer", raw)
            statements = parsed.get("statements", [])

            citations: list[Citation] = []
            for i, stmt in enumerate(statements):
                cite_ref = stmt.get("cite", "")
                # cite_ref is like "S1", "S2" — extract the number
                try:
                    source_id = int(cite_ref.lstrip("S"))
                    chunk_idx = source_id - 1
                    if 0 <= chunk_idx < len(chunks):
                        chunk = chunks[chunk_idx]
                        citations.append(
                            Citation(
                                source_id=source_id,
                                chunk_id=chunk.chunk_id,
                                filename=chunk.filename,
                                page_num=chunk.page_num,
                                cited_text=stmt.get("text", "")[:400],
                                score=chunk.score,
                            )
                        )
                except (ValueError, IndexError):
                    continue

            logger.info(
                "longcite_generate_done",
                answer_len=len(answer),
                citations=len(citations),
            )
            return GenerationResult(
                answer=answer,
                citations=citations,
                model_used=settings.generation_demo_model,
            )

        except (json.JSONDecodeError, KeyError):
            logger.warning("longcite_parse_failed", raw_preview=raw[:200])
            return GenerationResult(
                answer=raw,
                citations=[],
                model_used=settings.generation_demo_model,
            )
