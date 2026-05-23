from app.core.config import settings
from app.schemas.query import ScoredChunk
from app.services.generation.groq_gen import GenerationResult, GroqGenerator
from app.services.generation.longcite import LongCiteGenerator

_groq = GroqGenerator()


async def generate(query: str, chunks: list[ScoredChunk]) -> GenerationResult:
    """
    Pick between the Groq (free, always-on) and LongCite (demo, VESSL on-demand) path.

    Swapping generators requires only setting/unsetting LONGCITE_ENDPOINT in .env —
    no code changes. This is the "Strategy pattern": same interface, different backends.

    Default: Groq llama-3.3-70b-versatile ($0, 14,400 tokens/min free tier)
    Demo:    LongCite-8B on VESSL A10 (native sentence citations, ~$1.52/2hr session)
    """
    if settings.longcite_endpoint:
        return await LongCiteGenerator().generate(query, chunks)
    return await _groq.generate(query, chunks)
