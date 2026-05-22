from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM APIs
    anthropic_api_key: str
    groq_api_key: str

    # Infrastructure
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_base"
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/ragdb"
    redis_url: str = "redis://localhost:6379"
    redis_cache_db: int = 0

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Models
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    generation_model: str = "claude-sonnet-4-6"
    generation_fallback_model: str = "llama-3.3-70b-versatile"
    context_enrichment_model: str = "llama-3.3-70b-versatile"
    eval_judge_model: str = "claude-haiku-4-5-20251001"

    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 64
    context_window_size: int = 8192

    # Retrieval
    retrieval_top_k: int = 50
    rerank_top_k: int = 5
    rrf_k: int = 60
    max_agent_retries: int = 2

    # Caching
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 3600

    # App
    app_env: str = "development"
    app_log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


settings = Settings()
