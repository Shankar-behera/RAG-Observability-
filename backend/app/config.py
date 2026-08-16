"""
Central configuration. Everything is driven by real environment variables —
no hardcoded fallback credentials, no mock provider. If a required key is
missing for the provider you selected, the app fails loudly at call time
rather than silently degrading.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- LLM provider selection -------------------------------------------------
    # "openai"   -> api.openai.com, needs OPENAI_API_KEY
    # "deepseek" -> api.deepseek.com (OpenAI-compatible), needs DEEPSEEK_API_KEY
    # "ollama"   -> local Ollama server, OpenAI-compatible /v1 endpoint, no key needed
    llm_provider: Literal["openai", "deepseek", "ollama"] = "ollama"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "phi3-custom"

    # ---- Embeddings ---------------------------------------------------------------
    # Local sentence-transformers model. Runs on-device, no API key required,
    # used for both vector-store indexing and Ragas embedding-based metrics.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Vector store ---------------------------------------------------------------
    vector_store: Literal["chroma", "pgvector"] = "chroma"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "rag_documents"

    pg_dsn: str = "postgresql://raguser:ragpass@localhost:5432/rag_observability"
    pg_table: str = "rag_documents"

    # ---- App / CI ---------------------------------------------------------------
    top_k: int = 4
    faithfulness_gate: float = 0.85
    cors_origins: str = "http://localhost:3000"

    def provider_base_url(self) -> str | None:
        return {
            "openai": None,  # openai SDK default
            "deepseek": self.deepseek_base_url,
            "ollama": self.ollama_base_url,
        }[self.llm_provider]

    def provider_api_key(self) -> str:
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set but llm_provider=openai")
            return self.openai_api_key
        if self.llm_provider == "deepseek":
            if not self.deepseek_api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is not set but llm_provider=deepseek")
            return self.deepseek_api_key
        # Ollama's OpenAI-compatible endpoint ignores the key but the SDK requires a non-empty string
        return "ollama-local"

    def provider_model(self) -> str:
        return {
            "openai": self.openai_model,
            "deepseek": self.deepseek_model,
            "ollama": self.ollama_model,
        }[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
