"""
Real LLM provider wiring.

OpenAI, DeepSeek, and Ollama all expose an OpenAI-compatible /v1 chat
completions API, so a single `openai.OpenAI` client configured with the
right `base_url` + `api_key` talks to any of them. There is no mock or
stub client here — swapping LLM_PROVIDER in .env is the only thing that
changes which real network endpoint gets called.
"""
from __future__ import annotations

from openai import OpenAI

from app.config import Settings, get_settings

_client_cache: dict[str, OpenAI] = {}


def get_llm_client(settings: Settings | None = None) -> OpenAI:
    s = settings or get_settings()
    cache_key = f"{s.llm_provider}:{s.provider_base_url()}"
    if cache_key not in _client_cache:
        _client_cache[cache_key] = OpenAI(base_url=s.provider_base_url(), api_key=s.provider_api_key())
    return _client_cache[cache_key]


def generate_answer(query: str, context_chunks: list[str], settings: Settings | None = None) -> str:
    """Call the real configured LLM with retrieved context and return the generated answer.

    Raises whatever the underlying SDK raises (connection error, auth error, etc.) —
    callers are expected to surface that, not swallow it into a fake response.
    """
    s = settings or get_settings()
    client = get_llm_client(s)

    context_block = "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context_chunks))
    system_prompt = (
        "You are a retrieval-augmented assistant. Answer the user's question using ONLY "
        "the information in the provided context chunks. If the context does not contain "
        "the answer, say you don't have enough information — do not use outside knowledge."
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"

    resp = client.chat.completions.create(
        model=s.provider_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""
