"""
Real embedding model. Uses sentence-transformers locally (downloads the
model weights from HuggingFace on first run, then runs on-device — no API
key, no mock vectors). Shared by the vector store (indexing/retrieval) and
by the Ragas evaluation engine (embedding-based metrics).
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(settings: Settings | None = None) -> SentenceTransformer:
    s = settings or get_settings()
    if s.embedding_model not in _model_cache:
        _model_cache[s.embedding_model] = SentenceTransformer(s.embedding_model)
    return _model_cache[s.embedding_model]


def embed_texts(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    model = get_embedding_model(settings)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    return embed_texts([text], settings)[0]
