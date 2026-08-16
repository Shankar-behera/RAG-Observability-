from __future__ import annotations

import ssl
import certifi

# --- START SSL PATCH ---
# Force Python to use certifi instead of the corrupted Windows Certificate Store
def custom_ssl_context(*args, **kwargs):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(certifi.where())
    return ctx

ssl.create_default_context = custom_ssl_context
# --- END SSL PATCH ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_debug, routes_ingest, routes_query, routes_regression
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="RAG Observability Platform",
    description="Full-stack observability for RAG pipelines: offline/online eval, failure localization, CI regression gating.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_ingest.router)
app.include_router(routes_query.router)
app.include_router(routes_debug.router)
app.include_router(routes_regression.router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "vector_store": settings.vector_store,
    }
