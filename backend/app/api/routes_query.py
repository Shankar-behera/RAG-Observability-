from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from app.api.security import require_api_access

from app.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunkOut
from app.rag.pipeline import RAGPipeline

router = APIRouter(
    prefix="/query",
    tags=["query"],
    dependencies=[Depends(require_api_access)],
    responses={
        401: {"description": "Missing or invalid API access key."},
        503: {"description": "API access is not configured."},
    },
)


@router.post("", response_model=QueryResponse)
def run_query(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    try:
        trace = pipeline.run(req.query, req.role, top_k=req.top_k)
    except ValueError as exc:
        # Preserve a safe validation message without returning arbitrary exception text.
        if str(exc).startswith("Unknown role"):
            raise HTTPException(
                status_code=400,
                detail="Unknown role.",
            ) from None
        raise HTTPException(
            status_code=400,
            detail="Invalid query request.",
        ) from None
    except Exception:
        # Keep upstream failure details out of client-visible responses.
        raise HTTPException(
            status_code=502,
            detail="LLM provider call failed.",
        ) from None

    return QueryResponse(
        query=trace.query,
        role=trace.role,
        answer=trace.answer,
        retrieved_chunks=[
            RetrievedChunkOut(id=c.id, text=c.text, acl_role=c.acl_role, source=c.source, score=c.score)
            for c in trace.retrieved_chunks
        ],
    )
