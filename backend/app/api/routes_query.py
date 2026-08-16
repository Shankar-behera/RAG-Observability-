from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunkOut
from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def run_query(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    try:
        trace = pipeline.run(req.query, req.role, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Real upstream failures (LLM provider unreachable, auth error, etc.)
        # surface as a 502 rather than a silently-generated fake answer.
        raise HTTPException(status_code=502, detail=f"LLM provider call failed: {e}")

    return QueryResponse(
        query=trace.query,
        role=trace.role,
        answer=trace.answer,
        retrieved_chunks=[
            RetrievedChunkOut(id=c.id, text=c.text, acl_role=c.acl_role, source=c.source, score=c.score)
            for c in trace.retrieved_chunks
        ],
    )
