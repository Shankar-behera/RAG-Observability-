from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.eval.failure_classifier import audit_permissions, classify
from app.eval.ragas_eval import EvalSample, RagasEvaluator
from app.models.schemas import DebugTraceRequest, DebugTraceResponse, RetrievedChunkOut
from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/debug", tags=["debug"])


def _chunk_out(chunks) -> list[RetrievedChunkOut]:
    return [RetrievedChunkOut(id=c.id, text=c.text, acl_role=c.acl_role, source=c.source, score=c.score) for c in chunks]


@router.post("/trace", response_model=DebugTraceResponse)
async def debug_trace(req: DebugTraceRequest) -> DebugTraceResponse:
    settings = get_settings()
    pipeline = RAGPipeline(settings)

    try:
        trace = pipeline.run(req.query, req.role, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM provider call failed: {e}")

    # If retrieval is empty (e.g. a permission failure blocked everything),
    # faithfulness/context precision are meaningless -- short-circuit straight
    # to the permission audit rather than asking Ragas to score an empty context.
    blocked, leaked = audit_permissions(trace)
    if leaked or (not trace.retrieved_chunks and blocked):
        report = classify(trace, faithfulness=0.0, context_precision=0.0, context_recall=None)
        return DebugTraceResponse(
            query=trace.query, role=trace.role, answer=trace.answer,
            retrieved_chunks=_chunk_out(trace.retrieved_chunks),
            unfiltered_top_chunks=_chunk_out(trace.unfiltered_top_chunks),
            faithfulness=0.0, answer_relevancy=0.0, context_precision=0.0, context_recall=None,
            root_cause=report.root_cause.value, reasons=report.reasons,
            permission_blocked_doc_ids=report.permission_blocked_doc_ids or [],
            permission_leaked_doc_ids=report.permission_leaked_doc_ids or [],
        )

    try:
        evaluator = RagasEvaluator(settings)
        sample = EvalSample(
            query=trace.query, answer=trace.answer, contexts=trace.context_used, ground_truth=req.ground_truth,
        )
        result = await evaluator.evaluate_sample(sample)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ragas evaluation call failed: {e}")

    report = classify(
        trace, faithfulness=result.faithfulness, context_precision=result.context_precision,
        context_recall=result.context_recall,
    )

    return DebugTraceResponse(
        query=trace.query, role=trace.role, answer=trace.answer,
        retrieved_chunks=_chunk_out(trace.retrieved_chunks),
        unfiltered_top_chunks=_chunk_out(trace.unfiltered_top_chunks),
        faithfulness=result.faithfulness, answer_relevancy=result.answer_relevancy,
        context_precision=result.context_precision, context_recall=result.context_recall,
        root_cause=report.root_cause.value, reasons=report.reasons,
        permission_blocked_doc_ids=report.permission_blocked_doc_ids or [],
        permission_leaked_doc_ids=report.permission_leaked_doc_ids or [],
    )
