from __future__ import annotations

from app.eval.failure_classifier import RootCause, classify
from app.rag.pipeline import QueryTrace
from app.rag.vector_store import RetrievedChunk


def _chunk(id: str, role: str, text: str = "chunk text") -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, acl_role=role, source="s", score=0.9)


def _trace(role, allowed, retrieved, unfiltered) -> QueryTrace:
    return QueryTrace(
        query="q", role=role, allowed_roles=allowed,
        retrieved_chunks=retrieved, unfiltered_top_chunks=unfiltered,
        answer="a", context_used=[c.text for c in retrieved],
    )


def test_pass_when_scores_clear_thresholds():
    t = _trace("Engineering", ["Engineering", "General"], [_chunk("eng1", "Engineering")], [_chunk("eng1", "Engineering")])
    r = classify(t, faithfulness=0.9, context_precision=0.8)
    assert r.root_cause == RootCause.PASS


def test_retriever_failure_on_low_context_precision():
    t = _trace("Engineering", ["Engineering", "General"], [_chunk("eng1", "Engineering")], [_chunk("eng1", "Engineering")])
    r = classify(t, faithfulness=0.9, context_precision=0.2)
    assert r.root_cause == RootCause.RETRIEVER_FAILURE


def test_generator_failure_on_low_faithfulness_with_good_retrieval():
    t = _trace("Engineering", ["Engineering", "General"], [_chunk("eng1", "Engineering")], [_chunk("eng1", "Engineering")])
    r = classify(t, faithfulness=0.3, context_precision=0.9)
    assert r.root_cause == RootCause.GENERATOR_FAILURE


def test_permission_failure_on_leaked_document():
    """An HR-only doc appearing in an Engineering user's filtered results is
    a permission leak and must be classified as such regardless of scores."""
    t = _trace("Engineering", ["Engineering", "General"], [_chunk("hr1", "HR")], [_chunk("hr1", "HR")])
    r = classify(t, faithfulness=0.95, context_precision=0.95)
    assert r.root_cause == RootCause.PERMISSION_FAILURE
    assert "hr1" in r.permission_leaked_doc_ids


def test_permission_failure_on_blocked_document():
    """A doc the role IS allowed to see, present in the unfiltered top-k but
    missing from filtered results, is an over-restrictive ACL bug."""
    t = _trace("Engineering", ["Engineering", "General"], [], [_chunk("eng1", "Engineering")])
    r = classify(t, faithfulness=0.0, context_precision=0.0)
    assert r.root_cause == RootCause.PERMISSION_FAILURE
    assert "eng1" in r.permission_blocked_doc_ids


def test_permission_failure_takes_priority_over_generator_failure():
    """Even if faithfulness is also bad, a leaked document must be reported
    as a permission failure, not masked as a generator failure."""
    t = _trace("Engineering", ["Engineering", "General"], [_chunk("hr1", "HR")], [_chunk("hr1", "HR")])
    r = classify(t, faithfulness=0.1, context_precision=0.1)
    assert r.root_cause == RootCause.PERMISSION_FAILURE
