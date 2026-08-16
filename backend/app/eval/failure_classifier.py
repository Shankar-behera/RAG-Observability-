"""
Root-cause failure classification.

Given a QueryTrace (real retrieval + generation output) plus the Ragas
scores for that sample, decide which stage of the pipeline is responsible
for a failing response:

  PERMISSION_FAILURE - ACL filtering blocked a document the role should
                        see, OR would have leaked one it shouldn't, based
                        on comparing the filtered vs. unfiltered top-k.
  RETRIEVER_FAILURE   - the relevant chunks were never retrieved (low
                        context precision/recall) even though they exist
                        somewhere in the corpus for this role.
  GENERATOR_FAILURE   - relevant chunks WERE retrieved (good context
                        precision) but the LLM ignored them or hallucinated
                        (low faithfulness).
  PASS                - faithfulness and context metrics both clear the bar.

This is pure decision logic over real scores — it does not itself call an
LLM or fabricate scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.rag.pipeline import QueryTrace


class RootCause(str, Enum):
    PASS = "pass"
    RETRIEVER_FAILURE = "retriever_failure"
    GENERATOR_FAILURE = "generator_failure"
    PERMISSION_FAILURE = "permission_failure"


@dataclass
class FailureReport:
    root_cause: RootCause
    reasons: list[str]
    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    permission_blocked_doc_ids: list[str] | None = None
    permission_leaked_doc_ids: list[str] | None = None


def audit_permissions(trace: QueryTrace) -> tuple[list[str], list[str]]:
    """Compare filtered retrieval against the unfiltered top-k (audit-only)
    to find:
      - blocked: docs that appear in the unfiltered top-k, ARE within the
        user's allowed roles, but got dropped from the filtered results
        (a permission bug that's over-restricting).
      - leaked: docs that appear in the FILTERED results but whose acl_role
        is NOT in the user's allowed roles (a permission bug that's
        under-restricting -- the serious one).
    """
    filtered_ids = {c.id for c in trace.retrieved_chunks}
    allowed = set(trace.allowed_roles)

    leaked = [c.id for c in trace.retrieved_chunks if c.acl_role not in allowed]

    blocked = [
        c.id for c in trace.unfiltered_top_chunks
        if c.acl_role in allowed and c.id not in filtered_ids
    ]
    return blocked, leaked


def classify(
    trace: QueryTrace,
    faithfulness: float,
    context_precision: float,
    context_recall: float | None = None,
    faithfulness_threshold: float = 0.7,
    context_precision_threshold: float = 0.5,
) -> FailureReport:
    reasons: list[str] = []

    blocked, leaked = audit_permissions(trace)
    if leaked:
        reasons.append(
            f"{len(leaked)} retrieved chunk(s) belong to an ACL role the '{trace.role}' "
            f"user is not permitted to read: {leaked}"
        )
        return FailureReport(
            root_cause=RootCause.PERMISSION_FAILURE, reasons=reasons,
            faithfulness=faithfulness, context_precision=context_precision, context_recall=context_recall,
            permission_blocked_doc_ids=blocked, permission_leaked_doc_ids=leaked,
        )
    if blocked:
        reasons.append(
            f"{len(blocked)} chunk(s) the '{trace.role}' user IS permitted to read were "
            f"present in the unfiltered top-k but missing from the ACL-filtered results: {blocked}"
        )
        return FailureReport(
            root_cause=RootCause.PERMISSION_FAILURE, reasons=reasons,
            faithfulness=faithfulness, context_precision=context_precision, context_recall=context_recall,
            permission_blocked_doc_ids=blocked, permission_leaked_doc_ids=leaked,
        )

    retriever_ok = context_precision >= context_precision_threshold
    generator_ok = faithfulness >= faithfulness_threshold

    if retriever_ok and generator_ok:
        reasons.append("Faithfulness and context precision both clear their thresholds.")
        return FailureReport(
            root_cause=RootCause.PASS, reasons=reasons,
            faithfulness=faithfulness, context_precision=context_precision, context_recall=context_recall,
        )

    if not retriever_ok:
        reasons.append(
            f"Context precision {context_precision:.2f} is below threshold "
            f"{context_precision_threshold:.2f} — the retriever did not surface relevant chunks."
        )
        return FailureReport(
            root_cause=RootCause.RETRIEVER_FAILURE, reasons=reasons,
            faithfulness=faithfulness, context_precision=context_precision, context_recall=context_recall,
        )

    reasons.append(
        f"Context precision {context_precision:.2f} is healthy but faithfulness "
        f"{faithfulness:.2f} is below threshold {faithfulness_threshold:.2f} — the retriever "
        f"did its job; the LLM ignored or contradicted the retrieved context."
    )
    return FailureReport(
        root_cause=RootCause.GENERATOR_FAILURE, reasons=reasons,
        faithfulness=faithfulness, context_precision=context_precision, context_recall=context_recall,
    )
