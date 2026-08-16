from __future__ import annotations

from pydantic import BaseModel, Field


class IngestDocument(BaseModel):
    text: str
    acl_role: str = Field(..., description="One of: Engineering, HR, Executive, General")
    source: str = ""
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument]


class IngestResponse(BaseModel):
    ingested: int
    total_in_store: int


class QueryRequest(BaseModel):
    query: str
    role: str = Field(..., description="Engineering | HR | Executive")
    top_k: int | None = None


class RetrievedChunkOut(BaseModel):
    id: str
    text: str
    acl_role: str
    source: str
    score: float


class QueryResponse(BaseModel):
    query: str
    role: str
    answer: str
    retrieved_chunks: list[RetrievedChunkOut]


class DebugTraceRequest(BaseModel):
    query: str
    role: str
    ground_truth: str | None = None
    top_k: int | None = None


class DebugTraceResponse(BaseModel):
    query: str
    role: str
    answer: str
    retrieved_chunks: list[RetrievedChunkOut]
    unfiltered_top_chunks: list[RetrievedChunkOut]
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float | None
    root_cause: str
    reasons: list[str]
    permission_blocked_doc_ids: list[str]
    permission_leaked_doc_ids: list[str]


class RegressionSummary(BaseModel):
    total_samples: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    faithfulness_gate: float
    passed: bool
    per_sample: list[dict]
