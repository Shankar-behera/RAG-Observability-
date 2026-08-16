"""
The multi-tenant RAG pipeline under test.

Every query produces a QueryTrace: the full retrieval + generation record
needed downstream by both the evaluation engine and the failure-localization
dashboard. Permission auditing is done by running an *unfiltered* retrieval
in parallel (never used to build the answer) purely to detect whether ACL
filtering incorrectly blocked a document the user should have seen, or
would have leaked one they shouldn't.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.rag.embeddings import embed_query
from app.rag.llm_providers import generate_answer
from app.rag.vector_store import RetrievedChunk, VectorStoreBackend, get_vector_store

# Role -> set of ACL labels that role is allowed to read.
# Executive can read everything; Engineering/HR read their own docs + shared "General" docs.
ROLE_ACCESS: dict[str, list[str]] = {
    "Engineering": ["Engineering", "General"],
    "HR": ["HR", "General"],
    "Executive": ["Engineering", "HR", "Executive", "General"],
}


@dataclass
class QueryTrace:
    query: str
    role: str
    allowed_roles: list[str]
    retrieved_chunks: list[RetrievedChunk]
    unfiltered_top_chunks: list[RetrievedChunk]   # audit-only, never shown as "the answer's source"
    answer: str
    context_used: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, store: VectorStoreBackend | None = None):
        self.settings = settings or get_settings()
        self.store = store or get_vector_store(self.settings)

    def allowed_roles_for(self, role: str) -> list[str]:
        if role not in ROLE_ACCESS:
            raise ValueError(f"Unknown role '{role}'. Known roles: {list(ROLE_ACCESS)}")
        return ROLE_ACCESS[role]

    def run(self, query: str, role: str, top_k: int | None = None, generate: bool = True) -> QueryTrace:
        k = top_k or self.settings.top_k
        allowed = self.allowed_roles_for(role)

        q_emb = embed_query(query, self.settings)

        retrieved = self.store.query(q_emb, top_k=k, allowed_roles=allowed)
        unfiltered = self.store.query_unfiltered(q_emb, top_k=k)

        context_used = [c.text for c in retrieved]
        answer = generate_answer(query, context_used, self.settings) if generate else ""

        return QueryTrace(
            query=query,
            role=role,
            allowed_roles=allowed,
            retrieved_chunks=retrieved,
            unfiltered_top_chunks=unfiltered,
            answer=answer,
            context_used=context_used,
        )
