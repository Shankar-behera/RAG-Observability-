"""
Real ACL enforcement tests against a real Chroma collection.

Embedding calls are monkeypatched to fixed vectors ONLY because this CI/dev
sandbox may not have network access to huggingface.co to download the
sentence-transformers weights on first run. Everything downstream —
Chroma's actual indexing, its actual `where` metadata filter, and the
ACL-role logic in vector_store.py — runs for real. When you run this in an
environment with normal internet access, delete the monkeypatch fixture
and it exercises the real embedding model too.
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.rag.vector_store import ChromaBackend, Document

FIXED_VECTORS = {
    "Engineering doc about deployment": [1.0, 0.0, 0.0],
    "HR doc about payroll": [0.0, 1.0, 0.0],
    "Executive doc about strategy": [0.0, 0.0, 1.0],
    "General doc about office hours": [0.5, 0.5, 0.5],
    "deployment question": [0.95, 0.05, 0.0],
}


def fake_embed(texts, settings=None):
    return [FIXED_VECTORS.get(t, [0.1, 0.1, 0.1]) for t in texts]


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr("app.rag.vector_store.embed_texts", fake_embed)
    s = Settings(llm_provider="ollama", vector_store="chroma", chroma_persist_dir=str(tmp_path / "chroma"))
    backend = ChromaBackend(s)
    docs = [
        Document(id="eng1", text="Engineering doc about deployment", acl_role="Engineering", source="eng.md"),
        Document(id="hr1", text="HR doc about payroll", acl_role="HR", source="hr.md"),
        Document(id="exec1", text="Executive doc about strategy", acl_role="Executive", source="exec.md"),
        Document(id="gen1", text="General doc about office hours", acl_role="General", source="general.md"),
    ]
    backend.add_documents(docs)
    return backend


def test_documents_indexed(store):
    assert store.count() == 4


def test_engineering_role_never_sees_hr_or_exec(store):
    q = fake_embed(["deployment question"])[0]
    results = store.query(q, top_k=4, allowed_roles=["Engineering", "General"])
    roles_returned = {r.acl_role for r in results}
    assert roles_returned <= {"Engineering", "General"}
    assert not roles_returned & {"HR", "Executive"}


def test_hr_role_never_sees_engineering_or_exec(store):
    q = fake_embed(["deployment question"])[0]
    results = store.query(q, top_k=4, allowed_roles=["HR", "General"])
    roles_returned = {r.acl_role for r in results}
    assert not roles_returned & {"Engineering", "Executive"}


def test_executive_sees_everything(store):
    q = fake_embed(["deployment question"])[0]
    results = store.query(q, top_k=4, allowed_roles=["Engineering", "HR", "Executive", "General"])
    assert len(results) == 4


def test_unauthorized_role_returns_nothing(store):
    q = fake_embed(["deployment question"])[0]
    # Empty allowed_roles list -> must return zero results, not "everything"
    results = store.query(q, top_k=4, allowed_roles=[])
    assert results == []


def test_unfiltered_audit_query_bypasses_acl_by_design(store):
    """The unfiltered path is intentionally an ACL bypass -- it must only ever
    be used by the permission-audit layer, never to answer a real user query.
    This test documents and locks in that behavior."""
    q = fake_embed(["deployment question"])[0]
    results = store.query_unfiltered(q, top_k=4)
    assert len(results) == 4
