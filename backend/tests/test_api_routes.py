"""
Route-level tests using FastAPI's real TestClient. These verify request
validation, response schemas, and error handling — not LLM behavior (the
/query and /debug/trace happy paths that need a live LLM call are covered
separately in test_pipeline_live.py, which skips automatically when no
provider is reachable).
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "ollama")

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body
    assert "vector_store" in body


def test_openapi_schema_includes_all_routers():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for expected in ["/ingest", "/query", "/debug/trace", "/regression/latest", "/regression/history"]:
        assert expected in paths, f"missing route {expected}"


def test_query_rejects_unknown_role():
    r = client.post("/query", json={"query": "what is our vacation policy", "role": "Intern"})
    assert r.status_code == 400
    assert "Unknown role" in r.json()["detail"]


def test_query_requires_query_field():
    r = client.post("/query", json={"role": "Engineering"})
    assert r.status_code == 422  # pydantic validation error


def test_regression_latest_returns_404_when_no_reports(tmp_path, monkeypatch):
    import app.api.routes_regression as rr
    monkeypatch.setattr(rr, "REPORTS_DIR", tmp_path / "nonexistent")
    r = client.get("/regression/latest")
    assert r.status_code == 404


def test_regression_history_empty_list_when_no_reports(tmp_path, monkeypatch):
    import app.api.routes_regression as rr
    monkeypatch.setattr(rr, "REPORTS_DIR", tmp_path / "nonexistent")
    r = client.get("/regression/history")
    assert r.status_code == 200
    assert r.json() == []
