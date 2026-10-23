"""
Route-level tests using FastAPI's real TestClient. These verify request
validation, response schemas, and error handling — not LLM behavior (the
/query and /debug/trace happy paths that need a live LLM call are covered
separately in test_pipeline_live.py, which skips automatically when no
provider is reachable).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LLM_PROVIDER", "ollama")

import pytest
from fastapi.testclient import TestClient

from app.api.security import require_api_access
from app.main import app


@pytest.fixture(autouse=True)
def _allow_api_access_for_route_tests():
    had_override = require_api_access in app.dependency_overrides
    previous_override = app.dependency_overrides.get(require_api_access)
    app.dependency_overrides[require_api_access] = lambda: None
    try:
        yield
    finally:
        if had_override:
            app.dependency_overrides[require_api_access] = previous_override
        else:
            app.dependency_overrides.pop(require_api_access, None)

client = TestClient(app)


def test_debug_trace_never_returns_unfiltered_chunk_text(monkeypatch):
    import app.api.routes_debug as rd
    monkeypatch.setattr(
        rd, "get_settings",
        lambda: SimpleNamespace(debug_api_key="test-debug-key"),
    )
    from app.rag.pipeline import QueryTrace
    from app.rag.vector_store import RetrievedChunk

    visible = RetrievedChunk(
        id="eng-1",
        text="Engineering runbook: restart the worker.",
        acl_role="Engineering",
        source="engineering.md",
        score=0.91,
    )
    hidden = RetrievedChunk(
        id="hr-secret-1",
        text="UNFILTERED_SECRET_MARKER confidential HR record",
        acl_role="HR",
        source="hr-private.md",
        score=0.99,
    )
    trace = QueryTrace(
        query="How do I restart the worker?",
        role="Engineering",
        allowed_roles=["Engineering", "General"],
        retrieved_chunks=[visible],
        unfiltered_top_chunks=[hidden, visible],
        answer="Restart the worker using the runbook.",
        context_used=[visible.text],
    )

    class FakePipeline:
        def __init__(self, settings):
            pass

        def run(self, query, role, top_k=None):
            return trace

    class FakeEvaluator:
        def __init__(self, settings):
            pass

        async def evaluate_sample(self, sample):
            from app.eval.ragas_eval import EvalResult
            return EvalResult(
                faithfulness=0.9,
                answer_relevancy=0.8,
                context_precision=0.9,
                context_recall=None,
            )

    monkeypatch.setattr(rd, "RAGPipeline", FakePipeline)
    monkeypatch.setattr(rd, "RagasEvaluator", FakeEvaluator)

    response = client.post(
        "/debug/trace",
        headers={"X-Debug-API-Key": "test-debug-key"},
        json={"query": trace.query, "role": "Engineering"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["unfiltered_top_chunks"] == []
    assert "UNFILTERED_SECRET_MARKER" not in response.text
    assert "hr-private.md" not in response.text
    assert body["retrieved_chunks"][0]["id"] == "eng-1"


def test_debug_trace_hides_provider_exception_details(monkeypatch):
    import app.api.routes_debug as rd
    monkeypatch.setattr(
        rd, "get_settings",
        lambda: SimpleNamespace(debug_api_key="test-debug-key"),
    )

    class FakePipeline:
        def __init__(self, settings):
            pass

        def run(self, query, role, top_k=None):
            raise RuntimeError("SECRET_PROVIDER_MARKER internal endpoint and credentials")

    monkeypatch.setattr(rd, "RAGPipeline", FakePipeline)

    response = client.post(
        "/debug/trace",
        headers={"X-Debug-API-Key": "test-debug-key"},
        json={"query": "test query", "role": "Engineering"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "LLM provider call failed."
    assert "SECRET_PROVIDER_MARKER" not in response.text
    assert "credentials" not in response.text


def test_debug_trace_hides_evaluator_exception_details(monkeypatch):
    import app.api.routes_debug as rd
    monkeypatch.setattr(
        rd, "get_settings",
        lambda: SimpleNamespace(debug_api_key="test-debug-key"),
    )
    from app.rag.pipeline import QueryTrace
    from app.rag.vector_store import RetrievedChunk

    visible = RetrievedChunk(
        id="eng-2",
        text="Public engineering context.",
        acl_role="Engineering",
        source="engineering.md",
        score=0.9,
    )
    trace = QueryTrace(
        query="test query",
        role="Engineering",
        allowed_roles=["Engineering", "General"],
        retrieved_chunks=[visible],
        unfiltered_top_chunks=[visible],
        answer="A test answer.",
        context_used=[visible.text],
    )

    class FakePipeline:
        def __init__(self, settings):
            pass

        def run(self, query, role, top_k=None):
            return trace

    class FakeEvaluator:
        def __init__(self, settings):
            pass

        async def evaluate_sample(self, sample):
            raise RuntimeError("SECRET_EVALUATOR_MARKER private evaluator diagnostics")

    monkeypatch.setattr(rd, "RAGPipeline", FakePipeline)
    monkeypatch.setattr(rd, "RagasEvaluator", FakeEvaluator)

    response = client.post(
        "/debug/trace",
        headers={"X-Debug-API-Key": "test-debug-key"},
        json={"query": trace.query, "role": "Engineering"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Evaluation failed."
    assert "SECRET_EVALUATOR_MARKER" not in response.text
    assert "private evaluator diagnostics" not in response.text

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


def test_regression_reports_dir_points_to_backend_ci_reports():
    import app.api.routes_regression as rr

    expected = Path(rr.__file__).resolve().parents[2] / "ci" / "reports"
    assert rr.REPORTS_DIR.resolve() == expected.resolve()


def test_regression_latest_returns_newest_report(tmp_path, monkeypatch):
    import app.api.routes_regression as rr

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    older = reports_dir / "older.json"
    newer = reports_dir / "newer.json"

    older.write_text(
        json.dumps({"timestamp": "older", "total_samples": 2}),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps({"timestamp": "newer", "total_samples": 5}),
        encoding="utf-8",
    )

    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))
    monkeypatch.setattr(rr, "REPORTS_DIR", reports_dir)

    response = client.get("/regression/latest")

    assert response.status_code == 200
    assert response.json()["timestamp"] == "newer"
    assert response.json()["total_samples"] == 5


def test_regression_history_returns_report_summary(tmp_path, monkeypatch):
    import app.api.routes_regression as rr

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report = {
        "timestamp": "2026-08-04T13:01:16Z",
        "avg_faithfulness": 0.6048,
        "avg_answer_relevancy": 0.7644,
        "avg_context_precision": 0.8,
        "passed": False,
        "total_samples": 5,
        "per_sample": [{"query": "example"}],
    }
    (reports_dir / "20260804T130117Z.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    monkeypatch.setattr(rr, "REPORTS_DIR", reports_dir)

    response = client.get("/regression/history")

    assert response.status_code == 200
    assert response.json() == [{
        "timestamp": "20260804T130117Z",
        "avg_faithfulness": 0.6048,
        "avg_answer_relevancy": 0.7644,
        "avg_context_precision": 0.8,
        "passed": False,
        "total_samples": 5,
    }]


def test_regression_history_rejects_invalid_limits():
    for limit in ("0", "-1", "101"):
        response = client.get(f"/regression/history?limit={limit}")
        assert response.status_code == 422, f"limit={limit}"


def test_regression_history_accepts_maximum_limit(tmp_path, monkeypatch):
    import app.api.routes_regression as rr

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(rr, "REPORTS_DIR", reports_dir)

    response = client.get("/regression/history?limit=100")

    assert response.status_code == 200
    assert response.json() == []


def test_regression_latest_hides_malformed_report_details(tmp_path, monkeypatch):
    import app.api.routes_regression as rr

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "broken.json").write_text(
        '{"sensitive": ',
        encoding="utf-8",
    )
    monkeypatch.setattr(rr, "REPORTS_DIR", reports_dir)

    response = client.get("/regression/latest")

    assert response.status_code == 500
    assert response.json()["detail"] == "Regression report could not be read."
    assert "JSONDecodeError" not in response.text
    assert str(reports_dir) not in response.text


def test_regression_history_hides_malformed_report_details(tmp_path, monkeypatch):
    import app.api.routes_regression as rr

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "broken.json").write_text(
        '{"sensitive": ',
        encoding="utf-8",
    )
    monkeypatch.setattr(rr, "REPORTS_DIR", reports_dir)

    response = client.get("/regression/history")

    assert response.status_code == 500
    assert response.json()["detail"] == "Regression report could not be read."
    assert "JSONDecodeError" not in response.text
    assert str(reports_dir) not in response.text


def test_regression_latest_rejects_non_object_json(tmp_path, monkeypatch):
    import app.api.routes_regression as rr

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "array.json").write_text(
        '["not", "an", "object"]',
        encoding="utf-8",
    )
    monkeypatch.setattr(rr, "REPORTS_DIR", reports_dir)

    response = client.get("/regression/latest")

    assert response.status_code == 500
    assert response.json()["detail"] == "Regression report could not be read."


def test_regression_latest_directory_permission_error_is_generic(monkeypatch):
    import app.api.routes_regression as rr

    class UnreadableReportsDirectory:
        def exists(self):
            raise PermissionError("SECRET_PATH_MARKER")

    monkeypatch.setattr(rr, "REPORTS_DIR", UnreadableReportsDirectory())

    response = client.get("/regression/latest")

    assert response.status_code == 500
    assert response.json()["detail"] == "Regression report could not be read."
    assert "SECRET_PATH_MARKER" not in response.text


def test_regression_history_directory_permission_error_is_generic(monkeypatch):
    import app.api.routes_regression as rr

    class UnreadableReportsDirectory:
        def exists(self):
            raise PermissionError("SECRET_PATH_MARKER")

    monkeypatch.setattr(rr, "REPORTS_DIR", UnreadableReportsDirectory())

    response = client.get("/regression/history")

    assert response.status_code == 500
    assert response.json()["detail"] == "Regression report could not be read."
    assert "SECRET_PATH_MARKER" not in response.text

def test_debug_trace_disabled_when_api_key_is_not_configured(monkeypatch):
    import app.api.routes_debug as rd

    monkeypatch.setattr(
        rd, "get_settings",
        lambda: SimpleNamespace(debug_api_key=None),
    )

    response = client.post(
        "/debug/trace",
        json={"query": "test query", "role": "Engineering"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Debug endpoint is disabled because access is not configured."
    )


def test_debug_trace_rejects_missing_api_key(monkeypatch):
    import app.api.routes_debug as rd

    monkeypatch.setattr(
        rd, "get_settings",
        lambda: SimpleNamespace(debug_api_key="test-debug-key"),
    )

    response = client.post(
        "/debug/trace",
        json={"query": "test query", "role": "Engineering"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing debug API key."


def test_debug_trace_rejects_invalid_api_key(monkeypatch):
    import app.api.routes_debug as rd

    monkeypatch.setattr(
        rd, "get_settings",
        lambda: SimpleNamespace(debug_api_key="test-debug-key"),
    )

    response = client.post(
        "/debug/trace",
        headers={"X-Debug-API-Key": "wrong-test-key"},
        json={"query": "test query", "role": "Engineering"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing debug API key."

def test_debug_trace_hides_value_error_details(monkeypatch):
    import app.api.routes_debug as rd

    monkeypatch.setattr(
        rd, "get_settings",
        lambda: SimpleNamespace(debug_api_key="test-debug-key"),
    )

    class FakePipeline:
        def __init__(self, settings):
            pass

        def run(self, query, role, top_k=None):
            raise ValueError("SECRET_VALUE_ERROR_MARKER internal configuration")

    monkeypatch.setattr(rd, "RAGPipeline", FakePipeline)

    response = client.post(
        "/debug/trace",
        headers={"X-Debug-API-Key": "test-debug-key"},
        json={"query": "test query", "role": "Engineering"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid debug trace request."
    assert "SECRET_VALUE_ERROR_MARKER" not in response.text
    assert "internal configuration" not in response.text


def test_query_hides_provider_exception_details(monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import routes_query
    from app.main import app

    secret_marker = "SECRET_QUERY_PROVIDER_MARKER"

    class FailingPipeline:
        def __init__(self, settings):
            pass

        def run(self, *args, **kwargs):
            raise RuntimeError(secret_marker)

    monkeypatch.setattr(routes_query, "get_settings", lambda: object())
    monkeypatch.setattr(routes_query, "RAGPipeline", FailingPipeline)

    response = TestClient(app).post(
        "/query",
        json={"query": "test query", "role": "Engineering"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "LLM provider call failed."
    assert secret_marker not in response.text


def test_query_hides_value_error_details(monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import routes_query
    from app.main import app

    secret_marker = "SECRET_QUERY_VALIDATION_MARKER"

    class FailingPipeline:
        def __init__(self, settings):
            pass

        def run(self, *args, **kwargs):
            raise ValueError(secret_marker)

    monkeypatch.setattr(routes_query, "get_settings", lambda: object())
    monkeypatch.setattr(routes_query, "RAGPipeline", FailingPipeline)

    response = TestClient(app).post(
        "/query",
        json={"query": "test query", "role": "Engineering"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid query request."
    assert secret_marker not in response.text
