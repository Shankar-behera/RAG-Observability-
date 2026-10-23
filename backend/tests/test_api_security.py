from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes_ingest, routes_query, routes_regression, security


def _isolated_client() -> TestClient:
    isolated_app = FastAPI()
    isolated_app.include_router(routes_ingest.router)
    isolated_app.include_router(routes_query.router)
    isolated_app.include_router(routes_regression.router)
    return TestClient(isolated_app)


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/query", {"query": "test query", "role": "Engineering"}),
        ("post", "/ingest", {"documents": []}),
        ("get", "/regression/history", None),
        ("get", "/regression/latest", None),
    ],
)
def test_protected_api_routes_reject_missing_key(
    monkeypatch, method, path, payload
):
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: SimpleNamespace(api_access_key="test-api-access-key"),
    )
    client = _isolated_client()

    if payload is None:
        response = getattr(client, method)(path)
    else:
        response = getattr(client, method)(path, json=payload)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_protected_api_route_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: SimpleNamespace(api_access_key="test-api-access-key"),
    )

    response = _isolated_client().get(
        "/regression/history",
        headers={"X-API-Key": "wrong-test-key"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_protected_api_routes_fail_closed_without_configuration(monkeypatch):
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: SimpleNamespace(api_access_key=None),
    )

    response = _isolated_client().get("/regression/history")

    assert response.status_code == 503
    assert response.json()["detail"] == "API access is not configured."


def test_valid_api_key_allows_regression_history(monkeypatch, tmp_path):
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: SimpleNamespace(api_access_key="test-api-access-key"),
    )
    monkeypatch.setattr(routes_regression, "REPORTS_DIR", tmp_path)

    response = _isolated_client().get(
        "/regression/history",
        headers={"X-API-Key": "test-api-access-key"},
    )

    assert response.status_code == 200
    assert response.json() == []


def test_compose_ports_are_loopback_only_and_secrets_are_server_side():
    root = Path(__file__).resolve().parents[2]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    api_ts = (root / "frontend/src/lib/api.ts").read_text(encoding="utf-8")
    proxy = (
        root / "frontend/src/app/api/backend/[...path]/route.ts"
    ).read_text(encoding="utf-8")

    for port in ("5432", "11434", "8000", "3000"):
        assert f'"127.0.0.1:{port}:{port}"' in compose

    assert 'path === "/debug/trace" ? "/api/debug/trace"' in api_ts
    assert "/api/backend" in api_ts
    assert "process.env.API_ACCESS_KEY" in proxy
    assert '"X-API-Key": apiKey' in proxy
    assert "NEXT_PUBLIC_API_URL" not in api_ts
def test_generated_openapi_documents_authentication_responses():
    from app.main import app

    schema = app.openapi()
    protected = {
        ("/query", "post"): "X-API-Key",
        ("/ingest", "post"): "X-API-Key",
        ("/regression/latest", "get"): "X-API-Key",
        ("/regression/history", "get"): "X-API-Key",
        ("/debug/trace", "post"): "X-Debug-API-Key",
    }

    for (path, method), expected_header in protected.items():
        operation = schema["paths"][path][method]
        header_names = {
            parameter["name"]
            for parameter in operation.get("parameters", [])
            if parameter.get("in") == "header"
        }

        assert expected_header in header_names, f"{method.upper()} {path}"
        assert {"401", "503"}.issubset(operation["responses"]), (
            f"Authentication responses missing from {method.upper()} {path}"
        )
def test_frontend_proxy_allows_health_without_api_key():
    root = Path(__file__).resolve().parents[2]
    proxy = (
        root / "frontend/src/app/api/backend/[...path]/route.ts"
    ).read_text(encoding="utf-8")

    assert '"/health"' in proxy
    assert 'backendPath !== "/health" && !apiKey' in proxy
    assert '...(apiKey ? { "X-API-Key": apiKey } : {})' in proxy


def test_debug_api_key_trims_surrounding_configuration_whitespace(monkeypatch):
    from app.api import routes_debug

    monkeypatch.setattr(
        routes_debug,
        "get_settings",
        lambda: SimpleNamespace(debug_api_key=" test-debug-key "),
    )

    routes_debug.require_debug_access("test-debug-key")
