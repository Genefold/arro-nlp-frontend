"""Smoke tests for the FastAPI application scaffold."""

from __future__ import annotations

from arro_nlp_frontend.main import create_app


def test_health_endpoint(app_client):
    """GET /health returns 200 and required fields."""
    r = app_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "embed_backend" in body
    assert "embedder_dim" in body
    assert body["embedder_dim"] == 384


def test_openapi_schema_reachable(app_client):
    """OpenAPI schema must be served at /openapi.json."""
    r = app_client.get("/openapi.json")
    assert r.status_code == 200


def test_create_app_returns_fastapi_instance():
    from fastapi import FastAPI

    app = create_app()
    assert isinstance(app, FastAPI)
