"""Smoke tests for the FastAPI application scaffold."""

from __future__ import annotations

import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arro_nlp_frontend.arro_client import ArroClient
from arro_nlp_frontend.main import _check_single_worker, create_app, lifespan


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


def _mock_arro_init(self, base_url="", timeout=30.0):
    self._client = AsyncMock()


@pytest.mark.asyncio
async def test_lifespan_arro_client_closed_on_store_init_failure():
    """aclose() must be called even when DocumentStore raises during startup."""
    with (
        patch("arro_nlp_frontend.main.Embedder.from_settings", return_value=MagicMock(dim=384)),
        patch("arro_nlp_frontend.main.DocumentStore.__init__", side_effect=RuntimeError("schema migration required")),
        patch.object(ArroClient, "__init__", _mock_arro_init),
        patch.object(ArroClient, "aclose", new_callable=AsyncMock) as mock_aclose,
    ):
        app = create_app()
        with pytest.raises(RuntimeError):
            async with lifespan(app):
                pass

        assert mock_aclose.call_count == 1


@pytest.mark.asyncio
async def test_lifespan_store_not_closed_when_store_init_fails():
    """store.close() must not be called if DocumentStore init never completed."""
    with (
        patch("arro_nlp_frontend.main.Embedder.from_settings", return_value=MagicMock(dim=384)),
        patch("arro_nlp_frontend.main.DocumentStore.__init__", side_effect=RuntimeError("schema migration required")),
        patch("arro_nlp_frontend.main.DocumentStore.close", new_callable=MagicMock) as mock_close,
        patch.object(ArroClient, "__init__", _mock_arro_init),
        patch.object(ArroClient, "aclose", new_callable=AsyncMock),
    ):
        app = create_app()
        with pytest.raises(RuntimeError):
            async with lifespan(app):
                pass

        assert mock_close.call_count == 0


@pytest.mark.asyncio
async def test_lifespan_normal_startup_and_shutdown_closes_both():
    """On normal startup and shutdown, both arro_client.aclose() and store.close() are called."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

    def mock_arro_init_with_client(self, base_url="", timeout=30.0):
        self._client = mock_client

    with (
        patch("arro_nlp_frontend.main.Embedder.from_settings", return_value=MagicMock(dim=384)),
        patch("arro_nlp_frontend.main.DocumentStore.__init__", return_value=None),
        patch("arro_nlp_frontend.main.DocumentStore.close", new_callable=MagicMock) as mock_close,
        patch.object(ArroClient, "__init__", mock_arro_init_with_client),
        patch.object(ArroClient, "aclose", new_callable=AsyncMock) as mock_aclose,
    ):
        app = create_app()
        async with lifespan(app):
            pass

        assert mock_aclose.call_count == 1
        assert mock_close.call_count == 1


@pytest.mark.asyncio
async def test_lifespan_arro_client_closed_on_embedder_failure():
    """aclose() must be called even when Embedder.from_settings raises."""
    with (
        patch("arro_nlp_frontend.main.Embedder.from_settings", side_effect=FileNotFoundError("model path missing")),
        patch("arro_nlp_frontend.main.DocumentStore.close", new_callable=MagicMock) as mock_close,
        patch.object(ArroClient, "__init__", _mock_arro_init),
        patch.object(ArroClient, "aclose", new_callable=AsyncMock) as mock_aclose,
    ):
        app = create_app()
        with pytest.raises(FileNotFoundError):
            async with lifespan(app):
                pass

        assert mock_aclose.call_count == 1
        assert mock_close.call_count == 0


def test_check_single_worker_no_warning_when_absent(caplog):
    """No CRITICAL log when WEB_CONCURRENCY and UVICORN_WORKERS are not set."""
    with (
        patch.dict(os.environ, {"WEB_CONCURRENCY": "1", "UVICORN_WORKERS": "1"}),
        caplog.at_level(logging.CRITICAL, logger="arro_nlp_frontend.main"),
    ):
        _check_single_worker()
    assert not any(r.levelname == "CRITICAL" for r in caplog.records)


def test_check_single_worker_critical_on_web_concurrency(caplog):
    """CRITICAL is logged when WEB_CONCURRENCY is set to a value other than '1'."""
    with (
        patch.dict(os.environ, {"WEB_CONCURRENCY": "2"}, clear=True),
        caplog.at_level(logging.CRITICAL, logger="arro_nlp_frontend.main"),
    ):
        _check_single_worker()
    criticals = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert len(criticals) == 1
    assert "WEB_CONCURRENCY" in criticals[0].message
    assert "2" in criticals[0].message


def test_check_single_worker_critical_on_uvicorn_workers(caplog):
    """CRITICAL is logged when UVICORN_WORKERS is set to a value other than '1'."""
    with (
        patch.dict(os.environ, {"UVICORN_WORKERS": "4"}, clear=True),
        caplog.at_level(logging.CRITICAL, logger="arro_nlp_frontend.main"),
    ):
        _check_single_worker()
    criticals = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert len(criticals) == 1
    assert "UVICORN_WORKERS" in criticals[0].message
    assert "4" in criticals[0].message


def test_check_single_worker_no_warning_when_set_to_one(caplog):
    """No CRITICAL log when WEB_CONCURRENCY is explicitly set to '1'."""
    with (
        patch.dict(os.environ, {"WEB_CONCURRENCY": "1", "UVICORN_WORKERS": "1"}),
        caplog.at_level(logging.CRITICAL, logger="arro_nlp_frontend.main"),
    ):
        _check_single_worker()
    assert not any(r.levelname == "CRITICAL" for r in caplog.records)


def test_check_single_worker_critical_for_both_vars(caplog):
    """Two CRITICAL logs emitted when both WEB_CONCURRENCY and UVICORN_WORKERS are > 1."""
    with (
        patch.dict(os.environ, {"WEB_CONCURRENCY": "2", "UVICORN_WORKERS": "2"}, clear=True),
        caplog.at_level(logging.CRITICAL, logger="arro_nlp_frontend.main"),
    ):
        _check_single_worker()
    criticals = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert len(criticals) == 2
