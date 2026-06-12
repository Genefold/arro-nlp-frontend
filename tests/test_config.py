"""Tests for arro_nlp_frontend.config."""

import pytest
from pydantic import ValidationError


def test_default_settings_valid():
    """Settings with all defaults must not raise."""
    from arro_nlp_frontend.config import Settings

    s = Settings()
    assert s.embed_backend == "local"
    assert s.embed_scale_factor == 1.0
    assert 0.0 <= s.arro_server_search_tau <= 1.0


def test_invalid_backend_raises():
    from arro_nlp_frontend.config import Settings

    with pytest.raises(ValidationError):
        Settings(embed_backend="cohere")


def test_invalid_scale_raises():
    from arro_nlp_frontend.config import Settings

    with pytest.raises(ValidationError):
        Settings(embed_scale_factor=-1.0)


def test_invalid_tau_raises():
    from arro_nlp_frontend.config import Settings

    with pytest.raises(ValidationError):
        Settings(arro_server_search_tau=1.5)


def test_openai_without_key_raises():
    from arro_nlp_frontend.config import Settings

    with pytest.raises(ValidationError):
        Settings(embed_backend="openai", openai_api_key="")


def test_openai_with_key_valid():
    from arro_nlp_frontend.config import Settings

    s = Settings(embed_backend="openai", openai_api_key="sk-test")
    assert s.embed_backend == "openai"


def test_store_db_path_default():
    """Default store_db_path is './data/documents.sqlite'."""
    from arro_nlp_frontend.config import Settings
    s = Settings()
    assert s.store_db_path == "./data/documents.sqlite"


def test_store_db_path_env_override(monkeypatch):
    """STORE_DB_PATH env var overrides the default."""
    monkeypatch.setenv("STORE_DB_PATH", "/tmp/custom.sqlite")
    # Re-instantiate to pick up the patched env
    from arro_nlp_frontend.config import Settings
    s = Settings()
    assert s.store_db_path == "/tmp/custom.sqlite"


def test_store_db_path_empty_string_accepted():
    """Empty string is a valid value (path validation is the caller's job)."""
    from arro_nlp_frontend.config import Settings
    s = Settings(store_db_path="")
    assert s.store_db_path == ""
