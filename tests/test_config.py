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
