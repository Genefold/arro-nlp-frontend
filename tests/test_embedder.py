"""Tests for arro_nlp_frontend.embedder.

All tests run fully offline — no HF Hub downloads after first run, no arro-server.
"""

from __future__ import annotations

import numpy as np
import pytest

from arro_nlp_frontend.embedder import Embedder


class TestEmbedderLocal:
    def test_dtype_is_float64(self, local_embedder):
        vecs = local_embedder.encode_batch(["buffer overflow in OpenSSL"])
        assert vecs.dtype == np.float64

    def test_shape_single(self, local_embedder):
        vecs = local_embedder.encode_batch(["test"])
        assert vecs.ndim == 2
        assert vecs.shape[0] == 1
        assert vecs.shape[1] == local_embedder.dim

    def test_shape_batch(self, local_embedder):
        texts = ["SQL injection", "heap overflow", "privilege escalation"]
        vecs = local_embedder.encode_batch(texts)
        assert vecs.shape == (3, local_embedder.dim)

    def test_empty_batch(self, local_embedder):
        vecs = local_embedder.encode_batch([])
        assert vecs.shape == (0, local_embedder.dim)
        assert vecs.dtype == np.float64

    def test_scale_factor_applied(self, local_embedder):
        """Norms must be ~1× the raw norms (MiniLM produces ~unit vectors)."""
        vecs = local_embedder.encode_batch(["authentication bypass via JWT"])
        norm = float(np.linalg.norm(vecs[0]))
        # With scale_factor=1.0, norm should be close to 1 (unit vector from MiniLM)
        assert 0.5 < norm < 2.0, f"Expected scaled norm near 1.0, got {norm:.3f}"

    def test_not_unit_normalised(self, local_embedder):
        """encode_batch must NOT call normalize_embeddings=True.
        Two semantically different texts produce vectors with different norms;
        normalised vectors would all have norm exactly 1.0."""
        vecs = local_embedder.encode_batch(
            [
                "remote code execution via buffer overflow",
                "x",
            ]
        )
        norm_a = float(np.linalg.norm(vecs[0]))
        norm_b = float(np.linalg.norm(vecs[1]))
        assert norm_a != norm_b or abs(norm_a - 1.0) > 1e-6, (
            "Vectors appear to be L2-normalised — check normalize_embeddings=False"
        )

    def test_deterministic(self, local_embedder):
        text = ["use-after-free in browser renderer"]
        np.testing.assert_array_equal(
            local_embedder.encode_batch(text),
            local_embedder.encode_batch(text),
        )

    def test_dim_property(self, local_embedder):
        assert local_embedder.dim == 384


class TestEmbedderModelPath:
    def test_missing_path_raises_file_not_found(self, tmp_path):
        """Non-empty path that does not exist must raise FileNotFoundError immediately."""
        with pytest.raises(FileNotFoundError):
            Embedder(
                backend="local",
                model="all-MiniLM-L6-v2",
                model_path=str(tmp_path / "nonexistent"),
            )

    def test_file_not_dir_raises(self, tmp_path):
        """A file path (not a directory) must raise NotADirectoryError."""
        f = tmp_path / "model.bin"
        f.write_bytes(b"fake")
        with pytest.raises(NotADirectoryError):
            Embedder(backend="local", model="any", model_path=str(f))

    def test_valid_path_loads(self, tmp_path, monkeypatch):
        """Priority 1: existing directory → load from disk (SentenceTransformer mocked)."""
        model_dir = tmp_path / "domain_adapted_model"
        model_dir.mkdir()

        class _FakeModel:
            def get_sentence_embedding_dimension(self):
                return 384

            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 384), dtype=np.float32)

        monkeypatch.setattr(
            "arro_nlp_frontend.embedder.SentenceTransformer", lambda p: _FakeModel()
        )
        e = Embedder(backend="local", model="any", model_path=str(model_dir))
        vecs = e.encode_batch(["test"])
        assert vecs.dtype == np.float64
        assert vecs.shape == (1, 384)


class TestEmbedderOpenAI:
    def test_openai_mocked(self, monkeypatch):
        """OpenAI backend with mocked client — no real API key needed."""

        class _Emb:
            embedding = [0.05] * 384

        class _Resp:
            data = [_Emb()]

        class _EmbAPI:
            def create(self, **kwargs):
                return _Resp()

        class _Client:
            def __init__(self, api_key):
                self.embeddings = _EmbAPI()

        import openai

        monkeypatch.setattr(openai, "OpenAI", _Client)

        e = Embedder(
            backend="openai",
            model="text-embedding-3-small",
            api_key="sk-fake",
            scale_factor=1.0,
        )
        vecs = e.encode_batch(["test"])
        assert vecs.dtype == np.float64
        assert vecs.shape == (1, 384)

    def test_openai_no_key_raises(self):
        with pytest.raises(ValueError, match="api_key is required"):
            Embedder(backend="openai", model="text-embedding-3-small", api_key="")
