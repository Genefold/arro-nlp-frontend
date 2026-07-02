"""Embedder: encodes text → float64 vectors for arro-server.

Design rules (NON NEGOTIABLE):
  - Vectors are NEVER L2-normalised. arro-server expects raw scaled vectors.
  - scale_factor is applied on every encode_batch call.
  - model_path set but missing on disk → FileNotFoundError at construction time.
    Never fall back silently: a silent fallback would mix vector distributions
    in the same Zarr index and corrupt search quality with no error.
  - model_path set but directory is EMPTY → fall back to HF Hub model with a
    WARNING. An empty directory means no domain-adapted model has been placed
    there yet; using the base model is safe because the index is also empty.
  - Empty input returns shape (0, dim), never raises.
  - Logging after every batch: count, mean_norm, min_norm, max_norm, elapsed.

Backend resolution order:
  1. model_path non-empty AND exists AND non-empty dir → SentenceTransformer(model_path)
  2. model_path non-empty AND exists AND empty dir     → SentenceTransformer(model) via
     HF Hub (WARNING)
  3. model_path non-empty AND missing                 → FileNotFoundError (fail loud)
  4. model_path empty, backend=local                  → SentenceTransformer(model) via HF Hub
  5. backend=openai                                   → OpenAI Embeddings API
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[misc, assignment]

try:
    import openai as _openai
except ImportError:  # pragma: no cover
    _openai = None  # type: ignore[misc, assignment]


def _is_empty_dir(p: Path) -> bool:
    """Return True if *p* is a directory that contains no files (recursively)."""
    return p.is_dir() and not any(p.rglob("*"))


class Embedder:
    """Encodes text to float64 vectors compatible with arro-server.

    Parameters
    ----------
    backend:      "local" | "openai"
    model:        HuggingFace model name (local backend without model_path)
    scale_factor: multiplier applied after encoding (default 1.0)
    api_key:      OpenAI API key (required when backend="openai")
    model_path:   path to local fine-tuned model directory (overrides model)
    """

    def __init__(
        self,
        *,
        backend: str,
        model: str,
        scale_factor: float = 1.0,
        api_key: str = "",
        model_path: str = "",
    ) -> None:
        self.backend = backend
        self.model_name = model
        self.scale_factor = scale_factor
        self._model = None
        self._client = None

        if backend == "openai":
            if _openai is None:  # pragma: no cover
                raise ImportError("openai package is required: pip install openai")
            if not api_key:
                raise ValueError("api_key is required when backend='openai'")
            self._client = _openai.OpenAI(api_key=api_key)

        elif backend == "local":
            if SentenceTransformer is None:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required: pip install sentence-transformers"
                )
            if model_path:
                p = Path(model_path)
                if not p.exists():
                    raise FileNotFoundError(
                        "EMBEDDER_MODEL_PATH is set but the directory does not exist: "
                        f"{model_path}\n"
                        "Fix: either create the directory or unset EMBEDDER_MODEL_PATH."
                    )
                if not p.is_dir():
                    raise NotADirectoryError(
                        f"EMBEDDER_MODEL_PATH must be a directory, not a file: {model_path}"
                    )
                if _is_empty_dir(p):
                    # Volume mount created an empty placeholder directory — no
                    # domain-adapted model has been deployed yet. Fall back to
                    # the base HF Hub model so the service starts cleanly.
                    # WARNING (not ERROR) because this is expected in fresh
                    # deployments and CI environments.
                    logger.warning(
                        "EMBEDDER_MODEL_PATH=%r is an empty directory. "
                        "No domain-adapted model found — falling back to "
                        "HF Hub model %r. "
                        "Place a fine-tuned SentenceTransformer model in that "
                        "directory and restart to use it.",
                        model_path,
                        model,
                    )
                    self._model = SentenceTransformer(model)
                else:
                    self._model = SentenceTransformer(str(p))
            else:
                self._model = SentenceTransformer(model)
        else:
            raise ValueError(f"backend must be 'local' or 'openai', got {backend!r}")

    @classmethod
    def from_settings(cls) -> Embedder:
        """Construct from arro_nlp_frontend.config.settings singleton."""
        from arro_nlp_frontend.config import settings

        return cls(
            backend=settings.embed_backend,
            model=settings.embed_model,
            scale_factor=settings.embed_scale_factor,
            api_key=settings.openai_api_key,
            model_path=settings.embedder_model_path,
        )

    @property
    def dim(self) -> int:
        """Output dimension. 384 for all-MiniLM-L6-v2 family."""
        if self._model is not None:
            return int(self._model.get_embedding_dimension())  # type: ignore[union-attr]
        return 384  # pragma: no cover — openai dim depends on model

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Return float64 array shape (len(texts), dim), scaled, NOT normalised.

        Empty input returns shape (0, dim).
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float64)

        t0 = time.perf_counter()

        if self.backend == "openai":
            assert self._client is not None
            response = self._client.embeddings.create(model=self.model_name, input=texts)
            raw = np.array([d.embedding for d in response.data], dtype=np.float64)
        else:
            assert self._model is not None
            raw = self._model.encode(
                texts,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=False,  # NEVER normalise
                show_progress_bar=False,
            )

        scaled = raw.astype(np.float64) * self.scale_factor
        elapsed = time.perf_counter() - t0
        norms = np.linalg.norm(scaled, axis=1)
        logger.info(
            "[embedder] batch count=%d mean_norm=%.3f min=%.3f max=%.3f elapsed=%.2fs",
            len(texts),
            float(norms.mean()),
            float(norms.min()),
            float(norms.max()),
            elapsed,
        )
        return scaled
