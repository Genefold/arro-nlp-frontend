"""Async HTTP client for arro-server.

Single responsibility: translate HTTP calls to/from arro-server into
Python types. No business logic lives here.

All methods raise ArroServerError on any non-2xx response or network failure.
The caller (ingest endpoint) decides how to handle it.
"""

from __future__ import annotations

import logging

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class ArroServerError(Exception):
    """Raised when arro-server returns a non-2xx response or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ArroClient:
    """Async HTTP client for arro-server.

    Parameters
    ----------
    base_url:   e.g. "http://localhost:8001"
    dataset_id: e.g. "cve/embeddings"
    timeout:    httpx timeout in seconds (default 30.0)
    """

    def __init__(self, base_url: str, dataset_id: str, timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )
        self._dataset_id = dataset_id

    async def push_vectors(self, vectors: np.ndarray, start_row: int) -> None:
        """POST /datasets/{dataset_id}/vectors

        Body: { "start_row": int, "vectors": [[float, ...], ...] }
        Raises ArroServerError on failure.
        """
        payload = {
            "start_row": start_row,
            "vectors": vectors.tolist(),
        }
        url = f"/datasets/{self._dataset_id}/vectors"
        try:
            response = await self._client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise ArroServerError(str(exc), status_code=None) from exc

        if response.status_code >= 400:
            raise ArroServerError(
                f"arro-server POST {url} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

    async def row_count(self) -> int:
        """GET /datasets/{dataset_id}/info -> { "n_rows": int }

        Returns 0 if arro-server responds with 404 (dataset not yet created).
        Raises ArroServerError on other non-2xx.
        """
        url = f"/datasets/{self._dataset_id}/info"
        try:
            response = await self._client.get(url)
        except httpx.RequestError as exc:
            raise ArroServerError(str(exc), status_code=None) from exc

        if response.status_code == 404:
            return 0
        if response.status_code >= 400:
            raise ArroServerError(
                f"arro-server GET {url} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )
        return int(response.json()["n_rows"])

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient."""
        await self._client.aclose()
