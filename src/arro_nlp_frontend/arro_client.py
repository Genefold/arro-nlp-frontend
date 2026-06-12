"""Async HTTP client for arro-server.

Single responsibility: translate HTTP calls to/from arro-server into
Python types. No business logic lives here.

All methods raise ArroServerError on any non-2xx response or network failure.
The caller (ingest endpoint) decides how to handle it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import httpx

logger = logging.getLogger(__name__)


class ArroServerError(Exception):
    """Raised when arro-server returns a non-2xx response or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class UploadCommitResult:
    """Result returned by /api/upload/commit."""

    index_stale: bool
    shape: list[int]


class ArroClient:
    """Async HTTP client for arro-server.

    Parameters
    ----------
    base_url:   e.g. "http://localhost:8001"
    dataset_id: e.g. "main--cve_embeddings"
    root_label: e.g. "main" — must match an ARRO_SERVER_DATA_ROOTS key
    timeout:    httpx timeout in seconds (default 30.0)
    """

    def __init__(
        self,
        base_url: str,
        dataset_id: str,
        root_label: str,
        timeout: float = 30.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )
        self._dataset_id = dataset_id
        self._root_label = root_label

    async def dataset_metadata(self) -> dict | None:
        """GET /api/datasets/{dataset_id}/metadata

        Returns the metadata dict on 200, None on 404 (dataset not yet created).
        Raises ArroServerError on other non-2xx.
        """
        url = f"/api/datasets/{self._dataset_id}/metadata"
        try:
            response = await self._client.get(url)
        except httpx.RequestError as exc:
            raise ArroServerError(str(exc), status_code=None) from exc

        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise ArroServerError(
                f"arro-server GET {url} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )
        return cast(dict, response.json())

    async def upload_init(self) -> str:
        """POST /api/upload/init

        Body: {"dataset_id": self._dataset_id, "root": self._root_label}
        Returns upload_path (str) — the absolute filesystem path where
        the caller must write the Zarr v3 array.
        Raises ArroServerError on failure.
        """
        payload = {
            "dataset_id": self._dataset_id,
            "root": self._root_label,
        }
        url = "/api/upload/init"
        try:
            response = await self._client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise ArroServerError(str(exc), status_code=None) from exc

        if response.status_code >= 400:
            raise ArroServerError(
                f"arro-server POST {url} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )
        return str(response.json()["upload_path"])

    async def upload_commit(self, fs_path: str) -> UploadCommitResult:
        """POST /api/upload/commit

        Body: {"dataset_id": self._dataset_id, "fs_path": fs_path}
        Returns UploadCommitResult(index_stale, shape).
        Raises ArroServerError on failure.
        """
        payload = {
            "dataset_id": self._dataset_id,
            "fs_path": fs_path,
        }
        url = "/api/upload/commit"
        try:
            response = await self._client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise ArroServerError(str(exc), status_code=None) from exc

        if response.status_code >= 400:
            raise ArroServerError(
                f"arro-server POST {url} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )
        data = response.json()
        return UploadCommitResult(
            index_stale=bool(data["index_stale"]),
            shape=list(data["shape"]),
        )

    async def build_index(self, graph_params: dict | None = None) -> None:
        """POST /api/datasets/{dataset_id}/index

        Body: {"graph_params": graph_params} or {} for server defaults.
        Raises ArroServerError on failure.
        """
        payload: dict = {}
        if graph_params is not None:
            payload["graph_params"] = graph_params
        url = f"/api/datasets/{self._dataset_id}/index"
        try:
            response = await self._client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise ArroServerError(str(exc), status_code=None) from exc

        if response.status_code >= 400:
            raise ArroServerError(
                f"arro-server POST {url} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient."""
        await self._client.aclose()
