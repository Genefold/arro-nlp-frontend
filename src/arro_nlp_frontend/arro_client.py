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
import numpy as np

__all__ = [
    "ArroServerError",
    "ArroClient",
    "UploadCommitResult",
    "VectorAppendResult",
    "VectorOverwriteResult",
    "SearchHit",
]

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


@dataclass
class VectorAppendResult:
    """Result returned by POST /api/datasets/{id}/vectors/append.

    Attributes
    ----------
    start_row:  Index of the first appended row in the Zarr array.
    appended:   Number of rows actually appended (== len(vectors) sent).
    new_shape:  Full array shape after the append, e.g. [300200, 384].
    """

    start_row: int
    appended: int
    new_shape: list[int]


@dataclass
class VectorOverwriteResult:
    """Result returned by POST /api/datasets/{id}/vectors/overwrite.

    Attributes
    ----------
    overwritten: Number of rows that were updated in-place.
    """

    overwritten: int


@dataclass
class SearchHit:
    """A single result returned by arro-server /search."""

    index: int
    score: float


class ArroClient:
    """Async HTTP client for arro-server.

    Parameters
    ----------
    base_url:   e.g. "http://localhost:8001"
    timeout:    httpx timeout in seconds (default 30.0)
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def dataset_metadata(self, dataset_id: str) -> dict | None:
        """GET /api/datasets/{dataset_id}/metadata

        Returns the metadata dict on 200, None on 404 (dataset not yet created).
        Raises ArroServerError on other non-2xx.
        """
        url = f"/api/datasets/{dataset_id}/metadata"
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

    async def upload_init(self, dataset_id: str, root_label: str) -> str:
        """POST /api/upload/init

        Body: {"dataset_id": dataset_id, "root": root_label}
        Returns upload_path (str) — the absolute filesystem path where
        the caller must write the Zarr v3 array.
        Raises ArroServerError on failure.
        """
        payload = {
            "dataset_id": dataset_id,
            "root": root_label,
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

    async def upload_commit(self, dataset_id: str, fs_path: str) -> UploadCommitResult:
        """POST /api/upload/commit

        Body: {"dataset_id": dataset_id, "fs_path": fs_path}
        Returns UploadCommitResult(index_stale, shape).
        Raises ArroServerError on failure.
        """
        payload = {
            "dataset_id": dataset_id,
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

    async def build_index(self, dataset_id: str, graph_params: dict | None = None) -> None:
        """POST /api/datasets/{dataset_id}/index

        Body: {"graph_params": graph_params} or {} for server defaults.
        Raises ArroServerError on failure.
        """
        payload: dict = {}
        if graph_params is not None:
            payload["graph_params"] = graph_params
        url = f"/api/datasets/{dataset_id}/index"
        try:
            response = await self._client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise ArroServerError(str(exc), status_code=None) from exc

        if response.status_code >= 400:
            raise ArroServerError(
                f"arro-server POST {url} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

    async def append_vectors(
        self,
        dataset_id: str,
        vectors: np.ndarray,
    ) -> VectorAppendResult:
        """POST /api/datasets/{dataset_id}/vectors/append

        Appends *vectors* as new rows at the end of the existing Zarr array.
        The server assigns row indices starting at the current array length --
        the caller must NOT assume a specific start_row before the call returns.

        Parameters
        ----------
        dataset_id: arro-server dataset identifier (e.g. "main--cve").
        vectors:    Float64 array of shape (M, dim). M >= 1.

        Returns
        -------
        VectorAppendResult with start_row, appended, and new_shape.

        Raises
        ------
        ArroServerError: any non-2xx response or network failure.
        ValueError:      if vectors is empty (shape[0] == 0).
        """
        if vectors.shape[0] == 0:
            raise ValueError("append_vectors: vectors must not be empty (shape[0] == 0)")

        payload = {"vectors": vectors.tolist()}
        url = f"/api/datasets/{dataset_id}/vectors/append"
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
        return VectorAppendResult(
            start_row=int(data["start_row"]),
            appended=int(data["appended"]),
            new_shape=list(data["new_shape"]),
        )

    async def overwrite_vectors(
        self,
        dataset_id: str,
        updates: list[tuple[int, np.ndarray]],
    ) -> VectorOverwriteResult:
        """POST /api/datasets/{dataset_id}/vectors/overwrite

        Overwrites specific rows in the Zarr array in-place.
        The array shape does not change -- only values at the given row indices
        are replaced. Row indices must already exist in the array.

        Parameters
        ----------
        dataset_id: arro-server dataset identifier.
        updates:    Non-empty list of (row_index, vector) pairs.
                    Each vector must have the same dimensionality as the array.

        Returns
        -------
        VectorOverwriteResult with overwritten count.

        Raises
        ------
        ArroServerError: any non-2xx response or network failure.
        ValueError:      if updates is empty.
        """
        if not updates:
            raise ValueError("overwrite_vectors: updates must not be empty")

        payload = {
            "updates": [{"row_index": row_idx, "vector": vec.tolist()} for row_idx, vec in updates]
        }
        url = f"/api/datasets/{dataset_id}/vectors/overwrite"
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
        return VectorOverwriteResult(overwritten=int(data["overwritten"]))

    async def get_vector_count(self, dataset_id: str) -> int:
        """GET /api/datasets/{dataset_id}/vectors/count

        Returns the current row count (nrows) of the dataset from the
        arro-server registry cache. This is an O(1) call when the cache
        is warm. The value is a snapshot -- a concurrent append may produce
        a higher value immediately after this call returns.

        Used as a pre-flight consistency guard before incremental writes:
        if the returned count differs from store.next_row_index(), the two
        stores are out of sync and the incremental path must not proceed.

        Parameters
        ----------
        dataset_id: arro-server dataset identifier.

        Returns
        -------
        nrows as int.

        Raises
        ------
        ArroServerError: any non-2xx response or network failure,
                         including 404 if the dataset does not exist.
        """
        url = f"/api/datasets/{dataset_id}/vectors/count"
        try:
            response = await self._client.get(url)
        except httpx.RequestError as exc:
            raise ArroServerError(str(exc), status_code=None) from exc

        if response.status_code >= 400:
            raise ArroServerError(
                f"arro-server GET {url} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )
        return int(response.json()["nrows"])

    async def search(
        self,
        dataset_id: str,
        vector: np.ndarray,
        top_k: int,
        tau: float,
    ) -> list[SearchHit]:
        """POST /api/datasets/{dataset_id}/search

        Body: {"vector": [float, ...], "k": top_k, "tau": tau, "mode": "tau"}
        Returns list[SearchHit] ordered by score descending (arro-server contract).
        Raises ArroServerError on any non-2xx or network failure.
        """
        payload = {
            "vector": vector.tolist(),
            "k": top_k,
            "tau": tau,
            "mode": "taumode",
        }
        url = f"/api/datasets/{dataset_id}/search"
        try:
            response = await self._client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise ArroServerError(str(exc), status_code=None) from exc

        if response.status_code >= 400:
            raise ArroServerError(
                f"arro-server POST {url} returned {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        raw = response.json()
        hits: list[dict] = raw["results"]
        return [SearchHit(index=int(hit["index"]), score=float(hit["score"])) for hit in hits]

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient."""
        await self._client.aclose()
