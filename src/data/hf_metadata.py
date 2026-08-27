"""Read dataset METADATA from the Hugging Face datasets-server.

This exists so we can audit a dataset's structure -- resolutions, formats,
generator names, real-image sources -- without downloading hundreds of GB of
image bytes. Community Forensics is 260GB; its resolution metadata is a few
hundred kilobytes, and that metadata is enough to discover a fatal shortcut
before committing any compute.

The datasets-server is a public read-only REST API. It rate-limits and
occasionally returns a transient error while it warms a split, so every call
retries with backoff.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, Sequence
from urllib.parse import quote

import requests

__all__ = [
    "DatasetsServerError",
    "fetch_rows",
    "iter_rows",
    "summarize_column",
]

BASE_URL = "https://datasets-server.huggingface.co"

# The server caps a single response; 100 rows is the documented safe page size.
MAX_PAGE = 100


class DatasetsServerError(RuntimeError):
    """The datasets-server returned an error we could not recover from."""


def _get(
    endpoint: str,
    params: dict[str, str],
    retries: int = 5,
    backoff: float = 3.0,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """GET with retries. The server returns 200 with an {"error": ...} body for
    some failures, so we check the payload, not just the status code."""
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                f"{BASE_URL}/{endpoint}", params=params, timeout=timeout
            )
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
        else:
            if response.ok and "error" not in payload:
                return payload
            last_error = payload.get("error", f"HTTP {response.status_code}")

        if attempt < retries:
            time.sleep(backoff * attempt)

    raise DatasetsServerError(
        f"{endpoint} failed after {retries} attempts: {last_error}"
    )


def fetch_rows(
    dataset: str,
    columns: Sequence[str],
    where: str | None = None,
    offset: int = 0,
    length: int = MAX_PAGE,
    config: str = "default",
    split: str = "train",
) -> tuple[list[dict[str, Any]], int]:
    """Fetch one page of rows.

    `where` is a SQL-ish predicate understood by the /filter endpoint, e.g.
    '"label"=0'. When it is None we use /rows instead, which is cheaper.

    Returns (rows, total_row_count).
    """
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "columns": ",".join(columns),
        "offset": str(offset),
        "length": str(min(length, MAX_PAGE)),
    }

    if where is None:
        payload = _get("rows", params)
    else:
        params["where"] = where
        payload = _get("filter", params)

    rows = [item["row"] for item in payload.get("rows", [])]
    total = int(payload.get("num_rows_total", len(rows)))
    return rows, total


def iter_rows(
    dataset: str,
    columns: Sequence[str],
    where: str | None = None,
    limit: int = 1000,
    start_offset: int = 0,
    config: str = "default",
    split: str = "train",
) -> Iterator[dict[str, Any]]:
    """Page through up to `limit` rows.

    Stops early when the server runs out of rows, so asking for more than
    exists is safe.
    """
    fetched = 0
    offset = start_offset

    while fetched < limit:
        page_size = min(MAX_PAGE, limit - fetched)
        rows, total = fetch_rows(
            dataset,
            columns,
            where=where,
            offset=offset,
            length=page_size,
            config=config,
            split=split,
        )
        if not rows:
            return

        for row in rows:
            yield row

        fetched += len(rows)
        offset += len(rows)

        if offset >= total:
            return


@dataclass
class ColumnSummary:
    column: str
    counts: Counter
    n_rows: int

    def most_common(self, n: int = 10) -> list[tuple[Any, int]]:
        return self.counts.most_common(n)


def summarize_column(
    dataset: str,
    column: str,
    where: str | None = None,
    limit: int = 500,
    config: str = "default",
    split: str = "train",
) -> ColumnSummary:
    """Value distribution for one column, for auditing dataset composition."""
    counts: Counter = Counter()
    n_rows = 0

    for row in iter_rows(
        dataset, [column], where=where, limit=limit, config=config, split=split
    ):
        value = row.get(column)
        # Resolutions come back as lists, which aren't hashable.
        if isinstance(value, list):
            value = tuple(value)
        counts[value] += 1
        n_rows += 1

    return ColumnSummary(column=column, counts=counts, n_rows=n_rows)
