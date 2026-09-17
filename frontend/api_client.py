"""API client for the SmartUniversityAssistant backend.

Talks to the FastAPI backend's GET /health and POST /query endpoints over
plain HTTP (requests). Contains no Streamlit or other UI code - app.py is
expected to call these functions and handle the exceptions/data itself.

Configuration
-------------
The backend URL is read from the API_BASE_URL environment variable
(loaded from frontend/.env via python-dotenv). It is never hardcoded here -
see frontend/.env.example for the expected format.

Error handling
--------------
All failures are raised as one of the APIClientError subclasses below,
rather than returned as sentinel values, so app.py can catch a single base
class if it wants a generic "something went wrong" branch, or catch specific
subclasses for tailored messages (e.g. distinguishing "backend is down" from
"backend rejected the question").
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

FRONTEND_DIR = Path(__file__).resolve().parent
load_dotenv(FRONTEND_DIR / ".env")

# Generation through a local, possibly CPU-only Ollama model can take a
# while; /query gets a longer timeout than /health.
HEALTH_TIMEOUT_SECONDS = 5.0
QUERY_TIMEOUT_SECONDS = 60.0


class APIClientError(Exception):
    """Base class for all errors raised by this API client."""


class APIConfigError(APIClientError):
    """Raised when the client itself is misconfigured (e.g. missing API_BASE_URL)."""


class APIConnectionError(APIClientError):
    """Raised when the backend cannot be reached at all."""


class APITimeoutError(APIClientError):
    """Raised when the backend does not respond within the configured timeout."""


class APIHTTPError(APIClientError):
    """Raised when the backend responds with a non-2xx HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


class APIInvalidResponseError(APIClientError):
    """Raised when the backend responds with 2xx but an unexpected/malformed body."""


def _get_base_url() -> str:
    """Read and validate API_BASE_URL from the environment.

    Deliberately does not fall back to a hardcoded default (e.g.
    http://localhost:8000) - if it's missing, that's a configuration problem
    the person running the app should fix via frontend/.env, not something
    this client should silently paper over.
    """
    base_url = os.environ.get("API_BASE_URL")
    if not base_url or not base_url.strip():
        raise APIConfigError(
            "API_BASE_URL is not set. Copy frontend/.env.example to "
            "frontend/.env and set API_BASE_URL to the backend's URL "
            "(e.g. http://localhost:8000)."
        )
    return base_url.strip().rstrip("/")


def _extract_error_detail(response: requests.Response) -> Optional[str]:
    """Best-effort extraction of FastAPI's {"detail": "..."} error body."""
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
    return None


def _request(method: str, path: str, timeout: float, **kwargs: Any) -> requests.Response:
    """Shared request wrapper: builds the URL and converts requests-level
    failures into this module's exception types.
    """
    base_url = _get_base_url()
    url = f"{base_url}{path}"

    try:
        return requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.Timeout as exc:
        raise APITimeoutError(
            f"Request to {url} timed out after {timeout}s."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise APIConnectionError(
            f"Could not connect to backend at {url}. Is the backend running?"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise APIClientError(f"Unexpected error calling {url}: {exc}") from exc


def check_health() -> dict[str, Any]:
    """Call GET /health and return the backend's health payload.

    Returns
    -------
    dict
        The backend's JSON response, e.g.
        {"status": "healthy", "retrieval_ready": bool, "generation_ready": bool}

    Raises
    ------
    APIConfigError, APIConnectionError, APITimeoutError, APIHTTPError,
    APIInvalidResponseError
    """
    response = _request("GET", "/health", timeout=HEALTH_TIMEOUT_SECONDS)

    if response.status_code != 200:
        raise APIHTTPError(
            response.status_code,
            _extract_error_detail(response)
            or f"Health check failed with HTTP {response.status_code}.",
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise APIInvalidResponseError(
            f"/health returned a non-JSON response: {response.text[:200]!r}"
        ) from exc

    if not isinstance(data, dict) or "status" not in data:
        raise APIInvalidResponseError(f"/health returned an unexpected payload: {data!r}")

    return data


def ask_question(question: str) -> dict[str, Any]:
    """Call POST /query with {"question": question} and return the answer.

    Parameters
    ----------
    question : str
        The user's natural-language question. Must be non-empty.

    Returns
    -------
    dict
        {"answer": str, "sources": list[str]}

    Raises
    ------
    ValueError
        If `question` is empty/whitespace-only (checked client-side before
        any network call).
    APIConfigError, APIConnectionError, APITimeoutError, APIHTTPError,
    APIInvalidResponseError
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty.")

    response = _request(
        "POST",
        "/query",
        timeout=QUERY_TIMEOUT_SECONDS,
        json={"question": question.strip()},
    )

    if response.status_code != 200:
        raise APIHTTPError(
            response.status_code,
            _extract_error_detail(response)
            or f"/query failed with HTTP {response.status_code}.",
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise APIInvalidResponseError(
            f"/query returned a non-JSON response: {response.text[:200]!r}"
        ) from exc

    if not isinstance(data, dict) or "answer" not in data or "sources" not in data:
        raise APIInvalidResponseError(f"/query returned an unexpected payload shape: {data!r}")

    if not isinstance(data["answer"], str):
        raise APIInvalidResponseError(f"/query 'answer' field is not a string: {data['answer']!r}")

    if not isinstance(data["sources"], list):
        raise APIInvalidResponseError(f"/query 'sources' field is not a list: {data['sources']!r}")

    return {"answer": data["answer"], "sources": data["sources"]}
