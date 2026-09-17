"""Tests for the SmartUniversityAssistant API (backend/app/main.py).

These are real API-behavior tests: they exercise the FastAPI app through
TestClient (HTTP request/response cycle, including validation and the
lifespan startup handler), rather than calling service functions directly.

Two of the four tests below depend on infrastructure this project does not
mock or fake:

- The persisted Chroma vector store at settings.CHROMA_PERSIST_DIR
  (built by notebooks/rag_pipeline.ipynb).
- A running local Ollama server with settings.OLLAMA_MODEL pulled.

Rather than mocking those out (which would test our code but not the actual
API behavior against the real pipeline, and risks masking a real
integration bug), the "valid question" test checks /health first and skips
itself with a clear reason if either dependency isn't available in the
current environment. This keeps the test honest: when the infrastructure is
present (e.g. on a machine with the vector store copied over and
`ollama serve` running), it fully exercises the real retrieve -> generate ->
respond flow and asserts on the real response.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Using the app as a context manager runs the lifespan handler, so
    # retrieval_service/generation_service attempt their real .load() calls
    # exactly as they would under `uvicorn app.main:app`.
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------
# 1. GET /health
# ---------------------------------------------------------------------
def test_health_returns_healthy(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


# ---------------------------------------------------------------------
# 2. POST /query with a valid question
# ---------------------------------------------------------------------
def test_query_with_valid_question(client: TestClient) -> None:
    health = client.get("/health").json()
    if not (health.get("retrieval_ready") and health.get("generation_ready")):
        pytest.skip(
            "Retrieval service (persisted Chroma vector store) and/or "
            "generation service (local Ollama server) are not available in "
            "this environment. This test requires both to be running to "
            "exercise the real /query flow - see README/.env for setup. "
            f"health = {health}"
        )

    response = client.post(
        "/query",
        json={"question": "What are the prerequisites for 18.03?"},
    )

    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)


# ---------------------------------------------------------------------
# 3. POST /query with invalid input
# ---------------------------------------------------------------------
def test_query_missing_question_field_returns_422(client: TestClient) -> None:
    # No "question" key at all in the request body.
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_blank_question_returns_422(client: TestClient) -> None:
    # "question" is present but fails QueryRequest's validation
    # (empty/whitespace-only string is rejected by the field_validator).
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422
