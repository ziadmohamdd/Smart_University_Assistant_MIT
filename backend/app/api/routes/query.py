"""API routes: GET /health and POST /query.

Flow for /query:
    request -> validate (QueryRequest) -> RetrievalService.retrieve()
    -> GenerationService.answer_question() -> QueryResponse

Both services are loaded once at application startup (see app/main.py's
lifespan handler) and reused here as module-level singletons - no per-request
re-initialization of the embedding model, Chroma connection, or Ollama
client.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import generation_service
from app.services.retrieval import RetrievalServiceError, retrieval_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


def _format_source(source: dict[str, Any]) -> str:
    """Render a resolved source dict as a single human-readable string,
    e.g. '[S1] Course Catalog (p.42) — https://catalog.mit.edu'.
    """
    parts = [f"[{source['label']}]", source.get("title") or "Untitled"]
    if source.get("page") is not None:
        parts.append(f"(p.{source['page']})")
    if source.get("source"):
        parts.append(f"— {source['source']}")
    return " ".join(parts)


@router.get("/health")
def health() -> dict[str, Any]:
    """Simple liveness/readiness check.

    Confirms the application process is running and reports whether the
    retrieval and generation services finished loading successfully at
    startup.
    """
    return {
        "status": "healthy",
        "retrieval_ready": retrieval_service.is_ready,
        "generation_ready": generation_service.is_ready,
    }


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question using the persisted RAG pipeline.

    1. `request` is already validated by QueryRequest (non-empty question).
    2. Retrieve top-k chunks from the persisted Chroma collection.
    3. Build the grounded prompt and call the local Ollama model.
    4. Return the answer plus a human-readable list of cited sources.
    """
    question = request.question

    if not retrieval_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vector store is not available. The retrieval service failed "
                "to load at startup - check that CHROMA_PERSIST_DIR points to "
                "the persisted Chroma store and that the collection exists."
            ),
        )

    try:
        retrieved = retrieval_service.retrieve(question, k=settings.RETRIEVAL_TOP_K)
    except RetrievalServiceError as exc:
        logger.error("Retrieval failed for question %r: %s", question, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store error: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during retrieval for question %r", question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error during retrieval.",
        ) from exc

    if not generation_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Generation service is not available. Ollama failed to load "
                "at startup - check that `ollama serve` is running and that "
                "OLLAMA_MODEL is pulled."
            ),
        )

    try:
        result = generation_service.answer_question(question, retrieved)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during generation for question %r", question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error during generation.",
        ) from exc

    if result.get("error"):
        logger.error("Generation failed for question %r: %s", question, result["error"])
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama generation failed: {result['error']}",
        )

    sources = [_format_source(s) for s in result["sources"]]
    return QueryResponse(answer=result["answer"], sources=sources)
