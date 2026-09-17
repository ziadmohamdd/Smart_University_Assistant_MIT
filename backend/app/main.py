"""FastAPI application entrypoint.

Run from the backend/ directory with:
    uvicorn app.main:app --reload

Resource loading strategy
--------------------------
The embedding model, the persisted Chroma collection, and the Ollama client
are all expensive/stateful resources that must be created exactly once and
reused across requests - never rebuilt or reconnected per request. This is
done in the `lifespan` handler below, which runs once at process startup
(and once at shutdown), using the module-level `retrieval_service` and
`generation_service` singletons from app/services/.

If either service fails to load (e.g. the vector store is missing, or Ollama
isn't running), the app still starts - it does not crash on startup - but
/query will return 503 Service Unavailable until the underlying issue is
fixed and the process is restarted. This means /health can always be used to
check the app is up, and to see which of the two services are actually ready.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.generation import GenerationServiceError, generation_service
from app.services.retrieval import RetrievalServiceError, retrieval_service
from app.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SmartUniversityAssistant backend...")

    try:
        retrieval_service.load()
    except RetrievalServiceError as exc:
        logger.error(
            "Retrieval service failed to load - /query will return 503 until "
            "this is fixed and the app is restarted. Reason: %s",
            exc,
        )

    try:
        generation_service.load()
    except GenerationServiceError as exc:
        logger.error(
            "Generation service failed to load - /query will return 503 until "
            "this is fixed and the app is restarted. Reason: %s",
            exc,
        )

    logger.info(
        "Startup complete. retrieval_ready=%s generation_ready=%s",
        retrieval_service.is_ready,
        generation_service.is_ready,
    )

    yield

    logger.info("Shutting down SmartUniversityAssistant backend...")


app = FastAPI(
    title="SmartUniversityAssistant API",
    description=(
        "RAG backend for the SmartUniversityAssistant project. Answers "
        "questions grounded in a persisted Chroma vector store, using a "
        "local Ollama model for generation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
