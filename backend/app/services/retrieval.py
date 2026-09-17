"""
Retrieval service.

Loads the embedding model and the already-persisted Chroma collection built by
notebooks/rag_pipeline.ipynb, and exposes a `retrieve()` function whose
behavior mirrors the notebook's `retrieve()` cell as closely as possible:

- same embedding model (query vectors comparable to stored document vectors)
- same collection name
- same Chroma query shape (documents, metadatas, distances)
- same similarity formula (similarity = 1 - distance, valid because the
  collection was created with hnsw:space = "cosine")
- same "source" fallback chain (source_url -> domain -> source_file)
- same output shape: rank, chunk_id, text, title, page, source, distance,
  similarity

This module does NOT build embeddings, does NOT create/rebuild the Chroma
collection, and does NOT touch raw/processed documents. It only opens the
existing persisted store at settings.CHROMA_PERSIST_DIR and reads from it.

Intended usage: instantiate `RetrievalService` once (e.g. in FastAPI's
startup/lifespan), then call `.retrieve(question, k)` per request.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from app.core.config import Settings, settings

logger = logging.getLogger(__name__)


class RetrievalServiceError(RuntimeError):
    """Raised when the retrieval service fails to initialize or query."""


class RetrievalService:
    """Loads the persisted Chroma collection + embedding model once, and
    serves retrieval requests against them without ever rebuilding either.
    """

    def __init__(self, settings_obj: Settings = settings) -> None:
        self._settings = settings_obj
        self._embedder: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[Collection] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the embedding model and open the existing persisted Chroma
        collection. Call once at startup. Raises RetrievalServiceError if
        the store/collection is missing, unreadable, or empty.
        """
        persist_dir = Path(self._settings.CHROMA_PERSIST_DIR)

        if not persist_dir.exists():
            raise RetrievalServiceError(
                f"Chroma persist directory not found: {persist_dir}. "
                "This backend does not build the vector store — copy the "
                "persisted store produced by notebooks/rag_pipeline.ipynb "
                "into this location (see CHROMA_PERSIST_DIR in .env)."
            )

        logger.info("Loading embedding model '%s'...", self._settings.EMBEDDING_MODEL_NAME)
        try:
            self._embedder = SentenceTransformer(self._settings.EMBEDDING_MODEL_NAME)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalServiceError(
                f"Failed to load embedding model '{self._settings.EMBEDDING_MODEL_NAME}': {exc}"
            ) from exc

        logger.info("Opening persisted Chroma store at '%s'...", persist_dir)
        try:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        except Exception as exc:  # noqa: BLE001
            raise RetrievalServiceError(
                f"Failed to open persisted Chroma store at '{persist_dir}': {exc}"
            ) from exc

        collection_name = self._settings.CHROMA_COLLECTION_NAME
        try:
            # get_collection (not get_or_create_collection): the collection
            # must already exist. If it doesn't, that's a setup error, not
            # something this service should silently "fix" by creating one.
            self._collection = self._client.get_collection(name=collection_name)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalServiceError(
                f"Chroma collection '{collection_name}' not found in "
                f"'{persist_dir}'. Verify CHROMA_COLLECTION_NAME matches the "
                "notebook and that the persisted store was copied correctly. "
                f"Original error: {exc}"
            ) from exc

        count = self._collection.count()
        if count == 0:
            raise RetrievalServiceError(
                f"Chroma collection '{collection_name}' exists but is empty "
                "(0 chunks). Verify you copied the fully-populated persisted "
                "store from the notebook, not an empty/partial one."
            )

        logger.info(
            "Retrieval service ready: collection '%s' loaded with %d chunks.",
            collection_name,
            count,
        )

    @property
    def is_ready(self) -> bool:
        return self._embedder is not None and self._collection is not None

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            raise RetrievalServiceError(
                "RetrievalService.load() must succeed before calling retrieve(). "
                "The service is not initialized."
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, question: str, k: Optional[int] = None) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant chunks for a natural-language
        question from the persisted Chroma collection.

        Mirrors notebooks/rag_pipeline.ipynb's `retrieve()` function exactly:
        same query embedding step, same Chroma query, same similarity
        formula, same output fields.

        Parameters
        ----------
        question : str
            The user's natural-language query.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.RETRIEVAL_TOP_K
            (mirrors DEFAULT_K in the notebook).

        Returns
        -------
        list[dict]
            One dict per retrieved chunk: rank, chunk_id, text, title, page,
            source, distance, similarity.
        """
        self._ensure_ready()

        if not question or not question.strip():
            raise RetrievalServiceError("question must be a non-empty string.")

        top_k = k if k is not None else self._settings.RETRIEVAL_TOP_K
        if top_k < 1:
            raise RetrievalServiceError("k must be >= 1.")

        try:
            query_embedding = self._embedder.encode(
                [question],
                normalize_embeddings=True,
            ).tolist()
        except Exception as exc:  # noqa: BLE001
            raise RetrievalServiceError(f"Failed to embed question: {exc}") from exc

        try:
            res = self._collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalServiceError(f"Chroma query failed: {exc}") from exc

        results: list[dict[str, Any]] = []
        ids = res.get("ids", [[]])[0]
        docs_out = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for rank, (chunk_id, text, meta, dist) in enumerate(
            zip(ids, docs_out, metas, dists), start=1
        ):
            meta = meta or {}
            source = meta.get("source_url") or meta.get("domain") or meta.get("source_file") or ""
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "text": text,
                    "title": meta.get("title", ""),
                    "page": meta.get("page"),
                    "source": source,
                    "distance": float(dist),
                    "similarity": 1.0 - float(dist),  # cosine space: similarity = 1 - distance
                }
            )

        return results


# ----------------------------------------------------------------------
# Module-level singleton, intended to be loaded once during FastAPI startup
# (e.g. in a lifespan handler) and reused across requests.
# ----------------------------------------------------------------------
retrieval_service = RetrievalService()
