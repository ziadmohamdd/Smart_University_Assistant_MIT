"""
Application configuration.

All values here mirror the configuration actually used in the source-of-truth
notebook `notebooks/rag_pipeline.ipynb`, which built and persisted the Chroma
vector store the backend must read from. Defaults are set to match that
notebook exactly; anything environment-specific (paths, hosts) is overridable
via `.env` so the project runs on any machine without editing code.

Do NOT change the defaults for EMBEDDING_MODEL_NAME or CHROMA_COLLECTION_NAME
unless the vector store is regenerated with new values — the embedding model
must match the one used to build the stored embeddings, and the collection
name must match the persisted Chroma collection, or retrieval will silently
return nothing or raise a "collection not found" error.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Vector store (Chroma) --------------------------------------------
    # Path to the persisted Chroma store created by the notebook
    # (chromadb.PersistentClient(path=...)). Relative to the backend dir by
    # default so the same repo works on any machine/OS.
    CHROMA_PERSIST_DIR: Path = BACKEND_DIR / "data" / "vector_store"

    # Exact collection name used by `chroma_client.create_collection(...)`
    # in the notebook. Must match exactly or `get_collection` will fail.
    CHROMA_COLLECTION_NAME: str = "rag_assistant_chunks"

    # --- Embeddings ----------------------------------------------------------
    # Must be the exact model used to embed the persisted chunks
    # (SentenceTransformer(EMBEDDING_MODEL_NAME) in the notebook), since
    # query embeddings must live in the same vector space as the stored ones.
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval -----------------------------------------------------------
    # Matches DEFAULT_K in the notebook's retrieve()/answer_question() flow.
    RETRIEVAL_TOP_K: int = 5

    # --- Ollama generation -----------------------------------------------
    # Matches OLLAMA_HOST / OLLAMA_MODEL in the notebook.
    OLLAMA_HOST: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"

    # Matches the `options` passed to ollama_client.chat(...) inside
    # generate_answer() in the notebook.
    OLLAMA_TEMPERATURE: float = 0.0
    OLLAMA_NUM_CTX: int = 8192
    OLLAMA_NUM_PREDICT: int = 400

    # --- API / CORS ------------------------------------------------------
    # Not part of the notebook (backend-only concern). Comma-separated list
    # of origins allowed to call this API, e.g. your frontend dev server.
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance, so .env is parsed once per process."""
    return Settings()


settings = get_settings()
