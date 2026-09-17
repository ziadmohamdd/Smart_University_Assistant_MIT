"""
Generation service.

Reuses the exact grounding philosophy, prompt structure, and citation format
from notebooks/rag_pipeline.ipynb:

- SYSTEM_PROMPT: same wording as the notebook (retrieval-grounded assistant,
  no outside knowledge).
- GROUNDING_RULES: same 7 rules (cite every factual sentence with [S1]/[S2]/...,
  never invent a label, preserve exact names/dates/course numbers, refuse via
  NOT_IN_CONTEXT when unsupported, partial-answer handling, ~150 word limit,
  no preamble).
- format_context(): same "[S1] Title — page N — source" header + truncated
  body, joined with "---" separators.
- build_rag_prompt(): same CONTEXT / QUESTION / GROUNDING_RULES / ANSWER
  structure.
- generate_answer(): same ollama_client.chat(...) call shape and options
  (temperature, num_ctx, num_predict).
- extract_cited_sources(): same [S\\d+] regex, same ordered de-duplication,
  same resolved source fields.

This module does NOT call any external LLM API — only the local Ollama
server, via the `ollama` Python client, configured through Settings
(OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_NUM_CTX,
OLLAMA_NUM_PREDICT). The Ollama client is created once and reused.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import ollama

from app.core.config import Settings, settings

logger = logging.getLogger(__name__)


class GenerationServiceError(RuntimeError):
    """Raised when the generation service fails to initialize or generate."""


# ----------------------------------------------------------------------
# Prompt constants — copied verbatim from notebooks/rag_pipeline.ipynb
# (Section: RAG prompting), so backend behavior matches the tested notebook.
# ----------------------------------------------------------------------

CONTEXT_CHUNK_MAX_WORDS = 350  # per-chunk truncation so k=5 chunks fit the context window
NOT_IN_CONTEXT_PREFIX = "NOT_IN_CONTEXT"

SYSTEM_PROMPT = """You are a retrieval-grounded document assistant for a university \
document collection (MIT course catalog and registrar pages).

You answer questions using ONLY the numbered context passages given to you in each \
request. The passages are the single source of truth. You have no other knowledge about \
this institution, and you must never rely on anything you may have seen during training.

You are precise, concise and factual. You would rather say that the answer is not in the \
context than produce a plausible-sounding guess."""

GROUNDING_RULES = f"""GROUNDING AND CITATION RULES (follow all of them):
1. Use ONLY the information in the CONTEXT section above. Do not use outside or prior \
knowledge, and do not infer facts that are not written there.
2. Every factual sentence in your answer must end with the label(s) of the passage(s) it \
came from, e.g. "The add date is March 15 [S2]." Use several labels when a sentence draws \
on several passages, e.g. "[S1][S3]".
3. Never invent a source label. Only use labels that actually appear in the CONTEXT \
section.
4. Quote names, dates, course numbers and requirements exactly as they appear in the \
context. Do not round, reword or normalise them.
5. If the context does not contain enough information to answer the question, reply with \
exactly one line, and nothing else:
   {NOT_IN_CONTEXT_PREFIX}: The retrieved documents do not contain information to answer \
this question.
   Do this even if you believe you know the answer from general knowledge.
6. If the context only partially answers the question, answer the supported part with its \
citations and then state plainly which part is not covered by the retrieved documents.
7. Keep the answer under about 150 words. Do not add a preamble, and do not repeat these \
rules back."""

CITATION_RE = re.compile(r"\[(S\d+)\]")


def format_context(
    results: list[dict[str, Any]], max_words: int = CONTEXT_CHUNK_MAX_WORDS
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Render retrieved chunks as labelled, citable context blocks ([S1], [S2], ...).

    Returns (context_string, label_map) where label_map maps "S1" -> the
    result dict, so the caller can resolve the citations the model produced
    back to real documents. Mirrors the notebook's format_context() exactly.
    """
    blocks, label_map = [], {}
    for i, r in enumerate(results, start=1):
        label = f"S{i}"
        label_map[label] = r

        words = (r.get("text") or "").split()
        text = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")

        header = f'[{label}] "{r.get("title") or "Untitled"}" — page {r.get("page")}'
        source = r.get("source") or ""
        if source:
            header += f" — {source}"

        blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(blocks), label_map


def build_rag_prompt(
    question: str, results: list[dict[str, Any]]
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Assemble the full RAG user prompt: context + question + grounding/citation rules.

    The system instructions are sent separately as the chat `system` message,
    so the four required prompt parts are: SYSTEM_PROMPT, CONTEXT, QUESTION,
    GROUNDING_RULES. Mirrors the notebook's build_rag_prompt() exactly.
    """
    context_str, label_map = format_context(results)
    if not context_str:
        context_str = "(no passages were retrieved for this question)"

    prompt = f"""CONTEXT — numbered passages retrieved from the document collection:

{context_str}

=== END OF CONTEXT ===

QUESTION:
{question}

{GROUNDING_RULES}

ANSWER:"""
    return prompt, label_map


def extract_cited_sources(
    answer_text: str, label_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the ordered, de-duplicated list of sources the answer actually
    cites. Mirrors the notebook's extract_cited_sources() exactly.
    """
    cited = []
    for label in CITATION_RE.findall(answer_text or ""):
        if label in label_map and label not in cited:
            cited.append(label)

    sources = []
    for label in cited:
        r = label_map[label]
        sources.append(
            {
                "label": label,
                "title": r.get("title") or "Untitled",
                "page": r.get("page"),
                "source": r.get("source") or "",
                "chunk_id": r.get("chunk_id"),
                "similarity": round(float(r.get("similarity", 0.0)), 3),
            }
        )
    return sources


def _extract_message_content(response: Any) -> str:
    """Return the assistant text from an ollama chat response.

    Works with both the dict-style responses of older `ollama` versions and
    the pydantic-style objects returned by newer ones. Mirrors the notebook's
    _extract_message_content() exactly.
    """
    if isinstance(response, dict):
        return (response.get("message") or {}).get("content", "")
    message = getattr(response, "message", None)
    return getattr(message, "content", "") if message is not None else ""


class GenerationService:
    """Wraps the local Ollama client and the notebook's grounded RAG prompting
    logic. The Ollama client is created once and reused across requests.
    """

    def __init__(self, settings_obj: Settings = settings) -> None:
        self._settings = settings_obj
        self._client: Optional[ollama.Client] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Create the reusable Ollama client and verify the server/model are
        reachable. Call once at startup.
        """
        logger.info("Connecting to Ollama at '%s'...", self._settings.OLLAMA_HOST)
        try:
            self._client = ollama.Client(host=self._settings.OLLAMA_HOST)
        except Exception as exc:  # noqa: BLE001
            raise GenerationServiceError(
                f"Failed to create Ollama client for host '{self._settings.OLLAMA_HOST}': {exc}"
            ) from exc

        try:
            models_resp = self._client.list()
        except Exception as exc:  # noqa: BLE001
            raise GenerationServiceError(
                f"Could not reach Ollama server at '{self._settings.OLLAMA_HOST}'. "
                "Make sure `ollama serve` is running. "
                f"Original error: {exc}"
            ) from exc

        model_names = self._model_names_from_list(models_resp)
        target = self._settings.OLLAMA_MODEL
        if not any(
            n == target or n.split(":")[0] == target.split(":")[0] for n in model_names
        ):
            raise GenerationServiceError(
                f"Model '{target}' is not registered with the Ollama server at "
                f"'{self._settings.OLLAMA_HOST}'. Run `ollama pull {target}` first. "
                f"Models currently available: {model_names or '(none)'}"
            )

        logger.info(
            "Generation service ready: model '%s' verified on Ollama server.", target
        )

    @staticmethod
    def _model_names_from_list(models_resp: Any) -> list[str]:
        """Extract model name strings from ollama_client.list(), tolerating
        both dict-style and pydantic-style responses.
        """
        models = (
            models_resp.get("models")
            if isinstance(models_resp, dict)
            else getattr(models_resp, "models", None)
        ) or []
        names = []
        for m in models:
            name = m.get("name") if isinstance(m, dict) else getattr(m, "model", None)
            if name:
                names.append(name)
        return names

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            raise GenerationServiceError(
                "GenerationService.load() must succeed before calling generate(). "
                "The service is not initialized."
            )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_answer(self, prompt: str, model: Optional[str] = None) -> str:
        """Send the RAG prompt to the local Ollama LLM and return the raw
        answer text. Mirrors the notebook's generate_answer() exactly,
        with temperature/num_ctx/num_predict sourced from config instead of
        hardcoded defaults.
        """
        self._ensure_ready()

        try:
            response = self._client.chat(
                model=model or self._settings.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": self._settings.OLLAMA_TEMPERATURE,
                    "num_ctx": self._settings.OLLAMA_NUM_CTX,
                    "num_predict": self._settings.OLLAMA_NUM_PREDICT,
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationServiceError(f"Ollama generation failed: {exc}") from exc

        return _extract_message_content(response).strip()

    def answer_question(
        self,
        question: str,
        retrieved: list[dict[str, Any]],
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build the grounded prompt from already-retrieved chunks, call
        Ollama, and extract cited sources.

        This does NOT perform retrieval itself — the caller (e.g. an API
        route composing RetrievalService + GenerationService) passes in the
        chunks already retrieved via RetrievalService.retrieve(). This keeps
        the generation service independently testable and matches the
        notebook's separation of retrieve() / build_rag_prompt() /
        generate_answer().

        Returns
        -------
        dict with keys: question, answer, sources, grounded, prompt, error
        """
        prompt, label_map = build_rag_prompt(question, retrieved)

        out: dict[str, Any] = {
            "question": question,
            "answer": "",
            "sources": [],
            "grounded": False,
            "prompt": prompt,
            "error": None,
        }

        try:
            answer = self.generate_answer(prompt, model=model)
        except GenerationServiceError as exc:
            out["error"] = str(exc)
            out["answer"] = f"[generation unavailable] {exc}"
            return out

        out["answer"] = answer
        out["sources"] = extract_cited_sources(answer, label_map)
        out["grounded"] = (
            not answer.upper().startswith(NOT_IN_CONTEXT_PREFIX) and bool(out["sources"])
        )
        return out


# ----------------------------------------------------------------------
# Module-level singleton, intended to be loaded once during FastAPI startup
# (e.g. in a lifespan handler) and reused across requests.
# ----------------------------------------------------------------------
generation_service = GenerationService()
