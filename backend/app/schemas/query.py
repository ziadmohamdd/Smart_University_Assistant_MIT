"""Request/response schemas for the /query endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Incoming request body for POST /query."""

    question: str = Field(
        ...,
        min_length=1,
        description="The user's natural-language question.",
        examples=["What are the prerequisites for 18.03?"],
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty or whitespace-only")
        return stripped


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str = Field(..., description="The grounded answer, with [S#] citations.")
    sources: list[str] = Field(
        default_factory=list,
        description="Human-readable descriptions of the sources actually cited in the answer.",
    )
