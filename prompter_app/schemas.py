from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


AllowedRole = Literal["system", "user", "assistant", "developer"]
ResponseFormatType = Literal["text", "json_object"]


class PromptMessage(BaseModel):
    role: AllowedRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("content must not be blank")
        return normalized


class ResponseFormat(BaseModel):
    type: ResponseFormatType = "text"


class GenerateRequest(BaseModel):
    kind: str = Field(min_length=1, description="Logical request kind, such as supportive or extraction.")
    messages: list[PromptMessage] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional caller metadata for tracing or future routing.",
    )

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("kind must not be blank")
        return normalized

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return value

        normalized = value.strip()
        if not normalized:
            raise ValueError("model must not be blank")
        return normalized


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class GenerateResponse(BaseModel):
    kind: str
    provider: str
    model: str
    output_text: str
    finish_reason: str | None = None
    usage: Usage
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    provider: str
    provider_base_url: str
    configured_model: str
    provider_api_key_configured: bool


class ProviderGenerateResult(BaseModel):
    provider: str
    model: str
    output_text: str
    finish_reason: str | None = None
    usage: Usage
    request_id: str | None = None
    raw_response: dict[str, Any] | None = None
