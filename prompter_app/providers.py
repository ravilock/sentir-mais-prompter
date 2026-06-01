from __future__ import annotations

from typing import Any, Protocol

import httpx

from prompter_app.schemas import GenerateRequest, ProviderGenerateResult, Usage
from prompter_app.settings import Settings


class ProviderError(RuntimeError):
    """Raised when the backing LLM provider cannot fulfill a request."""


class PromptProvider(Protocol):
    def generate(self, request: GenerateRequest) -> ProviderGenerateResult: ...


class OpenAICompatibleProvider:
    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._http_client = http_client or httpx.Client(timeout=settings.request_timeout_seconds)

    def generate(self, request: GenerateRequest) -> ProviderGenerateResult:
        if not self._settings.llm_api_key:
            raise ProviderError("LLM_API_KEY is not configured")

        if request.model and not self._settings.allow_model_override:
            raise ProviderError("model override is disabled")

        payload: dict[str, Any] = {
            "model": request.model or self._settings.default_model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.response_format.type == "json_object":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.app_url:
            headers["HTTP-Referer"] = self._settings.app_url
        if self._settings.app_title:
            headers["X-Title"] = self._settings.app_title

        response = self._http_client.post(
            f"{self._settings.llm_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise ProviderError(f"provider returned status {response.status_code}: {response.text}")

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ProviderError("provider response did not include choices")

        message = choices[0].get("message") or {}
        content = _extract_message_content(message.get("content"))
        if not content:
            raise ProviderError("provider response did not include assistant content")

        usage = body.get("usage") or {}
        return ProviderGenerateResult(
            provider=self._settings.llm_provider,
            model=body.get("model") or payload["model"],
            output_text=content,
            finish_reason=choices[0].get("finish_reason"),
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            request_id=body.get("id"),
            raw_response=body,
        )


def build_provider(settings: Settings) -> PromptProvider:
    provider = settings.llm_provider.strip().lower()
    if provider in {"openrouter", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleProvider(settings)

    raise ValueError(f"unsupported LLM_PROVIDER: {settings.llm_provider}")


def _extract_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts)

    return ""
