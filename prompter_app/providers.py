from __future__ import annotations

import json
import logging
from typing import Any, Protocol

import httpx

from prompter_app.schemas import GenerateRequest, ProviderGenerateResult, Usage
from prompter_app.settings import Settings


logger = logging.getLogger(__name__)


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

        request_summary = _build_request_summary(request, payload)
        logger.info(
            "dispatching openai-compatible request provider=%s base_url=%s summary=%s",
            self._settings.llm_provider,
            self._settings.llm_base_url,
            json.dumps(request_summary, sort_keys=True),
        )

        try:
            response = self._http_client.post(
                f"{self._settings.llm_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as error:
            logger.exception(
                "openai-compatible provider request transport failure provider=%s base_url=%s error_type=%s summary=%s",
                self._settings.llm_provider,
                self._settings.llm_base_url,
                type(error).__name__,
                json.dumps(request_summary, sort_keys=True),
            )
            raise ProviderError(f"provider request failed: {type(error).__name__}: {error}") from error

        if response.status_code < 200 or response.status_code >= 300:
            upstream_request_id = _extract_upstream_request_id(response)
            response_preview = _truncate_text(response.text)
            logger.error(
                "openai-compatible provider returned non-success status provider=%s base_url=%s status_code=%s upstream_request_id=%s summary=%s response_preview=%s",
                self._settings.llm_provider,
                self._settings.llm_base_url,
                response.status_code,
                upstream_request_id or "unknown",
                json.dumps(request_summary, sort_keys=True),
                response_preview,
            )
            raise ProviderError(
                f"provider returned status {response.status_code}"
                f" (request_id={upstream_request_id or 'unknown'}): {response_preview}"
            )

        try:
            body = response.json()
        except json.JSONDecodeError as error:
            upstream_request_id = _extract_upstream_request_id(response)
            response_preview = _truncate_text(response.text)
            logger.exception(
                "openai-compatible provider returned invalid JSON provider=%s base_url=%s status_code=%s upstream_request_id=%s summary=%s response_preview=%s",
                self._settings.llm_provider,
                self._settings.llm_base_url,
                response.status_code,
                upstream_request_id or "unknown",
                json.dumps(request_summary, sort_keys=True),
                response_preview,
            )
            raise ProviderError(
                f"provider returned invalid JSON (request_id={upstream_request_id or 'unknown'}): {error}"
            ) from error

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


class OllamaProvider:
    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._http_client = http_client or httpx.Client(timeout=settings.request_timeout_seconds)

    def generate(self, request: GenerateRequest) -> ProviderGenerateResult:
        if request.model and not self._settings.allow_model_override:
            raise ProviderError("model override is disabled")

        payload: dict[str, Any] = {
            "model": request.model or self._settings.default_model,
            "messages": [message.model_dump() for message in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.response_format.type == "json_object":
            payload["format"] = "json"

        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens

        request_summary = _build_request_summary(request, payload)
        logger.info(
            "dispatching ollama request base_url=%s summary=%s",
            self._settings.llm_base_url,
            json.dumps(request_summary, sort_keys=True),
        )

        try:
            response = self._http_client.post(
                f"{self._settings.llm_base_url.rstrip('/')}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        except httpx.HTTPError as error:
            logger.exception(
                "ollama provider request transport failure base_url=%s error_type=%s summary=%s",
                self._settings.llm_base_url,
                type(error).__name__,
                json.dumps(request_summary, sort_keys=True),
            )
            raise ProviderError(f"provider request failed: {type(error).__name__}: {error}") from error

        if response.status_code < 200 or response.status_code >= 300:
            upstream_request_id = _extract_upstream_request_id(response)
            response_preview = _truncate_text(response.text)
            logger.error(
                "ollama provider returned non-success status base_url=%s status_code=%s upstream_request_id=%s summary=%s response_preview=%s",
                self._settings.llm_base_url,
                response.status_code,
                upstream_request_id or "unknown",
                json.dumps(request_summary, sort_keys=True),
                response_preview,
            )
            raise ProviderError(
                f"provider returned status {response.status_code}"
                f" (request_id={upstream_request_id or 'unknown'}): {response_preview}"
            )

        try:
            body = response.json()
        except json.JSONDecodeError as error:
            upstream_request_id = _extract_upstream_request_id(response)
            response_preview = _truncate_text(response.text)
            logger.exception(
                "ollama provider returned invalid JSON base_url=%s status_code=%s upstream_request_id=%s summary=%s response_preview=%s",
                self._settings.llm_base_url,
                response.status_code,
                upstream_request_id or "unknown",
                json.dumps(request_summary, sort_keys=True),
                response_preview,
            )
            raise ProviderError(
                f"provider returned invalid JSON (request_id={upstream_request_id or 'unknown'}): {error}"
            ) from error

        message = body.get("message") or {}
        content = _extract_message_content(message.get("content"))
        if not content:
            raise ProviderError("provider response did not include assistant content")

        prompt_eval_count = body.get("prompt_eval_count", 0)
        eval_count = body.get("eval_count", 0)

        return ProviderGenerateResult(
            provider="ollama",
            model=body.get("model") or payload["model"],
            output_text=content,
            finish_reason=body.get("done_reason"),
            usage=Usage(
                prompt_tokens=prompt_eval_count,
                completion_tokens=eval_count,
                total_tokens=prompt_eval_count + eval_count,
            ),
            request_id=None,
            raw_response=body,
        )


def build_provider(settings: Settings) -> PromptProvider:
    provider = settings.llm_provider.strip().lower()
    if provider in {"openrouter", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleProvider(settings)
    if provider == "ollama":
        return OllamaProvider(settings)

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


def _build_request_summary(request: GenerateRequest, payload: dict[str, Any]) -> dict[str, Any]:
    options = payload.get("options")
    return {
        "kind": request.kind,
        "message_count": len(request.messages),
        "message_roles": [message.role for message in request.messages],
        "message_lengths": [len(message.content) for message in request.messages],
        "model": payload.get("model"),
        "temperature": payload.get("temperature")
        if payload.get("temperature") is not None
        else options.get("temperature")
        if isinstance(options, dict)
        else None,
        "max_tokens": payload.get("max_tokens")
        if payload.get("max_tokens") is not None
        else options.get("num_predict")
        if isinstance(options, dict)
        else None,
        "response_format": request.response_format.type,
        "metadata_keys": sorted((request.metadata or {}).keys()),
    }


def _extract_upstream_request_id(response: httpx.Response) -> str | None:
    for header_name in ("x-request-id", "request-id"):
        value = response.headers.get(header_name)
        if value:
            return value

    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None

    request_id = body.get("id")
    if isinstance(request_id, str) and request_id.strip():
        return request_id.strip()
    return None


def _truncate_text(value: str, limit: int = 1000) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."
