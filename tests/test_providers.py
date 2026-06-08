from __future__ import annotations

import json

import httpx
import pytest

from prompter_app.providers import OllamaProvider, OpenAICompatibleProvider, ProviderError
from prompter_app.schemas import GenerateRequest, PromptMessage
from prompter_app.settings import Settings


def test_openai_compatible_provider_maps_request_and_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["referer"] = request.headers.get("HTTP-Referer")
        captured["title"] = request.headers.get("X-Title")
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            status_code=200,
            json={
                "id": "gen_123",
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "structured output",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    provider = OpenAICompatibleProvider(
        Settings(
            llm_api_key="provider-secret",
            app_url="https://sentir-mais.app",
            app_title="sentir-mais-prompter",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = provider.generate(
        GenerateRequest(
            kind="extraction",
            model="openai/gpt-4o-mini",
            messages=[
                PromptMessage(role="system", content="extract JSON"),
                PromptMessage(role="user", content="I argued with my boss"),
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.2,
        )
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["authorization"] == "Bearer provider-secret"
    assert captured["referer"] == "https://sentir-mais.app"
    assert captured["title"] == "sentir-mais-prompter"
    assert captured["payload"] == {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "extract JSON"},
            {"role": "user", "content": "I argued with my boss"},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    assert result.output_text == "structured output"
    assert result.request_id == "gen_123"
    assert result.usage.total_tokens == 15


def test_openai_compatible_provider_requires_llm_api_key() -> None:
    provider = OpenAICompatibleProvider(Settings(llm_api_key=""))

    with pytest.raises(ProviderError, match="LLM_API_KEY is not configured"):
        provider.generate(
            GenerateRequest(
                kind="supportive",
                messages=[PromptMessage(role="user", content="hello")],
            )
        )


def test_ollama_provider_maps_request_and_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen2.5:7b",
                "message": {
                    "role": "assistant",
                    "content": '{"enough_context":true}',
                },
                "done_reason": "stop",
                "prompt_eval_count": 42,
                "eval_count": 17,
            },
        )

    provider = OllamaProvider(
        Settings(
            local_llm=True,
            llm_provider="ollama",
            llm_base_url="http://127.0.0.1:11434",
            default_model="qwen2.5:7b",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = provider.generate(
        GenerateRequest(
            kind="extraction",
            messages=[
                PromptMessage(role="system", content="extract JSON"),
                PromptMessage(role="user", content="I argued with my boss"),
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.2,
        )
    )

    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"] == {
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": "extract JSON"},
            {"role": "user", "content": "I argued with my boss"},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 300,
        },
        "format": "json",
    }
    assert result.output_text == '{"enough_context":true}'
    assert result.model == "qwen2.5:7b"
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 42
    assert result.usage.completion_tokens == 17
    assert result.usage.total_tokens == 59
