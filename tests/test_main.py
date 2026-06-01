from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from prompter_app import main
from prompter_app.providers import ProviderError
from prompter_app.schemas import GenerateRequest, PromptMessage, ProviderGenerateResult, Usage
from prompter_app.settings import Settings


class FakeProvider:
    def __init__(self) -> None:
        self.generate_calls = []

    def generate(self, request):
        self.generate_calls.append(request)
        return ProviderGenerateResult(
            provider="openrouter",
            model="openrouter/auto",
            output_text="generated reply",
            finish_reason="stop",
            usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
            request_id="gen_abc",
        )


class FailingProvider:
    def generate(self, request):
        raise ProviderError("provider exploded")


def test_healthcheck_reports_current_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "settings",
        Settings(
            llm_provider="openrouter",
            llm_base_url="https://openrouter.ai/api/v1",
            default_model="openrouter/auto",
            llm_api_key="provider-secret",
        ),
    )

    response = asyncio.run(main.healthcheck())

    assert response.model_dump() == {
        "status": "ok",
        "provider": "openrouter",
        "provider_base_url": "https://openrouter.ai/api/v1",
        "configured_model": "openrouter/auto",
        "provider_api_key_configured": True,
    }


def test_generate_requires_api_key_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(main, "settings", Settings(api_key="secret-key"))

    with pytest.raises(HTTPException) as error:
        main.require_api_key(None)

    assert error.value.status_code == 401
    assert error.value.detail == "unauthorized"


def test_generate_accepts_request_with_api_key(monkeypatch) -> None:
    fake_provider = FakeProvider()
    monkeypatch.setattr(main, "settings", Settings(api_key="secret-key"))
    monkeypatch.setattr(main, "provider", fake_provider)

    request = GenerateRequest(
        kind="extraction",
        messages=[
            PromptMessage(role="system", content="extract"),
            PromptMessage(role="user", content="I felt anxious"),
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=256,
    )
    response = asyncio.run(main.generate(request, None))

    assert response.model_dump() == {
        "kind": "extraction",
        "provider": "openrouter",
        "model": "openrouter/auto",
        "output_text": "generated reply",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
        "request_id": "gen_abc",
    }
    assert len(fake_provider.generate_calls) == 1
    assert fake_provider.generate_calls[0].kind == "extraction"
    assert fake_provider.generate_calls[0].response_format.type == "json_object"


def test_generate_maps_provider_errors_to_bad_gateway(monkeypatch) -> None:
    monkeypatch.setattr(main, "settings", Settings())
    monkeypatch.setattr(main, "provider", FailingProvider())

    request = GenerateRequest(
        kind="supportive",
        messages=[PromptMessage(role="user", content="hello")],
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(main.generate(request, None))

    assert error.value.status_code == 502
    assert error.value.detail == "provider exploded"
