from __future__ import annotations

import pytest
from pydantic import ValidationError

from prompter_app.schemas import GenerateRequest, PromptMessage


def test_prompt_message_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        PromptMessage(role="user", content="   ")


def test_generate_request_rejects_blank_kind() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(
            kind="   ",
            messages=[PromptMessage(role="user", content="hello")],
        )


def test_generate_request_normalizes_valid_values() -> None:
    request = GenerateRequest(
        kind=" supportive ",
        model=" openrouter/auto ",
        messages=[PromptMessage(role="user", content=" hello ")],
    )

    assert request.kind == "supportive"
    assert request.model == "openrouter/auto"
    assert request.messages[0].content == "hello"
