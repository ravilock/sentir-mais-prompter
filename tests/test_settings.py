from __future__ import annotations

from prompter_app.settings import Settings


def test_local_llm_defaults_to_ollama_and_qwen25(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_LLM", "true")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)

    settings = Settings()

    assert settings.local_llm is True
    assert settings.llm_provider == "ollama"
    assert settings.llm_base_url == "http://127.0.0.1:11434"
    assert settings.default_model == "qwen2.5:7b"


def test_explicit_env_overrides_local_llm_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_LLM", "true")
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("DEFAULT_MODEL", "llama3.1:8b")

    settings = Settings()

    assert settings.local_llm is True
    assert settings.llm_provider == "ollama"
    assert settings.llm_base_url == "http://ollama:11434"
    assert settings.default_model == "llama3.1:8b"
