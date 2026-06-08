from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    return raw.strip().lower() == "true"


def _default_llm_provider() -> str:
    if _env_bool("LOCAL_LLM", False):
        return "ollama"

    return os.getenv("LLM_PROVIDER", "openrouter")


def _default_llm_base_url() -> str:
    if _env_bool("LOCAL_LLM", False):
        return os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434")

    return os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")


def _default_model() -> str:
    if _env_bool("LOCAL_LLM", False):
        return os.getenv("DEFAULT_MODEL", "qwen2.5:7b")

    return os.getenv("DEFAULT_MODEL", "openrouter/auto")


@dataclass(frozen=True)
class Settings:
    app_name: str = "sentir-mais-prompter"
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8020")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    local_llm: bool = field(default_factory=lambda: _env_bool("LOCAL_LLM", False))
    llm_provider: str = field(default_factory=_default_llm_provider)
    llm_base_url: str = field(default_factory=_default_llm_base_url)
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    default_model: str = field(default_factory=_default_model)
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    )
    connect_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("CONNECT_TIMEOUT_SECONDS", "10"))
    )
    pool_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("POOL_TIMEOUT_SECONDS", "10"))
    )
    max_connections: int = field(default_factory=lambda: int(os.getenv("MAX_CONNECTIONS", "100")))
    max_keepalive_connections: int = field(
        default_factory=lambda: int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "20"))
    )
    app_url: str | None = field(default_factory=lambda: os.getenv("APP_URL") or None)
    app_title: str | None = field(
        default_factory=lambda: os.getenv("APP_TITLE", "sentir-mais-prompter") or None
    )
    allow_model_override: bool = field(
        default_factory=lambda: _env_bool("ALLOW_MODEL_OVERRIDE", True)
    )
