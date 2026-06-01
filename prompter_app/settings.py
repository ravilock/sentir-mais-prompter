from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    return raw.strip().lower() == "true"


@dataclass(frozen=True)
class Settings:
    app_name: str = "sentir-mais-prompter"
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8020")))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openrouter"))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "openrouter/auto"))
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    )
    app_url: str | None = field(default_factory=lambda: os.getenv("APP_URL") or None)
    app_title: str | None = field(
        default_factory=lambda: os.getenv("APP_TITLE", "sentir-mais-prompter") or None
    )
    allow_model_override: bool = field(
        default_factory=lambda: _env_bool("ALLOW_MODEL_OVERRIDE", True)
    )
