from __future__ import annotations

import logging
import time
from uuid import uuid4
from json import dumps

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from prompter_app.providers import ProviderError, build_provider
from prompter_app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from prompter_app.settings import Settings

settings = Settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
provider = build_provider(settings)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or request.headers.get("X-Amzn-Trace-Id") or uuid4().hex
    started_at = time.monotonic()
    client = request.client.host if request.client else "unknown"
    logger.info(
        "http request started request_id=%s method=%s path=%s client=%s",
        request_id,
        request.method,
        request.url.path,
        client,
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "http request failed request_id=%s method=%s path=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            int((time.monotonic() - started_at) * 1000),
        )
        raise

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "http request completed request_id=%s method=%s path=%s status_code=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        int((time.monotonic() - started_at) * 1000),
    )
    return response


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return

    if authorization != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )


@app.get("/healthz", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(
        status="ok",
        provider=settings.llm_provider,
        provider_base_url=settings.llm_base_url,
        configured_model=settings.default_model,
        provider_api_key_configured=bool(settings.llm_api_key),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    _: None = Depends(require_api_key),
) -> GenerateResponse:
    request_summary = {
        "kind": request.kind,
        "message_count": len(request.messages),
        "message_roles": [message.role for message in request.messages],
        "model_override": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "response_format": request.response_format.type,
        "metadata_keys": sorted((request.metadata or {}).keys()),
    }
    logger.info("generate request received summary=%s", dumps(request_summary, sort_keys=True))

    try:
        result = provider.generate(request)
    except ProviderError as error:
        logger.exception("generate request failed summary=%s", dumps(request_summary, sort_keys=True))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error

    logger.info(
        "generate request completed summary=%s result=%s",
        dumps(request_summary, sort_keys=True),
        dumps(
            {
                "provider": result.provider,
                "resolved_model": result.model,
                "finish_reason": result.finish_reason,
                "request_id": result.request_id,
                "usage": result.usage.model_dump(),
                "output_length": len(result.output_text),
            },
            sort_keys=True,
        ),
    )
    return GenerateResponse(
        kind=request.kind,
        provider=result.provider,
        model=result.model,
        output_text=result.output_text,
        finish_reason=result.finish_reason,
        usage=result.usage,
        request_id=result.request_id,
    )


def run() -> None:
    logger.info(
        "starting prompter service provider=%s base_url=%s model=%s timeout_seconds=%s connect_timeout_seconds=%s pool_timeout_seconds=%s max_connections=%s max_keepalive_connections=%s log_level=%s",
        settings.llm_provider,
        settings.llm_base_url,
        settings.default_model,
        settings.request_timeout_seconds,
        settings.connect_timeout_seconds,
        settings.pool_timeout_seconds,
        settings.max_connections,
        settings.max_keepalive_connections,
        settings.log_level,
    )
    uvicorn.run(
        "prompter_app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
