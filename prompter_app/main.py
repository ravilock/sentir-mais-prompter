from __future__ import annotations

import logging
from json import dumps

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status

from prompter_app.providers import ProviderError, build_provider
from prompter_app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from prompter_app.settings import Settings

settings = Settings()
provider = build_provider(settings)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)


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
    uvicorn.run(
        "prompter_app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
