# sentir-mais-prompter

Python API service for prompt execution in Sentir Mais.

## What it does

- exposes `POST /generate` for prompt-oriented requests from `sentir-mais-backend`
- exposes `GET /healthz`
- authenticates inter-service calls with a simple API key
- forwards requests to a backing LLM provider through a provider abstraction

The provider layer supports both OpenAI-compatible backends and a local Ollama backend.

## Run locally

Install:

```bash
pip install .
```

Run:

```bash
sentir-mais-prompter
```

Or:

```bash
python -m prompter_app.main
```

Default address:

- `http://localhost:8020`

## Tests

Create a virtual environment if you want isolation:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dev dependencies:

```bash
pip install ".[dev]"
```

Run the full test suite:

```bash
pytest
```

Run a single test file:

```bash
pytest tests/test_main.py
```

## Environment

- `HOST`
- `PORT`
- `API_KEY`
- `LLM_PROVIDER`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `DEFAULT_MODEL`
- `REQUEST_TIMEOUT_SECONDS`
- `APP_URL`
- `APP_TITLE`
- `ALLOW_MODEL_OVERRIDE`

Default LLM provider settings:

- `LLM_PROVIDER=openrouter`
- `LLM_BASE_URL=https://openrouter.ai/api/v1`
- `DEFAULT_MODEL=openrouter/auto`

### Local Ollama mode

Set `LOCAL_LLM=true` to route generation to a local Ollama instance instead of OpenRouter.

Defaults in local mode:

- `LLM_PROVIDER=ollama`
- `LLM_BASE_URL=http://127.0.0.1:11434`
- `DEFAULT_MODEL=qwen2.5:7b`

Recommended local model for this project:

- `qwen2.5:7b`

Reason:

- it is available in Ollama at about 4.7GB
- it is a good fit for structured JSON extraction and instruction following
- it is conservative enough to run comfortably on a 32GB RAM machine even if GPU acceleration is unavailable

Example setup:

```bash
ollama pull qwen2.5:7b
export LOCAL_LLM=true
export API_KEY=sentir-mais-local-prompter-key
sentir-mais-prompter
```

## API

### `GET /healthz`

Returns service readiness and the current provider configuration.

### `POST /generate`

If `API_KEY` is configured, requests must send:

```text
Authorization: <API_KEY>
```

Example request:

```json
{
  "kind": "supportive",
  "messages": [
    {
      "role": "system",
      "content": "You are a calm emotional support assistant."
    },
    {
      "role": "user",
      "content": "I had a very hard day and I feel overwhelmed."
    }
  ],
  "temperature": 0.7,
  "max_tokens": 300,
  "response_format": {
    "type": "text"
  }
}
```

Example response:

```json
{
  "kind": "supportive",
  "provider": "openrouter",
  "model": "openrouter/auto",
  "output_text": "I am sorry this felt so heavy today. Can you tell me what happened first?",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 112,
    "completion_tokens": 47,
    "total_tokens": 159
  },
  "request_id": "gen_123"
}
```
