"""HTTP helpers for model providers: endpoint URLs, headers, content parts.

Chat Completions (OpenRouter and OpenAI-compatible servers such as vLLM, SGLang
or services/servers/serve_vlm.py) is the one wire format the runtime speaks. Messages use
the harness content parts (`input_text`, `input_image`).
"""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CHAT_COMPLETIONS = "/chat/completions"


def urllib_transport(url, headers, payload, timeout):
    encoded = json.dumps(payload, allow_nan=False).encode()
    request = Request(url, data=encoded, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Provider HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Provider request failed: {error.reason}") from error
    return json.loads(raw)


def chat_url(provider, base_url):
    """The Chat Completions URL of a model config (backend, base_url)."""
    if provider == "openrouter" and not base_url:
        return "https://openrouter.ai/api/v1/chat/completions"
    if not base_url:
        raise ValueError("openai_chat requires base_url")
    root = base_url.rstrip("/")
    if root.endswith(CHAT_COMPLETIONS):
        return root
    return root + CHAT_COMPLETIONS


def bearer(api_key):
    return {
        "content-type": "application/json",
        "user-agent": "the-harness",
        "authorization": f"Bearer {api_key}",
    }


def chat_part(part):
    if part["type"] == "input_text":
        return {"type": "text", "text": part["text"]}
    if part["type"] == "input_image":
        return {"type": "image_url", "image_url": {"url": part["image_url"]}}
    raise ValueError(f"Unsupported content part: {part['type']}")


def redact(text, secret):
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text
