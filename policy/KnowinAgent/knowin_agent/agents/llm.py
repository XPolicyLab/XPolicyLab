"""Model clients. Agents talk to a ChatModel: messages + tool schemas in, text + tool calls out.

OpenAICompatModel speaks the Chat Completions tool-calling protocol. ScriptedModel
replays decisions for tests and offline dry runs.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import numpy as np


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""
    parse_error: Optional[str] = None


@dataclass
class ModelTurn:
    text: str
    tool_calls: list
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    raw: Optional[dict] = None


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list, tools: list) -> ModelTurn: ...


def image_part(rgb: np.ndarray, *, max_side: int = 512, quality: int = 80) -> dict:
    from PIL import Image

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    data = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def _parse_arguments(raw: str) -> tuple[dict, Optional[str]]:
    if not raw:
        return {}, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"arguments are not valid JSON: {exc}"
    if not isinstance(value, dict):
        return {}, "arguments must be a JSON object"
    return value, None


@dataclass
class ModelConfig:
    model: str
    base_url: str
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 4096
    reasoning_effort: Optional[str] = None
    extra_body: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)
    timeout_s: float = 300.0
    max_retries: int = 4

    @staticmethod
    def from_dict(values: dict) -> "ModelConfig":
        known = {f for f in ModelConfig.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown model config keys {sorted(unknown)}")
        return ModelConfig(**values)


class OpenAICompatModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.model

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    def complete(self, messages: list, tools: list) -> ModelTurn:
        cfg = self.config
        body = {"model": cfg.model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.max_tokens is not None:
            body["max_tokens"] = cfg.max_tokens
        if cfg.reasoning_effort:
            body["reasoning_effort"] = cfg.reasoning_effort
        body.update(cfg.extra_body)
        headers = {"Content-Type": "application/json", **cfg.headers}
        if cfg.api_key_env:
            key = os.environ.get(cfg.api_key_env)
            if not key:
                raise RuntimeError(f"environment variable {cfg.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        payload = json.dumps(body).encode()
        delay = 2.0
        started = time.time()
        for attempt in range(cfg.max_retries + 1):
            request = urllib.request.Request(self._endpoint(), data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=cfg.timeout_s) as response:
                    data = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:600]
                if exc.code in (408, 409, 429, 500, 502, 503, 504) and attempt < cfg.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"model HTTP {exc.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < cfg.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"model request failed: {exc}") from None
        message = data["choices"][0]["message"]
        calls = []
        for raw_call in message.get("tool_calls") or []:
            raw = raw_call.get("function", {}).get("arguments") or ""
            args, error = _parse_arguments(raw)
            calls.append(ToolCall(raw_call.get("id") or f"call_{len(calls)}", raw_call["function"]["name"], args, raw, error))
        text = message.get("content") or ""
        if isinstance(text, list):
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return ModelTurn(text=text, tool_calls=calls, usage=data.get("usage") or {}, latency_s=time.time() - started,
                         raw={"id": data.get("id"), "finish_reason": data["choices"][0].get("finish_reason")})


class ScriptedModel:
    """Replays decisions. Each item is a ModelTurn or a callable(messages, tools) -> ModelTurn."""

    def __init__(self, script: Sequence, name: str = "scripted"):
        self.script = list(script)
        self.name = name
        self.calls = 0
        self.seen: list = []

    def complete(self, messages: list, tools: list) -> ModelTurn:
        self.seen.append({"messages": messages, "tools": [t["function"]["name"] for t in tools]})
        if self.calls >= len(self.script):
            raise RuntimeError("scripted model ran out of decisions")
        item = self.script[self.calls]
        self.calls += 1
        return item(messages, tools) if callable(item) else item


def call(name: str, rationale: str = "", /, **arguments) -> ModelTurn:
    """Build a one-tool-call turn (scripts and tests)."""
    return ModelTurn(text=rationale, tool_calls=[ToolCall(f"call_{name}_{id(arguments)}", name, arguments,
                                                     json.dumps(arguments))])


def build_model(spec: dict) -> ChatModel:
    kind = spec.get("provider", "openai_compat")
    values = {k: v for k, v in spec.items() if k != "provider"}
    if kind == "openai_compat":
        return OpenAICompatModel(ModelConfig.from_dict(values))
    raise ValueError(f"unknown model provider {kind!r}")


LOADERS: dict[str, Callable[[dict], ChatModel]] = {"openai_compat": build_model}
