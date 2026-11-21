"""Multi-turn text completion for the high level.

Chat Completions covers OpenRouter and self-hosted OpenAI-compatible servers
(vLLM, SGLang) through `openai_chat` with a `base_url`. Messages use the
harness content parts (`input_text`, `input_image`).
"""

from dataclasses import dataclass, field
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..core.http import bearer, chat_part, chat_url, redact, urllib_transport


@dataclass
class ChatReply:
    """One completion: its text, usage (input and output tokens, cost in USD when the
    provider reports it, finish_reason), the provider's response id, and the seconds the
    request took with its retries."""

    text: str
    usage: dict = field(default_factory=dict)
    response_id: str | None = None
    seconds: float = 0.0
    # Reasoning text or summary, when the provider returns one (e.g. DeepSeek, GPT-6
    # Luna on OpenRouter); for the trace only, never sent back to the model.
    reasoning: str = ""


# The keys of a model role (configs/models/<name>.toml: [main], [vlm], [pointer],
# [verifier]). Documentation; ChatModel reads them.
MODEL_KEYS = {
    "service": "a service of the deploy instead of a hosted model: its client config, with this role's keys on top",
    "backend": "openrouter (default) or openai_chat (any OpenAI-compatible server)",
    "model": "the model id",
    "base_url": "openai_chat: the server's URL, e.g. http://127.0.0.1:8800/v1",
    "base_url_env": "openai_chat: environment variable holding the server URL; overrides base_url when set",
    "api_key_env": "the environment variable that holds the key (default OPENROUTER_API_KEY for openrouter)",
    "reasoning_effort": "openrouter: none, low, medium or high",
    "max_output_tokens": "tokens per reply, reasoning included on OpenRouter (default 8192)",
    "temperature": "sampling temperature (default: the provider's)",
    "stream": "stream the reply: a dead connection shows within the read timeout while a model thinks",
    "timeout_seconds": "read timeout of one request (default 300)",
    "retries": "retries of a failed request (default 2)",
    "retry_seconds": "keep retrying transient failures until this long after the request started",
    "extra_body": "extra request fields for OpenAI-compatible servers (e.g. max_image_side)",
    "on_demand": "written by the resolution for an on-demand service: the broker starts it first",
}


class ChatModel:
    """A Chat Completions client for one model role (MODEL_KEYS): OpenRouter or an
    OpenAI-compatible server. The openrouter backend needs its key in the environment;
    `transport` replaces the HTTP call (tests)."""

    def __init__(self, config: dict, transport=None):
        self.provider = config.get("backend", "openrouter")
        if self.provider not in {"openrouter", "openai_chat"}:
            raise ValueError("The program harness supports the openrouter and openai_chat backends")
        self.model = config["model"]
        # A named variable wins over a URL written in the file, so an evaluator points
        # the same frozen agent at their own server without changing its identity.
        named = config.get("base_url_env")
        from_env = os.environ.get(named, "").strip() if named else ""
        self.base_url = from_env or config.get("base_url")
        if self.provider == "openai_chat" and not self.base_url:
            where = f" Set {named}." if named else ""
            raise ValueError(f"openai_chat requires base_url.{where}")
        # A self-hosted model placed on demand is started by the deploy's broker first.
        self.on_demand = config.get("on_demand")
        self.url = chat_url(self.provider, self.base_url)
        self.timeout = float(config.get("timeout_seconds", 300.0))
        self.max_output_tokens = int(config.get("max_output_tokens", 8192))
        self.reasoning_effort = config.get("reasoning_effort")
        self.temperature = config.get("temperature")
        self.retries = int(config.get("retries", 2))
        # Keep retrying transient failures until this many seconds have passed since the
        # request started (at least `retries` times). The simulator does not advance while
        # the model is being called, so waiting out an outage costs wall time only.
        self.retry_seconds = config.get("retry_seconds")
        self.retry_seconds = None if self.retry_seconds is None else float(self.retry_seconds)
        # Extra request fields for OpenAI-compatible servers (e.g. max_image_side for serve_vlm.py).
        self.extra_body = dict(config.get("extra_body") or {})
        # Streaming keeps bytes flowing while a reasoning model thinks; some networks reset
        # a connection that stays silent for minutes waiting for a non-streamed reply.
        self.stream = bool(config.get("stream", False))
        self.transport = transport or (sse_transport if self.stream else urllib_transport)
        key_env = config.get("api_key_env") or ("OPENROUTER_API_KEY" if self.provider == "openrouter" else None)
        self._api_key = os.environ.get(key_env) if key_env else None
        if self.provider == "openrouter" and not self._api_key:
            raise ValueError(f"Set {key_env} before selecting the openrouter backend")
        self._api_key = self._api_key or "EMPTY"

    def payload(self, system: str, messages: list[dict]) -> dict:
        """The request body: `system` and the harness messages as chat messages, the output
        limit and extra_body fields, and on OpenRouter usage accounting and reasoning."""
        # Some models (Molmo2) take no system message; an empty one is left out.
        converted = [{"role": "system", "content": system}] if system else []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                converted.append({"role": message["role"], "content": content})
            else:
                converted.append({"role": message["role"],
                                  "content": [chat_part(part) for part in content]})
        payload = {"model": self.model, "messages": converted, "max_tokens": self.max_output_tokens,
                   **self.extra_body}
        if self.stream:
            payload["stream"] = True
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.provider == "openrouter":
            payload["usage"] = {"include": True}
            if self.reasoning_effort == "none":
                payload["reasoning"] = {"enabled": False}
            elif self.reasoning_effort:
                payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    def complete(self, system: str, messages: list[dict]) -> ChatReply:
        """One completion of `messages` under `system`, starting an on-demand service through
        the broker first. Failed requests are retried (`retries` times, or until
        `retry_seconds` have passed), except client errors other than HTTP 429; the last
        failure is raised as a RuntimeError."""
        if self.on_demand:
            from ..services.broker import ensure

            ensure({"on_demand": self.on_demand})
        payload = self.payload(system, messages)
        started = time.perf_counter()
        last_error = None
        # A completion has no side effects, so transient provider failures may be retried.
        attempt = 0
        while True:
            if attempt:
                waited = time.perf_counter() - started
                if attempt > self.retries and (self.retry_seconds is None or waited >= self.retry_seconds):
                    raise last_error
                time.sleep(min(2 ** (attempt - 1) * 5, 30 if self.retry_seconds is None else 60))
            attempt += 1
            try:
                body = self.transport(self.url, bearer(self._api_key), payload, self.timeout)
            except Exception as error:
                last_error = RuntimeError(redact(str(error), self._api_key))
                if "Provider HTTP 4" in str(error) and "HTTP 429" not in str(error):
                    raise last_error from None  # a client error will not pass on retry
                continue
            if isinstance(body, dict) and body.get("error"):
                detail = body["error"]
                text = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
                last_error = RuntimeError(f"{self.provider} error: {redact(text, self._api_key)}")
                continue
            choices = body.get("choices") or []
            if len(choices) != 1:
                raise ValueError(f"Expected one completion choice, got {len(choices)}")
            message = choices[0].get("message") or {}
            usage = body.get("usage") or {}
            return ChatReply(
                text=message.get("content") or "",
                reasoning=message.get("reasoning") or "",
                usage={"input_tokens": usage.get("prompt_tokens", 0),
                       "output_tokens": usage.get("completion_tokens", 0),
                       "cost": usage.get("cost"),
                       "finish_reason": choices[0].get("finish_reason")},
                response_id=body.get("id"),
                seconds=time.perf_counter() - started,
            )


def sse_transport(url, headers, payload, timeout):
    """POST a streamed chat completion and assemble the chunks into the non-streamed
    response shape. `timeout` bounds each read, so keep-alive comments reset it."""
    request = Request(url, data=json.dumps(payload, allow_nan=False).encode(), headers=headers, method="POST")
    body = {"id": None, "usage": None}
    content = []
    try:
        with urlopen(request, timeout=timeout) as response:
            return _read_sse(response, body, content)
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Provider HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Provider request failed: {error.reason}") from error
    except OSError as error:
        raise RuntimeError(f"Provider request failed: {error}") from error


def _read_sse(lines, body, content):
    finish = None
    reasoning = []
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue  # blank separators and ": keep-alive" comments
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            return {"error": chunk["error"]}
        body["id"] = body["id"] or chunk.get("id")
        if chunk.get("usage"):
            body["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning") or "")
            finish = choice.get("finish_reason") or finish
    if finish is None:
        # The connection closed before the completion finished (the provider dropped the
        # stream): a transport failure to retry, not an empty reply.
        raise OSError(f"stream ended before the reply finished ({len(''.join(content))} characters received)")
    return {"id": body["id"], "usage": body["usage"] or {},
            "choices": [{"message": {"content": "".join(content),
                                     **({"reasoning": "".join(reasoning)} if any(reasoning) else {})},
                         "finish_reason": finish}]}


def scripted_models(script) -> dict:
    """The models of a scripted episode: `script` is a list of replies, or
    {"replies": [...], "verifier": [...]} as scripts/replay_script.py writes it; without
    recorded verifier answers the verifier says YES."""
    if isinstance(script, list):
        script = {"replies": script}
    return {"chat": ScriptedChat(script["replies"]),
            "verifier": ScriptedChat(script.get("verifier") or ["YES (scripted verifier)"] * 100)}


class ScriptedChat:
    """Replays fixed replies in order. For pipeline checks without model calls."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system: str, messages: list[dict]) -> ChatReply:
        """The next scripted reply (a `<finish>` block once they run out), with zero usage."""
        if self.calls >= len(self.replies):
            text = "<finish>scripted replies exhausted</finish>"
        else:
            text = self.replies[self.calls]
        self.calls += 1
        return ChatReply(text=text, usage={"input_tokens": 0, "output_tokens": 0, "cost": 0.0})
