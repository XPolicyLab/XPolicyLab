"""Small, explicit HTTP model boundary; no credentials or provider state on disk."""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str
    api_key_env: str = "PHYSICALRSI_API_KEY"
    protocol: str = "chat_completions"
    timeout_s: float = 120
    max_output_tokens: int = 4096
    max_tool_rounds: int = 8

    def __post_init__(self):
        url = urllib.parse.urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("base_url must be an HTTP(S) API root")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Credentials and query strings do not belong in base_url")
        if not self.model or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("Provide a model and an API key environment variable name")
        if self.protocol not in {"chat_completions", "responses"}:
            raise ValueError("protocol must be chat_completions or responses")
        if not 0 < self.timeout_s <= 600 or not 1 <= self.max_tool_rounds <= 32:
            raise ValueError("Invalid model timeout or tool round budget")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

    def public(self):
        return asdict(self)


class LanguageModel:
    def __init__(self, config, *, transport=None):
        self.config = config
        self.transport = transport or self._post

    def _post(self, endpoint, body):
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise ValueError(f"Set {self.config.api_key_env} before using conversation")
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + endpoint,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        # Never retry automatically: the caller may already have executed tools.
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_s
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Model API returned HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            raise RuntimeError(
                "Model API connection failed or returned invalid JSON"
            ) from None

    def complete(self, history, tools):
        """Return text/calls and native items to preserve reasoning/tool continuity."""
        if self.config.protocol == "responses":
            body = dict(
                model=self.config.model,
                input=history,
                tools=[dict(type="function", **tool) for tool in tools],
                max_output_tokens=self.config.max_output_tokens,
                store=False,
            )
            result = self.transport("/responses", body)
            if result.get("status") != "completed":
                raise RuntimeError("Model response did not complete; no tools executed")
            items = result["output"]
            calls = [
                dict(id=item["call_id"], name=item["name"], arguments=item["arguments"])
                for item in items
                if item["type"] == "function_call"
            ]
            text = "\n".join(
                part["text"]
                for item in items
                if item["type"] == "message"
                for part in item["content"]
                if part["type"] == "output_text"
            )
            return text, calls, items
        result = self.transport(
            "/chat/completions",
            dict(
                model=self.config.model,
                messages=history,
                tools=[dict(type="function", function=tool) for tool in tools],
                max_completion_tokens=self.config.max_output_tokens,
            ),
        )
        choice = result["choices"][0]
        if choice["finish_reason"] not in {"stop", "tool_calls"}:
            raise RuntimeError("Model response did not complete; no tools executed")
        message = choice["message"]
        calls = [
            dict(id=call["id"], **call["function"])
            for call in message.get("tool_calls") or []
        ]
        native = {
            key: message[key]
            for key in ("role", "content", "tool_calls")
            if key in message
        }
        return message.get("content") or "", calls, [native]

    def tool_result(self, call_id, value):
        output = json.dumps(value, ensure_ascii=False)
        if self.config.protocol == "responses":
            return dict(type="function_call_output", call_id=call_id, output=output)
        return dict(role="tool", tool_call_id=call_id, content=output)
