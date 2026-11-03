"""The review teacher over an OpenAI-compatible Chat Completions endpoint: any other model provider.

Gemini, Qwen, DeepSeek, Kimi, GLM and self-hosted servers (vLLM, SGLang) all serve ``POST {base_url}/chat/completions``
in OpenAI's shape. Same contract as ``teacher.Conversation`` + ``teacher.ResponsesClient``, so ``TeacherBrain`` is
unchanged: the frozen prompt as the system message, the ``robodojo_execute`` tool (strict schema) with the call
required, the same packets, images, limits, retries and pruning. The key is read from the file the frozen config names
and sent only in the Authorization header; provider errors are reduced like the Responses client's.

Provider differences, forced by the wire format: a tool result carries text only on most of these servers, so a packet
that answers a call is a ``tool`` message with the packet text followed by a ``user`` message with its camera views;
reasoning is not carried between calls (Chat Completions has no field for it); ``reasoning_effort`` is sent as the
request's ``reasoning_effort`` (servers without the knob ignore it or refuse it at preflight's first call).
"""
from __future__ import annotations

import copy
import json
import threading
import time
import urllib.error
import urllib.request

from .teacher import (FATAL_CODES, FATAL_STATUS, FATAL_TYPES, RETRYABLE_STATUS, TeacherFatal, TeacherUnavailable,
                      _compact_packet_text, mask, provider_error, read_api_key)
from .review import tool_spec

API = "openai_chat"
PRUNED_VIEW = {"type": "text", "text": "[earlier camera view omitted]"}


def chat_tool(spec: dict | None = None) -> dict:
    """The teacher's tool as a Chat Completions function (``spec``: the Responses-shaped tool, default RoboDojo's)."""
    spec = spec if spec is not None else tool_spec()
    return {"type": "function", "function": {"name": spec["name"], "description": spec["description"],
                                             "parameters": spec["parameters"], "strict": True}}


def _image(item: dict) -> dict:
    return {"type": "image_url", "image_url": {"url": item["image_url"]}}


class ChatConversation:
    """The persistent ``messages`` list (``items``), the system prompt first."""

    def __init__(self, cfg: dict, episode_id: str, prompt: str, tool: dict | None = None):
        self.cfg = cfg
        self.tool = chat_tool(tool)
        self.items: list = [{"role": "system", "content": prompt}]
        self.cache_key = f"oeh-review-{episode_id}"
        self.pending_call: str | None = None
        self.last_input_tokens = 0
        self.prunes = 0
        self.recorded = 0

    def body(self) -> dict:
        body = {"model": self.cfg["model"], "input": self.items, "tools": [self.tool], "tool_choice": "required",
                "transport": API, "reasoning": {"effort": self.cfg["reasoning_effort"]},
                "prompt_cache_key": self.cache_key}
        if self.cfg.get("max_output_tokens"):
            body["max_output_tokens"] = self.cfg["max_output_tokens"]
        return body

    def _answer(self, text_parts: list, images: list):
        if self.pending_call is not None:
            self.items.append({"role": "tool", "tool_call_id": self.pending_call,
                               "content": "\n\n".join(part["text"] for part in text_parts)})
            self.pending_call = None
            if images:
                self.items.append({"role": "user", "content": [
                    {"type": "text", "text": "Camera views for the packet above."}, *images]})
        else:
            self.items.append({"role": "user", "content": text_parts + images})

    def add_packet(self, text: str, images: list[dict], intro: str | None = None):
        head = [{"type": "text", "text": intro}] if intro else []
        self._answer(head + [{"type": "text", "text": text}], [_image(image) for image in images])

    def add_error(self, text: str):
        self._answer([{"type": "text", "text": text}], [])

    def add_output(self, output: list) -> dict | None:
        """``output`` holds the assistant message; return its last tool call in the brain's shape."""
        call = None
        for message in output:
            kept = {k: v for k, v in message.items() if k in ("role", "content", "tool_calls")}
            kept["role"] = "assistant"
            self.items.append(copy.deepcopy(kept))
            for tool_call in message.get("tool_calls") or []:
                call = tool_call
        if call is None:
            return None
        self.pending_call = call["id"]
        function = call.get("function") or {}
        if function.get("name") != "robodojo_execute":
            return None
        return {"type": "function_call", "call_id": call["id"], "name": function["name"],
                "arguments": function.get("arguments") or "{}"}

    def unrecorded(self) -> list:
        items, self.recorded = self.items[self.recorded:], len(self.items)
        return items

    def prune(self) -> dict | None:
        if self.last_input_tokens < self.cfg.get("prune_above_tokens", 180_000):
            return None
        keep, seen, views, tables = self.cfg.get("context_images", 6), 0, 0, 0
        # A packet is its text (a user message, or a tool message's content) and the views that follow it.
        for item in reversed(self.items[1:]):
            content = item.get("content")
            if item["role"] == "tool" and isinstance(content, str) and '"student_proposal"' in content:
                seen += 1
                if seen > keep:
                    compact = _compact_packet_text(content)
                    tables += compact != content
                    item["content"] = compact
                continue
            if item["role"] != "user" or not isinstance(content, list):
                continue
            holds_packet = any(p.get("type") == "text" and '"student_proposal"' in p.get("text", "") for p in content)
            if holds_packet:
                seen += 1
            stale = seen > keep if holds_packet else seen >= keep  # views after a tool message belong to it
            if not stale:
                continue
            replaced = []
            for part in content:
                if part.get("type") == "image_url":
                    replaced.append(dict(PRUNED_VIEW))
                    views += 1
                elif part.get("type") == "text" and '"student_proposal"' in part.get("text", ""):
                    compact = _compact_packet_text(part["text"])
                    tables += compact != part["text"]
                    replaced.append({**part, "text": compact})
                else:
                    replaced.append(part)
            item["content"] = replaced
        if not views and not tables:
            return None
        self.prunes += 1
        self.recorded = min(self.recorded, 1)
        return {"views_removed": views, "tables_removed": tables, "kept_packets": keep,
                "input_tokens_before": self.last_input_tokens}


class ChatClient:
    """POST /chat/completions with bounded retries. Blocking: the brain runs it in a worker thread."""

    def __init__(self, cfg: dict, key: str | None = None):
        self.cfg = cfg
        self.base = cfg["base_url"].rstrip("/")
        self._key = key if key is not None else read_api_key(cfg["api_key_file"])
        self.closed = threading.Event()

    def _request(self, path: str, body: dict | None, timeout: float):
        request = urllib.request.Request(self.base + path, method="POST" if body is not None else "GET",
                                         data=None if body is None else json.dumps(body).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + self._key,
                                                  "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def preflight(self) -> dict:
        """The key lists the frozen model (``GET /models``; ids with a ``models/`` prefix count as the bare id)."""
        try:
            data = self._request("/models", None, 30)
        except urllib.error.HTTPError as exc:
            error = provider_error(exc.code, exc.read()[:2000])
            raise TeacherFatal(f"teacher preflight refused: {error}", [error]) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TeacherUnavailable(f"teacher preflight unreachable: {type(exc).__name__}: {exc}") from None
        ids = {str(m.get("id", "")).removeprefix("models/") for m in data.get("data") or []}
        if self.cfg["model"] not in ids:
            raise TeacherFatal(f"teacher preflight: the endpoint does not list {self.cfg['model']!r}", [
                {"status": 404, "type": "invalid_request_error", "code": "model_not_found"}])
        return {"model": self.cfg["model"], "transport": API}

    def _payload(self, body: dict) -> dict:
        payload = {"model": body["model"], "messages": body["input"], "tools": body["tools"],
                   "tool_choice": body["tool_choice"]}
        if body["reasoning"]["effort"] != "none":
            payload["reasoning_effort"] = body["reasoning"]["effort"]
        if body.get("max_output_tokens"):
            payload["max_completion_tokens"] = body["max_output_tokens"]
        return payload

    def create(self, body: dict) -> tuple[dict, list[dict]]:
        attempts = self.cfg.get("max_attempts", 4)
        delays = self.cfg.get("retry_delays_s") or [5]
        errors = []
        for attempt in range(attempts):
            if self.closed.is_set():
                raise TeacherUnavailable("teacher client closed", errors)
            started = time.monotonic()
            try:
                data = self._request("/chat/completions", self._payload(body), self.cfg["request_timeout_s"])
                choice = (data.get("choices") or [{}])[0]
                if choice.get("finish_reason") == "length":
                    raise RuntimeError("response stopped at the output limit (finish_reason length)")
                return self._responses_shape(data, choice), errors
            except urllib.error.HTTPError as exc:
                error = provider_error(exc.code, exc.read()[:2000])
                fatal = (exc.code in FATAL_STATUS or error.get("code") in FATAL_CODES
                         or error.get("type") in FATAL_TYPES)
                retry = not fatal and exc.code in RETRYABLE_STATUS
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
                error = {"status": None, "type": type(exc).__name__,
                         "message": mask(str(exc))[:400]}
                fatal, retry = False, True
            errors.append({**error, "attempt": attempt + 1, "seconds": round(time.monotonic() - started, 3),
                           "retry": retry and attempt + 1 < attempts, "fatal": fatal})
            if fatal:
                raise TeacherFatal(f"teacher refused: {error}", errors)
            if not retry or attempt + 1 == attempts:
                break
            if self.closed.wait(delays[min(attempt, len(delays) - 1)]):
                break
        raise TeacherUnavailable(f"teacher unavailable after {len(errors)} attempt(s): {errors[-1] if errors else ''}",
                                 errors)

    @staticmethod
    def _responses_shape(data: dict, choice: dict) -> dict:
        usage = data.get("usage") or {}
        return {"id": data.get("id"), "status": "completed", "output": [choice.get("message") or {}],
                "stop": {"finish_reason": choice.get("finish_reason")},
                "usage": {"input_tokens": usage.get("prompt_tokens"),
                          "input_tokens_details": {"cached_tokens": (usage.get("prompt_tokens_details") or {})
                                                   .get("cached_tokens") or 0},
                          "output_tokens": usage.get("completion_tokens"),
                          "output_tokens_details": {"reasoning_tokens": (usage.get("completion_tokens_details") or {})
                                                    .get("reasoning_tokens")},
                          "total_tokens": usage.get("total_tokens")}}
