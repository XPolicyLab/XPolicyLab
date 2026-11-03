"""The review teacher over the Anthropic Messages API: a Claude model in the GPT teacher's place.

Same contract as ``teacher.Conversation`` + ``teacher.ResponsesClient``, so ``TeacherBrain`` is unchanged and a Claude
teacher and a GPT teacher differ only where the provider makes them: the same frozen prompt (as the ``system``
prompt), the same ``robodojo_execute`` tool (its strict schema as ``input_schema``), the same first-packet text, the
same packets, images, limits, retries and pruning thresholds. One persistent conversation per episode; every packet
answers the previous tool call as a ``tool_result``; the whole history, with the model's thinking blocks unchanged, is
sent back each call through the official ``anthropic`` SDK (streaming, so a long thinking turn cannot hit an HTTP
timeout before ``request_timeout_s``).

Provider differences, each forced by the API rather than chosen:

- A call cannot be forced (``tool_choice`` ``any``/``tool`` is refused by the current models): ``auto`` with one tool,
  parallel calls off, and the first-packet text asking for one ``robodojo_execute`` call. An answer without the call
  is returned to the model for correction exactly as a GPT answer without one is.
- ``reasoning_effort`` is ``output_config.effort`` with adaptive thinking (``low`` .. ``max``; Claude has no ``none``
  or ``minimal``). Thinking is summarized so call artifacts carry a readable trace; its signature is recorded as a
  hash, like GPT's encrypted reasoning.
- Pruning (``prune_above_tokens``, ``context_images``) edits earlier turns, and the API refuses a replayed thinking
  block whose earlier history changed. A prune therefore also drops every earlier thinking block (once, at the prune;
  later blocks are kept again). The prune also runs before the request would pass the API's own image count or
  request size limits, which a GPT conversation never meets first.

The key is read from the file the frozen config names and handed to the SDK client only; nothing here logs, records
or returns it, and provider errors are reduced to status, type and code like the Responses client's.
"""
from __future__ import annotations

import copy
import json
import threading
import time

from .review import tool_spec
from .teacher import (FATAL_STATUS, RETRYABLE_STATUS, TeacherFatal, TeacherUnavailable, _compact_packet_text, mask,
                      provider_error, read_api_key)

API = "anthropic_messages"
DEFAULT_BASE_URL = "https://api.anthropic.com"
EFFORTS = {"low", "medium", "high", "xhigh", "max"}
# 529 is Anthropic's "overloaded"; the rest are the Responses client's retryable statuses.
RETRYABLE = RETRYABLE_STATUS | {529}
# A spent balance arrives as 402 billing_error (or a 400 naming the credit balance): no retry can help, and the
# evaluator already counts an exhausted budget (type insufficient_quota) as infrastructure.
BILLING = {"status": 402, "type": "insufficient_quota", "code": "anthropic_billing_error"}
# The Messages API's own request limits (vision docs: 600 images per request on 1M-context models; 32 MB per
# request). The prune runs before either is reached: base64 images dominate the request size.
MAX_IMAGES = 600
MAX_IMAGE_BYTES = 24_000_000
# max_output_tokens: None means the model's own output limit, read at preflight (Responses: no cap).
FALLBACK_MAX_TOKENS = 64_000
PRUNED_VIEW = {"type": "text", "text": "[earlier camera view omitted]"}


def anthropic_tool(spec: dict | None = None) -> dict:
    """The teacher's tool as a Messages API tool (``spec``: the Responses-shaped tool, default RoboDojo's)."""
    spec = spec if spec is not None else tool_spec()
    return {"name": spec["name"], "description": spec["description"], "input_schema": spec["parameters"],
            "strict": True}


def image_block(item: dict) -> dict:
    """A Responses ``input_image`` (PNG data URL) as a Messages base64 image block (the same bytes)."""
    url = item["image_url"]
    prefix, _, data = url.partition(",")
    media = prefix[len("data:"):].split(";", 1)[0] or "image/png"
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def _images(blocks) -> int:
    count = 0
    for block in blocks if isinstance(blocks, list) else []:
        if block.get("type") == "image":
            count += 1
        elif block.get("type") == "tool_result":
            count += _images(block.get("content"))
    return count


def _image_bytes(blocks) -> int:
    size = 0
    for block in blocks if isinstance(blocks, list) else []:
        if block.get("type") == "image":
            size += len((block.get("source") or {}).get("data") or "")
        elif block.get("type") == "tool_result":
            size += _image_bytes(block.get("content"))
    return size


class AnthropicConversation:
    """The persistent ``messages`` list (``items``). User turns strictly alternate with assistant turns: a packet that
    follows a failed call joins the trailing user turn instead of starting a second one."""

    def __init__(self, cfg: dict, episode_id: str, prompt: str, tool: dict | None = None):
        self.cfg = cfg
        self.tool = anthropic_tool(tool)
        self.system = prompt
        self.items: list = []
        self.cache_key = f"oeh-review-{episode_id}"
        self.pending_call: str | None = None
        self.last_input_tokens = 0
        self.prunes = 0
        self.recorded = 0

    def body(self) -> dict:
        return {"model": self.cfg["model"], "input": self.items, "system": self.system, "tools": [self.tool],
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}, "transport": API,
                "reasoning": {"effort": self.cfg["reasoning_effort"]}, "prompt_cache_key": self.cache_key,
                "max_tokens": self.cfg.get("max_output_tokens")}  # None: the model's own limit (client preflight)

    def _user(self, content: list):
        if self.items and self.items[-1]["role"] == "user":
            last = self.items.pop()
            self.recorded = min(self.recorded, len(self.items))  # the merged turn is recorded again
            content = list(last["content"]) + content
        self.items.append({"role": "user", "content": content})

    def _answer(self, content: list):
        if self.pending_call is None:
            self._user(content)
        else:
            self._user([{"type": "tool_result", "tool_use_id": self.pending_call, "content": content}])
            self.pending_call = None

    def add_packet(self, text: str, images: list[dict], intro: str | None = None):
        head = [{"type": "text", "text": intro}] if intro else []
        self._answer(head + [{"type": "text", "text": text}] + [image_block(image) for image in images])

    def add_error(self, text: str):
        if self.pending_call is not None:
            self._user([{"type": "tool_result", "tool_use_id": self.pending_call, "is_error": True,
                         "content": [{"type": "text", "text": text}]}])
            self.pending_call = None
        else:
            self._user([{"type": "text", "text": text}])

    def add_output(self, output: list) -> dict | None:
        """Append the assistant turn (its content blocks, thinking included, unchanged); return its tool call in the
        shape the brain reads (``arguments`` as JSON text), if any."""
        blocks = [copy.deepcopy(block) for block in output]
        if not blocks:
            return None
        self.items.append({"role": "assistant", "content": blocks})
        call = None
        for block in blocks:
            if block.get("type") == "tool_use":
                call = block
        if call is None:
            return None
        self.pending_call = call["id"]
        if call.get("name") != "robodojo_execute":
            return None
        return {"type": "function_call", "call_id": call["id"], "name": call["name"],
                "arguments": json.dumps(call.get("input") or {})}

    def unrecorded(self) -> list:
        items, self.recorded = self.items[self.recorded:], len(self.items)
        return items

    def _needs_prune(self) -> str | None:
        if self.last_input_tokens >= self.cfg.get("prune_above_tokens", 180_000):
            return "input_tokens"
        contents = [item["content"] for item in self.items]
        if sum(_images(content) for content in contents) > MAX_IMAGES - 3 * self.cfg.get("context_images", 6):
            return "image_count"
        if sum(_image_bytes(content) for content in contents) > MAX_IMAGE_BYTES:
            return "request_bytes"
        return None

    def prune(self) -> dict | None:
        reason = self._needs_prune()
        if reason is None:
            return None
        keep, seen, views, tables = self.cfg.get("context_images", 6), 0, 0, 0

        def packet_holder(blocks):
            return any(b.get("type") == "image" or (b.get("type") == "text" and '"student_proposal"' in b.get("text", ""))
                       for b in blocks)

        def strip(blocks):
            nonlocal views, tables
            out = []
            for block in blocks:
                if block.get("type") == "image":
                    out.append(dict(PRUNED_VIEW))
                    views += 1
                elif block.get("type") == "text" and '"student_proposal"' in block.get("text", ""):
                    compact = _compact_packet_text(block["text"])
                    tables += compact != block["text"]
                    out.append({**block, "text": compact})
                else:
                    out.append(block)
            return out

        # Packets newest first: a packet is a user turn's own blocks or a tool_result's content.
        for item in reversed(self.items):
            if item["role"] != "user":
                continue
            content = item["content"]
            holders = [(None, content)] + [(block, block.get("content")) for block in content
                                           if block.get("type") == "tool_result"]
            for block, blocks in holders:
                if not isinstance(blocks, list) or not packet_holder(blocks):
                    continue
                seen += 1
                if seen <= keep:
                    continue
                if block is None:
                    item["content"] = content = strip(content)
                else:
                    block["content"] = strip(blocks)
        if not views and not tables:
            return None
        # The edited history invalidates every thinking block produced after the edit point; drop them all once.
        thinking = 0
        for item in self.items:
            if item["role"] == "assistant":
                kept = [b for b in item["content"] if b.get("type") not in ("thinking", "redacted_thinking")]
                thinking += len(item["content"]) - len(kept)
                item["content"] = kept or [{"type": "text", "text": "[reasoning omitted]"}]
        self.prunes += 1
        self.recorded = 0  # the next call artifact records the whole pruned history once
        return {"views_removed": views, "tables_removed": tables, "kept_packets": keep,
                "input_tokens_before": self.last_input_tokens, "trigger": reason, "thinking_blocks_removed": thinking}


def _error_record(exc) -> tuple[dict, bool, bool]:
    """(what may be recorded, fatal, retry) for an SDK exception."""
    import anthropic
    if isinstance(exc, anthropic.APIStatusError):
        body = exc.body if isinstance(exc.body, (dict, list)) else None
        error = provider_error(exc.status_code, json.dumps(body) if body is not None else str(exc.body or ""))
        message = str(((body or {}).get("error") or {}).get("message") or "") if isinstance(body, dict) else ""
        if exc.status_code == 402 or error.get("type") == "billing_error" or "credit balance" in message.lower():
            return {**error, **BILLING}, True, False
        fatal = exc.status_code in FATAL_STATUS or error.get("type") in {"authentication_error", "permission_error"}
        return error, fatal, not fatal and exc.status_code in RETRYABLE
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return {"status": None, "type": type(exc).__name__,
                "message": mask(str(exc))[:400]}, False, True
    return {"status": None, "type": type(exc).__name__,
            "message": mask(str(exc))[:400]}, False, True


class AnthropicClient:
    """Messages API calls with bounded retries (the SDK's own retries are off: ``max_attempts`` and
    ``retry_delays_s`` govern, as for the Responses client). Blocking: the brain runs it in a worker thread; it never
    records anything itself."""

    def __init__(self, cfg: dict, key: str | None = None):
        import anthropic
        self.cfg = cfg
        self.closed = threading.Event()
        self.model_max_tokens: int | None = None
        self._client = anthropic.Anthropic(
            api_key=key if key is not None else read_api_key(cfg["api_key_file"]),
            base_url=cfg.get("base_url") or DEFAULT_BASE_URL, max_retries=0, timeout=cfg["request_timeout_s"])

    def preflight(self) -> dict:
        """The key can read the frozen model (a model lookup costs no tokens); its output limit sizes ``max_tokens``
        when the config sets none."""
        import anthropic
        try:
            model = self._client.models.retrieve(self.cfg["model"], timeout=30)
        except anthropic.APIStatusError as exc:
            error, fatal, _ = _error_record(exc)
            raise (TeacherFatal if fatal else TeacherUnavailable)(f"teacher preflight refused: {error}",
                                                                   [error]) from None
        except anthropic.APIError as exc:
            raise TeacherUnavailable(f"teacher preflight unreachable: {type(exc).__name__}: "
                                     f"{mask(exc)[:400]}") from None
        data = model.model_dump() if hasattr(model, "model_dump") else dict(model)
        if data.get("id") != self.cfg["model"]:
            raise TeacherFatal(f"teacher preflight: the API reports model {data.get('id')!r}")
        self.model_max_tokens = data.get("max_tokens") or None
        return {"model": data.get("id"), "display_name": data.get("display_name"), "transport": API,
                "max_tokens": self.model_max_tokens}

    def _request(self, body: dict) -> dict:
        max_tokens = body.get("max_tokens") or self.model_max_tokens or FALLBACK_MAX_TOKENS
        with self._client.messages.stream(
                model=body["model"], max_tokens=max_tokens, system=body["system"], messages=body["input"],
                tools=body["tools"], tool_choice=body["tool_choice"],
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": body["reasoning"]["effort"]},
                cache_control={"type": "ephemeral"}) as stream:
            message = stream.get_final_message()
        return message.model_dump(mode="json", exclude_none=True)

    def create(self, body: dict) -> tuple[dict, list[dict]]:
        attempts = self.cfg.get("max_attempts", 4)
        delays = self.cfg.get("retry_delays_s") or [5]
        errors = []
        for attempt in range(attempts):
            if self.closed.is_set():
                raise TeacherUnavailable("teacher client closed", errors)
            started = time.monotonic()
            try:
                message = self._request(body)
                stop = message.get("stop_reason")
                if stop == "max_tokens":
                    raise RuntimeError(f"response stopped at max_tokens ({stop})")
                return self._responses_shape(message), errors
            except Exception as exc:  # noqa: BLE001 - classified below; nothing else escapes the worker thread
                error, fatal, retry = _error_record(exc)
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
    def _responses_shape(message: dict) -> dict:
        """The brain reads ``id``, ``output`` and a Responses-style ``usage``. Total input = uncached + cache writes +
        cache reads; ``cached_tokens`` = cache reads. Claude reports no separate thinking-token count."""
        usage = message.get("usage") or {}
        uncached = usage.get("input_tokens") or 0
        written = usage.get("cache_creation_input_tokens") or 0
        read = usage.get("cache_read_input_tokens") or 0
        output = usage.get("output_tokens") or 0
        return {"id": message.get("id"), "status": "completed", "output": message.get("content") or [],
                "stop": {"stop_reason": message.get("stop_reason"), "stop_details": message.get("stop_details")},
                "usage": {"input_tokens": uncached + written + read, "input_tokens_details": {
                    "cached_tokens": read, "cache_creation_tokens": written},
                    "output_tokens": output, "output_tokens_details": {"reasoning_tokens": None},
                    "total_tokens": uncached + written + read + output}}
