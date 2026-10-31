"""Chat requests to an OpenAI-compatible gateway (``/chat/completions`` with tools), or to an
Anthropic Messages endpoint for Claude (``AnthropicGateway``, at the end of this file).

The client never reads proxy settings from the environment.  Failures are sorted three ways:

* rate limited (429): waited out with a growing pause; does not count as a failed attempt;
* out of credit (403 billing error): waited out too, for at most BILLING_WAIT seconds - the account
  is often topped up within minutes, and every episode running then would otherwise end at once.
  An account close to empty refuses a share of the requests while several are in flight, and the
  next try often passes, so the first pauses are short;
* transient (no answer, 408, 409, 425, 5xx): retried a limited number of times;
* anything else (other 4xx): our request is wrong and retrying will not help.

Every attempt is reported through ``log`` so the episode record shows where the time went, and
``Gateway.lost_s`` adds up the time spent on failed attempts and the pauses after them.

Replies are streamed by default (``stream=True``; the gateway allows each user only a few
non-streaming requests a minute).  The pieces are put back together into the answer a
non-streaming request would have returned, so nothing past ``_post`` can tell the two apart.  A
stream that breaks off before its finish reason counts as a transient failure.
"""

from __future__ import annotations

import copy
import json
import random
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import httpx

#: Pauses after a rate limit: start, factor, ceiling (seconds).
RATE_PAUSE = (5.0, 2.0, 120.0)
#: Pauses after a transient failure: start, factor, ceiling (seconds).
RETRY_PAUSE = (2.0, 2.0, 60.0)
#: Pauses while the account is out of credit: start, factor, ceiling (seconds); and the longest such
#: wait for one reply before giving up (seconds).
BILLING_PAUSE = (4.0, 1.5, 60.0)
BILLING_WAIT = 900.0
TRANSIENT = {0, 408, 409, 425}


class GatewayError(Exception):
    """The gateway refused the request, or kept failing until the retries or the time ran out."""

    def __init__(self, message: str, kind: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind  # "refused", "failing", "time" or "stopped"
        self.status = status


@dataclass
class Reply:
    text: str
    calls: list[dict[str, Any]]
    finish_reason: str | None
    usage: dict[str, int]
    reasoning: str | None = None
    attempts: int = 1
    seconds: float = 0.0
    raw_message: dict[str, Any] = field(default_factory=dict)


def read_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt": int(usage.get("prompt_tokens") or 0),
        "completion": int(usage.get("completion_tokens") or 0),
        "cached": int(prompt_details.get("cached_tokens") or 0),
        "reasoning": int(completion_details.get("reasoning_tokens") or 0),
    }


class Gateway:
    #: how a streamed answer is put back together (AnthropicGateway reads other events)
    pieces: Callable[[], Any] = None  # set below, after _Pieces

    def __init__(self, base_url: str, api_key: str, model: str, *, reasoning_effort: str | None = "high",
                 max_tokens: int | None = 8192, request_timeout_s: float = 900.0, retries: int = 10,
                 stream: bool = True, log: Callable[[dict], None] | None = None, ca_file: str | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.request_timeout_s = request_timeout_s
        self.stream = stream
        self.retries = retries
        self.log = log or (lambda event: None)
        self.lost_s = 0.0  # seconds spent on failed attempts and the pauses after them
        # ca_file: a private CA the gateway's certificate is signed with (else the usual ones)
        verify = ssl.create_default_context(cafile=str(Path(ca_file).expanduser())) if ca_file else True
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=None, trust_env=False, verify=verify)

    def close(self) -> None:
        self._http.close()

    def settings(self) -> dict[str, Any]:
        """What every request carries besides the conversation (for the record)."""
        return {"model": self.model, "reasoning_effort": self.reasoning_effort, "max_tokens": self.max_tokens,
                "tool_choice": "auto", "request_timeout_s": self.request_timeout_s, "stream": self.stream}

    def body(self, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.stream:
            body.update(stream=True, stream_options={"include_usage": True})
        return body

    def ask(self, messages: list[dict], tools: list[dict], *, deadline: float,
            stop: Callable[[], bool] = lambda: False) -> Reply:
        """One reply.  ``deadline`` is a ``time.monotonic()`` instant; ``stop`` is polled while waiting.

        Raises :class:`GatewayError` when the request cannot succeed (or time / retries ran out).
        """
        payload = json.dumps(self.body(messages, tools), ensure_ascii=False).encode("utf-8")
        failures = 0
        limited = 0
        broke = 0
        broke_since = None
        attempts = 0
        t_first = time.monotonic()
        while True:
            left = deadline - time.monotonic()
            if stop():
                raise GatewayError("stopped while waiting for the model", "stopped")
            if left <= 1.0:
                raise GatewayError("out of time before the model answered", "time")
            attempts += 1
            t0 = time.monotonic()
            status, data, problem = self._post(payload, min(self.request_timeout_s, left))
            took = time.monotonic() - t0
            event = {"kind": "request", "attempt": attempts, "status": status, "seconds": round(took, 2),
                     "bytes": len(payload)}
            if status == 200 and isinstance(data, dict) and data.get("choices"):
                message = data["choices"][0].get("message") or {}
                usage = read_usage(data.get("usage"))
                if message.get("model"):  # the model that answered, when the gateway says
                    event["model"] = message["model"]
                event.update(usage=usage, finish_reason=data["choices"][0].get("finish_reason"))
                if "first_s" in data:  # streamed: when the first piece came (the model's thinking time)
                    event["first_s"] = data["first_s"]
                self.log(event)
                content = message.get("content")
                return Reply(text=content if isinstance(content, str) else "",
                             calls=list(message.get("tool_calls") or []),
                             finish_reason=data["choices"][0].get("finish_reason"), usage=usage,
                             reasoning=message.get("reasoning_content") or message.get("reasoning"),
                             attempts=attempts, seconds=time.monotonic() - t_first, raw_message=message)
            if status == 200:
                status, problem = 0, f"answer without choices: {str(data)[:300]}"
            event["error"] = (problem or "")[:500]
            self.log(event)
            self.lost_s += took
            if status == 429:
                limited += 1
                pause = _pause(RATE_PAUSE, limited)
            elif status == 403 and self._out_of_credit(problem):
                broke += 1
                broke_since = broke_since or t0
                if time.monotonic() - broke_since > BILLING_WAIT:
                    raise GatewayError(f"the gateway account stayed out of credit for "
                                       f"{time.monotonic() - broke_since:.0f}s: {problem}", "failing", status)
                pause = _pause(BILLING_PAUSE, broke)
            elif status in TRANSIENT or status >= 500:
                failures += 1
                if failures > self.retries:
                    raise GatewayError(f"the gateway failed {failures} times; last: {status} {problem}", "failing",
                                       status)
                pause = _pause(RETRY_PAUSE, failures)
            else:
                raise GatewayError(f"the gateway refused the request: {status} {problem}", "refused", status)
            t_pause = time.monotonic()
            self._sleep(min(pause, max(0.0, deadline - time.monotonic() - 1.0)), stop)
            self.lost_s += time.monotonic() - t_pause

    def _post(self, payload: bytes, timeout_s: float) -> tuple[int, Any, str | None]:
        """One attempt: the HTTP status (0 = no usable answer), the answer as a non-streaming
        request returns it, and what went wrong."""
        if self.stream:
            return self._post_stream(payload, timeout_s)
        try:
            response = self._http.post(self.url, content=payload, timeout=timeout_s)
        except httpx.HTTPError as exc:
            return 0, None, f"{type(exc).__name__}: {exc}"
        return _whole(response)

    def _post_stream(self, payload: bytes, timeout_s: float) -> tuple[int, Any, str | None]:
        t0 = time.monotonic()
        parts = self.pieces()
        done = False
        try:
            with self._http.stream("POST", self.url, content=payload, timeout=timeout_s) as response:
                if response.status_code != 200 or "event-stream" not in response.headers.get("content-type", ""):
                    response.read()
                    return _whole(response)
                # The body is read to its end, past [DONE] (as the OpenAI SDK does): the gateway logs a
                # stream the client closes before the server ends it as failed, complete answer or not.
                for data in _sse_data(_lines(response.iter_bytes())):
                    if done:
                        if time.monotonic() - t0 > timeout_s:
                            break
                        continue
                    if data == "[DONE]":
                        done = True
                        continue
                    if parts.first_s is None:
                        parts.first_s = time.monotonic() - t0
                    failed = parts.add(data)
                    if failed:
                        return failed
                    if time.monotonic() - t0 > timeout_s:
                        return 0, None, f"the answer was still coming after {timeout_s:.0f}s"
        except httpx.HTTPError as exc:
            if not done:  # a broken tail after [DONE] leaves the answer whole
                return 0, None, f"{type(exc).__name__}: {exc}"
        return parts.answer()

    @staticmethod
    def _out_of_credit(problem: str | None) -> bool:
        return _out_of_credit(problem)

    @staticmethod
    def _sleep(seconds: float, stop: Callable[[], bool]) -> None:
        end = time.monotonic() + seconds
        while not stop():
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(1.0, left))


def _out_of_credit(problem: str | None) -> bool:
    text = (problem or "").lower()
    return any(word in text for word in ("billing", "balance", "credit", "quota"))


def _claude_out_of_credit(problem: str | None) -> bool:
    """The Claude gateways (new-api relays) also report an empty account in Chinese: "预扣费额度失败"
    (``_whole`` passes the body on as ASCII JSON, so the word may come escaped)."""
    text = problem or ""
    return _out_of_credit(text) or "额度" in text or "\\u989d\\u5ea6" in text


def _pause(rule: tuple[float, float, float], n: int) -> float:
    start, factor, ceiling = rule
    base = min(ceiling, start * factor ** (n - 1))
    return base * random.uniform(0.75, 1.25)


def _whole(response: httpx.Response) -> tuple[int, Any, str | None]:
    """A complete (not streamed) answer or refusal."""
    try:
        data = response.json()
    except ValueError:
        data = None
    if response.status_code != 200:
        text = json.dumps(data)[:500] if data is not None else response.text[:500]
        return response.status_code, data, text
    if data is None:
        return 0, None, f"unreadable answer: {response.text[:300]}"
    return 200, data, None


def _lines(chunks: Iterable[bytes]) -> Iterator[str]:
    """The lines of a byte stream, split at newlines only (the text inside may hold other line breaks)."""
    rest = b""
    for chunk in chunks:
        rest += chunk
        *lines, rest = rest.split(b"\n")
        for line in lines:
            yield line.removesuffix(b"\r").decode("utf-8", "replace")
    if rest:
        yield rest.removesuffix(b"\r").decode("utf-8", "replace")


def _sse_data(lines: Iterable[str]) -> Iterator[str]:
    """The data of each server-sent event; comments and other fields are skipped."""
    data: list[str] = []
    for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


class _Pieces:
    """A streamed answer, put back together into the shape of a non-streaming one."""

    def __init__(self):
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish: str | None = None
        self.usage: dict[str, Any] | None = None
        self.first_s: float | None = None

    def add(self, data: str) -> tuple[int, Any, str] | None:
        """Take one event; returns the failure when the event reports one or cannot be read."""
        try:
            chunk = json.loads(data)
        except ValueError:
            chunk = None
        if not isinstance(chunk, dict):
            return 0, None, f"unreadable piece of the answer: {data[:300]}"
        if chunk.get("error"):
            problem = json.dumps(chunk)[:500]
            return (403 if _out_of_credit(problem) else 0), chunk, problem
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if (choice.get("index") or 0) != 0:
                continue
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                self.text.append(delta["content"])
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(thought, str):
                self.reasoning.append(thought)
            for piece in delta.get("tool_calls") or []:
                self._call(piece)
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
        return None

    def _call(self, piece: dict[str, Any]) -> None:
        index = piece.get("index")
        if not isinstance(index, int):  # no index: an id not seen before starts the next call
            seen = [i for i, c in self.calls.items() if c["id"] and c["id"] == piece.get("id")]
            index = seen[0] if seen else (len(self.calls) if piece.get("id") or not self.calls else max(self.calls))
        call = self.calls.setdefault(index, {"id": None, "type": "function",
                                             "function": {"name": "", "arguments": ""}})
        if piece.get("id"):
            call["id"] = piece["id"]
        if piece.get("type"):
            call["type"] = piece["type"]
        function = piece.get("function") or {}
        if function.get("name") and not call["function"]["name"]:
            call["function"]["name"] = function["name"]
        if isinstance(function.get("arguments"), str):
            call["function"]["arguments"] += function["arguments"]

    def answer(self) -> tuple[int, Any, str | None]:
        if self.finish is None:
            return 0, None, "the answer broke off before it was complete"
        message: dict[str, Any] = {"role": "assistant", "content": "".join(self.text) or None}
        if self.calls:
            message["tool_calls"] = [self.calls[i] for i in sorted(self.calls)]
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        return 200, {"choices": [{"index": 0, "message": message, "finish_reason": self.finish}],
                     "usage": self.usage, "first_s": round(self.first_s or 0.0, 2)}, None


Gateway.pieces = _Pieces


# -- Claude: the Anthropic Messages API ----------------------------------------------------------------
ANTHROPIC_VERSION = "2023-06-01"
#: why a reply stopped, in the chat-completions words the episode reads
STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length",
                "pause_turn": "pause_turn", "refusal": "refusal"}


class AnthropicGateway(Gateway):
    """The same client for Claude, against an Anthropic Messages endpoint (``<base>/messages``).

    The conversation stays in the chat-completions shape the rest of HexaShell uses; ``body`` turns it
    into a Messages request each time.  Thinking is adaptive, its summary returned (``display``), its
    depth set by ``reasoning_effort`` (``output_config.effort``).  A reply's content blocks - thinking
    blocks with their signatures, text, tool calls - come back in ``raw_message["_blocks"]``; the
    conversation keeps them and every later request sends them back unchanged, so the model keeps its
    earlier reasoning.  Streamed events are put back into the chat-completions shape, so ``ask`` and
    the episode handle both gateways alike.  Three cache breakpoints: the end of the system prompt, the
    end of the previous request and the end of the conversation.
    """

    def __init__(self, base_url: str, api_key: str, model: str, *, display: str = "summarized", **kwargs):
        super().__init__(base_url, api_key, model, **kwargs)
        if not self.stream:
            raise ValueError("AnthropicGateway streams its replies (stream=True)")
        self.url = base_url.rstrip("/") + "/messages"
        self.display = display
        self._http.headers.update({"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION})

    pieces = None  # set below, after _MessagePieces

    def settings(self) -> dict[str, Any]:
        out = super().settings()
        out.update(api="anthropic-messages", thinking={"type": "adaptive", "display": self.display})
        return out

    @staticmethod
    def _out_of_credit(problem: str | None) -> bool:
        return _claude_out_of_credit(problem)

    def body(self, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        system, turns = to_anthropic(messages)
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens or 8192, "system": system,
                                "messages": turns, "thinking": {"type": "adaptive", "display": self.display},
                                "stream": True}
        if tools:
            body["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                              "input_schema": t["function"]["parameters"]} for t in tools]
            body["tool_choice"] = {"type": "auto"}
        if self.reasoning_effort:
            body["output_config"] = {"effort": self.reasoning_effort}
        return body


def to_anthropic(messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """(system blocks, messages) of a Messages request for a chat-completions conversation.

    Tool results and the pictures that follow them merge into one user turn; an assistant turn is sent
    as the content blocks the API returned (``_blocks``), or rebuilt from its text and tool calls.
    """
    system: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})

    for message in messages:
        role = message["role"]
        if role == "system":
            system.append({"type": "text", "text": message["content"]})
        elif role == "assistant":
            blocks = message.get("_blocks")
            if blocks is None:
                blocks = [{"type": "text", "text": message["content"]}] if message.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                            "input": _json_object(c["function"].get("arguments"))}
                           for c in message.get("tool_calls") or []]
            add("assistant", copy.deepcopy(blocks) or [{"type": "text", "text": "(no reply)"}])
        elif role == "tool":
            add("user", [{"type": "tool_result", "tool_use_id": message["tool_call_id"],
                          "content": message.get("content") or "(no output)"}])
        else:
            add("user", _user_blocks(message.get("content")))
    if system:
        system[-1]["cache_control"] = {"type": "ephemeral"}
    # The end of the conversation, and the end of the previous request (the turn before the latest
    # reply): a breakpoint looks back only 20 blocks for an earlier cache entry, and a turn with many
    # pictures adds more than that.
    replies = [i for i, turn in enumerate(turns) if turn["role"] == "assistant"]
    for i in {len(turns) - 1, replies[-1] - 1 if replies else -1}:
        if i >= 0 and turns[i]["content"]:
            turns[i]["content"][-1] = dict(turns[i]["content"][-1], cache_control={"type": "ephemeral"})
    return system, turns


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content or "(empty)")}]
    blocks: list[dict[str, Any]] = []
    for part in content:
        if part.get("type") == "image_url":
            head, _, data = part["image_url"]["url"].partition(",")
            media = head.removeprefix("data:").split(";")[0] or "image/jpeg"
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        elif part.get("text"):
            blocks.append({"type": "text", "text": part["text"]})
    return blocks or [{"type": "text", "text": "(empty)"}]


def _json_object(text: Any) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    try:
        value = json.loads(text or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


class _MessagePieces:
    """A streamed Messages answer, put back together into the shape of a chat-completions one."""

    def __init__(self):
        self.blocks: dict[int, dict[str, Any]] = {}
        self.json: dict[int, str] = {}  # tool-call arguments as they stream in
        self.stop: str | None = None
        self.usage: dict[str, Any] = {}
        self.model: str | None = None
        self.first_s: float | None = None

    def add(self, data: str) -> tuple[int, Any, str] | None:
        try:
            event = json.loads(data)
        except ValueError:
            event = None
        if not isinstance(event, dict):
            return 0, None, f"unreadable piece of the answer: {data[:300]}"
        kind = event.get("type")
        if kind == "error" or event.get("error"):
            problem = json.dumps(event, ensure_ascii=False)[:500]
            error = (event.get("error") or {}).get("type") if isinstance(event.get("error"), dict) else None
            status = 429 if error == "rate_limit_error" else 403 if _claude_out_of_credit(problem) else 0
            return status, event, problem
        if kind == "message_start":
            message = event.get("message") or {}
            self.model = message.get("model")
            self.usage.update(message.get("usage") or {})
        elif kind == "content_block_start":
            index = event.get("index", len(self.blocks))
            self.blocks[index] = dict(event.get("content_block") or {})
            if self.blocks[index].get("type") == "tool_use":
                self.json[index] = ""
        elif kind == "content_block_delta":
            index = event.get("index", 0)
            block = self.blocks.setdefault(index, {"type": "text", "text": ""})
            delta = event.get("delta") or {}
            step = delta.get("type")
            if step == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif step == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            elif step == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.get("signature", "")
            elif step == "input_json_delta":
                self.json[index] = self.json.get(index, "") + delta.get("partial_json", "")
        elif kind == "message_delta":
            self.stop = (event.get("delta") or {}).get("stop_reason") or self.stop
            self.usage.update(event.get("usage") or {})
        return None

    def answer(self) -> tuple[int, Any, str | None]:
        if self.stop is None:
            return 0, None, "the answer broke off before it was complete"
        blocks, text, thoughts, calls = [], [], [], []
        for index in sorted(self.blocks):
            block = self.blocks[index]
            kind = block.get("type")
            if kind == "tool_use":
                arguments = self.json.get(index) or "{}"
                block["input"] = _json_object(arguments)
                calls.append({"id": block.get("id"), "type": "function",
                              "function": {"name": block.get("name", ""), "arguments": arguments}})
            elif kind == "text":
                if not block.get("text"):
                    continue  # the API refuses empty text blocks when they are sent back
                text.append(block["text"])
            elif kind == "thinking":
                if block.get("thinking"):
                    thoughts.append(block["thinking"])
                if not block.get("signature"):
                    continue  # cut off before it was signed (max_tokens): the API would refuse it back
            blocks.append(block)
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None, "_blocks": blocks,
                                   "model": self.model}
        if calls:
            message["tool_calls"] = calls
        if thoughts:
            message["reasoning_content"] = "\n\n".join(thoughts)
        u = self.usage
        read, written = int(u.get("cache_read_input_tokens") or 0), int(u.get("cache_creation_input_tokens") or 0)
        details = u.get("output_tokens_details") or {}
        usage = {"prompt_tokens": int(u.get("input_tokens") or 0) + read + written,
                 "completion_tokens": int(u.get("output_tokens") or 0),
                 "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": written},
                 "completion_tokens_details": {"reasoning_tokens": int(details.get("thinking_tokens") or 0)}}
        return 200, {"choices": [{"index": 0, "message": message,
                                  "finish_reason": STOP_REASONS.get(self.stop, self.stop)}],
                     "usage": usage, "first_s": round(self.first_s or 0.0, 2)}, None


AnthropicGateway.pieces = _MessagePieces
