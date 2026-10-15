"""Translate between chat/completions payloads and the OpenAI Responses API.

Agent policies build their dialogue as chat/completions ``messages`` and read
replies out of ``choices[0].message``. The Responses API is the only way to
control reasoning effort when function tools are registered: chat/completions
rejects any ``reasoning_effort`` other than ``"none"`` alongside tools. These
helpers keep the callers' message plumbing untouched by translating only at the
wire boundary.

Two provider constraints are baked in, both observed against ByteDance AIDP:

- A replayed ``function_call`` must not carry the server-assigned item ``id``.
  AIDP load-balances across Azure resources, and an id minted by one resource is
  rejected by another with "The requested item was created under a different
  Azure OpenAI resource".
- A ``reasoning`` item may only be replayed when the request pins the same
  resource through the stateful-session headers. Without them the encrypted
  payload fails to decrypt. `ReasoningReplayStore` therefore holds reasoning
  items outside the caller's message list and only replays them on request.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "ReasoningReplayStore",
    "chat_messages_to_responses_input",
    "chat_tools_to_responses_tools",
    "responses_to_chat_completion",
    "session_headers",
]


def session_headers(session_id: str) -> dict[str, str]:
    """AIDP headers that pin one dialogue to a single Azure resource.

    Required for prompt caching and for replaying reasoning items; without them
    every turn may land on a different resource.
    """
    return {
        "extra": json.dumps({"session_id": session_id}),
        "azureai-stateful-session-enabled": "true",
        "azureai-model-sessionid": session_id,
    }


def chat_tools_to_responses_tools(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Flatten ``{"type","function":{...}}`` into the Responses tool shape."""
    flattened: list[dict[str, Any]] = []
    for tool in tools or []:
        if tool.get("type") != "function":
            flattened.append(dict(tool))
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            flattened.append(dict(tool))
            continue
        entry: dict[str, Any] = {
            "type": "function",
            "name": function.get("name"),
            "parameters": function.get("parameters") or {"type": "object"},
        }
        description = function.get("description")
        if description is not None:
            entry["description"] = description
        if function.get("strict") is not None:
            entry["strict"] = function["strict"]
        flattened.append(entry)
    return flattened


def _content_parts(content: Any, *, role: str) -> Any:
    """Map chat content parts to Responses input/output parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    text_type = "output_text" if role == "assistant" else "input_text"
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append({"type": text_type, "text": str(part)})
            continue
        kind = part.get("type")
        if kind == "text":
            parts.append({"type": text_type, "text": part.get("text", "")})
        elif kind == "image_url":
            image_url = part.get("image_url")
            url = (
                image_url.get("url")
                if isinstance(image_url, dict)
                else image_url
            )
            image_part: dict[str, Any] = {
                "type": "input_image",
                "image_url": str(url),
            }
            detail = (
                image_url.get("detail") if isinstance(image_url, dict) else None
            )
            if detail is not None:
                image_part["detail"] = detail
            parts.append(image_part)
        else:
            parts.append(part)
    return parts


def _tool_call_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False, default=str)
        items.append(
            {
                "type": "function_call",
                # The server-assigned item id is deliberately dropped; see module docstring.
                "call_id": call.get("id"),
                "name": function.get("name") or call.get("name"),
                "arguments": arguments,
            }
        )
    return items


def chat_messages_to_responses_input(
    messages: list[dict[str, Any]],
    *,
    reasoning_store: "ReasoningReplayStore | None" = None,
) -> list[dict[str, Any]]:
    """Convert a chat ``messages`` list into Responses ``input`` items.

    When ``reasoning_store`` is given, the reasoning items recorded for an
    assistant turn are replayed immediately before that turn's tool call.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            output = message.get("content")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False, default=str)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": output,
                }
            )
            continue
        if role == "assistant":
            calls = _tool_call_items(message)
            if reasoning_store is not None and calls:
                items.extend(reasoning_store.items_for(calls[0].get("call_id")))
            content = message.get("content")
            if content:
                items.append(
                    {
                        "role": "assistant",
                        "content": _content_parts(content, role="assistant"),
                    }
                )
            items.extend(calls)
            continue
        items.append(
            {
                "role": role or "user",
                "content": _content_parts(message.get("content"), role=str(role)),
            }
        )
    return items


def _output_items(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("output") or [])
    return list(getattr(response, "output", []) or [])


def _item_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return dict(item)


def responses_to_chat_completion(
    response: Any,
    *,
    reasoning_store: "ReasoningReplayStore | None" = None,
) -> dict[str, Any]:
    """Render a Responses reply in the ``choices[0].message`` shape.

    Callers keep using ``parse_chat_message`` / ``assistant_message_from_result``
    unchanged. Reasoning items are handed to ``reasoning_store`` rather than
    returned, since they cannot live in a chat-shaped message.
    """
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_items: list[dict[str, Any]] = []
    for item in _output_items(response):
        payload = _item_dict(item)
        kind = payload.get("type")
        if kind == "reasoning":
            reasoning_items.append(payload)
        elif kind == "function_call":
            tool_calls.append(
                {
                    "id": payload.get("call_id"),
                    "type": "function",
                    "function": {
                        "name": payload.get("name"),
                        "arguments": payload.get("arguments") or "{}",
                    },
                }
            )
        elif kind == "message":
            for part in payload.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    text_chunks.append(str(part["text"]))
    if reasoning_store is not None and reasoning_items:
        reasoning_store.record(
            tool_calls[0]["id"] if tool_calls else None, reasoning_items
        )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_chunks),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = response.get("usage") if isinstance(response, dict) else None
    if usage is None:
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            dump = getattr(raw_usage, "model_dump", None)
            usage = dump() if callable(dump) else dict(raw_usage)
    finish_reason = "tool_calls" if tool_calls else "stop"
    status = (
        response.get("status")
        if isinstance(response, dict)
        else getattr(response, "status", None)
    )
    if status == "incomplete":
        finish_reason = "length"
    return {
        "id": (
            response.get("id")
            if isinstance(response, dict)
            else getattr(response, "id", None)
        ),
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage or {},
    }


class ReasoningReplayStore:
    """Hold reasoning items outside the caller's chat message list.

    Reasoning items carry provider-encrypted state that cannot be represented in
    a chat-shaped assistant message, so they are keyed by the tool call they
    accompanied and replayed from here.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._items: dict[str, list[dict[str, Any]]] = {}

    def record(self, call_id: str | None, items: list[dict[str, Any]]) -> None:
        if not self.enabled or not call_id or not items:
            return
        replayable: list[dict[str, Any]] = []
        for item in items:
            payload = dict(item)
            # The SDK includes this output-only field in its input type, but
            # AIDP rejects it as -4003 at `input[N].status`.
            payload.pop("status", None)
            replayable.append(payload)
        self._items[str(call_id)] = replayable

    def items_for(self, call_id: str | None) -> list[dict[str, Any]]:
        if not self.enabled or not call_id:
            return []
        return [dict(item) for item in self._items.get(str(call_id), [])]

    def clear(self) -> None:
        self._items.clear()
