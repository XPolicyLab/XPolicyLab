# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert RoboRSI's full runtime trace into the public training episode format."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from roborsi.evaluation.trajectory import TrajectoryReader
from roborsi.training_data import convert_episode

_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


def _content(value: Any) -> list[dict[str, Any]]:
    """Map SDK text and content-addressed images without discarding tool data."""
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if isinstance(value, list):
        return [part for item in value for part in _content(item)]
    if isinstance(value, Mapping):
        if value.get("kind") == "cache-point":
            return [{"type": "cache_point", "ttl": value.get("ttl", "5m")}]
        if value.get("kind") == "image-url":
            raise ValueError(
                "remote image URL cannot be exported; provide image bytes to the agent"
            )
        if "artifact_ref" in value and not str(value.get("media_type", "")).startswith(
            "image/"
        ):
            raise ValueError(
                "training export does not support non-image binary content"
            )
        if "artifact_ref" in value and str(value.get("media_type", "")).startswith(
            "image/"
        ):
            return [
                {
                    "type": "image",
                    "media_type": value["media_type"],
                    "asset_ref": value["artifact_ref"],
                    "sha256": value.get("sha256"),
                }
            ]
        if "return_value" in value:
            return [
                *_content(value["return_value"]),
                *_content(value.get("content")),
            ]
        # Keep arbitrary structured observations as exact JSON text, then
        # materialize any nested image references as separate visual blocks.
        images = [
            part
            for nested in value.values()
            for part in _content(nested)
            if part["type"] == "image"
        ]
        return [
            {
                "type": "text",
                "text": json.dumps(value, ensure_ascii=False, sort_keys=True),
            },
            *images,
        ]
    return [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]


def _tool_call(part: Mapping[str, Any], index: int) -> dict[str, Any]:
    arguments = part.get("args", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_raw": arguments}
    if not isinstance(arguments, Mapping):
        arguments = {"_raw": arguments}
    return {
        "id": part.get("tool_call_id") or f"missing-tool-id-{index}",
        "name": part["tool_name"],
        "arguments": dict(arguments),
    }


def _assistant(parts: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": [], "tool_calls": []}
    for index, part in enumerate(parts, 1):
        kind = part.get("part_kind")
        if kind == "tool-call":
            message["tool_calls"].append(_tool_call(part, index))
        elif kind == "thinking":
            block = {"type": "reasoning", "text": str(part.get("content", ""))}
            if part.get("signature") is not None:
                signature = part["signature"]
                if not isinstance(signature, str):
                    raise TypeError("thinking signature must be text")
                block["encrypted_content"] = signature
            message["content"].append(block)
        elif kind == "text":
            message["content"].append(
                {"type": "text", "text": str(part.get("content", ""))}
            )
        else:
            message["content"].append(
                {
                    "type": "text",
                    "text": json.dumps(part, ensure_ascii=False, sort_keys=True),
                }
            )
    return message


def _messages(request: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    system_prompt = ""
    messages: list[dict[str, Any]] = []
    for sdk_message in request.get("messages", []):
        if sdk_message.get("kind") == "response":
            messages.append(_assistant(sdk_message.get("parts", [])))
            continue
        instructions = sdk_message.get("instructions")
        if isinstance(instructions, str) and instructions:
            system_prompt = instructions
        for part in sdk_message.get("parts", []):
            kind = part.get("part_kind")
            if kind == "user-prompt":
                messages.append(
                    {"role": "user", "content": _content(part.get("content"))}
                )
            elif kind == "tool-return":
                messages.append(
                    {
                        "role": "tool",
                        "name": part.get("tool_name"),
                        "tool_call_id": part.get("tool_call_id") or "missing-tool-id",
                        "content": _content(part.get("content")),
                    }
                )
            elif kind == "retry-prompt":
                role = "tool" if part.get("tool_call_id") else "user"
                message = {"role": role, "content": _content(part.get("content"))}
                if role == "tool":
                    message.update(
                        tool_call_id=part["tool_call_id"],
                        name=part.get("tool_name"),
                    )
                messages.append(message)
            elif kind in {"system-prompt", "instruction"}:
                system_prompt = "\n\n".join(
                    filter(None, (system_prompt, str(part.get("content", ""))))
                )
            else:
                messages.append({"role": "user", "content": _content(part)})
    return system_prompt, messages


def _request(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    system_prompt, messages = _messages(snapshot)
    parameters = snapshot.get("model_request_parameters") or {}
    if not system_prompt:
        system_prompt = "\n\n".join(
            str(part["content"])
            for part in (parameters.get("instruction_parts") or [])
            if isinstance(part, Mapping) and part.get("content")
        )
    tools = []
    for tool in [
        *parameters.get("function_tools", []),
        *parameters.get("output_tools", []),
        *parameters.get("native_tools", []),
    ]:
        if not isinstance(tool, Mapping) or not tool.get("name"):
            continue
        tools.append(
            {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "input_schema": tool.get("parameters_json_schema") or {},
            }
        )
    return {
        "provider": snapshot.get("provider"),
        "model": snapshot.get("model"),
        "system_prompt": system_prompt,
        "messages": messages,
        "tools": tools,
        "settings": {
            "model_settings": snapshot.get("model_settings"),
            "model_request_parameters": parameters,
        },
    }


def _usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {key: value[key] for key in _USAGE_FIELDS if key in value}


def _response(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "message": _assistant(snapshot.get("parts", [])),
        "finish_reason": snapshot.get("finish_reason"),
    }


def _read_snapshot(reader: TrajectoryReader, reference: str | None) -> dict[str, Any]:
    if not reference:
        raise ValueError("full trace is missing a required snapshot reference")
    path = reader.artifact_path(reference)
    if not path.is_file():
        raise FileNotFoundError(f"trace snapshot is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _tool_result(
    snapshot: Mapping[str, Any], event: Mapping[str, Any]
) -> dict[str, Any]:
    payload = event.get("payload") or {}
    content = _content(snapshot)
    if payload.get("status") != "completed":
        content.append(
            {
                "type": "text",
                "text": json.dumps({"error_type": payload.get("error_type")}),
            }
        )
    return {
        "role": "tool",
        "name": payload.get("name"),
        "tool_call_id": event.get("tool_call_id"),
        "content": content,
    }


def convert_roborsi_trace(
    run_dir: str | Path,
    task: Mapping[str, Any],
    *,
    output_dir: str | Path,
    outcome: Mapping[str, Any] | None = None,
    allow_partial: bool = False,
    versions: Mapping[str, str] | None = None,
) -> Path:
    """Export one full RoboRSI trace, preserving every logical model call.

    Retries appear in each call's attempts and do not create duplicate training
    targets. Root, child and context-compression calls remain distinguishable by
    agent identity and purpose. Benchmark-native scores are supplied explicitly.

    Args:
        run_dir: RoboRSI run containing trace/manifest.json and full snapshots.
        task: Benchmark task identity and instruction.
        output_dir: New standard episode directory.
        outcome: Benchmark-native result, if scoring has completed.
        allow_partial: Include an unfinished run with missing responses.
        versions: Versions recorded at run time, such as agent commit, simulator,
            scorer, prompt and skill revisions. Unknown values should be omitted.

    Returns:
        Path to the published standard episode.json file.
    """
    reader = TrajectoryReader(run_dir)
    manifest = reader.manifest()
    if manifest.get("mode") != "full":
        raise ValueError("training export requires a full RoboRSI trace")
    if not allow_partial and not manifest.get("complete"):
        raise ValueError("training export requires a complete RoboRSI trace")
    projection = reader.update()
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    attempts: dict[str, list[dict[str, Any]]] = {}
    tools: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for event in projection.events:
        request_id = event.get("request_id")
        kind = event["type"]
        if kind == "model_request_start" and request_id:
            starts[request_id] = event
        elif kind == "model_request_end" and request_id:
            ends[request_id] = event
        elif kind == "model_attempt_end" and request_id:
            attempts.setdefault(request_id, []).append(event)
        elif kind == "tool_end":
            tools[(event.get("turn_id"), event.get("tool_call_id"))] = event
    calls = []
    for request_id, start in starts.items():
        end = ends.get(request_id)
        if end is None and not allow_partial:
            raise ValueError(f"model request has no end event: {request_id}")
        request = _request(
            _read_snapshot(reader, (start.get("payload") or {}).get("request_ref"))
        )
        payload = (end or {}).get("payload") or {}
        response_ref = payload.get("response_ref")
        response = (
            _response(_read_snapshot(reader, response_ref)) if response_ref else None
        )
        request_attempts = sorted(
            attempts.get(request_id, []), key=lambda item: item.get("attempt") or 0
        )
        attempt_records = [
            {
                "number": attempt.get("attempt") or 1,
                "status": (attempt.get("payload") or {}).get("status", "error"),
                "error_type": (attempt.get("payload") or {}).get("error_type"),
                "usage": _usage((attempt.get("payload") or {}).get("usage")),
            }
            for attempt in request_attempts
        ]
        tool_results = []
        if response is not None:
            for tool_call in response["message"]["tool_calls"]:
                tool_event = tools.get((start.get("turn_id"), tool_call["id"]))
                if tool_event is None:
                    continue
                tool_payload = tool_event.get("payload") or {}
                snapshot = (
                    _read_snapshot(reader, tool_payload["result_ref"])
                    if tool_payload.get("result_ref")
                    else {}
                )
                tool_results.append(_tool_result(snapshot, tool_event))
        calls.append(
            {
                "call_id": request_id,
                "agent_id": start.get("agent_id"),
                "parent_agent_id": start.get("parent_agent_id"),
                "turn_id": start.get("turn_id"),
                "purpose": (start.get("payload") or {}).get("purpose", "agent"),
                "request": request,
                "response": response,
                "tool_results": tool_results,
                "status": payload.get("status", "partial"),
                "usage": _usage(
                    (request_attempts[-1].get("payload") or {}).get("usage")
                    if request_attempts
                    else None
                ),
                "attempts": attempt_records,
            }
        )
    if not calls:
        raise ValueError("RoboRSI trace has no model requests")
    return convert_episode(
        task,
        calls,
        output_dir=output_dir,
        outcome=outcome,
        episode_id=manifest.get("run_id"),
        source_dir=run_dir,
        provenance={
            "source_kind": "roborsi_trace",
            "source_run_id": manifest.get("run_id"),
            "source_manifest_sha256": hashlib.sha256(
                (Path(run_dir) / "trace/manifest.json").read_bytes()
            ).hexdigest(),
            "captured_at": manifest.get("started_at"),
            "versions": dict(versions or {}),
        },
    )
