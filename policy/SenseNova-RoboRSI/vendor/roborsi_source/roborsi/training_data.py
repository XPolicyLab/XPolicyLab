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

"""Benchmark-neutral conversion of task and LLM calls into training episodes."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

_IMAGE_SUFFIXES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


def training_episode_schema() -> dict[str, Any]:
    """Return the published version 1 JSON Schema.

    Returns:
        A fresh copy of the schema so callers cannot mutate validation rules.
    """
    source = files("roborsi.schemas").joinpath("training_episode.v1.json")
    return json.loads(source.read_text(encoding="utf-8"))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _items(value: Any, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, (list, tuple)
    ):
        raise TypeError(f"{name} must be a sequence")
    return list(value)


def _copy_provider_json(value: Any) -> Any:
    """Copy native provider context without changing its finite JSON values."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("provider item JSON object keys must be text")
        return {key: _copy_provider_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_provider_json(item) for item in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("provider item must contain only finite JSON values")


class _Assets:
    """Copy caller-owned image data into one immutable episode directory."""

    def __init__(self, root: Path, source_dir: Path | None) -> None:
        self.root = root
        self.source_dir = source_dir.resolve() if source_dir is not None else None

    def image(self, value: Mapping[str, Any]) -> dict[str, str]:
        media_type = value.get("media_type")
        if media_type not in _IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image media type: {media_type!r}")
        sources = [
            key for key in ("data", "base64", "path", "asset_ref") if key in value
        ]
        if len(sources) != 1:
            raise ValueError(
                "image content needs exactly one data, base64, path or asset_ref"
            )
        source = sources[0]
        if source == "data":
            data = value[source]
            if not isinstance(data, bytes):
                raise TypeError("image data must be bytes")
        elif source == "base64":
            encoded = value[source]
            if not isinstance(encoded, str):
                raise TypeError("image base64 must be text")
            try:
                data = base64.b64decode(encoded, validate=True)
            except binascii.Error as exc:
                raise ValueError("invalid image base64") from exc
        else:
            reference = Path(value[source])
            if source == "asset_ref":
                if self.source_dir is None:
                    raise ValueError("source_dir is required for image asset_ref")
                path = (self.source_dir / reference).resolve()
                if reference.is_absolute() or not path.is_relative_to(self.source_dir):
                    raise ValueError("image asset_ref escapes source_dir")
            else:
                path = reference.resolve()
            data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        supplied_digest = value.get("sha256")
        if supplied_digest is not None and supplied_digest != digest:
            raise ValueError("image sha256 does not match its content")
        destination = self.root / "assets" / f"{digest}.{_IMAGE_SUFFIXES[media_type]}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
        return {
            "type": "image",
            "media_type": media_type,
            "asset_ref": destination.relative_to(self.root).as_posix(),
            "sha256": digest,
        }


def _content(value: Any, assets: _Assets) -> list[dict[str, Any]]:
    if value is None:
        return []
    parts = [value] if isinstance(value, (str, Mapping)) else _items(value, "content")
    result = []
    for part in parts:
        if isinstance(part, str):
            result.append({"type": "text", "text": part})
            continue
        item = _mapping(part, "content part")
        if item.get("type") in {"text", "reasoning"}:
            content = {"type": item["type"], "text": item["text"]}
            if item["type"] == "reasoning" and "encrypted_content" in item:
                encrypted_content = item["encrypted_content"]
                if not isinstance(encrypted_content, str):
                    raise TypeError("reasoning encrypted_content must be text")
                content["encrypted_content"] = encrypted_content
            result.append(content)
        elif item.get("type") == "provider_item":
            result.append(
                {
                    "type": "provider_item",
                    "provider": item["provider"],
                    "item": _copy_provider_json(
                        _mapping(item["item"], "provider item JSON object")
                    ),
                }
            )
        elif item.get("type") == "cache_point":
            result.append({"type": "cache_point", "ttl": item["ttl"]})
        elif item.get("type") == "image":
            result.append(assets.image(item))
        else:
            raise ValueError(f"unsupported content type: {item.get('type')!r}")
    return result


def _message(value: Any, assets: _Assets) -> dict[str, Any]:
    item = _mapping(value, "message")
    calls = []
    for raw in _items(item.get("tool_calls", []), "message tool_calls"):
        tool = _mapping(raw, "tool call")
        calls.append(
            {"id": tool["id"], "name": tool["name"], "arguments": tool["arguments"]}
        )
    message = {
        "role": item["role"],
        "content": _content(item.get("content", []), assets),
        "tool_calls": calls,
    }
    for key in ("name", "tool_call_id"):
        if key in item and item[key] is not None:
            message[key] = item[key]
    return message


def _request(value: Any, assets: _Assets) -> dict[str, Any]:
    item = _mapping(value, "LLM request")
    tools = []
    for raw in _items(item.get("tools", []), "request tools"):
        tool = _mapping(raw, "tool definition")
        tools.append(
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool["input_schema"],
            }
        )
    request = {
        "system_prompt": item.get("system_prompt", ""),
        "messages": [
            _message(message, assets)
            for message in _items(item["messages"], "request messages")
        ],
        "tools": tools,
        "settings": dict(_mapping(item.get("settings", {}), "request settings")),
    }
    for key in ("model", "provider"):
        if key in item:
            request[key] = item[key]
    return request


def _response(value: Any, assets: _Assets) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = {"message": {"role": "assistant", "content": value}}
    item = _mapping(value, "LLM response")
    raw_message = item.get("message", item)
    message = _message(raw_message, assets)
    if message["role"] != "assistant":
        raise ValueError("LLM response message must have assistant role")
    response = {"message": message}
    if "finish_reason" in item:
        response["finish_reason"] = item["finish_reason"]
    return response


def _call(value: Any, index: int, assets: _Assets) -> dict[str, Any]:
    item = _mapping(value, f"LLM call {index}")
    status = item.get("status", "completed")
    response = _response(item.get("response"), assets)
    if status == "completed" and response is None:
        raise ValueError(f"completed LLM call {index} needs a response")
    result = {
        "call_id": item.get("call_id", f"call-{index:06d}"),
        "purpose": item.get("purpose", "agent"),
        "request": _request(item["request"], assets),
        "response": response,
        "tool_results": [
            _message(message, assets)
            for message in _items(item.get("tool_results", []), "tool_results")
        ],
        "status": status,
        "usage": dict(_mapping(item.get("usage", {}), "call usage")),
        "attempts": [
            dict(_mapping(attempt, "LLM attempt"))
            for attempt in _items(item.get("attempts", []), "attempts")
        ],
    }
    for key in ("agent_id", "parent_agent_id", "turn_id"):
        if key in item:
            result[key] = item[key]
    if any(message["role"] != "tool" for message in result["tool_results"]):
        raise ValueError("tool_results must contain tool-role messages")
    return result


def validate_training_episode(episode: Mapping[str, Any], root: Path | str) -> None:
    """Validate the versioned record and verify every referenced image.

    Args:
        episode: Standard episode record.
        root: Directory that owns its relative media references.
    """
    Draft202012Validator(
        training_episode_schema(), format_checker=FormatChecker()
    ).validate(episode)
    calls = episode["calls"]
    ids = [call["call_id"] for call in calls]
    if len(ids) != len(set(ids)):
        raise ValueError("LLM call IDs must be unique within an episode")
    root = Path(root).resolve()
    for call in calls:
        if call["status"] == "completed" and call["response"] is None:
            raise ValueError("completed LLM call needs a response")
        if (
            call["response"] is not None
            and call["response"]["message"]["role"] != "assistant"
        ):
            raise ValueError("LLM response message must have assistant role")
        tool_call_ids = (
            {tool["id"] for tool in call["response"]["message"]["tool_calls"]}
            if call["response"] is not None
            else set()
        )
        if len(tool_call_ids) != len(
            call["response"]["message"]["tool_calls"] if call["response"] else ()
        ):
            raise ValueError("tool call IDs must be unique within an LLM response")
        for result in call["tool_results"]:
            if (
                result["role"] != "tool"
                or result.get("tool_call_id") not in tool_call_ids
            ):
                raise ValueError(
                    "tool results must match this LLM response's tool calls"
                )
        messages = [*call["request"]["messages"], *call["tool_results"]]
        if call["response"] is not None:
            messages.append(call["response"]["message"])
        for message in messages:
            if message["role"] != "assistant" and message["tool_calls"]:
                raise ValueError("only assistant messages may contain tool calls")
            if message["role"] == "tool" and "tool_call_id" not in message:
                raise ValueError("tool messages need tool_call_id")
            for part in message["content"]:
                if part["type"] == "provider_item":
                    _copy_provider_json(part["item"])
                if part["type"] != "image":
                    continue
                path = (root / part["asset_ref"]).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f"missing image asset: {part['asset_ref']}")
                if hashlib.sha256(path.read_bytes()).hexdigest() != part["sha256"]:
                    raise ValueError(f"image checksum mismatch: {part['asset_ref']}")


def convert_episode(
    task: Mapping[str, Any],
    calls: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    outcome: Mapping[str, Any] | None = None,
    episode_id: str | None = None,
    source_dir: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Convert one benchmark task and its ordered LLM calls to the standard format.

    Each call supplies the *actual* LLM request and response, including the
    messages, exposed tool definitions and multimodal content. Images may be
    supplied as bytes, base64, file paths or references relative to source_dir.
    ``provider_item`` blocks preserve opaque provider context, not normalized text.
    The destination must not already exist; it is published only after all
    records and images validate.

    Args:
        task: Benchmark name, task ID, instruction, seed and optional metadata.
        calls: Ordered mappings with request, response and optional tool results.
        output_dir: New episode directory to publish.
        outcome: Benchmark-native success and score; unknown if not yet scored.
        episode_id: Stable ID to use in the record, defaulting to the directory name.
        source_dir: Root for image asset_ref values, such as a RoboRSI run directory.
        provenance: Original run identity and recorded component versions.

    Returns:
        Path to the published episode.json file.
    """
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"training episode already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    partial.mkdir()
    try:
        assets = _Assets(partial, Path(source_dir) if source_dir is not None else None)
        source = _mapping(task, "task")
        task_record = {
            "benchmark": source["benchmark"],
            "task_id": source["task_id"],
            "instruction": source["instruction"],
            "metadata": dict(_mapping(source.get("metadata", {}), "task metadata")),
        }
        for key in ("benchmark_version", "seed"):
            if key in source:
                task_record[key] = source[key]
        scored = _mapping(outcome or {}, "outcome")
        result = {
            "status": scored.get("status", "unknown"),
            "success": scored.get("success"),
            "score": scored.get("score"),
            "metadata": dict(_mapping(scored.get("metadata", {}), "outcome metadata")),
        }
        if result["status"] == "unknown" and result["success"] is not None:
            result["status"] = "success" if result["success"] else "failure"
        source_info = _mapping(provenance or {}, "provenance")
        source_record = {
            "source_kind": source_info.get("source_kind", "benchmark"),
            "source_run_id": source_info.get("source_run_id"),
            "source_manifest_sha256": source_info.get("source_manifest_sha256"),
            "captured_at": source_info.get("captured_at"),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "versions": dict(_mapping(source_info.get("versions", {}), "versions")),
            "metadata": dict(
                _mapping(source_info.get("metadata", {}), "provenance metadata")
            ),
        }
        episode = {
            "schema_version": 1,
            "episode_id": episode_id or destination.name,
            "task": task_record,
            "outcome": result,
            "provenance": source_record,
            "calls": [
                _call(call, index, assets) for index, call in enumerate(calls, 1)
            ],
        }
        validate_training_episode(episode, partial)
        document = partial / "episode.json"
        document.write_text(
            json.dumps(episode, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, destination)
        return destination / "episode.json"
    except BaseException:
        shutil.rmtree(partial)
        raise


def load_training_episode(path: str | Path) -> dict[str, Any]:
    """Load and validate a previously published training episode.

    Args:
        path: Episode JSON file or its containing directory.

    Returns:
        Validated standard episode record.
    """
    source = Path(path)
    document = source / "episode.json" if source.is_dir() else source
    episode = json.loads(document.read_text(encoding="utf-8"))
    validate_training_episode(episode, document.parent)
    return episode
