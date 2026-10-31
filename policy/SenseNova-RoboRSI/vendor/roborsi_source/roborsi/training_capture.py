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

"""Convert one explicit provider capture into a portable training episode.

Requests are observations, not inferred logical turns or deduplicated retries.
Only each call's response is a training target; assistant history stays context.
Provider ciphertext remains opaque throughout conversion.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from roborsi.training_data import convert_episode

_REQUEST_NAME = re.compile(r"([0-9]{6,})\.request\.json\Z")


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _json(body: bytes | str) -> Any:
    return json.loads(body, object_pairs_hook=_object, parse_constant=_invalid_constant)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _source_file(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).name != name or not path.resolve().is_relative_to(root):
        raise ValueError(f"capture path escapes its directory: {name}")
    if not path.is_file():
        raise ValueError(f"capture file is missing: {name}")
    return path


def _redacted(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_redacted(item) for item in value.values())
    if isinstance(value, list):
        return any(_redacted(item) for item in value)
    return value == "[REDACTED]"


def _check_references(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "item_reference":
            raise ValueError("unresolved Responses item_reference")
        if value.get("file_id"):
            raise ValueError("unresolved provider file_id media reference")
        if value.get("type") in {"input_file", "input_audio", "audio", "video"}:
            raise ValueError(
                "unsupported provider media; supply a supported self-contained capture"
            )
        if value.get("type") in {"input_image", "image_url"}:
            url = value.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise ValueError(
                    "unresolved image URL; conversion does not download media"
                )
        for item in value.values():
            _check_references(item)
    elif isinstance(value, list):
        for item in value:
            _check_references(item)


def _provider_item(item: dict[str, Any]) -> dict[str, Any]:
    _check_references(item)
    return {"type": "provider_item", "provider": "openai.responses", "item": item}


def _content(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        raise ValueError("Responses message content must be text or an array")
    content = []
    for part in value:
        part = _mapping(part, "Responses content item")
        kind = part.get("type")
        if kind in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("Responses text content must contain text")
            content.append({"type": "text", "text": text})
        elif kind in {"input_image", "image_url"}:
            _check_references(part)
            url = part["image_url"]
            url = url["url"] if isinstance(url, dict) else url
            header, separator, data = url.partition(",")
            if not separator or not header.endswith(";base64"):
                raise ValueError("image data URL must use base64")
            content.append(
                {"type": "image", "media_type": header[5:-7], "base64": data}
            )
        else:
            content.append(_provider_item(part))
    return content


def _reasoning(item: dict[str, Any]) -> dict[str, Any]:
    summary = item.get("summary", [])
    if not isinstance(summary, list):
        raise ValueError("Responses reasoning summary must be an array")
    texts = []
    for entry in summary:
        entry = _mapping(entry, "reasoning summary item")
        if entry.get("type") != "summary_text" or not isinstance(
            entry.get("text"), str
        ):
            raise ValueError("unsupported reasoning summary content")
        texts.append(entry["text"])
    content = {"type": "reasoning", "text": "\n".join(texts)}
    if "encrypted_content" in item and item["encrypted_content"] is not None:
        if not isinstance(item["encrypted_content"], str):
            raise ValueError("reasoning encrypted_content must be text")
        content["encrypted_content"] = item["encrypted_content"]
    return content


def _message(item: dict[str, Any]) -> dict[str, Any]:
    kind = item.get("type")
    if kind == "message" or kind is None and "role" in item:
        if item.get("role") not in {"system", "developer", "user", "assistant"}:
            raise ValueError("unsupported Responses message role")
        return {"role": item["role"], "content": _content(item.get("content", []))}
    if kind == "reasoning":
        return {"role": "assistant", "content": [_reasoning(item)]}
    if kind in {"function_call", "custom_tool_call"}:
        name = _text(item.get("name"), "tool name")
        if item.get("namespace"):
            name = _text(item["namespace"], "tool namespace") + "." + name
        if kind == "custom_tool_call":
            raw = item.get("input")
            if not isinstance(raw, str):
                raise ValueError("custom_tool_call.input must be text")
            arguments = {"_raw": raw}
        else:
            raw = item.get("arguments")
            if isinstance(raw, str):
                try:
                    arguments = _json(raw)
                except ValueError:
                    arguments = {"_raw": raw}
                if not isinstance(arguments, dict):
                    arguments = {"_raw": raw}
            elif isinstance(raw, dict):
                arguments = raw
            else:
                raise ValueError(
                    "function_call.arguments must be JSON text or an object"
                )
        return {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                {
                    "id": _text(item.get("call_id"), "tool call_id"),
                    "name": name,
                    "arguments": arguments,
                }
            ],
        }
    if kind in {"function_call_output", "custom_tool_call_output"}:
        return {
            "role": "tool",
            "tool_call_id": _text(item.get("call_id"), "tool result call_id"),
            "content": _content(item.get("output")),
        }
    if not isinstance(kind, str):
        raise ValueError("Responses item requires a type")
    return {"role": "assistant", "content": [_provider_item(item)]}


def _function_tools(tools: list[Any], namespace: str = "") -> list[dict[str, Any]]:
    result = []
    for raw in tools:
        tool = _mapping(raw, "Responses tool definition")
        if tool.get("type") == "namespace":
            nested = tool.get("tools")
            if not isinstance(nested, list):
                raise ValueError("namespace tools must be an array")
            name = namespace + _text(tool.get("name"), "namespace name") + "."
            result.extend(_function_tools(nested, name))
        elif tool.get("type") == "function":
            result.append(
                {
                    "name": namespace + _text(tool.get("name"), "function name"),
                    "description": tool.get("description") or "",
                    "input_schema": _mapping(
                        tool.get("parameters", {}), "function parameters"
                    ),
                }
            )
    return result


def _request(value: dict[str, Any], provider: str) -> dict[str, Any]:
    if value.get("previous_response_id") or value.get("conversation"):
        raise ValueError(
            "unresolved provider previous_response_id or conversation state"
        )
    _check_references(value.get("input"))
    inputs = value.get("input", [])
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    if not isinstance(inputs, list):
        raise ValueError("Responses input must be text or an array")
    instructions = value.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("Responses instructions must be text or null")
    tools = value.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("Responses tools must be an array")
    settings = {
        key: item
        for key, item in value.items()
        if key not in {"input", "instructions", "model", "tools"}
    }
    settings["responses_tools"] = tools
    return {
        "model": _text(value.get("model"), "request model"),
        "provider": provider,
        "system_prompt": instructions or "",
        "messages": [
            _message(_mapping(item, "Responses input item")) for item in inputs
        ],
        "tools": _function_tools(tools),
        "settings": settings,
    }


def _response(value: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = value.get("output", [])
    if not isinstance(output, list):
        raise ValueError("Responses output must be an array")
    content, tool_calls, order = [], [], []
    for index, raw in enumerate(output):
        item = _mapping(raw, "Responses output item")
        message = _message(item)
        if message["role"] != "assistant":
            raise ValueError("Responses target output must have assistant role")
        blocks, calls = message["content"], message.get("tool_calls", [])
        order.append(
            {
                "index": index,
                "type": item.get("type"),
                "id": item.get("id"),
                "content_indices": list(
                    range(len(content), len(content) + len(blocks))
                ),
                "tool_call_indices": list(
                    range(len(tool_calls), len(tool_calls) + len(calls))
                ),
            }
        )
        content.extend(blocks)
        tool_calls.extend(calls)
    return {
        "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        "finish_reason": value.get("status"),
    }, order


def _usage(response: Mapping[str, Any]) -> dict[str, Any]:
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        raise ValueError("Responses usage must be an object")
    details = usage.get("input_tokens_details") or {}
    if not isinstance(details, dict):
        raise ValueError("Responses input_tokens_details must be an object")
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read_tokens": details.get("cached_tokens"),
        "cache_write_tokens": details.get("cache_write_tokens"),
    }


def _read_response(
    root: Path, stem: str, metadata: dict[str, Any]
) -> tuple[dict[str, Any], bytes]:
    if metadata.get("raw_response") != f"{stem}.response.bin":
        raise ValueError("capture raw_response path does not match its numbered record")
    body = _source_file(root, metadata["raw_response"]).read_bytes()
    if metadata.get("response_sha256") != _digest(body):
        raise ValueError("capture response checksum mismatch")
    if metadata.get("response_bytes") != len(body):
        raise ValueError("capture response byte count mismatch")
    encoding = metadata.get("content_encoding", "").lower()
    try:
        decoded = (
            gzip.decompress(body)
            if encoding == "gzip"
            else zlib.decompress(body)
            if encoding == "deflate"
            else body
        )
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError("capture response compression is invalid") from exc
    if encoding not in {"", "identity", "gzip", "deflate"}:
        raise ValueError("unsupported capture response encoding")
    from roborsi.planner.provider_capture import parse_response_body

    parsed = parse_response_body(
        decoded,
        metadata.get("content_type", ""),
        endpoint=metadata.get("endpoint", "/responses"),
        include_response=True,
    )
    return parsed, body


def convert_codex_capture(
    capture_dir: str | Path,
    task: Mapping[str, Any],
    *,
    output_dir: str | Path,
    outcome: Mapping[str, Any] | None = None,
    episode_id: str | None = None,
    versions: Mapping[str, str] | None = None,
    allow_partial: bool = False,
    require_encrypted_content: bool = False,
) -> Path:
    """Convert captured actual requests and raw responses to a training episode.

    Args:
        capture_dir: One leaf provider capture directory with summary.json.
        task: Explicit benchmark task identity; never inferred from model text.
        output_dir: New immutable episode directory, published after validation.
        outcome: Optional externally supplied native outcome, otherwise unknown.
        episode_id: Optional stable episode identifier.
        versions: Explicit recorded component versions.
        allow_partial: Retain failed or unfinished attempts with non-completed status.
        require_encrypted_content: Require ciphertext in a completed response target.

    Returns:
        Path to the standard episode.json document.
    """
    if type(allow_partial) is not bool or type(require_encrypted_content) is not bool:
        raise ValueError("capture conversion flags must be booleans")
    root = Path(capture_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("capture_dir must be one existing leaf capture directory")
    destination = Path(output_dir).expanduser().resolve()
    if destination.is_relative_to(root):
        raise ValueError("training output must be outside the source capture directory")
    summary_bytes = _source_file(root, "summary.json").read_bytes()
    summary = _mapping(_json(summary_bytes), "capture summary")
    if summary.get("schema_version") != 1:
        raise ValueError("unsupported provider capture schema_version")
    count = summary.get("requests")
    if type(count) is not int or count < 1:
        raise ValueError("capture summary must identify at least one request")
    if not allow_partial and (
        summary.get("pending_requests") or summary.get("capture_errors")
    ):
        raise ValueError("capture contains pending requests or capture errors")
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("capture summary records must be an array")
    recorded = {}
    for record in records:
        record = _mapping(record, "capture response record")
        number = record.get("request_id")
        if (
            type(number) is not int
            or number < 1
            or number > count
            or number in recorded
        ):
            raise ValueError("invalid or duplicate capture request_id")
        recorded[number] = record
    names = {
        int(match[1]): entry.name
        for entry in root.iterdir()
        if (match := _REQUEST_NAME.fullmatch(entry.name))
    }
    if set(names) != set(range(1, count + 1)):
        raise ValueError("capture request records are missing or exceed the summary")
    provider = _text(summary.get("model_provider"), "capture model_provider")
    calls, evidence = [], []
    pending_tools: dict[str, list[int]] = {}
    results_seen: dict[tuple[int, str], dict[str, Any]] = {}
    cipher_targets = 0
    for number in sorted(names):
        stem = names[number].split(".", 1)[0]
        request_bytes = _source_file(root, names[number]).read_bytes()
        request_record = _mapping(_json(request_bytes), "capture request record")
        if (
            request_record.get("request_id") != number
            or request_record.get("method") != "POST"
        ):
            raise ValueError("capture request record identity mismatch")
        if (
            request_record.get("redacted_paths")
            or request_record.get("request_capture_complete") is False
        ):
            raise ValueError("capture request is redacted or incomplete")
        forwarded = _mapping(
            request_record.get("forwarded"), "captured forwarded request"
        )
        if _redacted(forwarded):
            raise ValueError("capture request contains redacted content")
        expected = request_record.get("forwarded_record_sha256")
        if expected is not None and expected != _digest(
            json.dumps(
                forwarded,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ):
            raise ValueError("capture forwarded request checksum mismatch")
        endpoint = request_record.get("endpoint")
        if endpoint not in {"/responses", "/responses/compact"}:
            raise ValueError("unsupported capture request endpoint")
        record = recorded.get(number)
        parsed = None
        if record is not None:
            metadata = _mapping(
                _json(_source_file(root, f"{stem}.response.json").read_bytes()),
                "capture response metadata",
            )
            if metadata != record or metadata.get("endpoint") != endpoint:
                raise ValueError("capture response record does not match its summary")
            parsed, _ = _read_response(root, stem, metadata)
            if parsed.get("response_conflicts"):
                raise ValueError("capture response contains conflicting output items")
            complete = bool(
                record.get("transport_complete")
                and 200 <= record.get("status_code", 0) < 300
                and parsed["response_completed"]
                and parsed["parse_status"] == "parsed"
            )
        else:
            complete = False
            metadata = {"status_code": 0, "transport_complete": False}
        if not complete and not allow_partial:
            raise ValueError(
                f"capture request {number} has an incomplete or failed response"
            )
        detail = {
            "request_id": number,
            "endpoint": endpoint,
            "request_file": names[number],
            "request_sha256": _digest(request_bytes),
            "forwarded_sha256": request_record.get("forwarded_sha256"),
            "forwarded_record_verified": expected is not None,
            "mutated": request_record.get("mutated", False),
            "response_sha256": metadata.get("response_sha256"),
            "response_id": parsed.get("response_id") if parsed else None,
            "transport_complete": metadata.get("transport_complete", False),
            "http_status": metadata.get("status_code", 0),
            "parse_status": parsed.get("parse_status") if parsed else "pending",
            "complete": complete,
        }
        evidence.append(detail)
        if endpoint == "/responses/compact":
            detail["training_target"] = False
            continue
        normalized = _request(forwarded, provider)
        for message in normalized["messages"]:
            if message["role"] != "tool":
                continue
            call_id = message["tool_call_id"]
            owners = pending_tools.get(call_id, [])
            if not owners:
                detail.setdefault("unmatched_context_tool_results", []).append(call_id)
                continue
            if len(owners) != 1:
                detail.setdefault("ambiguous_context_tool_results", []).append(call_id)
                continue
            owner = owners[0]
            key = (owner, call_id)
            if key in results_seen:
                if results_seen[key] != message:
                    raise ValueError(
                        "conflicting tool feedback for one provider call_id"
                    )
            else:
                results_seen[key] = message
                calls[owner]["tool_results"].append(message)
        response = parsed.get("response") if parsed else None
        status_code = metadata.get("status_code", 0)
        status = (
            "completed"
            if complete
            else "error"
            if status_code >= 400 or metadata.get("transport_error")
            else "partial"
        )
        normalized_response = None
        order = []
        if isinstance(response, dict) and 200 <= status_code < 300:
            normalized_response, order = _response(response)
        detail["response_output_order"] = order
        detail["call_id"] = f"capture-{number:06d}"
        detail["status"] = status
        detail["training_target"] = complete
        call = {
            "call_id": detail["call_id"],
            "request": normalized,
            "response": normalized_response,
            "tool_results": [],
            "status": status,
            "usage": _usage(response) if isinstance(response, dict) else {},
            "attempts": [],
        }
        if complete and normalized_response is None:
            raise ValueError("completed capture has no reconstructable model response")
        if normalized_response is not None:
            for tool in normalized_response["message"]["tool_calls"]:
                pending_tools.setdefault(tool["id"], []).append(len(calls))
            if complete:
                cipher_targets += sum(
                    bool(item.get("encrypted_content"))
                    for item in normalized_response["message"]["content"]
                    if item["type"] == "reasoning"
                )
        calls.append(call)
    if not calls:
        raise ValueError("capture contains no Responses model calls to convert")
    if require_encrypted_content and not cipher_targets:
        raise ValueError(
            "capture contains no nonempty encrypted_content in completed response targets"
        )
    details = {item.get("call_id"): item for item in evidence if "call_id" in item}
    for call in calls:
        expected_ids = (
            {item["id"] for item in call["response"]["message"]["tool_calls"]}
            if call["response"]
            else set()
        )
        observed_ids = {item["tool_call_id"] for item in call["tool_results"]}
        details[call["call_id"]]["missing_tool_results"] = sorted(
            expected_ids - observed_ids
        )
    return convert_episode(
        task,
        calls,
        output_dir=destination,
        outcome=outcome,
        episode_id=episode_id,
        provenance={
            "source_kind": "codex_provider_capture",
            "source_manifest_sha256": _digest(summary_bytes),
            "versions": dict(versions or {}),
            "metadata": {
                "training_target": "response_only",
                "ordering": "request_id; concurrent requests may overlap",
                "capture_dir": str(root),
                "model_provider": provider,
                "upstream_base_url": summary.get("upstream_base_url"),
                "allow_partial": allow_partial,
                "require_encrypted_content": require_encrypted_content,
                "nonempty_encrypted_targets": cipher_targets,
                "capture_errors": summary.get("capture_errors", []),
                "requests": evidence,
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    """Convert one explicitly named capture without contacting any service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outcome-file")
    parser.add_argument("--episode-id")
    parser.add_argument("--versions-file")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-encrypted-content", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = convert_codex_capture(
            args.capture_dir,
            _json(Path(args.task_file).read_bytes()),
            output_dir=args.output_dir,
            outcome=_json(Path(args.outcome_file).read_bytes())
            if args.outcome_file
            else None,
            episode_id=args.episode_id,
            versions=_json(Path(args.versions_file).read_bytes())
            if args.versions_file
            else None,
            allow_partial=args.allow_partial,
            require_encrypted_content=args.require_encrypted_content,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
