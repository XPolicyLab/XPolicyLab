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

"""Capture provider Responses bodies without changing the agent's planner.

Raw response bytes are authoritative. Parsed reasoning items are a convenience
index; encrypted content remains opaque and is never decrypted. Capture is
explicitly enabled and may contain private prompts, tool results and responses.
Credential headers are forwarded but never persisted.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import math
import os
import re
import socket
import threading
import time
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import IPv6Address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import proxy_bypass_environment

import httpx
import uvicorn

_HOP_HEADERS = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
    b"host",
}
_CREDENTIAL_KEYS = {
    "authorization",
    "cookie",
    "set_cookie",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "client_secret",
}


@dataclass(frozen=True)
class ProviderCaptureConfig:
    """A fixed upstream and optional, recorded encrypted-content request change."""

    upstream_base_url: str
    model_provider: str
    request_encrypted_content: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.upstream_base_url, str):
            raise ValueError("provider_capture.upstream_base_url must be a URL")
        parsed = urlsplit(self.upstream_base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "?" in self.upstream_base_url
            or "#" in self.upstream_base_url
            or any(char.isspace() for char in self.upstream_base_url)
        ):
            raise ValueError(
                "provider_capture.upstream_base_url requires an HTTP(S) URL without credentials, query or fragment"
            )
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError(
                "provider_capture.upstream_base_url has an invalid port"
            ) from exc
        if not isinstance(self.model_provider, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", self.model_provider
        ):
            raise ValueError(
                "provider_capture.model_provider must be a nonempty provider identifier"
            )
        if type(self.request_encrypted_content) is not bool:
            raise ValueError(
                "provider_capture.request_encrypted_content must be boolean"
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderCaptureConfig:
        """Validate capture options at the configuration boundary."""
        if not isinstance(value, dict):
            raise ValueError("provider_capture must be a mapping")
        unknown = set(value) - {
            "upstream_base_url",
            "model_provider",
            "request_encrypted_content",
        }
        if unknown:
            raise ValueError(f"Unknown provider_capture fields: {sorted(unknown)}")
        for key in ("upstream_base_url", "model_provider"):
            if key not in value:
                raise ValueError(f"provider_capture.{key} is required")
        return cls(**value)


def _redact(
    value: Any, *, redacted_paths: list[str] | None = None, path: str = ""
) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            child_path = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key.lower().replace("-", "_") in _CREDENTIAL_KEYS:
                result[key] = "[REDACTED]"
                if item != "[REDACTED]" and redacted_paths is not None:
                    redacted_paths.append(child_path)
            else:
                result[key] = _redact(
                    item, redacted_paths=redacted_paths, path=child_path
                )
        return result
    if isinstance(value, list):
        return [
            _redact(item, redacted_paths=redacted_paths, path=f"{path}/{index}")
            for index, item in enumerate(value)
        ]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _forward_headers(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    excluded = set(_HOP_HEADERS)
    for name, value in headers:
        if name.lower() == b"connection":
            excluded.update(part.strip().lower() for part in value.split(b","))
    return [(name, value) for name, value in headers if name.lower() not in excluded]


def _merge_output_items(
    response: dict[str, Any],
    done_items: list[tuple[Any, dict[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    """Combine final output with complete stream items, reporting conflicts."""
    output = response.get("output", [])
    if not isinstance(output, list):
        return response, ["Response output is not a list"]
    output = [dict(item) if isinstance(item, dict) else item for item in output]
    conflicts: list[str] = []
    for output_index, item in done_items:
        item_id = item.get("id")
        matches = [
            index
            for index, existing in enumerate(output)
            if item_id and isinstance(existing, dict) and existing.get("id") == item_id
        ]
        if len(matches) > 1:
            conflicts.append("Duplicate output item identifier")
            continue
        if output_index is not None and (
            type(output_index) is not int or output_index < 0
        ):
            conflicts.append("Invalid output item index")
            continue
        if matches:
            index = matches[0]
            if output_index is not None and index != output_index:
                conflicts.append("Output item identifier and index disagree")
                continue
        elif output_index is not None:
            index = output_index
            # Partial streams may finish a later item before earlier positions.
            # Bound allocation from untrusted indices while retaining these gaps.
            if index >= max(10_000, len(output) + len(done_items)):
                conflicts.append("Output item index is out of range")
                continue
            output.extend([None] * max(0, index + 1 - len(output)))
        else:
            index = len(output)
            output.append(None)
        existing = output[index]
        if existing is None:
            output[index] = dict(item)
            continue
        if not isinstance(existing, dict):
            conflicts.append("Output item is not an object")
            continue
        if item_id and existing.get("id") not in (None, "", item_id):
            conflicts.append("Conflicting output item identifiers")
            continue
        for key, value in item.items():
            if key not in existing or (
                key in {"id", "encrypted_content"} and existing[key] in (None, "")
            ):
                existing[key] = value
            elif key == "encrypted_content" and value in (None, ""):
                continue
            elif existing[key] != value:
                conflicts.append(f"Conflicting output item field: {key}")
    if any(not isinstance(item, dict) for item in output):
        conflicts.append("Response output contains missing or invalid items")
    return {**response, "output": output}, conflicts


class _ResponseIntegrityError(ValueError):
    """JSON cannot represent one unambiguous, finite provider response."""


def _strict_response_json(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise _ResponseIntegrityError(f"Duplicate JSON object key: {key}")
            result[key] = value
        return result

    def nonfinite(_value: str) -> Any:
        raise _ResponseIntegrityError("Nonfinite JSON numeric value")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise _ResponseIntegrityError("Nonfinite JSON numeric value")
        return parsed

    return json.loads(
        text,
        object_pairs_hook=object_pairs,
        parse_constant=nonfinite,
        parse_float=finite_float,
    )


def parse_response_body(
    body: bytes,
    content_type: str,
    *,
    endpoint: str = "/responses",
    include_response: bool = False,
) -> dict[str, Any]:
    """Index JSON or SSE reasoning items, preserving opaque values exactly.

    This parses a captured body, not a Codex transcript. A stream's final response
    replaces repeated output-item events in the index, unless that would discard
    captured encrypted content. The raw file retains every event. Missing
    completion is reported even when individual items parse. Compaction replies
    are JSON output lists without a required status; their ciphertext remains
    in the raw body and is not counted as newly generated reasoning.

    ``include_response`` also exposes the complete provider response for offline
    conversion. Partial responses remain marked incomplete; inconsistent stream
    items are reported as malformed rather than silently overwritten. The normal
    capture summary does not contain a second copy of the full response payload.
    """
    records: dict[tuple[str, str], dict[str, Any]] = {}
    response_id = ""
    completed = False
    malformed_events = 0
    truncated = False
    is_compaction = endpoint == "/responses/compact"
    captured_response: dict[str, Any] | None = None
    created_response: dict[str, Any] | None = None
    done_items: list[tuple[Any, dict[str, Any]]] = []
    response_conflicts: list[str] = []

    def set_response_id(value: Any) -> None:
        nonlocal response_id
        if not value:
            return
        if include_response and response_id and response_id != str(value):
            if "Conflicting response identifiers" not in response_conflicts:
                response_conflicts.append("Conflicting response identifiers")
        response_id = str(value)
        for key in list(records):
            if key[0] == "":
                item = records.pop(key)
                item["response_id"] = response_id
                records[(response_id, key[1])] = item

    def add_item(item: Any, index: Any = None) -> None:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            return
        identity = str(item.get("id") or f"output:{index}")
        encrypted = item.get("encrypted_content")
        state = (
            "absent"
            if encrypted is None
            else "empty"
            if encrypted == ""
            else "nonempty"
            if isinstance(encrypted, str)
            else "malformed"
        )
        previous = records.get((response_id, identity))
        if (
            previous is not None
            and previous["encrypted_content_state"] == "nonempty"
            and state != "nonempty"
        ):
            return
        records[(response_id, identity)] = {
            "response_id": response_id or None,
            "item_id": item.get("id"),
            "output_index": index,
            "encrypted_content_state": state,
            "item": item,
        }

    def consume_response(response: dict[str, Any]) -> None:
        nonlocal completed
        set_response_id(response.get("id"))
        output = response.get("output", None if is_compaction else [])
        if not isinstance(output, list):
            raise ValueError("Response output must be a list")
        if is_compaction:
            if any(not isinstance(item, dict) for item in output):
                raise ValueError("Compaction output items must be objects")
            completed = True
            return
        completed = response.get("status") == "completed"
        for index, item in enumerate(output):
            add_item(item, index)

    try:
        text = body.decode("utf-8-sig")
        if not is_compaction and "text/event-stream" in content_type.lower():
            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            frames = normalized.split("\n\n")
            truncated = bool(frames[-1].strip())
            for frame in frames:
                data = "\n".join(
                    line[5:].lstrip(" ")
                    for line in frame.split("\n")
                    if line.startswith("data:")
                )
                if not data or data == "[DONE]":
                    continue
                try:
                    event = (
                        _strict_response_json(data)
                        if include_response
                        else json.loads(data)
                    )
                except _ResponseIntegrityError as exc:
                    response_conflicts.append(str(exc))
                    continue
                except ValueError:
                    malformed_events += 1
                    continue
                if not isinstance(event, dict):
                    malformed_events += 1
                    continue
                response = event.get("response")
                if isinstance(response, dict):
                    set_response_id(response.get("id"))
                else:
                    set_response_id(event.get("response_id"))
                event_type = event.get("type")
                if (
                    include_response
                    and event_type == "response.created"
                    and isinstance(response, dict)
                ):
                    created_response = response
                if event_type in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                } and isinstance(response, dict):
                    if include_response:
                        captured_response = response
                    consume_response(response)
                    completed = (
                        event_type == "response.completed"
                        and response.get("status", "completed") == "completed"
                    )
                elif event_type == "response.output_item.done":
                    add_item(event.get("item"), event.get("output_index"))
                    if include_response:
                        item = event.get("item")
                        if isinstance(item, dict):
                            done_items.append((event.get("output_index"), item))
                        else:
                            response_conflicts.append(
                                "Completed output item is not an object"
                            )
        else:
            response = (
                _strict_response_json(text) if include_response else json.loads(text)
            )
            if not isinstance(response, dict):
                raise ValueError("Response must be an object")
            if include_response:
                captured_response = response
            consume_response(response)
    except _ResponseIntegrityError as exc:
        response_conflicts.append(str(exc))
    except (ValueError, TypeError):
        malformed_events += 1
    if include_response:
        if captured_response is None and (created_response is not None or done_items):
            captured_response = {**(created_response or {}), "output": []}
            if response_id:
                captured_response.setdefault("id", response_id)
        if (
            captured_response is not None
            and done_items
            and "Conflicting response identifiers" not in response_conflicts
        ):
            captured_response, conflicts = _merge_output_items(
                captured_response, done_items
            )
            response_conflicts.extend(conflicts)
        malformed_events += len(response_conflicts)
    result = {
        "response_id": response_id or None,
        "response_kind": "compaction" if is_compaction else "response",
        "response_completed": completed,
        "parse_status": "malformed"
        if malformed_events
        else "truncated"
        if truncated
        else "parsed",
        "malformed_events": malformed_events,
        "truncated": truncated,
        "reasoning_items": list(records.values()),
    }
    if include_response:
        result["response"] = captured_response
        result["response_conflicts"] = response_conflicts
    return result


def _upstream_proxy(url: str, env: Mapping[str, str]) -> str | None:
    """Resolve the fixed upstream's proxy without reading or mutating global env."""
    proxies = {}
    for scheme in ("http", "https", "all", "no"):
        key = f"{scheme}_proxy"
        # The lowercase value wins even when empty, matching standard env policy.
        value = env.get(key, env.get(key.upper(), ""))
        if value:
            proxies[scheme] = value
    target = urlsplit(url)
    if proxy_bypass_environment(target.netloc, proxies):
        return None
    # urllib matches bracketed authorities, but NO_PROXY commonly uses bare IPv6.
    if target.hostname and ":" in target.hostname:
        for entry in proxies.get("no", "").split(","):
            try:
                if IPv6Address(entry.strip()) == IPv6Address(target.hostname):
                    return None
            except ValueError:
                continue
    proxy = proxies.get(target.scheme, proxies.get("all"))
    if proxy and "://" not in proxy:
        proxy = "http://" + proxy
    return proxy


class ProviderCapture:
    """Serve a local, fixed-destination Responses proxy until explicitly stopped.

    Args:
        config: Validated upstream and opt-in request mutation settings.
        output_dir: Per-run capture directory. Existing capture records are refused.
        env: Effective agent process environment. None snapshots the current
            process. An explicit empty mapping disables environment proxies.
    """

    def __init__(
        self,
        config: ProviderCaptureConfig,
        output_dir: Path | str,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        # Resolve once from the launch environment; credentials stay out of records.
        self._proxy = _upstream_proxy(
            config.upstream_base_url, dict(os.environ if env is None else env)
        )
        self._prefix = urlsplit(config.upstream_base_url).path.rstrip("/")
        self._port = 0
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._records: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._sequence = 0
        self._started = False
        self._closed = False
        self._owns_output = False

    @property
    def base_url(self) -> str:
        """Return the local base URL after startup."""
        if not self._started:
            raise RuntimeError("ProviderCapture has not started")
        return f"http://127.0.0.1:{self._port}{self._prefix}"

    @property
    def summary(self) -> dict[str, Any]:
        """Return counts with explicit HTTP, parsing and transport completeness."""
        with self._lock:
            records = list(self._records)
            requests = self._sequence
        return {
            "schema_version": 1,
            "upstream_base_url": self.config.upstream_base_url,
            "model_provider": self.config.model_provider,
            "request_encrypted_content": self.config.request_encrypted_content,
            "requests": requests,
            "recorded_responses": len(records),
            "complete_responses": sum(record["complete"] for record in records),
            "incomplete_responses": sum(not record["complete"] for record in records),
            "pending_requests": requests - len(records),
            "capture_errors": list(self._errors),
            "http_errors": sum(
                record.get("status_code", 0) >= 400 for record in records
            ),
            "reasoning_items": sum(record["reasoning_items"] for record in records),
            "nonempty_encrypted_items": sum(
                record["nonempty_encrypted_items"] for record in records
            ),
            "encrypted_content_chars": sum(
                record["encrypted_content_chars"] for record in records
            ),
            "records": records,
        }

    def start(self, *, ready_timeout_s: float = 5.0) -> str:
        """Start a listener bound only to loopback, without an upstream request."""
        if self._closed:
            raise RuntimeError("ProviderCapture cannot be restarted")
        if self._started:
            return self.base_url
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            raise ValueError("Provider capture output directory must be empty")
        self._owns_output = True
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        self._port = int(listener.getsockname()[1])
        config = uvicorn.Config(
            self._app,
            interface="asgi3",
            log_level="critical",
            access_log=False,
            lifespan="off",
            timeout_graceful_shutdown=1,
        )
        self._server = uvicorn.Server(config)

        def run() -> None:
            asyncio.run(self._run(listener))

        self._thread = threading.Thread(
            target=run, name="provider-capture", daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + ready_timeout_s
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() >= deadline:
                self.stop()
                raise RuntimeError("Provider capture listener failed to start")
            time.sleep(0.01)
        self._started = True
        return self.base_url

    async def _run(self, listener: socket.socket) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            assert self._server is not None
            await self._server.serve(sockets=[listener])
        finally:
            listener.close()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        """Cancel active transfers, stop serving and flush the summary boundedly."""
        if self._closed:
            return
        if self._server is not None:
            self._server.should_exit = True
        if self._loop is not None and not self._loop.is_closed():

            def cancel() -> None:
                for task in list(self._tasks):
                    task.cancel()

            self._loop.call_soon_threadsafe(cancel)
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise RuntimeError(
                    "Provider capture did not stop within its cleanup deadline"
                )
        self._closed = True
        if self._owns_output:
            _write_json(self.output_dir / "summary.json", self.summary)
        if self._errors:
            raise RuntimeError(
                "Provider capture could not persist all response records"
            )

    def __enter__(self) -> ProviderCapture:
        self.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop()

    async def _app(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        if scope["method"] != "POST" or scope["path"] not in {
            self._prefix + "/responses",
            self._prefix + "/responses/compact",
        }:
            await self._error(send, 404, b"Unsupported provider capture route")
            return
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        task = asyncio.create_task(self._forward(scope, b"".join(chunks), send))
        self._tasks.add(task)

        async def disconnected() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass

        watcher = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait(
                {task, watcher}, return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done and not task.done():
                task.cancel()
            results = await asyncio.gather(task, return_exceptions=True)
            if isinstance(results[0], Exception):
                self._errors.append(type(results[0]).__name__)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            self._tasks.discard(task)

    @staticmethod
    async def _error(send: Any, status: int, message: bytes) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": message})

    async def _forward(self, scope: dict[str, Any], body: bytes, send: Any) -> None:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        stem = f"{sequence:06d}"
        endpoint = scope["path"][len(self._prefix) :]
        upstream_url = self.config.upstream_base_url.rstrip("/") + endpoint
        query = scope.get("query_string", b"")
        upstream_target = httpx.URL(upstream_url)
        if query:
            upstream_target = upstream_target.copy_with(query=query)
        query_metadata = {
            "query_present": bool(query),
            "query_sha256": hashlib.sha256(query).hexdigest() if query else None,
        }
        raw_path = self.output_dir / f"{stem}.response.bin"
        record: dict[str, Any] = {
            "request_id": sequence,
            "endpoint": endpoint,
            **query_metadata,
            "raw_response": raw_path.name,
            "status_code": 0,
            "transport_complete": False,
            "content_type": "",
        }
        started = False
        encoding = ""
        try:
            try:
                original = json.loads(body)
            except ValueError:
                original = None
            forwarded = original
            forwarded_body = body
            if self.config.request_encrypted_content and endpoint == "/responses":
                if not isinstance(original, dict):
                    raise ValueError("Capture mutation requires a JSON object")
                includes = original.get("include", [])
                if not isinstance(includes, list) or any(
                    not isinstance(value, str) for value in includes
                ):
                    raise ValueError(
                        "Capture mutation requires include to be a string array"
                    )
                forwarded = {**original, "store": False, "include": list(includes)}
                if "reasoning.encrypted_content" not in forwarded["include"]:
                    forwarded["include"].append("reasoning.encrypted_content")
                forwarded_body = json.dumps(
                    forwarded, ensure_ascii=False, separators=(",", ":")
                ).encode()
            redacted_paths: list[str] = []
            original_record = _redact(original, redacted_paths=redacted_paths)
            forwarded_record = _redact(forwarded, redacted_paths=redacted_paths)
            redacted_paths = sorted(set(redacted_paths))
            try:
                canonical_record = json.dumps(
                    forwarded_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                forwarded_record_sha256 = hashlib.sha256(canonical_record).hexdigest()
            except (TypeError, ValueError):
                forwarded_record_sha256 = None
            _write_json(
                self.output_dir / f"{stem}.request.json",
                {
                    "request_id": sequence,
                    "method": "POST",
                    "endpoint": endpoint,
                    "upstream_url": upstream_url,
                    **query_metadata,
                    "original": original_record,
                    "forwarded": forwarded_record,
                    "redacted_paths": redacted_paths,
                    "request_redacted": bool(redacted_paths),
                    "request_capture_complete": isinstance(forwarded_record, dict)
                    and not redacted_paths
                    and forwarded_record_sha256 is not None,
                    "forwarded_record_sha256": forwarded_record_sha256,
                    "original_sha256": hashlib.sha256(body).hexdigest(),
                    "forwarded_sha256": hashlib.sha256(forwarded_body).hexdigest(),
                    "mutated": forwarded_body != body,
                    "requested_encrypted_content": isinstance(forwarded, dict)
                    and isinstance(forwarded.get("include"), list)
                    and "reasoning.encrypted_content" in forwarded["include"],
                },
            )
            headers = _forward_headers(scope["headers"])
            headers = [
                (name, value)
                for name, value in headers
                if name.lower() != b"content-length"
            ]
            timeout = httpx.Timeout(None, connect=30.0, write=30.0, pool=30.0)
            with raw_path.open("wb") as raw:
                async with httpx.AsyncClient(
                    timeout=timeout,
                    proxy=self._proxy,
                    trust_env=False,
                    follow_redirects=False,
                ) as client:
                    async with client.stream(
                        "POST",
                        upstream_target,
                        content=forwarded_body,
                        headers=headers,
                    ) as response:
                        record["status_code"] = response.status_code
                        record["content_type"] = response.headers.get(
                            "content-type", ""
                        )
                        encoding = response.headers.get("content-encoding", "")
                        response_headers = [
                            (name, value)
                            for name, value in _forward_headers(
                                list(response.headers.raw)
                            )
                            if name.lower() != b"content-length"
                        ]
                        await send(
                            {
                                "type": "http.response.start",
                                "status": response.status_code,
                                "headers": response_headers,
                            }
                        )
                        started = True
                        async for chunk in response.aiter_raw():
                            raw.write(chunk)
                            raw.flush()
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": chunk,
                                    "more_body": True,
                                }
                            )
                        record["transport_complete"] = True
        except asyncio.CancelledError:
            record["transport_error"] = "cancelled_or_disconnected"
        except Exception as exc:
            # Exception text and HTTP headers can contain credentials.
            record["transport_error"] = type(exc).__name__
            if not started:
                await self._error(
                    send,
                    400 if isinstance(exc, ValueError) else 502,
                    b"Provider capture forwarding failed",
                )
        finally:
            self._finish_record(record, raw_path, encoding)
        if started and record["transport_complete"]:
            await send({"type": "http.response.body", "body": b"", "more_body": False})

    def _finish_record(self, record: dict[str, Any], path: Path, encoding: str) -> None:
        if not path.exists():
            path.write_bytes(b"")
        body = path.read_bytes()
        record["response_bytes"] = len(body)
        record["response_sha256"] = hashlib.sha256(body).hexdigest()
        record["content_encoding"] = encoding
        try:
            if encoding.lower() == "gzip":
                body = gzip.decompress(body)
            elif encoding.lower() == "deflate":
                body = zlib.decompress(body)
            elif encoding and encoding.lower() != "identity":
                raise ValueError("Unsupported encoding for derived index")
            parsed = parse_response_body(
                body, record["content_type"], endpoint=record["endpoint"]
            )
        except (ValueError, OSError, EOFError, zlib.error):
            parsed = {
                "parse_status": "malformed",
                "response_completed": False,
                "reasoning_items": [],
            }
        items = parsed.pop("reasoning_items")
        record.update(parsed)
        record["complete"] = bool(
            record["transport_complete"]
            and 200 <= record["status_code"] < 300
            and record["response_completed"]
            and record["parse_status"] == "parsed"
        )
        record["reasoning_items"] = len(items)
        record["nonempty_encrypted_items"] = sum(
            item["encrypted_content_state"] == "nonempty" for item in items
        )
        record["encrypted_content_chars"] = sum(
            len(item["item"]["encrypted_content"])
            for item in items
            if item["encrypted_content_state"] == "nonempty"
        )
        _write_json(path.with_suffix(".json"), record)
        with self._lock:
            with (self.output_dir / "reasoning.jsonl").open(
                "a", encoding="utf-8"
            ) as index:
                for item in items:
                    index.write(
                        json.dumps(
                            {
                                "request_id": record["request_id"],
                                "raw_response": path.name,
                                "complete": record["complete"],
                                **item,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            self._records.append(record)
