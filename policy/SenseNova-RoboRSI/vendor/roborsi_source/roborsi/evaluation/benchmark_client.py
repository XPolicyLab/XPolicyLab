# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed Benchmark HTTP client; downloads replace shared output directories."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from http.client import HTTPException
from pathlib import Path
from typing import Any

from roborsi.evaluation.http_artifacts import save_artifact
from roborsi.evaluation.media import (
    MAX_MEDIA_BYTES,
    MAX_OBSERVATION_MEDIA_BYTES,
    MAX_OBSERVATION_MEDIA_FILES,
    CaseMediaChannel,
)
from roborsi.evaluation.messages import (
    ArtifactInfo,
    NativeResult,
    Observation,
    ServiceError,
    Task,
)
from roborsi.evaluation.services import AdapterClient
from roborsi.utils.rpc import RpcClient, RpcError, RpcTransportError
from roborsi.utils.rpc.http_abort import HttpCallGuard

MAX_JSON_BYTES = 8 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class BenchmarkTransport(RpcClient):
    """Map the typed Adapter operations to the public REST endpoints."""

    def __init__(self, endpoint: str, *, token: str | None = None) -> None:
        super().__init__()
        parsed = urllib.parse.urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port == 0
            or any(c.isspace() for c in parsed.netloc)
        ):
            raise ValueError("Benchmark endpoint must be an HTTP(S) origin URL")
        if token is not None and (not token or any(c in token for c in "\r\n")):
            raise ValueError("invalid Benchmark token")
        self.endpoint, self.token = endpoint.rstrip("/"), token
        self.episode_id: str | None = None
        self.direct = parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict | None = None,
        timeout_s: float = 30.0,
        artifact: ArtifactInfo | None = None,
    ) -> tuple[int, Any]:
        headers = {
            "Accept": "application/json" if artifact is None else artifact.mime_type
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.endpoint + path, data=encoded, headers=headers, method=method
        )
        limit = MAX_JSON_BYTES if artifact is None else artifact.size_bytes
        with HttpCallGuard() as guard:
            opener = guard.opener(
                direct=self.direct, connect_timeout_s=30.0 if timeout_s == 0 else None
            )
            opener.add_handler(_NoRedirect())
            try:
                with opener.open(
                    request, timeout=None if timeout_s == 0 else timeout_s
                ) as response:
                    status = response.status
                    length = response.headers.get("Content-Length")
                    if length is not None and (
                        not length.isdecimal() or int(length) > limit
                    ):
                        raise ValueError("Benchmark response exceeds size limit")
                    raw = response.read(limit + 1)
                    if len(raw) > limit:
                        raise ValueError("Benchmark response exceeds size limit")
                    if length is not None and len(raw) != int(length):
                        raise RpcTransportError(
                            path, "Benchmark HTTP response was interrupted"
                        )
                    if artifact is not None:
                        if (
                            response.headers.get("Content-Type") != artifact.mime_type
                            or response.headers.get("X-Content-SHA256")
                            != artifact.sha256
                        ):
                            raise ValueError(
                                "artifact download headers differ from metadata"
                            )
                        return status, raw
            except urllib.error.HTTPError as exc:
                with exc:
                    raw = exc.read(MAX_JSON_BYTES + 1)
                try:
                    error = ServiceError.from_dict(json.loads(raw)["error"])
                except (ValueError, TypeError, KeyError) as parse_error:
                    raise RpcError(
                        path,
                        f"Benchmark HTTP status {exc.code}",
                        code="infrastructure_error",
                    ) from parse_error
                raise RpcError(path, error.message, code=error.code) from exc
            except (OSError, HTTPException) as exc:
                raise RpcTransportError(
                    path, "Benchmark HTTP connection failed"
                ) from exc
        try:
            return status, json.loads(
                raw,
                parse_constant=lambda _: (_ for _ in ()).throw(
                    ValueError("nonfinite JSON")
                ),
            )
        except (ValueError, UnicodeDecodeError) as exc:
            raise RpcTransportError(path, "invalid Benchmark JSON response") from exc

    def episode_path(self, episode_id: str | None = None) -> str:
        identifier = episode_id or self.episode_id
        if identifier is None:
            raise ValueError("Benchmark episode has not started")
        return "/v1/episodes/" + urllib.parse.quote(identifier, safe="")

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        timeout_s = 30.0 if timeout_s is None else timeout_s
        if method == "healthz":
            _, reply = self.request("GET", "/v1/health", timeout_s=timeout_s)
            if reply.get("benchmark_api_versions") != [1]:
                raise RpcError(
                    method,
                    "unsupported Benchmark HTTP API version",
                    code="unsupported_version",
                )
            return {
                **reply,
                "roborsi_protocol_versions": [1],
                "runtime_info_versions": [1],
            }
        if method == "runtime_info":
            _, reply = self.request("GET", "/v1/runtime", timeout_s=timeout_s)
            return json.dumps(reply["runtime"], allow_nan=False)
        if method == "start":
            body = json.loads(args[0])
            self.episode_id = body["episode_id"]
            _, reply = self.request(
                "POST", "/v1/episodes", body=body, timeout_s=timeout_s
            )
            return json.dumps(reply)
        if method in {"step", "stop", "query", "finish"}:
            body = (
                {"request_id": args[0]} if method == "finish" else json.loads(args[0])
            )
            path = (
                self.episode_path()
                + {
                    "step": "/actions",
                    "stop": "/stop",
                    "query": "/query",
                    "finish": "/finish",
                }[method]
            )
            _, reply = self.request("POST", path, body=body, timeout_s=timeout_s)
            return json.dumps(reply)
        if method == "close_episode":
            if self.episode_id is None:
                return {"ok": True}
            end = None if timeout_s == 0 else time.monotonic() + timeout_s

            def remaining() -> float:
                return 0 if end is None else max(0.001, end - time.monotonic())

            _, state = self.request("GET", self.episode_path(), timeout_s=timeout_s)
            if state["state"] not in {"completed", "failed", "cancelled", "closed"}:
                self.request(
                    "POST",
                    self.episode_path() + "/cancel",
                    body={},
                    timeout_s=remaining(),
                )
                while state["state"] not in {
                    "completed",
                    "failed",
                    "cancelled",
                    "closed",
                }:
                    if end is not None and time.monotonic() >= end:
                        raise TimeoutError("Benchmark cancellation is still pending")
                    time.sleep(
                        0.05
                        if end is None
                        else max(0, min(0.05, end - time.monotonic()))
                    )
                    _, state = self.request(
                        "GET",
                        self.episode_path(),
                        timeout_s=remaining(),
                    )
            if (state.get("error") or {}).get("details", {}).get(
                "operation"
            ) == "close":
                raise RpcError(
                    "close_episode",
                    "Benchmark cleanup failed",
                    code="infrastructure_error",
                )
            self.request(
                "POST",
                self.episode_path() + "/close",
                body={},
                timeout_s=remaining(),
            )
            return {"ok": True}
        if method == "shutdown":
            # HTTP Benchmark processes belong to their deployer; Docker removes its container.
            return {"ok": True}
        raise ValueError(f"unknown Benchmark operation: {method}")


class BenchmarkClient(AdapterClient):
    """Run tests and download outputs using only the Benchmark HTTP API."""

    def __init__(
        self, endpoint: str, *, token: str | None = None, close_timeout_s: float = 5.0
    ) -> None:
        self.transport = BenchmarkTransport(endpoint, token=token)
        super().__init__(self.transport, close_timeout_s=close_timeout_s)
        self.downloads_complete = False
        self._downloads: dict[str, ArtifactInfo] = {}

    def tasks(self, *, timeout_s: float = 30.0) -> list[Task]:
        _, reply = self.transport.request("GET", "/v1/tasks", timeout_s=timeout_s)
        return [Task.from_dict(t) for t in reply["tasks"]]

    def artifacts(
        self,
        *,
        episode_id: str | None = None,
        step_index: int | None = None,
        kind: str | None = None,
        timeout_s: float = 30.0,
    ) -> list[ArtifactInfo]:
        query = {}
        if step_index is not None:
            query["step_index"] = step_index
        if kind is not None:
            query["kind"] = kind
        path = self.transport.episode_path(episode_id) + "/artifacts"
        if query:
            path += "?" + urllib.parse.urlencode(query)
        _, reply = self.transport.request("GET", path, timeout_s=timeout_s)
        infos = [ArtifactInfo.from_dict(a) for a in reply["artifacts"]]
        if any(
            a.episode_id != (episode_id or self.transport.episode_id) for a in infos
        ):
            raise ValueError("artifact belongs to another episode")
        return infos

    def download(
        self, info: ArtifactInfo, directory: Path, *, timeout_s: float = 30.0
    ) -> None:
        path = (
            self.transport.episode_path(info.episode_id)
            + "/artifacts/"
            + info.artifact_id
        )
        _, data = self.transport.request(
            "GET", path, timeout_s=timeout_s, artifact=info
        )
        save_artifact(directory, info, data)
        self._downloads[info.ref] = info

    @property
    def artifact_records(self) -> tuple[ArtifactInfo, ...]:
        return tuple(
            info
            for info in self._downloads.values()
            if info.episode_id == self.transport.episode_id
        )

    def collect_observation(
        self, observation: Observation, channel: CaseMediaChannel, *, timeout_s: float
    ) -> None:
        if (
            len(observation.media) > MAX_OBSERVATION_MEDIA_FILES
            or sum(m.size_bytes for m in observation.media)
            > MAX_OBSERVATION_MEDIA_BYTES
            or any(m.size_bytes > MAX_MEDIA_BYTES for m in observation.media)
        ):
            raise ValueError("observation media exceeds size limits")
        end = time.monotonic() + timeout_s
        infos = self.artifacts(step_index=observation.step_index, timeout_s=timeout_s)
        files = {item.ref: item for item in infos}
        for ref in observation.media:
            info = files.get(ref.ref)
            if (
                info is None
                or info.kind != "image"
                or info.step_index != observation.step_index
                or info.episode_id != observation.episode_id
                or (info.size_bytes, info.sha256, info.mime_type)
                != (ref.size_bytes, ref.sha256, ref.mime_type)
            ):
                raise ValueError("image reference differs from HTTP artifact metadata")
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("observation download timed out")
            self.download(info, channel.media_dir.parent, timeout_s=remaining)
        channel.verify_observation(observation)

    def collect_native(
        self, native: NativeResult, directory: Path, *, timeout_s: float
    ) -> None:
        if native.source is None:
            return
        end = time.monotonic() + timeout_s
        infos = self.artifacts(kind="native_result", timeout_s=timeout_s)
        matches = [
            a
            for a in infos
            if a.ref == native.source
            and a.sha256 == native.source_sha256
            and a.episode_id == native.episode_id
        ]
        if len(matches) != 1:
            raise ValueError("native result is missing its HTTP artifact")
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("native result download timed out")
        self.download(matches[0], directory, timeout_s=remaining)

    def collect_artifacts(self, directory: Path, *, timeout_s: float) -> None:
        end = time.monotonic() + timeout_s
        for info in self.artifacts(timeout_s=timeout_s):
            if self._downloads.get(info.ref) == info:
                continue
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Benchmark artifact download timed out")
            self.download(info, directory, timeout_s=remaining)
        self.downloads_complete = True

    def collect_failure_artifacts(self, directory: Path, *, timeout_s: float) -> None:
        """Flush the failed remote episode and archive its remaining evidence.

        Use the existing cleanup route without requesting native finish or a
        score. Keep this transport open until verified downloads complete.
        """
        if self.transport.episode_id is None:
            return
        self.transport.call("close_episode", timeout_s=self._close_timeout_s)
        self.collect_artifacts(directory, timeout_s=timeout_s)

    def submit_test(
        self,
        episode_id: str,
        task: Task,
        actions: list[dict],
        *,
        request_id: str,
        repeat_last: bool = False,
        timeout_s: float = 30.0,
    ) -> dict:
        _, reply = self.transport.request(
            "POST",
            "/v1/tests",
            body={
                "episode_id": episode_id,
                "task": task.as_dict(),
                "request_id": request_id,
                "actions": actions,
                "repeat_last": repeat_last,
            },
            timeout_s=timeout_s,
        )
        return reply

    def status(self, episode_id: str, *, timeout_s: float = 30.0) -> dict:
        return self.transport.request(
            "GET", self.transport.episode_path(episode_id), timeout_s=timeout_s
        )[1]

    def result(self, episode_id: str, *, timeout_s: float = 30.0) -> tuple[int, dict]:
        return self.transport.request(
            "GET",
            self.transport.episode_path(episode_id) + "/result",
            timeout_s=timeout_s,
        )

    def cancel(self, episode_id: str, *, timeout_s: float = 30.0) -> dict:
        return self.transport.request(
            "POST",
            self.transport.episode_path(episode_id) + "/cancel",
            body={},
            timeout_s=timeout_s,
        )[1]

    def delete(self, episode_id: str, *, timeout_s: float = 30.0) -> None:
        self.transport.request(
            "DELETE", self.transport.episode_path(episode_id), timeout_s=timeout_s
        )
