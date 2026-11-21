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

"""Connect a case to an externally owned RoboRSI HTTP Agent service."""

from __future__ import annotations

import math
from pathlib import Path
from urllib.parse import urlsplit

from roborsi.evaluation.deadline import CaseDeadline
from roborsi.evaluation.services import AgentClient
from roborsi.utils.rpc import RpcClient, wait_for_ready
from roborsi.utils.rpc.http_rpc import HttpRpcClient


class ExternalAgentSession:
    """Own the connection, while the external service keeps its own process."""

    def __init__(self, rpc: RpcClient) -> None:
        self.client = AgentClient(rpc)
        self._rpc = rpc

    def __enter__(self) -> ExternalAgentSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self._rpc.close()


class HttpAgentLauncher:
    """Use a running HTTP(S) Agent that implements RoboRSI's episode RPC contract."""

    def __init__(self, endpoint: str, *, startup_timeout_s: float = 30.0) -> None:
        if not isinstance(endpoint, str):
            raise ValueError("Agent endpoint must be an HTTP(S) URL")
        try:
            parsed = urlsplit(endpoint)
            parsed.port
        except ValueError as exc:
            raise ValueError("Agent endpoint must be an HTTP(S) origin URL") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.port == 0
            or any(character.isspace() for character in parsed.netloc)
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("Agent endpoint must be an HTTP(S) origin URL")
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        self.endpoint = endpoint.rstrip("/")
        self.startup_timeout_s = startup_timeout_s

    def start(
        self,
        case_dir: str | Path,
        *,
        deadline: CaseDeadline | None = None,
        media_dir: str | Path | None = None,
    ) -> ExternalAgentSession:
        """Check health; the remote service must already know its media mount."""
        if deadline is not None:
            deadline.check()
        rpc = HttpRpcClient(self.endpoint)
        try:
            wait_for_ready(
                rpc,
                timeout_s=self.startup_timeout_s,
                poll_interval_s=0.1,
                remaining_s=deadline.remaining if deadline is not None else None,
            )
            if deadline is not None:
                deadline.check()
        except Exception:
            rpc.close()
            raise
        return ExternalAgentSession(rpc)
