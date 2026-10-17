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

"""Client utility functions: endpoint parsing, server discovery, and readiness polling."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from roborsi.utils.logging import get_logger

if TYPE_CHECKING:
    from roborsi.utils.daemon import ProcessDaemon
    from roborsi.utils.rpc.rpc_client import RpcClient

logger = get_logger("client_utils")


def parse_endpoint(endpoint: str) -> tuple[str, str, int]:
    """Parse ``[protocol://]host:port`` into ``(protocol, host, port)``.

    Protocol defaults to ``http`` when the prefix is omitted.
    """
    if "://" in endpoint:
        protocol, _, rest = endpoint.partition("://")
    else:
        protocol, rest = "http", endpoint
    host, _, port = rest.partition(":")
    if not host or not port:
        raise ValueError(f"endpoint must be [protocol://]host:port, got {endpoint!r}")
    return protocol, host, int(port)


def make_rpc_client(endpoint: str, *, enable_sessions: bool = False) -> "RpcClient":
    """Build an HTTP or socket client for ``endpoint``."""
    from roborsi.utils.rpc.http_rpc import HttpRpcClient
    from roborsi.utils.rpc.socket_rpc import SocketRpcClient

    protocol, host, port = parse_endpoint(endpoint)
    if protocol == "http":
        return HttpRpcClient(f"http://{host}:{port}", enable_sessions=enable_sessions)
    if protocol == "socket":
        return SocketRpcClient(host, port, enable_sessions=enable_sessions)
    raise ValueError(f"endpoint protocol must be http or socket, got {protocol!r}")


def wait_for_ready(
    client: "RpcClient",
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.5,
    daemon: "ProcessDaemon | None" = None,
    remaining_s: Callable[[], float] | None = None,
) -> None:
    """Poll ``client.call("healthz")`` until it succeeds or ``timeout_s`` elapses.

    If ``daemon`` is given and its subprocess exits before becoming ready,
    fail fast with ``RuntimeError`` (carrying the exit code) instead of
    blocking until ``timeout_s``.

    Once the server is reachable, if the client is session-aware (it owns a
    private ``_session_id``), this function registers the session with the
    server once. The sid stays private to the client; business code never
    sees it.
    """
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        available = deadline - time.monotonic()
        if remaining_s is not None:
            available = min(available, remaining_s())
        if available <= 0:
            break
        if daemon is not None:
            rc = daemon.poll()
            if rc is not None:
                detail = last_err if last_err is not None else "no healthz attempt yet"
                raise RuntimeError(
                    f"{daemon.name} exited with code {rc} before becoming "
                    f"ready; check its log. last healthz error: {detail}"
                )
        try:
            client.call("healthz", timeout_s=min(1.0, available))
        except Exception as exc:
            last_err = exc
            time.sleep(min(poll_interval_s, max(0.0, deadline - time.monotonic())))
            continue
        sid = getattr(client, "_session_id", None)
        if sid is not None:
            available = deadline - time.monotonic()
            if remaining_s is not None:
                available = min(available, remaining_s())
            if available <= 0:
                break
            try:
                client.call("session.register", timeout_s=min(30.0, available))
            except Exception as exc:
                if remaining_s is not None:
                    remaining_s()
                raise RuntimeError(
                    f"server is reachable but session.register failed for "
                    f"sid={sid!r}: {exc}. The server process is now in a "
                    f"half-ready state; check its log."
                ) from exc
        if remaining_s is not None:
            remaining_s()
        if time.monotonic() >= deadline:
            break
        return
    if remaining_s is not None:
        remaining_s()
    raise TimeoutError(
        f"server did not become ready within {timeout_s:.0f}s: {last_err}"
    )
