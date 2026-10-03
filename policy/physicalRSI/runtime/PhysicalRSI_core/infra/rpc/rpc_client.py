#
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

"""RPC client protocol: error type, transport-agnostic client base, response
envelope validation.

Server-side counterparts live in :mod:`PhysicalRSI_core.infra.rpc.rpc_facade` (the
``ServiceHost`` base and the ``make_error_response`` envelope builder).
"""

from __future__ import annotations

import atexit
import uuid
from typing import Any


class RpcError(RuntimeError):
    """Raised when a remote method call returns an error."""

    def __init__(self, method: str, message: str, *, traceback: str | None = None):
        super().__init__(f"{method}: {message}")
        self.method = method
        self.server_traceback = traceback


class RpcClient:
    """Base for transport-specific RPC clients.

    Owns the session id (transport-private; business code never sees it)
    and the atexit close hook. Subclasses implement :meth:`call` for the
    transport-specific request path.
    """

    def __init__(self, *, enable_sessions: bool = False) -> None:
        self._session_id: str | None = (
            f"rpc_{uuid.uuid4().hex}" if enable_sessions else None
        )
        self._closed = False
        if self._session_id is not None:
            atexit.register(self._close_at_exit)

    def close(self) -> None:
        """Notify the server to drop this client's session.

        Called automatically at exit via atexit. Failures are swallowed
        (the process is exiting anyway; the server's sweep thread is the
        fallback for crashed clients). Idempotent: a ``_closed`` flag guards
        against double-close when atexit fires after a manual ``close()``.
        """
        if self._closed:
            return
        if self._session_id is not None:
            self.call("session.close", timeout_s=1.0)
        self._closed = True

    def _close_at_exit(self):
        try:
            self.close()
        except Exception:
            pass

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Invoke a remote method and return its result. Override in subclasses."""
        raise NotImplementedError

    def invoke(self, request_id, method, *, timeout_s=None, **kwargs):
        """Execute with a durable ID; after timeout query status before retry."""
        try:
            return self.call(
                "request.execute",
                kwargs={"request_id": request_id, "method": method, "kwargs": kwargs},
                timeout_s=timeout_s,
            )
        except Exception as error:
            error.request_id = request_id
            raise

    def status(self, request_id, *, timeout_s=5):
        return self.call(
            "request.status", kwargs={"request_id": request_id}, timeout_s=timeout_s
        )


def check_response(response: Any, method: str) -> Any:
    """Validate RPC response envelope; raise ``RpcError`` on failure, return result."""
    if not isinstance(response, dict):
        raise RpcError(method, f"bad response type: {type(response).__name__}")
    if not response.get("ok"):
        raise RpcError(
            method,
            str(response.get("error", "<no error message>")),
            traceback=response.get("traceback"),
        )
    return response.get("result")


__all__ = ["RpcClient", "RpcError", "check_response"]
