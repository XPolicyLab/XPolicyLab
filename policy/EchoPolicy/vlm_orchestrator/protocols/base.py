# SPDX-License-Identifier: Apache-2.0

"""Abstract Frontend / Backend interfaces for protocol modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable


class FrontendSession(ABC):
    """One logical session with an eval client.

    For long-lived connections (WebSocket): one session per connection.
    Request/response adapters may choose a longer-lived session if their
    transport has no connection boundary.
    """

    @abstractmethod
    async def send_metadata(self, metadata: dict) -> None:
        """Send the initial metadata frame, if the protocol expects one.

        OpenPI expects a metadata frame before the first observation;
        adapters without one may no-op.
        """

    @abstractmethod
    async def recv_obs(self) -> dict | None:
        """Receive the next observation in the canonical schema.

        Returns ``None`` when the session has ended (client disconnect,
        EOF, etc.).  Proxy-specific keys like ``__step`` and
        ``__episode_id`` should be passed through unchanged in the
        canonical dict so the orchestrator can extract them.
        """

    async def recv_batch(self) -> list[dict] | None:
        """Receive one or more canonical observations.

        Frontends without a native batch frame retain single-observation
        behavior through this default implementation.
        """
        observation = await self.recv_obs()
        return None if observation is None else [observation]

    @abstractmethod
    async def send_action(self, canonical_action: dict) -> int | None:
        """Send a canonical action back, translating to native protocol.

        ``canonical_action`` is a dict containing at least ``"actions"``
        and may contain orchestrator metadata. Return the number of action
        timesteps actually emitted when the protocol can report it.
        """

    async def send_action_batch(self, canonical_actions: list[dict]) -> list[int | None]:
        """Send one action response per observation in the last batch."""
        return [await self.send_action(action) for action in canonical_actions]


class Frontend(ABC):
    """Listens for eval-client connections in a specific protocol."""

    @abstractmethod
    async def serve(
        self,
        host: str,
        port: int,
        on_session: Callable[[FrontendSession], Awaitable[None]],
    ) -> None:
        """Bind to ``host:port`` and accept connections.

        For each new session, call ``on_session(session)`` and wait for
        it to finish.  Concurrent sessions are protocol-specific:
        WebSocket allows many; other transports may serialize sessions.
        """


class BackendConnection(ABC):
    """One open connection to a VLA server."""

    @abstractmethod
    async def recv_metadata(self) -> dict:
        """Read the VLA server's metadata, or synthesize a minimal one
        for protocols without a metadata frame.
        """

    @abstractmethod
    async def infer(self, canonical_obs: dict) -> dict:
        """Send a canonical observation, return a canonical action dict.

        The implementation is responsible for translating to/from the
        VLA's native schema.
        """

    async def infer_batch(self, canonical_obs: list[dict]) -> list[dict]:
        """Infer a list of observations, preserving input order.

        Backends without a native batch API retain compatibility through the
        existing single-observation method.
        """
        return [await self.infer(observation) for observation in canonical_obs]

    @abstractmethod
    async def close(self) -> None: ...


class Backend(ABC):
    """Connects to a VLA server in a specific protocol."""

    @abstractmethod
    async def connect(self) -> BackendConnection:
        """Open a fresh connection to the VLA server.

        Called once per eval-client session.  The orchestrator uses one
        backend connection per frontend session.
        """
