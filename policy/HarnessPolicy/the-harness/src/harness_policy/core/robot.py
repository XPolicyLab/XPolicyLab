"""JSON/HTTP transport to a worker in a separate Python environment.

This is the framework's protocol, NOT the native RoboDojo/XPolicyLab protocol.
An integration must implement Environment on the worker side.
"""

import json
import os
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .contracts import Action, Observation, Transition


class RemoteEnvironment:
    """A worker reached over HTTP (POST /rpc), with the bearer token from
    $HARNESS_WORKER_TOKEN when it is set. A worker error is raised as a RuntimeError, and
    no request is retried: a motion that timed out may have executed."""

    name = "remote"

    def __init__(self, endpoint: str, timeout: float = 120.0):
        url = urlparse(endpoint)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
            raise ValueError("Worker endpoint must be an HTTP(S) URL without embedded credentials")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.token = os.environ.get("HARNESS_WORKER_TOKEN")

    def _call(self, method: str, **arguments):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(self.endpoint + "/rpc", data=json.dumps({
            "method": method, "arguments": arguments,
        }, allow_nan=False).encode(), headers=headers)
        # Deliberately no automatic retry: a timed-out motion may have executed.
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"Worker HTTP {error.code}: {detail}") from error
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    def tools(self) -> list[dict]:
        """The worker's tool schemas. Checks its protocol version and takes its name; it
        also serves as the check that a worker answers."""
        description = self._call("describe")
        if description["protocol_version"] != 1:
            raise ValueError("Unsupported worker protocol version")
        self.name = description["name"]
        return description["tools"]

    def reset(self, task: str, seed: int) -> Observation:
        """Start an episode of `task` with `seed` and return its first observation."""
        return Observation(**self._call("reset", task=task, seed=seed))

    def execute(self, action: Action, episode_id: str, step: int) -> Transition:
        """Execute `action` on the observation (`episode_id`, `step`); the worker rejects a
        stale pair. Returns the transition to the next observation."""
        return Transition.from_dict(self._call(
            "execute", action=action.to_dict(), episode_id=episode_id, step=step,
        ))

    def call(self, name: str, **arguments):
        """Invoke an environment-specific worker method."""
        return self._call("call", name=name, arguments=arguments)

    def close(self) -> None:
        """Nothing: the worker belongs to the process that launched it."""
        # Worker lifecycle is explicitly owned by its launching process.
        pass
