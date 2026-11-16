"""Credentials for authenticated APIs or an explicitly local bridge."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time


class CredentialError(RuntimeError):
    def __init__(self, message, *, retryable=True):
        super().__init__(message)
        self.retryable = retryable


def validate_auth(settings):
    if not isinstance(settings, dict):
        raise ValueError("supervisor.auth must be an object")
    kind = settings.get("type", "env")
    if kind == "copilot":
        if settings.keys() - {"type", "token_file", "timeout_s"}:
            raise ValueError("Invalid supervisor.auth settings")
        path = settings.get("token_file")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("supervisor.auth.token_file is required for Copilot")
        timeout = settings.get("timeout_s", 30)
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("supervisor.auth.timeout_s must be positive and finite")
        return {"type": kind, "token_file": path, "timeout_s": timeout}
    allowed = (
        {"type"}
        if kind in {"env", "none"}
        else {"type", "command", "refresh_interval_s", "timeout_s"}
    )
    if (
        not isinstance(kind, str)
        or kind not in {"env", "command", "none"}
        or settings.keys() - allowed
    ):
        raise ValueError("Invalid supervisor.auth settings")
    if kind in {"env", "none"}:
        return {"type": kind}
    command = settings.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(arg, str) or not arg.strip() or "\0" in arg
            for arg in command
        )
    ):
        raise ValueError("supervisor.auth.command must be a nonempty argv list")
    result = {"type": kind, "command": list(command)}
    for key, default in (("refresh_interval_s", 300), ("timeout_s", 30)):
        value = settings.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"supervisor.auth.{key} must be positive and finite")
        result[key] = value
    return result


def _token(value):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 16384
        or any(ord(char) < 33 or ord(char) > 126 for char in value)
    ):
        raise CredentialError("Invalid supervisor credential", retryable=False)
    return value


class StaticCredential:
    refreshable = False

    def __init__(self, token):
        self._token = _token(token)

    def get(self, deadline):
        return self._token


class LocalBridgeCredential:
    """The loopback bridge owns upstream authentication; send no dummy token."""

    refreshable = False
    loopback_only = True

    def get(self, deadline):
        return None


class CommandCredential:
    refreshable = True

    def __init__(self, settings):
        self.settings = validate_auth(settings)
        if self.settings["type"] != "command":
            raise ValueError("A command credential requires command auth")
        self._lock = threading.Lock()
        self._token = None
        self._expires = 0.0

    def invalidate(self, rejected_token, *, deadline=None):
        remaining = (
            self.settings["timeout_s"]
            if deadline is None
            else max(0, deadline - time.monotonic())
        )
        if not self._lock.acquire(timeout=remaining):
            raise CredentialError("Supervisor credential wait timed out")
        try:
            if self._token == rejected_token:
                self._token = None
        finally:
            self._lock.release()

    def get(self, deadline):
        if not self._lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise CredentialError("Supervisor credential wait timed out")
        try:
            if self._token is not None and time.monotonic() < self._expires:
                return self._token
            timeout = min(self.settings["timeout_s"], deadline - time.monotonic())
            if timeout <= 0:
                raise CredentialError("Supervisor credential deadline exceeded")
            try:
                process = subprocess.Popen(
                    self.settings["command"],
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError:
                raise CredentialError(
                    "Cannot start supervisor credential command", retryable=False
                ) from None
            try:
                stdout, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                raise CredentialError(
                    "Supervisor credential command timed out"
                ) from None
            if process.returncode:
                raise CredentialError("Supervisor credential command failed")
            try:
                token = stdout.decode("ascii").strip()
            except UnicodeError:
                raise CredentialError(
                    "Invalid supervisor credential", retryable=False
                ) from None
            self._token = _token(token)
            self._expires = time.monotonic() + self.settings["refresh_interval_s"]
            return self._token
        finally:
            self._lock.release()
