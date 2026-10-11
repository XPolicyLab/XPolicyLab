"""Minimal OpenAI-compatible chat client for the RoboDojo VLM policy.

Deliberately built on ``urllib`` so the policy server runs in the plain
``RoboDojo`` conda env with no extra install. Credentials come from a
``secrets.json`` searched upward from this file; the file is never logged.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SECRETS_FILENAME = "secrets.json"
SECRETS_ENV_VAR = "DEXBOTIC_SECRETS"
PROFILE_ENV_VAR = "VLM_AGENT_PROFILE"

# A profile is a leaf dict when it carries any of these; used to tell a single
# profile apart from a name -> profile mapping.
_PROFILE_MARKERS = ("api_key", "base_url", "model")


class VLMError(RuntimeError):
    """Raised for invalid VLM configuration or an unsuccessful VLM request."""

    def __init__(self, message, *, attempts=0, latency_s=None, status_code=None, retryable=False, retry_history=None):
        super().__init__(message)
        self.attempts = attempts
        self.latency_s = latency_s
        self.status_code = status_code
        self.retryable = retryable
        self.retry_history = list(retry_history or [])


def _candidate_secret_paths(explicit: str | None = None) -> list[Path]:
    """Every place a secrets.json may live, most specific first.

    Both the logical path (through the XPolicyLab symlink) and the real path
    are walked so the policy works whether this directory is symlinked into
    RoboDojo or copied there.
    """
    if explicit is not None:
        return [Path(explicit).expanduser()]
    env_path = os.environ.get(SECRETS_ENV_VAR)
    if env_path is not None:
        return [Path(env_path).expanduser()]

    paths: list[Path] = []
    here = Path(__file__)
    for start in (here.resolve(), here.absolute()):
        for parent in start.parents:
            paths.append(parent / SECRETS_FILENAME)

    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _normalize_profiles(data: Any) -> list[dict]:
    """Flatten the accepted secrets.json shapes into a list of profiles.

    Accepted shapes:
    ``{"vlm": {...}}``, ``{"vlm": [{...}, ...]}``,
    ``{"vlm": {"name": {...}, ...}}`` and a bare top-level list.
    """
    if isinstance(data, dict) and "vlm" in data:
        data = data["vlm"]

    if isinstance(data, dict):
        if any(marker in data for marker in _PROFILE_MARKERS):
            return [dict(data)]
        profiles = []
        for name, profile in data.items():
            if not isinstance(profile, dict):
                raise VLMError("VLM profile mapping entries must be objects")
            profile = dict(profile)
            profile.setdefault("name", name)
            profiles.append(profile)
        return profiles

    if isinstance(data, list):
        if any(not isinstance(profile, dict) for profile in data):
            raise VLMError("VLM profile list entries must be objects")
        return [dict(profile) for profile in data]

    raise VLMError("VLM configuration must be a profile object, list, or named mapping")


def load_vlm_profiles(secrets_path: str | None = None) -> list[dict]:
    """Load and validate VLM profiles in configuration order.

    An explicit path or ``DEXBOTIC_SECRETS`` is authoritative. Otherwise,
    search upward from this file. Unnamed profiles use their model ID as name.
    """
    tried = _candidate_secret_paths(secrets_path)
    for path in tried:
        if not path.is_file():
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise VLMError(
                f"invalid JSON in {path} at line {exc.lineno}, column {exc.colno}"
            ) from None
        except (OSError, UnicodeError):
            raise VLMError(f"cannot read VLM configuration from {path}") from None
        profiles = _normalize_profiles(data)
        if not profiles:
            continue
        profiles = [_validate_profile(profile, path) for profile in profiles]
        names: set[str] = set()
        for profile in profiles:
            if profile["name"] in names:
                raise VLMError(f"duplicate VLM profile name {profile['name']!r} in {path}")
            names.add(profile["name"])
        return profiles

    raise VLMError(
        f"no {SECRETS_FILENAME} with a usable VLM profile found; looked in: "
        + ", ".join(str(p) for p in tried[:6])
    )


def load_vlm_profile(name: str | None = None, secrets_path: str | None = None) -> dict:
    """Select by explicit name, then ``VLM_AGENT_PROFILE``, then first profile.

    Profile names take precedence over model IDs. A model ID can select a
    profile only when exactly one profile uses it.
    """
    profiles = load_vlm_profiles(secrets_path)
    selected = name if name is not None else os.environ.get(PROFILE_ENV_VAR)
    if selected is None:
        return profiles[0]
    for profile in profiles:
        if profile["name"] == selected:
            return profile
    matches = [profile for profile in profiles if profile["model"] == selected]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise VLMError(f"VLM model ID {selected!r} is ambiguous; select a profile name")
    available = [profile["name"] for profile in profiles]
    raise VLMError(f"VLM profile {selected!r} not found (available: {available})")


def _validate_profile(profile: dict, path: Path) -> dict:
    invalid = [
        key for key in ("base_url", "api_key", "model")
        if not isinstance(profile.get(key), str) or not profile[key].strip()
    ]
    if invalid:
        raise VLMError(f"VLM profile in {path} requires nonempty string field(s): {invalid}")
    profile.setdefault("name", profile["model"])
    if not isinstance(profile["name"], str) or not profile["name"].strip():
        raise VLMError(f"VLM profile name in {path} must be a nonempty string")
    if any(char in profile[key] for key in ("name", "model") for char in "\n\r\t"):
        raise VLMError(f"VLM profile name and model in {path} must be single-line values")
    if profile.get("extra_body") is not None and not isinstance(profile["extra_body"], dict):
        raise VLMError(f"VLM profile extra_body in {path} must be an object")
    return profile


class VLMClient:
    """Blocking chat-completions client with bounded retries.

    The policy server runs every model call in a worker thread, so blocking
    here is fine. The simulation RPC timeout must cover the caller-supplied
    ``deadline``, including retries.
    """

    def __init__(
        self,
        profile: dict,
        timeout_s: float = 60.0,
        max_attempts: int = 6,
        retry_delay_s: float = 2.0,
    ):
        self.profile = profile
        self.model = profile["model"]
        self.base_url = profile["base_url"].rstrip("/")
        self._api_key = profile["api_key"]
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay_s = float(retry_delay_s)
        self.extra_body = dict(profile.get("extra_body") or {})

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 3000,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        deadline: float | None = None,
    ) -> dict:
        """Run one chat completion.

        Args:
            messages: OpenAI-style message list (text and image_url parts).
            max_tokens: completion cap. Reasoning models spend part of this on
                hidden reasoning tokens, so keep headroom above the JSON size.
            temperature: sampling temperature, omitted from the request when
                ``None`` (some gateways reject it for reasoning models).
            reasoning_effort: forwarded verbatim when set. A low setting may
                still consume reasoning tokens, depending on the endpoint.
            deadline: absolute ``time.monotonic()`` value after which no new
                attempt is started.

        Returns:
            ``{"text", "usage", "finish_reason", "latency_s", "attempts"}``.

        Raises:
            VLMError: every attempt failed, or the endpoint returned no text.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if reasoning_effort:
            payload["reasoning_effort"] = str(reasoning_effort)
        payload.update(self.extra_body)
        payload["model"] = self.model
        # An explicit run setting wins over profile defaults such as Qwen's
        # enable_thinking=false / reasoning_effort=none.
        if reasoning_effort is not None:
            payload["reasoning_effort"] = str(reasoning_effort)
            if "enable_thinking" in self.extra_body:
                payload["enable_thinking"] = reasoning_effort != "none"

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        started = time.monotonic()
        last_error = "decision deadline expired before request"
        attempts = 0
        status_code = None
        retryable = False
        retry_history = []
        for attempt in range(1, self.max_attempts + 1):
            if deadline is not None and time.monotonic() >= deadline:
                break
            timeout = self.timeout_s
            if deadline is not None:
                timeout = min(timeout, deadline - time.monotonic())
                if timeout <= 0:
                    break
            try:
                attempt_started = time.monotonic()
                request = urllib.request.Request(self.url, data=body, headers=headers)
                attempts += 1
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                text = _extract_text(data)
                if not text:
                    raise VLMError(f"empty completion (finish_reason={_finish_reason(data)})", retryable=True)
                return {
                    "text": text,
                    "usage": data.get("usage", {}),
                    "finish_reason": _finish_reason(data),
                    "latency_s": time.monotonic() - started,
                    "attempts": attempts,
                    "retry_history": retry_history,
                }
            except Exception as exc:  # noqa: BLE001 - retried and reported below
                last_error = _describe(exc, self._api_key)
                status_code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                # 400 is retried too: gateways in front of the endpoints return it for
                # transient upstream failures
                retryable = (status_code in (400, 408, 409, 425, 429) or status_code >= 500
                             if status_code else isinstance(exc, (OSError, urllib.error.URLError,
                                                                  http.client.HTTPException, json.JSONDecodeError,
                                                                  UnicodeError))
                             or (isinstance(exc, VLMError) and exc.retryable))
                failure = {"attempt": attempt, "error": last_error, "status_code": status_code,
                           "latency_s": round(time.monotonic() - attempt_started, 3)}
                retry_history.append(failure)
                retry_after = None
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        retry_after = max(0.0, float(exc.headers.get("Retry-After", "")))
                    except (ValueError, TypeError, AttributeError):
                        pass
                    exc.close()
                if not retryable or attempt >= self.max_attempts:
                    break
                # exponential backoff: a rate-limited endpoint (429) needs the gap to grow
                backoff = min(self.retry_delay_s * (2 ** (attempt - 1)), 30.0)
                delay = max(backoff, retry_after or 0.0) + random.uniform(0, min(1.0, backoff))
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= delay:
                    break
                failure["retry_delay_s"] = round(delay, 3)
                print(f"[vlm_agent] request attempt {attempt}/{self.max_attempts}: {last_error}; "
                      f"retrying in {delay:.1f}s", flush=True)
                time.sleep(delay)

        raise VLMError(
            f"VLM request failed after {attempts} attempt(s): {last_error}",
            attempts=attempts, latency_s=time.monotonic() - started,
            status_code=status_code, retryable=retryable,
            retry_history=retry_history,
        )


def _extract_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # Some gateways return content parts instead of a plain string.
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return (content or "").strip()


def _finish_reason(data: dict) -> str:
    choices = data.get("choices") or []
    return str(choices[0].get("finish_reason")) if choices else "unknown"


# Client errors whose body explains what the request got wrong; auth and rate-limit replies are
# never echoed, they can carry account details.
_ECHOED_STATUSES = (400, 413, 422)


def _describe(exc: Exception, secret: str | None = None) -> str:
    """Readable one-liner for an HTTP failure, without leaking the API key."""
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        if exc.code in _ECHOED_STATUSES:
            # a 400 body says what the proxy objected to (an image, a field, a filter); without it a
            # dead episode cannot be diagnosed afterwards
            try:
                raw_body = exc.read(600)
            except Exception:  # noqa: BLE001 - the status alone is still worth reporting
                raw_body = b""
            if secret:
                key = secret.encode("utf-8")
                hit_read_limit = len(raw_body) == 600
                raw_body = raw_body.replace(key, b"<redacted>")
                if hit_read_limit:
                    # The bounded read may end inside the key, before a full match is available.
                    for length in range(min(len(key) - 1, len(raw_body)), 0, -1):
                        if raw_body.endswith(key[:length]):
                            raw_body = raw_body[:-length] + b"<redacted>"
                            break
            body = " ".join(raw_body.decode("utf-8", "replace").split())[:300]
        return f"HTTP {exc.code}" + (f": {body}" if body else "")
    if isinstance(exc, TimeoutError) or (isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)):
        return "request timeout"
    if isinstance(exc, urllib.error.URLError):
        return f"connection failed ({type(exc.reason).__name__})"
    if isinstance(exc, VLMError):
        return str(exc)
    return type(exc).__name__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List or select configured VLM profiles")
    parser.add_argument("--list-models", action="store_true", help="show profile names and model IDs")
    parser.add_argument("--model", "--profile", dest="profile", help="profile name or unique model ID")
    parser.add_argument("--secrets", help="explicit secrets.json path")
    args = parser.parse_args(argv)
    try:
        if args.list_models:
            profiles = load_vlm_profiles(args.secrets)
            width = max(len("NAME"), *(len(profile["name"]) for profile in profiles))
            print(f"{'NAME':<{width}}  MODEL")
            for profile in profiles:
                print(f"{profile['name']:<{width}}  {profile['model']}")
        else:
            print(load_vlm_profile(args.profile, args.secrets)["name"])
    except VLMError as exc:
        print(f"VLM configuration error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
