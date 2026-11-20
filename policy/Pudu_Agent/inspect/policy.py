"""Standalone RoboDojo L4 inspect-inspired joint agent policy."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from XPolicyLab.utils.openai_responses import (
    ReasoningReplayStore,
    chat_messages_to_responses_input,
    chat_tools_to_responses_tools,
    responses_to_chat_completion,
    session_headers,
)

from .prior_attempt import load_prior_messages
from .trace import source_revision
from .types import JOINT_CHANNELS, Action, ActionChunk, ActionSpace, Observation

_PLANNER_WORKER_PATH = Path(__file__).with_name("planner_http_worker.py")


class CapabilityFailure(Exception):
    """Model-side failure: repair exhaustion, no tool call, give_up, or budget."""


class InfrastructureFailure(BaseException):
    """Provider-side failure: missing key, auth, misconfiguration, exhausted retries.

    Derived from ``BaseException`` on purpose. RoboDojo's layout loop in
    ``src/eval_client/main.py`` wraps each episode in ``except Exception`` and
    advances to the next layout, which would silently fold provider outages into
    the success rate. Every adapter site that must observe one catches it by
    name, so the only handler this bypasses is a blanket ``except Exception``.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after_s: float | None = None,
        status: int | None = None,
        key_unusable: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.status = status
        # Set when the provider refused the credential rather than the request:
        # the key is not one this account can serve the model with, now or in an
        # hour. Waiting cannot fix it; another key can.
        self.key_unusable = key_unusable


@dataclass(frozen=True)
class RoboDojoActionSpec:
    """The semantics RoboDojo exposes to the joint agent policy."""

    labels: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray
    control_hz: float
    docs: str
    max_step: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        width = len(self.labels)
        channel_width = ActionSpace(JOINT_CHANNELS).width
        if width != channel_width:
            raise ValueError(
                f"action spec has {width} labels but the joint channels carry "
                f"{channel_width}; labels name channels by position"
            )
        if self.low.shape != (width,) or self.high.shape != (width,):
            raise ValueError("action bounds must be flat and match labels")
        if not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high)):
            raise ValueError("action bounds must be finite")
        if np.any(self.low > self.high):
            raise ValueError("action lower bounds must not exceed upper bounds")
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if self.max_step is None:
            return
        if len(self.max_step) != width:
            raise ValueError("max_step must be one entry per action dimension")
        for entry in self.max_step:
            if entry is not None and (not np.isfinite(entry) or entry <= 0):
                raise ValueError("max_step entries must be finite and > 0 or None")


@dataclass(frozen=True)
class MotionOutcome:
    """Result of validating one model motion call."""

    chunk: ActionChunk | None
    tool_result: str
    repairable: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Planner:
    """One model, with the call surface it will answer function tools on.

    The three travel together because the surface decides how much of the
    model an arm actually gets: on chat/completions astra accepts no
    reasoning_effort but "none" once tools are registered, so a condition put
    on the wrong surface can still return plausible tool calls while thinking
    not at all.
    """

    model: str
    api_style: str
    api_version: str


#: The planners a run can name, keyed by the short name that identifies the
#: condition everywhere else: the run id, the arm, the console column.
#:
#: Both are on the Responses API so that an arm differs from another arm by
#: its model and nothing else. gpt-5.5 does answer on chat/completions, and
#: with a reasoning_effort there, unlike astra -- but running the two
#: conditions on different surfaces would make the comparison about the
#: surface as much as the model.
PLANNERS: dict[str, Planner] = {
    "astra": Planner("gpt-6-astra", "responses", "2024-03-01-preview"),
    "gpt55": Planner("gpt-5.5-2026-04-24", "responses", "2024-03-01-preview"),
    # Claude on an OpenAI-compatible chat/completions endpoint; point L4_INSPECT_BASE_URL at it and
    # give the token's env var via L4_INSPECT_API_KEY_ENV (see the policy README).
    "fable": Planner("claude-fable-5-2", "chat", "2025-04-01-preview"),
    "opus55": Planner("claude-opus-5-5", "chat", "2025-04-01-preview"),
}
DEFAULT_PLANNER = "astra"

_DEFAULT_ENDPOINT = "https://aidp.bytedance.net/api/modelhub/online/v2/crawl"
#: Variables the run reads keys from, in the order it reaches for them.
#:
#: More than one because a rate limit is a property of the account, not of the
#: model: a sweep's ceiling is one key's quota, and a second key raises it
#: without changing what is being measured. Names that are unset contribute
#: nothing, so the same launch works unchanged on a machine that has only the
#: first one.
_DEFAULT_KEY_ENV = "ARK_API_KEY,ARK_API_KEY_BACKUP"
_DEFAULT_MODEL = PLANNERS[DEFAULT_PLANNER].model
_DEFAULT_API_VERSION = PLANNERS[DEFAULT_PLANNER].api_version
_DEFAULT_API_STYLE = PLANNERS[DEFAULT_PLANNER].api_style
_DEFAULT_REASONING_EFFORT = "medium"
_DEFAULT_MAX_LLM_CALLS = 100
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_HARD_TIMEOUT_S = 0.0
_DEFAULT_IMAGE_HORIZON = 2
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_MIN_SECRET_LEN = 4
_RETRY_BACKOFF = (1.0, 2.0)


def _optional_str(env: Mapping[str, str], key: str, default: str) -> str:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw)


def _optional_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        raise InfrastructureFailure(
            f"{key}={raw!r} is not a whole number; unset it to use {default}."
        ) from None


def _optional_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    token = str(raw).strip().lower()
    if token in _TRUE_VALUES:
        return True
    if token in _FALSE_VALUES:
        return False
    raise InfrastructureFailure(
        f"{key}={raw!r} is not a boolean; use 1/0, true/false, or unset it "
        f"to use {int(default)}."
    ) from None


def _optional_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        raise InfrastructureFailure(
            f"{key}={raw!r} is not a number; unset it to use {default}."
        ) from None


def validate_rgb_only_depth(env: Mapping[str, str]) -> None:
    """Reject depth rendering; misconfiguration is infrastructure, not capability."""
    depth = _optional_str(env, "L4_INSPECT_DEPTH", "off").lower()
    if depth != "off":
        raise InfrastructureFailure(
            "RoboDojo_Agent_L4_Inspect is RGB-only; set L4_INSPECT_DEPTH=off or unset it.",
            retryable=False,
        )


def planner_name(env: Mapping[str, str]) -> str:
    """Which planner this run names, defaulting to the one it always was.

    Unset means ``astra``, so every run recorded before there was a choice is
    still correctly described by its own trace.
    """
    name = _optional_str(env, "L4_INSPECT_PLANNER", DEFAULT_PLANNER).strip()
    if name not in PLANNERS:
        known = ", ".join(sorted(PLANNERS))
        raise InfrastructureFailure(
            f"L4_INSPECT_PLANNER={name!r} is not a planner this adapter knows; "
            f"expected one of: {known}"
        )
    return name


def client_config_from_env(env: Mapping[str, str]) -> dict[str, Any]:
    """Resolve Azure client settings from ``L4_INSPECT_*`` keys.

    The planner supplies the model and the surface it is served on; the three
    ``L4_INSPECT_MODEL`` / ``_API_STYLE`` / ``_API_VERSION`` keys still win
    where they are set, which is how a new model is tried before it earns a
    name in ``PLANNERS``.
    """
    planner = PLANNERS[planner_name(env)]
    hard_timeout_s = _optional_float(
        env, "L4_INSPECT_HARD_TIMEOUT_S", _DEFAULT_HARD_TIMEOUT_S
    )
    if not 0 <= hard_timeout_s < float("inf"):
        raise InfrastructureFailure(
            "L4_INSPECT_HARD_TIMEOUT_S must be finite and nonnegative; "
            "use 0 to disable it."
        )
    return {
        "planner": planner_name(env),
        "model": _optional_str(env, "L4_INSPECT_MODEL", planner.model),
        "azure_endpoint": _optional_str(env, "L4_INSPECT_BASE_URL", _DEFAULT_ENDPOINT),
        "api_version": _optional_str(env, "L4_INSPECT_API_VERSION", planner.api_version),
        "api_key_env": _optional_str(env, "L4_INSPECT_API_KEY_ENV", _DEFAULT_KEY_ENV),
        "timeout_s": _optional_float(env, "L4_INSPECT_TIMEOUT_S", _DEFAULT_TIMEOUT_S),
        "hard_timeout_s": hard_timeout_s,
        "api_style": _optional_str(
            env, "L4_INSPECT_API_STYLE", planner.api_style
        ).lower(),
        "reasoning_effort": _optional_str(
            env, "L4_INSPECT_REASONING_EFFORT", _DEFAULT_REASONING_EFFORT
        ),
    }


def api_key_pool(env: Mapping[str, str], names: str) -> list[tuple[str, str]]:
    """The keys this run may spend, paired with the variable each came from.

    ``names`` is a comma-separated list so that adding a key is adding a value
    to the environment, not a decision made at launch: the run reaches for the
    next one itself when the provider throttles the one it is on.

    Keys are deduplicated by value, since the same key reached through two
    names is one account and rotating between them would only look like relief.
    """
    pool: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in (part.strip() for part in names.split(",")):
        if not name:
            continue
        key = str(env.get(name, "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        pool.append((name, key))
    return pool


def responses_base_url(endpoint: str) -> str:
    """Derive the Responses base URL from the chat/completions endpoint.

    AIDP serves chat/completions under ``.../online/v2/crawl`` while the
    Responses API lives directly under ``.../online``.
    """
    trimmed = endpoint.rstrip("/")
    for suffix in ("/v2/crawl", "/v1/crawl", "/crawl"):
        if trimmed.endswith(suffix):
            return trimmed[: -len(suffix)]
    return trimmed


def _retry_delay_seconds(attempt: int, error: InfrastructureFailure) -> float:
    if error.retry_after_s is not None and error.retry_after_s > 0:
        return float(error.retry_after_s)
    if attempt < len(_RETRY_BACKOFF):
        return _RETRY_BACKOFF[attempt]
    return _RETRY_BACKOFF[-1]


def _infrastructure_failure(
    message: str,
    *,
    status: int | None = None,
    retry_after_s: float | None = None,
) -> InfrastructureFailure:
    retryable = False
    if status in {408, 429} or (status is not None and status >= 500):
        retryable = True
    return InfrastructureFailure(
        message, retryable=retryable, retry_after_s=retry_after_s, status=status
    )


def _model_refusal_from_error(error: Exception) -> CapabilityFailure | None:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        code = str(body.get("error", {}).get("code", "")).lower()
        if any(token in code for token in ("content_filter", "content_policy", "refusal")):
            return CapabilityFailure(f"model refusal: {code}")
    message = str(error).lower()
    if any(token in message for token in ("content filter", "content_policy", "refusal")):
        return CapabilityFailure(f"model refusal: {error}")
    return None


_SAFE_PROVIDER_FIELD = re.compile(r"[A-Za-z0-9_.\[\]/:-]{1,120}")


def _provider_error_suffix(error: Exception) -> str:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return ""
    detail = body.get("error", body)
    if not isinstance(detail, dict):
        return ""
    fields: list[str] = []
    for name in ("type", "code", "param"):
        value = detail.get(name)
        if value is None:
            continue
        text = str(value)
        if _SAFE_PROVIDER_FIELD.fullmatch(text):
            fields.append(f"{name}={text}")
    return f" ({', '.join(fields)})" if fields else ""


def classify_openai_error(error: Exception) -> CapabilityFailure | InfrastructureFailure:
    status = getattr(error, "status_code", None)
    if status is not None:
        detail = _provider_error_suffix(error)
        if (refusal := _model_refusal_from_error(error)) is not None:
            return refusal
        if status in {408, 429} or status >= 500:
            wrapped = _infrastructure_failure(f"HTTP {status}{detail}", status=status)
        elif status in {401, 403}:
            # The credential, not the request: a key with no grant for this
            # deployment, a revoked key, or an account out of quota. The
            # provider's own message is still dropped -- it sometimes quotes the
            # key back -- so the run carries only the status and the verdict.
            return InfrastructureFailure(
                f"HTTP {status} client error{detail}",
                retryable=False,
                status=status,
                key_unusable=True,
            )
        elif 400 <= status < 500:
            return InfrastructureFailure(
                f"HTTP {status} client error{detail}",
                retryable=False,
                status=status,
            )
        else:
            return InfrastructureFailure(
                f"HTTP {status} provider error{detail}", retryable=False
            )
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wrapped.retry_after_s = float(retry_after)
                except (TypeError, ValueError):
                    pass
        return wrapped
    error_name = type(error).__name__.lower()
    if "timeout" in error_name or "connection" in error_name:
        return InfrastructureFailure(str(error), retryable=True)
    if (refusal := _model_refusal_from_error(error)) is not None:
        return refusal
    return InfrastructureFailure(str(error), retryable=False)


def _validate_completion_shape(response: dict[str, Any]) -> None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise InfrastructureFailure("provider returned no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise InfrastructureFailure("provider returned a malformed message")
    finish_reason = str(choices[0].get("finish_reason") or "")
    if finish_reason in {"content_filter", "content_policy"}:
        raise CapabilityFailure(f"model refusal: finish_reason={finish_reason}")


class AzureAgentClient:
    """Azure client with bounded retry on transient failures.

    Talks to the Responses API by default and returns chat-shaped results, so
    the agent keeps its chat ``messages`` plumbing. ``L4_INSPECT_API_STYLE=chat``
    restores chat/completions, which cannot carry reasoning alongside tools.
    """

    def __init__(
        self,
        *,
        model: str,
        azure_endpoint: str,
        api_version: str,
        api_key: str | None = None,
        api_keys: Sequence[tuple[str, str]] | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        hard_timeout_s: float = _DEFAULT_HARD_TIMEOUT_S,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        complete_fn: Callable[..., dict[str, Any]] | None = None,
        session_id: str | None = None,
        api_style: str = _DEFAULT_API_STYLE,
        reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
    ) -> None:
        self.model = model
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self._keys = list(api_keys or [])
        if api_key and not self._keys:
            self._keys = [(_DEFAULT_KEY_ENV.split(",")[0], api_key)]
        if not self._keys:
            raise InfrastructureFailure("no API key was given to the client")
        self._active = 0
        # Which slot each key occupies, fixed at construction. Retiring a key
        # shortens self._keys, so its position there renumbers the survivors and
        # would reattribute every status recorded before the retirement.
        self._key_slot = {name: index for index, (name, _) in enumerate(self._keys)}
        self._status_tally: dict[str, dict[str, int]] = {}
        self.timeout_s = timeout_s
        self.hard_timeout_s = hard_timeout_s
        self.max_retries = max_retries
        self._complete_fn = complete_fn
        self.session_id = session_id
        self.api_style = api_style
        self.reasoning_effort = reasoning_effort
        # Reasoning items carry resource-bound encrypted state, so they live
        # here rather than in the agent's chat message list.
        self.reasoning_store = ReasoningReplayStore()

    @property
    def api_key(self) -> str:
        """The key the next request will be signed with."""
        return self._keys[self._active][1]

    @property
    def api_key_env(self) -> str:
        """The variable the active key came from. Safe to log; the key is not."""
        return self._keys[self._active][0]

    @property
    def api_key_envs(self) -> list[str]:
        return [name for name, _ in self._keys]

    def provider_status(self) -> dict[str, dict[str, int]]:
        """What the provider answered, per key, as a count per status code.

        The only durable record of a provider-side failure. The messages are
        dropped on purpose -- they sometimes quote the key back -- and the
        status otherwise reaches nothing but the stderr of a log that may not
        outlive the machine, so a sweep that lost its slots to 429s leaves no
        evidence of it.

        Keyed by slot rather than by variable name because the transcript
        redacts both any field whose name contains ``api_key`` and the key
        variables' own names, either of which would blank this out. Slot N is
        the Nth name in ``L4_INSPECT_API_KEY_ENV``.
        """
        return {slot: dict(counts) for slot, counts in self._status_tally.items()}

    def _record_status(self, error: InfrastructureFailure) -> None:
        """Attributed to the key in use now, before any rotation moves off it."""
        slot = f"key{self._key_slot.get(self.api_key_env, '?')}"
        # Timeouts and connection resets carry no status and are the other way a
        # slot is lost, so they are counted rather than dropped.
        code = "no_status" if error.status is None else str(error.status)
        counts = self._status_tally.setdefault(slot, {})
        counts[code] = counts.get(code, 0) + 1

    def _moved_off_the_active_key(
        self, error: InfrastructureFailure, throttled: set[str]
    ) -> bool:
        """Spend a different key when this one, not the provider, is the problem.

        Returns True when the call is worth repeating right away rather than
        after a backoff, which is the whole reason to carry a second key: a rate
        limit belongs to the account, so another account can serve the request
        now. ``throttled`` remembers, within one call, which keys have already
        answered with one, so two exhausted keys fall through to the ordinary
        backoff instead of ping-ponging between themselves.
        """
        if error.key_unusable:
            return self._retire_active_key()
        if error.status != 429:
            return False
        throttled.add(self.api_key_env)
        spare = next(
            (
                index
                for index, (name, _) in enumerate(self._keys)
                if name not in throttled
            ),
            None,
        )
        if spare is None:
            return False
        was = self.api_key_env
        self._active = spare
        print(
            f"[L4 inspect] {was} is rate limited; continuing on {self.api_key_env}",
            flush=True,
        )
        return True

    def _retire_active_key(self) -> bool:
        """Drop a key the provider will not serve this model with at all.

        Not transient and not the model's fault: the key belongs to an account
        that was never granted this deployment, or has lost the grant. Keeping
        it in the pool costs a wasted call every time rotation reaches it, and
        failing the run on it would make one account's missing permission look
        like the model refusing the task.
        """
        retired = self.api_key_env
        self._keys = [(name, key) for name, key in self._keys if name != retired]
        if not self._keys:
            return False
        self._active = 0
        print(
            f"[L4 inspect] {retired} cannot serve {self.model}; "
            f"continuing on {self.api_key_env}",
            flush=True,
        )
        return True

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str],
        *,
        complete_fn: Callable[..., dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> AzureAgentClient:
        config = client_config_from_env(env)
        key_env = str(config["api_key_env"])
        api_keys = api_key_pool(env, key_env)
        if not api_keys:
            raise InfrastructureFailure(
                f"API key environment variable {key_env!r} is unset or empty"
            )
        max_retries = _optional_int(env, "L4_INSPECT_MAX_RETRIES", _DEFAULT_MAX_RETRIES)
        keep_all_images = _optional_bool(env, "L4_INSPECT_KEEP_ALL_IMAGES", True)
        api_style = str(config["api_style"])
        if session_id is None and (keep_all_images or api_style == "responses"):
            session_id = uuid.uuid4().hex
        if not keep_all_images and api_style != "responses":
            # Stubbing older images rewrites history, which breaks a stateful
            # session; on the Responses path the session is required regardless,
            # since it also pins the Azure resource that owns reasoning items.
            session_id = None
        return cls(
            model=str(config["model"]),
            azure_endpoint=str(config["azure_endpoint"]),
            api_version=str(config["api_version"]),
            api_keys=api_keys,
            timeout_s=float(config["timeout_s"]),
            hard_timeout_s=float(config["hard_timeout_s"]),
            max_retries=max_retries,
            complete_fn=complete_fn,
            session_id=session_id,
            api_style=api_style,
            reasoning_effort=str(config["reasoning_effort"]),
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        attempts = 0
        # Per call, so a key throttled on one turn is reached for again on the
        # next: a rate limit lasts seconds, and an episode lasts minutes.
        throttled: set[str] = set()
        while True:
            try:
                if self._complete_fn is not None:
                    response = self._complete_fn(messages, tools)
                elif self.hard_timeout_s > 0:
                    response = self._isolated_azure_complete(messages, tools)
                else:
                    response = self._azure_complete(messages, tools)
                _validate_completion_shape(response)
                return response
            except CapabilityFailure:
                raise
            except InfrastructureFailure as error:
                failure = error
            except Exception as error:
                classified = classify_openai_error(error)
                if isinstance(classified, CapabilityFailure):
                    raise classified
                failure = classified
            self._record_status(failure)
            # Raised out here rather than in the handler: another key is worth
            # trying before the retry budget is, and a failure re-raised outside
            # the block carries no provider exception as its context, which is
            # what keeps a quoted-back key out of the traceback.
            if self._moved_off_the_active_key(failure, throttled):
                continue
            if not failure.retryable or attempts >= self.max_retries:
                raise failure
            time.sleep(_retry_delay_seconds(attempts, failure))
            attempts += 1

    def _isolated_azure_complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        request = (
            self._responses_request(messages, tools)
            if self.uses_responses_api()
            else self._chat_request(messages, tools)
        )
        payload = {
            "api_style": self.api_style,
            "model": self.model,
            "azure_endpoint": self.azure_endpoint,
            "api_version": self.api_version,
            "key_env": self.api_key_env,
            "timeout_s": self.timeout_s,
            "request": request,
        }
        worker_env = os.environ.copy()
        worker_env[self.api_key_env] = self.api_key
        process = subprocess.Popen(
            [sys.executable, str(_PLANNER_WORKER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=worker_env,
        )
        try:
            stdout, _stderr = process.communicate(
                json.dumps(payload), timeout=self.hard_timeout_s
            )
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise InfrastructureFailure(
                f"planner HTTP hard deadline exceeded after {self.hard_timeout_s:g}s",
                retryable=True,
            ) from None
        if process.returncode != 0:
            raise InfrastructureFailure(
                f"planner HTTP worker exited with status {process.returncode}",
                retryable=True,
            )
        try:
            envelope = json.loads(stdout)
            failure = envelope.get("failure")
            if failure is not None:
                if failure.get("kind") == "capability":
                    raise CapabilityFailure(str(failure["message"]))
                if failure.get("kind") != "infrastructure":
                    raise KeyError("unknown worker failure kind")
                raise InfrastructureFailure(
                    str(failure["message"]),
                    retryable=bool(failure.get("retryable")),
                    retry_after_s=failure.get("retry_after_s"),
                    status=failure.get("status"),
                    key_unusable=bool(failure.get("key_unusable")),
                )
            response = dict(envelope["response"])
        except (CapabilityFailure, InfrastructureFailure):
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise InfrastructureFailure(
                "planner HTTP worker returned an unreadable response",
                retryable=True,
            ) from None
        if self.uses_responses_api():
            return responses_to_chat_completion(
                response, reasoning_store=self.reasoning_store
            )
        return response

    def uses_responses_api(self) -> bool:
        return self.api_style == "responses"

    def _chat_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model.split("/", 1)[-1],
            "messages": messages,
            "tools": tools or None,
            "parallel_tool_calls": False,
            "reasoning_effort": self.reasoning_effort,
        }
        if self.session_id:
            request["extra_headers"] = session_headers(self.session_id)
        return request

    def _responses_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model.split("/", 1)[-1],
            "input": chat_messages_to_responses_input(
                messages, reasoning_store=self.reasoning_store
            ),
        }
        if tools:
            request["tools"] = chat_tools_to_responses_tools(tools)
            request["parallel_tool_calls"] = False
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if self.session_id:
            request["extra_headers"] = session_headers(self.session_id)
        return request

    def _azure_complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self.uses_responses_api():
            return self._responses_complete(messages, tools)
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise InfrastructureFailure("openai package is not installed") from None
        try:
            client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                max_retries=0,
                timeout=self.timeout_s,
            )
        except Exception:
            raise InfrastructureFailure("failed to construct Azure client") from None
        request = self._chat_request(messages, tools)
        try:
            response = client.chat.completions.create(**request)
        except Exception as error:
            raise classify_openai_error(error) from None
        try:
            return json.loads(response.model_dump_json())
        except (TypeError, ValueError, json.JSONDecodeError):
            raise InfrastructureFailure("provider returned malformed JSON") from None

    def _responses_complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError:
            raise InfrastructureFailure("openai package is not installed") from None
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=responses_base_url(self.azure_endpoint),
                max_retries=0,
                timeout=self.timeout_s,
            )
        except Exception:
            raise InfrastructureFailure("failed to construct Azure client") from None
        request = self._responses_request(messages, tools)
        try:
            response = create_response(client, request)
        except Exception as error:
            raise classify_openai_error(error) from None
        try:
            return responses_to_chat_completion(
                response, reasoning_store=self.reasoning_store
            )
        except (TypeError, ValueError) as error:
            raise InfrastructureFailure(
                f"provider returned an unreadable response: {error}"
            ) from None



def create_response(client: Any, request: dict[str, Any]) -> Any:
    """Create one Responses API turn, streaming by default.

    Some OpenAI-compatible relays only accept ``stream=true``. Streaming also
    keeps long reasoning turns from idling a proxy connection. The returned
    object is the final ``response.completed`` payload, the same object a
    non-streaming ``create`` would return. ``L4_INSPECT_STREAM=0`` restores the
    plain call.
    """
    if os.environ.get("L4_INSPECT_STREAM", "1").strip().lower() in {"0", "false", "no", "off"}:
        return client.responses.create(**request)
    final = None
    stream = client.responses.create(stream=True, **request)
    try:
        for event in stream:
            kind = getattr(event, "type", "")
            if kind in ("response.completed", "response.incomplete", "response.failed"):
                final = getattr(event, "response", None)
            elif kind == "error":
                raise RuntimeError(f"stream error event: {getattr(event, 'message', event)}")
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    if final is None:
        raise RuntimeError("stream ended without a final response")
    return final


def _format_bound(value: float) -> str:
    """Format one finite action bound compactly for the model-facing schema."""
    return f"{float(value):.4g}"


def _tool_schemas(action_spec: RoboDojoActionSpec) -> list[dict[str, Any]]:
    bounds = ", ".join(
        f"{label}: [{_format_bound(low)}, {_format_bound(high)}]"
        for label, low, high in zip(
            action_spec.labels, action_spec.low, action_spec.high, strict=True
        )
    )
    return [
        {
            "type": "function",
            "function": {
                "name": "move_joints",
                "description": (
                    "Move the robot toward absolute joint targets. "
                    "Name only the joints you intend to change. "
                    f"Per-dimension bounds: {bounds}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                            "description": "Named absolute joint targets in radians or gripper units.",
                        },
                        "note": {
                            "type": "string",
                            "description": "Brief rationale for the motion.",
                        },
                    },
                    "required": ["targets", "note"],
                },
            },
        },
        # There is deliberately no tool for declaring the task finished.
        # RoboDojo ends the episode itself the moment its reward fires, so any
        # moment the model is still being asked to act in is one where the goal
        # has not been met, and declaring otherwise only forfeits the rest of
        # the run. All 22 recorded episodes that ended that way scored zero.
        {
            "type": "function",
            "function": {
                "name": "give_up",
                "description": "End the episode because the task cannot be completed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "hindsight": {"type": "string"},
                    },
                    "required": ["reason", "hindsight"],
                },
            },
        },
    ]


def use_task_recipe(env: Mapping[str, str] | None = None) -> bool:
    """Always False: this adapter carries no per-task text.

    The upstream adapter could append ``recipes/<task>.md`` chosen by task
    name. Under the Agent Track rules the official Detail Description, which
    the client supplies as the instruction, is the only task-conditioned text,
    so the recipe files and the switch are gone.
    """
    del env
    return False


def _goal_content(
    *,
    instruction: str | None,
    task_name: str | None,
    env: Mapping[str, str],
) -> str:
    del task_name, env
    return f"Goal (official task description):\n{instruction or ''}"


def _system_prompt(embodiment_docs: str | None = None) -> str:
    prompt = (
        "You are controlling a real robot embodiment named 'robodojo-arx-x5'. "
        "You receive RGB camera images, the current state of every dimension "
        "you can command, and a task instruction. Respond with exactly one "
        "tool call per turn. The environment has its own step limit, reported "
        "with each observation as the env steps remaining, and an accepted "
        "motion reports how many env steps it spent. Steps are spent by "
        "distance travelled, not by turns taken, so a small correction is "
        "nearly free and there is no reason to cover extra ground in one turn."
    )
    if embodiment_docs and embodiment_docs.strip():
        prompt += "\n\nEmbodiment notes:\n" + embodiment_docs.strip()
    return prompt


def encode_jpeg_data_uri(image: np.ndarray) -> str:
    """Encode an RGB ndarray as a JPEG data URI without channel swapping."""
    array = np.asarray(image)
    if array.dtype.kind == "f":
        raise ValueError("expected an integer RGB image array, not floating point")
    array = array.astype(np.uint8, copy=False)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an RGB image with shape (H, W, 3)")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=95)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _state_text(labels: Sequence[str], values: Sequence[float], instruction: str | None) -> str:
    pairs = [f"{label}={values[index]:.4f}" for index, label in enumerate(labels)]
    lines = [f"Instruction: {instruction or ''}", "Joint state:"]
    lines.extend(pairs)
    return "\n".join(lines)


def _label_index(labels: Sequence[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(labels)}


def _named_targets_to_vector(
    *,
    labels: Sequence[str],
    current: np.ndarray,
    targets: Mapping[str, Any],
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray | None, str | None, list[str]]:
    vector = current.astype(np.float64, copy=True)
    label_map = _label_index(labels)
    clamp_notes: list[str] = []
    for name, raw in targets.items():
        if name not in label_map:
            return None, f"unknown dimension {name!r}", []
        index = label_map[name]
        requested = float(raw)
        if not np.isfinite(requested):
            return None, f"target for {name!r} must be finite", []
        clamped = float(np.clip(requested, low[index], high[index]))
        if clamped != requested:
            clamp_notes.append(f"{name}: requested {requested:.4f}, clamped to {clamped:.4f}")
        vector[index] = clamped
    return vector, None, clamp_notes


def _interpolate_chunk(
    *,
    current: np.ndarray,
    target: np.ndarray,
    max_step: Sequence[float | None],
    control_hz: float,
    low: np.ndarray,
    high: np.ndarray,
) -> ActionChunk:
    action_space = ActionSpace(JOINT_CHANNELS)
    steps = max(1, _interpolation_steps(current, target, max_step))
    waypoints: list[np.ndarray] = []
    for step_index in range(steps):
        alpha = (step_index + 1) / steps
        waypoint = current + alpha * (target - current)
        waypoint = np.clip(waypoint, low, high)
        waypoints.append(waypoint)
    actions = []
    for index, waypoint in enumerate(waypoints):
        decoded = action_space.decode(waypoint.tolist())
        meta: dict[str, Any] = {}
        if index == len(waypoints) - 1:
            meta["chunk_final"] = True
        actions.append(Action(data=decoded.data, meta=meta))
    return ActionChunk(actions=actions, control_hz=control_hz)


def _interpolation_steps(
    current: np.ndarray, target: np.ndarray, max_step: Sequence[float | None]
) -> int:
    if np.allclose(current, target, atol=1e-9, rtol=0.0):
        return 1
    steps = 1
    for index, (start, end) in enumerate(zip(current, target, strict=True)):
        delta = abs(float(end) - float(start))
        if delta <= 1e-12:
            continue
        limit = max_step[index]
        if limit is None or limit <= 0:
            continue
        steps = max(steps, int(np.ceil(delta / float(limit))))
    return steps


def _vector_from_state(state: Mapping[str, Any]) -> np.ndarray:
    """Flatten a RoboDojo state dict in joint-channel order.

    Labels, bounds and per-step limits are all indexed by that same position, so
    nothing here may be derived from the text of a label.
    """
    return np.asarray(
        ActionSpace(JOINT_CHANNELS).encode(state), dtype=np.float64
    )


def _state_dict_from_vector(vector: Sequence[float]) -> dict[str, np.ndarray]:
    return dict(ActionSpace(JOINT_CHANNELS).decode(vector).data)


def _compact_message_history_in_place(
    messages: list[dict[str, Any]], *, image_horizon: int
) -> None:
    if image_horizon < 1:
        raise ValueError("image_horizon must be at least 1")
    observation_turns = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and isinstance(message.get("content"), list)
        and any(part.get("type") == "image_url" for part in message["content"])
    ]
    for index in observation_turns[:-image_horizon]:
        content = messages[index].get("content")
        if not isinstance(content, list):
            continue
        stubbed: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image_url":
                stubbed.append(
                    {
                        "type": "text",
                        "text": "[earlier camera image omitted to save context]",
                    }
                )
            else:
                stubbed.append(part)
        messages[index] = {"role": "user", "content": stubbed}


class JointAgentPolicy:
    """Local inspect-inspired joint-target policy backed by Azure chat completions."""

    motion_tool_name = "move_joints"

    def __init__(
        self,
        *,
        action_spec: RoboDojoActionSpec,
        env: Mapping[str, str],
        client: Any | None = None,
    ) -> None:
        self.action_spec = action_spec
        self._env = dict(env)
        validate_rgb_only_depth(self._env)
        # Resolved once, here, so a planner name nothing recognises is refused
        # before the simulator spends its cold start rather than at the first
        # transcript write -- and so the trace cannot disagree with the client
        # about which model ran.
        self._client_config = client_config_from_env(self._env)
        self._tools = self._build_tools(action_spec)
        self._max_llm_calls = _optional_int(
            self._env, "L4_INSPECT_MAX_LLM_CALLS", _DEFAULT_MAX_LLM_CALLS
        )
        self._keep_all_images = _optional_bool(
            self._env, "L4_INSPECT_KEEP_ALL_IMAGES", True
        )
        self._image_horizon = _optional_int(
            self._env, "L4_INSPECT_IMAGE_HORIZON", _DEFAULT_IMAGE_HORIZON
        )
        if not self._keep_all_images and self._image_horizon < 1:
            raise InfrastructureFailure(
                f"L4_INSPECT_IMAGE_HORIZON={self._image_horizon} would stub every "
                "camera image out of the conversation; set it to 1 or more, or "
                f"unset it to use {_DEFAULT_IMAGE_HORIZON}."
            )
        self._depth = _optional_str(self._env, "L4_INSPECT_DEPTH", "off").lower()
        self._layout_id: int | None = None
        self._task_name: str | None = None
        self._messages: list[dict[str, Any]] = []
        self._goal_sent = False
        self._calls = 0
        self._hindsight: str | None = None
        self._transcript: list[dict[str, Any]] = []
        if client is not None:
            self._client = client
        else:
            self._client = AzureAgentClient.from_env(
                self._env,
                session_id=uuid.uuid4().hex if self._keep_all_images else None,
            )
        self.reset()

    def _build_tools(self, action_spec: RoboDojoActionSpec) -> list[dict[str, Any]]:
        return _tool_schemas(action_spec)

    def _system_message(self) -> str:
        return _system_prompt(self.action_spec.docs)

    def reset(self) -> None:
        azure = self._client if isinstance(self._client, AzureAgentClient) else None
        # The Responses path needs a session even when history is rewritten: it
        # pins the Azure resource that owns this episode's reasoning items.
        if self._keep_all_images or (azure is not None and azure.uses_responses_api()):
            self._session_id = uuid.uuid4().hex
            if azure is not None:
                azure.session_id = self._session_id
        else:
            self._session_id = None
        if azure is not None:
            azure.reasoning_store.clear()
        self._messages = [
            {
                "role": "system",
                "content": self._system_message(),
            }
        ]
        self._goal_sent = False
        self._goal_text: str | None = None
        self._calls = 0
        self._hindsight = None
        self._layout_id = None
        self._task_name = None
        self._transcript = []

    def prepare(self, observation: Observation) -> None:
        layout_id = observation.extra.get("layout_id")
        if isinstance(layout_id, int):
            self._layout_id = layout_id
        task_name = observation.extra.get("task")
        if isinstance(task_name, str) and task_name.strip():
            self._task_name = task_name.strip()
        if not self._goal_sent:
            self._goal_text = _goal_content(
                instruction=observation.instruction,
                task_name=self._task_name,
                env=self._env,
            )
            self._messages.append({"role": "user", "content": self._goal_text})
            self._append_prior_attempt()
            self._goal_sent = True

    def _append_prior_attempt(self) -> None:
        """Replay an earlier attempt's conversation, when one is configured."""
        path = self._prior_transcript_path()
        if not path:
            return
        note = _optional_str(self._env, "L4_INSPECT_PRIOR_NOTE", "") or None
        try:
            messages = load_prior_messages(
                path, render_state=self._render_prior_state, note=note
            )
        except ValueError as error:
            raise InfrastructureFailure(str(error)) from error
        self._messages.extend(messages)

    def _prior_transcript_path(self) -> str:
        # Disabled on purpose. Replaying another episode's conversation would
        # carry one episode's experience into another, which the Agent Track
        # rules forbid; every episode starts from the same frozen state.
        return ""

    def _render_prior_state(
        self, state: Mapping[str, Any], instruction: str | None
    ) -> str:
        """A replayed turn's joint state, in the same shape as a live one."""
        return _state_text(
            list(state), [float(value) for value in state.values()], instruction
        )

    def act(self, observation: Observation) -> ActionChunk:
        if self._calls >= self._max_llm_calls:
            return self._give_up_chunk("LLM call budget exhausted", observation)
        self.prepare(observation)
        self._messages.append(self._observation_message(observation))
        repair_attempts = 0
        while True:
            if self._calls >= self._max_llm_calls:
                return self._give_up_chunk("LLM call budget exhausted", observation)
            if not self._keep_all_images:
                _compact_message_history_in_place(
                    self._messages, image_horizon=self._image_horizon
                )
            started = time.monotonic()
            response = self._client.complete(self._messages, self._tools)
            latency_s = time.monotonic() - started
            self._calls += 1
            turn_record = self._record_turn(
                response,
                policy_step=observation.step,
                repair_attempt=repair_attempts,
                latency_s=latency_s,
            )
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            content = message.get("content")
            if not tool_calls:
                turn_record["validation_error"] = (
                    "model returned text without a tool call"
                    if content
                    else "model returned an empty response"
                )
                if content:
                    self._messages.append({"role": "assistant", "content": content})
                    self._messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Reply with exactly one tool call from the provided tools."
                            ),
                        }
                    )
                    repair_attempts += 1
                    if repair_attempts >= 3:
                        raise CapabilityFailure("Model kept failing: no tool call")
                    continue
                self._messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The model returned an empty response. "
                            "Reply with exactly one tool call from the provided tools."
                        ),
                    }
                )
                repair_attempts += 1
                if repair_attempts >= 3:
                    raise CapabilityFailure("Model kept failing: empty response")
                continue
            tool_calls = self._normalize_tool_calls(tool_calls)
            if len(tool_calls) != 1:
                turn_record["validation_error"] = (
                    f"model returned {len(tool_calls)} tool calls; expected exactly one"
                )
                raise CapabilityFailure("Model must return exactly one tool call")
            call = tool_calls[0]
            name = call["function"]["name"]
            turn_record["tool"] = name
            try:
                arguments = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError as error:
                repair = f"invalid JSON arguments: {error}"
                turn_record["validation_error"] = repair
                turn_record["tool_result"] = repair
                self._append_tool_repair(call, content, repair)
                repair_attempts += 1
                if repair_attempts >= 3:
                    raise CapabilityFailure("Model kept failing to produce a valid tool call")
                continue
            turn_record["arguments"] = arguments
            self._messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )
            if self._is_motion_tool(name):
                outcome = self._handle_motion(name, arguments, observation)
                turn_record["tool_result"] = outcome.tool_result
                self._messages.append(
                    self._tool_result(call["id"], outcome.tool_result)
                )
                if outcome.chunk is not None:
                    turn_record["accepted"] = True
                    return outcome.chunk
                if outcome.repairable:
                    turn_record["validation_error"] = outcome.tool_result
                    repair_attempts += 1
                    if repair_attempts >= 3:
                        raise CapabilityFailure(
                            "Model kept failing to produce a valid tool call"
                        )
                else:
                    # A valid Cartesian request can still be unreachable. Let
                    # the model choose another pose without counting it as a
                    # schema-repair attempt.
                    turn_record["accepted"] = True
                continue
            # Query tools (perception helpers) answer from the current
            # observation and move nothing, so the model gets the answer and
            # keeps the turn. They are not schema repairs.
            query_result = self._handle_query(name, arguments, observation)
            if query_result is not None:
                turn_record["accepted"] = True
                turn_record["tool_result"] = query_result
                self._messages.append(self._tool_result(call["id"], query_result))
                continue
            # "done" is not in this set even though the stop path still knows
            # the name: a hallucinated one would forfeit a run that the episode
            # itself would otherwise have been allowed to finish.
            if name == "give_up":
                turn_record["accepted"] = True
                turn_record["tool_result"] = f"Acknowledged {name}."
                self._messages.append(
                    self._tool_result(call["id"], turn_record["tool_result"])
                )
                return self._stop_chunk(name, arguments, observation)
            repair = f"unknown tool {name!r}"
            turn_record["validation_error"] = repair
            turn_record["tool_result"] = repair
            self._messages.append(self._tool_result(call["id"], repair))
            repair_attempts += 1
            if repair_attempts >= 3:
                raise CapabilityFailure("Model kept failing to produce a valid tool call")

    def confirm_executed(self, played: int) -> None:
        del played

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def hindsight(self) -> str | None:
        return self._hindsight

    def transcript(self) -> list[dict[str, Any]] | None:
        return list(self._transcript)

    def audit_config(self) -> dict[str, Any]:
        client = self._client_config
        return {
            "adapter": "robodojo-agent-l4-inspect",
            "planner": client["planner"],
            "model": client["model"],
            "azure_endpoint": client["azure_endpoint"],
            "api_version": client["api_version"],
            # How many accounts this episode could draw on, not which ones. A
            # throttled run that had one key reads very differently from one
            # that had two. Deliberately not named after the key: the transcript
            # redacts any field whose name looks like one, which would blank
            # this count out.
            "provider_keys": len(api_key_pool(self._env, str(client["api_key_env"]))),
            # What the provider actually answered this episode, per key slot.
            # Empty on a run that never saw a provider error, which is most of
            # them; non-empty is the record of a run that spent time on one.
            "provider_status": (
                self._client.provider_status()
                if hasattr(self._client, "provider_status")
                else {}
            ),
            "scene": {"init_seed": self._layout_id},
            "prior_transcript": self._prior_transcript_path() or None,
            # Every word the model was given, so a published transcript can be
            # read without the code that produced it. The three cover the whole
            # prompt between them: the system message carries the embodiment
            # notes, the goal turn carries the task recipe, and the tool schema
            # carries the bounds, the units and the angle convention. Taken
            # from the messages as sent rather than regenerated, because what
            # this has to record is what the model saw.
            "prompt": {
                "system": self._messages[0]["content"] if self._messages else None,
                "goal": self._goal_text,
                "tools": self._tools,
            },
            "code": source_revision(),
            "embodiment": {
                "labels": list(self.action_spec.labels),
                "low": self.action_spec.low.tolist(),
                "high": self.action_spec.high.tolist(),
                "control_hz": self.action_spec.control_hz,
                "max_step": (
                    list(self.action_spec.max_step)
                    if self.action_spec.max_step is not None
                    else None
                ),
                "docs": self.action_spec.docs,
            },
            "policy_config": {
                "depth": self._depth,
                "max_llm_calls": self._max_llm_calls,
                "use_recipe": use_task_recipe(self._env),
                "task": self._task_name,
                "keep_all_images": self._keep_all_images,
                "cache_session_id": self._session_id,
                "image_horizon": self._image_horizon,
                "hard_timeout_s": client["hard_timeout_s"],
                "api_style": client["api_style"],
                "reasoning_effort": _optional_str(
                    self._env, "L4_INSPECT_REASONING_EFFORT", _DEFAULT_REASONING_EFFORT
                ),
            },
        }

    def _state_block(self, observation: Observation) -> str:
        """Render the state in the space the model commands, one line per dim.

        A subclass whose model-facing action space is not the joint space has to
        replace this whole block rather than append to it: the leading state is
        the reference its absolute targets are measured against, so rendering
        some other space here would hand the model the wrong starting point.
        """
        current = _vector_from_state(observation.state)
        return _state_text(
            self.action_spec.labels,
            current.tolist(),
            observation.instruction,
        )

    def _observation_message(self, observation: Observation) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": self._state_block(observation)}
        ]
        if observation.remaining_steps is not None:
            # The step limit is the budget that actually ends most episodes,
            # and until it is reported the model has no way to know it exists.
            parts[0]["text"] += (
                f"\nEnv steps remaining before the episode ends: "
                f"{observation.remaining_steps}"
            )
        for name, image in observation.images.items():
            parts.append({"type": "text", "text": f"camera '{name}' (step {observation.step}):"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_jpeg_data_uri(image)},
                }
            )
        return {"role": "user", "content": parts}

    def _handle_move(
        self, arguments: Mapping[str, Any], observation: Observation
    ) -> tuple[ActionChunk | None, str | None, list[str]]:
        targets = arguments.get("targets")
        if not isinstance(targets, Mapping):
            return None, "move_joints.targets must be an object of named joint targets", []
        current = _vector_from_state(observation.state)
        target, error, clamp_notes = _named_targets_to_vector(
            labels=self.action_spec.labels,
            current=current,
            targets=targets,
            low=self.action_spec.low,
            high=self.action_spec.high,
        )
        if error is not None:
            return None, error, []
        assert target is not None
        max_step = self.action_spec.max_step or tuple(None for _ in self.action_spec.labels)
        actions = _interpolate_chunk(
            current=current,
            target=target,
            max_step=max_step,
            control_hz=self.action_spec.control_hz,
            low=self.action_spec.low,
            high=self.action_spec.high,
        )
        requested_targets = {str(name): float(value) for name, value in targets.items()}
        clamped_targets = {
            name: float(target[self.action_spec.labels.index(name)])
            for name in requested_targets
        }
        return (
            ActionChunk(
                actions=actions.actions,
                control_hz=actions.control_hz,
                meta={
                    "trace": {
                        "tool": "move_joints",
                        "requested_targets": requested_targets,
                        "clamped_targets": clamped_targets,
                        "target": dict(
                            zip(
                                self.action_spec.labels,
                                (float(value) for value in target),
                                strict=True,
                            )
                        ),
                        "clamp_notes": list(clamp_notes),
                        "planned_waypoints": len(actions),
                    }
                },
            ),
            None,
            clamp_notes,
        )

    def _handle_motion(
        self,
        name: str,
        arguments: Mapping[str, Any],
        observation: Observation,
    ) -> MotionOutcome:
        del name
        chunk, repair, clamp_notes = self._handle_move(arguments, observation)
        if repair is not None:
            return MotionOutcome(
                chunk=None,
                tool_result=repair,
                repairable=True,
            )
        assert chunk is not None
        message = (
            "Accepted after clamping: " + "; ".join(clamp_notes)
            if clamp_notes
            else "Accepted."
        )
        return MotionOutcome(
            chunk=chunk,
            tool_result=message,
            notes=tuple(clamp_notes),
        )

    def _normalize_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return tool_calls

    def _handle_query(
        self, name: str, arguments: Mapping[str, Any], observation: Observation
    ) -> str | None:
        """Answer a non-motion tool, or return None when ``name`` is not one."""
        del name, arguments, observation
        return None

    def _is_motion_tool(self, name: str) -> bool:
        return name == self.motion_tool_name

    def _stop_chunk(
        self, name: str, arguments: Mapping[str, Any], observation: Observation
    ) -> ActionChunk:
        hindsight = str(arguments.get("hindsight") or "")
        self._hindsight = hindsight or None
        detail = arguments.get("summary") if name == "done" else arguments.get("reason")
        hold = _state_dict_from_vector(_vector_from_state(observation.state))
        meta = {
            "request_stop": True,
            "stop_reason": name,
            "stop_detail": detail,
        }
        return ActionChunk(
            actions=[Action(data=hold, meta=meta)],
            control_hz=self.action_spec.control_hz,
            meta={
                "trace": {
                    "tool": name,
                    "arguments": dict(arguments),
                    "planned_waypoints": 1,
                }
            },
        )

    def _give_up_chunk(self, reason: str, observation: Observation) -> ActionChunk:
        self._hindsight = None
        hold = _state_dict_from_vector(_vector_from_state(observation.state))
        return ActionChunk(
            actions=[
                Action(
                    data=hold,
                    meta={
                        "request_stop": True,
                        "stop_reason": "give_up",
                        "stop_detail": reason,
                    },
                )
            ],
            control_hz=self.action_spec.control_hz,
            meta={
                "trace": {
                    "tool": "give_up",
                    "arguments": {"reason": reason},
                    "planned_waypoints": 1,
                }
            },
        )

    def _append_tool_repair(
        self, call: Mapping[str, Any], content: Any, message: str
    ) -> None:
        self._messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [call],
            }
        )
        self._messages.append(self._tool_result(str(call["id"]), message))

    def _tool_result(self, tool_call_id: str, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def _record_turn(
        self,
        response: dict[str, Any],
        *,
        policy_step: int,
        repair_attempt: int,
        latency_s: float,
    ) -> dict[str, Any]:
        sanitized = json.loads(json.dumps(response))
        choice = (sanitized.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        record = {
            "call_index": self._calls,
            "policy_step": int(policy_step),
            "repair_attempt": int(repair_attempt),
            "latency_s": float(latency_s),
            "response_id": sanitized.get("id"),
            "model": sanitized.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "usage": sanitized.get("usage") or {},
            "content": message.get("content"),
            "accepted": False,
            "tool": None,
            "arguments": None,
            "validation_error": None,
            "tool_result": None,
            "response": sanitized,
        }
        self._transcript.append(record)
        return record
