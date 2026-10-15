"""Standalone RoboDojo L3 inspect-inspired joint agent policy."""

from __future__ import annotations

import base64
import io
import json
import os
import random
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
from .recipes import task_recipe
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

    A planner may leave ``endpoint`` / ``api_key_env`` unset and inherit the
    public runtime defaults. Provider-specific endpoints and credentials are
    supplied through environment variables.
    """

    model: str
    api_style: str
    api_version: str
    endpoint: str | None = None
    api_key_env: str | None = None
    reasoning_effort: str | None = None
    timeout_s: float | None = None
    provider: str = "azure"
    pin_session: bool = True


#: The planners a run can name, keyed by the short name that identifies the
#: condition everywhere else: the run id, the arm, the console column.
#:
#: astra and gpt55 stay on the Responses API so that an arm differs from
#: another arm by its model and nothing else. gpt-5.5 does answer on
#: chat/completions, and with a reasoning_effort there, unlike astra -- but
#: running those two conditions on different surfaces would make the
#: comparison about the surface as much as the model. kimi is the same
#: Responses shape against Moonshot rather than AIDP.
PLANNERS: dict[str, Planner] = {
    "astra": Planner("gpt-6-astra", "responses", "2024-03-01-preview"),
    "gpt55": Planner("gpt-5.5-2026-04-24", "responses", "2024-03-01-preview"),
    "kimi": Planner(
        "kimi-k3",
        "responses",
        "2024-03-01-preview",
        endpoint="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        reasoning_effort="high",
        timeout_s=180.0,
        provider="openai",
        pin_session=False,
    ),
}
DEFAULT_PLANNER = "astra"

#: Variables the run reads keys from, in the order it reaches for them.
#:
#: More than one because a rate limit is a property of the account, not of the
#: model: a sweep's ceiling is one key's quota, and a second key raises it
#: without changing what is being measured. Names that are unset contribute
#: nothing, so the same launch works unchanged on a machine that has only the
#: first one.
_DEFAULT_KEY_ENV = "OPENAI_API_KEY,OPENAI_API_KEY_BACKUP"
_DEFAULT_MODEL = PLANNERS[DEFAULT_PLANNER].model
_DEFAULT_API_VERSION = PLANNERS[DEFAULT_PLANNER].api_version
_DEFAULT_API_STYLE = PLANNERS[DEFAULT_PLANNER].api_style
_DEFAULT_REASONING_EFFORT = "medium"
#: Kimi K3 only accepts low / high / max. The L3 default "medium" and the
#: unused "none" are folded onto the nearest legal values so a kimi run does
#: not have to restate effort just to start.
_KIMI_REASONING_EFFORT = {
    "none": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "max": "max",
}
_DEFAULT_MAX_LLM_CALLS = 100
_DEFAULT_MAX_RETRIES = 12
_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_HARD_TIMEOUT_S = 0.0
_DEFAULT_IMAGE_HORIZON = 2
# imitate_sorting_sequence: the opposite arm replays a per-layout trajectory
# while the policy must hold still, and moving one step early ends the episode
# at zero. Across the 65 scripted trajectories that replay takes 396-566 env
# steps, so the wait covers the longest with margin rather than the average.
# Calling the planner during it would burn the LLM budget on hold-still turns
# and still never see the sequence unless those frames are kept.
_IMITATE_SORTING_WATCH_S = 24.0
_IMITATE_SORTING_WATCH_SAMPLE_S = 1.0
_WATCH_MESSAGE_PREFIX = "DEMONSTRATION WATCH FRAME"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_MIN_SECRET_LEN = 4
# A sweep runs every rollout against one account, so a rate limit is the
# steady state and not a blip: the ladder has to outlast a throttling window
# rather than probe it twice. It stops doubling at the cap because past that
# point a retry is only waiting, and a shard that waits longer than a few
# minutes is holding its GPU for nothing.
_RETRY_BACKOFF = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)


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


def _optional_absent_int(env: Mapping[str, str], key: str) -> int | None:
    """Like ``_optional_int`` but ``None`` when the key is unset.

    Unset and ``0`` are different for ``L3_INSPECT_DEFER_LLM_UNTIL_ENV_STEP``:
    unset keeps per-task defaults, ``0`` turns every defer off.
    """
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        raise InfrastructureFailure(
            f"{key}={raw!r} is not a whole number."
        ) from None


def _base_task_name(task_name: str | None) -> str:
    if not task_name:
        return ""
    return task_name.strip().removesuffix("_random")


def _defer_llm_plan(
    *,
    task_name: str | None,
    control_hz: float,
    env_until: int | None,
    env_tasks: set[str],
    env_sample_every: int | None,
) -> tuple[int, int]:
    """How long to hold before the first LLM call, and how often to sample.

    ``sample_every`` of 0 means one blind chunk with no watch-frame history.
    """
    base = _base_task_name(task_name)
    builtin_until = 0
    builtin_sample = 0
    if (
        base == "imitate_sorting_sequence"
        and np.isfinite(control_hz)
        and control_hz > 0
    ):
        builtin_until = int(round(_IMITATE_SORTING_WATCH_S * control_hz))
        builtin_sample = max(1, int(round(_IMITATE_SORTING_WATCH_SAMPLE_S * control_hz)))
    def _with_sample(until: int, default_sample: int) -> tuple[int, int]:
        if until <= 0:
            return 0, 0
        sample = default_sample
        if env_sample_every is not None:
            sample = max(0, env_sample_every)
        return until, sample

    if env_until is None:
        return _with_sample(builtin_until, builtin_sample)
    if env_until <= 0:
        return 0, 0
    named = _base_task_name(task_name)
    if env_tasks and (
        named in env_tasks or (task_name or "").strip() in env_tasks
    ):
        return _with_sample(env_until, builtin_sample)
    return _with_sample(builtin_until, builtin_sample)


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
    depth = _optional_str(env, "L3_INSPECT_DEPTH", "off").lower()
    if depth != "off":
        raise InfrastructureFailure(
            "RoboDojo_Agent_L3_Inspect is RGB-only; set L3_INSPECT_DEPTH=off or unset it.",
            retryable=False,
        )


def planner_name(env: Mapping[str, str]) -> str:
    """Which planner this run names, defaulting to the one it always was.

    Unset means ``astra``, so every run recorded before there was a choice is
    still correctly described by its own trace.
    """
    name = _optional_str(env, "L3_INSPECT_PLANNER", DEFAULT_PLANNER).strip()
    if name not in PLANNERS:
        known = ", ".join(sorted(PLANNERS))
        raise InfrastructureFailure(
            f"L3_INSPECT_PLANNER={name!r} is not a planner this adapter knows; "
            f"expected one of: {known}"
        )
    return name


def _required_base_url(env: Mapping[str, str], planner: Planner) -> str:
    """The host is a launch choice, not a planner default.

    A missing URL used to fall through to ``https://api.openai.com/v1`` and
    then time out on the first call. Refuse here so that failure is named.
    """
    raw = env.get("L3_INSPECT_BASE_URL")
    if raw is None or str(raw).strip() == "":
        hint = ""
        if planner.endpoint:
            hint = f" For {planner_name(env)}, that is typically {planner.endpoint}."
        raise InfrastructureFailure(
            "L3_INSPECT_BASE_URL is unset or empty. Export the provider's "
            "OpenAI-compatible /v1 base URL; there is no default host."
            + hint
        )
    return str(raw).strip()


def _reasoning_effort_for(name: str, requested: str) -> str:
    """Map a requested effort onto the values this planner's provider accepts."""
    if name != "kimi":
        return requested
    mapped = _KIMI_REASONING_EFFORT.get(requested.strip().lower())
    if mapped is None:
        legal = ", ".join(sorted(_KIMI_REASONING_EFFORT))
        raise InfrastructureFailure(
            f"L3_INSPECT_REASONING_EFFORT={requested!r} is not a Kimi K3 "
            f"effort; expected one of: {legal}"
        )
    return mapped


def client_config_from_env(env: Mapping[str, str]) -> dict[str, Any]:
    """Resolve Azure client settings from ``L3_INSPECT_*`` keys.

    The planner supplies the model and the surface it is served on; the three
    ``L3_INSPECT_MODEL`` / ``_API_STYLE`` / ``_API_VERSION`` keys still win
    where they are set, which is how a new model is tried before it earns a
    name in ``PLANNERS``. Key names, timeout and session pinning travel with
    the planner as well: kimi is Moonshot, not AIDP. The host does not: a
    missing ``L3_INSPECT_BASE_URL`` used to fall through to api.openai.com
    and time out, so it is required.
    """
    name = planner_name(env)
    planner = PLANNERS[name]
    hard_timeout_s = _optional_float(
        env, "L3_INSPECT_HARD_TIMEOUT_S", _DEFAULT_HARD_TIMEOUT_S
    )
    if not 0 <= hard_timeout_s < float("inf"):
        raise InfrastructureFailure(
            "L3_INSPECT_HARD_TIMEOUT_S must be finite and nonnegative; "
            "use 0 to disable it."
        )
    timeout_default = (
        planner.timeout_s if planner.timeout_s is not None else _DEFAULT_TIMEOUT_S
    )
    reasoning_default = planner.reasoning_effort or _DEFAULT_REASONING_EFFORT
    return {
        "planner": name,
        "model": _optional_str(env, "L3_INSPECT_MODEL", planner.model),
        "azure_endpoint": _required_base_url(env, planner),
        "api_version": _optional_str(env, "L3_INSPECT_API_VERSION", planner.api_version),
        "api_key_env": _optional_str(
            env, "L3_INSPECT_API_KEY_ENV", planner.api_key_env or _DEFAULT_KEY_ENV
        ),
        "timeout_s": _optional_float(env, "L3_INSPECT_TIMEOUT_S", timeout_default),
        "hard_timeout_s": hard_timeout_s,
        "api_style": _optional_str(
            env, "L3_INSPECT_API_STYLE", planner.api_style
        ).lower(),
        "reasoning_effort": _reasoning_effort_for(
            name,
            _optional_str(env, "L3_INSPECT_REASONING_EFFORT", reasoning_default),
        ),
        "provider": planner.provider,
        "pin_session": planner.pin_session,
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


def _retry_delay_seconds(
    attempt: int,
    error: InfrastructureFailure,
    *,
    jitter: Callable[[], float] = random.random,
) -> float:
    """How long to wait before the next attempt, spread across callers.

    A sweep starts every rollout at once, so their first planner calls -- and,
    on a fixed backoff, every retry after them -- arrive in the same second and
    draw the same rate limit again. The delay is jittered across one backoff
    step so a burst that was rejected together comes back spread out.
    """
    if error.retry_after_s is not None and error.retry_after_s > 0:
        return float(error.retry_after_s)
    step = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
    return step * (1.0 + jitter())


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


# AIDP wraps its business error codes in HTTP 400, so the status alone reads as
# a malformed request that will never succeed. -4003 is "downstream parameter
# error", which the vendor documents as recoverable by retry -- it is what a
# PTU/PayGo switch behind the gateway looks like. Named here because a fatal
# verdict throws the whole rollout attempt away, cold start included.
_RETRYABLE_GATEWAY_CODES = frozenset({"-4003"})


def _gateway_code_is_retryable(error: Exception) -> bool:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    detail = body.get("error", body)
    if not isinstance(detail, dict):
        return False
    return str(detail.get("code", "")).strip() in _RETRYABLE_GATEWAY_CODES


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
                retryable=_gateway_code_is_retryable(error),
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
    the agent keeps its chat ``messages`` plumbing. ``L3_INSPECT_API_STYLE=chat``
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
        provider: str = "azure",
    ) -> None:
        self.model = model
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.provider = provider
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
        the Nth name in ``L3_INSPECT_API_KEY_ENV``.
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
        self._left_the_account_behind()
        print(
            f"[L3 inspect] {was} is rate limited; continuing on {self.api_key_env}",
            flush=True,
        )
        return True

    def _left_the_account_behind(self) -> None:
        """Drop the state the key that just went away owned.

        Reasoning items are encrypted against the resource that produced them
        and the sticky session names a resource on that same account, so under
        the next key the provider rejects both as bad input items -- an HTTP
        400 whose -4003 points at ``input[N]`` -- instead of answering. The
        dialogue itself is untouched; only the reasoning chain restarts.
        """
        self.reasoning_store.clear()
        if self.session_id:
            self.session_id = uuid.uuid4().hex

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
        self._left_the_account_behind()
        print(
            f"[L3 inspect] {retired} cannot serve {self.model}; "
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
        max_retries = _optional_int(env, "L3_INSPECT_MAX_RETRIES", _DEFAULT_MAX_RETRIES)
        keep_all_images = _optional_bool(env, "L3_INSPECT_KEEP_ALL_IMAGES", True)
        api_style = str(config["api_style"])
        pin_session = bool(config["pin_session"])
        if not pin_session:
            # Moonshot caches prefixes itself; AIDP sticky-session headers
            # would be unrecognized extra fields on that host.
            session_id = None
        elif session_id is None and (keep_all_images or api_style == "responses"):
            session_id = uuid.uuid4().hex
        elif not keep_all_images and api_style != "responses":
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
            provider=str(config["provider"]),
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
            "provider": self.provider,
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
            if self.provider == "openai":
                from openai import OpenAI

                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.azure_endpoint,
                    max_retries=0,
                    timeout=self.timeout_s,
                )
            else:
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                    max_retries=0,
                    timeout=self.timeout_s,
                )
        except ImportError:
            raise InfrastructureFailure("openai package is not installed") from None
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
            response = client.responses.create(**request)
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


def _format_bound(value: float) -> str:
    """Format one finite action bound compactly for the model-facing schema."""
    return f"{float(value):.4g}"


def _tool_schemas(
    action_spec: RoboDojoActionSpec,
    *,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    bounds = ", ".join(
        f"{label}: [{_format_bound(low)}, {_format_bound(high)}]"
        for label, low, high in zip(
            action_spec.labels, action_spec.low, action_spec.high, strict=True
        )
    )
    tools: list[dict[str, Any]] = [
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
    ]
    # Default: no "done". RoboDojo ends the episode itself when its reward
    # fires; declaring finished early used to forfeit otherwise-scorable runs.
    # Opt in with L3_INSPECT_ALLOW_DONE=1 for probe arms that need an explicit
    # terminal call.
    if _optional_bool(env or {}, "L3_INSPECT_ALLOW_DONE", False):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "done",
                    "description": (
                        "End the episode because the task is complete from "
                        "your point of view."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "hindsight": {"type": "string"},
                        },
                        "required": ["summary"],
                    },
                },
            }
        )
    # Opt out with L3_INSPECT_DISABLE_GIVE_UP=1 so the model must keep acting
    # until the env step budget or reward ends the episode.
    if not _optional_bool(env or {}, "L3_INSPECT_DISABLE_GIVE_UP", False):
        tools.append(
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
            }
        )
    return tools


def use_task_recipe(env: Mapping[str, str] | None = None) -> bool:
    """Whether to append recipes/<task>.md when that file exists.

    Default on. ``L3_INSPECT_USE_RECIPE=0`` keeps the files on disk but omits
    ``TASK RECIPE:`` from the Goal turn.
    """
    return _optional_bool(env or {}, "L3_INSPECT_USE_RECIPE", True)


def _general_pickup_bimanual_enabled(env: Mapping[str, str]) -> bool:
    return _optional_bool(env, "L3_INSPECT_GENERAL_PICKUP_BIMANUAL", False)


def _general_pickup_bimanual_lift_enabled(env: Mapping[str, str]) -> bool:
    """Dual-arm lift skill that forbids jaw-grasp pickup."""
    return _optional_bool(env, "L3_INSPECT_GENERAL_PICKUP_BIMANUAL_LIFT", False)


def _goal_content(
    *,
    instruction: str | None,
    task_name: str | None,
    env: Mapping[str, str],
) -> str:
    goal = f"Goal: {instruction or ''}"
    if task_name == "press_by_number":
        goal += (
            " After completing each red button, press the blue confirmation "
            "button immediately; do not finish both reds before pressing blue."
        )
    if task_name == "general_pickup" and _general_pickup_bimanual_lift_enabled(env):
        goal += (
            " Lift the target with both arms by pressing from opposite sides; "
            "do not pick it up by grasping with a gripper jaw. Keep the "
            "grippers from closing onto the object as the lift method."
        )
    elif task_name == "general_pickup" and _general_pickup_bimanual_enabled(env):
        goal += (
            " Pick with both arms together: first close both grippers into "
            "fists, then press the two closed grippers onto opposite sides of "
            "the target object and lift it with both arms. Do not grasp with "
            "a single hand."
        )
    if not task_name or not use_task_recipe(env):
        return goal
    recipe = task_recipe(task_name)
    if recipe is None:
        return goal
    _, text = recipe
    if task_name == "general_pickup" and _general_pickup_bimanual_lift_enabled(env):
        text = (
            text.rstrip()
            + "\n\n## Notes\n\n"
            "Raise the official target with both arms. Press from opposite "
            "sides and lift together. Do not use a gripper-jaw grasp as the "
            "pickup method: do not close a gripper onto the object to pinch "
            "and carry it.\n"
        )
    elif task_name == "general_pickup" and _general_pickup_bimanual_enabled(env):
        text = (
            text.rstrip()
            + "\n\n## Notes\n\n"
            "Use a bimanual press-lift, not a one-handed grasp. Close both "
            "grippers first, seat them on opposite faces of the object, and "
            "raise both arms together so the object is pinched between the "
            "two fists.\n"
        )
    return f"{goal}\n\nTASK RECIPE:\n{text.rstrip()}"


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


def vision_flip_array(
    image: np.ndarray, env: Mapping[str, str] | None = None
) -> np.ndarray:
    """Apply the model-facing camera flips configured in ``env``.

    Eval MP4s stay upright; only the JPEG handed to the planner (and any
    recorder that opts into this helper) mirrors what the model saw.
    """
    array = np.asarray(image)
    if _optional_bool(env or {}, "L3_INSPECT_FLIP_VISION_UD", False):
        array = np.flipud(array)
    if _optional_bool(env or {}, "L3_INSPECT_FLIP_VISION_LR", False):
        array = np.fliplr(array)
    return array


def masked_camera_names(env: Mapping[str, str] | None = None) -> frozenset[str]:
    """Cameras whose model-facing image should be blanked.

    ``L3_INSPECT_MASK_CAMERAS`` is a comma list. ``head`` and ``cam_head``
    match each other; same for the wrist aliases.
    """
    raw = _optional_str(env or {}, "L3_INSPECT_MASK_CAMERAS", "").strip()
    names: set[str] = set()
    for part in raw.split(","):
        token = part.strip().lower()
        if not token:
            continue
        names.add(token)
        if token.startswith("cam_"):
            names.add(token[len("cam_") :])
        else:
            names.add(f"cam_{token}")
    return frozenset(names)


def camera_is_masked(name: str, env: Mapping[str, str] | None = None) -> bool:
    return str(name).strip().lower() in masked_camera_names(env)


def vision_model_view_array(
    image: np.ndarray,
    name: str,
    env: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Flips plus optional per-camera black mask for the planner JPEG."""
    array = vision_flip_array(image, env)
    if camera_is_masked(name, env):
        return np.zeros(array.shape, dtype=np.uint8)
    return np.asarray(array)


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


_ICL_MESSAGE_PREFIX = "IN-CONTEXT DEMONSTRATION"
_DEFAULT_ICL_TEXT_FIELD = "with_both_modifiers"


def _icl_text_root(env: Mapping[str, str]) -> str:
    return _optional_str(env, "L3_INSPECT_ICL_TEXT_ROOT", "").strip()


def _icl_image_root(env: Mapping[str, str]) -> str:
    return _optional_str(env, "L3_INSPECT_ICL_ROOT", "").strip()


def _icl_mode(env: Mapping[str, str]) -> str | None:
    """Which ICL channel is active. Text wins when both roots are set."""
    if _icl_text_root(env):
        return "text"
    if _icl_image_root(env):
        return "image"
    return None


def _icl_enabled_for_task(task_name: str, env: Mapping[str, str]) -> bool:
    if _icl_mode(env) is None:
        return False
    configured = _optional_str(env, "L3_INSPECT_ICL_TASKS", "").strip()
    if not configured:
        return True
    tasks = {item.strip() for item in configured.split(",") if item.strip()}
    return task_name in tasks


def _icl_text_message(task_name: str, env: Mapping[str, str]) -> dict[str, Any] | None:
    """Load a layout-generic textual demonstration; no images, no action vectors."""
    if _icl_mode(env) != "text" or not _icl_enabled_for_task(task_name, env):
        return None
    root = Path(_icl_text_root(env)).expanduser()
    episode = _optional_int(env, "L3_INSPECT_ICL_TEXT_EPISODE", 0)
    path = root / f"{task_name}_ep{episode:07d}.json"
    if not path.is_file():
        raise InfrastructureFailure(f"ICL text file does not exist: {path}")
    field = (
        _optional_str(env, "L3_INSPECT_ICL_TEXT_FIELD", _DEFAULT_ICL_TEXT_FIELD).strip()
        or _DEFAULT_ICL_TEXT_FIELD
    )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        description = record["annotation"]["descriptions"][field]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise InfrastructureFailure(f"invalid ICL text file {path}: {error}") from error
    if not isinstance(description, str) or not description.strip():
        raise InfrastructureFailure(f"ICL text field {field!r} is empty in {path}")
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"{_ICL_MESSAGE_PREFIX}\n"
                    f"Task: {task_name}\n"
                    "A layout-generic textual expert demonstration "
                    "(no images, no action vectors):\n"
                    f"{description.strip()}"
                ),
            }
        ],
    }


def _format_icl_group_row(group: Any, index: int) -> str:
    rows = []
    for name in sorted(group.keys()):
        if name.endswith("_delta_ee_poses"):
            continue
        values = np.asarray(group[name][index]).reshape(-1)
        rendered = ", ".join(f"{float(value):.6f}" for value in values)
        rows.append(f"{name}=[{rendered}]")
    return "\n".join(rows)


IclFrameFormatter = Callable[[Any, Any, int, int], str]


def _format_icl_state_action(
    state: Any, action: Any, index: int, frame_count: int
) -> str:
    del frame_count
    return (
        "Expert state:\n"
        f"{_format_icl_group_row(state, index)}\n"
        "Expert action:\n"
        f"{_format_icl_group_row(action, index)}"
    )


def _icl_message(
    task_name: str,
    env: Mapping[str, str],
    *,
    frame_formatter: IclFrameFormatter = _format_icl_state_action,
    description: str = (
        "These chronological expert keyframes show the head-camera observation "
        "and the corresponding full robot state/action. Use them as a "
        "task-specific reference, not as the current state."
    ),
) -> dict[str, Any] | None:
    """Load task-keyed head-camera ICL keyframes without decoding RGB JPEGs."""
    if _icl_mode(env) == "text":
        return _icl_text_message(task_name, env)
    if not _icl_enabled_for_task(task_name, env):
        return None
    root = Path(_icl_image_root(env)).expanduser()
    path = root / f"{task_name}.hdf5"
    if not path.is_file():
        raise InfrastructureFailure(f"ICL file does not exist: {path}")
    try:
        import h5py
    except ImportError as error:
        raise InfrastructureFailure(
            "L3_INSPECT_ICL_ROOT is set but h5py is not installed"
        ) from error

    camera = _optional_str(env, "L3_INSPECT_ICL_CAMERA", "cam_head").strip()
    parts: list[dict[str, Any]] = []
    try:
        with h5py.File(path, "r") as episode:
            colors = episode[f"vision/{camera}/colors"]
            state = episode["state"]
            action = episode["action"]
            source_frames = list(
                np.asarray(
                    episode.attrs.get("gpt_icl_source_frames", np.arange(len(colors)))
                ).reshape(-1)
            )
            if not (len(colors) == len(source_frames)):
                raise ValueError(
                    "camera frame count does not match gpt_icl_source_frames"
                )
            instruction = episode["instruction"][()]
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")
            parts.append(
                {
                    "type": "text",
                    "text": (
                        f"{_ICL_MESSAGE_PREFIX}\n"
                        f"Task: {task_name}\n"
                        f"Instruction: {instruction}\n"
                        f"{description}"
                    ),
                }
            )
            for index, source_frame in enumerate(source_frames):
                raw = bytes(colors[index])
                if not raw.startswith(b"\xff\xd8"):
                    raise ValueError(f"{camera} keyframe {index} is not JPEG")
                parts.extend(
                    [
                        {
                            "type": "text",
                            "text": (
                                f"Expert keyframe {index + 1}/{len(colors)} "
                                f"(source frame {int(source_frame)}, camera '{camera}'):"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(raw).decode("ascii")
                            },
                        },
                        {
                            "type": "text",
                            "text": frame_formatter(
                                state, action, index, len(colors)
                            ),
                        },
                    ]
                )
    except (KeyError, OSError, ValueError, UnicodeError) as error:
        raise InfrastructureFailure(f"invalid ICL file {path}: {error}") from error
    return {"role": "user", "content": parts}


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


def _is_protected_history_turn(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    if not isinstance(first, Mapping):
        return False
    text = str(first.get("text") or "")
    return text.startswith(_ICL_MESSAGE_PREFIX) or text.startswith(
        _WATCH_MESSAGE_PREFIX
    )


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
        and not _is_protected_history_turn(message)
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
    _default_max_llm_calls = _DEFAULT_MAX_LLM_CALLS

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
            self._env, "L3_INSPECT_MAX_LLM_CALLS", self._default_max_llm_calls
        )
        self._defer_llm_until_env_step = _optional_absent_int(
            self._env, "L3_INSPECT_DEFER_LLM_UNTIL_ENV_STEP"
        )
        if (
            self._defer_llm_until_env_step is not None
            and self._defer_llm_until_env_step < 0
        ):
            raise InfrastructureFailure(
                "L3_INSPECT_DEFER_LLM_UNTIL_ENV_STEP must be >= 0"
            )
        self._defer_llm_tasks = {
            item.strip()
            for item in _optional_str(
                self._env, "L3_INSPECT_DEFER_LLM_TASKS", ""
            ).split(",")
            if item.strip()
        }
        self._defer_llm_sample_every = _optional_absent_int(
            self._env, "L3_INSPECT_DEFER_LLM_SAMPLE_EVERY_ENV_STEP"
        )
        if (
            self._defer_llm_sample_every is not None
            and self._defer_llm_sample_every < 0
        ):
            raise InfrastructureFailure(
                "L3_INSPECT_DEFER_LLM_SAMPLE_EVERY_ENV_STEP must be >= 0"
            )
        self._keep_all_images = _optional_bool(
            self._env, "L3_INSPECT_KEEP_ALL_IMAGES", True
        )
        self._image_horizon = _optional_int(
            self._env, "L3_INSPECT_IMAGE_HORIZON", _DEFAULT_IMAGE_HORIZON
        )
        if not self._keep_all_images and self._image_horizon < 1:
            raise InfrastructureFailure(
                f"L3_INSPECT_IMAGE_HORIZON={self._image_horizon} would stub every "
                "camera image out of the conversation; set it to 1 or more, or "
                f"unset it to use {_DEFAULT_IMAGE_HORIZON}."
            )
        self._depth = _optional_str(self._env, "L3_INSPECT_DEPTH", "off").lower()
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
        return _tool_schemas(action_spec, env=self._env)

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
            if self._task_name:
                demonstration = self._icl_demonstration(self._task_name)
                if demonstration is not None:
                    self._messages.append(demonstration)
            self._append_prior_attempt()
            self._goal_sent = True

    def _icl_demonstration(self, task_name: str) -> dict[str, Any] | None:
        return _icl_message(task_name, self._env)

    def _append_prior_attempt(self) -> None:
        """Replay an earlier attempt's conversation, when one is configured."""
        path = self._prior_transcript_path()
        if not path:
            return
        note = _optional_str(self._env, "L3_INSPECT_PRIOR_NOTE", "") or None
        try:
            messages = load_prior_messages(
                path, render_state=self._render_prior_state, note=note
            )
        except ValueError as error:
            raise InfrastructureFailure(str(error)) from error
        self._messages.extend(messages)

    def _prior_transcript_path(self) -> str:
        return _optional_str(self._env, "L3_INSPECT_PRIOR_TRANSCRIPT", "").strip()

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
        deferred = self._deferred_llm_chunk(observation)
        if deferred is not None:
            return deferred
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
            # "done" is refused unless L3_INSPECT_ALLOW_DONE opted it into the
            # tool list: a hallucinated finish would forfeit a run the episode
            # itself would otherwise have been allowed to complete.
            if name == "give_up" or (
                name == "done"
                and _optional_bool(self._env, "L3_INSPECT_ALLOW_DONE", False)
            ):
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

    def _defer_plan(self) -> tuple[int, int]:
        return _defer_llm_plan(
            task_name=self._task_name,
            control_hz=self.action_spec.control_hz,
            env_until=self._defer_llm_until_env_step,
            env_tasks=self._defer_llm_tasks,
            env_sample_every=self._defer_llm_sample_every,
        )

    def _deferred_llm_chunk(self, observation: Observation) -> ActionChunk | None:
        until_env_step, sample_every = self._defer_plan()
        if until_env_step <= 0:
            return None
        current_env_step = observation.extra.get("env_step")
        if not isinstance(current_env_step, int):
            return None
        wait_steps = until_env_step - current_env_step
        if wait_steps <= 0:
            return None
        chunk_steps = wait_steps
        if sample_every > 0:
            chunk_steps = min(sample_every, wait_steps)
            self._messages.append(
                self._observation_message(
                    observation,
                    watch_until_env_step=until_env_step,
                )
            )
        hold = _state_dict_from_vector(_vector_from_state(observation.state))
        return ActionChunk(
            actions=[Action(data=hold) for _ in range(chunk_steps)],
            control_hz=self.action_spec.control_hz,
            meta={
                "trace": {
                    "tool": "defer_llm",
                    "arguments": {
                        "from_env_step": current_env_step,
                        "until_env_step": until_env_step,
                        "sample_every_env_step": sample_every,
                    },
                    "planned_waypoints": chunk_steps,
                }
            },
        )

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
            "adapter": "robodojo-agent-l3-inspect",
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
                "icl_root": _optional_str(self._env, "L3_INSPECT_ICL_ROOT", "") or None,
                "icl_text_root": _icl_text_root(self._env) or None,
                "icl_text_field": _optional_str(
                    self._env, "L3_INSPECT_ICL_TEXT_FIELD", _DEFAULT_ICL_TEXT_FIELD
                )
                or _DEFAULT_ICL_TEXT_FIELD,
                "icl_mode": _icl_mode(self._env),
                "icl_tasks": _optional_str(self._env, "L3_INSPECT_ICL_TASKS", "") or None,
                "icl_camera": _optional_str(
                    self._env, "L3_INSPECT_ICL_CAMERA", "cam_head"
                ),
                "defer_llm_until_env_step": self._defer_plan()[0],
                "defer_llm_sample_every_env_step": self._defer_plan()[1],
                "defer_llm_tasks": sorted(self._defer_llm_tasks),
                "task": self._task_name,
                "keep_all_images": self._keep_all_images,
                "cache_session_id": self._session_id,
                "image_horizon": self._image_horizon,
                "hard_timeout_s": client["hard_timeout_s"],
                "api_style": client["api_style"],
                "reasoning_effort": _optional_str(
                    self._env, "L3_INSPECT_REASONING_EFFORT", _DEFAULT_REASONING_EFFORT
                ),
                "flip_vision_ud": _optional_bool(
                    self._env, "L3_INSPECT_FLIP_VISION_UD", False
                ),
                "flip_vision_lr": _optional_bool(
                    self._env, "L3_INSPECT_FLIP_VISION_LR", False
                ),
                "mask_cameras": sorted(masked_camera_names(self._env)),
                "general_pickup_bimanual": _general_pickup_bimanual_enabled(
                    self._env
                ),
                "general_pickup_bimanual_lift": _general_pickup_bimanual_lift_enabled(
                    self._env
                ),
                "allow_done": _optional_bool(
                    self._env, "L3_INSPECT_ALLOW_DONE", False
                ),
                "disable_give_up": _optional_bool(
                    self._env, "L3_INSPECT_DISABLE_GIVE_UP", False
                ),
                "env_step_limit_override": _optional_int(
                    self._env, "L3_INSPECT_ENV_STEP_LIMIT", 0
                )
                or None,
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

    def _observation_message(
        self,
        observation: Observation,
        *,
        watch_until_env_step: int | None = None,
    ) -> dict[str, Any]:
        text = self._state_block(observation)
        if watch_until_env_step is not None:
            env_step = observation.extra.get("env_step")
            text = (
                f"{_WATCH_MESSAGE_PREFIX}\n"
                "Hold still. This is a sampled frame from the opposite arm's "
                f"demonstration (env step {env_step} of {watch_until_env_step}). "
                "Do not move and do not give_up; write down which object was "
                "just placed.\n"
                f"{text}"
            )
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if observation.remaining_steps is not None:
            # The step limit is the budget that actually ends most episodes,
            # and until it is reported the model has no way to know it exists.
            parts[0]["text"] += (
                f"\nEnv steps remaining before the episode ends: "
                f"{observation.remaining_steps}"
            )
        for name, image in observation.images.items():
            array = vision_model_view_array(image, name, self._env)
            label = f"camera '{name}' (step {observation.step})"
            if camera_is_masked(name, self._env):
                label += ", MASKED"
            parts.append({"type": "text", "text": f"{label}:"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_jpeg_data_uri(array)},
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
