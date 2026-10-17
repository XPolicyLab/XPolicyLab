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
