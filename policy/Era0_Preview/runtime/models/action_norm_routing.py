from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACTION_NORM_ROUTING_VERSION = "v1"
DEFAULT_DEPLOY_PROFILE = "default"
GLOBAL_ACTION_NORM_KEY = "__global__"


def normalize_bank_keys(bank_keys: Any) -> list[str]:
    if bank_keys is None:
        return [GLOBAL_ACTION_NORM_KEY]
    if isinstance(bank_keys, (tuple, list)):
        keys = [str(key).strip() for key in bank_keys if str(key).strip()]
    else:
        keys = [str(bank_keys).strip()]
    if not keys:
        keys = [GLOBAL_ACTION_NORM_KEY]
    return keys


def load_deploy_profile_map(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"deploy profile map must be a JSON object: {path}")
    return payload


def normalize_action_norm_routing(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    mode = str(payload.get("mode", "")).strip().lower().replace("-", "_")
    version = str(payload.get("version", ACTION_NORM_ROUTING_VERSION)).strip() or ACTION_NORM_ROUTING_VERSION
    profiles_payload = payload.get("profiles")
    if not isinstance(profiles_payload, dict):
        direct_profiles = {
            key: value
            for key, value in payload.items()
            if key not in {"version", "mode", "requires_explicit_mapping"}
        }
        profiles_payload = direct_profiles if direct_profiles else None
    profiles: dict[str, dict[str, str]] = {}
    if isinstance(profiles_payload, dict):
        for profile_name, value in profiles_payload.items():
            name = str(profile_name).strip()
            if not name:
                continue
            if isinstance(value, str):
                action_norm_key = value.strip()
                description = ""
            elif isinstance(value, dict):
                action_norm_key = str(value.get("action_norm_key", "")).strip()
                description = str(value.get("description", "")).strip()
            else:
                continue
            if not action_norm_key:
                continue
            profiles[name] = {
                "action_norm_key": action_norm_key,
                "description": description,
            }
    if mode not in {"single_bank", "multi_bank"}:
        if profiles:
            mode = "multi_bank" if len(profiles) > 1 or DEFAULT_DEPLOY_PROFILE not in profiles else "single_bank"
        else:
            mode = ""
    out = {
        "version": version,
        "mode": mode,
        "profiles": profiles,
    }
    if "requires_explicit_mapping" in payload:
        out["requires_explicit_mapping"] = bool(payload.get("requires_explicit_mapping"))
    return out


def build_action_norm_routing(
    *,
    bank_keys: Any,
    raw_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    keys = normalize_bank_keys(bank_keys)
    normalized = normalize_action_norm_routing(raw_mapping) if raw_mapping else None
    if len(keys) <= 1:
        action_norm_key = keys[0]
        profiles: dict[str, dict[str, str]] = {
            DEFAULT_DEPLOY_PROFILE: {
                "action_norm_key": action_norm_key,
                "description": "single-bank checkpoint",
            }
        }
        if normalized and normalized.get("profiles"):
            for profile_name, payload in normalized["profiles"].items():
                resolved_key = str(payload.get("action_norm_key", "")).strip()
                if resolved_key != action_norm_key:
                    continue
                profiles[profile_name] = {
                    "action_norm_key": action_norm_key,
                    "description": str(payload.get("description", "")).strip(),
                }
        for profile_name, payload in profiles.items():
            resolved_key = str(payload.get("action_norm_key", "")).strip()
            if resolved_key not in keys:
                raise ValueError(
                    f"deploy profile '{profile_name}' points to unknown action_norm_key={resolved_key}. "
                    f"Available keys: {keys}"
                )
        return {
            "version": ACTION_NORM_ROUTING_VERSION,
            "mode": "single_bank",
            "profiles": profiles,
        }

    profiles: dict[str, dict[str, str]] = {}
    if normalized and normalized.get("profiles"):
        for profile_name, payload in normalized["profiles"].items():
            action_norm_key = str(payload.get("action_norm_key", "")).strip()
            if action_norm_key not in keys:
                raise ValueError(
                    f"deploy profile '{profile_name}' points to unknown action_norm_key={action_norm_key}. "
                    f"Available keys: {keys}"
                )
            profiles[profile_name] = {
                "action_norm_key": action_norm_key,
                "description": str(payload.get("description", "")).strip(),
            }
    if len(profiles) == 0:
        # Auto-register each bank_key as its own profile name (identity mapping).
        # Any multi-bank ckpt is directly deployable via deploy_profile=<bank_key>.
        # `target=state` banks normalize proprio only — they are never an action
        # target, so they must not be offered as deployable profiles.
        action_keys = [k for k in keys if "|target=state" not in k]
        profiles = {
            k: {"action_norm_key": k, "description": "auto"} for k in (action_keys or keys)
        }
    return {
        "version": ACTION_NORM_ROUTING_VERSION,
        "mode": "multi_bank",
        "profiles": profiles,
    }


def resolve_action_norm_key(
    *,
    routing: dict[str, Any] | None,
    bank_keys: Any,
    deploy_profile: Any | None,
) -> tuple[str | None, dict[str, Any]]:
    keys = normalize_bank_keys(bank_keys)
    resolved = build_action_norm_routing(bank_keys=keys, raw_mapping=routing)
    profiles = resolved.get("profiles", {})
    mode = resolved.get("mode", "")

    if mode == "single_bank" or len(keys) <= 1:
        profile_name = str(deploy_profile or "").strip()
        if profile_name:
            if profile_name not in profiles:
                available = sorted(profiles.keys())
                raise ValueError(
                    f"Unknown deploy_profile={profile_name}. Available profiles: {available}"
                )
            action_norm_key = str(profiles[profile_name]["action_norm_key"])
        else:
            action_norm_key = str(profiles.get(DEFAULT_DEPLOY_PROFILE, {}).get("action_norm_key", keys[0]))
        return action_norm_key, resolved

    profile_name = str(deploy_profile or "").strip()
    if not profile_name:
        available = sorted(profiles.keys())
        raise ValueError(
            f"deploy_profile is required for multi-bank checkpoints. Available profiles: {available}"
        )
    if profile_name not in profiles:
        available = sorted(profiles.keys())
        raise ValueError(
            f"Unknown deploy_profile={profile_name}. Available profiles: {available}"
        )
    action_norm_key = str(profiles[profile_name]["action_norm_key"])
    if action_norm_key not in keys:
        raise ValueError(
            f"deploy_profile={profile_name} resolved to unknown action_norm_key={action_norm_key}."
        )
    return action_norm_key, resolved


def available_deploy_profiles(routing: dict[str, Any] | None, bank_keys: Any) -> list[str]:
    resolved = build_action_norm_routing(bank_keys=bank_keys, raw_mapping=routing)
    return sorted(str(key) for key in resolved.get("profiles", {}).keys())
