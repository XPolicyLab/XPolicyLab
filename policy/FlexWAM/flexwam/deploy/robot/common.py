"""Shared helpers for robot deployment profiles."""
from __future__ import annotations

from typing import Any, Mapping


CLIENT_KEY_FIELDS = ("state", "prompt", "action", "cameras")


def derive_router_display_order(camera_order: tuple[str, ...]) -> tuple[str, ...]:
    wrist_views = [name for name in camera_order if "wrist" in name]
    other_views = [name for name in camera_order if "wrist" not in name]
    return tuple(wrist_views + other_views)


def require_client_keys(robot: Any) -> dict[str, Any]:
    robot_name = getattr(robot, "name", type(robot).__name__)
    keys = getattr(robot, "client_keys", None)
    if not isinstance(keys, Mapping):
        raise ValueError(
            f"Robot {robot_name!r} must define client_keys mapping model fields to client "
            "websocket keys: {{'state': ..., 'prompt': ..., 'action': ..., "
            "'cameras': {{model_name: client_key, ...}}}}."
        )
    missing = [field for field in CLIENT_KEY_FIELDS if field not in keys]
    if missing:
        raise ValueError(f"Robot {robot_name!r} client_keys missing fields: {missing}.")
    cameras = keys["cameras"]
    if not isinstance(cameras, Mapping) or not cameras:
        raise ValueError(
            f"Robot {robot_name!r} client_keys['cameras'] must map each model camera name "
            "to the client observation key."
        )
    return {
        "state": str(keys["state"]),
        "prompt": str(keys["prompt"]),
        "action": str(keys["action"]),
        "cameras": {str(name): str(client_key) for name, client_key in cameras.items()},
    }


def client_field_key(robot: Any, field: str) -> str:
    if field == "cameras":
        raise TypeError("Use camera_client_key() for camera mappings.")
    return str(require_client_keys(robot)[field])


def camera_client_key(robot: Any, model_name: str) -> str:
    cameras = require_client_keys(robot)["cameras"]
    if model_name not in cameras:
        known = ", ".join(cameras) or "<none>"
        raise KeyError(
            f"Robot {getattr(robot, 'name', type(robot).__name__)!r} has no client key for "
            f"model camera {model_name!r}. Add it to client_keys['cameras']. Known: {known}."
        )
    return cameras[model_name]


def lookup_client_value(payload: Mapping[str, Any], key: str) -> Any:
    """Look up the configured client key. Dotted keys nest only if the literal key is absent."""
    if key in payload:
        return payload[key]
    if "." not in key:
        return None
    current: Any = payload
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current
