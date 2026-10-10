# SPDX-License-Identifier: Apache-2.0

"""Backend for RoboDojo/XPolicyLab's binary WebSocket policy protocol."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from .base import Backend, BackendConnection
from .robodojo import RoboDojoFrame, pack_frame, unpack_frame

logger = logging.getLogger(__name__)


class XPolicyLabWsBackendConnection(BackendConnection):
    def __init__(
        self,
        ws,
        *,
        evaluation_id: str,
        trial_id: str,
        timeout_s: float = 900.0,
    ):
        self._ws = ws
        self._evaluation_id = evaluation_id
        self._trial_id = trial_id
        self._timeout_s = timeout_s

    async def recv_metadata(self) -> dict:
        response = await self._request("hello", {})
        return dict(response.payload or {})

    async def infer(self, canonical_obs: dict) -> dict:
        observation = _canonical_to_xpolicylab(canonical_obs)
        response = await self._request(
            "infer",
            {"observation": observation},
            step=_step_from_observation(canonical_obs),
        )
        return _normalize_xpolicylab_response(dict(response.payload or {}))

    async def infer_batch(self, canonical_obs: list[dict]) -> list[dict]:
        if not canonical_obs:
            return []
        observations = [_canonical_to_xpolicylab(obs) for obs in canonical_obs]
        response = await self._request(
            "call",
            {"func_name": "infer_batch", "obs": observations},
            step=max(_step_from_observation(obs) for obs in canonical_obs),
        )
        payload = dict((response.payload or {}).get("result") or {})
        actions = payload.get("actions")
        if not isinstance(actions, list) or len(actions) != len(canonical_obs):
            raise ValueError(
                "XPolicyLab batch response length mismatch: "
                f"expected {len(canonical_obs)}, got "
                f"{len(actions) if isinstance(actions, list) else type(actions).__name__}"
            )
        overlays = payload.get("video_overlay")
        if not isinstance(overlays, list):
            overlays = [None] * len(actions)
        if len(overlays) != len(actions):
            raise ValueError(
                "XPolicyLab batch video_overlay length mismatch: "
                f"expected {len(actions)}, got {len(overlays)}"
            )
        return [
            _normalize_xpolicylab_response(
                {"actions": action, "video_overlay": overlays[index]}
            )
            for index, action in enumerate(actions)
        ]

    async def close(self) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.close()
        finally:
            self._ws = None

    async def _request(
        self,
        message_type: str,
        payload: dict[str, Any],
        *,
        step: int = 0,
    ) -> RoboDojoFrame:
        if self._ws is None:
            raise RuntimeError("XPolicyLab backend connection is closed")
        request = RoboDojoFrame(
            message_type=message_type,
            message_id=str(uuid4()),
            evaluation_id=self._evaluation_id,
            trial_id=self._trial_id,
            step=step,
            payload=payload,
        )
        await self._ws.send(pack_frame(request.to_wire()))
        raw = await asyncio.wait_for(self._ws.recv(), self._timeout_s)
        if isinstance(raw, str):
            raise ValueError("XPolicyLab backend returned a text WebSocket frame")
        response = RoboDojoFrame.from_wire(unpack_frame(raw))
        if response.message_id != request.message_id:
            raise ValueError(
                "XPolicyLab response message_id does not match request"
            )
        if response.message_type == "error":
            error = response.payload or {}
            raise RuntimeError(
                f"XPolicyLab {error.get('code', 'error')}: "
                f"{error.get('message', 'policy server error')}"
            )
        expected = {
            "hello": "hello_ack",
            "infer": "infer_result",
            "call": "call_result",
        }.get(message_type)
        if response.message_type != expected:
            raise ValueError(
                f"expected XPolicyLab {expected}, got {response.message_type}"
            )
        return response


class XPolicyLabWsBackend(Backend):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        evaluation_id: str = "echopolicy",
        trial_id: str = "echopolicy-trial",
        timeout_s: float = 900.0,
    ):
        self._host = host
        self._port = port
        self._evaluation_id = evaluation_id
        self._trial_id = trial_id
        self._timeout_s = timeout_s

    async def connect(self) -> BackendConnection:
        try:
            from websockets.asyncio.client import connect
        except ModuleNotFoundError:  # websockets < 13 (for example RoboDojo's env)
            from websockets.legacy.client import connect

        uri = f"ws://{self._host}:{self._port}"
        ws = await connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=None,
        )
        logger.info("XPolicyLabWsBackend connected to %s", uri)
        # One backend connection belongs to exactly one frontend WebSocket.
        # A static trial id aliases per-session policy state when two evaluator
        # GPUs connect concurrently.
        connection_trial_id = f"{self._trial_id}-{uuid4().hex}"
        return XPolicyLabWsBackendConnection(
            ws,
            evaluation_id=self._evaluation_id,
            trial_id=connection_trial_id,
            timeout_s=self._timeout_s,
        )


def _canonical_to_xpolicylab(observation: Mapping[str, Any]) -> dict[str, Any]:
    images = {
        "cam_high": _to_chw_uint8(observation["observation/exterior_image_1_left"]),
    }
    for name, key in (
        ("cam_left_wrist", "observation/wrist_image_left"),
        ("cam_right_wrist", "observation/wrist_image_right"),
    ):
        if observation.get(key) is not None:
            images[name] = _to_chw_uint8(observation[key])
    required_cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    missing = [name for name in required_cameras if name not in images]
    if missing:
        raise ValueError(
            "XPolicyLab observation is missing required camera(s): "
            + ", ".join(missing)
        )

    # Pi05's XPolicyLab model accepts the same encoded images as its Aloha
    # transport, but the custom envelope names the text field ``instruction``.
    result = {
        "state": _flat_state(observation),
        "images": {name: images[name] for name in required_cameras},
        # The proxy has already synchronized this with the current subgoal.
        "instruction": str(
            observation.get("instruction", observation.get("prompt", ""))
        ),
    }
    encoded_keys = {
        "state",
        "images",
        "vision",
        "instruction",
        "prompt",
        "image_masks",
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/wrist_image_right",
        "observation/depth_exterior_image_1_left",
        "observation/depth_wrist_image_left",
        "observation/depth_wrist_image_right",
        "observation/joint_position",
        "observation/gripper_position",
    }
    for key, value in observation.items():
        if (
            key in encoded_keys
            or key.startswith("__")
            or key.endswith("_raw")
        ):
            continue
        result[key] = value
    return result


def _to_chw_uint8(value: Any) -> Any:
    import numpy as np

    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"VLA image must be 3D, got {image.shape}")
    if image.shape[-1] in (1, 3, 4):
        image = np.moveaxis(image, -1, 0)
    elif image.shape[0] not in (1, 3, 4):
        raise ValueError(f"unsupported VLA image shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return image.astype(np.uint8, copy=False)


def _flat_state(observation: Mapping[str, Any]) -> Any:
    import numpy as np

    state = observation.get("state")
    if isinstance(state, Mapping):
        keys = (
            "left_arm_joint_state", "left_ee_joint_state",
            "right_arm_joint_state", "right_ee_joint_state",
        )
        if all(key in state for key in keys):
            values = [state[key] for key in keys]
        elif "arm_joint_state" in state and "ee_joint_state" in state:
            values = [state["arm_joint_state"], state["ee_joint_state"]]
        else:
            values = None
        if values is not None:
            result = np.concatenate([np.asarray(value).reshape(-1) for value in values])
        else:
            result = None
    else:
        result = None
    if result is None:
        result = np.concatenate([
            np.asarray(observation["observation/joint_position"]).reshape(-1),
            np.asarray(observation["observation/gripper_position"]).reshape(-1),
        ])
    result = result.astype(np.float32, copy=False)
    if result.shape != (14,):
        raise ValueError(f"Pi05 VLA state must have shape (14,), got {result.shape}")
    return result


def _step_from_observation(observation: Mapping[str, Any]) -> int:
    try:
        return int(observation.get("__step", 0))
    except (TypeError, ValueError):
        return 0


def _normalize_xpolicylab_response(response: dict[str, Any]) -> dict[str, Any]:
    """Adapt XPolicyLab's single-arm action names to RoboDojo's contract."""
    actions = response.get("actions")
    if isinstance(actions, Mapping):
        actions = [actions]
    if not isinstance(actions, list) or not all(
        isinstance(action, Mapping) for action in actions
    ):
        return response

    normalized = []
    for action in actions:
        item = dict(action)
        if "joint_state" in item and "arm_joint_state" not in item:
            item["arm_joint_state"] = item.pop("joint_state")
        normalized.append(item)
    response["actions"] = normalized
    return response
