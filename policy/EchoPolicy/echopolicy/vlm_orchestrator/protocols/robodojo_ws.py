# SPDX-License-Identifier: Apache-2.0

"""RoboDojo WebSocket frontend adapted to the legacy Frontend interface."""

from __future__ import annotations

import http
import logging
from typing import Awaitable, Callable

from .base import Frontend, FrontendSession
from .robodojo import (
    ActionLayout,
    RoboDojoFrame,
    RoboDojoObservationAdapter,
    pack_frame,
    unpack_frame,
)

logger = logging.getLogger(__name__)


def _decode_images(observation: dict) -> None:
    # Plain RGB arrays also work when the standalone orchestrator is installed.
    # Encoded XPolicyLab images require its canonical, marker-aware decoder.
    import numpy as np

    for camera in observation.get("vision", {}).values():
        values = [camera.get(key) for key in ("color", "colors", "rgb", "image") if key in camera] if isinstance(camera, dict) else [camera]
        if any(value is not None and not (isinstance(value, np.ndarray) and value.ndim == 3) for value in values):
            from XPolicyLab.utils.process_data import decode_obs_images
            decode_obs_images(observation)
            return


def _wire_action_count(actions) -> int:
    return len(actions) if isinstance(actions, list) else 1


_RESPONSE_TYPES = {
    "hello": "hello_ack",
    "prepare_case": "prepare_case_ack",
    "reset": "reset_result",
    "call": "call_result",
    "infer": "infer_result",
    "batch_infer": "batch_infer_result",
    "feedback": "feedback_ack",
    "trial_end": "trial_end_ack",
    "heartbeat": "heartbeat_ack",
}
_REQUEST_TYPES = set(_RESPONSE_TYPES) | {"close"}
_LIFECYCLE_TYPES = {
    "hello",
    "prepare_case",
    "reset",
    "feedback",
    "trial_end",
    "heartbeat",
}


class RoboDojoWsSession(FrontendSession):
    """Translate RoboDojo frames into the original ``FrontendSession`` API.

    Lifecycle frames are consumed here. For a batch frame, observations are
    yielded one at a time to the unchanged proxy and actions are collected
    until the matching RoboDojo response can be sent.
    """

    def __init__(self, ws):
        self._ws = ws
        self._adapter = RoboDojoObservationAdapter()
        self._frame: RoboDojoFrame | None = None
        self._layout: ActionLayout | None = None
        self._episode_generation = 0
        self._batch_frame: RoboDojoFrame | None = None
        self._batch_observations: list[dict] = []
        self._batch_layouts: list[ActionLayout | None] = []
        self._batch_actions: list = []
        self._batch_index = 0
        self._call_observation: dict | None = None
        self._call_observations: list[dict] | None = None

    async def send_metadata(self, metadata: dict) -> None:
        # RoboDojo has hello/hello_ack instead of OpenPI's metadata frame.
        return None

    async def recv_obs(self) -> dict | None:
        if self._batch_observations:
            return self._batch_observations.pop(0)

        while True:
            raw = None
            try:
                raw = await self._ws.recv()
                if isinstance(raw, str):
                    raise ValueError("RoboDojo frontend only accepts binary frames")
                frame = RoboDojoFrame.from_wire(unpack_frame(raw))
                self._frame = frame
                if frame.message_type not in _REQUEST_TYPES:
                    raise ValueError(
                        f"unsupported RoboDojo request: {frame.message_type}"
                    )

                if frame.message_type in _LIFECYCLE_TYPES:
                    if frame.message_type in {"reset", "trial_end"}:
                        self._episode_generation += 1
                        self._clear_call_observations()
                    payload = {
                        "ok": True,
                        "metadata": {},
                    } if frame.message_type == "hello" else {
                        "ok": True,
                        "result": None,
                    }
                    await self._send_frame(frame, payload)
                    continue

                if frame.message_type == "call":
                    observations = await self._handle_call(frame)
                    if observations is None:
                        continue
                    if len(observations) == 1:
                        canonical, self._layout = self._convert_observation(
                            frame, observations[0]
                        )
                        return canonical
                    self._start_batch(frame, observations)
                    return self._batch_observations.pop(0)

                if frame.message_type == "close":
                    return None
                if frame.message_type == "infer":
                    observation = frame.payload.get("observation")
                    if not isinstance(observation, dict):
                        raise ValueError("infer payload missing observation map")
                    canonical, self._layout = self._convert_observation(
                        frame, observation
                    )
                    return canonical
                if frame.message_type == "batch_infer":
                    observations = frame.payload.get("observations")
                    if not isinstance(observations, list) or not observations:
                        raise ValueError(
                            "batch_infer payload missing observations list"
                        )
                    if not all(isinstance(item, dict) for item in observations):
                        raise ValueError("batch observations must be maps")
                    self._start_batch(frame, observations)
                    return self._batch_observations.pop(0)
            except Exception as exc:
                if raw is None:
                    return None
                await self.send_error(exc)

    async def recv_batch(self) -> list[dict] | None:
        """Receive a native single- or multi-environment inference frame."""
        while True:
            raw = None
            try:
                raw = await self._ws.recv()
                if isinstance(raw, str):
                    raise ValueError("RoboDojo frontend only accepts binary frames")
                frame = RoboDojoFrame.from_wire(unpack_frame(raw))
                self._frame = frame
                if frame.message_type not in _REQUEST_TYPES:
                    raise ValueError(f"unsupported RoboDojo request: {frame.message_type}")

                if frame.message_type in _LIFECYCLE_TYPES:
                    if frame.message_type in {"reset", "trial_end"}:
                        self._episode_generation += 1
                        self._clear_call_observations()
                    payload = (
                        {"ok": True, "metadata": {}}
                        if frame.message_type == "hello"
                        else {"ok": True, "result": None}
                    )
                    await self._send_frame(frame, payload)
                    continue

                if frame.message_type == "call":
                    observations = await self._handle_call(frame)
                    if observations is None:
                        continue
                    if len(observations) == 1:
                        canonical, self._layout = self._convert_observation(
                            frame, observations[0]
                        )
                        return [canonical]
                    self._start_batch(frame, observations)
                    return list(self._batch_observations)

                if frame.message_type == "close":
                    return None
                if frame.message_type == "infer":
                    observation = frame.payload.get("observation")
                    if not isinstance(observation, dict):
                        raise ValueError("infer payload missing observation map")
                    canonical, self._layout = self._convert_observation(frame, observation)
                    return [canonical]
                if frame.message_type == "batch_infer":
                    observations = frame.payload.get("observations")
                    if not isinstance(observations, list) or not observations:
                        raise ValueError("batch_infer payload missing observations list")
                    if not all(isinstance(item, dict) for item in observations):
                        raise ValueError("batch observations must be maps")
                    self._start_batch(frame, observations)
                    return list(self._batch_observations)
            except Exception as exc:
                if raw is None:
                    return None
                await self.send_error(exc)

    async def send_action(self, canonical_action: dict) -> int:
        try:
            if self._batch_frame is not None:
                index = self._batch_index
                self._batch_actions.append(
                    self._adapter.to_robodojo_action(
                        canonical_action, self._batch_layouts[index]
                    )
                )
                self._batch_index += 1
                if self._batch_index < len(self._batch_layouts):
                    return 0
                frame = self._batch_frame
                payload = {"actions": self._batch_actions}
                sent_count = sum(
                    _wire_action_count(actions)
                    for actions in self._batch_actions
                )
                self._clear_batch()
                await self._send_frame(frame, payload)
                return sent_count

            if self._frame is None or self._frame.message_type not in {"infer", "call"}:
                raise RuntimeError("no pending RoboDojo inference request")
            actions = self._adapter.to_robodojo_action(
                canonical_action, self._layout
            )
            payload = (
                {"ok": True, "result": actions, "video_overlay": canonical_action.get("video_overlay")}
                if self._frame.message_type == "call"
                else {**canonical_action, "actions": actions}
            )
            await self._send_frame(self._frame, payload)
            return _wire_action_count(actions)
        except Exception as exc:
            await self.send_error(exc)
            self._clear_batch()
            return 0

    async def send_action_batch(self, canonical_actions: list[dict]) -> list[int]:
        """Translate and send all actions for one pending RoboDojo batch."""
        try:
            if self._batch_frame is None:
                if len(canonical_actions) != 1:
                    raise RuntimeError("single RoboDojo inference expects one action")
                return [await self.send_action(canonical_actions[0])]
            if len(canonical_actions) != len(self._batch_layouts):
                raise ValueError(
                    "action batch length does not match observation batch: "
                    f"{len(canonical_actions)} != {len(self._batch_layouts)}"
                )
            wire_actions = [
                self._adapter.to_robodojo_action(action, layout)
                for action, layout in zip(canonical_actions, self._batch_layouts, strict=True)
            ]
            counts = [_wire_action_count(actions) for actions in wire_actions]
            frame = self._batch_frame
            self._clear_batch()
            payload = (
                {
                    "ok": True,
                    "result": wire_actions,
                    "video_overlay": [
                        action.get("video_overlay") if isinstance(action, dict) else None
                        for action in canonical_actions
                    ],
                }
                if frame.message_type == "call"
                else {
                    "actions": wire_actions,
                    "video_overlay": [
                        action.get("video_overlay") if isinstance(action, dict) else None
                        for action in canonical_actions
                    ],
                }
            )
            await self._send_frame(frame, payload)
            return counts
        except Exception as exc:
            await self.send_error(exc)
            self._clear_batch()
            raise

    async def send_error(self, error: Exception) -> None:
        if self._frame is None:
            raise error
        frame = self._frame
        await self._send_frame(
            frame,
            {
                "code": "infer_failed"
                if frame.message_type in {"infer", "batch_infer", "call"}
                else "invalid_frame",
                "message": str(error),
                "details": {},
            },
            message_type="error",
        )

    def _convert_observation(
        self, frame: RoboDojoFrame, observation: dict
    ) -> tuple[dict, ActionLayout | None]:
        _decode_images(observation)
        marker = frame.trial_id or frame.action_case_id or frame.evaluation_id
        marker = f"{marker}:{self._episode_generation}"
        canonical, layout = self._adapter.to_canonical(
            observation,
            step=frame.step,
            episode_marker=marker,
        )
        self._add_task_name(canonical, frame)
        return canonical, layout

    @staticmethod
    def _add_task_name(observation: dict, frame: RoboDojoFrame) -> None:
        if observation.get("task_name") or observation.get("__task_name"):
            return
        action_case_id = frame.action_case_id or ""
        if action_case_id.endswith("_case"):
            observation["__task_name"] = action_case_id[:-5]

    def _start_batch(self, frame: RoboDojoFrame, observations: list[dict]) -> None:
        self._batch_frame = frame
        self._batch_index = 0
        self._batch_actions = []
        self._batch_observations = []
        self._batch_layouts = []
        marker = frame.trial_id or frame.action_case_id or frame.evaluation_id
        for index, observation in enumerate(observations):
            _decode_images(observation)
            env_idx = observation.get("env_idx", index)
            episode_marker = f"{marker}:{self._episode_generation}"
            if "env_idx" in observation:
                episode_marker += f":env{env_idx}"
            canonical, layout = self._adapter.to_canonical(
                observation,
                step=frame.step,
                episode_marker=episode_marker,
            )
            self._add_task_name(canonical, frame)
            self._batch_observations.append(canonical)
            self._batch_layouts.append(layout)

    async def _handle_call(self, frame: RoboDojoFrame) -> list[dict] | None:
        func_name = frame.payload.get("func_name")
        if func_name == "update_obs":
            observation = frame.payload.get("obs")
            if not isinstance(observation, dict):
                raise ValueError("update_obs payload missing obs map")
            self._call_observation = observation
            self._call_observations = None
            await self._send_frame(frame, {"ok": True, "result": None})
            return None
        if func_name == "update_obs_batch":
            observations = frame.payload.get("obs")
            if (
                not isinstance(observations, list)
                or not observations
                or not all(isinstance(item, dict) for item in observations)
            ):
                raise ValueError("update_obs_batch payload missing obs list")
            self._call_observations = observations
            self._call_observation = None
            await self._send_frame(frame, {"ok": True, "result": None})
            return None
        if func_name == "get_action":
            if self._call_observation is None:
                raise ValueError("get_action called before update_obs")
            observation = self._call_observation
            self._call_observation = None
            return [observation]
        if func_name == "get_action_batch":
            if self._call_observations is None:
                raise ValueError("get_action_batch called before update_obs_batch")
            observations = self._call_observations
            self._call_observations = None
            return observations
        raise ValueError(f"unsupported RoboDojo call: {func_name}")

    def _clear_call_observations(self) -> None:
        self._call_observation = None
        self._call_observations = None

    def _clear_batch(self) -> None:
        self._batch_frame = None
        self._batch_observations = []
        self._batch_layouts = []
        self._batch_actions = []
        self._batch_index = 0

    async def _send_frame(
        self,
        request: RoboDojoFrame,
        payload: dict,
        *,
        message_type: str | None = None,
    ) -> None:
        response_type = message_type or _RESPONSE_TYPES[request.message_type]
        await self._ws.send(
            pack_frame(
                RoboDojoFrame(
                    message_type=response_type,
                    message_id=request.message_id,
                    evaluation_id=request.evaluation_id,
                    action_case_id=request.action_case_id,
                    trial_id=request.trial_id,
                    repeat_index=request.repeat_index,
                    step=request.step,
                    payload=payload,
                ).to_wire()
            )
        )


def _health_check(connection, request):
    path = getattr(request, "path", connection)
    if path == "/healthz":
        respond = getattr(connection, "respond", None)
        if respond is not None:
            return respond(http.HTTPStatus.OK, "OK\n")
        return http.HTTPStatus.OK, [], b"OK\n"
    return None


class RoboDojoWsFrontend(Frontend):
    async def serve(
        self,
        host: str,
        port: int,
        on_session: Callable[[FrontendSession], Awaitable[None]],
    ) -> None:
        try:
            import websockets.asyncio.server as ws_server
        except ModuleNotFoundError:  # websockets < 13 (for example RoboDojo's env)
            import websockets.server as ws_server

        async def _ws_handler(ws):
            session = RoboDojoWsSession(ws)
            try:
                await on_session(session)
            except Exception:
                logger.exception("RoboDojo frontend session failed")
                try:
                    await session.send_error(Exception("proxy session failed"))
                except Exception:
                    pass

        async with ws_server.serve(
            _ws_handler,
            host,
            port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=60,
            ping_timeout=120,
        ) as server:
            logger.info("RoboDojoWsFrontend listening on %s:%s", host, port)
            await server.serve_forever()
