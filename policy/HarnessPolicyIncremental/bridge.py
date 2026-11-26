"""Step 2c: Independent XPolicyLab <-> episode thread bridge for HarnessPolicyIncremental.

Replaces pyrualean.hosts.xpolicylab_bridge and xpolicylab_robodojo.BridgeEnv.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .exceptions import MotionUnacknowledged
from .instructions import episode_instruction

STATUS_KEY = "harness_status"
MOTION_BINDING = "harness_motion_binding"
WAIT = [{"harness_wait": True}]
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CAMERA_FIELDS = ("color", "rgb", "shape", "intrinsic_matrix", "extrinsic_matrix", "extrinsics_matrix")
ROBOT_STATE_FIELDS = tuple(
    f"{arm}_{field}"
    for arm in ("left", "right")
    for field in ("arm_joint_state", "ee_joint_state", "ee_pose", "tcp_pose", "delta_ee_pose")
)
STATUS_FIELDS = ("step", "step_limit", "pictures_behind")
VERDICT_FIELDS = (*STATUS_FIELDS, "episode_id", "receipt", "success", "client_error")


def robot_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Project only named RGB cameras/calibration and robot proprioception."""
    raw_state = obs.get("state") or {}
    raw_vision = obs.get("vision") or {}
    return {
        "state": {key: raw_state[key] for key in ROBOT_STATE_FIELDS if key in raw_state},
        "vision": {
            name: {key: raw_vision[name][key] for key in CAMERA_FIELDS if key in raw_vision[name]}
            for name in CAMERAS if isinstance(raw_vision.get(name), dict)
        },
    }


def receipt_matches(receipt: Any, episode: str, step: int) -> bool:
    return (isinstance(receipt, list) and len(receipt) == 2
            and isinstance(receipt[0], str) and receipt[0] == episode
            and type(receipt[1]) is int and receipt[1] == step)


class EpisodeOver(BaseException):
    """Raised when the simulator client has ended the current episode."""


class EpisodeBridge:
    """Thread-safe rendezvous between XPolicyLab RPCs and the skill execution thread."""

    def __init__(self, poll_s: float = 20.0) -> None:
        self._cv = threading.Condition()
        self.poll_s = float(poll_s)
        self._obs: dict[str, Any] | None = None
        self._action: dict[str, Any] | None = None
        self._binding: tuple[str, int] | None = None
        self._instruction = ""
        self._pending: tuple[str, int] | None = None
        self._dispatched = False
        self._receipts: list[tuple[str, int]] = []
        self.motion_error: str | None = None
        self._finished = False
        self._over: dict[str, Any] | None = None
        self._status: dict[str, Any] = {}
        self._pictured: Any = None
        self.resets = 0

    def reset(self) -> None:
        with self._cv:
            self.resets += 1
            self._cv.notify_all()

    def update_obs(self, obs: dict[str, Any]) -> None:
        with self._cv:
            if self._finished or self._over is not None:
                return
            info = (obs.get("additional_info") or {}).get(STATUS_KEY)
            info = info if isinstance(info, dict) else {}
            episode, observation_step = info.get("episode_id"), info.get("observation_step")
            binding = (episode, observation_step)
            if not isinstance(episode, str) or not episode or type(observation_step) is not int:
                self._fail_motion("missing episode or observation-step binding")
                return
            if self._binding is None:
                valid = observation_step == 0 and info.get("receipt") is None
            elif binding == self._binding:
                return
            else:
                valid = (self._dispatched and self._pending == self._binding
                         and binding == (self._binding[0], self._binding[1] + 1)
                         and receipt_matches(info.get("receipt"), *self._binding))
            if not valid:
                self._fail_motion("observation does not acknowledge the dispatched motion")
                return
            if self._binding is not None:
                self._receipts.append(self._binding)
            else:
                self._instruction = episode_instruction(obs)
            self._binding = binding
            self._status = {key: info[key] for key in STATUS_FIELDS if key in info}
            obs = robot_observation(obs)
            obs["instruction"] = self._instruction
            state = obs["state"]
            poses = {
                name: cam.get(key)
                for name, cam in (obs.get("vision") or {}).items()
                for key in ("extrinsic_matrix", "extrinsics_matrix")
                if isinstance(cam, dict) and cam.get(key) is not None
            }
            behind = int(self._status.get("pictures_behind") or 0)
            if behind >= 1 and self._pictured is not None:
                obs["image_state"], prev_poses = self._pictured
                obs["vision"] = {name: dict(cam) if isinstance(cam, dict) else cam
                                 for name, cam in (obs.get("vision") or {}).items()}
                for name, cam in (obs.get("vision") or {}).items():
                    if name in prev_poses and isinstance(cam, dict):
                        for key in ("extrinsic_matrix", "extrinsics_matrix"):
                            if cam.get(key) is not None:
                                cam[key] = prev_poses[name]
            else:
                obs["image_state"] = state
            self._pictured = (state, poses)
            self._obs = obs
            self._cv.notify_all()

    def get_action(self) -> list[dict[str, Any]]:
        with self._cv:
            self._cv.wait_for(lambda: self._action is not None or self._finished or self._over is not None,
                              self.poll_s)
            if self._finished or self._over is not None:
                return []
            if self._action is not None and self._pending is not None:
                action, self._action = self._action, None
                self._dispatched = True
                self._cv.notify_all()
                return [{**action, MOTION_BINDING: list(self._pending)}]
            return WAIT

    def episode_end(self, info: dict[str, Any] | None) -> None:
        with self._cv:
            verdict = {key: info[key] for key in VERDICT_FIELDS if info is not None and key in info}
            if self._binding is not None and verdict.get("episode_id") != self._binding[0]:
                return
            if self._pending is not None and self._binding == self._pending:
                if not self._dispatched or not receipt_matches(verdict.get("receipt"), *self._pending):
                    self._fail_motion("episode ended without a receipt for its last motion")
                elif self._over is None:
                    self._receipts.append(self._pending)
            self._over = verdict
            self._action = None
            self._cv.notify_all()

    def first_obs(self, timeout_s: float, alive: Any = None) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self._obs is None:
                if self.motion_error is not None:
                    raise MotionUnacknowledged(self.motion_error)
                if self._finished:
                    raise EpisodeOver("episode finished before first observation")
                if self._over is not None:
                    raise EpisodeOver("episode ended before first observation")
                if alive is not None and not alive():
                    raise RuntimeError("evaluation client exited before first observation")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"no observation from client in {timeout_s:.0f} s")
                self._cv.wait(min(left, 2.0))
            return self._obs

    def get_obs(self) -> dict[str, Any]:
        with self._cv:
            if self._obs is None:
                raise RuntimeError("no observation received yet")
            return self._obs

    def status(self) -> dict[str, Any]:
        with self._cv:
            st = dict(self._status)
            if self._over is not None:
                st["over"] = True
                st["step_limit"] = st.get("step", 0)
            return st

    @property
    def over(self) -> dict[str, Any] | None:
        with self._cv:
            return None if self._over is None else dict(self._over)

    @property
    def receipts(self) -> list[tuple[str, int]]:
        with self._cv:
            return list(self._receipts)

    def step(self, action: dict[str, Any], timeout_s: float = 600.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        with self._cv:
            if self.motion_error is not None:
                raise MotionUnacknowledged(self.motion_error)
            if self._over is not None or self._finished:
                raise EpisodeOver("episode is over")
            if self._binding is None or self._pending is not None:
                self._fail_motion("cannot dispatch without a fresh episode observation")
                raise MotionUnacknowledged(self.motion_error)
            before = self._binding
            self._pending = self._binding
            self._dispatched = False
            self._action = action
            self._cv.notify_all()
            while self._binding == before and self._over is None and self.motion_error is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    self._fail_motion(f"no motion receipt from the evaluation client in {timeout_s:g} s")
                    break
                self._cv.wait(min(left, 2.0))
            self._pending = None
            if self.motion_error is not None:
                raise MotionUnacknowledged(self.motion_error)
            if self._binding == before:
                raise EpisodeOver("episode ended on this action")
            assert self._obs is not None
            return self._obs

    def _fail_motion(self, error: str) -> None:
        self.motion_error = self.motion_error or error
        self._action = None
        self._finished = True
        self._cv.notify_all()

    def finish(self) -> None:
        with self._cv:
            self._finished = True
            self._action = None
            self._cv.notify_all()

    def abandon(self) -> None:
        with self._cv:
            if self._pending is not None and self._dispatched and self._binding == self._pending:
                self._fail_motion("episode reset without a receipt for its last motion")
            self._finished = True
            self._action = None
            if self._over is None:
                self._over = {"success": False, "abandoned": True}
            self._cv.notify_all()

    def verdict(self, timeout_s: float = 300.0, alive: Any = None) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self._over is None:
                left = deadline - time.monotonic()
                if left <= 0 or (alive is not None and not alive()):
                    return None
                self._cv.wait(min(left, 2.0))
            return dict(self._over)


class EpisodeEnv:
    """Environment facade over EpisodeBridge for robot toolchains."""

    def __init__(self, bridge: EpisodeBridge, alive: Any = None) -> None:
        self.bridge = bridge
        self.alive = alive

    def get_obs(self) -> dict[str, Any]:
        return self.bridge.get_obs()

    def get_status(self) -> dict[str, Any]:
        return self.bridge.status()

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        obs = self.bridge.step(action)
        return obs, 0.0, False, {"status": self.bridge.status()}

    def episode_over(self) -> bool:
        return self.bridge.over is not None

    def conclude(self) -> None:
        self.bridge.finish()
        self.bridge.verdict(alive=self.alive)
