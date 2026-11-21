"""RoboDojo (Isaac Sim) behind the KnowinAgent `RobotBackend` contract.

One backend instance owns one layout (bank, ordinal) in one process. Each
`step` is one native ``take_action`` for both arms, i.e. one 25 Hz control tick
counted against the task's native ``step_lim``. `reset` replays the same layout
the way the official evaluator moves between episodes: native ``close()`` (stage
torn down) followed by ``reset(seed=[ordinal])`` (stage rebuilt), never the
in-place soft reload. Resets are for practice; a scored run uses none.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from knowin_agent.backends.robodojo import native as boot
from knowin_agent.backends.robodojo.calibration import calibrate_cameras, to_numpy
from knowin_agent.backends.robodojo.layouts import SEED_STRIDE, decode_scene, resolve_root
from knowin_agent.backends.robodojo.profile import arx_x5_profile
from knowin_agent.backends.robodojo.rig import rig_calibration
from knowin_agent.backends.robodojo.scoring import condition_report, object_poses, partial_credit_supported, read_score
from knowin_agent.core.geometry import Pose
from knowin_agent.runtime.backend import (
    ArmCommand,
    ArmState,
    Budget,
    CameraFrame,
    Observation,
    RobotProfile,
    StepOutcome,
)

DEPTH_MODES = ("none", "native")
# sim: read from the simulator cameras (not in the official observation); constant: rig constants (rig.py);
# none: no calibration (metric perception unavailable).
CALIBRATION_MODES = ("sim", "constant", "none")
FrameSink = Callable[[Observation], None]


def _is_physx_break(exc: BaseException) -> bool:
    return type(exc).__name__ == "PhysXBrokenError"


def _depth_map(value: Any, name: str) -> np.ndarray:
    depth = np.asarray(value, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"{name}: depth must be HxW, got shape {depth.shape}")
    return depth


def _rgb_image(value: Any, name: str) -> np.ndarray:
    rgb = np.asarray(value)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.dtype != np.uint8:
        raise ValueError(f"{name}: color must be uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
    return np.ascontiguousarray(rgb[..., :3])


class RoboDojoBackend:
    """A RoboDojo evaluation episode driven tick by tick by the KnowinAgent runtime."""

    profile: RobotProfile
    supports_reset = True

    def __init__(
        self,
        task: str,
        seed: int = 0,
        *,
        robodojo_root: Optional[os.PathLike | str] = None,
        kit_args: Optional[str] = None,
        depth: str = "none",
        calibration: bool | str = True,
        official_action_limit: bool = True,
        profile: Optional[RobotProfile] = None,
        frame_sink: Optional[FrameSink] = None,
    ):
        self._configure(task, seed, depth, calibration, official_action_limit, profile, frame_sink)
        self.root = resolve_root(robodojo_root)
        self.bank, self.ordinal, self.layout_path = decode_scene(task, seed, self.root)
        try:
            with boot.robodojo_cwd(self.root):
                run_id = boot.prepare_process(self.root)
                monitor = boot.start_physx_monitor(self.root, task)
                self.app, launch_info = boot.launch_app(kit_args)
                env_cfg = boot.build_env_cfg(
                    task, self.bank, native_depth=depth == "native", monitor_enabled=monitor is not None, run_id=run_id
                )
                self.native = boot.create_native(env_cfg, self.app)
                boot.begin_episode(self.native, self.ordinal, monitor)
            self._monitor = monitor
            self._boot_info.update(launch_info, run_id=run_id, physx_monitor=monitor is not None,
                                   physics_device=str(getattr(self.native, "device", "unknown")))
            self._attach()
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_native(
        cls,
        native: Any,
        app: Any = None,
        *,
        task: str = "unknown",
        seed: int = 0,
        robodojo_root: Optional[os.PathLike | str] = None,
        depth: str = "none",
        calibration: bool | str = False,
        official_action_limit: bool = True,
        profile: Optional[RobotProfile] = None,
        frame_sink: Optional[FrameSink] = None,
    ) -> "RoboDojoBackend":
        """Wrap a native env that is already reset (tests, or a caller that booted it itself)."""
        self = cls.__new__(cls)
        self._configure(task, seed, depth, calibration, official_action_limit, profile, frame_sink)
        self.root = Path(robodojo_root) if robodojo_root is not None else None
        self.bank, self.ordinal = divmod(seed, SEED_STRIDE)
        self.layout_path = None
        self.app, self.native = app, native
        self._attach()
        return self

    def _configure(self, task, seed, depth, calibration, official_action_limit, profile, frame_sink) -> None:
        if depth not in DEPTH_MODES:
            raise ValueError(f"depth must be one of {DEPTH_MODES}, got {depth!r}")
        self.task = task
        self.seed = seed
        self.depth = depth
        mode = {True: "sim", False: "none"}.get(calibration, calibration)
        if mode not in CALIBRATION_MODES:
            raise ValueError(f"calibration must be one of {CALIBRATION_MODES} (or a bool), got {calibration!r}")
        self.calibration_mode = mode
        self.calibration = mode == "sim"
        self.official_action_limit = bool(official_action_limit)
        self.profile = profile if profile is not None else arx_x5_profile()
        self.frame_sink = frame_sink
        self.app = self.native = None
        self.root: Optional[Path] = None
        self._boot_info: dict = {"policy_name": boot.POLICY_NAME}
        self._episode_over = False
        self._reason: Optional[str] = None
        self._error: Optional[str] = None
        self._kinematics_check: dict = {}
        self._last: Optional[Observation] = None
        self._final_evaluation: Optional[dict] = None
        self._grip_commands: dict[str, float] = {}
        self._monitor: Any = None
        self._resets: list[dict] = []
        self._ticks_before_reset = 0

    def _attach(self) -> None:
        self.native_step_limit = int(self.native.step_lim)
        if not self.official_action_limit:
            self.native.step_lim = math.inf
        self._last = self._fetch()
        self._instruction = self._last.instruction
        self._kinematics_check = self._check_kinematics(self._last)
        self._emit(self._last)

    def _cwd(self):
        return boot.robodojo_cwd(self.root)

    def _fetch(self) -> Observation:
        with self._cwd():
            raw = self.native.get_obs()
        return self.convert_observation(raw)

    def convert_observation(self, raw: Mapping[str, Any]) -> Observation:
        """Native observation dict -> Observation with logical camera names."""
        state = raw["state"]
        arms = {}
        for arm in self.profile.arms:
            opening = float(np.asarray(state[f"{arm}_ee_joint_state"], dtype=float).reshape(-1)[0])
            arms[arm] = ArmState(
                ee_pose=Pose.from_array(np.asarray(state[f"{arm}_ee_pose"], dtype=float)),
                joints=np.asarray(state[f"{arm}_arm_joint_state"], dtype=float).reshape(-1),
                gripper_command=self._grip_commands.setdefault(arm, opening),
                gripper_opening=opening,
            )
        vision = raw.get("vision") or {}
        missing = [name for name in self.profile.cameras.values() if name not in vision]
        if missing:
            raise KeyError(f"native observation lacks cameras {missing} (has {sorted(vision)})")
        rgb = {name: _rgb_image(vision[name]["color"], name) for name in self.profile.cameras.values()}
        calibration = {}
        if self.calibration:
            origins = getattr(self.native, "env_origins", None)
            calibration = calibrate_cameras(
                self.native.camera_manager, rgb, env_origin=None if origins is None else to_numpy(origins)[0]
            )
        elif self.calibration_mode == "constant":
            calibration = rig_calibration({arm: arms[arm].ee_pose for arm in arms}, rgb)
        cameras = {}
        for logical, name in self.profile.cameras.items():
            depth = None
            if self.depth == "native":
                if "depth" not in vision[name]:
                    raise KeyError(f"{name}: native depth requested but missing from the observation")
                depth = _depth_map(vision[name]["depth"], name)
                if depth.shape != rgb[name].shape[:2]:
                    raise ValueError(f"{name}: depth {depth.shape} and rgb {rgb[name].shape[:2]} sizes differ")
            calib = calibration.get(name)
            cameras[logical] = CameraFrame(
                name=logical,
                rgb=rgb[name],
                intrinsics=None if calib is None else calib.intrinsics,
                extrinsics=None if calib is None else calib.extrinsics,
                depth=depth,
            )
        return Observation(
            tick=int(self.native.take_action_cnt[0]),
            arms=arms,
            cameras=cameras,
            instruction=str(raw.get("instruction", "")),
        )

    def native_action(self, commands: Mapping[str, ArmCommand]) -> dict[str, np.ndarray]:
        """The native action dict for one tick: joint mode when every arm carries joint targets, else ee mode."""
        arms = set(self.profile.arms)
        if set(commands) != arms:
            raise ValueError(f"need exactly one command per arm {sorted(arms)}, got {sorted(commands)}")
        joint_mode = all(commands[arm].joints is not None for arm in self.profile.arms)
        action = {}
        for arm in self.profile.arms:
            command = commands[arm]
            if joint_mode:
                action[f"{arm}_arm_joint_state"] = np.asarray(command.joints, dtype=np.float64)
            else:
                pose = command.ee_pose
                if pose is None:
                    if self.profile.kinematics is None:
                        raise ValueError(f"{arm}: joint-only command in an ee tick and no kinematics to convert it")
                    pose = self.profile.kinematics.fk(arm, command.joints)
                action[f"{arm}_ee_pose"] = np.asarray(pose.array(), dtype=np.float64)
            action[f"{arm}_ee_joint_state"] = np.array([float(command.gripper)], dtype=np.float64)
        return action

    def observe(self) -> Observation:
        return self._last

    def step(self, commands: Mapping[str, ArmCommand]) -> StepOutcome:
        if self._episode_over:
            raise RuntimeError("episode is over; call reset() to replay the layout, or start a new backend")
        action = self.native_action(commands)
        mode = "joint" if any(key.endswith("_arm_joint_state") for key in action) else "ee"
        native = self.native
        try:
            with self._cwd():
                native.validate_action_dict(action)
                if native.get_action_type(action) != mode:
                    raise ValueError(f"native did not classify the action as a {mode} action")
                for arm in self.profile.arms:
                    self._grip_commands[arm] = float(commands[arm].gripper)
                before = int(native.take_action_cnt[0])
                native.take_action(action)
                after = int(native.take_action_cnt[0])
            if after != before + 1:
                raise RuntimeError(f"native take_action advanced {after - before} ticks instead of 1")
            self._last = self._fetch()
        except Exception as exc:
            if not (_is_physx_break(exc) or 0 in native.unstable_envs):
                raise
            return self._finish("unstable", f"{type(exc).__name__}: {exc}")
        if 0 in native.unstable_envs:
            return self._finish("unstable", "native marked the episode unstable")
        if bool(native.end_flag[0]):
            limit_hit = not bool(native.success[0]) and int(native.take_action_cnt[0]) >= native.step_lim
            return self._finish("action_limit" if limit_hit else "task_ended")
        self._emit(self._last)
        return StepOutcome(self._last)

    def _finish(self, reason: str, error: Optional[str] = None) -> StepOutcome:
        self._episode_over, self._reason, self._error = True, reason, error
        self._emit(self._last)
        return StepOutcome(self._last, episode_over=True, reason=reason)

    def reset(self) -> Observation:
        """Replay this backend's layout in the same process and return the first observation.

        Renews the native action allowance (the native tick counter restarts at 0), keeps the
        cumulative tick count for reporting, restores the original instruction text, re-reads
        camera calibration from the rebuilt cameras and forgets the last gripper commands.
        Raises `LayoutUnstable` if the replayed layout fails the native stability check; the
        backend then refuses further steps and must be closed.
        """
        if self.native is None:
            raise RuntimeError("backend is closed")
        if self._resets and self._resets[-1]["status"] != "ok":
            raise RuntimeError("a previous reset failed; the native env is in an unknown state, close this backend")
        attempt_ticks = int(self.native.take_action_cnt[0])
        event = {
            "reset": len(self._resets) + 1,
            "tick": attempt_ticks,
            "cumulative_tick": self._ticks_before_reset + attempt_ticks,
            "previous_reason": self._reason,
            "previous_score": self._attempt_score(),
            "status": "failed",
        }
        self._resets.append(event)
        self._ticks_before_reset += attempt_ticks
        self._episode_over, self._reason, self._error = True, "reset_failed", None
        try:
            with self._cwd():
                self._end_native_episode()
                boot.begin_episode(self.native, self.ordinal, self._monitor)
            if 0 in self.native.unstable_envs:
                raise boot.LayoutUnstable("native marked the replayed layout unstable")
        except boot.LayoutUnstable as exc:
            self._reason, self._error = "unstable", str(exc)
            raise boot.LayoutUnstable(
                f"same-scene reset {event['reset']} of {self.task} (bank {self.bank}, ordinal {self.ordinal}) "
                f"failed the native stability check: {exc}; close this backend and start a new one"
            ) from exc
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            raise
        self._renew_action_allowance()
        self._restore_instruction()
        self._grip_commands.clear()
        self._last = self._fetch()
        if self._last.instruction != self._instruction:
            raise RuntimeError("native observation did not keep the restored instruction")
        self._episode_over, self._reason = False, None
        self._kinematics_check = self._check_kinematics(self._last)
        event["status"] = "ok"
        self._emit(self._last)
        return self._last

    def _attempt_score(self) -> Optional[float]:
        try:
            return read_score(self.native)["score"]
        except Exception:
            return None

    def _end_native_episode(self) -> None:
        """End the current episode as main.py does between batches (``env.close()``, main.py:444).

        Native ``close`` destroys the capture/camera/robot/scene managers' handles and the USD
        stage; the following ``reset`` finds ``sim is None`` and relaunches, so the replay takes
        the same code path as the first episode rather than the in-place soft reload.
        """
        self.native.close()
        self._discard_destroyed_render_products()

    def _discard_destroyed_render_products(self) -> None:
        """Drop render-product handles that native ``TiledCaptureManager.destroy`` already destroyed.

        ``destroy`` calls ``destroy()`` on every entry of ``tiled_render_products`` but never empties
        the list, while ``init_cameras`` appends a new product per camera on every relaunch. Without
        this, each later close would destroy the handles of every earlier (already closed) stage again.
        """
        capture = getattr(self.native, "capture_manager", None)
        products = getattr(capture, "tiled_render_products", None)
        if isinstance(products, list):
            products.clear()

    def _renew_action_allowance(self) -> None:
        """Restore the task's step limit; native ``EvalEnv.reset`` already zeroed ``take_action_cnt``."""
        self.native.step_lim = self.native_step_limit if self.official_action_limit else math.inf
        if int(self.native.take_action_cnt[0]) != 0:
            raise RuntimeError("native reset did not restart the action counter")

    def _restore_instruction(self) -> None:
        """Put the episode's original instruction back.

        Native ``ObsManager.reset`` (called by ``EvalEnv.close`` and ``EvalEnv.reset``) draws a new
        description, which can be a different paraphrase of the same goal.
        """
        manager = getattr(self.native, "obs_manager", None)
        current = getattr(manager, "instruction", None)
        if not isinstance(current, list) or not current:
            raise RuntimeError("native obs_manager has no per-env instruction list to restore")
        current[0] = self._instruction

    def _emit(self, observation: Observation) -> None:
        if self.frame_sink is not None:
            self.frame_sink(observation)

    def _check_kinematics(self, observation: Observation) -> dict:
        kinematics = self.profile.kinematics
        if kinematics is None or not hasattr(kinematics, "model_residual"):
            return {}
        out = {}
        for arm, state in observation.arms.items():
            try:
                dp, da = kinematics.model_residual(arm, state.joints, state.ee_pose)
                out[arm] = {"fk_position_error_mm": round(dp * 1000, 3), "fk_angle_error_deg": round(math.degrees(da), 3)}
            except ValueError as exc:
                out[arm] = {"error": str(exc)}
        return out

    def cumulative_ticks(self) -> int:
        """Ticks over every attempt since boot (resets included); for reporting only."""
        return self._ticks_before_reset + self.budget().ticks_used

    def budget(self) -> Budget:
        """The current attempt's native allowance; `reset` renews it."""
        used = int(self.native.take_action_cnt[0]) if self.native is not None else self._last.tick
        return Budget(used, self.native_step_limit if self.official_action_limit else None)

    def instruction(self) -> str:
        return self._instruction

    def non_official_conditions(self) -> list[str]:
        conditions = []
        if self.depth == "native":
            conditions.append("native_depth: distance_to_image_plane annotators and vision.depth enabled")
        if self.calibration:
            conditions.append("camera_calibration: optical intrinsics/extrinsics read from the simulator cameras")
        if not self.official_action_limit:
            conditions.append("unlimited_action_budget: native step_lim disabled")
        if self._resets:
            conditions.append(f"same_scene_reset: {len(self._resets)} reset(s); score and budget cover the last attempt")
        return conditions

    def evaluation(self) -> dict:
        """Host-only: native score and the conditions it was obtained under. Never give this to an agent."""
        if self.native is None:
            if self._final_evaluation is None:
                raise RuntimeError("backend closed before an evaluation was recorded")
            return dict(self._final_evaluation)
        result = read_score(self.native)
        unstable = self._reason == "unstable"
        result.update(
            task=self.task,
            seed=self.seed,
            bank=self.bank,
            ordinal=self.ordinal,
            layout=None if self.layout_path is None else str(self.layout_path),
            policy_name=boot.POLICY_NAME,
            episode_over=self._episode_over,
            reason=self._reason,
            error=self._error,
            excluded=unstable,
            partial_credit_supported=partial_credit_supported(self.native),
            budget=self.budget().as_dict(),
            cumulative_ticks=self.cumulative_ticks(),
            resets={"count": len(self._resets), "events": [dict(e) for e in self._resets]},
            native_step_limit=self.native_step_limit,
            official_action_limit=self.official_action_limit,
            depth=self.depth,
            official_observation=self.depth == "none" and not self.calibration,
            non_official_conditions=self.non_official_conditions(),
            kinematics_check=self._kinematics_check,
            boot=dict(self._boot_info),
        )
        return result

    def ground_truth(self) -> Optional[dict]:
        """Host-only: every scoring predicate evaluated now and every object's pose. Never give this to an agent."""
        if self.native is None:
            return None
        return {"conditions": condition_report(self.native), "objects": object_poses(self.native)}

    def close(self) -> None:
        """Shut down the native env, then the Isaac Sim app; the last evaluation stays readable."""
        native, app = getattr(self, "native", None), getattr(self, "app", None)
        if native is not None and getattr(self, "_last", None) is not None:
            try:
                self._final_evaluation = self.evaluation()
            except Exception as exc:
                self._final_evaluation = {"error": f"evaluation unavailable at close: {exc}"}
        self.native = self.app = None
        try:
            if native is not None:
                with self._cwd():
                    native.close()
        finally:
            if app is not None:
                app.close()
