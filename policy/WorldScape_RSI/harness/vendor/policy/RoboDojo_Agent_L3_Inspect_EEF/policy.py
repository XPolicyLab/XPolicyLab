"""Cartesian action surface for the RGB-only Inspect agent."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import (
    InfrastructureFailure,
    JointAgentPolicy,
    MotionOutcome,
    RoboDojoActionSpec,
    _icl_message,
    _icl_text_message,
    _optional_bool,
    _optional_float,
)
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.trace import named_state, source_revision
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import (
    JOINT_CHANNELS,
    Action,
    ActionChunk,
    ActionSpace,
    Observation,
)

from . import pose

Planner = Callable[..., Mapping[str, Any]]

# How far one env step may carry a motion comes from the action spec's
# per-dimension ``max_step``, the way the upstream Inspect agent takes it from
# the embodiment's declaration rather than deciding a speed for itself. A fixed
# count cannot work: resampling is uniform, so a 480-waypoint CuRobo plan
# replayed as 15 joint targets asks for a jump the controller cannot track, and
# the arm lands hundreds of millimetres away.
#
# Upstream's playout cap. Nothing in the recorded sweep comes close, so this
# only exists to refuse a motion rather than silently compress it.
_DEFAULT_MAX_DURATION_S = 10.0
# The jaw is not limited by tracking but by dwell: it is one position-controlled
# degree of freedom, so what matters is how long its setpoint is held, not how
# finely its travel is divided. The embodiment declares what the jaw *can* do,
# its full range in four env steps; this asks for half again as many, leaving a
# full close at six steps and a small correction at one. Nothing measured so
# far separates four steps from six on grasp success, so this keeps the count
# the recorded sweep ran with rather than speeding the jaw up on a refactor.
_DEFAULT_GRIPPER_DWELL_STEP = 0.18

_LEGACY_KEYS = ("xyz", "quat", "arm", "left", "right")

_EEF_STATE_HEADER = (
    "World-frame grasp-point state, named as move_eef takes them "
    "(metres; degrees from the straight-down reference):"
)


def _icl_eef_values(group: Any, index: int, arm: str) -> dict[str, float]:
    values = pose.pose_to_values(np.asarray(group[f"{arm}_ee_poses"][index]))
    gripper = np.asarray(group[f"{arm}_ee_joint_states"][index]).reshape(-1)
    if gripper.size != 1:
        raise ValueError(f"{arm}_ee_joint_states must contain one value per frame")
    values["gripper"] = float(gripper[0])
    return values


def _icl_eef_14(group: Any, index: int) -> dict[str, float]:
    return {
        f"{arm}_{axis}": value
        for arm in pose.ARMS
        for axis, value in _icl_eef_values(group, index, arm).items()
    }


def _render_eef_14(values: Mapping[str, float]) -> list[str]:
    """Match the Cartesian lines a live observation prints for ``move_eef``."""
    lines = [_EEF_STATE_HEADER]
    for label in pose.labels():
        _, _, axis = label.partition("_")
        value = values[label]
        if axis == "gripper":
            lines.append(f"{label}={value:.2f}")
        elif axis in pose.ANGLE_AXES:
            lines.append(f"{label}={value:.1f}")
        else:
            lines.append(f"{label}={value:.4f}")
    return lines


def _format_eef_icl_frame(
    state: Any, action: Any, index: int, frame_count: int
) -> str:
    del action, frame_count
    lines = ["Expert observed:"]
    lines.extend(_render_eef_14(_icl_eef_14(state, index)))
    return "\n".join(lines)


def _function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


class EefAgentPolicy(JointAgentPolicy):
    """Inspect policy that asks the model for world-frame grasp-point dimensions."""

    motion_tool_name = "move_eef"
    # Official 2100 eef-astra default. imitate_sorting_sequence holds still
    # for the first 24 s (600 env steps at 25 Hz) and samples that wait into
    # the first planner call, so those turns no longer spend the LLM budget.
    _default_max_llm_calls = 170

    def __init__(self, *, planner: Planner, **kwargs: Any) -> None:
        self._planner = planner
        # The tool description quotes the playout cap, so the limits have to be
        # known before the base constructor builds the tools.
        env = kwargs.get("env") or {}
        self._max_duration_s = _optional_float(
            env, "L3_INSPECT_EEF_MAX_DURATION_S", _DEFAULT_MAX_DURATION_S
        )
        self._gripper_dwell_step = _optional_float(
            env, "L3_INSPECT_EEF_GRIPPER_DWELL_STEP", _DEFAULT_GRIPPER_DWELL_STEP
        )
        # Model-facing xyz is multiplied by -1 on the way in and out together,
        # so the controller still moves in the real world while the prompt
        # speaks a mirrored position frame.
        self._negate_xyz = _optional_bool(env, "L3_INSPECT_EEF_NEGATE_XYZ", False)
        # Fixed world-frame bias applied to every Cartesian move after the model
        # names its target. Arrival checks still compare against the un-biased
        # request so the next observation can close the loop. Axis offsets and
        # a uniform-on-sphere draw add: fixed radius R once per episode, or
        # exponential length with mean MEAN_R (optionally redrawn every move).
        self._jitter_dx = _optional_float(env, "L3_INSPECT_EEF_JITTER_DX", 0.0)
        self._jitter_dy = _optional_float(env, "L3_INSPECT_EEF_JITTER_DY", 0.0)
        self._jitter_dz = _optional_float(env, "L3_INSPECT_EEF_JITTER_DZ", 0.0)
        self._jitter_sphere_r = _optional_float(
            env, "L3_INSPECT_EEF_JITTER_SPHERE_R", 0.0
        )
        self._jitter_sphere_mean_r = _optional_float(
            env, "L3_INSPECT_EEF_JITTER_SPHERE_MEAN_R", 0.0
        )
        self._jitter_sphere_per_move = _optional_bool(
            env, "L3_INSPECT_EEF_JITTER_SPHERE_PER_MOVE", False
        )
        seed_raw = str(env.get("L3_INSPECT_EEF_JITTER_SPHERE_SEED") or "").strip()
        self._jitter_sphere_seed: int | None = int(seed_raw) if seed_raw else None
        self._jitter_rng = np.random.default_rng(self._jitter_sphere_seed)
        self._jitter_sphere_dx = 0.0
        self._jitter_sphere_dy = 0.0
        self._jitter_sphere_dz = 0.0
        for key, value in (
            ("L3_INSPECT_EEF_MAX_DURATION_S", self._max_duration_s),
            ("L3_INSPECT_EEF_GRIPPER_DWELL_STEP", self._gripper_dwell_step),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise InfrastructureFailure(f"{key} must be finite and > 0")
        for key, value in (
            ("L3_INSPECT_EEF_JITTER_DX", self._jitter_dx),
            ("L3_INSPECT_EEF_JITTER_DY", self._jitter_dy),
            ("L3_INSPECT_EEF_JITTER_DZ", self._jitter_dz),
            ("L3_INSPECT_EEF_JITTER_SPHERE_R", self._jitter_sphere_r),
            ("L3_INSPECT_EEF_JITTER_SPHERE_MEAN_R", self._jitter_sphere_mean_r),
        ):
            if not np.isfinite(value):
                raise InfrastructureFailure(f"{key} must be finite")
        if self._jitter_sphere_r < 0.0:
            raise InfrastructureFailure(
                "L3_INSPECT_EEF_JITTER_SPHERE_R must be >= 0"
            )
        if self._jitter_sphere_mean_r < 0.0:
            raise InfrastructureFailure(
                "L3_INSPECT_EEF_JITTER_SPHERE_MEAN_R must be >= 0"
            )
        super().__init__(**kwargs)
        self._resample_sphere_jitter()
        limits = self._declared_step_limits()
        located = ActionSpace(JOINT_CHANNELS).offsets()
        self._arm_step_limits: dict[str, np.ndarray] = {}
        self._gripper_step_limits: dict[str, float] = {}
        for arm in pose.ARMS:
            start, size = located[f"{arm}_arm_joint_state"]
            self._arm_step_limits[arm] = limits[start : start + size]
            start, _ = located[f"{arm}_ee_joint_state"]
            # The declaration is a ceiling on what the jaw can do; the dwell
            # margin is this policy asking for less than that.
            self._gripper_step_limits[arm] = min(
                float(limits[start]), self._gripper_dwell_step
            )

    @property
    def _max_steps(self) -> int:
        return max(1, int(np.ceil(self._max_duration_s * self.action_spec.control_hz)))

    def reset(self) -> None:
        self._last_targets: dict[str, np.ndarray] = {}
        self._gripper_goals: dict[str, float] = {}
        # Episode-constant sphere draws re-roll here; per-move mode redraws
        # inside each Cartesian plan instead.
        if not self._jitter_sphere_per_move:
            self._resample_sphere_jitter()
        super().reset()

    def _resample_sphere_jitter(self) -> None:
        """Draw a world xyz offset on a random sphere direction.

        ``SPHERE_R`` fixes the length. ``SPHERE_MEAN_R`` draws the length from
        an exponential with that mean (so E[||v||] matches). Mean mode wins
        when both are set. Radius 0 is a no-op.
        """
        if self._jitter_sphere_mean_r > 0.0:
            radius = float(self._jitter_rng.exponential(self._jitter_sphere_mean_r))
        elif self._jitter_sphere_r > 0.0:
            radius = float(self._jitter_sphere_r)
        else:
            self._jitter_sphere_dx = 0.0
            self._jitter_sphere_dy = 0.0
            self._jitter_sphere_dz = 0.0
            return
        vector = self._jitter_rng.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0.0:
            vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            norm = 1.0
        vector = vector / norm * radius
        self._jitter_sphere_dx = float(vector[0])
        self._jitter_sphere_dy = float(vector[1])
        self._jitter_sphere_dz = float(vector[2])

    def _jitter_offset(self) -> tuple[float, float, float]:
        return (
            self._jitter_dx + self._jitter_sphere_dx,
            self._jitter_dy + self._jitter_sphere_dy,
            self._jitter_dz + self._jitter_sphere_dz,
        )

    def _icl_demonstration(self, task_name: str) -> dict[str, Any] | None:
        # Text ICL is layout-generic prose with no pose channel. Prefer it when
        # configured so a text-arm launch cannot accidentally keep image+EEF.
        text = _icl_text_message(task_name, self._env)
        if text is not None:
            return text
        return _icl_message(
            task_name,
            self._env,
            frame_formatter=_format_eef_icl_frame,
            description=(
                "These chronological expert keyframes use the same 14 world-frame "
                "grasp-point dimensions as each live observation. Each image is "
                "paired with the 14-dimensional pose at that frame, not a "
                "next-step action. Use them as a task-specific reference, not as "
                "the current state."
            ),
        )

    # ------------------------------------------------------------------ #
    # Tool surface
    # ------------------------------------------------------------------ #

    def _build_tools(self, action_spec: RoboDojoActionSpec) -> list[dict[str, Any]]:
        terminal_tools = super()._build_tools(action_spec)[1:]
        move = _function(
            "move_eef",
            (
                "Move to absolute world-frame grasp-point targets. Name only the "
                "dimensions you intend to change; every unnamed dimension holds "
                "its observed value, so lowering one hand 2 cm is a single "
                "entry. Positions are metres. Orientation is three angles in "
                "degrees measured from a straight-down grasp, not absolute "
                "world angles: at "
                "pitch_deg=roll_deg=yaw_deg=0 the outward tool axis points at "
                "world -z. pitch_deg turns about world +x (0 is straight down, "
                "+90 is horizontal pointing at world +y), roll_deg about world "
                "+y, and yaw_deg about world +z, which is the jaw rotation of a "
                "straight-down grasp. Naming dimensions of both arms moves them "
                "simultaneously along time-aligned paths. The motion plays at a "
                "fixed safe speed, so how long it takes follows from how far it "
                f"goes; a move needing over {self._max_duration_s:g}s is refused "
                "and has to be split. A gripper value named alongside a pose "
                "moves the jaw only after the arm has arrived, so naming both "
                "in one call is a safe way to grasp at a pose. "
                f"Per-dimension bounds: {self._bounds_text()}."
            ),
            {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": (
                            "Map of dimension name to value. Valid names: "
                            + ", ".join(pose.labels())
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "What you observe right now and why you chose this "
                            "motion, in one or two plain sentences."
                        ),
                    },
                },
                "required": ["targets", "note"],
            },
        )
        return [move, *terminal_tools]

    def _axis_bounds(self, label: str) -> tuple[float, float]:
        low, high = pose.bounds(label)
        _, _, axis = label.partition("_")
        if self._negate_xyz and axis in pose.POSITION_AXES:
            return (-high, -low)
        return (low, high)

    def _bounds_text(self) -> str:
        return ", ".join(
            f"{label}: [{low:.4g}, {high:.4g}]"
            for label, (low, high) in (
                (label, self._axis_bounds(label)) for label in pose.labels()
            )
        )

    def _to_model_xyz(self, values: dict[str, float]) -> dict[str, float]:
        if not self._negate_xyz:
            return values
        out = dict(values)
        for axis in pose.POSITION_AXES:
            out[axis] = -float(out[axis])
        return out

    def _from_model_xyz(self, values: dict[str, float]) -> dict[str, float]:
        # Same transform both ways: observing -p and commanding -p recovers p.
        return self._to_model_xyz(values)

    def _system_message(self) -> str:
        return (
            "You are controlling a real robot embodiment named "
            "'robodojo-arx-x5'. You receive RGB camera images, the current "
            "world-frame grasp-point state of both arms in the same 14 dimensions "
            "move_eef takes, arm joint angles as context you cannot command, "
            "and a task instruction. Move with move_eef by "
            "naming only the world-frame dimensions you want to change. "
            "Cartesian targets must be estimated from RGB; no depth or "
            "world-coordinate query is available. Respond with exactly one tool "
            "call per turn. After each motion the next observation reports how "
            "far the grasp point ended up from what you asked for, so check it "
            "before assuming a motion landed. Two budgets run down at once and "
            f"whichever empties first ends the episode: {self._max_llm_calls} "
            "LLM calls, one per turn, and the environment's own step limit, "
            "reported with each observation as the env steps remaining. A "
            "motion spends env steps in proportion to how far it travels, so a "
            "small correction is cheap and a long reach is not."
            f"\n\nEmbodiment notes:\n{self.action_spec.docs.strip()}"
        )

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _state_block(self, observation: Observation) -> str:
        """Render the 14 Cartesian dimensions, then joints as read-only context.

        The Cartesian vector leads and is emitted in ``pose.labels()`` order, so
        it is dimension-for-dimension the space ``move_eef`` targets are
        absolute in. That alignment is the whole point: an absolute target only
        says where to go, so the model needs its starting point named the same
        way to reason about a correction at all.

        Joint angles follow as context the model cannot command. Their two
        gripper dimensions are the same quantity as ``<arm>_gripper`` above, so
        they are left out rather than repeated under a second name.
        """
        measured = {arm: self._measured_values(observation, arm) for arm in pose.ARMS}
        lines = [
            f"Instruction: {observation.instruction or ''}",
            "World-frame grasp-point state, named as move_eef takes them "
            "(metres; degrees from the straight-down reference):",
        ]
        for label in pose.labels():
            arm, _, axis = label.partition("_")
            if axis == "gripper":
                lines.append(f"{label}={self._measured_gripper(observation, arm):.2f}")
            elif measured[arm] is None:
                # Never silently drop a dimension: a missing line reads as
                # "nothing to say about it" when it means the reference the
                # model is about to aim from is not there.
                lines.append(f"{label}=unavailable")
            elif axis in pose.ANGLE_AXES:
                lines.append(f"{label}={measured[arm][axis]:.1f}")
            else:
                lines.append(f"{label}={measured[arm][axis]:.4f}")
        joints = named_state(self.action_spec.labels, observation.state)
        for arm in pose.ARMS:
            joints.pop(f"{arm}_gripper", None)
        lines.append("Arm joint angles, radians, for context; not commandable:")
        lines.extend(f"{label}={value:.4f}" for label, value in joints.items())
        return "\n".join(lines)

    def _observation_message(
        self,
        observation: Observation,
        *,
        watch_until_env_step: int | None = None,
    ) -> dict[str, Any]:
        message = super()._observation_message(
            observation, watch_until_env_step=watch_until_env_step
        )
        arrival = self._arrival_text(observation)
        if arrival:
            message["content"][0]["text"] += "\n" + arrival
        return message

    def _arrival_text(self, observation: Observation) -> str:
        """Report how far the previous motion ended up from what was requested."""
        pending, self._last_targets = self._last_targets, {}
        parts = []
        for arm, target in sorted(pending.items()):
            measured = self._measured_pose(observation, arm)
            if measured is None:
                continue
            # Both sides are stored as flange poses because that is what the
            # planner takes, but the model reasons in grasp points, and the
            # two distances differ whenever the orientation missed as well.
            reached = pose.pose_to_values(measured)
            wanted = pose.pose_to_values(target)
            offset = float(
                np.linalg.norm(
                    [reached[axis] - wanted[axis] for axis in pose.POSITION_AXES]
                )
            )
            angle = pose.orientation_error_deg(measured[3:7], target[3:7])
            parts.append(f"{arm} {offset * 1000:.0f} mm and {angle:.1f} deg away")
        if not parts:
            return ""
        return (
            "Arrival check for the previous move_eef target: "
            + "; ".join(parts)
            + "."
        )

    @staticmethod
    def _measured_pose(observation: Observation, arm: str) -> np.ndarray | None:
        raw = np.asarray(
            observation.state.get(f"{arm}_ee_pose", []), dtype=np.float64
        ).reshape(-1)
        if raw.size < 7 or not np.isfinite(raw[:7]).all():
            return None
        if float(np.linalg.norm(raw[3:7])) <= 1e-8:
            return None
        return raw[:7]

    def _measured_values(
        self, observation: Observation, arm: str
    ) -> dict[str, float] | None:
        measured = self._measured_pose(observation, arm)
        if measured is None:
            return None
        return self._to_model_xyz(pose.pose_to_values(measured))

    @staticmethod
    def _measured_gripper(observation: Observation, arm: str) -> float:
        raw = np.asarray(
            observation.state.get(f"{arm}_ee_joint_state", [0.0]), dtype=np.float64
        ).reshape(-1)
        return float(raw[0]) if raw.size else 0.0

    # ------------------------------------------------------------------ #
    # Motion
    # ------------------------------------------------------------------ #

    def _handle_motion(
        self,
        name: str,
        arguments: Mapping[str, Any],
        observation: Observation,
    ) -> MotionOutcome:
        del name
        legacy = [key for key in _LEGACY_KEYS if key in arguments]
        if legacy:
            return self._repair(
                f"move_eef no longer takes {legacy[0]!r}. Name world-frame "
                'dimensions instead, for example {"targets": {"left_z": 0.90, '
                '"left_yaw_deg": -90}}.'
            )
        by_arm, error = self._parse_targets(arguments.get("targets"))
        if error:
            return self._repair(error)
        assert by_arm is not None

        paths: dict[str, np.ndarray] = {}
        target_poses: dict[str, np.ndarray] = {}
        requested_poses: dict[str, np.ndarray] = {}
        if self._jitter_sphere_per_move:
            self._resample_sphere_jitter()
        jx, jy, jz = self._jitter_offset()
        for arm in sorted(by_arm):
            axes = {
                axis: value
                for axis, value in by_arm[arm].items()
                if axis != "gripper"
            }
            if not axes:
                continue
            measured = self._measured_values(observation, arm)
            if measured is None:
                raise InfrastructureFailure(
                    f"RoboDojo reported no usable {arm}_ee_pose; an absolute "
                    "Cartesian target cannot be built from a missing flange pose"
                )
            # measured is already in the model frame; fold overrides there, then
            # map positions back to the world before CuRobo sees them.
            merged = {**measured, **axes}
            world = self._from_model_xyz(merged)
            requested_pose = pose.values_to_pose(world)
            executed = dict(world)
            if jx or jy or jz:
                executed["x"] = float(executed["x"]) + jx
                executed["y"] = float(executed["y"]) + jy
                executed["z"] = float(executed["z"]) + jz
            target_pose = pose.values_to_pose(executed)
            path, failure = self._plan_arm(arm, target_pose)
            if failure is not None:
                return MotionOutcome(
                    chunk=None,
                    tool_result=f"{arm} arm {failure.tool_result} Neither arm moved.",
                    repairable=failure.repairable,
                )
            assert path is not None
            current = np.asarray(observation.state[arm + "_arm_joint_state"], dtype=np.float64)
            if np.max(np.abs(path[0] - current)) > 1e-3:
                raise InfrastructureFailure(f"planner_start_discontinuity: {arm}; first waypoint does not match public joint state")
            path = path.copy()
            path[0] = current
            start, size = ActionSpace(JOINT_CHANNELS).offsets()[arm + "_arm_joint_state"]
            if np.any(path < self.action_spec.low[start:start+size] - 1e-6) or np.any(path > self.action_spec.high[start:start+size] + 1e-6):
                raise InfrastructureFailure(f"planner_joint_limit_violation: {arm}")
            paths[arm] = path
            # Arrival checks the model-named pose; the planner flew to the
            # jittered one. Keeping those distinct is what makes the bias
            # visible as residual error on the next turn.
            requested_poses[arm] = requested_pose
            target_poses[arm] = target_pose

        grippers = {
            arm: (self._measured_gripper(observation, arm), overrides["gripper"])
            for arm, overrides in by_arm.items()
            if overrides.get("gripper") is not None
        }
        # The jaw moves only once the arm has arrived. Closing it while the
        # hand is still travelling sweeps the object out of the way before the
        # grasp, which is the most common complaint in the recorded give_up
        # reasons. Sequencing costs the jaw's own waypoints on top of the
        # arm's, which is why the derivation adds the two phases.
        arm_steps = self._arm_steps(paths)
        if paths:
            arm_steps = max(1, arm_steps)
        gripper_steps = self._gripper_steps(grippers)
        steps = arm_steps + gripper_steps
        if steps > self._max_steps:
            return self._repair(
                f"this motion needs {steps / self.action_spec.control_hz:.1f}s at "
                f"the safe speed, over the {self._max_duration_s:g}s playout cap; "
                "split it into smaller motions"
            )
        if steps == 0:
            # Every named dimension already holds the value it was asked for.
            arm_steps = steps = 1
        # The count is never clamped to the planner's own point count.
        # Resampling interpolates along the planned polyline, so asking for
        # more waypoints than CuRobo returned adds points on that same
        # collision-checked path; clamping instead would be the compression
        # this derivation exists to avoid, only in miniature.
        synchronized = {
            arm: self._resample_path(path, arm_steps, self._arm_step_limits[arm]) for arm, path in paths.items()
        }
        actions = []
        for index in range(steps):
            data = self._held_action_data(observation)
            for arm in by_arm:
                if arm in synchronized:
                    # Both arms travel over the same waypoints and arrive
                    # together, then hold while the jaws move.
                    data[f"{arm}_arm_joint_state"] = synchronized[arm][
                        min(index, arm_steps - 1)
                    ]
                if arm in grippers and index >= arm_steps:
                    data[f"{arm}_ee_joint_state"][:] = grippers[arm][1]
            actions.append(
                Action(
                    data=data,
                    meta={"chunk_final": True} if index == steps - 1 else {},
                )
            )

        self._last_targets = dict(requested_poses)
        # Latched after the waypoints are built, so the arm travels to the
        # release pose still holding the previous grip and only lets go on
        # arrival.
        for arm, (_, target) in grippers.items():
            self._gripper_goals[arm] = float(target)
        trace = {
            "tool": "move_eef",
            "mode": "bimanual" if len(by_arm) == 2 else "single_arm",
            "arms": {
                arm: {
                    "named": {
                        f"{arm}_{axis}": value
                        for axis, value in sorted(by_arm[arm].items())
                    },
                    "target_pose": (
                        requested_poses[arm].round(6).tolist()
                        if arm in requested_poses
                        else None
                    ),
                    "executed_pose": (
                        target_poses[arm].round(6).tolist()
                        if arm in target_poses
                        else None
                    ),
                    "source_waypoints": len(paths[arm]) if arm in paths else 0,
                }
                for arm in sorted(by_arm)
            },
            "plan_status": "Success",
            "planned_waypoints": steps,
            "arm_waypoints": arm_steps,
            "gripper_waypoints": gripper_steps,
            "normalized_from_parallel_calls": bool(
                arguments.get("_normalized_from_parallel_calls")
            ),
            "jitter_dx": jx,
            "jitter_dy": jy,
            "jitter_dz": jz,
            "jitter_sphere_r": self._jitter_sphere_r,
            "jitter_sphere_mean_r": self._jitter_sphere_mean_r,
            "jitter_sphere_per_move": self._jitter_sphere_per_move,
        }
        return MotionOutcome(
            chunk=ActionChunk(
                actions=actions,
                control_hz=self.action_spec.control_hz,
                meta={"trace": trace},
            ),
            tool_result=self._accepted_text(steps, gripper_steps, target_poses),
        )

    def _accepted_text(
        self,
        steps: int,
        gripper_steps: int,
        target_poses: Mapping[str, np.ndarray],
    ) -> str:
        text = f"Accepted: playing {steps} waypoints"
        text += f" ({steps / self.action_spec.control_hz:.1f}s)."
        if gripper_steps:
            text += (
                f" The last {gripper_steps} move the gripper, once the arm has "
                "arrived."
            )
        if target_poses:
            text += " The next observation reports how close the grasp point landed."
        return text

    def _parse_targets(
        self, raw: Any
    ) -> tuple[dict[str, dict[str, float]] | None, str | None]:
        if not isinstance(raw, Mapping) or not raw:
            return None, (
                "move_eef.targets must be a non-empty object of dimension "
                "name to value"
            )
        valid = pose.labels()
        by_arm: dict[str, dict[str, float]] = {}
        for raw_label, raw_value in raw.items():
            label = str(raw_label)
            if label not in valid:
                return None, (
                    f"unknown dimension {label!r}; valid names: {', '.join(valid)}"
                )
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                return None, (
                    f"value for {label!r} must be a finite number, got {raw_value!r}"
                )
            value = float(raw_value)
            if not np.isfinite(value):
                return None, (
                    f"value for {label!r} must be a finite number, got {raw_value!r}"
                )
            low, high = self._axis_bounds(label)
            if not low <= value <= high:
                return None, (
                    f"target for {label!r} is outside [{low:.4g}, {high:.4g}]"
                )
            arm, _, axis = label.partition("_")
            by_arm.setdefault(arm, {})[axis] = value
        return by_arm, None

    def _plan_arm(
        self, arm: str, target_pose: np.ndarray
    ) -> tuple[np.ndarray | None, MotionOutcome | None]:
        planned = self._planner(arm=arm, target_pose=target_pose)
        if planned.get("status") != "Success" or planned.get("position") is None:
            status = str(planned.get("status", "Fail"))
            return None, MotionOutcome(
                chunk=None,
                tool_result=(
                    f"pose unreachable (planner status: {status}). The robot did "
                    "not move; choose another pose."
                ),
                repairable=False,
            )
        path = np.asarray(planned["position"], dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 6 or not np.isfinite(path).all():
            raise InfrastructureFailure(
                f"EEF planner returned path shape {path.shape}; expected finite (N, 6)"
            )
        if len(path) == 0:
            raise InfrastructureFailure("EEF planner returned an empty joint path")
        return path, None

    def _declared_step_limits(self) -> np.ndarray:
        """The per-step budget for each of the 14 joint-space dimensions.

        The embodiment's declaration governs. A dimension that declares nothing
        is one whose whole range already fits inside a step, so its range is the
        only bound left to apply.
        """
        labels = self.action_spec.labels
        declared = self.action_spec.max_step or (None,) * len(labels)
        span = self.action_spec.high - self.action_spec.low
        limits = np.asarray(
            [
                span[index] if entry is None else entry
                for index, entry in enumerate(declared)
            ],
            dtype=np.float64,
        )
        if not np.isfinite(limits).all() or bool(np.any(limits <= 0.0)):
            offenders = [
                label for label, limit in zip(labels, limits, strict=True) if limit <= 0.0
            ]
            raise InfrastructureFailure(
                "every action dimension needs a positive per-step budget to "
                f"derive a waypoint count from; {offenders} have none"
            )
        return limits

    def _arm_steps(self, paths: Mapping[str, np.ndarray]) -> int:
        """Use the same normalized path metric for timing and interpolation."""
        steps = 0
        for arm, path in paths.items():
            if len(path) < 2:
                continue
            travel = np.abs(np.diff(path.astype(np.float64), axis=0)) / self._arm_step_limits[arm]
            needed = np.ceil(np.max(travel, axis=1).sum())
            steps = max(steps, int(needed))
        return steps

    def _gripper_steps(self, grippers: Mapping[str, tuple[float, float]]) -> int:
        """Env steps the jaw's setpoint is held for, scaled to its travel."""
        steps = 0
        for arm, (start, target) in grippers.items():
            travel = abs(target - start)
            steps = max(
                steps, int(np.ceil(travel / self._gripper_step_limits[arm]))
            )
        return steps

    @staticmethod
    def _resample_path(path: np.ndarray, steps: int, limits: np.ndarray) -> np.ndarray:
        """Bound every command increment, including nonuniform paths/reversals.

        A segment's duration is its largest joint delta divided by that joint's
        budget. Uniform sampling of cumulative duration makes every joint
        Lipschitz-bounded by its budget. Waypoint index is not elapsed duration.
        """
        distance = np.max(np.abs(np.diff(path.astype(np.float64), axis=0)) / limits, axis=1)
        source = np.r_[0.0, np.cumsum(distance)]
        keep = np.r_[True, np.diff(source) > 0]
        target = np.linspace(0.0, source[-1], steps + 1)[1:]
        return np.stack(
            [np.interp(target, source[keep], path[keep, joint]) for joint in range(6)],
            axis=1,
        ).astype(np.float32)

    def _normalize_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fold several parallel move_eef calls into the one call per turn."""
        if len(tool_calls) < 2:
            return tool_calls
        merged: dict[str, float] = {}
        notes: list[str] = []
        for call in tool_calls:
            function = call.get("function") or {}
            if function.get("name") != "move_eef":
                return tool_calls
            try:
                arguments = json.loads(function.get("arguments") or "")
            except (TypeError, json.JSONDecodeError):
                return tool_calls
            targets = arguments.get("targets")
            if not isinstance(targets, Mapping):
                return tool_calls
            for label, value in targets.items():
                if str(label) in merged:
                    return tool_calls
                merged[str(label)] = value
            note = arguments.get("note")
            if isinstance(note, str) and note.strip():
                notes.append(note.strip())
        combined: dict[str, Any] = {
            "targets": merged,
            "note": " | ".join(notes),
            "_normalized_from_parallel_calls": True,
        }
        return [
            {
                "id": str(tool_calls[0].get("id") or "bimanual-call"),
                "type": "function",
                "function": {
                    "name": "move_eef",
                    "arguments": json.dumps(combined),
                },
            }
        ]

    @staticmethod
    def _repair(message: str) -> MotionOutcome:
        return MotionOutcome(chunk=None, tool_result=message, repairable=True)

    def _held_action_data(self, observation: Observation) -> dict[str, np.ndarray]:
        """Hold the arms where they are and the jaws on what they were told.

        The jaw cannot hold by echoing its own reading. RoboDojo reports
        ``<arm>_ee_joint_state`` from the previous command after its
        20%-of-range clamp, not from the finger position, so re-sending it
        parks the jaw at the width it already has: the position error, and with
        it the grip force, goes to zero and the object slides out of a closed
        hand during the next transfer. Re-sending the value the model last
        asked for keeps the clamp a full 20% ahead of the fingers, which is the
        same squeeze the closing phase applied. Until the model has named a
        jaw, the reading is all there is.
        """
        data = {
            name: np.asarray(observation.state[name], dtype=np.float32).copy()
            for name in (
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            )
        }
        for arm, goal in self._gripper_goals.items():
            data[f"{arm}_ee_joint_state"][:] = goal
        return data

    def audit_config(self) -> dict[str, Any]:
        config = super().audit_config()
        config["adapter"] = "robodojo-agent-l3-inspect-eef"
        # The base adapter hashes only itself. This surface is the other half
        # of what drives the run, so a trace from it has to cover both.
        config["code"] = source_revision(Path(__file__).resolve().parent)
        labels = pose.labels()
        config["embodiment"] = {
            **config["embodiment"],
            "labels": list(labels),
            "low": [self._axis_bounds(label)[0] for label in labels],
            "high": [self._axis_bounds(label)[1] for label in labels],
            # A Cartesian dimension has no honest per-step budget: how far one
            # env step carries it depends on the arm's pose. The budget that
            # governs is declared in joint space and reported below.
            "max_step": None,
            "rotation_reference_quat_wxyz": list(pose.TOP_DOWN_QUAT_WXYZ),
            # Per dimension, where the reference the absolute targets are
            # measured against was read from. Aligned with "labels".
            "state_reference_keys": list(pose.state_reference_keys()),
            "joint_labels": list(self.action_spec.labels),
            # The per-step budget every waypoint count is derived from, so a
            # trace shows what actually governed the run. Aligned with
            # "joint_labels".
            "joint_max_step": [float(value) for value in self._declared_step_limits()],
        }
        config["policy_config"]["model_action_space"] = "eef"
        config["policy_config"]["max_duration_s"] = self._max_duration_s
        config["policy_config"]["gripper_dwell_step"] = self._gripper_dwell_step
        config["policy_config"]["negate_xyz"] = self._negate_xyz
        config["policy_config"]["jitter_dx"] = self._jitter_offset()[0]
        config["policy_config"]["jitter_dy"] = self._jitter_offset()[1]
        config["policy_config"]["jitter_dz"] = self._jitter_offset()[2]
        config["policy_config"]["jitter_sphere_r"] = self._jitter_sphere_r
        config["policy_config"]["jitter_sphere_mean_r"] = self._jitter_sphere_mean_r
        config["policy_config"]["jitter_sphere_per_move"] = (
            self._jitter_sphere_per_move
        )
        return config
