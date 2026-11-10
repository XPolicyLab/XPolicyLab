"""Shared motion with optional public-shape cross-arm checks."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from orimapi.skills.code_skills.play_tic_tac_toe.motion import Motion as BaseMotion, down_rotation
from .geometry import verify_pose


class Motion(BaseMotion):
    def __init__(self, geometry, emit, clearance=None, *, require_cartesian_completion=False):
        if type(require_cartesian_completion) is not bool:
            raise ValueError("Cartesian completion option must be boolean")
        self.require_cartesian_completion = require_cartesian_completion
        self.clearance = clearance
        super().__init__(geometry, emit)

    def reset(self):
        super().reset()
        self.cartesian_completion = None

    def step(self, observation):
        """Do not retire a move while its existing TCP verification still fails.

        Keep the same target, joint-step limits and bounded tracking allowance.
        A residual can be lag, contact or another tracking issue; this neither
        diagnoses contact nor opens a hand or retries an altered target.
        """
        # Existing non-pinned controllers and private motion subclasses keep
        # their prior behavior unless their frozen configuration opts in.
        if not self.require_cartesian_completion:
            return super().step(observation)
        previous_index = self.index
        previous_start = self.operation_start
        original_emit = self.emit
        pending = []
        self.emit = lambda *args, **kwargs: pending.append((args, kwargs))
        try:
            action = super().step(observation)
        finally:
            self.emit = original_emit
        if (
            self.failure is None and self.plan is not None
            and 0 <= previous_index < len(self.plan)
            and self.index == previous_index + 1
        ):
            row = self.plan[previous_index]
            if row["op"] in {"move", "dual_move"}:
                poses = (
                    {row["arm"]: row}
                    if row["op"] == "move"
                    else {side: row[side] for side in ("left", "right")}
                )
                checks = {}
                for side, target in poses.items():
                    goal = np.eye(4)
                    goal[:3, :3] = Rotation.from_euler(
                        "XY", [target.get("tilt_x", 0), target.get("tilt_y", 0)]
                    ).as_matrix() @ down_rotation(target.get("yaw", 0))
                    goal[:3, 3] = target["position"]
                    actual = self.geometry.fk(
                        side, observation.robot_state[f"{side}_arm_joint_state"]
                    )
                    checks[side] = verify_pose(actual, goal)
                deferred = any(check["status"] != "pass" for check in checks.values())
                self.cartesian_completion = {
                    "step": observation.step, "deferred": deferred, "checks": checks,
                    "scope": "Existing observed TCP verification, not contact or retention evidence",
                }
                if deferred:
                    self.index = previous_index
                    self.operation_start = (
                        previous_start if previous_start is not None else observation.step
                    )
                    self.hold = None
                    pending = [
                        (args, kwargs) for args, kwargs in pending
                        if not args or args[0] not in {"stage_completed", "program_completed"}
                    ]
                    pending.append((("cartesian_completion_deferred",), {
                        "step": observation.step, "index": previous_index,
                        "checks": checks,
                        "scope": "Retain the unchanged target within the original bounded tracking allowance",
                    }))
        for args, kwargs in pending:
            original_emit(*args, **kwargs)
        return action

    def preflight(self, plan, observation, budget=None):
        report = super().preflight(plan, observation, budget)
        if not any(row["op"] == "home" for row in plan):
            return report
        joints = {
            side: np.asarray(
                observation.robot_state[f"{side}_arm_joint_state"], dtype=float
            ).copy()
            for side in ("left", "right")
        }
        home = self.home or observation.robot_state
        targets = {
            side: np.asarray(home[f"{side}_arm_joint_state"], dtype=float)
            for side in joints
        }
        # A preview must not mutate the live other-arm observation.
        saved_static = self.clearance.static if self.clearance is not None else None
        checked = 0
        try:
            for index, row in enumerate(plan):
                if row["op"] != "home":
                    if self.clearance is not None:
                        self.clearance.static = {
                            side: self.clearance.objects(side, value)
                            for side, value in joints.items()
                        }
                    _, joints = self._preflight(
                        [row], observation, initial_joints=joints
                    )
                    continue
                limit = row.get("max_joint_step", self.max_joint_step)
                if (
                    type(limit) not in (int, float)
                    or not math.isfinite(limit)
                    or limit <= 0
                ):
                    raise ValueError("Home joint step must be positive and finite")
                count = max(
                    1,
                    math.ceil(
                        max(float(np.max(np.abs(targets[s] - joints[s]))) for s in joints)
                        / limit
                    ),
                )
                if count > 10000:
                    raise ValueError("Nominal home path exceeds bounded preflight")
                for sample in range(count):
                    # Match home.step: each joint independently advances by at
                    # most limit, rather than one straight, normalized q-line.
                    if self.clearance is not None:
                        self.clearance.static = {
                            side: self.clearance.objects(side, value)
                            for side, value in joints.items()
                        }
                    proposed = {
                        side: current + np.clip(targets[side] - current, -limit, limit)
                        for side, current in joints.items()
                    }
                    for side, value in proposed.items():
                        try:
                            self._guard(side, value)
                        except ValueError as error:
                            raise ValueError(
                                f"Home preflight stage {index}, sample {sample + 1}, "
                                f"{side} rejected: {error}. No home action was issued. "
                                "Choose a newly checked clear staging route; do not "
                                "repeat this home or relax a guard."
                            ) from error
                    joints = proposed
                    checked += 1
        finally:
            if self.clearance is not None:
                self.clearance.static = saved_static
        report["home_path_preflight"] = {
            "nominal_samples": checked,
            "scope": (
                "Nominal clipped-joint home path under existing public robot/table "
                "guards only; not full scene/payload clearance or tracking proof. "
                "Every actual proposed action remains guarded."
            ),
        }
        return report

    def _guard(self, side, joints):
        try:
            super()._guard(side, joints)
        except ValueError as error:
            if (
                self.clearance is None
                or str(error) != "Arm enters the other arm's workspace"
            ):
                raise
        if self.clearance is not None:
            self.clearance.check(side, joints)
