"""Original jaw planning and persistent command semantics; native bridge only."""
from __future__ import annotations

from typing import Any

import numpy as np


class BaseGripperMotion:
    def _gripper_controls(self, normalized_value: float, steps: int, robot: Any | None = None) -> list[dict[str, Any]]:
        robot = robot or self.robot
        return self.rm.plan_endeffector_joint(
            env_idx=0,
            arm_tag=robot.arm_name,
            target_val=float(np.clip(normalized_value, 0.0, 1.0)),
            need_plan=True,
            control_step_num=max(1, int(steps)),
        )


class GripperMotion(BaseGripperMotion):
    def _official_motion_profile(self):
        return bool(getattr(self.args, "official_eval", False))

    def _gripper_controls(self, normalized_value, steps, robot=None):
        robot = robot or self.robot
        value = float(np.clip(normalized_value, 0., 1.))
        unchanged = self._gripper_targets.get(robot.arm_name) == value
        self._gripper_targets[robot.arm_name] = value
        if unchanged:
            # Interpolating again from the measured jaw width releases the
            # contact preload at every segment boundary. Hold the commanded
            # target continuously until an actual open/close transition.
            command = super()._gripper_controls(value, 1, robot)
            return command * max(1, int(steps))
        return super()._gripper_controls(value, steps, robot)

    def _execute(self, controls, phase):
        if not controls:
            return
        if self._official_motion_profile():
            held_grippers = {}
            for arm, value in self._gripper_targets.items():
                held_grippers.update(super()._gripper_controls(value, 1, self.robots[arm])[0])
            # Reassert idle jaws' commanded positions, rather than their
            # contact-displaced measurements. An explicit active-hand ramp
            # takes precedence over these persistent targets.
            controls = [{**held_grippers, **control} for control in controls]
        return self._official_bridge.execute(controls, phase)

    def _grip(self, arm, value, phase):
        self.phase = phase
        seconds = self.args.gripper_settle_seconds
        if self._official_motion_profile() and phase.endswith("_lift_settle"):
            # Closure already settled before lifting. Keep a short loaded
            # hold before measuring the grasp, with unchanged lift/slip gates.
            seconds = min(seconds, .12)
        steps = max(10, round(seconds / self.env.dt))
        self._execute(self._gripper_controls(value, steps, self.robots[arm]), phase)


__all__ = ('BaseGripperMotion', 'GripperMotion')
