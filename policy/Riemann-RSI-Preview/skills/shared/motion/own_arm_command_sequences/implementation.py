"""Own-robot command sequences shared by the three mature rule imports."""
from __future__ import annotations
from typing import Any
import numpy as np
from skills.shared.geometry.native_rigid_pose.implementation import as_numpy

class OwnArmCommandSequences:

    def _actual_ee_pose(self, robot: Any) -> np.ndarray:
        pose = self.rm.get_real_endpose(robot, env_idx_list=[0], is_relative=True)[0]
        return as_numpy(pose).reshape(7).astype(np.float32)

    def _gripper_controls(self, normalized_value: float, steps: int, robot: Any | None=None) -> list[dict[str, Any]]:
        robot = robot or self.robot
        return self.rm.plan_endeffector_joint(env_idx=0, arm_tag=robot.arm_name, target_val=float(np.clip(normalized_value, 0.0, 1.0)), need_plan=True, control_step_num=max(1, int(steps)))

    @staticmethod
    def _merge_controls(*sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not sequences:
            return []
        n = max((len(seq) for seq in sequences))
        result: list[dict[str, Any]] = []
        for i in range(n):
            merged: dict[str, Any] = {}
            for seq in sequences:
                if not seq:
                    continue
                merged.update(seq[min(i, len(seq) - 1)])
            result.append(merged)
        return result

    def _hold_gripper(self, normalized_value: float, phase: str, seconds: float=0.8) -> None:
        steps = max(10, int(round(seconds / float(self.env.dt))))
        self._execute(self._gripper_controls(normalized_value, steps), phase)

    def _hold_both_grippers(self, normalized_value: float, phase: str, seconds: float=0.8) -> None:
        steps = max(10, int(round(seconds / float(self.env.dt))))
        controls = self._merge_controls(*[self._gripper_controls(normalized_value, steps, robot) for robot in self.robots.values()])
        self._execute(controls, phase)
