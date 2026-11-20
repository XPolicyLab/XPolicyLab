"""Mature home, pose planning and bounded feedback with public joint limits."""
from __future__ import annotations

from typing import Any

import numpy as np

from skills.shared.motion.limited_target_gripper_motion.implementation import GripperMotion
from skills.shared.geometry.rigid_pose_math.implementation import (
    as_numpy, mat_to_quat_wxyz, quat_to_mat_wxyz, quaternion_angle_deg,
)


class PosePlanningError(RuntimeError):
    """The planner explicitly reported an unreachable or unplannable pose."""


class PoseFeedbackError(RuntimeError):
    """A precise pose exhausted feedback corrections without converging."""

    def __init__(self, phase, position_error, rotation_error):
        self.phase = phase
        self.position_error = float(position_error)
        self.rotation_error = float(rotation_error)
        super().__init__(f"Pose feedback did not converge at {phase}: "
                         f"{self.position_error:.4f} m, {self.rotation_error:.2f} deg")


class NeutralArmMotion(GripperMotion):
    def _plan_joint_home(self, robot: Any) -> list[dict[str, Any]]:
        print(f"[data-engine] cuRobo plan_joint begin arm={robot.arm_name}", flush=True)
        current = self.rm.get_joint(robot, env_idx_list=[0])[0]
        result = self.rm.planner[robot.robot_name].plan_joint(current, self.home_joints[robot.arm_name])
        print(f"[data-engine] cuRobo plan_joint result={result.get('status')}", flush=True)
        if result.get("status") != "Success":
            raise RuntimeError(f"cuRobo plan_joint failed for {robot.arm_name}")
        controls = self.rm.plan_ee(env_idx=0, arm_tag=robot.arm_name, result=result)
        return controls or []

    @staticmethod
    def _merge_controls(*sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not sequences:
            return []
        n = max(len(seq) for seq in sequences)
        result: list[dict[str, Any]] = []
        for i in range(n):
            merged: dict[str, Any] = {}
            for seq in sequences:
                if not seq:
                    continue
                merged.update(seq[min(i, len(seq) - 1)])
            result.append(merged)
        return result

    def _plan(self, arm, pose):
        robot = self.robots[arm]
        try:
            seq = self._plan_pose_for(robot, pose)
            current = as_numpy(self.rm.get_joint(robot, env_idx_list=[0])[0])
            goal = as_numpy(seq[-1][self.rm.process_name(arm)]["position"])
            # A pose solution may wind an elbow through an unnecessary full
            # revolution. Re-plan an equivalent, nearby joint endpoint.
            if np.any(np.abs(goal - current) > np.pi):
                raise RuntimeError("Pose plan uses a distant joint winding; retry canonical IK")
            return seq
        except RuntimeError as error:
            # The shared planner reports unsuccessful paths with this exact
            # prefix. Do not reinterpret CUDA, recorder or other runtime faults
            # as a reachability failure that another arm could recover.
            if not (str(error).startswith("cuRobo plan_path failed for pose=") or
                    str(error) == "Pose plan uses a distant joint winding; retry canonical IK"):
                raise
            # Equivalent +/-2pi IK solutions must respect the USD joint
            # limits; this is also needed by the existing swap_T engine.
            ik = self._ik(arm, pose)
            if ik.get("status") != "Success":
                raise PosePlanningError(str(error)) from error
            current = as_numpy(self.rm.get_joint(robot, env_idx_list=[0])[0])
            goal = as_numpy(ik["joint_value"]).reshape(-1)
            limits = self.rm.joint_limits[arm]
            for j in range(len(goal)):
                choices = [goal[j] + k * 2 * np.pi for k in range(-3, 4)
                           if limits[j, 0] <= goal[j] + k * 2 * np.pi <= limits[j, 1]]
                if not choices:
                    raise PosePlanningError(f"No in-limit IK solution: {arm}, joint {j}")
                goal[j] = min(choices, key=lambda v: abs(v - current[j]))
            result = self.rm.planner[robot.robot_name].plan_joint(current, goal)
            if result.get("status") != "Success":
                raise PosePlanningError(f"cuRobo canonical IK planning failed: {arm}, {self.phase}")
            return self.rm.plan_ee(env_idx=0, arm_tag=arm, result=result)

    def _move(self, arm, target, phase, grip, precise=False, insertion=False, initial_plan=None):
        self.phase = phase
        target = np.asarray(target, dtype=np.float32)
        command = target.copy()
        robot = self.robots[arm]
        attempts = self.args.pose_corrections + 1 if precise else 1
        position_tolerance = self.args.insertion_tolerance if insertion else self.args.pose_tolerance
        rotation_tolerance = .5 if insertion else 3.
        for attempt in range(attempts):
            seq = initial_plan if attempt == 0 and initial_plan is not None else self._plan(arm, command)
            if not seq:
                raise RuntimeError(f"Empty cuRobo trajectory at {phase}")
            self._execute(self._merge_controls(seq, self._gripper_controls(grip, len(seq), robot)),
                          phase if attempt == 0 else f"{phase}_correct_{attempt}")
            actual = self._actual_ee_pose(robot)
            dp = target[:3] - actual[:3]
            angle = quaternion_angle_deg(target[3:], actual[3:])
            print(f"[fill-egg-holder] {phase} arm={arm} position_error={np.linalg.norm(dp):.5f} "
                  f"rotation_error={angle:.2f} target={target.tolist()} actual={actual.tolist()}", flush=True)
            if not precise or (np.linalg.norm(dp) <= position_tolerance and angle <= rotation_tolerance):
                return
            if np.linalg.norm(dp) > .04 or angle > 20.:
                raise RuntimeError(f"Large tracking error at {phase}; stopping feedback: "
                                   f"{np.linalg.norm(dp):.3f} m / {angle:.1f} deg")
            # Correct model/simulator mismatch with measured SE(3) error;
            # never use the align_blocks servo, which forcibly closes jaws.
            command[:3] += dp
            rotation = (quat_to_mat_wxyz(target[3:]) @ quat_to_mat_wxyz(actual[3:]).T
                        @ quat_to_mat_wxyz(command[3:]))
            command[3:] = mat_to_quat_wxyz(rotation)
        raise PoseFeedbackError(phase, np.linalg.norm(dp), angle)
