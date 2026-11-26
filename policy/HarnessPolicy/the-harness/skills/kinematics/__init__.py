"""Forward and inverse kinematics and motion planning, through the robot's kinematics."""

import math


# ik: a solution that differs from the current joints by a full turn is the same
# angle on another branch and is turned back toward them (within the joint limits).
# One that still swings a joint more than IK_FLIP_JUMP_RAD for a target within
# IK_FLIP_NEAR_M and IK_FLIP_TURN_RAD of the current link6 pose is a branch flip: a
# small Cartesian step would become a large joint motion, so it counts as no solution.
IK_FLIP_JUMP_RAD, IK_FLIP_NEAR_M, IK_FLIP_TURN_RAD = 2.0, 0.05, 0.5


class Kinematics:
    def fk(self, joints=None):
        """Link6 poses of both arms, {"left_arm": pose, "right_arm": pose}, for 14 joint
        values (the robot's current joints when None)."""
        return self.env.call("fk", command=list(joints) if joints is not None else None)

    def ik(self, arm, pose):
        """6 joints placing link6 at `pose`, the solution nearest the current joints (the
        solver starts from them), or None. A solution a full turn away on some joint is
        turned back; a branch flip (a large joint swing for a small change of pose) is
        no solution."""
        group = self._arm(arm)
        pose = [float(v) for v in pose]
        joints = self.env.call("ik", episode_id=self.episode_id, arm=arm, pose=pose)["joints"]
        return None if joints is None else self._unflipped(arm, group, joints, pose)

    def _unflipped(self, arm, group, joints, pose):
        """`joints` turned by whole turns toward the current joints where the joint
        limits allow; None when it is still a branch flip (see IK_FLIP_JUMP_RAD)."""
        state = getattr(self, "state", None) or {}
        if not state.get("joint_position"):
            return [float(v) for v in joints]
        profile = getattr(self, "profile", None) or {}
        low, high = profile.get("action_low"), profile.get("action_high")
        result = []
        for index, value in zip(group["joints"], joints):
            value, now = float(value), float(state["joint_position"][index])
            while value - now > math.pi and (low is None or value - 2 * math.pi >= low[index]):
                value -= 2 * math.pi
            while now - value > math.pi and (high is None or value + 2 * math.pi <= high[index]):
                value += 2 * math.pi
            result.append(value)
        jump = max(abs(value - float(state["joint_position"][index])) for index, value in zip(group["joints"], result))
        here = state.get(f"{arm.split('_')[0]}_ee_pose")
        if jump > IK_FLIP_JUMP_RAD and here and len(pose) == 7:
            dot = abs(sum(float(a) * float(b) for a, b in zip(here[3:7], pose[3:7])))
            dot /= max(math.sqrt(sum(float(a) ** 2 for a in here[3:7]) * sum(b * b for b in pose[3:7])), 1e-12)
            turn = 2 * math.acos(min(1.0, dot))
            if math.dist(here[:3], pose[:3]) < IK_FLIP_NEAR_M and turn < IK_FLIP_TURN_RAD:
                return None
        return result

    def plan_path(self, arm, pose=None, joints=None):
        """A path for `arm` from its current joints to a link6 `pose` or to 6 `joints`
        (exactly one): 6-joint waypoints at the control rate, or None when planning fails.
        The planner avoids self collision and the table only, not objects or the other arm."""
        self._arm(arm)
        result = self.env.call("plan", episode_id=self.episode_id, arm=arm,
                               pose=None if pose is None else [float(v) for v in pose],
                               joints=None if joints is None else [float(v) for v in joints])
        return result["waypoints"] if result["success"] else None
