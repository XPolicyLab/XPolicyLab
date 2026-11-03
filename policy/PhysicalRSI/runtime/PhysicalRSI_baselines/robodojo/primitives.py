"""Observable-state primitives producing complete RoboDojo dual-arm actions.

Movement requires an injected collision-aware planner; no straight-line joint
fallback. Native perception, planner and control adapters must honor deadlines.
"""

import math
import inspect
import time
from copy import deepcopy

from PhysicalRSI_core.contracts import Contract


def vector(value, size, name):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != size
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
    ):
        raise ValueError(name + " must contain finite real values of the declared size")
    return list(value)


class RoboDojoPrimitives:
    def __init__(
        self,
        *,
        observe,
        plan,
        step,
        dimensions,
        joint_limits,
        max_joint_step,
        timeout_s=30,
        max_waypoints=500,
        gripper_steps=5,
        perceive=None,
    ):
        expected = {
            side + suffix
            for side in ("left", "right")
            for suffix in ("_arm_joint_state", "_ee_joint_state")
        }
        if set(dimensions) != expected or any(
            type(n) is not int or n < 1 for n in dimensions.values()
        ):
            raise ValueError("Explicit dual-arm action dimensions required")
        if (
            not math.isfinite(max_joint_step)
            or max_joint_step <= 0
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
            or type(max_waypoints) is not int
            or max_waypoints < 1
            or type(gripper_steps) is not int
            or gripper_steps < 1
        ):
            raise ValueError("Positive primitive execution limits required")
        if perceive is not None and not callable(perceive):
            raise ValueError("Perception callback must be callable")
        self.perceive = perceive
        self.observe, self.plan, self.step = observe, plan, step
        self.dimensions = dict(dimensions)
        self.joint_limits = deepcopy(joint_limits)
        for side in ("left", "right"):
            n = dimensions[side + "_arm_joint_state"]
            low = vector(joint_limits[side]["low"], n, "joint lower limits")
            high = vector(joint_limits[side]["high"], n, "joint upper limits")
            if any(a >= b for a, b in zip(low, high)):
                raise ValueError("Joint limit interval is empty")
        self.max_joint_step, self.timeout_s = max_joint_step, timeout_s
        self.max_waypoints, self.gripper_steps = max_waypoints, gripper_steps
        self.handlers = dict(
            approach=self.approach, gripper=self.gripper, reobserve=self.reobserve
        )

        if perceive is not None:
            self.handlers["scene"] = self.scene

    def bindings(self, *, revision, actor):
        """Client contracts; revision must pin planner/control/config dependencies."""

        def project(result):
            state = {
                key: vector(result["state"][key], n, key)
                for key, n in self.dimensions.items()
            }
            return dict(state=state)

        def validate(function):
            signature = inspect.signature(function)
            return lambda args, kwargs: dict(signature.bind(*args, **kwargs).arguments)

        bindings = {
            name: dict(
                revision=revision,
                input=Contract("robodojo." + name + "/v1", embodiment="dual-arm"),
                output=Contract("robodojo.proprioception/v1", embodiment="dual-arm"),
                effects=[actor],
                validate_input=validate(function),
                project_output=project,
            )
            for name, function in self.handlers.items()
        }
        if "scene" in bindings:
            from .perception import project_scene

            bindings["scene"].update(
                output=Contract(
                    "robodojo.scene/v1", unit="m", frame="world", embodiment="dual-arm"
                ),
                project_output=project_scene,
            )
        return bindings

    def _state(self, deadline):
        self._check(deadline)
        raw = self.observe(deadline=deadline)
        state = {key: vector(raw[key], n, key) for key, n in self.dimensions.items()}
        for side in ("left", "right"):
            self._joints(side, state[side + "_arm_joint_state"])
            if any(x < 0 or x > 1 for x in state[side + "_ee_joint_state"]):
                raise ValueError("Expected normalized gripper state in [0, 1]")
        self._check(deadline)
        return state  # Deliberate field projection; no scores, layouts or simulator objects.

    @staticmethod
    def _check(deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError("Primitive execution deadline reached")

    def _joints(self, arm, values):
        if arm not in ("left", "right"):
            raise ValueError("Arm must be left or right")
        values = vector(
            values, self.dimensions[arm + "_arm_joint_state"], "joint position"
        )
        limits = self.joint_limits[arm]
        if any(
            x < a or x > b for x, a, b in zip(values, limits["low"], limits["high"])
        ):
            raise ValueError("Joint position outside configured limits")
        return values

    def scene(self):
        from .perception import project_scene

        if self.perceive is None:
            raise RuntimeError("No perception capability configured")
        deadline = time.monotonic() + self.timeout_s
        state = self._state(deadline)
        result = self.perceive(state=state, deadline=deadline)
        self._check(deadline)
        return project_scene(result)

    def reobserve(self):
        return {"state": self._state(time.monotonic() + self.timeout_s)}

    def approach(self, *, arm, position_m, quaternion_wxyz):
        if arm not in ("left", "right"):
            raise ValueError("Arm must be left or right")
        position = vector(position_m, 3, "world position in metres")
        quaternion = vector(quaternion_wxyz, 4, "world quaternion wxyz")
        if abs(sum(x * x for x in quaternion) - 1) > 1e-4:
            raise ValueError("Expected a unit quaternion in wxyz order")
        deadline = time.monotonic() + self.timeout_s
        initial = self._state(deadline)
        key = arm + "_arm_joint_state"
        path = self.plan(
            arm=arm,
            start=initial[key],
            position_m=position,
            quaternion_wxyz=quaternion,
            state=deepcopy(initial),
            deadline=deadline,
        )
        if (
            not isinstance(path, (list, tuple))
            or not 1 <= len(path) <= self.max_waypoints
        ):
            raise ValueError("Planner must return a bounded nonempty joint trajectory")
        path = [self._joints(arm, row) for row in path]
        previous = initial[key]
        for row in path:
            if any(abs(a - b) > self.max_joint_step for a, b in zip(previous, row)):
                raise ValueError("Planner trajectory exceeds configured joint step")
            previous = row
        for row in path:
            action = self._state(deadline)
            if any(abs(a - b) > self.max_joint_step for a, b in zip(action[key], row)):
                raise ValueError("Observed robot diverged from planned trajectory")
            action[key] = list(row)
            self.step(action, deadline=deadline)
        return {"state": self._state(deadline), "action_steps": len(path)}

    def gripper(self, *, arm, command):
        if (
            arm not in ("left", "right")
            or type(command) not in (int, float)
            or not math.isfinite(command)
            or not 0 <= command <= 1
        ):
            raise ValueError("Expected an arm and normalized gripper command in [0, 1]")
        deadline = time.monotonic() + self.timeout_s
        key = arm + "_ee_joint_state"
        for _ in range(self.gripper_steps):
            action = self._state(deadline)
            action[key] = [command] * self.dimensions[key]
            self.step(action, deadline=deadline)
        return {"state": self._state(deadline), "action_steps": self.gripper_steps}
