"""A persistent Python session that yields actions to an external evaluator."""

import copy

import numpy as np

from simulators.execution import CodeTask
from simulators.robodojo_api import RoboDojoLowLevelApi


def policy_observation(obs):
    # No evaluator metadata, layout identity, score or ground truth crosses
    # this boundary. These are the existing handwritten primitive's fields.
    return copy.deepcopy({k: obs[k] for k in
                          ("instruction", "vision", "state", "action", "additional_info")
                          if k in obs})


def validate_action(action, state):
    if not isinstance(action, dict) or not action:
        raise ValueError("Expected a complete native action dictionary")
    action = {k: np.asarray(v, dtype=np.float64) for k, v in action.items()}
    ee = any(k.endswith("ee_pose") for k in action)
    required = {k for k in state if k.endswith("ee_joint_state") or
                k.endswith("ee_pose" if ee else "arm_joint_state")}
    if set(action) != required:
        raise ValueError(f"Expected action keys {sorted(required)}")
    for key, value in action.items():
        if value.ndim != 1 or value.shape != np.asarray(state[key]).shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape or nonfinite target: {key}")
    return action


class BridgeTask(CodeTask):
    def __init__(self, connection, observation, step_limit, *, phase="test"):
        super().__init__()
        # NumPy lazily imports its formatters using the caller's import hook.
        # Initialize them before snippets run with restricted builtins.
        str(np.zeros(1))
        repr(np.zeros(1))
        self.connection = connection
        self.initial = policy_observation(observation)
        self.native_step_limit = self.max_steps = int(step_limit)
        self.phase = phase
        self.initialize(RoboDojoLowLevelApi(self))

    def reset(self, *, seed=None, options=None):
        if self._observation is not None:
            if self.phase != "playground" or not self._restoring_scene:
                raise PermissionError("reset() is forbidden in Test")
            self.connection.send({"kind": "reset"})
            reply = self.connection.recv()
            if reply.get("kind") != "reset":
                raise RuntimeError("Native Playground reset failed")
            self.max_steps = self.control_steps + self.native_step_limit
            return self.start_episode(seed, policy_observation(reply["observation"]))
        self._task_prompt = self.initial["instruction"]
        return self.start_episode(seed, self.initial)

    def control_step(self, action):
        if not self.done:
            action = validate_action(action, self._observation["state"])
            self.connection.send({"kind": "action", "action": action})
            transition = self.connection.recv()
            if transition.get("kind") != "transition":
                raise RuntimeError("Evaluator stopped while a native action was pending")
            self.control_steps += 1
            self.success = bool(transition["success"])
            self.terminated = bool(transition["terminated"])
            self.truncated = bool(transition["truncated"])
            self.observe(policy_observation(transition["observation"]))
        return self.transition(float(self.success))

    def close(self):
        pass
