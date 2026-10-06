"""Shared host execution for the single Gymnasium step(code) interface."""

from __future__ import annotations

import contextlib
import copy
import io
import traceback

from XPolicyLab.policy.EmbodiedRSI.runtime.execution import Environment
from XPolicyLab.policy.EmbodiedRSI.runtime.execution.python import NUMPY, SAFE_BUILTINS, validate_snippet
from XPolicyLab.policy.EmbodiedRSI.runtime.execution.primitives import ObservationSpace, rgb_images


class CodeTask(Environment):
    """Execute code with only the selected native policy entrypoints."""

    def initialize(self, api):
        self._api = api
        self._exec_globals = {}
        self._observation = None
        self._frames = {}
        self.success = False
        self.terminated = False
        self.truncated = False
        self.control_steps = 0

    def start_episode(self, seed, obs):
        self.begin_episode(seed)
        self.control_steps = 0
        self._exec_globals = {}
        self._frames = {}
        self.success = self.terminated = False
        self.truncated = self.max_steps is not None and self.control_steps >= self.max_steps
        self.observe(obs)
        self.observation_space = ObservationSpace(obs)
        return self.public_observation(), {"task_prompt": self.task_prompt()}

    def reset_from_program(self):
        raise PermissionError("reset() is forbidden in Test")

    def observe(self, obs):
        self._observation = copy.deepcopy(obs)
        for camera, frame in rgb_images(obs).items():
            self._frames.setdefault(camera, []).append(frame.copy())

    def public_observation(self):
        if self._observation is None:
            raise RuntimeError("Call reset before reading observations")
        return copy.deepcopy(self._observation)

    def task_prompt(self):
        return getattr(self, "_task_prompt", "")

    @property
    def done(self):
        return self.terminated or self.truncated

    def transition(self, reward=0.0):
        # Reward is host bookkeeping. Policies receive only native success.
        return (
            self.public_observation(),
            float(reward),
            self.terminated,
            self.truncated,
            {"success": self.success},
        )

    def _step(self, code):
        stdout, stderr = io.StringIO(), io.StringIO()
        ok = True
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                validate_snippet(code)
                self._exec_globals.update(__builtins__=dict(SAFE_BUILTINS), np=NUMPY)
                self._exec_globals.update(self._api.functions())
                exec(compile(code, "<agent-program>", "exec"), self._exec_globals)
            except Exception:
                ok = False
                traceback.print_exc()
        frames, self._frames = self._frames, {}
        obs, reward, terminated, truncated, _ = self.transition(self.reward())
        return self.finish_step(
            obs,
            reward,
            terminated,
            truncated,
            {
                "ok": ok,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "native_steps": self.control_steps,
            },
            frames=frames,
            success=self.success,
        )

    def reward(self):
        return float(self.success)

    def render(self):
        frames = rgb_images(self.public_observation())
        return next(iter(frames.values())).copy()
