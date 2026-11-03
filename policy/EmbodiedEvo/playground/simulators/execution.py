"""Shared host execution for the single Gymnasium step(code) interface."""

from __future__ import annotations

import contextlib
import copy
import io
import traceback

from environments import Environment
from simulators.interactive import NUMPY, SAFE_BUILTINS, validate_snippet
from simulators.primitives import ObservationSpace, rgb_images


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
        self._restoring_scene = False
        self._reset_seed = None
        self._reset_options = None
        self._reset_info = {}

    def start_episode(self, seed, obs, *, reset_info=None, reset_options=None):
        if not self._restoring_scene:
            self.begin_episode(seed)
            self.control_steps = 0
            self._exec_globals = {}
            self._frames = {}
        self._reset_seed = seed
        self._reset_options = copy.deepcopy(reset_options)
        self._reset_info = copy.deepcopy(reset_info or {})
        self.success = self.terminated = False
        self.truncated = self.max_steps is not None and self.control_steps >= self.max_steps
        self.observe(obs)
        self.observation_space = ObservationSpace(obs)
        return self.public_observation(), {"task_prompt": self.task_prompt()}

    def reset_from_program(self):
        """Restore the current scene without granting a new execution budget."""
        if self.execution_phase != "playground":
            raise PermissionError("reset() is only allowed in PlayGround; forbidden in Test")
        if self._observation is None or self._reset_seed is None:
            raise RuntimeError("The host must initialize a seeded scene before reset()")
        self._restoring_scene = True
        try:
            obs, _ = self.reset(
                seed=self._reset_seed, options=copy.deepcopy(self._reset_options)
            )
            return obs, copy.deepcopy(self._reset_info)
        except Exception:
            # A failed rebuild must not leave a partially initialized scene live.
            self.truncated = True
            raise
        finally:
            self._restoring_scene = False

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
