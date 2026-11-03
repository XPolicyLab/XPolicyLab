"""Temporary head-camera cover driven only by acknowledged measured states."""
from collections import deque
from dataclasses import dataclass
import math
import secrets

import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    try:
        result = int(value)
        # Do not round large replay seeds through float (which loses bits >2**53).
        valid = (isinstance(value, str) or value == result) and result >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return result


def _positive(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return result


@dataclass(frozen=True)
class StallSettings:
    fix: str
    steps: int
    cover_chunks: int
    arm_threshold: float
    gripper_threshold: float
    action_hz: float
    seed: int | None

    @classmethod
    def from_config(cls, config, native_hz):
        fix = str(config.get('phoenix_stall_fix') or 'none').lower()
        if fix not in ('none', 'cover'):
            raise ValueError('phoenix_stall_fix must be none or cover')
        hz = _positive(config.get('action_hz') or native_hz, 'action_hz')
        steps = config.get('phoenix_stall_steps')
        if steps is None or str(steps).lower() in ('none', 'null', ''):
            seconds = _positive(config.get('phoenix_stall_seconds', 3.),
                                'phoenix_stall_seconds')
            steps = math.ceil(seconds * hz)
        else:
            steps = _integer(steps, 'phoenix_stall_steps', 0)
        seed = config.get('phoenix_stall_seed', -1)
        if seed is None or str(seed).lower() in ('none', 'null', ''):
            seed = None
        else:
            seed = _integer(seed, 'phoenix_stall_seed', -1)
            if seed > 2**63 - 1:
                raise ValueError('phoenix_stall_seed must fit a signed 64-bit integer')
        return cls(fix, steps,
                   _integer(config.get('phoenix_stall_cover_chunks', 2),
                            'phoenix_stall_cover_chunks', 1),
                   _positive(config.get('phoenix_stall_arm_threshold', .02),
                             'phoenix_stall_arm_threshold'),
                   _positive(config.get('phoenix_stall_gripper_threshold', .05),
                             'phoenix_stall_gripper_threshold'), hz, seed)

    @property
    def enabled(self):
        return self.fix == 'cover' and self.steps > 0


@dataclass(frozen=True)
class CoverDecision:
    step: int
    stalled: bool
    covered: bool
    phase: int
    arm_range: float | None
    gripper_range: float | None


class StallCover:
    def __init__(self, settings, action_dim, gripper_slices):
        self.settings = settings
        self.action_dim = action_dim
        self.grippers = np.zeros(action_dim, dtype=bool)
        for section in gripper_slices:
            self.grippers[section] = True
        if not self.grippers.any() or self.grippers.all():
            raise ValueError('Stall detection requires both arm and gripper dimensions')
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.settings.steps + 1)
        self.phase = 0
        self.stalled_chunks = 0
        self.covered_chunks = 0
        self.reseeded_chunks = 0
        self.last_decision = None
        self.last_requested_seed = None
        self.last_effective_seed = None

    def observe(self, state, executed_step):
        if not self.settings.enabled:
            return
        step = _integer(executed_step, 'executed_step', 0)
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError('Stall detection requires a finite measured joint state')
        if self.history and step < self.history[-1][0]:
            raise ValueError('Executed step moved backwards; reset the episode first')
        if self.history and step == self.history[-1][0]:
            # Extra uploads before an action are not additional control steps.
            self.history[-1] = (step, state.copy())
        else:
            self.history.append((step, state.copy()))

    def plan(self, observation):
        """Plan a cover without advancing counters until inference succeeds."""
        step = self.history[-1][0] if self.history else 0
        arm_range = gripper_range = None
        stalled = False
        if (self.settings.enabled and len(self.history) == self.settings.steps + 1
                and step - self.history[0][0] == self.settings.steps):
            # Check the entire window, not just equal endpoints of an out-and-back
            # motion. This conservatively avoids masking a moving robot.
            states = np.stack([state for _, state in self.history])
            span = np.ptp(states, axis=0)
            arm_range = float(span[~self.grippers].max())
            gripper_range = float(span[self.grippers].max())
            stalled = (arm_range < self.settings.arm_threshold
                       and gripper_range < self.settings.gripper_threshold)
        phase = self.phase if stalled else 0
        covered = stalled and phase < self.settings.cover_chunks
        decision = CoverDecision(step, stalled, covered, phase, arm_range, gripper_range)
        if not covered:
            return observation, decision
        # Only replace the already-decoded head RGB pixels, before normalization.
        # uint8 zero is black; zero after [-1,1] normalization would be gray.
        views = observation['observation']
        head = views['head_camera']
        pixels = np.asarray(head['rgb'])
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise ValueError('Head cover requires decoded HWC uint8 RGB')
        result = dict(observation)
        result['observation'] = dict(views)
        result['observation']['head_camera'] = dict(head, rgb=np.zeros_like(pixels))
        return result, decision

    def inference_seed(self, decision, normal_seed):
        """Resolve -1 here: the frozen sampler treats literal -1 as a fixed seed."""
        requested = self.settings.seed if decision.stalled else None
        if requested is None:
            return None, normal_seed
        if requested == -1:
            effective = secrets.randbits(63)
            # Avoid repeating the previous draw or the normal deterministic seed.
            while effective in (normal_seed, self.last_effective_seed):
                effective = secrets.randbits(63)
            return requested, effective
        return requested, requested

    def commit(self, decision, *, requested_seed=None, effective_seed=None):
        self.last_decision = decision
        self.last_requested_seed = requested_seed
        self.last_effective_seed = effective_seed
        self.reseeded_chunks += int(requested_seed == -1)
        if decision.stalled:
            self.stalled_chunks += 1
            self.covered_chunks += int(decision.covered)
            self.phase = (decision.phase + 1) % (self.settings.cover_chunks + 1)
        else:
            self.phase = 0

    def snapshot(self):
        last = self.last_decision
        return dict(fix=self.settings.fix, enabled=self.settings.enabled,
                    steps=self.settings.steps, action_hz=self.settings.action_hz,
                    nominal_seconds=self.settings.steps / self.settings.action_hz,
                    cover_chunks=self.settings.cover_chunks,
                    arm_threshold_rad=self.settings.arm_threshold,
                    gripper_threshold_native=self.settings.gripper_threshold,
                    stall_seed=self.settings.seed,
                    reseeded=self.reseeded_chunks,
                    last_requested_seed=self.last_requested_seed,
                    last_effective_seed=self.last_effective_seed,
                    stalled=self.stalled_chunks, changed=self.covered_chunks,
                    last_covered=False if last is None else last.covered,
                    last_stalled=False if last is None else last.stalled,
                    executed_step=0 if last is None else last.step,
                    arm_range=None if last is None else last.arm_range,
                    gripper_range=None if last is None else last.gripper_range)
