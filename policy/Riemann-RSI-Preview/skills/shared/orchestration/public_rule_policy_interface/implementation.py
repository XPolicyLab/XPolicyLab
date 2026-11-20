"""Policy-side blocking-rule transport; no simulator or evaluation hooks."""
from __future__ import annotations

from copy import deepcopy
import threading
import time
from typing import Callable

import numpy as np

from skills.shared.motion.normalized_pair_joint_execution.implementation import project_observation


class PolicyEpisodeClosed(BaseException):
    """Cancel a legacy workflow without entering its ordinary retry handlers."""


class RulePolicyFailure(RuntimeError):
    """An unexpected workflow error, with its original cause retained."""


ACTION_WIDTHS = {
    'left_arm_joint_state': 6, 'right_arm_joint_state': 6,
    'left_ee_joint_state': 1, 'right_ee_joint_state': 1,
}


def normalized_action(action):
    pose_widths = {side + suffix: width for side in ('left', 'right')
                   for suffix, width in (('_ee_pose', 7), ('_ee_joint_state', 1))}
    widths = pose_widths if isinstance(action, dict) and set(action) == set(pose_widths) else ACTION_WIDTHS
    if not isinstance(action, dict) or set(action) != set(widths):
        raise ValueError('Expected exactly the four public joint target fields')
    result = {}
    for key, width in widths.items():
        value = np.asarray(action[key], dtype=float)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError('Invalid public action: ' + key)
        if key.endswith('ee_joint_state') and np.any((value < 0) | (value > 1)):
            raise ValueError('Gripper commands must be normalized, not measured widths')
        if key.endswith('ee_pose') and abs(np.linalg.norm(value[3:]) - 1) > .01:
            raise ValueError('Expected an official unit quaternion: ' + key)
        result[key] = value.tolist()
    return result


class BlockingRulePolicy:
    """One official caller, one workflow, one action acknowledged by one frame.

    The runner owns its task skills and resources. It receives no native object.
    The official websocket server decodes RGB before calling update_obs.
    """

    def __init__(self, runner: Callable, *, wait_timeout=180.0, join_timeout=5.0,
                 planning_poll_seconds=None):
        if not callable(runner):
            raise TypeError('A public-observation workflow runner is required')
        if not np.isfinite([wait_timeout, join_timeout]).all() or min(wait_timeout, join_timeout) <= 0:
            raise ValueError('Policy waits and cleanup must be bounded')
        self._runner = runner
        self._wait_timeout = float(wait_timeout)
        self._join_timeout = float(join_timeout)
        if planning_poll_seconds is not None and (not np.isfinite(planning_poll_seconds)
                                                  or planning_poll_seconds <= 0):
            raise ValueError('Planning poll duration must be finite and positive')
        self._planning_poll_seconds = planning_poll_seconds
        self._condition = threading.Condition()
        self._thread = None
        self._generation = 0
        self._closed = False
        self._clear()

    def _clear(self):
        self._observation = None
        self._fresh_observation = False
        self._pending = None
        self._awaiting_ack = False
        self._ack = None
        self._last_action = None
        self._error = None
        self._completed = False
        self._observations_received = 0
        self._actions_returned = 0
        self._empty_planning_chunks = 0

    def _stop(self):
        with self._condition:
            self._generation += 1
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(self._join_timeout)
            if thread.is_alive():
                with self._condition:
                    self._closed = True
                raise RuntimeError('Previous workflow did not exit; refusing episode reuse')
        with self._condition:
            self._thread = None

    def reset(self):
        if self._closed:
            raise RuntimeError('Policy is closed')
        self._stop()
        with self._condition:
            self._clear()

    def close(self):
        with self._condition:
            self._closed = True
        self._stop()

    def update_obs(self, observation):
        # Validate the projected contract now, not after returning an action.
        # PublicObservationMemory already defines this shared RGB/state contract.
        from skills.shared.motion.normalized_pair_joint_execution.implementation import PublicObservationMemory
        frame = project_observation(observation)
        PublicObservationMemory(frame, '.')
        if 'instruction' in observation:
            instruction = observation['instruction']
            if not isinstance(instruction, str) or len(instruction) > 5000:
                raise ValueError('Expected the official public instruction string')
            frame['instruction'] = instruction
        with self._condition:
            if self._closed:
                raise RuntimeError('Policy is closed')
            self._observation = frame
            self._observations_received += 1
            self._fresh_observation = True
            if self._awaiting_ack:
                self._ack = deepcopy(frame)
                self._awaiting_ack = False
                self._pending = None
            self._condition.notify_all()

    def _exchange(self, generation, action):
        target = normalized_action(action)
        with self._condition:
            self._require_episode(generation)
            if self._pending is not None:
                raise RuntimeError('Only one workflow command may be outstanding')
            self._ack = None
            self._pending = target
            self._condition.notify_all()
            deadline = time.monotonic() + self._wait_timeout
            while self._ack is None:
                self._require_episode(generation)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('No official observation acknowledged the policy action')
                self._condition.wait(remaining)
            self._require_episode(generation)
            frame = self._ack
            self._ack = None
            return frame

    def _require_episode(self, generation):
        if self._closed or generation != self._generation:
            raise PolicyEpisodeClosed()

    def _run(self, generation, initial):
        try:
            self._runner(initial, lambda action: self._exchange(generation, action))
        except PolicyEpisodeClosed:
            return
        except BaseException as error:
            with self._condition:
                if generation == self._generation:
                    self._error = error
        finally:
            with self._condition:
                if generation == self._generation:
                    self._completed = True
                self._condition.notify_all()

    def get_action(self):
        with self._condition:
            if self._closed:
                raise RuntimeError('Policy is closed')
            if self._observation is None or not self._fresh_observation or self._awaiting_ack:
                raise RuntimeError('A fresh official observation is required before each action')
            generation = self._generation
            if self._thread is None:
                self._last_action = normalized_action({key: self._observation['state'][key] for key in ACTION_WIDTHS})
                self._thread = threading.Thread(
                    target=self._run, args=(generation, deepcopy(self._observation)),
                    name='public-rule-workflow', daemon=True,
                )
                self._thread.start()
            deadline = time.monotonic() + (self._planning_poll_seconds or self._wait_timeout)
            while self._pending is None and not self._completed:
                self._require_episode(generation)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self._planning_poll_seconds is not None:
                        # The official deploy loop accepts an empty chunk:
                        # it polls again without calling env.take_action.
                        # No command was delivered, so the next public frame
                        # must NOT acknowledge a pending workflow action.
                        self._fresh_observation = False
                        self._empty_planning_chunks += 1
                        return []
                    raise TimeoutError('Public workflow did not produce an action')
                self._condition.wait(remaining)
            self._require_episode(generation)
            if self._error is not None:
                raise RulePolicyFailure('Public workflow failed: ' + str(self._error)) from self._error
            target = self._last_action if self._pending is None else self._pending
            self._last_action = deepcopy(target)
            self._fresh_observation = False
            self._awaiting_ack = True
            self._actions_returned += 1
            return [deepcopy(target)]

    def diagnostics(self):
        with self._condition:
            return {
                'observations_received': self._observations_received,
                'actions_returned': self._actions_returned,
                'empty_planning_chunks': self._empty_planning_chunks,
                'workflow_completed': self._completed,
                'workflow_error': None if self._error is None else repr(self._error),
                'worker_alive': self._thread is not None and self._thread.is_alive(),
                'official_success_claimed': False,
            }
