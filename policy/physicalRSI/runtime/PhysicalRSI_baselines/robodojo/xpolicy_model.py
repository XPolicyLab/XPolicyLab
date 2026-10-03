"""Policy service model interface for isolated actors of one frozen harness.

The installed runtime supplies actor_factory(identity=...). It must construct a
fresh, unstarted isolated actor with the pinned policy, memory and primitive closure.
This boundary does not select skill_choices, evolve policies or judge task success.
"""

import math
import time
import uuid
from copy import deepcopy


class Model:
    def __init__(
        self,
        *,
        actor_factory,
        harness_revision,
        max_actors=10,
        timeout_s=60,
        admit_batch=None,
    ):
        if not isinstance(harness_revision, str) or not harness_revision:
            raise ValueError("A frozen harness revision is required")
        if type(max_actors) is not int or not 1 <= max_actors <= 10:
            raise ValueError("The policy service supports one to ten actors")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Positive action timeout required")
        self.factory = actor_factory
        if admit_batch is not None and not callable(admit_batch):
            raise ValueError("Batch admission must be callable")
        self.admit_batch = admit_batch
        self.revision = harness_revision
        self.max_actors, self.timeout_s = max_actors, timeout_s
        self.actors = {}
        self.order = []
        self.failed = False
        self.closed = False
        self.episode = uuid.uuid4().hex

    def _ready(self):
        if self.closed or self.failed:
            raise RuntimeError("Model closed or interrupted; successful reset required")

    @staticmethod
    def _indices(indices):
        if any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("Environment indices must be nonnegative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate environment indices")

    def update_obs(self, obs):
        self.update_obs_batch([dict(obs, env_idx=obs.get("env_idx", 0))])

    def update_obs_batch(self, obs_list):
        self._ready()
        observations = list(obs_list)
        indices = [obs["env_idx"] for obs in observations]
        self._indices(indices)
        if not indices or len(set(self.actors) | set(indices)) > self.max_actors:
            raise ValueError("Expected a nonempty batch within actor capacity")
        if self.admit_batch is not None:
            self.admit_batch(len(set(self.actors) | set(indices)))
        try:
            created = []
            for index in indices:
                if index not in self.actors:
                    identity = dict(
                        actor=str(index),
                        episode=self.episode,
                        harness_revision=self.revision,
                    )
                    actor = self.factory(identity=deepcopy(identity))
                    # Retain ownership even if validation/start fails.
                    self.actors[index] = actor
                    if actor.identity != identity:
                        raise ValueError("Actor identity differs from frozen binding")
                    created.append(index)
            # Constructors may load multiple GPU services. Do not start an
            # early policy's episode deadline while later actors still load.
            for index in created:
                self.actors[index].start()
            for index, obs in zip(indices, observations):
                self.actors[index].update_obs(obs)
            self.order = indices
        except BaseException:
            # Partial updates/actions must never be silently retried.
            self.failed = True
            raise

    def get_action(self):
        if len(self.order) != 1:
            raise ValueError("Single action requires one observed environment")
        return self.get_action_batch(self.order)[0]

    def get_action_batch(self, env_idx_list=None):
        self._ready()
        indices = list(self.order if env_idx_list is None else env_idx_list)
        self._indices(indices)
        if not indices or any(index not in self.actors for index in indices):
            raise ValueError("Actions require previously observed environments")
        deadline = time.monotonic() + self.timeout_s
        try:
            return [self.actors[i].get_action(deadline=deadline) for i in indices]
        except BaseException:
            self.failed = True
            raise

    def _cleanup(self):
        errors = []
        for index, actor in list(self.actors.items()):
            try:
                actor.close()
            except BaseException as error:
                errors.append(error)
            else:
                del self.actors[index]
        if errors:
            self.failed = True
            raise RuntimeError("Actor cleanup failed; ownership retained") from errors[
                0
            ]
        self.order = []

    def reset(self):
        if self.closed:
            raise RuntimeError("Closed model cannot be reset")
        self._cleanup()
        self.episode = uuid.uuid4().hex
        self.failed = False

    def close(self):
        self._cleanup()
        self.closed = True

    def on_trial_end(self, result=None):
        # Evaluator results are not trusted as deployment/qualification evidence.
        self.reset()
