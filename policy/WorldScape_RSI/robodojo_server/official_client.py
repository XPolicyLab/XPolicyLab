"""Adapt WSP's explicit commit to the pinned official per-action observation loop.

Each update after a prediction confirms one executed action for the previously
active environments. The H-th update is the next chunk's initial observation:
commit BEFORE forwarding it, just as the explicit deployment loop does.
"""

from collections.abc import Mapping
from numbers import Integral


def environment_ids(values):
    ids = list(values)
    if not ids or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ids):
        raise ValueError("Expected non-empty non-negative integer environment IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate environment IDs")
    return [int(i) for i in ids]


class OfficialClientMixin:
    """Only for WSP. Never changes the model config, predictions or GPU routing."""

    def __init__(self, *args, **kwargs):
        self._official_clear()
        super().__init__(*args, **kwargs)

    def _official_clear(self):
        self._official_pending = None
        self._official_latest = []
        self._official_failed = False

    def _official_check(self):
        if self._official_failed:
            raise RuntimeError("Official observation sequence failed; reset before continuing")

    def get_runtime_info(self):
        return {**super().get_runtime_info(), "requires_explicit_commit": False,
                "client_mode": "official", "observation_contract": "one_update_per_executed_action"}

    def reset(self):
        self._official_failed = True
        result = super().reset()
        self._official_clear()
        return result

    def update_obs(self, obs):
        return self.update_obs_batch([obs])

    def update_obs_batch(self, observations):
        self._official_check()
        try:
            if not isinstance(observations, (list, tuple)) or not all(isinstance(o, Mapping) for o in observations):
                raise TypeError("Expected an observation list")
            ids = environment_ids(o.get("env_idx", i) for i, o in enumerate(observations))
            pending = self._official_pending
            if pending is not None:
                active = set(ids)
                if not active.issubset(pending["active"]):
                    raise ValueError("New or retired environment reappeared before reset")
                # Environments disappearing on this update executed the same
                # last action as survivors, but no longer supply observations.
                for env in pending["active"]:
                    pending["executed"][env] += 1
                pending["active"] = active
                counts = pending["executed"]
                horizons = pending["horizons"]
                if any(counts[env] > horizons[env] for env in counts):
                    raise ValueError("Too many execution observations")
                if all(counts[env] == horizons[env] for env in active):
                    super().commit_action({
                        "executed_steps": dict(counts),
                        "finished_env_idx_list": sorted(set(horizons) - active),
                    })
                    self._official_pending = None
            result = super().update_obs_batch(observations)
            self._official_latest = ids
            return result
        except Exception:
            self._official_failed = True
            raise

    def get_action(self):
        return self.get_action_batch(self._official_latest[:1])[0]

    def get_action_batch(self, env_idx_list=None):
        self._official_check()
        try:
            if self._official_pending is not None:
                raise RuntimeError("Incomplete action observations; reset instead of guessing execution")
            ids = environment_ids(self._official_latest if env_idx_list is None else env_idx_list)
            if set(ids) != set(self._official_latest):
                raise ValueError("Prediction IDs must match the latest active observation batch")
            actions = super().get_action_batch(ids)
            if len(actions) != len(ids):
                raise ValueError("Wrong action batch size")
            horizons = {env: len(chunk) for env, chunk in zip(ids, actions, strict=True)}
            if len(set(horizons.values())) != 1 or not next(iter(horizons.values())):
                raise ValueError("Expected equal non-empty action horizons")
            self._official_pending = {"horizons": horizons, "active": set(ids),
                                      "executed": dict.fromkeys(ids, 0)}
            return actions
        except Exception:
            self._official_failed = True
            raise

    def commit_action(self, execution=None):
        raise RuntimeError("Explicit commit is disabled in official mode; use --client-mode explicit")

    def discard_action(self, execution=None):
        self._official_failed = True
        return super().discard_action(execution)

    def on_trial_end(self, result=None):
        # No final observation is guaranteed. Do not infer an execution count
        # for ended episodes; reset discards pending state and all memory.
        self.reset()
        return super().on_trial_end(result)
