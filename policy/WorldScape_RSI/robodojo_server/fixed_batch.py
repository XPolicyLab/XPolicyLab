"""Fixed, shrinking environment batches on one GPU; no model algorithm changes."""

from collections.abc import Mapping

from .official_client import environment_ids


class FixedBatchMixin:
    """Validate the per-action observation contract before native state changes.

    A protocol trial owns an entire batch. Retired IDs cannot rejoin until reset.
    The last surviving environment ends through TRIAL_END, not an empty update.
    """

    def __init__(self, *args, **kwargs):
        self._batch_clear()
        super().__init__(*args, **kwargs)

    def _batch_clear(self):
        self._batch_active = None
        self._batch_remaining = None
        self._official_failed = False

    def _batch_check(self):
        if self._official_failed:
            raise RuntimeError("Observation sequence failed; reset before continuing")

    def reset(self):
        self._official_failed = True
        result = super().reset()
        self._batch_clear()
        return result

    def update_obs(self, obs):
        return self.update_obs_batch([obs])

    def update_obs_batch(self, rows):
        self._batch_check()
        try:
            if not isinstance(rows, (list, tuple)) or not all(isinstance(row, Mapping) for row in rows):
                raise TypeError("Expected an observation list")
            if not 1 <= len(rows) <= self.max_envs_per_session:
                raise ValueError(f"Batch size exceeds capacity: expected 1-{self.max_envs_per_session} environments")
            if len(rows) > 1 and any('env_idx' not in row for row in rows):
                raise ValueError("Batch observations require explicit stable env_idx")
            ids = environment_ids(row.get("env_idx", 0) for row in rows)
            if self._batch_active is not None:
                if not set(ids).issubset(self._batch_active):
                    raise ValueError("New or retired environment reappeared; reset the batch first")
                if self._batch_remaining == 0:
                    raise ValueError("Unexpected observation: request actions before the next update")
            result = super().update_obs_batch(rows)
            self._batch_active = ids
            self._batch_remaining = 0 if self._batch_remaining is None else self._batch_remaining - 1
            return result
        except Exception:
            self._official_failed = True
            raise

    def get_action(self):
        if self._batch_active is None or len(self._batch_active) != 1:
            raise ValueError("get_action requires one active environment; use get_action_batch")
        return self.get_action_batch(self._batch_active)[0]

    def get_action_batch(self, request=None):
        self._batch_check()
        try:
            if self._batch_remaining != 0:
                raise ValueError("Incomplete action observations; update observations before predicting")
            ids = request.get("env_idx_list") if isinstance(request, Mapping) else request
            ids = environment_ids(self._batch_active if ids is None else ids)
            if set(ids) != set(self._batch_active):
                raise ValueError("Prediction IDs must match the latest active observation batch")
            result = super().get_action_batch(ids)
            if len(result) != len(ids) or not result or not result[0] or len({len(row) for row in result}) != 1:
                raise ValueError("Expected one equal non-empty action chunk per environment")
            self._batch_remaining = len(result[0])
            return result
        except Exception:
            self._official_failed = True
            raise

    def get_runtime_info(self):
        return {**super().get_runtime_info(), "supports_batch": self.max_envs_per_session > 1,
                "supports_single_env_batch": True, "max_envs_per_session": self.max_envs_per_session,
                "batch_lifecycle": "fixed_shrinking_until_reset"}
