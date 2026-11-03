"""Align positional upstream observation batches with explicit environment IDs."""

import numpy as np


def _select(value, positions, count):
    if isinstance(value, dict):
        return {key: _select(item, positions, count) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.ndim < 1 or value.shape[0] != count:
            raise ValueError("Observation tensor has an inconsistent batch dimension")
        return value[positions]
    if isinstance(value, (list, tuple)) and len(value) == count:
        selected = [value[index] for index in positions]
        return tuple(selected) if isinstance(value, tuple) else selected
    raise ValueError("Unsupported stacked observation field")


class EnvironmentOrder:
    def update_obs_batch(self, obs_list):
        rows = list(obs_list)
        indices = [row["env_idx"] for row in rows]
        if (
            not 1 <= len(indices) <= 10
            or len(set(indices)) != len(indices)
            or any(type(index) is not int or index < 0 for index in indices)
        ):
            raise ValueError("One to ten distinct integer environment IDs required")
        return super().update_obs_batch(rows)

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise ValueError(
                "Observe the active environments before requesting actions"
            )
        observed = list(self._latest_env_idx_list)
        requested = list(observed if env_idx_list is None else env_idx_list)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(
                type(index) is not int or index not in observed for index in requested
            )
        ):
            raise ValueError(
                "Actions require distinct currently observed environment IDs"
            )
        window, indices = self.observation_window, self._latest_env_idx_list
        positions = [observed.index(index) for index in requested]
        try:
            if positions != list(range(len(observed))):
                self.observation_window = _select(window, positions, len(observed))
            self._latest_env_idx_list = requested
            # Never replay update_obs_batch: sparse memory must receive each
            # physical observation exactly once. Preserve upstream RNG/history.
            return super().get_action_batch(env_idx_list=requested, **kwargs)
        finally:
            self.observation_window, self._latest_env_idx_list = window, indices
