"""XPolicyLab adapter for X-VLA followed by GeoRefiner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from XPolicyLab.policy.X_VLA.model import (
    Model as XVLAModel,
    action_chunk_to_ee_dict_list,
    encode_obs as encode_xvla_obs,
    resolve_prompt,
)

from .georefiner_postprocessor import GeoRefinerObservation, GeoRefinerPostProcessor


_POLICY_DIR = Path(__file__).resolve().parent


class Model(XVLAModel):
    """Reuse X-VLA inference and refine each nominal action chunk once."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        adapter_cfg = dict(model_cfg)
        georefiner_cfg = adapter_cfg.get("georefiner")

        # Some development worktrees contain the legacy GeoRefiner hook inside
        # X_VLA.Model. Disable that hook while constructing the base policy so
        # this standalone adapter never loads or applies the refiner twice.
        base_cfg = dict(adapter_cfg)
        base_cfg["georefiner"] = {"mode": "disabled"}
        super().__init__(base_cfg)

        # Keep all normal X-VLA runtime options (steps, domain_id, paths, ...)
        # visible to the inherited infer() implementation.
        self.model_cfg = adapter_cfg
        self.georefiner_postprocessor = GeoRefinerPostProcessor(
            georefiner_cfg,
            action_horizon=self._resolve_action_horizon(),
            policy_dir=_POLICY_DIR,
        )
        self._georefiner_observations: dict[int, GeoRefinerObservation] = {}

    def _resolve_action_horizon(self) -> int:
        """Read the action horizon without requiring changes to X_VLA.Model."""

        candidates = (
            self.model,
            getattr(self.model, "model", None),
            getattr(getattr(self.model, "base_model", None), "model", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            value = getattr(candidate, "num_actions", None)
            if value is None:
                value = getattr(getattr(candidate, "config", None), "num_actions", None)
            if value is not None:
                return int(value)
        raise AttributeError("Loaded X-VLA model does not expose its action horizon.")

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: Sequence[dict[str, Any]]) -> None:
        env_indices = [int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)]
        encoded_observations: list[dict[str, Any]] = []
        refiner_observations: dict[int, GeoRefinerObservation] = {}

        for obs, env_idx in zip(obs_list, env_indices):
            # This is the official X-VLA preprocessing function. GeoRefiner
            # only reads additional fields and does not alter its result.
            encoded = encode_xvla_obs(obs, self.default_prompt)
            encoded["env_idx"] = env_idx
            encoded_observations.append(encoded)

            if self.georefiner_postprocessor.enabled:
                prompt = resolve_prompt(obs, self.default_prompt)
                refiner_observations[env_idx] = GeoRefinerObservation.from_observation(
                    obs,
                    instruction=prompt,
                    env_idx=env_idx,
                    head_rgb=encoded["images"][0],
                )

        self._latest_env_idx_list = env_indices
        self.observation_window = encoded_observations
        self._georefiner_observations = refiner_observations

    def get_action(self, **kwargs: Any) -> list[dict[str, np.ndarray]]:
        return self.get_action_batch(
            env_idx_list=[self._latest_env_idx_list[0]], **kwargs
        )[0]

    def get_action_batch(
        self, env_idx_list: Sequence[int] | None = None, **kwargs: Any
    ) -> list[list[dict[str, np.ndarray]]]:
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        requested_indices = list(env_idx_list or self._latest_env_idx_list)
        observations_by_env = {
            int(observation["env_idx"]): observation
            for observation in self.observation_window
        }
        try:
            encoded_observations = [
                observations_by_env[int(env_idx)] for env_idx in requested_indices
            ]
        except KeyError as exc:
            raise KeyError(
                f"No cached X-VLA observation for env_idx={exc.args[0]}; "
                f"cached envs are {sorted(observations_by_env)}."
            ) from exc

        # Keep the already validated X-VLA inference path byte-for-byte reused.
        nominal_chunks = [
            self.infer(observation, steps=kwargs.get("steps"))
            for observation in encoded_observations
        ]
        refiner_observations = None
        if self.georefiner_postprocessor.enabled:
            try:
                refiner_observations = [
                    self._georefiner_observations[int(env_idx)]
                    for env_idx in requested_indices
                ]
            except KeyError as exc:
                raise KeyError(
                    f"No cached GeoRefiner observation for env_idx={exc.args[0]}."
                ) from exc

        action_chunks = self.georefiner_postprocessor.process(
            nominal_chunks, refiner_observations
        )
        return [action_chunk_to_ee_dict_list(chunk) for chunk in action_chunks]

    def reset(self) -> None:
        self.observation_window = None
        self._latest_env_idx_list = [0]
        self._georefiner_observations = {}
        self.georefiner_postprocessor.reset_stats()
