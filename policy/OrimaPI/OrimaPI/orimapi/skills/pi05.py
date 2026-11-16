# coding=utf-8
"""Pi0.5 inference through the official OpenPI policy interface."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ..state import Observation
from .common import is_new_observation, decode_actions, encode_observation, action_limit
from .openpi_backend import load_openpi

if TYPE_CHECKING:
    from ..policy import PolicyConfig


class Pi05Policy:
    def __init__(self, config: PolicyConfig):
        self.config = config
        self._backend = None
        self._latest = {}
        self._encoded: dict[int, dict] = {}
        self.closed = False
        self.defer_encoding = False
        self._warmed = False

    def prepare_job(self, seed):
        import time
        import jax
        import numpy as np

        self.load()
        self.reset()
        if getattr(self._backend, "_is_pytorch_model", False) or not hasattr(
            self._backend, "_rng"
        ):
            raise RuntimeError(
                "Suite RNG lifecycle requires the pinned native JAX OpenPI Policy"
            )
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("policy_seed must be a nonnegative 31-bit integer")
        # Confine the pinned provider's RNG access to this lifecycle boundary.
        self._backend._rng = jax.random.key(seed)
        initial_rng = self._backend._rng
        warmup_s = 0.0
        if not self._warmed:
            started = time.monotonic()
            try:
                inputs = {
                    "images": {
                        name: np.zeros((3, 224, 224), dtype=np.uint8)
                        for name in self.config.camera_map
                    },
                    "state": np.zeros(sum(self.config.state_dims), dtype=np.float32),
                    "prompt": "Initialize policy",
                }
                output = self._backend.infer(inputs)
                decode_actions(output["actions"], self.config)
                self._warmed = True
                warmup_s = time.monotonic() - started
            finally:
                self._backend._rng = initial_rng
        self.defer_encoding = True
        return {
            "policy_seed": seed,
            "warmup_s": warmup_s,
            "warmed": self._warmed,
            "devices": [str(device) for device in jax.devices()],
        }

    def _open(self) -> None:
        if self.closed:
            raise RuntimeError("The action skill is closed")

    def load(self) -> None:
        self._open()
        if self._backend is not None:
            return
        self._backend = load_openpi(self.config)
        try:
            if not callable(getattr(self._backend, "infer", None)):
                raise TypeError("OpenPI provider must expose infer(observation)")
        except Exception:
            self.close()
            raise

    def describe(self) -> dict:
        return {
            "name": self.config.name,
            "loaded": self._backend is not None and not self.closed,
            "checkpoint": (
                str(self.config.checkpoint) if self.config.checkpoint else None
            ),
            "train_config_name": self.config.train_config_name,
            "normalization_id": self.config.normalization_id,
            "tokenizer_path": (
                str(self.config.tokenizer_path) if self.config.tokenizer_path else None
            ),
            "supports_subgoals": True,
            "description": "Language-conditioned action inference. Physical capabilities are limited to the supplied checkpoint; no task success is guaranteed.",
            "state_fields": list(self.config.state_fields),
            "state_dims": list(self.config.state_dims),
            "action_fields": list(self.config.action_fields),
            "action_dims": list(self.config.action_dims),
            "camera_map": dict(self.config.camera_map),
        }

    def observe(self, observation: Observation) -> None:
        self.load()
        if not is_new_observation(self._latest.get(observation.env_id), observation):
            return
        encoded = (
            None
            if self.defer_encoding
            else encode_observation(observation, self.config)
        )
        self._encoded[observation.env_id] = encoded
        self._latest[observation.env_id] = deepcopy(observation)

    def predict(
        self, observation: Observation, *, prompt: str, max_actions: int | None = None
    ) -> list[dict]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty active subgoal is required")
        action_limit(max_actions)
        self.observe(observation)
        if self._encoded[observation.env_id] is None:
            self._encoded[observation.env_id] = encode_observation(
                observation, self.config
            )
        inputs = deepcopy(self._encoded[observation.env_id])
        inputs["prompt"] = prompt
        output = self._backend.infer(inputs)
        if not isinstance(output, dict) or "actions" not in output:
            raise ValueError("OpenPI did not return an actions array")
        return decode_actions(output["actions"], self.config, max_actions)

    def reset(self, env_id: int | None = None) -> None:
        self._open()
        for values in (self._latest, self._encoded):
            if env_id is None:
                values.clear()
            else:
                values.pop(env_id, None)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.reset()
        finally:
            try:
                close = getattr(self._backend, "close", None)
                if callable(close):
                    close()
            finally:
                self._backend = None
                self._latest.clear()
                self._encoded.clear()
                self.closed = True
