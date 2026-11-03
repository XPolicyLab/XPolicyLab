"""Executable sparse visual-memory skill using the installed OpenPI bridge."""
from .sparse_backend import Model as UpstreamModel
from .metadata import ExternalPolicyMetadata
from .order import EnvironmentOrder


class Model(ExternalPolicyMetadata, EnvironmentOrder, UpstreamModel):
    policy_name = "pi05-sparse-memory"
    memory_mode = "causal_visual_history"

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.ensemble_seeds = []
        self.reset_policy_rng_each_call = False
        self.policy_seed = int(model_cfg.get("seed") or 0)
        self.temporal_ensemble_decay = None
        self.exec_horizon = int(model_cfg.get("exec_horizon") or self.policy._model.action_horizon)
        self._action_history = {}
        self._debug_action_calls = 0

    def reset(self):
        super().reset()
        self._action_history.clear()
        self._debug_action_calls = 0
