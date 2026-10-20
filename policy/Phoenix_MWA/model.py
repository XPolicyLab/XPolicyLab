"""Training-faithful WAM bridge for the Phoenix_MWA RoboDojo evaluation."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from XPolicyLab.model_template import ModelTemplate

from ._adapter_base import StatefulWAMAdapter, as_bool

POLICY_DIR = Path(__file__).resolve().parent


def _activate_runtime(runtime_root: Path) -> None:
    source = str(runtime_root / "src")
    sys.path[:] = [source] + [entry for entry in sys.path if entry != source]
    for module_name in list(sys.modules):
        if module_name == "wam" or module_name.startswith("wam."):
            del sys.modules[module_name]


class Model(ModelTemplate):
    """XPolicyLab model adapter with one stateful WAM session per RoboDojo environment."""

    def __init__(self, model_cfg):
        if as_bool(model_cfg.get('allow_dummy_policy', False)):
            raise ValueError('Dummy actions are disabled in the H5 deployment')
        StatefulWAMAdapter.__init__(self, model_cfg)
    _ingest = StatefulWAMAdapter._ingest
    update_obs = StatefulWAMAdapter.update_obs
    _session_for = StatefulWAMAdapter._session_for
    update_obs_batch = StatefulWAMAdapter.update_obs_batch
    _dummy_actions = StatefulWAMAdapter._dummy_actions
    get_action = StatefulWAMAdapter.get_action
    get_action_batch = StatefulWAMAdapter.get_action_batch
    get_timing_rollout = StatefulWAMAdapter.get_timing_rollout
    reset = StatefulWAMAdapter.reset

    def _load_real_policy(self):
        mode = str(self.model_cfg.get('runtime_mode') or 'joint96_real_phoenix')
        if mode != 'joint96_real_phoenix':
            raise ValueError('Only the current H5 Phoenix deployment is supported')
        self.model_cfg['runtime_mode'] = mode
        self._load_phoenix_policy()

    def _load_phoenix_policy(self) -> None:
        from .joint96_assets import read_source
        _, assets = read_source()
        _activate_runtime(assets['runtime_root'])
        from .phoenix_backend import PhoenixSession, build_phoenix_runtime
        self.runtime, session_cfg = build_phoenix_runtime(self.model_cfg)
        self.session_factory = partial(PhoenixSession, self.runtime, session_cfg)
        self.session = self.session_factory()

    def _adapt_obs(self, obs):
        if self.model_cfg.get('runtime_mode') == 'joint96_real_phoenix':
            import numpy as np
            for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                image = np.asarray(obs['vision'][camera]['color'])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError('New real training requires explicit uint8 RGB input: ' + camera)
        return StatefulWAMAdapter._adapt_obs(self, obs)

    def _predict(self, session, observation, instruction_text):
        if self.model_cfg.get('runtime_mode') == 'joint96_real_phoenix' and not self.allow_dummy_policy:
            from .phoenix_backend import predict_phoenix
            return predict_phoenix(self, session, observation, instruction_text)
        return StatefulWAMAdapter._predict(self, session, observation, instruction_text)


__all__ = ["Model"]
