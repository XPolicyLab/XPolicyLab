"""Training-faithful WAM bridge for the Phoenix_WAM RoboDojo evaluation."""

from __future__ import annotations

import sys
import json
import logging
from functools import partial
from pathlib import Path
from XPolicyLab.model_template import ModelTemplate

from ._adapter_base import StatefulWAMAdapter, as_bool
from .stall_recovery import StallCover, StallSettings

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
        self._stall_trackers = {}
        # Real initialization resolves the configured rate before loading weights.
        self._stall_settings = StallSettings.from_config(
            {'phoenix_stall_fix': 'none', 'phoenix_stall_steps': 0}, 1.)
        StatefulWAMAdapter.__init__(self, model_cfg)
    update_obs = StatefulWAMAdapter.update_obs
    _session_for = StatefulWAMAdapter._session_for
    update_obs_batch = StatefulWAMAdapter.update_obs_batch
    _dummy_actions = StatefulWAMAdapter._dummy_actions
    get_action = StatefulWAMAdapter.get_action
    get_action_batch = StatefulWAMAdapter.get_action_batch
    get_timing_rollout = StatefulWAMAdapter.get_timing_rollout

    def _stall_for(self, session):
        if session not in self._stall_trackers:
            self._stall_trackers[session] = StallCover(
                self._stall_settings, self.action_dim, self.gripper_slices)
        return self._stall_trackers[session]

    def _ingest(self, session, obs):
        adapted, text = StatefulWAMAdapter._ingest(self, session, obs)
        if self._stall_settings.enabled:
            self._stall_for(session).observe(
                adapted['joint_action']['vector'],
                session.get_timing_rollout()['executed_action_steps'])
        return adapted, text

    def get_stall_rollout(self):
        sessions = self.env_sessions if self.env_sessions else {'single': self.session}
        return {str(key): self._stall_for(session).snapshot()
                for key, session in sessions.items() if session is not None}

    def _record_stall(self, session, event):
        env_idx = next((key for key, value in self.env_sessions.items()
                        if value is session), 'single')
        record = dict(event=event, env_idx=env_idx,
                      inference_call=session.get_timing_rollout()['replan_calls'],
                      **self._stall_for(session).snapshot())
        # Logging must not discard a successfully queued action chunk.
        try:
            with (session.runtime.audit_dir / 'stall_recovery.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        except OSError:
            logging.getLogger(__name__).exception('Could not record stall recovery')
        if event == 'episode_end' or record['last_stalled']:
            print('[PHOENIX_STALL] ' + json.dumps(record), flush=True)

    def reset(self):
        for session in self._stall_trackers:
            if self._stall_settings.enabled:
                self._record_stall(session, 'episode_end')
        StatefulWAMAdapter.reset(self)
        self._stall_trackers.clear()

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
        from .phoenix_contract import ROBOTS
        self._stall_settings = StallSettings.from_config(
            self.model_cfg, ROBOTS[self.env_cfg_type][2])
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
            if session.pending_model_actions:
                raise RuntimeError('Previous actions have not all been acknowledged')
            if not self._stall_settings.enabled:
                return predict_phoenix(self, session, observation, instruction_text)
            tracker = self._stall_for(session)
            supplied, decision = tracker.plan(observation)
            normal_seed = session.seed
            requested_seed, effective_seed = tracker.inference_seed(decision, normal_seed)
            try:
                # Override only this session for this call, including the real-image
                # chunk of a persistent stall. Never change the shared model config.
                session.seed = effective_seed
                actions = predict_phoenix(self, session, supplied, instruction_text)
            finally:
                session.seed = normal_seed
            tracker.commit(decision, requested_seed=requested_seed,
                           effective_seed=effective_seed)
            self._record_stall(session, 'prediction')
            return actions
        return StatefulWAMAdapter._predict(self, session, observation, instruction_text)


__all__ = ["Model"]
