"""XPolicyLab entry point of Auriga (runs inside the policy server).

The policy server hosts the whole agent: one Auriga episode (Codex app-server session, tool
registry, motion planning, perception clients) per RoboDojo episode, started on the first
observation after ``reset`` and driven through :class:`auriga_xpl.bridge.Bridge`.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from XPolicyLab.model_template import ModelTemplate  # noqa: E402
from XPolicyLab.utils.process_data import get_robot_action_dim_info  # noqa: E402

from auriga_xpl.bridge import Bridge, hold_still  # noqa: E402


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = dict(model_cfg or {})
        self.action_type = self.cfg.get("action_type") or "joint"
        self.env_cfg_type = self.cfg.get("env_cfg_type") or "arx_x5"
        if self.action_type != "joint":
            raise ValueError("Auriga sends joint targets: action_type must be 'joint'")
        dims = get_robot_action_dim_info(self.env_cfg_type)
        if list(dims["arm_dim"]) != [6, 6] or list(dims["ee_dim"]) != [1, 1]:
            raise ValueError(f"Auriga supports the dual-arm ARX X5 only, got {dims} for {self.env_cfg_type}")
        self.debug = str(os.environ.get("EVAL_ENV_TYPE") or "").lower() == "debug"
        self.poll_s = float(self.cfg.get("poll_s") or 20.0)
        self.wrap_up_s = float(self.cfg.get("wrap_up_s") or 60.0)
        self.bridge = Bridge(self.poll_s)
        self.thread = None
        self.last_obs = {}
        self.episodes = 0
        self.runtime = None
        if not self.debug:
            # One runtime per server process: Auriga checkout on sys.path, frozen configuration
            # (configs/auriga.json of the pinned commit + auriga_eval.json), private Codex provider
            # home (Responses endpoint and key from AURIGA_API_* only), planners and calibration.
            from auriga_xpl.runtime import AurigaRuntime

            self.runtime = AurigaRuntime(self.cfg)

    # -- XPolicyLab calls -------------------------------------------------------------------------
    def reset(self):
        self.last_obs = {}
        self.bridge.abandon()  # an episode still running belongs to a trial the client has left
        self.bridge = Bridge(self.poll_s)
        self.thread = None

    def update_obs(self, obs):
        self.last_obs = obs
        if self.debug:
            return
        self.bridge.offer_obs(obs)
        if self.thread is None:
            self.episodes += 1
            self.thread = threading.Thread(target=self._episode, args=(self.bridge,),
                                           name=f"auriga-episode-{self.episodes}", daemon=True)
            self.thread.start()

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("Auriga runs one agent session per environment (eval_batch: false)")

    def get_action(self):
        if self.debug:
            return [hold_still(self.last_obs)]
        return self.bridge.next_actions()

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("Auriga runs one agent session per environment (eval_batch: false)")

    def on_trial_end(self, result=None):
        if self.debug:
            return True
        self.bridge.conclude(dict(result or {}))
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(self.wrap_up_s)
        return True

    # -- one episode ------------------------------------------------------------------------------
    def _episode(self, bridge):
        try:
            self.runtime.play(bridge, task_name=self.cfg.get("task_name"), seed=self.cfg.get("seed"))
        except Exception:  # noqa: BLE001 - logged; the client is released below
            print("[Auriga] episode failed:\n" + traceback.format_exc(), flush=True)
        finally:
            bridge.finish()


__all__ = ["Model"]
