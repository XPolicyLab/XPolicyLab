"""The policy server's view of HexaShell.

XPolicyLab calls ``reset``, then ``update_obs`` / ``get_action`` once per action, then
``on_trial_end`` with RoboDojo's verdict.  The first observation of an episode starts an episode
thread (:func:`hexashell.runtime.episode.play`): the language model works in it, and every action
its motions produce is handed to ``get_action`` through a :class:`Bridge`.
"""

from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path
from typing import Any

from ..state import STATUS_KEY
from .bridge import Bridge

#: The checkout this package lives in; relative result paths are taken from here.
HEXASHELL_ROOT = Path(__file__).resolve().parents[2]
#: Two six-joint arms with one gripper joint each, driven by joint targets.
ARM_DIM, EE_DIM = [6, 6], [1, 1]


def hold_still(obs: dict[str, Any]) -> dict[str, list[float]]:
    """An action that keeps every joint and gripper where the observation has them."""
    state = obs.get("state") or {}
    action = {}
    for key, value in state.items():
        if key.endswith("_arm_joint_state") or key.endswith("_ee_joint_state"):
            values = value.tolist() if hasattr(value, "tolist") else list(value)
            action[key] = [float(v) for v in values]
    return action


class PolicyHost:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = dict(cfg or {})
        self.poll_s = float(self.cfg.get("poll_s") or 20.0)
        self.wrap_up_s = float(self.cfg.get("wrap_up_s") or 30.0)
        root = Path(os.environ.get("HEXASHELL_RESULT_DIR") or self.cfg.get("result_dir") or "results").expanduser()
        self.result_dir = root if root.is_absolute() else HEXASHELL_ROOT / root
        self.debug = str(os.environ.get("EVAL_ENV_TYPE") or "").lower() == "debug"
        self.bridge = Bridge(self.poll_s)
        self.thread: threading.Thread | None = None
        self.last_obs: dict[str, Any] = {}
        self.episodes = 0
        self._check_robot()

    def _check_robot(self) -> None:
        action_type = self.cfg.get("action_type")
        if action_type not in (None, "", "joint"):
            raise ValueError(f"HexaShell sends joint targets: action_type must be 'joint', not {action_type!r}")
        env_cfg_type = self.cfg.get("env_cfg_type")
        if not env_cfg_type:
            return
        from XPolicyLab.utils.process_data import get_robot_action_dim_info

        dims = get_robot_action_dim_info(env_cfg_type)
        if list(dims["arm_dim"]) != ARM_DIM or list(dims["ee_dim"]) != EE_DIM:
            raise ValueError(f"HexaShell knows the two-arm ARX X5 (arm_dim {ARM_DIM}, ee_dim {EE_DIM}); "
                             f"{env_cfg_type} has arm_dim {dims['arm_dim']}, ee_dim {dims['ee_dim']}")

    # -- XPolicyLab calls ------------------------------------------------------------------------
    def reset(self) -> None:
        self.last_obs = {}
        if self.debug:
            return
        self.bridge.abandon()  # an episode still running belongs to a trial the client has left
        self.bridge = Bridge(self.poll_s)
        self.thread = None

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.last_obs = obs
        if self.debug:
            return
        self.bridge.offer_obs(obs)
        if self.thread is None:
            info = dict(((obs.get("additional_info") or {}).get(STATUS_KEY)) or {})
            if self.cfg.get("seed") is not None:  # the run's seed (policy server argument), for the record
                info.setdefault("eval_seed", self.cfg["seed"])
            self.episodes += 1
            self.thread = threading.Thread(target=self._episode, args=(self.bridge, info),
                                           name=f"hexashell-episode-{self.episodes}", daemon=True)
            self.thread.start()

    def get_action(self) -> list:
        if self.debug:
            return [hold_still(self.last_obs)]
        return self.bridge.next_actions()

    def on_trial_end(self, result: dict[str, Any] | None = None) -> bool:
        if self.debug:
            return True
        self.bridge.conclude(dict(result or {}))
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(self.wrap_up_s)  # the record is written before the client moves on
        return True

    # -- the episode -----------------------------------------------------------------------------
    def _episode(self, bridge: Bridge, info: dict[str, Any]) -> None:
        try:
            from ..runtime.episode import play

            play(bridge, info, self.cfg, self.result_dir)
        except Exception:  # noqa: BLE001 - reported, and the client is released below
            print("[HexaShell] the episode failed:\n" + traceback.format_exc(), flush=True)
        finally:
            bridge.finish()
