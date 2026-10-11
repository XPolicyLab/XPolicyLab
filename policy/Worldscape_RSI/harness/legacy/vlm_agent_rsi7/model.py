"""XPolicyLab policy-server entry point for the RoboDojo VLM agent.

The policy server instantiates ``Model(deploy_cfg)`` from ``deploy.yml`` and
then dispatches every ``model_client.call(func_name=...)`` onto a method of
this class, each on a worker thread. All VLM traffic happens here, in a plain
CPU process - the Isaac Sim client never talks to the VLM directly.
"""

from __future__ import annotations

import json
import os
import re
import time

from XPolicyLab.model_template import ModelTemplate

from .agent import AgentConfig, VLMAgent
from .vlm_client import PROFILE_ENV_VAR, VLMClient, load_vlm_profile

# JSON blob merged over the deploy.yml `vlm_agent` block, for one-off runs
# (`VLM_AGENT_OVERRIDES='{"cameras": ["cam_head", "cam_left_wrist"]}'`).
OVERRIDES_ENV_VAR = "VLM_AGENT_OVERRIDES"


class Model(ModelTemplate):
    """Closed-loop VLM policy: image and state in, one motion envelope out."""

    def __init__(self, model_cfg: dict):
        self.model_cfg = dict(model_cfg or {})
        raw_cfg = dict(self.model_cfg.get("vlm_agent") or {})
        raw_cfg.update(_env_overrides())

        self.cfg = AgentConfig(raw_cfg)
        desktop_root = os.environ.get('ROBODAWN_DESKTOP_BRIDGE')
        profile = ({'name': 'codex-desktop'} if desktop_root else
                   load_vlm_profile(os.environ.get(PROFILE_ENV_VAR) or self.cfg.vlm_profile))
        self.cfg.vlm_profile = profile["name"]
        if desktop_root:
            from .desktop_client import DesktopClient
            self.client = DesktopClient(desktop_root)
        else:
            self.client = VLMClient(
            profile,
            timeout_s=self.cfg.request_timeout_s,
            max_attempts=self.cfg.max_attempts,
            retry_delay_s=self.cfg.retry_delay_s,
        )

        task_name = self.model_cfg.get("task_name") or "task"
        profile_tag = re.sub(r"[^A-Za-z0-9._-]", "_", profile["name"])
        run_tag = os.environ.get("VLM_AGENT_RUN_TAG") or f"{task_name}_{profile_tag}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        if re.fullmatch(r"[A-Za-z0-9._-]+", run_tag) is None or run_tag in (".", ".."):
            raise ValueError("VLM_AGENT_RUN_TAG must be a simple directory name")
        self.agent = VLMAgent(self.cfg, self.client, run_tag=run_tag, task_name=task_name)

        # RoboDojo only uses action_type to label the result directory; the
        # environment infers the real type from each action's keys. The planned
        # trajectories are joint actions, so the label must say so.
        action_type = self.model_cfg.get("action_type")
        if action_type not in (None, "joint"):
            print(
                f"[vlm_agent] WARNING: action_type={action_type!r} but the policy emits joint actions; "
                "run it with --action-type joint",
                flush=True,
            )

        print(
            "[vlm_agent] ready\n"
            f"  profile    : {self.cfg.vlm_profile}\n"
            f"  model      : {self.client.model}\n"
            f"  task       : {task_name}\n"
            f"  demo bank  : {self.cfg.icl['demo_bank']}\n"
            f"  cameras    : {self.cfg.cameras}\n"
            f"  logs       : {self.agent.run_root}",
            flush=True,
        )

    # ------------------------------------------------- XPolicyLab model hooks

    def update_obs(self, obs):
        """Take one RoboDojo observation (images already decoded by the server)."""
        self.agent.observe(obs)

    def get_action(self):
        """Return this decision's motion envelope (see ``motion.py``)."""
        return self.agent.act()

    def reset(self):
        """Clear history at the start of an episode."""
        self.agent.reset()
        print(f"[vlm_agent] episode {self.agent.episode_index} start", flush=True)

    def report_execution(self, obs):
        """Custom RPC: the sim client reports what it actually executed.

        Called by ``deploy.eval_one_episode`` after every chunk, so the next
        prompt can mention IK failures, truncated chunks and the steps used.
        """
        self.agent.note_execution(obs)
        return {"ok": True}

    # RoboDojo only calls these when deploy.yml sets eval_batch: true. The VLM
    # loop is sequential by construction, so fail loudly instead of silently
    # evaluating a single env.
    def update_obs_batch(self, obs_list):
        raise NotImplementedError("vlm_agent runs one env at a time; keep eval_batch: false")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("vlm_agent runs one env at a time; keep eval_batch: false")


def _env_overrides() -> dict:
    raw = os.environ.get(OVERRIDES_ENV_VAR)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{OVERRIDES_ENV_VAR} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"{OVERRIDES_ENV_VAR} must be a JSON object, got {type(parsed).__name__}")
    print(f"[vlm_agent] applying {OVERRIDES_ENV_VAR}: {sorted(parsed)}", flush=True)
    return parsed
