"""The policy server's side of the PyRUA_Lean adapter.

There are no weights to load.  The "model" is an episode of PyRUA-Lean: a language model
writes Python against the ``robo`` API, one cell at a time, and each motion call becomes native
joint actions.  XPolicyLab's two calls reach it through a bridge
(``pyrualean.hosts.xpolicylab_bridge``): ``update_obs`` hands the program its next observation,
``get_action`` answers with the next action it has asked for, or with "wait" while it is
thinking, or with nothing once it has finished.

One episode at a time: ``reset`` ends whatever is running and the first observation after it
starts the next (the observation names the task and the layout, see ``deploy.py``).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from XPolicyLab.model_template import ModelTemplate

from pyrualean.hosts import xpolicylab_robodojo as host
from pyrualean.hosts.xpolicylab_bridge import STATUS_KEY, Bridge

from . import offline_programs

#: What the kinematics behind ``robo`` drive: two six-joint arms, one gripper joint each.
ARM_DIM, EE_DIM = [6, 6], [1, 1]
#: The robot of this adapter: colour images and the robot's state, no depth image.
ROBOT = "robodojo_xpl_rgb"


def _hold(obs: dict[str, Any]) -> dict[str, list[float]]:
    """The native action that keeps every joint and gripper where the observation has it."""

    state = obs.get("state") or {}
    return {
        key: [float(v) for v in getattr(value, "tolist", lambda value=value: value)()]
        for key, value in state.items()
        if key.endswith(("arm_joint_state", "ee_joint_state"))
    }


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = dict(model_cfg or {})
        self.poll_s = float(self.cfg.get("poll_s") or 20.0)
        self.wrap_up_s = float(self.cfg.get("wrap_up_s") or 30.0)
        self.bridge = Bridge(poll_s=self.poll_s)
        # fixed now: an episode makes its workspace the working directory
        self.result_dir = Path(
            os.environ.get("PYRUALEAN_RESULT_DIR") or self.cfg.get("result_dir") or "results"
        ).resolve()
        self._episode: threading.Thread | None = None
        self.results: list[dict[str, Any]] = []
        #: XPolicyLab's wiring check (EVAL_ENV_TYPE=debug) has no simulator behind it and no
        #: task to do: the policy then holds the robot where the observation has it, which
        #: runs the closed loop without a model writing programs
        self.debug = str(os.environ.get("EVAL_ENV_TYPE") or "").lower() == "debug"
        self._seen: dict[str, Any] = {}
        self._check_embodiment()
        # the code of every episode comes from programs/<task>.py, not from a language model
        if not self.debug:
            os.environ["OPENAI_BASE_URL"] = offline_programs.start()
            os.environ.setdefault("OPENAI_API_KEY", "offline")
            for var in ("PYRUALEAN_BASE_URL", "PYRUALEAN_API_KEY_ENV"):
                os.environ.pop(var, None)

    def _check_embodiment(self) -> None:
        """Joint targets for the ARX-X5 pair are all this policy sends."""

        action_type = self.cfg.get("action_type")
        if action_type not in (None, "joint"):
            raise ValueError(
                f"PyRUA_Lean sends joint targets: action_type must be 'joint', got {action_type!r}"
            )
        env_cfg_type = self.cfg.get("env_cfg_type")
        if not env_cfg_type:
            return
        from XPolicyLab.utils.process_data import get_robot_action_dim_info

        dims = get_robot_action_dim_info(env_cfg_type)
        if list(dims["arm_dim"]) != ARM_DIM or list(dims["ee_dim"]) != EE_DIM:
            raise ValueError(
                f"PyRUA_Lean drives two six-joint arms with one gripper joint each (arx_x5); "
                f"{env_cfg_type} has arm_dim {dims['arm_dim']} and ee_dim {dims['ee_dim']}"
            )

    # -- what XPolicyLab's policy server calls -------------------------------------------------

    def reset(self) -> None:
        if self.debug:
            self._seen = {}
            return
        # whatever was running belongs to an episode the client has left
        self.bridge.abandon()
        self.bridge = Bridge(poll_s=self.poll_s)
        self._episode = None
        self.bridge.reset()

    def update_obs(self, obs: dict[str, Any]) -> None:
        if self.debug:
            self._seen = obs
            return
        first = self._episode is None
        robot = self._robot() if first else ""
        self.bridge.update_obs(obs)
        if first:
            info = (obs.get("additional_info") or {}).get(STATUS_KEY) or {}
            self._episode = threading.Thread(
                target=self._play,
                args=(self.bridge, dict(info), robot),
                name="pyrualean-episode",
                daemon=True,
            )
            self._episode.start()

    def get_action(self) -> list[dict[str, Any]]:
        if self.debug:
            return [_hold(self._seen)]
        return self.bridge.get_action()

    def on_trial_end(self, result: dict[str, Any] | None = None) -> bool:
        """The benchmark's verdict, reported by ``deploy.py`` when the episode is over.

        Answered once the episode's record is on disk, or after ``wrap_up_s``: after its
        last trial the client exits, and the script that started this server stops it at
        once.  The wait must stay far below the client's request timeout (120 s): a late
        answer there is an exception in the client, and the trial is not recorded.
        """

        if self.debug:
            return True
        self.bridge.episode_end(result)
        episode = self._episode
        if episode is not None and episode is not threading.current_thread():
            episode.join(self.wrap_up_s)
        return True

    def update_obs_batch(self, obs_list: Any) -> None:
        raise NotImplementedError("one conversation drives one environment: eval_batch is false")

    def get_action_batch(self, env_idx_list: Any = None) -> Any:
        raise NotImplementedError("one conversation drives one environment: eval_batch is false")

    # -- the episode ---------------------------------------------------------------------------

    def _robot(self) -> str:
        """The robot the programs are written against (``robot`` of ``deploy.yml``)."""

        return str(self.cfg.get("robot") or ROBOT)

    def _play(self, bridge: Bridge, info: dict[str, Any], robot: str) -> None:
        from pyrualean.play import play_episode

        task = str(info.get("task_name") or self.cfg.get("task_name"))
        layout = int(info.get("layout", 0))
        gpu = self.cfg.get("gpu_id")
        cell = host.Cell(
            suite=str(self.cfg.get("env_cfg_type") or "arx_x5"),
            task=task,
            seed=layout,
            cuda_device=int(gpu) if gpu not in (None, "") else None,
        )
        offline_programs.select(task)
        out = self.result_dir / f"{task}_l{layout}_s{info.get('eval_seed', 0)}-{int(time.time())}"
        try:
            result = play_episode(
                cell,
                out,
                arm="cells",
                model=str(self.cfg.get("model") or "offline-program"),
                robot=robot,
                runtime=str(self.cfg.get("runtime") or "openai"),
                primitives=str(self.cfg.get("primitives") or "no-vla"),
                images=str(self.cfg.get("images") or "on-demand"),
                feedback=str(self.cfg.get("feedback") or "pure"),
                max_turns=int(self.cfg.get("max_turns") or 100),
                budget_s=float(self.cfg.get("budget_s") or 5400.0),
                base_url=os.environ.get("PYRUALEAN_BASE_URL"),
                api_key_env=os.environ.get("PYRUALEAN_API_KEY_ENV"),
                boot_options={"bridge": bridge},
                openai_options=dict(self.cfg.get("openai_options") or {}),
            )
            self.results.append(
                {k: result.get(k) for k in ("status", "environment_success", "task", "robot")}
            )
            print(
                f"[Model] {task} layout {layout}: status {result.get('status')}, "
                f"success {result.get('environment_success')}  ({out})",
                flush=True,
            )
        finally:
            # whatever happened, the client must not wait for an action that will not come
            bridge.finish()
