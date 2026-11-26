"""Run one instruction-selected skill per episode against the native joint-action bridge."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from XPolicyLab.model_template import ModelTemplate

from . import episode_runner, sam3_http_client, selector
from .bridge import MOTION_BINDING, STATUS_KEY, EpisodeBridge, robot_observation

#: What the kinematics behind ``robo`` drive: two six-joint arms, one gripper joint each.
ARM_DIM, EE_DIM = [6, 6], [1, 1]
#: The robot of this adapter: colour images and the robot's state, no depth image.
ROBOT = "robodojo_xpl_rgb"
MODEL_CONFIG_FIELDS = ("poll_s", "wrap_up_s", "result_dir", "action_type", "env_cfg_type", "sam3_url")


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
        self.cfg = {key: value for key, value in (model_cfg or {}).items() if key in MODEL_CONFIG_FIELDS}
        self.poll_s = float(self.cfg.get("poll_s") or 20.0)
        self.wrap_up_s = float(self.cfg.get("wrap_up_s") or 30.0)
        self.bridge = EpisodeBridge(poll_s=self.poll_s)
        # fixed now: an episode makes its workspace the working directory
        self.result_dir = Path(
            os.environ.get("HARNESS_RESULT_DIR") or self.cfg.get("result_dir") or "results"
        ).resolve()
        self._episode: threading.Thread | None = None
        self.results: list[dict[str, Any]] = []
        #: XPolicyLab's wiring check (EVAL_ENV_TYPE=debug) has no simulator behind it and no
        #: task to do: the policy then holds the robot where the observation has it, which
        #: runs the closed loop without a model writing programs
        self.debug = str(os.environ.get("EVAL_ENV_TYPE") or "").lower() == "debug"
        self._seen: dict[str, Any] = {}
        self._check_embodiment()
        # Step 2a: route SAM3 segmentation through our persistent HTTP service instead of loading LocalSam3 in-process
        sam3_url = os.environ.get("HARNESS_SAM3_URL") or str(self.cfg.get("sam3_url") or "http://127.0.0.1:8810")
        self._sam3_client = sam3_http_client.HttpSam3(base_url=sam3_url)
        print(f"[Model] Mode: {'debug hold' if self.debug else 'task program'}; SAM3: {sam3_url}", flush=True)

    def _check_embodiment(self) -> None:
        """Joint targets for the ARX-X5 pair are all this policy sends."""

        action_type = self.cfg.get("action_type")
        if action_type not in (None, "joint"):
            raise ValueError(
                f"HarnessTree sends joint targets: action_type must be 'joint', got {action_type!r}"
            )
        env_cfg_type = self.cfg.get("env_cfg_type")
        if not env_cfg_type:
            return
        from XPolicyLab.utils.process_data import get_robot_action_dim_info

        dims = get_robot_action_dim_info(env_cfg_type)
        if list(dims["arm_dim"]) != ARM_DIM or list(dims["ee_dim"]) != EE_DIM:
            raise ValueError(
                f"HarnessTree drives two six-joint arms with one gripper joint each (arx_x5); "
                f"{env_cfg_type} has arm_dim {dims['arm_dim']} and ee_dim {dims['ee_dim']}"
            )

    # -- what XPolicyLab's policy server calls -------------------------------------------------

    def reset(self) -> None:
        if self.debug:
            self._seen = {}
            return
        # whatever was running belongs to an episode the client has left
        self.bridge.abandon()
        self.bridge = EpisodeBridge(poll_s=self.poll_s)
        self._episode = None
        self.bridge.reset()

    def update_obs(self, obs: dict[str, Any]) -> None:
        if self.debug:
            info = (obs.get("additional_info") or {}).get(STATUS_KEY) or {}
            self._seen = {
                "state": robot_observation(obs)["state"],
                "additional_info": {STATUS_KEY: {key: info.get(key) for key in ("episode_id", "observation_step")}},
            }
            return
        first = self._episode is None
        self.bridge.update_obs(obs)
        if first:
            self._episode = threading.Thread(
                target=self._play,
                args=(self.bridge,),
                name="harnesstree-episode",
                daemon=True,
            )
            self._episode.start()

    def get_action(self) -> list[dict[str, Any]]:
        if self.debug:
            info = (self._seen.get("additional_info") or {}).get(STATUS_KEY) or {}
            action = _hold(self._seen)
            return [{**action, MOTION_BINDING: [info.get("episode_id"), info.get("observation_step")]}]
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

    def _play(self, bridge: EpisodeBridge) -> None:
        try:
            instruction = bridge.first_obs(120.0)["instruction"]
            skill = selector.select_skill(instruction)
            print(f"[Model] selected skill={skill!r} from instruction={instruction[:100]!r}", flush=True)
            out = self.result_dir / f"{skill}-{uuid4().hex}"
            result = episode_runner.run_episode(
                bridge=bridge,
                sam3_client=self._sam3_client,
                skill=skill,
                out_dir=out,
            )
            self.results.append(
                {k: result.get(k) for k in ("status", "environment_success", "task", "robot")}
            )
            print(
                f"[Model] {skill}: status {result.get('status')}, "
                f"success {result.get('environment_success')}  ({out})",
                flush=True,
            )
        finally:
            # whatever happened, the client must not wait for an action that will not come
            bridge.finish()
