"""Wall-X policy server for XPolicyLab.

Wraps the wall-x serving stack (``wall_x.serving.robodojo``), which implements the
native RoboDojo observation/action layout and the checkpoint's keyframe memory.
One instance per server process; ``deploy.eval_one_episode`` (one environment)
or ``deploy.eval_one_episode_batch`` (``eval_batch: true``) drives it.

The keyframe memory is per episode, but the client holds it: every request
carries its episode's memory state and gets the updated state back. The server
therefore keeps nothing between requests, and requests from different
environments can share one forward pass.

RPC surface (reachable through ``model_client.call(func_name=...)``):

======================  ====================================================
``reset``               new episode; returns the initial client memory state
``infer(obs)``          observation in, ``{actions, kfmem_state}`` out
``infer_batch(obs[])``  one ``infer`` result per observation, one forward
``update_obs`` /        the legacy pair; prefer ``infer`` so the two halves
``get_action``          cannot interleave with another request
======================  ====================================================
"""

from pathlib import Path

import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

#: The checkpoint predicts this many steps; the client replays the first half.
DEFAULT_ACTION_CHUNK = 32
DEFAULT_REPLAN = 16

#: Memory knobs that are safe to pin from deploy.yml; null keeps the checkpoint.
_MEMORY_KEYS = ("capacity", "num_candidates", "select_stride", "write_tail")

#: Action keys this checkpoint emits (dual arm, EE pose, native RoboDojo layout).
_EE_ACTION_KEYS = {
    "left_ee_pose": 7,
    "right_ee_pose": 7,
    "left_ee_joint_state": 1,
    "right_ee_joint_state": 1,
}


class Model(ModelTemplate):
    """XPolicyLab entry point; ``model_cfg`` is ``deploy.yml`` plus overrides."""

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg or {})
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        self.action_type = self.model_cfg.get("action_type")

        # Robot dimensions come from the shared lookup, never from a local table.
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self._check_action_layout()

        checkpoint_path = self._resolve_checkpoint()
        policy_config, memory_overrides = self._policy_config(checkpoint_path)

        from wall_x.serving.robodojo.memory_policy import (
            KeyframeMemoryRoboDojoPolicy,
        )
        from wall_x.serving.robodojo.policy import build_infer_config

        infer_config = build_infer_config(policy_config)
        self.policy = KeyframeMemoryRoboDojoPolicy(
            policy_config, infer_config, memory_overrides=memory_overrides
        )
        self.model = self.policy
        #: Episode state travels with each request, so environments may batch.
        self.eval_batch = True
        print(
            f"[Model] WallX ready: ckpt={checkpoint_path} "
            f"chunk={policy_config.action_chunk_length} "
            f"replan={policy_config.replan_steps} "
            f"memory={sorted(memory_overrides) or 'checkpoint defaults'}"
        )

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    def _check_action_layout(self):
        if self.action_type != "ee":
            raise ValueError(
                "WallX emits native EE-pose actions; deploy with action_type=ee "
                f"(got {self.action_type!r})"
            )
        ee_dim = self.robot_action_dim_info.get("ee_dim") or []
        if len(ee_dim) != 2 or any(int(dim) != 1 for dim in ee_dim):
            raise ValueError(
                "WallX expects a dual-arm EE robot (ee_dim [1, 1]), got "
                f"{self.robot_action_dim_info}"
            )

    def _resolve_checkpoint(self):
        explicit = self.model_cfg.get("checkpoint_path")
        if explicit:
            return str(explicit)
        policy_dir = Path(__file__).resolve().parent
        return str(
            resolve_checkpoint_root(
                self.model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir
            )
        )

    def _policy_config(self, checkpoint_path):
        from wall_x.serving.robodojo.policy import RoboDojoPolicyConfig

        action_chunk = self.model_cfg.get("action_chunk_length", DEFAULT_ACTION_CHUNK)
        config = RoboDojoPolicyConfig(
            checkpoint_path=checkpoint_path,
            train_config_path=self.model_cfg.get("train_config_path"),
            norm_key=self.model_cfg.get("norm_key", "ex_normal"),
            model_device=self.model_cfg.get("device", "cuda:0"),
            robot_type=self.model_cfg.get("robot_type", "desktop"),
            action_horizon=self.model_cfg.get("action_horizon"),
            num_inference_timesteps=int(self.model_cfg.get("num_inference_timesteps", 10)),
            action_start_index=int(self.model_cfg.get("action_start_index", 0)),
            replan_steps=int(self.model_cfg.get("replan_steps", DEFAULT_REPLAN)),
            action_chunk_length=None if action_chunk is None else int(action_chunk),
            image_resize=self.model_cfg.get("image_resize", "serving"),
            default_instruction=self.model_cfg.get("default_instruction", ""),
        )
        overrides = {
            key: int(self.model_cfg[key])
            for key in _MEMORY_KEYS
            if self.model_cfg.get(key) is not None
        }
        return config, overrides

    def _validate_actions(self, actions):
        for action in actions:
            if set(action) != set(_EE_ACTION_KEYS):
                raise ValueError(f"unexpected action keys: {sorted(action)}")
            for name, dim in _EE_ACTION_KEYS.items():
                array = np.asarray(action[name])
                if array.shape != (dim,) or not np.isfinite(array).all():
                    raise ValueError(f"invalid action entry: {name}")
        return actions

    # ------------------------------------------------------------------
    # ModelTemplate contract
    # ------------------------------------------------------------------
    def update_obs(self, obs):
        self.policy.update_obs(obs)

    def update_obs_batch(self, obs_list):
        raise NotImplementedError(
            "WallX returns each request's memory state with its actions; "
            "use infer_batch"
        )

    def get_action(self):
        return self._validate_actions(self.policy.get_action()["actions"])

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError(
            "WallX returns each request's memory state with its actions; "
            "use infer_batch"
        )

    def reset(self):
        return self.policy.reset()

    def infer(self, obs):
        """One request; safer than update_obs + get_action over two RPCs."""
        self.policy.update_obs(obs)
        result = self.policy.get_action()
        self._validate_actions(result["actions"])
        return result

    def infer_batch(self, obs_list):
        """Several environments' requests in one forward; results keep their order."""
        results = self.policy.infer_batch(list(obs_list))
        for result in results:
            self._validate_actions(result["actions"])
        return results
