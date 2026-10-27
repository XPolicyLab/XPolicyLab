"""XPolicyLab bridge for Phoenix-MWA RoboTwin joint-action inference."""

import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    get_batch_size,
    pack_robot_state,
    unpack_robot_state,
)
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from .runtime.loader import POLICY_DIR, build_runtime


class Model(ModelTemplate):
    """Translate XPolicyLab RGB observations and joint actions for Phoenix-MWA."""

    def __init__(self, model_cfg):
        self.cfg = dict(model_cfg)
        if (
            self.cfg.get("action_type") != "joint"
            or self.cfg.get("env_cfg_type") != "arx_x5"
        ):
            raise ValueError(
                "RoboTwin aloha-agilex uses env_cfg_type=arx_x5 and action_type=joint"
            )
        self.dims = get_robot_action_dim_info(self.cfg["env_cfg_type"])
        if list(self.dims["arm_dim"]) != [6, 6] or list(self.dims["ee_dim"]) != [1, 1]:
            raise ValueError(
                "Checkpoint requires two 6-DoF arms and one scalar gripper per arm"
            )
        self.action_dim = sum(self.dims["arm_dim"]) + sum(self.dims["ee_dim"])
        self.batch_size = get_batch_size(self.cfg["env_cfg_type"])
        root = resolve_checkpoint_root(
            self.cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR
        )
        self.runtime, self.new_session = build_runtime(root, self.cfg)
        self.single = self.new_session()
        self.sessions, self.observations, self.order = {}, {}, []
        self.last = None

    def _adapt(self, obs):
        state = np.asarray(
            pack_robot_state(
                obs, "joint", self.dims, source_type="obs", state_type="state"
            ),
            dtype=np.float32,
        )
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError("Invalid native joint state")
        images = {}
        for source, target in [
            ("cam_head", "head_camera"),
            ("cam_left_wrist", "left_camera"),
            ("cam_right_wrist", "right_camera"),
        ]:
            image = np.asarray(obs["vision"][source]["color"])
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{source} must be decoded uint8 HWC RGB")
            images[target] = {"rgb": np.ascontiguousarray(image)}
        instruction = obs.get("instruction", obs.get("instructions"))
        if isinstance(instruction, (list, tuple)) and len(instruction) == 1:
            instruction = instruction[0]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("A nonempty instruction is required")
        return {"observation": images, "joint_action": {"vector": state}}, instruction

    def _ingest(self, session, obs):
        adapted, prompt = self._adapt(obs)
        if session.pending_model_actions:
            session.update_obs(adapted)
        return adapted, prompt

    def update_obs(self, obs):
        self.last = self._ingest(self.single, obs)

    def update_obs_batch(self, obs_list):
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        indices = [int(o["env_idx"]) for o in obs_list]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Expected unique env_idx values")
        for env_idx, obs in zip(indices, obs_list):
            if env_idx not in self.sessions:
                self.sessions[env_idx] = self.new_session()
            self.observations[env_idx] = self._ingest(self.sessions[env_idx], obs)
        self.order = indices

    def _predict(self, session, pair):
        if pair is None:
            raise ValueError("Call update_obs before get_action")
        obs, prompt = pair
        chunk = np.asarray(
            session.get_action({"observation": obs, "instruction": prompt}),
            dtype=np.float32,
        )
        if (
            chunk.ndim != 2
            or chunk.shape[1] != self.action_dim
            or not np.isfinite(chunk).all()
        ):
            raise ValueError("Invalid policy action chunk")
        # No extra clipping: preserve the released policy's native joint/gripper outputs.
        return unpack_robot_state(chunk, "joint", self.dims, source_type="obs")

    def get_action(self):
        return self._predict(self.single, self.last)

    def get_action_batch(self, env_idx_list=None):
        indices = self.order if env_idx_list is None else [int(i) for i in env_idx_list]
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(i not in self.observations for i in indices)
        ):
            raise ValueError("Missing or duplicate environment observations")
        return [self._predict(self.sessions[i], self.observations[i]) for i in indices]

    def reset(self):
        self.single.reset_model()
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()
        self.observations.clear()
        self.order = []
        self.last = None
