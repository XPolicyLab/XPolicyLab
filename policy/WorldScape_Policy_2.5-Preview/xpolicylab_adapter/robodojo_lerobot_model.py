"""RoboDojo image and action normalization."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import yaml

from .robodojo_model import (
    RoboDojoModel,
    _fields_to_robodojo_actions,
    _instruction,
    _rgb_camera,
    _state_fields,
)


class RoboDojoLeRobotModel(RoboDojoModel):
    """One current RGB anchor predicts 24 absolute 14-D joint actions.

    Only the transport and transactional execution loop are shared with the old
    Q99 adapter. Numeric transforms come exclusively from the new checkpoint.
    """

    _checkpoint_embodiment = "robodojo"
    _compose_diffusion_in_adapter = False
    _reseed_independent_blocks = False
    _camera_order = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    _expected_observation_interval = 3
    _expected_observation_history_segments = 8
    _expected_history_memory = False
    _expected_mode = "interactive"

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        # The builder now casts before GPU transfer. Keep the adapter's explicit
        # BF16 contract, matching the training parameters, for subclass loaders.
        self.policy.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def get_action_batch(self, env_idx_list=None):
        # Training also runs the forward under BF16 autocast. State/image
        # normalization remains FP32 before the model's typed input transfer.
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            return super().get_action_batch(env_idx_list)

    def _validate_checkpoint_contract(self, checkpoint_config):
        memory = getattr(checkpoint_config, "history_memory", None)
        memory_enabled = bool(memory is not None and memory.enabled)
        if memory_enabled != self._expected_history_memory:
            raise ValueError(
                "Checkpoint History memory does not match the deployment; "
                "select deploy.yml for History checkpoints"
            )
        if self.model_cfg.get("history_memory", False) != self._expected_history_memory:
            raise ValueError("Deployment history_memory must match the selected adapter")
        if (list(self.robot_action_dim_info["arm_dim"]) != [6, 6]
                or list(self.robot_action_dim_info["ee_dim"]) != [1, 1]):
            raise ValueError("LeRobot joint checkpoint requires dual-arm widths [6,1,6,1]")
        embodiment = self.bundle.transform.embodiment
        for name, key in (("state_fields", "state.vector"), ("action_fields", "action.vector")):
            fields = tuple(getattr(embodiment, name))
            if len(fields) != 1 or (fields[0].key, fields[0].size,
                                    fields[0].absolute, fields[0].normalization) != (
                                        key, 14, True, "mean_std"):
                raise ValueError(f"LeRobot {name} must contain one absolute Z-score {key} of width 14")
            field = fields[0]
            stats = field.statistics
            if field.per_horizon_statistics is not None:
                raise ValueError("LeRobot checkpoint requires global Z-score statistics")
            for stat, expected in (("clip_min", -5.0), ("clip_max", 5.0)):
                if stat not in stats or not np.all(np.asarray(stats[stat]) == expected):
                    raise ValueError(f"LeRobot {key} requires {stat}={expected}")
        if (self.bundle.max_state_dim, self.bundle.max_action_dim) != (64, 32):
            raise ValueError("LeRobot checkpoint requires padded state/action widths 64/32")
        if self.bundle.embodiment_id != 0:
            raise ValueError("LeRobot checkpoint requires embodiment_id=0")
        shape = checkpoint_config.shape
        if (shape.action_horizon, shape.actions_per_block, shape.num_frames) != (24, 24, 9):
            raise ValueError("LeRobot checkpoint requires a 24-action / 9-frame training block")
        if (shape.max_state_dim, shape.action_dim) != (64, 32):
            raise ValueError("LeRobot model requires padded state/action widths 64/32")
        parameters = checkpoint_config.wam.core.parameters
        if checkpoint_config.wam.plugin != "wan22" or checkpoint_config.wam.variant != "ti2v-5b":
            raise ValueError("This adapter requires the Wan2.2 TI2V 5B backbone")
        layout = "mosaic_T"
        frame_seqlen = 120
        if parameters.get("max_chunk_size") != 1 or parameters.get("frame_seqlen") != frame_seqlen:
            raise ValueError("Deployment requires max_chunk_size=1 and frame_seqlen=120")
        if (getattr(self.mode, "value", self.mode) != self._expected_mode
                or getattr(checkpoint_config.mode, "value", checkpoint_config.mode) != self._expected_mode):
            raise ValueError(f"LeRobot deployment requires mode={self._expected_mode}")
        required = {
            "input_color_order": "rgb",
            "diffusion_concat_height": 384,
            "diffusion_concat_width": 320,
            "single_window_reference": "latest",
            "use_observation_history": False,
            "use_causal_memory": False,
            "reset_independent_blocks": True,
            "constrain_action_padding": False,
            "wam_history_mode": "periodic",
            "t5_prompt_template": None,
            "inference_precision": "bf16",
        }
        if (self.action_horizon, self.action_steps, self.observation_interval) != (
            24, 24, self._expected_observation_interval,
        ):
            raise ValueError(
                "LeRobot deployment requires horizon=24, replan_steps=24, "
                f"observation_interval={self._expected_observation_interval}"
            )
        if self.observation_history_segments != self._expected_observation_history_segments:
            raise ValueError(
                "LeRobot deployment requires "
                f"observation_history_segments={self._expected_observation_history_segments}"
            )
        for key, expected in required.items():
            if self.model_cfg.get(key) != expected:
                raise ValueError(
                    f"LeRobot deployment requires {key}={expected!r}; "
                    "select deploy.yml"
                )
        if tuple(self.model_cfg.get("diffusion_concat_view_order", ())) != (0, 1, 2):
            raise ValueError("LeRobot native camera order must be [head,left,right] / [0,1,2]")
        if (checkpoint_config.visual_memory.diffusion_view_layout != layout
                or self.model_cfg.get("diffusion_view_layout") != layout):
            raise ValueError(f"LeRobot deployment requires diffusion_view_layout={layout}")
        if self.bundle.transform.image_input_range != "zero_one":
            raise ValueError("LeRobot checkpoint image input range must be zero_one")

    def _checkpoint_builder_options(self):
        # The training checkout's native sampler has no padding-constraint API.
        overrides = self.model_cfg.get("generation_overrides")
        if not overrides:
            return {"inference_dtype": torch.bfloat16}
        generation = yaml.safe_load((self.checkpoint / "generation_config.yaml").read_text())
        return {"generation_config": {**generation, **overrides}, "inference_dtype": torch.bfloat16}

    def update_obs_batch(self, obs_list):
        super().update_obs_batch(obs_list)
        for env_idx in self._latest_env_idx_list:
            # This checkpoint has no observation history. Avoid retaining native
            # camera frames that cannot condition its single training block.
            self._obs_history_by_env[env_idx] = [self._latest_obs_by_env[env_idx]]

    def _prepare_observation_cameras(self, env_idx):
        obs = self._latest_obs_by_env[env_idx]
        return [_rgb_camera(obs, camera)[None] for camera in self._camera_order]

    def _adapt_observation(self, env_idx, *, cameras=None):
        obs = self._latest_obs_by_env.get(env_idx)
        if obs is None:
            raise RuntimeError(f"Call update_obs_batch() for env {env_idx} before predicting")
        if cameras is None:
            cameras = self._prepare_observation_cameras(env_idx)
        fields = _state_fields(obs, robot_action_dim_info=self.robot_action_dim_info)
        raw_state = {"state.vector": np.concatenate([
            fields["state.left_joint"], fields["state.left_gripper"],
            fields["state.right_joint"], fields["state.right_gripper"],
        ], axis=-1)}
        # Interactive training bypasses _apply_prompt_schedule, including its
        # optional T5 wrapper. Preserve the raw task text used by that path.
        instruction = _instruction(obs, self.default_instruction)
        adapter = self._observation_adapter_class(
            self.bundle, mode=self.mode, instruction=instruction,
            visual_input_range=self.visual_input_range, device=self.device,
            session_id=f"robodojo-{env_idx}",
        )
        # Retain raw native views. The checkpoint runtime normalizes and resizes
        # once, identically for the VAE input and current reference image.
        adapted = adapter((*cameras, raw_state, self._observation_index_by_env[env_idx]))
        adapted.prompts.visual_prompt = "none"
        return adapted

    def _decode_actions(self, output, current_state: Mapping):
        action = output.require_action().detach().float().cpu()
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        if action.shape != (24, self.bundle.max_action_dim):
            raise ValueError(f"LeRobot policy must return [24,{self.bundle.max_action_dim}], got {tuple(action.shape)}")
        vector = self.bundle.transform.unapply({"action": action})["action.vector"]
        vector = torch.as_tensor(vector).numpy()
        if vector.shape != (24, 14) or not np.isfinite(vector).all():
            raise ValueError("LeRobot decoded actions must be finite with shape [24,14]")
        fields = {
            "action.left_joint": vector[:, :6],
            "action.left_gripper": vector[:, 6:7],
            "action.right_joint": vector[:, 7:13],
            "action.right_gripper": vector[:, 13:14],
        }
        return _fields_to_robodojo_actions(
            fields, action_steps=self.action_steps, robot_action_dim_info=self.robot_action_dim_info,
        )
