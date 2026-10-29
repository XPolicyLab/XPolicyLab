import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

POLICY_DIR = Path(__file__).resolve().parent
if str(POLICY_DIR / "src") not in sys.path:
    sys.path.insert(0, str(POLICY_DIR / "src"))

logger = logging.getLogger(__name__)
# RoboDojo camera name -> camera name in the policy input.
CAMERAS = {"cam_head": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}
# AR memory settings switched per episode between the default and memory_context.
CONTEXT_KEYS = ("infer_sliding_window", "infer_reset_every", "infer_reset_keep_chunks")


def _standardize_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _get_instruction(obs: dict) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("RoboDojo observation contains an empty instruction list.")
        value = value[0]
    if value is None:
        raise KeyError("RoboDojo observation is missing task_instruction/instruction/instructions.")
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip()
    if not text:
        raise ValueError("RoboDojo observation contains an empty instruction.")
    return text


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        from galaxea_va.inference.gripper_latch import GripperLatch
        from galaxea_va.inference.memory_router import MemoryRouter
        from galaxea_va.inference.policy import FastWAMARPolicy

        self.model_cfg = dict(model_cfg)
        if not isinstance(self.model_cfg["gripper_latch"], bool):
            raise TypeError(f"gripper_latch must be true/false, got {self.model_cfg['gripper_latch']!r}")
        self.action_type = self.model_cfg["action_type"]
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        if (self.env_cfg_type, self.action_type) != ("arx_x5", "joint"):
            raise ValueError("G07_preview supports env_cfg_type=arx_x5 and action_type=joint.")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.last_obs = None
        self.last_instruction = None

        checkpoint_root = resolve_checkpoint_root(self.model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
        self.model = FastWAMARPolicy(
            inference_config_path=checkpoint_root / "inference_config.yaml",
            checkpoint_path=checkpoint_root / "checkpoints/weights/step_020904.pt",
            **{key: self.model_cfg[key] for key in (
                "device", "action_infer_mode", "replan_steps", "num_action_denoise_steps",
                "num_video_denoise_steps", "sigma_shift", "negative_prompt", "rand_device",
                "compile_action_infer", "timing_enabled", *CONTEXT_KEYS)},
        )
        self.gripper_latch = (GripperLatch(self.model.robot_processor.gripper_full_scale)
                              if self.model_cfg["gripper_latch"] else None)
        self.action_horizon = self.model.action_horizon
        self.replan_steps = self.model.replan_steps
        # Per-episode AR memory: the policy's own UND decides from the instruction whether to use memory_context.
        self.memory_router = MemoryRouter(self.model.model)
        self.default_context = tuple(self.model_cfg[key] for key in CONTEXT_KEYS)
        self.memory_context = tuple(self.model_cfg["memory_context"][key] for key in CONTEXT_KEYS)
        for context in sorted({self.default_context, self.memory_context}):
            self._set_context(context)
            self.model.warmup_inference()
        self._set_context(self.default_context)

    def _set_context(self, context):
        for key, value in zip(CONTEXT_KEYS, context):
            setattr(self.model.model, key, value)

    def _encode_obs(self, obs: dict) -> dict:
        vision = obs["vision"]
        return {
            "observation": {
                target: {"rgb": _standardize_rgb(vision[source]["color"])} for source, target in CAMERAS.items()
            },
            "joint_action": {
                "vector": pack_robot_state(
                    obs, self.action_type, self.robot_action_dim_info, source_type="obs", state_type="state",
                ).astype(np.float32)
            },
        }

    def get_camera_sizes(self):
        """RoboDojo camera name -> (width, height) in the model input; deploy.py uploads images at this size."""
        from galaxea_va.inference.policy import CAMERA_SIZES_WH

        return {source: CAMERA_SIZES_WH[target] for source, target in CAMERAS.items()}

    def update_obs(self, obs):
        instruction = _get_instruction(obs)
        if self.last_instruction is None:
            # Choose once after episode reset, using only the visible instruction.
            margin = self.memory_router.margin(instruction)
            context = self.memory_context if margin > 0 else self.default_context
            self._set_context(context)
            logger.info("Memory context | window=%d reset=%d keep=%d | memory margin %+.3f | %s",
                        *context, margin, instruction)
        self.last_obs = self._encode_obs(obs)
        self.last_instruction = instruction

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("G07_preview uses one episode per server; set eval_batch=false.")

    def get_action(self):
        """Advance the policy until it needs the next real observation.

        AR horizon=16, observation stride=4:
        obs(0) -> actions[0:4] -> obs(4) -> actions[4:8] -> ...
        Each server owns one episode's cache; parallel envs need separate servers.
        """
        if self.last_obs is None:
            raise ValueError("No observation is available. Call update_obs() before get_action().")
        actions = []
        task_env = SimpleNamespace(
            get_instruction=lambda: self.last_instruction,
            take_action=lambda action, action_type: actions.append(action),
        )
        self.model.step(task_env, self.last_obs)
        while not self.model.should_request_observation():
            self.model.step(task_env, None)
        self.last_obs = None
        actions = unpack_robot_state(
            np.asarray(actions, dtype=np.float32), self.action_type, self.robot_action_dim_info, source_type="obs",
        )
        # These are denormalized simulator commands, not model-space values.
        if self.gripper_latch is not None:
            for action in actions:
                for key in ("left_ee_joint_state", "right_ee_joint_state"):
                    action[key] = np.array([self.gripper_latch.process(key, float(action[key][0]))], dtype=np.float32)
        return actions

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("G07_preview uses one episode per server; set eval_batch=false.")

    def get_timing(self):
        return {**self.model.get_timing_rollout(), "action_steps": self.model.step_count}

    def reset(self):
        self.last_obs = None
        self.last_instruction = None
        if self.gripper_latch is not None:
            self.gripper_latch.reset()
        self.model.reset()
