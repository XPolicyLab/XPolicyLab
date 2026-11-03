"""XPolicyLab boundary for the hash-bound RSIWAM inference composition.

The vendored native runtime is unchanged. This boundary resolves XPolicyLab
paths and robot metadata, and separates the evaluation seed from model seed 1.
No model weights are loaded, and CUDA is not initialized, on module import.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from .bundle.runtime import rsiwam_loader
from .bundle.runtime.rsiwam_native_wrapper import Model as NativeModel

POLICY_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = POLICY_DIR / "bundle"
POSE_DIM = 7  # The XPolicyLab EE pose contract: xyz + qwxyz, independent of joints.
MODEL_SEED = 1


def prepare_configuration(model_cfg):
    """Validate the frozen policy contract before loading any base assets."""
    cfg = dict(model_cfg)
    fixed = {
        "bench_name": "RoboDojo",
        "env_cfg_type": "arx_x5",
        "action_type": "ee",
        "deployment_adapter": "robodojo_ee16",
        "model_seed": MODEL_SEED,
        "num_inference_steps": 10,
        "sigma_shift": 1,
        "replan_steps": 24,
        "action_horizon": 32,
        "mixed_precision": "bf16",
        "device": "cuda",
        "eval_batch": False,
    }
    for key, value in fixed.items():
        if cfg.get(key) is not None and cfg[key] != value:
            raise ValueError(f"RSIWAM requires {key}={value!r}")
    cfg.update(fixed)
    if cfg.get("recovery_adapter") is not None:
        raise ValueError("RSIWAM always uses the bundled hash-bound recovery parent")
    dimensions = get_robot_action_dim_info(cfg["env_cfg_type"])
    arm_dims, ee_dims = dimensions["arm_dim"], dimensions["ee_dim"]
    if len(arm_dims) != 2 or list(ee_dims) != [1, 1]:
        raise ValueError("The frozen RSIWAM checkpoint requires two arms with one gripper value each")
    action_dim = len(arm_dims) * POSE_DIM + sum(ee_dims)
    if cfg.get("action_dim") is not None and cfg["action_dim"] != action_dim:
        raise ValueError(f"action_dim must match the registered EE dimensions: {action_dim}")
    cfg["action_dim"] = action_dim
    # XPolicyLab's standard seed is an environment/evaluation seed. It must
    # never alter the seed used by the fixed trained model's inference sampler.
    evaluation_seed = cfg.get("seed")
    if evaluation_seed is not None and (
        isinstance(evaluation_seed, bool)
        or not isinstance(evaluation_seed, (int, np.integer))
        or evaluation_seed < 0
    ):
        raise ValueError("seed must be a non-negative integer evaluation seed")
    return cfg, dimensions, evaluation_seed


def resolve_base_root(model_cfg):
    """Use the shared precedence; every selected base asset is then rehashed."""
    cfg = dict(model_cfg)
    explicit = cfg.get("rsiwam_weights_root") or os.environ.get("RSIWAM_BASE_WEIGHTS")
    if explicit:
        cfg["rsiwam_weights_root"] = explicit
    return resolve_checkpoint_root(
        cfg,
        POLICY_DIR / "checkpoints",
        policy_dir=POLICY_DIR,
        explicit_keys=(
            "rsiwam_weights_root", "model_path", "checkpoint_path",
            "ckpt_path", "model_dir", "pretrained_path",
        ),
    )


class _RegisteredNativeModel(NativeModel):
    """Keep native history/client handling with registered action dimensions."""

    def __init__(self, model_cfg, dimensions):
        self.robot_action_dim_info = dimensions
        self._ee_dims = tuple(dimensions["ee_dim"])
        self._packed_action_dim = len(dimensions["arm_dim"]) * POSE_DIM + sum(self._ee_dims)
        super().__init__(model_cfg)

    def _pack_robodojo_ee16_state(self, obs):
        parts = []
        for side, ee_dim in zip(("left", "right"), self._ee_dims):
            parts.append(np.asarray(obs["state"][f"{side}_ee_pose"], dtype=np.float32).reshape(POSE_DIM))
            parts.append(np.asarray(obs["state"][f"{side}_ee_joint_state"], dtype=np.float32).reshape(ee_dim))
        return np.concatenate(parts, axis=0)

    def _unpack_robodojo_ee16_actions(self, actions):
        packed = np.asarray(actions, dtype=np.float32)
        if packed.ndim == 1:
            packed = packed[None, :]
        if packed.ndim != 2 or packed.shape[1] != self._packed_action_dim:
            raise ValueError(f"Expected actions [T,{self._packed_action_dim}], got {packed.shape}")
        result = []
        for row in packed:
            action, offset = {}, 0
            for side, ee_dim in zip(("left", "right"), self._ee_dims):
                action[f"{side}_ee_pose"] = row[offset:offset + POSE_DIM]
                offset += POSE_DIM
                action[f"{side}_ee_joint_state"] = row[offset:offset + ee_dim]
                offset += ee_dim
            result.append(action)
        return result


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the policy process")
        cfg, self.robot_action_dim_info, self.evaluation_seed = prepare_configuration(model_cfg)
        self.model_cfg = dict(cfg)
        self.action_type, self.env_cfg_type = cfg["action_type"], cfg["env_cfg_type"]
        self.action_dim = cfg["action_dim"]
        artifacts = rsiwam_loader.check_package()
        weights = rsiwam_loader.check_base(resolve_base_root(cfg))
        # install.sh installs this exact source tree in editable mode. Refuse a
        # different RecoverWAM package before importing its model implementation.
        import recoverwam
        if Path(recoverwam.__file__).resolve() != BUNDLE_DIR / "src/recoverwam/__init__.py":
            raise ValueError("Install this adapter with install.sh in the selected policy environment")
        cfg.update(
            seed=MODEL_SEED,
            policy_name="RSIWAM",
            base_identity=rsiwam_loader.BASE,
            recovery_adapter=None,
            checkpoint_path=str(weights / "joint2b_s100000/action.pt"),
            video_checkpoint_path=str(weights / "joint2b_s100000/video.pt"),
            dataset_stats_path=str(weights / "joint2b_s100000/dataset_stats.json"),
            wan_model_dir=str(weights / "Wan2.1-I2V-14B-480P"),
        )
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(MODEL_SEED)
        np.random.seed(MODEL_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = False
        self._native = _RegisteredNativeModel(cfg, self.robot_action_dim_info)
        self.model = self._native.model
        self.model.model.requires_grad_(False)
        self.model.model.eval()
        rsiwam_loader.compose(self.model, artifacts)

    def update_obs(self, obs):
        # Colors are already RGB arrays, decoded by the XPolicyLab server.
        return self._native.update_obs(obs)

    def get_action(self, obs=None):
        import torch
        with torch.inference_mode():
            return self._native.get_action(obs)

    def reset(self):
        return self._native.reset()

    def reset_client(self, obs=None):
        return self._native.reset_client(obs)

    def set_request_context(self, context):
        return self._native.set_request_context(context)

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("RSIWAM requires eval_batch=false; use the single-environment deployment loop")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("RSIWAM requires eval_batch=false; use the single-environment deployment loop")
