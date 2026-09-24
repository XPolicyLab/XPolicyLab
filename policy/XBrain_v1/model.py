import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.XBrain_v1.gripper_thresholds import (
    apply_gripper_thresholds,
    load_gripper_thresholds,
    normalize_prompt,
)
from XPolicyLab.utils.process_data import get_batch_size, get_robot_action_dim_info, pack_robot_state


class Model(ModelTemplate):
    _ROBOT_RESOURCES = {
        "piper_x": {"embodiment_id": 6, "norm_env": "XBRAIN_PIPERX_NORM_STATS_PATH", "ckpt_env": "XBRAIN_PIPERX_CHECKPOINT_PATH"},
        "piper": {"embodiment_id": 6, "norm_env": "XBRAIN_PIPER_NORM_STATS_PATH", "ckpt_env": "XBRAIN_PIPER_CHECKPOINT_PATH"},
        "arx_x5": {"embodiment_id": 0, "norm_env": "XBRAIN_ARX_X5_NORM_STATS_PATH", "ckpt_env": "XBRAIN_ARX_X5_CHECKPOINT_PATH"},
    }
    def __init__(self, model_cfg):
        # Store the configuration
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if self.env_cfg_type not in self._ROBOT_RESOURCES:
            raise ValueError(f"XBrain_v1 does not support env_cfg_type={self.env_cfg_type!r}")

        if self.action_type != "joint":
            raise ValueError(f"XBrain_v1 currently supports action_type=joint, got {self.action_type!r}")
        self.action_horizon = int(model_cfg.get("action_horizon", 30))
        if self.action_horizon <= 0 or self.action_horizon > 30:
            raise ValueError("action_horizon must be in [1, 30]")
        threshold_path = Path(__file__).with_name("gripper_thresholds.json")
        self._gripper_thresholds = load_gripper_thresholds(threshold_path, self.env_cfg_type)
        self._reported_gripper_prompts = set()
        self._warned_gripper_prompts = set()

        # Get robot action dimension metadata
        # Example:
        # {
        #     "arm_dim": [7] or [7, 7],
        #     "ee_dim": [1] or [1, 1]
        # }
        try:
            self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        except (FileNotFoundError, KeyError) as exc:
            raise RuntimeError(
                f"XPolicyLab robot metadata is missing for env_cfg_type={self.env_cfg_type!r}; "
                "install the RoboDojo env_cfg entry and matching utils/robot/_robot_info.json."
            ) from exc

        # Real-robot evaluation is always single-environment. Batch size comes
        # from simulator metadata only when the adapter is explicitly launched
        # with eval_batch=true.
        if bool(model_cfg.get("eval_batch", False)):
            try:
                self.batch_size = get_batch_size(self.env_cfg_type)
            except (FileNotFoundError, KeyError) as exc:
                raise RuntimeError(
                    f"Simulator batch metadata is missing for env_cfg_type={self.env_cfg_type!r}; "
                    "set eval_batch=false for real-robot evaluation."
                ) from exc
        else:
            self.batch_size = 1

        # The number of arm and EE entries must match, e.g. both are 2 for dual-arm robots
        assert len(self.robot_action_dim_info["arm_dim"]) == len(self.robot_action_dim_info["ee_dim"]), \
            "Arm and EE action dimensions must match"
        if self.robot_action_dim_info != {"arm_dim": [6, 6], "ee_dim": [1, 1]}:
            raise ValueError(f"XBrain_v1 requires dual 6-DoF arms for {self.env_cfg_type}: {self.robot_action_dim_info}")

        self._obs = None
        self._obs_batch = None
        self._pipe = self._load_pipeline(model_cfg)
        print(f"[XBrain_v1] initialized real pipeline for {self.env_cfg_type}")

    def _load_pipeline(self, model_cfg):
        resource = self._ROBOT_RESOURCES[self.env_cfg_type]
        checkpoint = (os.environ.get(resource["ckpt_env"])
                      or os.environ.get("XBRAIN_CHECKPOINT_PATH")
                      or model_cfg.get("checkpoint_path"))
        if not checkpoint and model_cfg.get("ckpt_name"):
            policy_dir = Path(__file__).resolve().parent
            checkpoint = policy_dir / model_cfg.get("checkpoint_root", "./checkpoints") / model_cfg["ckpt_name"]
        if checkpoint:
            checkpoint = str(Path(checkpoint).expanduser())
        if not checkpoint or not Path(checkpoint).is_dir():
            raise FileNotFoundError(f"Missing {self.env_cfg_type} checkpoint: {checkpoint!r}")
        runtime_root = os.environ.get("XBRAIN_RUNTIME_ROOT") or model_cfg.get("runtime_root")
        if runtime_root:
            runtime_path = Path(runtime_root).resolve()
            if str(runtime_path) not in sys.path:
                sys.path.insert(0, str(runtime_path))
        norm_path = (os.environ.get(resource["norm_env"])
                     or os.environ.get("XBRAIN_NORM_STATS_PATH")
                     or model_cfg.get("norm_stats_path"))
        tokenizer_path = os.environ.get("XBRAIN_TOKENIZER_PATH") or model_cfg.get("tokenizer_model_path")
        fast_tokenizer_path = os.environ.get("XBRAIN_FAST_TOKENIZER_PATH") or model_cfg.get("fast_tokenizer_path")
        required = {"norm_stats_path": norm_path, "tokenizer_model_path": tokenizer_path,
                    "fast_tokenizer_path": fast_tokenizer_path}
        missing = [name for name, value in required.items() if not value or not Path(value).exists()]
        if missing:
            raise FileNotFoundError(f"XBrain_v1 checkpoint is set but resources are missing: {missing}")
        # This must be the checkpoint-matched XBrain runtime, not an arbitrary
        # upstream giga-models installation visible on PYTHONPATH.
        from xbrain_v1_runtime import GigaBrain0Pipeline, runtime_info
        print(f"[XBrain_v1] runtime={runtime_info()}")
        with open(norm_path, encoding="utf-8") as handle:
            stats = json.load(handle)["norm_stats"]
        mask = [True, True, True, True, True, True, False,
                True, True, True, True, True, True, False]
        pipe = GigaBrain0Pipeline(
            model_path=str(checkpoint), tokenizer_model_path=str(tokenizer_path),
            fast_tokenizer_path=str(fast_tokenizer_path), embodiment_id=resource["embodiment_id"],
            state_norm_stats=stats["observation.state"], action_norm_stats=stats["action"],
            delta_mask=mask, original_action_dim=14, discrete_state_input=False,
            encode_sub_task_input=False, resize_imgs_with_padding=(224, 224),
            prompt_max_length=int(model_cfg.get("prompt_max_length", 120)),
            enable_control_mode_token=True, enable_end_effector_token=True,
        )
        # Training masks noise on the padded action dimensions. Keep the same
        # runtime contract: only the 14 RoboDojo action dimensions are denoised.
        denoise_mask = [True] * 14 + [False] * (pipe.policy.max_action_dim - 14)
        pipe.policy.set_action_denoise_mask(denoise_mask)
        print("[XBrain_v1] action denoise mask enabled: 14/{} dims".format(pipe.policy.max_action_dim))
        device = os.environ.get("XBRAIN_DEVICE", "cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "XBrain_v1 requires CUDA by default, but CUDA is unavailable. "
                "Set XBRAIN_DEVICE=cpu only for a functional smoke test."
            )
        pipe.to(device)
        return pipe

    def update_obs(self, obs):
        # Update a single observation here if needed
        self._obs = obs
        self._obs_batch = None

    def update_obs_batch(self, obs_list):
        # Update a batch of observations here if needed
        self._obs_batch = list(obs_list)
        self._obs = None

    def get_action(self):
        if self._pipe is not None:
            if self._obs is None:
                raise RuntimeError("update_obs must be called before get_action")
            obs = self._obs
            vision = obs.get("vision", {})
            images = {}
            for source, target in (("cam_head", "observation.images.cam_high"),
                                   ("cam_left_wrist", "observation.images.cam_left_wrist"),
                                   ("cam_right_wrist", "observation.images.cam_right_wrist")):
                # Websocket/msgpack may deserialize images as read-only views.
                # Copy before torch.from_numpy so the tensor always owns writable memory.
                image = self._prepare_image(vision[source]["color"], source)
                images[target] = torch.from_numpy(image).to(self._pipe.device)
            raw_state = obs["state"]
            if isinstance(raw_state, dict):
                state = pack_robot_state(
                    obs,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                ).astype(np.float32)
            else:
                state = np.asarray(raw_state, dtype=np.float32)
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError(f"state must be finite with shape (14,), got {state.shape}")
            prompt = str(obs.get("instruction") or obs.get("task_instruction") or
                         self.model_cfg.get("default_prompt", "")).strip()
            predicted = self._pipe(images, prompt, torch.from_numpy(state).to(self._pipe.device),
                                    is_robot_moving=False, is_body_moving=False)
            predicted = np.asarray(predicted.detach().float().cpu())
            if predicted.shape != (50, 14) or not np.isfinite(predicted).all():
                raise ValueError(f"pipeline returned unexpected action shape/values: {predicted.shape}")
            # The checkpoint predicts 50 steps, but the executor receives at
            # most the first 30 before collecting a fresh observation.
            predicted = predicted[:self.action_horizon]
            predicted = self._apply_prompt_gripper_thresholds(predicted, prompt)
        return self._format_pipeline_actions(predicted)

    def _apply_prompt_gripper_thresholds(self, predicted, prompt):
        prompt_key = normalize_prompt(prompt)
        rule = self._gripper_thresholds.get(prompt_key)
        if rule is None:
            if prompt_key not in self._warned_gripper_prompts:
                print(
                    f"[XBrain_v1] no gripper thresholds matched env={self.env_cfg_type} "
                    f"prompt={prompt!r}; preserving predicted gripper values"
                )
                self._warned_gripper_prompts.add(prompt_key)
            return predicted

        if prompt_key not in self._reported_gripper_prompts:
            print(
                f"[XBrain_v1] gripper thresholds task={rule.task} "
                f"left={rule.left} right={rule.right}"
            )
            self._reported_gripper_prompts.add(prompt_key)
        return apply_gripper_thresholds(predicted, rule)

    @staticmethod
    def _prepare_image(image, source):
        """Normalize decoded RGB input to the model's uint8 CHW resolution."""
        image = np.array(image, copy=True)
        if image.ndim != 3 or image.dtype != np.uint8:
            raise ValueError(f"{source} must be uint8 RGB image, got {image.shape} {image.dtype}")
        if image.shape[-1] == 3:
            image = np.moveaxis(image, -1, 0)
        if image.shape[0] != 3:
            raise ValueError(f"{source} must have three RGB channels, got {image.shape}")
        if image.shape[1:] != (224, 224):
            tensor = torch.from_numpy(image).to(dtype=torch.float32).unsqueeze(0)
            tensor = torch_functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
            image = tensor.squeeze(0).round().clamp(0, 255).to(dtype=torch.uint8).numpy()
        return image

        # Select action keys from the arm count and action type
        num_arms = len(self.robot_action_dim_info["arm_dim"])

        if num_arms == 1:  # single arm
            arm_keys = ["arm_joint_state"] if self.action_type == "joint" else ["ee_pose"]
            ee_keys = ["ee_joint_state"]

        elif num_arms == 2:  # dual arm
            arm_keys = ["left_arm_joint_state", "right_arm_joint_state"] if self.action_type == "joint" else ["left_ee_pose", "right_ee_pose"]
            ee_keys = ["left_ee_joint_state", "right_ee_joint_state"]

        else:
            raise NotImplementedError(f"Unsupported number of arms: {num_arms}")

        steps = self.action_horizon
        action_list = []

        for _ in range(steps):
            action_dict = {}

            for i, (arm_key, ee_key) in enumerate(zip(arm_keys, ee_keys)):
                # Arm action
                # joint mode: dimensions are determined by arm_dim
                # ee mode: defaults to a 7-D pose [x, y, z, qw, qx, qy, qz]
                if self.action_type == "joint":
                    action_dict[arm_key] = np.zeros(
                        self.robot_action_dim_info["arm_dim"][i],
                        dtype=np.float32,
                    )
                else:
                    action_dict[arm_key] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        dtype=np.float32,
                    )

                # Gripper / end-effector joint action
                action_dict[ee_key] = np.zeros(
                    self.robot_action_dim_info["ee_dim"][i],
                    dtype=np.float32,
                )

            action_list.append(action_dict)

        return action_list

    @staticmethod
    def _format_pipeline_actions(predicted):
        return [{
            "left_arm_joint_state": row[:6].astype(np.float32),
            "left_ee_joint_state": row[6:7].astype(np.float32),
            "right_arm_joint_state": row[7:13].astype(np.float32),
            "right_ee_joint_state": row[13:14].astype(np.float32),
        } for row in predicted]

    def get_action_batch(self, env_idx_list=None):
        if self._obs_batch is None:
            batch_size = len(env_idx_list) if env_idx_list is not None else self.batch_size
            return [self.get_action() for _ in range(batch_size)]
        action_batch = []
        for obs in self._obs_batch:
            self._obs = obs
            action_batch.append(self.get_action())
        self._obs = None
        return action_batch

    def reset(self):
        # Reset model state here if it has internal state, such as an RNN hidden state
        self._obs = None
        self._obs_batch = None
        if self._pipe is not None:
            self._pipe.reset_observation_memory()
