"""AR inference policy for the Cosmos3 FastWAM Reference AR checkpoint.

Inference-only counterpart of the training repository's ``experiments/robotwin/fastwam_policy/deploy_policy.py``
(AR branch) and ``galaxea_va.runtime.create_cosmos3_reference_ar``. The model and processor are built from the
checkpoint's ``inference_config.yaml`` and ``dataset_entry.yaml`` instead of the training Hydra tree.

Observation/action timing (horizon 16, 4 actions per video frame)::

    obs(0) -> infer -> actions[0:16]; frames observed at steps 4, 8, 12, 16 feed the next inference.
"""
import inspect
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from galaxea_va.datasets.prompt_builder import build_text_prompt
from galaxea_va.inference.robot_processor import RobotProcessor
from galaxea_va.models.cosmos3.cosmos3_fastwam_reference_ar import Cosmos3FastWAMReferenceAR

logger = logging.getLogger(__name__)

# (width, height) of each camera in the [384, 320] model frame: head on top, wrists side by side below.
CAMERA_SIZES_WH = {"head_camera": (320, 256), "left_camera": (160, 128), "right_camera": (160, 128)}
MODEL_DTYPES = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(pil_image.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def build_model(model_cfg, *, model_dtype: torch.dtype, device: str) -> Cosmos3FastWAMReferenceAR:
    """Construct the published Cosmos3-Nano backbone with the checkpoint's AR head configuration."""
    max_attn_window_size = model_cfg.max_attn_window_size
    return Cosmos3FastWAMReferenceAR.from_pretrained(
        action_idm_prob=model_cfg.action_idm_prob,
        video_cond_noise_prob=model_cfg.video_cond_noise_prob,
        proprio_dim=int(model_cfg.proprio_dim),
        action_dim=int(model_cfg.action_dim),
        action_horizon=int(model_cfg.action_horizon),
        train_num_chunks=int(model_cfg.train_num_chunks),
        action_video_freq_ratio=int(model_cfg.action_video_freq_ratio),
        infer_sliding_window=int(model_cfg.infer_sliding_window),
        chunk_offset_range=int(model_cfg.chunk_offset_range),
        infer_reset_every=int(model_cfg.infer_reset_every),
        infer_reset_keep_chunks=int(model_cfg.infer_reset_keep_chunks),
        max_attn_window_size=None if max_attn_window_size is None else int(max_attn_window_size),
        random_attn_window_size=bool(model_cfg.random_attn_window_size),
        tokenizer_max_len=int(model_cfg.tokenizer_max_len),
        action_lr_mult=float(model_cfg.action_lr_mult),
        trainable_scope="generator",
        gradient_checkpointing=bool(model_cfg.gradient_checkpointing),
        video_fdm_prob=model_cfg.video_fdm_prob,
        compile_training_denoise=bool(model_cfg.compile_training_denoise),
        video_scheduler=model_cfg.video_scheduler,
        action_scheduler=model_cfg.action_scheduler,
        loss=model_cfg.loss,
        device=device,
        torch_dtype=model_dtype,
    )


class FastWAMARPolicy:
    def __init__(
        self,
        *,
        inference_config_path: str | Path,
        checkpoint_path: str | Path,
        device: str,
        action_infer_mode: str,
        replan_steps: int,
        num_action_denoise_steps: int,
        num_video_denoise_steps: int,
        sigma_shift: float | None,
        negative_prompt: str,
        rand_device: str,
        compile_action_infer: bool,
        timing_enabled: bool,
        infer_sliding_window: int,
        infer_reset_every: int,
        infer_reset_keep_chunks: int,
    ) -> None:
        cfg = OmegaConf.load(inference_config_path)
        cfg.model.infer_sliding_window = int(infer_sliding_window)
        cfg.model.infer_reset_every = int(infer_reset_every)
        cfg.model.infer_reset_keep_chunks = int(infer_reset_keep_chunks)
        self.model = build_model(cfg.model, model_dtype=MODEL_DTYPES[cfg.mixed_precision], device=device)
        self.model.load_checkpoint(str(checkpoint_path))
        self.model = self.model.to(device).eval()

        self.robot_processor = RobotProcessor(OmegaConf.to_container(cfg.processor))
        self.embodiment_type = str(cfg.embodiment_type)
        self.include_embodiment_in_prompt = bool(cfg.include_embodiment_in_prompt)
        self.include_control_type_in_prompt = bool(cfg.include_control_type_in_prompt)

        self.action_horizon = int(self.model.action_horizon)
        if int(cfg.action_chunk_size) != self.action_horizon:
            raise ValueError(f"action_chunk_size={cfg.action_chunk_size} != model.action_horizon={self.action_horizon}.")
        self.replan_steps = int(replan_steps)
        self.num_action_denoise_steps = int(num_action_denoise_steps)
        self.num_video_denoise_steps = int(num_video_denoise_steps)
        self.sigma_shift = sigma_shift
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.compile_action_infer = bool(compile_action_infer)
        self.timing_enabled = bool(timing_enabled)
        if action_infer_mode not in ("first_frame", "idm"):
            raise ValueError(f"action_infer_mode must be first_frame or idm, got {action_infer_mode!r}.")
        self.action_infer_mode = action_infer_mode

        action_video_freq_ratio = int(cfg.action_video_freq_ratio)
        if self.action_horizon % action_video_freq_ratio != 0:
            raise ValueError(f"action_horizon={self.action_horizon} is not divisible by "
                             f"action_video_freq_ratio={action_video_freq_ratio}.")
        self._num_video_frames = self.action_horizon // action_video_freq_ratio + 1
        self._single_frame_ar = self.model.infer_sliding_window == 0
        if not self._single_frame_ar and self.replan_steps != self.action_horizon:
            raise ValueError(f"AR evaluation requires replan_steps == action_horizon, "
                             f"got {self.replan_steps} and {self.action_horizon}.")
        new_video_frames = self._num_video_frames - 1
        action_obs_stride = self.action_horizon // new_video_frames
        token_aligned_steps = action_obs_stride * int(self.model.vae.temporal_downsample_factor)
        if self.action_horizon % token_aligned_steps != 0:
            raise ValueError(f"action_horizon={self.action_horizon} must be divisible by {token_aligned_steps} "
                             "to align with VAE temporal tokens.")
        self._action_obs_stride = self.replan_steps if self._single_frame_ar else action_obs_stride

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._episode_reference_image: torch.Tensor | None = None
        self._observation_frames: list[torch.Tensor] = []
        self._reset_ar_cache_on_infer = True
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        logger.info("Initialized FastWAMARPolicy | ckpt=%s | horizon=%d | replan=%d",
                    checkpoint_path, self.action_horizon, self.replan_steps)

    def _prompt(self, instruction: str) -> str:
        return build_text_prompt(instruction, embodiment_type=self.embodiment_type,
                                 include_embodiment=self.include_embodiment_in_prompt,
                                 include_control_type=self.include_control_type_in_prompt)

    def warmup_inference(self) -> None:
        """Warm the compiled inference paths before accepting simulator requests."""
        if not self.compile_action_infer:
            return
        image = self._build_image_tensor({
            "observation": {name: {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)} for name in CAMERA_SIZES_WH}
        })
        self.model.warmup_inference(
            video_size=tuple(image.shape[-2:]),
            prompt=self._prompt("Move the object."),
            num_video_frames=self._num_video_frames,
            num_action_denoise_steps=self.num_action_denoise_steps,
            num_video_denoise_steps=self.num_video_denoise_steps,
            action_infer_mode=self.action_infer_mode,
            sigma_shift=self.sigma_shift,
            rand_device=self.rand_device,
            proprio=torch.zeros(self.model.proprio_dim),
        )

    def _build_image_tensor(self, observation: dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head, left, right = (_resize_rgb(obs_data[name]["rgb"], CAMERA_SIZES_WH[name]) for name in CAMERA_SIZES_WH)
        image = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)  # [384, 320, 3]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.torch_dtype)
        return image_tensor * (2.0 / 255.0) - 1.0

    def _infer_action_chunk(self, observation: dict[str, Any], instruction: str, *,
                            image_tensor: torch.Tensor, observed_video: torch.Tensor) -> np.ndarray:
        """Infer one action chunk from the frames observed since the previous inference (one frame on cold start)."""
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio, action_dim_is_pad = self.robot_processor.preprocess_proprio(state_vector)
        if self._episode_reference_image is None:
            self._episode_reference_image = image_tensor.clone()
            logger.info("Captured episode reference image | episode=%d shape=%s",
                        self.episode_count, tuple(image_tensor.shape))
        infer_kwargs = {
            "prompt": self._prompt(instruction),
            "proprio": proprio,
            "action_dim_is_pad": action_dim_is_pad.unsqueeze(0),
            "negative_prompt": self.negative_prompt,
            "sigma_shift": self.sigma_shift,
            "rand_device": self.rand_device,
            "num_action_denoise_steps": self.num_action_denoise_steps,
            "num_video_denoise_steps": self.num_video_denoise_steps,
            "num_video_frames": self._num_video_frames,
            "action_infer_mode": self.action_infer_mode,
            "observed_video": observed_video,
            "reset_cache": self._reset_ar_cache_on_infer,
            "reference_image": self._episode_reference_image,
        }
        if self.compile_action_infer:
            infer_kwargs["compile_action_infer"] = True
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        self._reset_ar_cache_on_infer = self._single_frame_ar
        if self.timing_enabled:
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        return self.robot_processor.postprocess_action(pred["action"], proprio)

    def should_request_observation(self) -> bool:
        return self.step_count % self._action_obs_stride == 0

    def step(self, task_env, observation: dict[str, Any] | None) -> None:
        """Execute one simulator step, buffering only the frames new since the previous AR update."""
        observation_required = self.should_request_observation()
        if observation_required != (observation is not None):
            raise ValueError(f"Observation {'missing' if observation_required else 'unexpected'} at action step "
                             f"{self.step_count}; configured stride is {self._action_obs_stride}.")
        current_image = None
        if observation is not None:
            current_image = self._build_image_tensor(observation)
            self._observation_frames.append(current_image)

        if not self.pending_actions:
            if current_image is None:
                raise ValueError("AR replan step requires a current observation.")
            if self._reset_ar_cache_on_infer:
                observed_video = current_image.unsqueeze(2)
            else:
                expected_frames = self._num_video_frames - 1
                if len(self._observation_frames) != expected_frames:
                    raise ValueError(f"AR observation buffer must hold every frame since the previous inference: "
                                     f"expected {expected_frames}, got {len(self._observation_frames)}.")
                observed_video = torch.stack(self._observation_frames, dim=2)
            action_chunk = self._infer_action_chunk(observation, task_env.get_instruction(),
                                                    image_tensor=current_image, observed_video=observed_video)
            for action in action_chunk[: self.replan_steps]:
                self.pending_actions.append(np.asarray(action, dtype=np.float32))
            self._observation_frames = []

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def get_timing_rollout(self) -> dict[str, float]:
        return {key: float(value) for key, value in self._timing_rollout.items()}

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self._episode_reference_image = None
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        self._observation_frames.clear()
        self._reset_ar_cache_on_infer = True
        self.model.reset_ar_cache()
