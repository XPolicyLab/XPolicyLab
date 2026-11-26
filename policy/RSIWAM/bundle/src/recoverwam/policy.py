"""EE16 RoboDojo policy: T-shaped RGB, history8, cached-video Euler action inference."""

from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from .data import DEFAULT_PROMPT
from .checkpoint_compat import action_config
from .datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

ROBODOJO_TSHAPE_VIDEO_SIZE = (240, 416)
ROBODOJO_HORIZONTAL_VIDEO_SIZE = (240, 960)
ROBODOJO_HORIZONTAL_PER_CAMERA_SIZE = (240, 320)
DELTA_EEF14_REPRESENTATION = "unsupported_delta_eef"
DELTA_JOINT14_REPRESENTATION = "unsupported_delta_joint"


def _normalized_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """Keep one composed RGB frame compactly in the CPU history buffer."""
    frame = image.detach().float().cpu()
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise ValueError(f"Expected one image batch, got {tuple(image.shape)}")
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"Expected normalized [3,H,W], got {tuple(image.shape)}")
    return frame.clamp(-1, 1).add(1.0).mul(127.5).round().to(torch.uint8).contiguous()


def _uint8_to_normalized(frames: list[torch.Tensor]) -> torch.Tensor:
    """Stack compact CPU RGB frames into normalized ``[3,T,H,W]`` form."""
    if not frames:
        raise ValueError("Cannot build a history condition from no frames.")
    stacked = torch.stack(frames, dim=1).to(torch.float32)
    return stacked.div(255.0).mul(2.0).sub(1.0)


def _model_dtype(value: Any) -> torch.dtype:
    key = str(value or "bf16").strip().lower()
    if key == "bf16":
        return torch.bfloat16
    if key == "fp16":
        return torch.float16
    if key in {"no", "fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported mixed precision: {value!r}")


def _as_uint8_hwc(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"RoboDojo RGB view must be HWC, got {array.shape}")
    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] < 3:
        raise ValueError(f"RoboDojo RGB view needs three channels, got {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.5 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def compose_training_condition(
    head: Any,
    left_wrist: Any,
    right_wrist: Any,
    *,
    layout: str = "robodojo_tshape",
    video_size: tuple[int, int] = ROBODOJO_TSHAPE_VIDEO_SIZE,
    per_camera_video_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Reproduce the checkpoint's camera composition and normalization."""
    views = []
    for image in (head, left_wrist, right_wrist):
        tensor = torch.from_numpy(np.array(_as_uint8_hwc(image), copy=True))
        views.append(tensor.permute(2, 0, 1).to(torch.float32).div_(255.0))
    layout = str(layout).strip().lower()
    target_size = tuple((int(value) for value in video_size))
    if layout == "horizontal":
        camera_size = tuple(
            (int(value) for value in per_camera_video_size or ROBODOJO_HORIZONTAL_PER_CAMERA_SIZE)
        )
        resized_views = [
            transforms_F.resize(
                view,
                size=list(camera_size),
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            for view in views
        ]
        condition = torch.cat(resized_views, dim=2)
        if tuple(condition.shape[-2:]) != target_size:
            condition = transforms_F.resize(
                condition,
                size=list(target_size),
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
    elif layout == "robodojo_tshape":
        if target_size != ROBODOJO_TSHAPE_VIDEO_SIZE:
            raise ValueError(
                f"RoboDojo T-shape deployment requires video_size={ROBODOJO_TSHAPE_VIDEO_SIZE}, got {target_size}."
            )
        head_tensor = transforms_F.resize(
            views[0],
            size=[480, 640],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        left_tensor = transforms_F.resize(
            views[1],
            size=[240, 240],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        right_tensor = transforms_F.resize(
            views[2],
            size=[240, 240],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        raw_layout = torch.cat([head_tensor, torch.cat([left_tensor, right_tensor], dim=1)], dim=2)
        condition = transforms_F.resize(
            raw_layout,
            size=list(target_size),
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
    else:
        raise ValueError(f"Unsupported RoboDojo camera layout: {layout!r}.")
    return condition.mul(2.0).sub(1.0).unsqueeze(0).contiguous()


class RoboDojoRecoverWAMPolicy:
    def __init__(
        self,
        checkpoint_path,
        wan_model_dir,
        dataset_stats_path=None,
        video_checkpoint_path=None,
        device="cuda",
        mixed_precision="bf16",
        num_inference_steps=10,
        sigma_shift=1.0,
        seed=1,
        replan_steps=24,
    ):
        checkpoint_path = Path(checkpoint_path).resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        cfg = OmegaConf.create(action_config(payload))
        data = cfg.data.train
        if (
            int(cfg.model.action_dit_config.hidden_dim),
            int(data.processor.action_output_dim),
            int(data.processor.proprio_output_dim),
        ) != (1024, 16, 16):
            raise ValueError("Expected Action2B / EE16 architecture")
        if list(data.video_size) != [240, 416] or data.video_resize_mode != "direct":
            raise ValueError("Expected direct 240x416 T-shape preprocessing")
        if (data.condition_history_frames, data.condition_history_stride, data.action_horizon) != (
            8,
            25,
            32,
        ):
            raise ValueError("Expected history8/stride25/horizon32")
        if not data.condition_include_episode_first or not data.condition_history_target_prefix:
            raise ValueError("Episode anchor and history prefix are required")
        video_path = Path(video_checkpoint_path or payload["video_checkpoint"]["path"])
        if not video_path.is_absolute():
            video_path = checkpoint_path.parent / video_path
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        video_meta = torch.load(video_path, map_location="cpu", mmap=True, weights_only=True)
        if video_meta.get("step") != payload.get("step") or "dit" not in video_meta:
            raise ValueError("Video/Action step or format mismatch")
        del video_meta
        model_cfg = cfg.model
        model_cfg._target_ = "recoverwam.runtime.create_recoverwam_wan21_14b"
        model_cfg.model_id = str(Path(wan_model_dir).resolve())
        model_cfg.tokenizer_model_id = str(Path(model_cfg.model_id) / "google/umt5-xxl")
        model_cfg.dit_checkpoint_path = str(video_path.resolve())
        model_cfg.action_dit_pretrained_path = None
        model_cfg.load_text_encoder = True
        model_cfg.video_dit_config.use_gradient_checkpointing = False
        model_cfg.action_dit_config.use_gradient_checkpointing = False
        model = instantiate(model_cfg, model_dtype=_model_dtype(mixed_precision), device=device)
        model.action_expert.load_state_dict(payload["action_expert"], strict=True)
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        self.model = model.eval()
        self.processor = instantiate(data.processor).eval()
        stats = (
            Path(dataset_stats_path)
            if dataset_stats_path
            else checkpoint_path.parent / "dataset_stats.json"
        )
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
        self.action_dim = self.proprio_dim = self.output_action_dim = 16
        self.action_representation = "absolute_dual_arm_ee_pose_qwxyz_plus_gripper"
        self.action_horizon = 32
        self.replan_steps = int(replan_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = float(sigma_shift)
        if (
            not 1 <= self.replan_steps <= 32
            or self.num_inference_steps < 1
            or self.sigma_shift <= 0
        ):
            raise ValueError("Invalid inference horizon/steps/shift")
        self.base_seed = int(seed)
        self.video_seed_offset = 1000003
        self.text_cfg_scale = 1.0
        self.negative_prompt = ""
        self.rand_device = "cpu"
        self.tiled = False
        self.video_size = (240, 416)
        self.visual_layout = "robodojo_tshape"
        self.per_camera_video_size = None
        self.num_video_frames = 17
        self.history_condition_frames = 8
        self.history_stride = 25
        self.history_include_episode_first = True
        self.history_buffer_size = 177
        self.visual_planning_output_root = None
        self.visual_planning_full_steps = 10
        self._history_by_env = {}
        self._history_anchor_by_env = {}
        self._history_step_by_env = {}
        self._latest_image_by_env = {}
        self._inference_index_by_env = {}
        self._inference_index = 0
        self._request_context = {}

    _PER_ENV_STATE = (
        "_history_by_env",
        "_history_anchor_by_env",
        "_history_step_by_env",
        "_latest_image_by_env",
        "_inference_index_by_env",
    )

    def set_request_context(self, context: Dict[str, Any]) -> None:
        self._request_context = dict(context or {})

    def _compose_observation_image(self, observation: Dict[str, Any]) -> torch.Tensor:
        images = observation["observation"]
        return compose_training_condition(
            images["head_camera"]["rgb"],
            images["left_camera"]["rgb"],
            images["right_camera"]["rgb"],
            layout=self.visual_layout,
            video_size=self.video_size,
            per_camera_video_size=self.per_camera_video_size,
        ).to(device=self.model.device, dtype=self.model.torch_dtype)

    def observe(self, observation: Dict[str, Any]) -> None:
        """Consume one native simulator observation for history conditioning.

        The official RoboDojo client calls ``update_obs`` for every action
        inside a returned chunk.  Recording here, rather than only when a new
        chunk is requested, preserves the native 25-Hz history timeline.
        """
        env_idx = int(observation.get("env_idx", 0))
        image = self._compose_observation_image(observation)
        self._latest_image_by_env[env_idx] = image
        if self.history_condition_frames > 0:
            self._record_history_frame(image, env_idx)

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.output_action_dim,):
            raise ValueError(
                f"Expected RoboDojo state [{self.output_action_dim}], got {state.shape}."
            )
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected one merged state key.")
        key = state_meta[0]["key"]
        batch = {"state": {key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        key = self.processor.shape_meta["action"][0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        return normalizer.backward(action.float().cpu()).numpy()

    def _record_history_frame(self, image: torch.Tensor, env_idx: int) -> None:
        """Append one simulator observation to the per-environment history."""
        env_idx = int(env_idx)
        history = self._history_by_env.get(env_idx)
        if history is None:
            history = deque(maxlen=self.history_buffer_size)
            self._history_by_env[env_idx] = history
            step = 0
            self._history_anchor_by_env[env_idx] = _normalized_to_uint8(image)
        else:
            step = int(self._history_step_by_env[env_idx]) + 1
        frame = _normalized_to_uint8(image)
        history.append((step, frame))
        self._history_step_by_env[env_idx] = step

    def _build_history_condition(
        self, image: torch.Tensor, env_idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[dict[str, Any]]]:
        """Build the exact anchor+history RGB condition expected by aligned history."""
        if self.history_condition_frames <= 0:
            return (None, None)
        env_idx = int(env_idx)
        history = self._history_by_env.get(env_idx)
        if not history:
            self._record_history_frame(image, env_idx)
            history = self._history_by_env[env_idx]
        current_step = int(self._history_step_by_env[env_idx])
        anchor = self._history_anchor_by_env.get(env_idx)
        if anchor is None:
            anchor = history[0][1]
            self._history_anchor_by_env[env_idx] = anchor
        requested_offsets = [
            -(self.history_condition_frames - 1 - index) * self.history_stride
            for index in range(self.history_condition_frames)
        ]
        step_to_frame = {int(step): frame for (step, frame) in history}
        selected_frames: list[torch.Tensor] = []
        selected_steps: list[int] = []
        for offset in requested_offsets:
            requested_step = current_step + int(offset)
            if requested_step <= 0:
                selected_step = 0
                selected_frame = anchor
            elif requested_step in step_to_frame:
                selected_step = requested_step
                selected_frame = step_to_frame[requested_step]
            else:
                prior = [
                    (int(step), frame) for (step, frame) in history if int(step) <= requested_step
                ]
                if prior:
                    (selected_step, selected_frame) = prior[-1]
                else:
                    (selected_step, selected_frame) = (0, anchor)
            selected_steps.append(int(selected_step))
            selected_frames.append(selected_frame)
        condition_frames = [anchor, *selected_frames]
        condition = _uint8_to_normalized(condition_frames).unsqueeze(0)
        condition = condition.to(device=image.device, dtype=image.dtype)
        metadata = {
            "enabled": True,
            "env_idx": env_idx,
            "current_step": current_step,
            "requested_offsets": requested_offsets,
            "selected_steps": [0, *selected_steps],
            "rgb_frames": len(condition_frames),
            "latent_frames": (len(condition_frames) - 1) // 4 + 1,
            "stride_native_steps": self.history_stride,
        }
        return (condition.contiguous(), metadata)

    def _seeds(self, env_idx: int = 0) -> tuple[Optional[int], Optional[int]]:
        env_idx = int(env_idx)
        replan_index = int(self._inference_index_by_env.get(env_idx, 0))
        self._inference_index_by_env[env_idx] = replan_index + 1
        self._inference_index += 1
        if self.base_seed is None:
            return (None, None)
        action_seed = int(self.base_seed + replan_index)
        video_seed = int(action_seed + self.video_seed_offset)
        return (action_seed, video_seed)

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        env_idx = int(observation.get("env_idx", 0))
        image = self._latest_image_by_env.get(env_idx)
        if image is None:
            image = self._compose_observation_image(observation)
            if self.history_condition_frames > 0:
                self._record_history_frame(image, env_idx)
        history_condition = None
        history_metadata = None
        if self.history_condition_frames > 0:
            (history_condition, history_metadata) = self._build_history_condition(
                image=image, env_idx=env_idx
            )
        delta_eef_reference = state
        proprio = self._normalize_state(delta_eef_reference)
        (action_seed, video_seed) = self._seeds(env_idx)
        capture_visual_plan = self.visual_planning_output_root is not None
        infer_kwargs = {
            "prompt": DEFAULT_PROMPT.format(task=instruction),
            "input_image": image,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": action_seed,
            "video_seed": video_seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
            "num_video_frames": self.num_video_frames,
            "return_one_step_video": capture_visual_plan,
            "return_full_video": capture_visual_plan,
            "full_video_num_inference_steps": self.visual_planning_full_steps
            if capture_visual_plan
            else None,
        }
        if history_condition is not None:
            infer_kwargs["condition_video"] = history_condition
        with torch.no_grad():
            prediction = self.model.infer_action(**infer_kwargs)
        action = self._denormalize_action(prediction["action"])[0]
        if action.ndim != 2 or action.shape[1] != self.output_action_dim:
            raise ValueError(
                f"Model returned an invalid RoboDojo action chunk: expected [T,{self.output_action_dim}], got {action.shape}."
            )
        return action

    def reset(self) -> None:
        self._inference_index = 0
        for name in self._PER_ENV_STATE:
            state = getattr(self, name, None)
            if state is not None:
                state.clear()

    def reset_env(self, env_idx: int) -> None:
        """Clear the episode state of one environment only."""
        for name in self._PER_ENV_STATE:
            state = getattr(self, name, None)
            if state is not None:
                state.pop(int(env_idx), None)


def get_model(args):
    policy_type = RoboDojoRecoverWAMPolicy
    extra = {}
    if args.get("recovery_adapter"):
        from .recovery.policy import RecoverWAMPolicy
        policy_type = RecoverWAMPolicy
        extra = dict(recovery_adapter=args["recovery_adapter"], base_identity=args.get("base_identity"))
    return policy_type(
        checkpoint_path=args.get("checkpoint_path") or args.get("ckpt_setting"),
        wan_model_dir=args["wan_model_dir"],
        dataset_stats_path=args.get("dataset_stats_path"),
        video_checkpoint_path=args.get("video_checkpoint_path"),
        device=args.get("device") or "cuda",
        mixed_precision=args.get("mixed_precision") or "bf16",
        num_inference_steps=args.get("num_inference_steps") or 10,
        sigma_shift=args.get("sigma_shift") or 1,
        seed=args.get("seed", 1),
        replan_steps=args.get("replan_steps") or 24,
        **extra,
    )
