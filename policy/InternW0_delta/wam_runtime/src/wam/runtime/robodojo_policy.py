import logging
import hashlib
import os
import sys
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wam.runtime.checkpoint import load_wam_checkpoint
from wam.runtime.preprocessing import (
    DEFAULT_PROMPT,
    RuntimeProcessor,
    build_runtime_processor,
    load_dataset_stats_from_json,
)
from wam.inference.online_action_policy import infer_online_action_chunk
from wam.memory_history import resolve_recent_history_index
from wam.model.modules.conditioning.video_rope import (
    sequence_start_video_rope_time_ids,
)

logger = logging.getLogger(__name__)


def _profile_section(profiler: Any, name: str):
    return profiler.section(name) if profiler is not None else nullcontext()


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _normalize_device(device: str) -> str:
    normalized = str(device).strip()
    # RoboTwin restricts each worker with CUDA_VISIBLE_DEVICES. Some model
    # components call torch.cuda.set_device, which requires an explicit index.
    if normalized.lower() == "cuda":
        return "cuda:0"
    return normalized


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _static_dimension_is_pad(processor: RuntimeProcessor, field: str) -> torch.Tensor:
    """Derive a fixed canonical-dimension padding mask from the merger."""

    if field not in {"action", "state"}:
        raise ValueError(f"Unsupported merged field: {field}")
    merger = processor.action_state_merger
    meta = processor.shape_meta[field]
    source_dim = sum(int(item["shape"]) for item in meta)
    target_dim = getattr(merger, f"{field}_target_dim")
    target_dim = source_dim if target_dim is None else int(target_dim)
    target_slices = getattr(merger, f"{field}_target_slices")
    mask = torch.ones(target_dim, dtype=torch.bool)
    if target_slices is None:
        mask[:source_dim] = False
    else:
        for entry in target_slices:
            start, end = (int(value) for value in entry["target_slice"])
            mask[start:end] = False
    return mask


@dataclass(frozen=True)
class RobotWinSharedRuntime:
    """Objects loaded once and shared by all RoboTwin policy sessions."""

    model: Any
    processor: RuntimeProcessor
    checkpoint_path: str
    dataset_stats_path: Path
    device: str
    model_dtype: torch.dtype
    action_dim_is_pad: torch.Tensor


@dataclass(frozen=True)
class RobotWinSessionConfig:
    """Immutable inference settings copied into each policy session."""

    action_horizon: int
    action_hz: float
    replan_steps: int
    num_inference_steps: int
    sigma_shift: Optional[float]
    seed: Optional[int]
    text_cfg_scale: float
    negative_prompt: str
    rand_device: str
    tiled: bool
    timing_enabled: bool
    num_video_frames: int
    video_size: tuple[int, int]
    video_layout: str
    video_view_names: str


def _resolve_training_video_contract(train_cfg: Any) -> tuple[str, str]:
    """Resolve the model-facing VAE layout exactly as RobotVideoDataset does."""

    single_canvas = bool(train_cfg.get("single_canvas", False))
    canvas_layout = str(
        train_cfg.get("concat_multi_camera", "single") or "single"
    )
    if single_canvas:
        return "single", "canvas"
    shape_meta = train_cfg.get("shape_meta", {})
    images = shape_meta.get("images", [])
    view_names = "|".join(
        str(item.get("key", f"view_{idx}"))
        for idx, item in enumerate(images)
    )
    return canvas_layout, view_names


def build_robotwin_shared_runtime(
    *,
    model_cfg: DictConfig,
    processor_cfg: DictConfig,
    checkpoint_path: str,
    dataset_stats_path: Path,
    device: str,
    model_dtype: torch.dtype,
    profiler: Any = None,
) -> RobotWinSharedRuntime:
    """Load the model and processor exactly once for a server process."""

    with _profile_section(profiler, "model_config_prepare"):
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True
        os.environ.setdefault(
            "DIFFSYNTH_MODEL_BASE_PATH",
            str((PROJECT_ROOT / "checkpoints").resolve()),
        )

    with _profile_section(profiler, "model_instantiate"):
        model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
    with _profile_section(profiler, "model_runtime_config"):
        if os.environ.get("WAM_DISABLE_CAUSAL_CONV1D_FAST_PATH", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            disabled_count = 0
            for module in model.modules():
                if getattr(module, "causal_conv1d_fn", None) is not None:
                    module.causal_conv1d_fn = None
                    disabled_count += 1
            logger.warning(
                "Disabled causal-conv1d fast path in %d modules; using PyTorch fallback.",
                disabled_count,
            )
    with _profile_section(profiler, "checkpoint_load"):
        load_wam_checkpoint(model, checkpoint_path)
    with _profile_section(profiler, "model_to_device"):
        model = model.to(device).eval()

    with _profile_section(profiler, "processor_instantiate"):
        processor = build_runtime_processor(processor_cfg)
    with _profile_section(profiler, "dataset_stats_load"):
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        processor.set_normalizer_from_stats(dataset_stats)

    action_dim_is_pad = _static_dimension_is_pad(processor, "action")
    model_action_dim = int(
        getattr(getattr(model, "action_expert", None), "action_dim", 0)
    )
    if model_action_dim <= 0:
        raise ValueError("model.action_expert.action_dim is unavailable.")
    if int(action_dim_is_pad.numel()) != model_action_dim:
        raise ValueError(
            "RoboTwin processor/model action dimension mismatch: "
            f"processor={action_dim_is_pad.numel()}, model={model_action_dim}."
        )
    if bool(action_dim_is_pad.all().item()):
        raise ValueError("RoboTwin action merger marks every action dimension invalid.")

    return RobotWinSharedRuntime(
        model=model,
        processor=processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=Path(dataset_stats_path),
        device=str(device),
        model_dtype=model_dtype,
        action_dim_is_pad=action_dim_is_pad,
    )


class RobotWinPolicySession:
    """All mutable state for one RoboTwin simulator connection."""

    def __init__(
        self,
        runtime: RobotWinSharedRuntime,
        config: RobotWinSessionConfig,
        profiler: Any = None,
    ) -> None:
        self.runtime = runtime
        self.profiler = profiler
        self.model = runtime.model
        self.processor = runtime.processor
        self.action_dim_is_pad = runtime.action_dim_is_pad

        self.action_horizon = int(config.action_horizon)
        self.action_hz = float(config.action_hz)
        if not np.isfinite(self.action_hz) or self.action_hz <= 0.0:
            raise ValueError(
                f"action_hz must be finite and positive, got {config.action_hz}."
            )
        self.replan_steps = int(config.replan_steps)
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")
        if not 1 <= self.replan_steps <= self.action_horizon:
            raise ValueError(
                "replan_steps must satisfy 1 <= replan_steps <= action_horizon; "
                f"got {self.replan_steps} and {self.action_horizon}"
            )
        self.num_inference_steps = int(config.num_inference_steps)
        self.sigma_shift = config.sigma_shift
        self.seed = config.seed
        self.text_cfg_scale = float(config.text_cfg_scale)
        self.negative_prompt = str(config.negative_prompt)
        self.rand_device = str(config.rand_device)
        self.tiled = bool(config.tiled)
        self.timing_enabled = bool(config.timing_enabled)
        self._num_video_frames = int(config.num_video_frames)
        self.video_height, self.video_width = (int(value) for value in config.video_size)
        if self.video_height <= 0 or self.video_width <= 0:
            raise ValueError(f"Invalid RoboTwin video_size: {config.video_size}")
        self.video_layout = str(config.video_layout)
        self.video_view_names = str(config.video_view_names)

        self.pending_actions: deque[np.ndarray] = deque()
        self.pending_model_actions: deque[torch.Tensor] = deque()
        self.action_history: deque[torch.Tensor] = deque()
        self.use_ar_action_history = bool(getattr(self.model, "ar_history_enabled", False)) and bool(
            getattr(self.model, "use_ar_action_history", False)
        )
        self.use_memory = bool(
            int(getattr(self.model, "memory_video_anchor_frames", 0) or 0) > 0
            or int(getattr(self.model, "memory_video_recent_frames", 0) or 0) > 0
        )
        self.ar_max_history_actions = int(getattr(self.model, "max_history_actions", 0) or 0)
        self.ar_effective_history_actions = self.ar_max_history_actions
        if self.use_ar_action_history:
            ar_num_chunks = int(getattr(self.model, "num_history_chunks", 0) or 0)
            if ar_num_chunks > 0:
                self.ar_effective_history_actions = max(
                    1,
                    min(self.ar_max_history_actions, ar_num_chunks * self.replan_steps),
                )
            if self.ar_max_history_actions <= 0:
                raise ValueError("AR action history is enabled, but model.max_history_actions is not positive.")
        action_dim = int(getattr(getattr(self.model, "action_expert", None), "action_dim", 0))
        if (self.use_ar_action_history or self.use_memory) and action_dim <= 0:
            raise ValueError("Action history is enabled, but model.action_expert.action_dim is unavailable.")

        self.action_video_freq_ratio = max(1, self.action_horizon // max(1, self._num_video_frames - 1))
        self.memory_video_anchor_frames = int(getattr(self.model, "memory_video_anchor_frames", 0) or 0)
        self.memory_video_recent_frames = int(getattr(self.model, "memory_video_recent_frames", 0) or 0)
        self.memory_video_anchor_raw_steps = self.memory_video_anchor_frames if self.memory_video_anchor_frames > 0 else 0
        self.visual_history: list[torch.Tensor] = []
        self.proprio_history: list[torch.Tensor] = []
        self._last_visual_history_step: Optional[int] = None
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout: Dict[str, float | int] = {}
        self._action_chunk_hashes: list[str] = []
        self.reset_timing_rollout()

        logger.info(
            "Initialized RobotWinPolicySession | ckpt=%s | stats=%s | "
            "horizon=%d | replan=%d | video_size=%dx%d | layout=%s | views=%s",
            runtime.checkpoint_path,
            runtime.dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.video_height,
            self.video_width,
            self.video_layout,
            self.video_view_names,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        state_batch = self.processor.action_state_merger.forward(state_batch)
        return state_batch["state"]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action = action.to(dtype=torch.float32, device="cpu")
        batch = {"action": action, "state": action.new_zeros(action.shape)}
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        return batch["action"][action_meta[0]["key"]].numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        top_height = (self.video_height * 2) // 3
        bottom_height = self.video_height - top_height
        left_width = self.video_width // 2
        right_width = self.video_width - left_width
        head = _resize_rgb(
            obs_data["head_camera"]["rgb"],
            (self.video_width, top_height),
        )
        left = _resize_rgb(
            obs_data["left_camera"]["rgb"],
            (left_width, bottom_height),
        )
        right = _resize_rgb(
            obs_data["right_camera"]["rgb"],
            (right_width, bottom_height),
        )
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)
        if image.shape[:2] != (self.video_height, self.video_width):
            raise RuntimeError(
                "RoboTwin camera canvas shape mismatch: "
                f"got {image.shape[:2]}, expected {(self.video_height, self.video_width)}"
            )

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _build_robotwin_vlm_images(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        return torch.stack(
            [
                torch.from_numpy(np.ascontiguousarray(obs_data[key]["rgb"])).permute(2, 0, 1)
                for key in ("head_camera", "left_camera", "right_camera")
            ],
            dim=0,
        ).to(device=self.model.device)

    def _build_action_history_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        action_dim = int(getattr(getattr(self.model, "action_expert", None), "action_dim", 0))
        if action_dim <= 0:
            raise ValueError("AR action history is enabled, but model.action_expert.action_dim is unavailable.")
        target_len = int(self.ar_effective_history_actions)
        actions = torch.zeros((target_len, action_dim), dtype=torch.float32)
        is_pad = torch.ones((target_len,), dtype=torch.bool)
        take = list(self.action_history)[-target_len:]
        if take:
            stacked = torch.stack([x.to(dtype=torch.float32, device="cpu") for x in take], dim=0)
            if stacked.shape[-1] != action_dim:
                raise ValueError(
                    f"AR action history dim mismatch: got {stacked.shape[-1]}, expected {action_dim}"
                )
            actions[-stacked.shape[0] :] = stacked
            is_pad[-stacked.shape[0] :] = False
        return actions, is_pad

    @staticmethod
    def _trim_history(history: list[torch.Tensor], *, anchor_len: int, recent_len: int) -> list[torch.Tensor]:
        anchor_len = max(0, int(anchor_len))
        recent_len = max(0, int(recent_len))
        if len(history) <= anchor_len + recent_len:
            return history
        return history[:anchor_len] + history[-recent_len:]

    def _record_memory_observation(self, observation: Dict[str, Any]) -> None:
        if not self.use_memory:
            return
        if self._last_visual_history_step == self.step_count:
            return
        frame = self._build_robotwin_image_tensor(observation)[0].detach().to(device="cpu", dtype=torch.float32)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)[0].detach().to(device="cpu", dtype=torch.float32)
        if self._last_visual_history_step is not None:
            missing_steps = self.step_count - self._last_visual_history_step - 1
            if missing_steps < 0:
                raise RuntimeError(
                    "RobotWin memory step moved backwards: "
                    f"last={self._last_visual_history_step}, current={self.step_count}"
                )
            if missing_steps and self.visual_history and self.proprio_history:
                # Preserve absolute step indexing without paying for unused
                # camera reads. Only exact replan/current and 32-step-recent
                # frames are consumed; intermediate slots may repeat the last
                # real frame safely.
                self.visual_history.extend([self.visual_history[-1]] * missing_steps)
                self.proprio_history.extend([self.proprio_history[-1]] * missing_steps)
        self.visual_history.append(frame)
        self.proprio_history.append(proprio)

        keep_recent = self.action_horizon + 1
        self.visual_history = self._trim_history(
            self.visual_history,
            anchor_len=self.memory_video_anchor_raw_steps,
            recent_len=keep_recent,
        )
        self.proprio_history = self._trim_history(
            self.proprio_history,
            anchor_len=self.memory_video_anchor_raw_steps,
            recent_len=keep_recent,
        )
        self._last_visual_history_step = self.step_count
        self._timing_rollout["memory_observation_calls"] += 1

    def _build_memory_video_tensor(
        self,
        history: list[torch.Tensor],
        *,
        target_len: int,
        height: int,
        width: int,
        from_start: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_len = int(target_len)
        frames = torch.zeros((3, target_len, height, width), dtype=torch.float32)
        is_pad = torch.ones((target_len,), dtype=torch.bool)
        if target_len <= 0:
            return frames, is_pad
        take = history[:target_len] if from_start else history[-target_len:]
        if not take:
            return frames, is_pad
        stacked = torch.stack([x.to(dtype=torch.float32, device="cpu") for x in take], dim=0)
        if stacked.shape[1:] != (3, height, width):
            raise ValueError(
                "Memory video shape mismatch: "
                f"got {tuple(stacked.shape[1:])}, expected {(3, height, width)}"
            )
        if from_start:
            frames[:, : len(take)] = stacked.permute(1, 0, 2, 3).contiguous()
            is_pad[: len(take)] = False
        else:
            frames[:, -len(take) :] = stacked.permute(1, 0, 2, 3).contiguous()
            is_pad[-len(take) :] = False
        return frames, is_pad

    def _build_memory_inputs(
        self,
        *,
        action_horizon: int,
        action_dim: int,
        height: int,
        width: int,
        memory_chunk_index: int,
        current_memory_chunks: Optional[int] = None,
        history_video_frames: Optional[int] = None,
        include_current_recent: bool = False,
    ) -> Optional[dict[str, torch.Tensor]]:
        if not self.use_memory:
            return None
        del (
            action_horizon,
            action_dim,
            current_memory_chunks,
            history_video_frames,
        )
        memory_chunk_index = max(0, int(memory_chunk_index))
        memory_inputs: dict[str, torch.Tensor] = {}

        if self.memory_video_anchor_frames > 0:
            sampled_anchor: list[torch.Tensor] = []
            sampled_anchor_proprio: list[torch.Tensor] = []
            for anchor_idx in range(self.memory_video_anchor_frames):
                raw_idx = anchor_idx
                if raw_idx < len(self.visual_history):
                    sampled_anchor.append(self.visual_history[raw_idx])
                    if raw_idx < len(self.proprio_history):
                        sampled_anchor_proprio.append(self.proprio_history[raw_idx])
                elif sampled_anchor:
                    sampled_anchor.append(sampled_anchor[-1])
                    if sampled_anchor_proprio:
                        sampled_anchor_proprio.append(sampled_anchor_proprio[-1])
                elif self.visual_history:
                    sampled_anchor.append(self.visual_history[0])
                    if self.proprio_history:
                        sampled_anchor_proprio.append(self.proprio_history[0])
            frames, is_pad = self._build_memory_video_tensor(
                sampled_anchor,
                target_len=self.memory_video_anchor_frames,
                height=height,
                width=width,
                from_start=True,
            )
            memory_inputs["memory_video_anchor"] = frames
            memory_inputs["memory_video_anchor_is_pad"] = is_pad
            memory_inputs["memory_video_anchor_frame_ids"] = (
                sequence_start_video_rope_time_ids(
                    1,
                    self.memory_video_anchor_frames,
                    device=torch.device("cpu"),
                )[0]
            )
            if sampled_anchor_proprio:
                memory_inputs["memory_video_anchor_proprio"] = torch.stack(
                    sampled_anchor_proprio, dim=0
                )
                memory_inputs["memory_video_anchor_proprio_is_pad"] = torch.zeros(
                    (len(sampled_anchor_proprio),), dtype=torch.bool
                )

        if (
            include_current_recent
            and self.memory_video_recent_frames > 0
            and self.visual_history
            and "memory_video_recent" not in memory_inputs
        ):
            recent_video_frames = int(self.memory_video_recent_frames)
            recent_layout = [
                resolve_recent_history_index(
                    len(self.visual_history),
                    self.action_horizon
                    + (recent_video_frames - 1 - slot)
                    * self.action_video_freq_ratio,
                    episode_step=self.step_count,
                )
                for slot in range(recent_video_frames)
            ]
            recent_sampled_history = [
                self.visual_history[index] for index, _ in recent_layout
            ]
            recent_sampled_proprio = [
                self.proprio_history[index]
                for index, _ in recent_layout
                if self.proprio_history
            ]
            recent_frames, recent_is_pad = self._build_memory_video_tensor(
                recent_sampled_history,
                target_len=recent_video_frames,
                height=height,
                width=width,
                from_start=False,
            )
            recent_is_real = torch.tensor(
                [is_real for _, is_real in recent_layout], dtype=torch.bool
            )
            recent_is_pad = recent_is_pad | ~recent_is_real
            memory_inputs["memory_video_recent"] = recent_frames
            memory_inputs["memory_video_recent_is_pad"] = recent_is_pad
            latent_frames_per_chunk = int(
                getattr(self.model, "memory_video_latents_per_chunk", 1) or 1
            )
            vae_temporal_factor = int(
                getattr(self.model, "memory_video_vae_temporal_factor", 1) or 1
            )
            recent_latent_start = (
                max(0, memory_chunk_index - 1) * latent_frames_per_chunk
            )
            memory_inputs["memory_video_recent_frame_ids"] = torch.arange(
                recent_latent_start,
                recent_latent_start + latent_frames_per_chunk,
                dtype=torch.long,
            ).repeat_interleave(vae_temporal_factor)
            if recent_sampled_proprio:
                memory_inputs["memory_video_recent_proprio"] = torch.stack(
                    recent_sampled_proprio, dim=0
                )
                memory_inputs["memory_video_recent_proprio_is_pad"] = (
                    ~recent_is_real
                )

        return memory_inputs


    def _prepare_memory_for_replan(
        self,
        *,
        image_tensor: torch.Tensor,
    ) -> Optional[dict[str, torch.Tensor]]:
        if not self.use_memory:
            return None
        _, _, height, width = image_tensor.shape
        action_dim = int(getattr(getattr(self.model, "action_expert", None), "action_dim", 0))
        memory_inputs = self._build_memory_inputs(
            action_horizon=self.action_horizon,
            action_dim=action_dim,
            height=height,
            width=width,
            memory_chunk_index=self.step_count // self.replan_steps,
            current_memory_chunks=self.step_count // self.replan_steps,
            include_current_recent=True,
        )
        return memory_inputs


    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> tuple[np.ndarray, torch.Tensor]:
        if not self.use_memory:
            raise ValueError("RobotWin deploy now requires memory AR inference.")
        with _profile_section(self.profiler, "image_canvas_prepare"):
            image_tensor = self._build_robotwin_image_tensor(observation)
        with _profile_section(self.profiler, "vlm_images_prepare"):
            vlm_current_images = self._build_robotwin_vlm_images(observation)
        with _profile_section(self.profiler, "current_frame_to_cpu"):
            current_frame = image_tensor[0].detach().to(device="cpu", dtype=torch.float32)
        with _profile_section(self.profiler, "state_preprocess"):
            state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
            proprio = self._normalize_state(state_vector)
            current_proprio = proprio[0].detach().to(device="cpu", dtype=torch.float32)
        with _profile_section(self.profiler, "initial_memory_record"):
            if not self.visual_history:
                self.visual_history.append(current_frame)
                self.proprio_history.append(current_proprio)
                self._last_visual_history_step = self.step_count
        with _profile_section(self.profiler, "memory_inputs_prepare"):
            prompt = DEFAULT_PROMPT.format(task=instruction)
            memory_inputs = self._prepare_memory_for_replan(image_tensor=image_tensor)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with _profile_section(self.profiler, "online_inference"):
            with torch.no_grad():
                pred = infer_online_action_chunk(
                    self.model,
                    prompt=prompt,
                    input_image=image_tensor,
                    vlm_current_images=vlm_current_images,
                    vlm_view_names="cam_high|cam_left_wrist|cam_right_wrist",
                    action_horizon=self.action_horizon,
                    action_hz=self.action_hz,
                    proprio=proprio,
                    action_dim_is_pad=self.action_dim_is_pad,
                    memory_inputs=memory_inputs,
                    num_inference_steps=self.num_inference_steps,
                    sigma_shift=self.sigma_shift,
                    seed=self.seed,
                    rand_device=self.rand_device,
                    tiled=self.tiled,
                    memory_chunk_index=self.step_count // self.replan_steps,
                    video_layout=self.video_layout,
                    video_view_names=self.video_view_names,
                )
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
        with _profile_section(self.profiler, "action_denormalize"):
            action_tensor = pred["action"]
            action_chunk = self._denormalize_action(action_tensor)[0]
        with _profile_section(self.profiler, "model_action_to_cpu"):
            model_action_chunk = action_tensor.detach().to(device="cpu", dtype=torch.float32)
            if model_action_chunk.ndim == 3:
                if model_action_chunk.shape[0] != 1:
                    raise ValueError(
                        "RobotWin inference expects batch size 1, got action shape "
                        f"{tuple(model_action_chunk.shape)}"
                    )
                model_action_chunk = model_action_chunk[0]
            if model_action_chunk.ndim != 2:
                raise ValueError(
                    "Expected per-step model action chunk [T,D], got "
                    f"{tuple(model_action_chunk.shape)}"
                )
        self._timing_rollout["replan_calls"] += 1
        self._timing_rollout["predicted_action_steps"] += int(action_chunk.shape[0])
        self._action_chunk_hashes.append(
            hashlib.sha256(
                np.ascontiguousarray(action_chunk, dtype=np.float32).tobytes()
            ).hexdigest()[:16]
        )
        return action_chunk, model_action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk, model_action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        self._timing_rollout["scheduled_action_steps"] += int(n_exec)
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))
            if self.use_ar_action_history or self.use_memory:
                self.pending_model_actions.append(model_action_chunk[min(i, model_action_chunk.shape[0] - 1)].clone())

    def should_request_observation(self) -> bool:
        if not self.pending_actions:
            return True
        if not self.use_memory:
            return False
        recent_capture_mod = (-self.action_horizon) % self.replan_steps
        return self.step_count % self.replan_steps == recent_capture_mod

    def get_action(self, request: Dict[str, Any]) -> np.ndarray:
        """Predict one replan chunk for a simulator running in another env.

        The returned chunk is limited to ``replan_steps`` (24 for the current
        evaluation), while the model still predicts its full 32-step horizon.
        Post-action observations must be acknowledged through ``update_obs``
        so memory and action history advance at exactly the executed cadence.
        """
        if not isinstance(request, dict):
            raise TypeError(f"Expected get_action request dict, got {type(request)!r}")
        observation = request.get("observation")
        instruction = request.get("instruction")
        if not isinstance(observation, dict):
            raise ValueError("get_action requires a RoboTwin observation dict")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("get_action requires a non-empty instruction")
        if self.pending_model_actions:
            raise RuntimeError(
                "A new replan was requested before all prior actions were acknowledged: "
                f"pending={len(self.pending_model_actions)}"
            )

        with _profile_section(self.profiler, "memory_observation_record"):
            self._record_memory_observation(observation)
        with _profile_section(self.profiler, "action_chunk_inference"):
            action_chunk, model_action_chunk = self._infer_action_chunk(
                observation=observation,
                instruction=instruction,
            )
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        self._timing_rollout["scheduled_action_steps"] += int(n_exec)
        with _profile_section(self.profiler, "action_queue_prepare"):
            for index in range(n_exec):
                self.pending_model_actions.append(
                    model_action_chunk[min(index, model_action_chunk.shape[0] - 1)].clone()
                )
        return np.asarray(action_chunk[:n_exec], dtype=np.float32)

    def update_obs(self, observation: Optional[Dict[str, Any]] = None) -> int:
        """Acknowledge one action and ingest the resulting simulator state."""
        if observation is not None and not isinstance(observation, dict):
            raise TypeError(f"Expected RoboTwin observation dict, got {type(observation)!r}")
        if not self.pending_model_actions:
            raise RuntimeError("update_obs received without a pending executed action")
        with _profile_section(self.profiler, "action_history_update"):
            model_action = self.pending_model_actions.popleft()
            self.action_history.append(model_action.detach().to(device="cpu", dtype=torch.float32))
            if self.use_ar_action_history and not self.use_memory:
                while len(self.action_history) > self.ar_max_history_actions:
                    self.action_history.popleft()
            self.step_count += 1
            self._timing_rollout["executed_action_steps"] += 1
        if observation is not None:
            with _profile_section(self.profiler, "memory_observation_record"):
                self._record_memory_observation(observation)
        return self.step_count

    def reset_model(self) -> bool:
        self.reset()
        return True

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if self.use_memory and observation is not None:
            self._record_memory_observation(observation)
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for wam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        model_action = self.pending_model_actions.popleft() if self.pending_model_actions else None
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        if (self.use_ar_action_history or self.use_memory) and model_action is not None:
            self.action_history.append(model_action.detach().to(device="cpu", dtype=torch.float32))
            if self.use_ar_action_history and not self.use_memory:
                while len(self.action_history) > self.ar_max_history_actions:
                    self.action_history.popleft()
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout.update(
            {
                "infer_s": 0.0,
                "sim_s": 0.0,
                "replan_calls": 0,
                "predicted_action_steps": 0,
                "scheduled_action_steps": 0,
                "executed_action_steps": 0,
                "memory_observation_calls": 0,
            }
        )

    def get_timing_rollout(self) -> Dict[str, Any]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
            "replan_calls": int(self._timing_rollout["replan_calls"]),
            "predicted_action_steps": int(self._timing_rollout["predicted_action_steps"]),
            "scheduled_action_steps": int(self._timing_rollout["scheduled_action_steps"]),
            "executed_action_steps": int(self._timing_rollout["executed_action_steps"]),
            "memory_observation_calls": int(self._timing_rollout["memory_observation_calls"]),
            "action_chunk_hashes": list(self._action_chunk_hashes),
        }

    def handle(self, command: str, payload: object) -> object:
        """Dispatch one validated RPC while preserving per-session failures."""
        methods = {
            "get_action": self.get_action,
            "update_obs": self.update_obs,
            "reset_model": self.reset_model,
            "get_timing_rollout": self.get_timing_rollout,
            "should_request_observation": self.should_request_observation,
        }
        method = methods.get(str(command))
        if method is None:
            raise SessionError(f"unsupported RoboTwin policy command: {command!r}")
        try:
            if command in {"reset_model", "get_timing_rollout", "should_request_observation"}:
                if payload is not None:
                    raise ValueError(f"{command} does not accept a payload")
                return method()
            return method(payload)
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as exc:
            message = str(exc).lower()
            if any(
                marker in message
                for marker in (
                    "cuda error",
                    "cuda out of memory",
                    "device-side assert",
                    "illegal memory access",
                    "cublas",
                    "cudnn",
                )
            ):
                raise
            raise SessionError(str(exc)) from exc
        except (KeyError, TypeError, ValueError, ProtocolError) as exc:
            raise SessionError(str(exc)) from exc

    def close(self) -> None:
        """Release this connection's state without touching shared runtime."""
        self.pending_actions.clear()
        self.pending_model_actions.clear()
        self.action_history.clear()
        self.visual_history.clear()
        self.proprio_history.clear()
        self._last_visual_history_step = None
        self._action_chunk_hashes.clear()

    def reset(self) -> None:
        with _profile_section(self.profiler, "policy_reset"):
            self.pending_actions.clear()
            self.pending_model_actions.clear()
            self.action_history.clear()
            self.visual_history.clear()
            self.proprio_history.clear()
            self._last_visual_history_step = None
            self._action_chunk_hashes.clear()
            self.episode_count += 1
            self.step_count = 0
            self.reset_timing_rollout()


class WorldActionRobotWinPolicy(RobotWinPolicySession):
    """Compatibility facade for existing single-policy evaluation callers."""


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def build_robotwin_runtime_and_session_config(
    usr_args: Dict[str, Any],
) -> tuple[RobotWinSharedRuntime, RobotWinSessionConfig]:
    profiler = usr_args.get("profiler")
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    with _profile_section(profiler, "config_compose"):
        cfg = _compose_sim_cfg(
            sim_cfg_path=sim_cfg_path,
            sim_cfg_name=sim_cfg_name,
            sim_task=sim_task,
        )
    with _profile_section(profiler, "model_overrides_apply"):
        _apply_model_overrides(cfg, usr_args)

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = _normalize_device(
        str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    with _profile_section(profiler, "dataset_stats_resolve"):
        dataset_stats_path = _resolve_dataset_stats_path(
            dataset_stats_path=usr_args.get("dataset_stats_path"),
        )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    action_hz = _parse_optional_float(usr_args.get("action_hz"))
    if action_hz is None:
        action_hz = _parse_optional_float(cfg.EVALUATION.get("action_hz"))
    if action_hz is None:
        raise ValueError(
            "Physical-time Action RoPE requires EVALUATION.action_hz."
        )

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )

    runtime = build_robotwin_shared_runtime(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        profiler=profiler,
    )
    # Keep online VAE encoding identical to the training dataset contract.
    # RoboTwin training currently builds one pixel-space T-shaped canvas and
    # publishes it as video_layout="single" when single_canvas=true.  Passing
    # "robotwin" here would make video_latent_codec split the canvas into
    # camera slots and encode them independently, which is not equivalent to
    # the whole-canvas VAE encode used by training.
    model_video_layout, model_video_view_names = (
        _resolve_training_video_contract(cfg.data.train)
    )

    session_config = RobotWinSessionConfig(
        action_horizon=action_horizon,
        action_hz=action_hz,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        # Match the exact pixel-space T-shaped canvas and VAE layout contract
        # used by training. Future Delta has one learned query per latent
        # spatial token, so changing this geometry or the VAE boundary context
        # at evaluation is not a valid resize-only operation.
        video_size=tuple(int(value) for value in cfg.data.train.video_size),
        video_layout=model_video_layout,
        video_view_names=model_video_view_names,
    )
    return runtime, session_config


def get_model(usr_args: Dict[str, Any]):
    runtime, session_config = build_robotwin_runtime_and_session_config(usr_args)
    return WorldActionRobotWinPolicy(
        runtime=runtime,
        config=session_config,
        profiler=usr_args.get("profiler"),
    )


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
