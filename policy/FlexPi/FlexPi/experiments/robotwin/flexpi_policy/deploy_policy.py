import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flexpi.datasets.lerobot.processors.flexpi_processor import FlexPiProcessor
from flexpi.datasets.lerobot.transforms.mixed_embodiment_align import (
    _PAD_SLOTS as ALIGN34_PAD_SLOTS,
    AlignTo34DByWidth,
)
from flexpi.datasets.lerobot.transforms.misc import StaticActionDimPad
from flexpi.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from flexpi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from flexpi.per_cam_compose import compose_robotwin_from_per_cam
from flexpi.datasets.lerobot.robot_video_dataset import _PER_CAM_HW

logger = logging.getLogger(__name__)


# Single source of truth for the dataset-cam-name → RoboTwin obs-key mapping.
_CAM_NAME_TO_OBS = {
    "cam_high":        "head_camera",
    "cam_left_wrist":  "left_camera",
    "cam_right_wrist": "right_camera",
}
_CAM_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _build_intrinsics_from_obs(observation: Dict[str, Any]) -> torch.Tensor:
    """Stack per-cam K from RoboTwin's obs dict at sim-native resolution.

    RoboTwin's ``get_obs()`` includes ``observation[cam]['intrinsic_cv']``
    (3×3) at the camera's native render resolution. The companion
    ``observation[cam]['depth']`` lives on the same grid, so K and depth
    are coherent by construction — the model can consume both as-is.
    """
    obs_data = observation["observation"]
    Ks = []
    for cam_name in _CAM_ORDER:
        obs_key = _CAM_NAME_TO_OBS[cam_name]
        K_raw = obs_data[obs_key]["intrinsic_cv"]
        K = np.asarray(K_raw, dtype=np.float32).copy()
        Ks.append(torch.from_numpy(K))
    return torch.stack(Ks, dim=0)


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


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _find_trained_config(ckpt_path: Path) -> Optional[Path]:
    """Locate the `config.yaml` saved next to a training checkpoint.

    Training writes `<run_dir>/config.yaml` (see runtime.run_training). The ckpt
    typically lives at `<run_dir>/checkpoints/weights/step_XXXX.pt`, so we walk
    up a few parents looking for it.
    """
    for parent in list(ckpt_path.resolve().parents)[:5]:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _resize_rgb_antialiased(rgb_uint8: np.ndarray, hw: tuple[int, int]) -> torch.Tensor:
    """Resize a [H, W, 3] uint8 RGB image to (h, w) with antialiased bilinear,
    returning a [3, h, w] float tensor in [0, 1]. Matches the dataset workers'
    ``torchvision.transforms.functional.resize(antialias=True)`` semantics —
    using PIL's BILINEAR (no antialias) here would diverge from training and
    fold high-frequency texture into low frequencies on the downsample path.
    """
    import torchvision.transforms.functional as TF
    t = torch.from_numpy(rgb_uint8.astype(np.uint8)).permute(2, 0, 1).float() / 255.0
    return TF.resize(
        t, size=list(hw),
        interpolation=TF.InterpolationMode.BILINEAR, antialias=True,
    )


def _depth_to_train_grid(depth: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Resample a [H, W] uint16 depth map onto (h, w) the way the training loaders
    do (anydata ``_convert`` / ``RobotPerCamDepthVideoDataset``): nearest gather at
    ``round(linspace(0, src-1, dst))`` rows/cols. Pairs with ``_endpoint_rescale_K``.
    """
    h, w = int(hw[0]), int(hw[1])
    src_h, src_w = int(depth.shape[-2]), int(depth.shape[-1])
    if (src_h, src_w) == (h, w):
        return depth
    ys = torch.linspace(0, src_h - 1, h).round().long()
    xs = torch.linspace(0, src_w - 1, w).round().long()
    # uint16 has no advanced-indexing kernel on CPU -> int32 round-trip, as in training.
    return depth.to(torch.int32)[..., ys[:, None], xs[None, :]].to(torch.uint16)


def _endpoint_rescale_K(K: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    """Rescale a 3x3 K from a (H, W) grid to (h, w) with the endpoint rule the
    training loaders use: s = (dst - 1) / (src - 1), applied to fx, cx / fy, cy."""
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    h, w = int(dst_hw[0]), int(dst_hw[1])
    sx = (w - 1) / (src_w - 1) if src_w > 1 else 1.0
    sy = (h - 1) / (src_h - 1) if src_h > 1 else 1.0
    k = K.to(torch.float64)
    return torch.tensor(
        [[k[0, 0] * sx, 0.0, k[0, 2] * sx],
         [0.0, k[1, 1] * sy, k[1, 2] * sy],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        offload_text_encoder: bool = False,
        torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        torch_compile_scope: str = "step",
        torch_compile_backend: str = "inductor",
        quantization: Optional[str] = None,
        attn_backend: str = "sdpa",
        compile_encoders: bool = False,
        trt_joint_engine_path: Optional[str] = None,
        trt_joint_free_video_blocks: bool = False,
        trt_joint_prefill_split_engine_path: Optional[str] = None,
        trt_joint_decode_split_engine_path: Optional[str] = None,
        solver: str = "euler",
        glue_cache: bool = False,
        encoder_cuda_graph: bool = False,
        joint_loop_cuda_graph: bool = False,
        use_per_cam: bool = False,
        dynamic_step_skip: bool = False,
        step_skip_thresholds: Optional[list] = None,
        step_skip_sim_sources: Optional[list] = None,
        step_skip_sim_aggregation: str = "mean",
        # Flex runtime overrides (FlexPiUnifiedJoint flex_joint runs only).
        # ``None`` => preserve trained-config default (no override).
        infer_joint_video: Optional[bool] = None,
        infer_joint_dino: Optional[bool] = None,
        infer_joint_pointmap: Optional[bool] = None,
        # Flex presence overrides — drop a stream from the input layout at
        # deploy. None => present at inference (default). Setting False forces
        # the joint denoise path so the presence-aware mask builder fires.
        infer_present_video: Optional[bool] = None,
        infer_present_dino: Optional[bool] = None,
        infer_present_pointmap: Optional[bool] = None,
        # Pad-channel mask on the action latent. None => derive from the
        # processor (scatter ids / AlignTo34DByWidth); False => never arm, for
        # checkpoints whose training left those channels unmasked; True => arm
        # whenever the processor pads.
        action_dim_mask: Optional[bool] = None,
        # True => build per-cam RGB, depth and K exactly as the training loader
        # did: RGB through the checkpoint processor's own image transforms, then
        # the per-cam resize; depth resampled onto processor shape_meta.depth with
        # the endpoint K. False => the single native->per-cam resize and
        # native-resolution depth + K.
        obs_train_parity: bool = False,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        if offload_text_encoder:
            # Skip T5 during model init — load it to CPU separately below.
            model_cfg_copy.load_text_encoder = False
        else:
            model_cfg_copy.load_text_encoder = True
        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        if offload_text_encoder:
            from flexpi.models.helpers.loader import load_text_encoder_to_device
            te, tok = load_text_encoder_to_device(
                device="cpu",
                torch_dtype=getattr(torch, model_dtype) if isinstance(model_dtype, str) else model_dtype,
                model_id=model_cfg_copy.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
                tokenizer_model_id=model_cfg_copy.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
                tokenizer_max_len=int(model_cfg_copy.get("tokenizer_max_len", 512)),
            )
            self.model.text_encoder = te
            self.model.tokenizer = tok
            self.model.offload_text_encoder = True
            print("[FlexPi] Text encoder loaded on CPU (never touches GPU). Saves ~10GB VRAM.")
        self.model = self.model.to(device).eval()

        if hasattr(self.model, "prepare_for_inference"):
            self.model.prepare_for_inference(
                torch_compile=torch_compile,
                torch_compile_mode=torch_compile_mode,
                quantization=quantization,
                torch_compile_scope=torch_compile_scope,
                attn_backend=attn_backend,
                torch_compile_backend=torch_compile_backend,
                compile_encoders=compile_encoders,
                trt_joint_engine_path=trt_joint_engine_path,
                trt_joint_free_video_blocks=trt_joint_free_video_blocks,
                trt_joint_prefill_split_engine_path=trt_joint_prefill_split_engine_path,
                trt_joint_decode_split_engine_path=trt_joint_decode_split_engine_path,
                solver=solver,
                glue_cache=glue_cache,
                encoder_cuda_graph=encoder_cuda_graph,
                joint_loop_cuda_graph=joint_loop_cuda_graph,
            )

        self.processor: FlexPiProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)
        self._action_dim_mask_override = action_dim_mask
        self.obs_train_parity = bool(obs_train_parity)
        if self.obs_train_parity:
            self._depth_train_hw = {
                m["key"]: (int(m["shape"][-2]), int(m["shape"][-1]))
                for m in (self.processor.shape_meta.get("depth") or [])
            }
            print(
                f"[FlexPi] obs_train_parity=True: RGB via processor val_transforms + per-cam "
                f"resize; depth/K resampled to {self._depth_train_hw}"
            )
        # Must precede warmup: compiled / CUDA-graphed loops bake the mask in.
        self._set_action_pad_mask()

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.use_per_cam = bool(use_per_cam)
        model_accepts_per_cam = "per_cam" in inspect.signature(self.model.infer_action).parameters
        if self.use_per_cam and model_accepts_per_cam:
            print("[FlexPi] use_per_cam=True, model accepts per_cam → per-cam path active")
        elif self.use_per_cam and not model_accepts_per_cam:
            print(
                "[FlexPi] WARNING: use_per_cam=True but model.infer_action does NOT accept "
                "`per_cam` kwarg → per-cam tensors will be built and silently dropped"
            )
        else:
            print("[FlexPi] use_per_cam=False → composite path (cameras resized directly into 384×320 slots)")
        self.dynamic_step_skip = bool(dynamic_step_skip)
        self.infer_joint_video = (None if infer_joint_video is None else bool(infer_joint_video))
        self.infer_joint_dino = (None if infer_joint_dino is None else bool(infer_joint_dino))
        self.infer_joint_pointmap = (None if infer_joint_pointmap is None else bool(infer_joint_pointmap))
        self.infer_present_video = (None if infer_present_video is None else bool(infer_present_video))
        self.infer_present_dino = (None if infer_present_dino is None else bool(infer_present_dino))
        self.infer_present_pointmap = (None if infer_present_pointmap is None else bool(infer_present_pointmap))
        if (
            self.infer_joint_video is not None
            or self.infer_joint_dino is not None
            or self.infer_joint_pointmap is not None
            or self.infer_present_video is not None
            or self.infer_present_dino is not None
            or self.infer_present_pointmap is not None
        ):
            print(
                "[FlexPi] flex infer overrides: "
                f"joint_video={self.infer_joint_video} "
                f"joint_dino={self.infer_joint_dino} "
                f"joint_pointmap={self.infer_joint_pointmap} "
                f"present_video={self.infer_present_video} "
                f"present_dino={self.infer_present_dino} "
                f"present_pointmap={self.infer_present_pointmap}"
            )
        # Normalize thresholds to list-of-tuples; OmegaConf delivers list-of-lists.
        if step_skip_thresholds is not None:
            step_skip_thresholds = [
                (float(t), int(n)) for t, n in step_skip_thresholds
            ]
        self.step_skip_thresholds = step_skip_thresholds
        self.step_skip_sim_sources = (
            None if step_skip_sim_sources is None
            else [str(s) for s in step_skip_sim_sources]
        )
        self.step_skip_sim_aggregation = str(step_skip_sim_aggregation)

        # K is sourced from the RoboTwin obs every step (see
        # `_infer_action_chunk` → `_build_intrinsics_from_obs`). Sim emits
        # K and depth on the same grid, so they're coherent by construction.
        self._obs_intrinsics_logged = False
        self._zero_anchor_warned = False

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._cached_prompt: Optional[str] = None
        self._cached_context: Optional[torch.Tensor] = None
        self._cached_context_mask: Optional[torch.Tensor] = None
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        self._infer_calls = 0
        self._episode_start_time = time.perf_counter()

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

        # Trigger torch.compile / CUDA Graph capture before timed inference.
        # Compile cost (10–30 s) lands here instead of in the first real call.
        if bool(torch_compile):
            self._warmup()

    def _warmup(self) -> None:
        """Run one dummy infer_action to trigger torch.compile + CUDA-Graph capture.

        The compiled denoise step is shape-keyed, so we feed the same shapes the
        real path will: composite ``input_image`` always, and ``per_cam`` /
        ``per_cam_depth`` when the model declares those kwargs.
        """
        warmup_t0 = time.perf_counter()

        infer_sig = inspect.signature(self.model.infer_action).parameters
        device = self.model.device
        dtype = self.model.torch_dtype

        # Composite input image — RoboTwin layout per FlexPi CLAUDE.md.
        dummy_image = torch.zeros((1, 3, 384, 320), device=device, dtype=dtype)

        # Encode a dummy prompt (cleared after warmup so first real prompt re-encodes).
        dummy_prompt = DEFAULT_PROMPT.format(task="warmup")
        with torch.no_grad():
            dummy_context, dummy_context_mask = self.model.encode_prompt(dummy_prompt)

        proprio_dim = getattr(self.model, "proprio_dim", None)
        dummy_proprio = (
            torch.zeros((1, proprio_dim), dtype=torch.float32) if proprio_dim is not None else None
        )

        warmup_kwargs: Dict[str, Any] = {
            "prompt": None,
            "input_image": dummy_image,
            "action_horizon": self.action_horizon,
            "proprio": dummy_proprio,
            "context": dummy_context,
            "context_mask": dummy_context_mask,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": 0,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in infer_sig:
            warmup_kwargs["num_video_frames"] = int(self._num_video_frames)
        if "return_stream_latents" in infer_sig:
            # Deploy reads only out["action"]; skip the blocking D2H copies of
            # the denoised video/dino/pointmap latents.
            warmup_kwargs["return_stream_latents"] = False

        if self.use_per_cam and "per_cam" in infer_sig:
            warmup_kwargs["per_cam"] = {
                cam_name: torch.zeros((1, 3, h_cam, w_cam), device=device, dtype=dtype)
                for cam_name, (h_cam, w_cam) in _PER_CAM_HW
            }

        if "per_cam_depth" in infer_sig:
            warmup_kwargs["per_cam_depth"] = {
                cam_name: torch.zeros((1, 1, h_cam, w_cam), device=device, dtype=dtype)
                for cam_name, (h_cam, w_cam) in _PER_CAM_HW
            }
            # Set dummy intrinsics so set_camera_intrinsics doesn't trip.
            if hasattr(self.model, "set_camera_intrinsics"):
                K_dummy = torch.eye(3, device=device).unsqueeze(0).expand(len(_PER_CAM_HW), -1, -1).contiguous()
                self.model.set_camera_intrinsics(K_dummy)

        # Pass the user's flex overrides into warmup so the compile entry that
        # gets warmed matches the regime the real eval will run. Without this,
        # warmup compiles for the trained default (e.g. (T,T,T)) and the first
        # real call under an override (e.g. (F,T,T) + present_v=False) pays a
        # fresh 10–30 s compile cost.
        if "joint_video" in infer_sig and self.infer_joint_video is not None:
            warmup_kwargs["joint_video"] = self.infer_joint_video
        if "joint_dino" in infer_sig and self.infer_joint_dino is not None:
            warmup_kwargs["joint_dino"] = self.infer_joint_dino
        if "joint_pointmap" in infer_sig and self.infer_joint_pointmap is not None:
            warmup_kwargs["joint_pointmap"] = self.infer_joint_pointmap
        if "present_video" in infer_sig and self.infer_present_video is not None:
            warmup_kwargs["present_video"] = self.infer_present_video
        if "present_dino" in infer_sig and self.infer_present_dino is not None:
            warmup_kwargs["present_dino"] = self.infer_present_dino
        if "present_pointmap" in infer_sig and self.infer_present_pointmap is not None:
            warmup_kwargs["present_pointmap"] = self.infer_present_pointmap

        with torch.no_grad():
            self.model.infer_action(**warmup_kwargs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Clear cache so first real prompt re-encodes (DEFAULT_PROMPT differs by task).
        self._cached_prompt = None
        self._cached_context = None
        self._cached_context_mask = None

        warmup_elapsed = time.perf_counter() - warmup_t0
        logger.info("torch.compile warmup done in %.2f s", warmup_elapsed)

    def _set_action_pad_mask(self) -> None:
        """Zero the action-head channels this embodiment never fills.

        Training zeroes padded channels on the noisy/target action and masks them
        out of the loss. At deploy nothing supplies that mask, so they would start
        as N(0,1) and leak into the real channels through the action DiT. Two
        sources: a scatter merger's unused channel ids, or AlignTo34DByWidth's
        per-sample pad slots (the YAM 32-wide state is always padded there).

        ``action_dim_mask=False`` skips this: a checkpoint trained with an
        identity merger (e.g. the 34D-native RoboDojo runs, whose training
        processor emitted an all-False ``action_dim_is_pad``) learned those
        channels as ordinary noised inputs, so zeroing them is out of distribution.

        A third source, StaticActionDimPad, is a mask the checkpoint WAS trained with
        (e.g. RoboDojo ABS_JOINTONLY: EEF channels 0..17 + 26/33), so it is ORed in and
        ``action_dim_mask=False`` is refused for such a chain.
        """
        transforms = self.processor.action_state_transforms or []
        static_pads = [t for t in transforms if isinstance(t, StaticActionDimPad)]
        if self._action_dim_mask_override is False:
            if static_pads:
                raise ValueError(
                    "action_dim_mask=false, but this checkpoint was trained with StaticActionDimPad "
                    f"(pad ids {static_pads[0].pad_ids}): its padded channels must be zeroed at "
                    "sampling too. Set action_dim_mask=true."
                )
            self.model.set_action_dim_mask(None)
            print("[FlexPi] action_dim_mask: disabled (action_dim_mask=false)")
            return
        merger = getattr(self.processor, "action_state_merger", None)
        used_ids = getattr(merger, "used_action_channel_ids", None)
        pad: Optional[torch.Tensor] = None
        if used_ids is not None:
            pad = torch.ones(int(merger.total_dim), dtype=torch.bool)
            pad[list(used_ids)] = False
        else:
            for trans in transforms:
                if not isinstance(trans, AlignTo34DByWidth):
                    continue
                if trans.backward_mode != "yam32":
                    raise ValueError(
                        f"AlignTo34DByWidth backward_mode={trans.backward_mode!r}; "
                        "this deploy only supports the 'yam32' path."
                    )
                pad = torch.zeros(int(getattr(merger, "total_dim", 34)), dtype=torch.bool)
                pad[list(ALIGN34_PAD_SLOTS)] = True
                break
        for trans in static_pads:
            pad = trans.mask() if pad is None else pad | trans.mask()
        if pad is None:
            print("[FlexPi] action_dim_mask: none (identity merger)")
            return
        self.model.set_action_dim_mask(pad)
        print(
            f"[FlexPi] action_dim_mask: {int(pad.sum())}/{int(pad.numel())} channels padded "
            f"(ids {torch.nonzero(pad).flatten().tolist()})"
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        # Scatter/pad the raw state into the model's proprio layout (e.g. 14→32 for
        # the AgiBot-unified ScatterToChannels). For the native ConcatLeftAlign (no
        # target dim) this is an identity concat — same values, just routed through
        # the merger so train and deploy agree. Mirrors LIBERO eval_libero_single.
        state_batch = self.processor.action_state_merger.forward(state_batch)
        return state_batch["state"]

    def _denormalize_action(
        self, action: torch.Tensor, anchor_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Invert the processor chain on a predicted action chunk.

        ``anchor_state`` is the raw (un-normalized) state vector the chunk was
        predicted from. Relative action transforms (e.g. Yam32DRelativeAction)
        read it as the base to add deltas back onto; without it they see a
        zero state and return the deltas themselves as absolute targets.
        Configs with no action_state_transforms ignore it.
        """
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
        action_key = action_meta[0]["key"]
        action = action.to(dtype=torch.float32, device="cpu")

        # Inverse of the training processor chain: merger.backward gathers the
        # model's total_dim output back to the raw per-key dims, then normalizer
        # denormalizes. For native ConcatLeftAlign this is an identity crop (same
        # values). Mirrors experiments/libero/eval_libero_single.py:_denormalize_action.
        merger = self.processor.action_state_merger
        batch = {"action": action}
        # merger.backward processes both action and state unconditionally; pass a
        # dummy zero state to satisfy its shape assert (its output is discarded).
        state_dim = getattr(merger, "state_target_dim", None) or getattr(merger, "total_dim", None)
        if state_dim is not None:
            batch["state"] = torch.zeros(
                action.shape[0], action.shape[1], int(state_dim),
                dtype=action.dtype, device=action.device,
            )
        batch = merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        if self.processor.action_state_transforms is not None:
            if anchor_state is not None:
                state_key = self.processor.shape_meta["state"][0]["key"]
                anchor = torch.as_tensor(np.asarray(anchor_state), dtype=torch.float32)
                anchor = anchor.reshape(1, 1, -1).expand(action.shape[0], 1, -1)
                # Widen the anchor exactly as the forward path does: a chain that
                # starts with AlignTo34DByWidth scatters 32 -> 34 before the
                # relative transform, which then expects a 34D base. For chains
                # that leave state untouched this is a no-op.
                anchor = self.processor.action_state_transform(
                    {"state": {state_key: anchor.clone()}}
                )["state"][state_key]
                batch["state"] = {state_key: anchor}
            elif not self._zero_anchor_warned:
                logger.warning(
                    "action_state_transforms are set but no anchor_state was given; "
                    "relative actions are being decoded against a zero state."
                )
                self._zero_anchor_warned = True
            for trans in reversed(self.processor.action_state_transforms):
                batch = trans.backward(batch)
        return batch["action"][action_key].numpy()

    def _build_robotwin_per_cam_tensors(
        self, observation: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Build per-camera tensors at per-cam sizes (head 256x320, wrists
        224x224 by default — driven by ``_PER_CAM_HW``). Each tensor is
        ``[1, 3, h, w]`` in [-1, 1] on the model's device/dtype.

        Matches RobotVideoDataset's per-cam pipeline exactly: one
        antialiased bilinear resize per camera from native MP4 resolution
        to per-cam target size, then linear normalization to [-1, 1].
        """
        obs_data = observation["observation"]
        per_cam: Dict[str, torch.Tensor] = {}
        for cam_name, (h_cam, w_cam) in _PER_CAM_HW:
            obs_key = _CAM_NAME_TO_OBS[cam_name]
            rgb = obs_data[obs_key]["rgb"]
            if self.obs_train_parity:
                t = self._train_chain_rgb(cam_name, rgb, (h_cam, w_cam))  # [3, h, w] in [0, 1]
            else:
                t = _resize_rgb_antialiased(rgb, (h_cam, w_cam))  # [3, h, w] in [0, 1]
            t = t * 2.0 - 1.0  # → [-1, 1]
            t = t.unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)
            per_cam[cam_name] = t  # [1, 3, h, w]
        return per_cam

    def _build_robotwin_per_cam_depth_tensors(
        self, observation: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Build per-camera GT depth tensors at sim-native resolution.

        RoboTwin's ``observation[cam]['depth']`` is uint16 mm on the same
        grid as ``observation[cam]['intrinsic_cv']``. We pass it through
        without resizing — depth and K are coherent at sim-native, and the
        model's pointmap encoder is grid-agnostic (it unprojects to metric
        XYZ then resamples internally to fixed encoder grids).
        Emits ``[1, 1, H, W]`` uint16 on the model's device; no float cast
        — the encoder does ``depth.float() / 1000.0`` itself.
        """
        obs_data = observation["observation"]
        per_cam_depth: Dict[str, torch.Tensor] = {}
        for cam_name in _CAM_ORDER:
            obs_key = _CAM_NAME_TO_OBS[cam_name]
            depth_np = obs_data[obs_key]["depth"]
            depth = torch.from_numpy(np.asarray(depth_np)).contiguous()
            if depth.dtype != torch.uint16:
                # RoboTwin emits uint16 but be defensive for real-world sensors.
                depth = depth.to(torch.int32).clamp_(0, 65535).to(torch.uint16)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth.squeeze(0)
            if depth.ndim != 2:
                raise ValueError(
                    f"observation[{obs_key}]['depth'] must be [H, W] or [1, H, W]; "
                    f"got shape {tuple(depth.shape)}"
                )
            if self.obs_train_parity and cam_name in self._depth_train_hw:
                depth = _depth_to_train_grid(depth, self._depth_train_hw[cam_name])
            # [1, 1, H, W] — T=1 slot mirrors the dataset's [B, T, H, W] shape.
            per_cam_depth[cam_name] = depth.unsqueeze(0).unsqueeze(0).to(
                device=self.model.device
            )
        return per_cam_depth

    def _train_chain_rgb(self, cam_name: str, rgb_uint8: np.ndarray, hw: tuple[int, int]) -> torch.Tensor:
        """Per-cam RGB exactly as the training loader built it: the checkpoint
        processor's own image transforms (e.g. ToTensor + Resize([256, 320]))
        on the uint8 frame, then the antialiased resize to the per-cam size (a
        no-op when the sizes already match). Returns [3, h, w] in [0, 1]."""
        import torchvision.transforms.functional as TF
        transforms = self.processor.val_transforms
        if isinstance(transforms, (dict, DictConfig)):
            transforms = transforms[cam_name]
        x = torch.from_numpy(np.ascontiguousarray(rgb_uint8, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)
        for trans in transforms:
            x = trans(x)  # [1, 3, H', W'] float in [0, 1]
        x = TF.resize(
            x, size=list(hw),
            interpolation=TF.InterpolationMode.BILINEAR, antialias=True,
        )
        return x[0]

    def _build_intrinsics(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Per-cam K in _CAM_ORDER; endpoint-rescaled onto the training depth
        grid when obs_train_parity is on (must match _build_robotwin_per_cam_depth_tensors)."""
        K = _build_intrinsics_from_obs(observation)
        if not self.obs_train_parity:
            return K
        obs_data = observation["observation"]
        Ks = []
        for i, cam_name in enumerate(_CAM_ORDER):
            dst = self._depth_train_hw.get(cam_name)
            if dst is None:
                Ks.append(K[i])
                continue
            src = np.asarray(obs_data[_CAM_NAME_TO_OBS[cam_name]]["depth"]).shape[-2:]
            Ks.append(_endpoint_rescale_K(K[i], src, dst))
        return torch.stack(Ks, dim=0)

    def _build_robotwin_image_tensor(
        self, observation: Dict[str, Any],
        per_cam: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Build the [1, 3, 384, 320] composite for the VAE input path.

        When ``per_cam`` is provided, derives the composite from the per-cam
        tensors via the shared ``compose_robotwin_from_per_cam`` helper —
        train and deploy then share the exact same compose pipeline (single
        per-cam resize on native, then 224→128 wrist downsample on GPU).

        When ``per_cam`` is None, falls back to the legacy direct
        native→slot resize path (for composite-only-trained models).
        """
        if per_cam is not None:
            # Promote per_cam to [B, 3, T=1, H, W] for the helper, then
            # collapse to [1, 3, H, W] for the VAE first-frame API.
            per_cam_5d = {k: v.unsqueeze(2) for k, v in per_cam.items()}
            composite = compose_robotwin_from_per_cam(per_cam_5d)  # [1, 3, 1, 384, 320]
            return composite.squeeze(2)  # [1, 3, 384, 320]
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        # When use_per_cam is on (deploying a model trained with
        # RobotVideoDataset), build per-cam first and derive the
        # composite from it via the shared compose helper. Train and
        # deploy then share exactly one resize chain. Models that consume
        # per_cam (FlexPiLatent, FlexPiFlexDino) bypass the composite-slice
        # path entirely.
        per_cam: Optional[Dict[str, torch.Tensor]] = None
        if self.use_per_cam:
            per_cam = self._build_robotwin_per_cam_tensors(observation)
        # Build per_cam_depth only when the model accepts it (the infer_action
        # signature is the contract — no model-type checks). K and depth are
        # both sourced from the RoboTwin obs at sim-native — the model's
        # pointmap encoder unprojects to metric XYZ then resamples to its
        # own internal grids, so it's resolution-agnostic at this boundary.
        # Skipped when the pointmap stream is dropped at runtime
        # (infer_present_pointmap=False): the model then never reads depth or K,
        # so observations without them stay valid.
        per_cam_depth: Optional[Dict[str, torch.Tensor]] = None
        infer_sig = inspect.signature(self.model.infer_action).parameters
        if "per_cam_depth" in infer_sig and self.infer_present_pointmap is not False:
            K_obs = self._build_intrinsics(observation)
            self.model.set_camera_intrinsics(K_obs.to(device=self.model.device))
            if not self._obs_intrinsics_logged:
                logger.info("Camera intrinsics sourced from RoboTwin obs (sim-native)")
                self._obs_intrinsics_logged = True
            per_cam_depth = self._build_robotwin_per_cam_depth_tensors(observation)
        image_tensor = self._build_robotwin_image_tensor(observation, per_cam=per_cam)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        if self._cached_prompt != prompt:
            print(f"[FlexPi] Encoding prompt (episode {self.episode_count}): {prompt!r}")
            self._cached_context, self._cached_context_mask = self.model.encode_prompt(prompt)
            self._cached_prompt = prompt
        infer_kwargs = {
            "prompt": None,
            "context": self._cached_context,
            "context_mask": self._cached_context_mask,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in infer_sig:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        if "return_stream_latents" in infer_sig:
            # Deploy reads only out["action"]; skip the blocking D2H copies of
            # the denoised video/dino/pointmap latents.
            infer_kwargs["return_stream_latents"] = False
        # Forward per_cam only when the model declares the kwarg — keeps
        # the deploy compatible with older composite-only models that
        # don't have per_cam in their infer_action signature.
        if per_cam is not None and "per_cam" in infer_sig:
            infer_kwargs["per_cam"] = per_cam
        if per_cam_depth is not None:
            infer_kwargs["per_cam_depth"] = per_cam_depth
        # Step-skipping (joint-variant models only — FlexPiLatentJoint /
        # FlexPi3DJoint). Other models' signatures don't accept these; gate
        # each one individually so this stays a no-op elsewhere.
        if "dynamic_step_skip" in infer_sig:
            infer_kwargs["dynamic_step_skip"] = bool(self.dynamic_step_skip)
        if "step_skip_thresholds" in infer_sig and self.step_skip_thresholds is not None:
            infer_kwargs["step_skip_thresholds"] = self.step_skip_thresholds
        if "step_skip_sim_sources" in infer_sig and self.step_skip_sim_sources is not None:
            infer_kwargs["step_skip_sim_sources"] = list(self.step_skip_sim_sources)
        if "step_skip_sim_aggregation" in infer_sig:
            infer_kwargs["step_skip_sim_aggregation"] = self.step_skip_sim_aggregation
        # Flex runtime joint overrides (FlexPiUnifiedJoint flex_joint runs).
        # Gate by signature so older models silently ignore.
        if "joint_video" in infer_sig and self.infer_joint_video is not None:
            infer_kwargs["joint_video"] = self.infer_joint_video
        if "joint_dino" in infer_sig and self.infer_joint_dino is not None:
            infer_kwargs["joint_dino"] = self.infer_joint_dino
        if "joint_pointmap" in infer_sig and self.infer_joint_pointmap is not None:
            infer_kwargs["joint_pointmap"] = self.infer_joint_pointmap
        if "present_video" in infer_sig and self.infer_present_video is not None:
            infer_kwargs["present_video"] = self.infer_present_video
        if "present_dino" in infer_sig and self.infer_present_dino is not None:
            infer_kwargs["present_dino"] = self.infer_present_dino
        if "present_pointmap" in infer_sig and self.infer_present_pointmap is not None:
            infer_kwargs["present_pointmap"] = self.infer_present_pointmap
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            if self.model.device.type == "cuda":
                torch.cuda.synchronize()
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
            self._infer_calls += 1

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor, anchor_state=state_vector)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for flexpi)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        if self.timing_enabled and self.step_count > 0:
            rollout = self.get_timing_rollout()
            wall_s = time.perf_counter() - self._episode_start_time
            n_calls = max(self._infer_calls, 1)
            n_steps = max(self.step_count, 1)
            infer_ms_per_call = 1000.0 * rollout["infer_s"] / n_calls
            sim_ms_per_step = 1000.0 * rollout["sim_s"] / n_steps
            policy_ms_per_step = infer_ms_per_call / self.replan_steps + sim_ms_per_step
            policy_fps = 1000.0 / policy_ms_per_step if policy_ms_per_step > 0 else float("inf")
            print(
                f"[FlexPi timing] episode={self.episode_count} steps={self.step_count} "
                f"wall_s={wall_s:.3f} fps={self.step_count / wall_s:.2f} "
                f"policy_fps={policy_fps:.2f} "
                f"infer_calls={self._infer_calls} infer_s={rollout['infer_s']:.3f} "
                f"infer_ms/call={infer_ms_per_call:.1f} "
                f"sim_s={rollout['sim_s']:.3f}",
                flush=True,
            )
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self._cached_prompt = None
        self._cached_context = None
        self._cached_context_mask = None
        self.reset_timing_rollout()
        self._infer_calls = 0
        self._episode_start_time = time.perf_counter()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    # Load architecture + processor from the config saved next to the checkpoint
    # at training time, so eval can't drift from training (wrong joint flags,
    # layer counts, etc.). Fall back to the preset if the file is missing.
    trained_cfg_path = _find_trained_config(Path(str(checkpoint_path)))
    trained_cfg = None
    if trained_cfg_path is not None:
        trained_cfg = OmegaConf.load(trained_cfg_path)
        cfg.model = trained_cfg.model
        cfg.data.train.processor = trained_cfg.data.train.processor
        print(f"[FlexPi] Loaded model + processor from trained config: {trained_cfg_path}")
    else:
        print(
            f"[FlexPi] WARNING: No config.yaml found next to checkpoint "
            f"{checkpoint_path}; using task preset. Risk of train/eval drift.",
            file=sys.stderr,
        )

    # Name the variant so the per-class logger lines below (FlexPiLatent /
    # FlexPi3D / FlexPiUnified / *Joint) read in context.
    print(f"[FlexPi] model._target_={cfg.model.get('_target_', '?')}")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 32))

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

    offload_text_encoder = _parse_bool(
        usr_args.get("offload_text_encoder", cfg.EVALUATION.get("offload_text_encoder", False))
    )
    torch_compile = _parse_bool(
        usr_args.get("torch_compile", cfg.EVALUATION.get("torch_compile", False))
    )
    torch_compile_mode = str(
        usr_args.get("torch_compile_mode", cfg.EVALUATION.get("torch_compile_mode", "reduce-overhead"))
    )
    torch_compile_scope = str(
        usr_args.get("torch_compile_scope", cfg.EVALUATION.get("torch_compile_scope", "step"))
    )
    attn_backend = str(
        usr_args.get("attn_backend", cfg.EVALUATION.get("attn_backend", "sdpa"))
    )
    torch_compile_backend = str(
        usr_args.get("torch_compile_backend", cfg.EVALUATION.get("torch_compile_backend", "inductor"))
    )
    compile_encoders = _parse_bool(
        usr_args.get("compile_encoders", cfg.EVALUATION.get("compile_encoders", False))
    )
    quantization_raw = usr_args.get("quantization", cfg.EVALUATION.get("quantization"))
    quantization = None if _is_none_like(quantization_raw) else str(quantization_raw)
    trt_joint_engine_raw = usr_args.get(
        "trt_joint_engine_path", cfg.EVALUATION.get("trt_joint_engine_path")
    )
    trt_joint_engine_path = (
        None if _is_none_like(trt_joint_engine_raw) else str(trt_joint_engine_raw)
    )
    _pf_split_raw = usr_args.get(
        "trt_joint_prefill_split_engine_path",
        cfg.EVALUATION.get("trt_joint_prefill_split_engine_path"),
    )
    trt_joint_prefill_split_engine_path = None if _is_none_like(_pf_split_raw) else str(_pf_split_raw)
    _dec_split_raw = usr_args.get(
        "trt_joint_decode_split_engine_path",
        cfg.EVALUATION.get("trt_joint_decode_split_engine_path"),
    )
    trt_joint_decode_split_engine_path = None if _is_none_like(_dec_split_raw) else str(_dec_split_raw)
    trt_joint_free_video_blocks = _parse_bool(
        usr_args.get(
            "trt_joint_free_video_blocks",
            cfg.EVALUATION.get("trt_joint_free_video_blocks", False),
        )
    )
    solver = str(usr_args.get("solver", cfg.EVALUATION.get("solver", "euler")))
    glue_cache = _parse_bool(
        usr_args.get("glue_cache", cfg.EVALUATION.get("glue_cache", False))
    )
    encoder_cuda_graph = _parse_bool(
        usr_args.get("encoder_cuda_graph", cfg.EVALUATION.get("encoder_cuda_graph", False))
    )
    joint_loop_cuda_graph = _parse_bool(
        usr_args.get("joint_loop_cuda_graph", cfg.EVALUATION.get("joint_loop_cuda_graph", False))
    )

    # Per-cam deploy: produce per-cam tensors at per-cam sizes (head 256×320,
    # wrists 224×224) and forward them to infer_action when the model accepts
    # `per_cam`. Resolution order:
    #   1. explicit override (CLI `use_per_cam=...` or `EVALUATION.use_per_cam`)
    #   2. auto-detect from trained config's `data.train._target_`
    #      (a per-cam loader → true, anything else → false)
    #   3. default false
    explicit_use_per_cam = usr_args.get("use_per_cam", cfg.EVALUATION.get("use_per_cam"))
    if not _is_none_like(explicit_use_per_cam):
        use_per_cam = _parse_bool(explicit_use_per_cam)
        print(f"[FlexPi] use_per_cam={use_per_cam} (explicit override)")
    elif trained_cfg is not None:
        trained_target = str(trained_cfg.data.train.get("_target_", ""))
        # Two spellings mean per-cam RGB: today's single loader
        # (robot_video_dataset.RobotVideoDataset) and the pre-merge
        # RobotPerCam{,Depth}VideoDataset it replaced — every one of the 25
        # saved run configs carries the latter. The only loader that ever
        # emitted a composite `video` was the pre-merge RobotVideoDataset,
        # which no task config referenced and no checkpoint was trained with.
        use_per_cam = (
            trained_target.endswith("RobotVideoDataset")
            or "RobotPerCam" in trained_target
        )
        print(
            f"[FlexPi] use_per_cam={use_per_cam} "
            f"(auto-detected from trained data.train._target_={trained_target})"
        )
    else:
        use_per_cam = False
        print("[FlexPi] use_per_cam=False (default; no trained config and no override)")

    # Step-skipping (joint-variant models only; other models silently ignore).
    def _materialize_list(raw):
        """Convert raw value to plain list. Handles OmegaConf ListConfig, plain
        list/tuple, and stringified Python literals (e.g. "[[0.95, 4], [0.93, 2]]")
        as forwarded from the parent process via subprocess argv."""
        if raw is None or _is_none_like(raw):
            return None
        if isinstance(raw, str):
            import ast
            return list(ast.literal_eval(raw))
        try:
            return list(OmegaConf.to_container(raw, resolve=True))
        except (ValueError, TypeError):
            return list(raw)

    dynamic_step_skip = _parse_bool(
        usr_args.get("dynamic_step_skip", cfg.EVALUATION.get("dynamic_step_skip", False))
    )
    step_skip_thresholds = _materialize_list(
        usr_args.get("step_skip_thresholds", cfg.EVALUATION.get("step_skip_thresholds"))
    )
    step_skip_sim_sources = _materialize_list(
        usr_args.get("step_skip_sim_sources", cfg.EVALUATION.get("step_skip_sim_sources"))
    )
    step_skip_sim_aggregation = str(
        usr_args.get(
            "step_skip_sim_aggregation",
            cfg.EVALUATION.get("step_skip_sim_aggregation", "mean"),
        )
    )

    # Flex runtime overrides — live under EVALUATION.infer_joint_* (not
    # clobbered by the trained-config autoload, which only overwrites cfg.model
    # and cfg.data.train.processor). None => preserve trained default.
    def _parse_optional_bool_arg(raw):
        if raw is None or _is_none_like(raw):
            return None
        return _parse_bool(raw)

    infer_joint_video = _parse_optional_bool_arg(
        usr_args.get("infer_joint_video", cfg.EVALUATION.get("infer_joint_video"))
    )
    infer_joint_dino = _parse_optional_bool_arg(
        usr_args.get("infer_joint_dino", cfg.EVALUATION.get("infer_joint_dino"))
    )
    infer_joint_pointmap = _parse_optional_bool_arg(
        usr_args.get("infer_joint_pointmap", cfg.EVALUATION.get("infer_joint_pointmap"))
    )
    infer_present_video = _parse_optional_bool_arg(
        usr_args.get("infer_present_video", cfg.EVALUATION.get("infer_present_video"))
    )
    infer_present_dino = _parse_optional_bool_arg(
        usr_args.get("infer_present_dino", cfg.EVALUATION.get("infer_present_dino"))
    )
    infer_present_pointmap = _parse_optional_bool_arg(
        usr_args.get("infer_present_pointmap", cfg.EVALUATION.get("infer_present_pointmap"))
    )
    # Train/deploy alignment switches (see WorldActionRobotWinPolicy.__init__).
    action_dim_mask = _parse_optional_bool_arg(
        usr_args.get("action_dim_mask", cfg.EVALUATION.get("action_dim_mask"))
    )
    obs_train_parity = _parse_bool(
        usr_args.get("obs_train_parity", cfg.EVALUATION.get("obs_train_parity", False))
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
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
        offload_text_encoder=offload_text_encoder,
        torch_compile=torch_compile,
        torch_compile_mode=torch_compile_mode,
        torch_compile_scope=torch_compile_scope,
        torch_compile_backend=torch_compile_backend,
        compile_encoders=compile_encoders,
        quantization=quantization,
        attn_backend=attn_backend,
        trt_joint_engine_path=trt_joint_engine_path,
        trt_joint_free_video_blocks=trt_joint_free_video_blocks,
        trt_joint_prefill_split_engine_path=trt_joint_prefill_split_engine_path,
        trt_joint_decode_split_engine_path=trt_joint_decode_split_engine_path,
        solver=solver,
        glue_cache=glue_cache,
        encoder_cuda_graph=encoder_cuda_graph,
        joint_loop_cuda_graph=joint_loop_cuda_graph,
        use_per_cam=use_per_cam,
        dynamic_step_skip=dynamic_step_skip,
        step_skip_thresholds=step_skip_thresholds,
        step_skip_sim_sources=step_skip_sim_sources,
        step_skip_sim_aggregation=step_skip_sim_aggregation,
        infer_joint_video=infer_joint_video,
        infer_joint_dino=infer_joint_dino,
        infer_joint_pointmap=infer_joint_pointmap,
        infer_present_video=infer_present_video,
        infer_present_dino=infer_present_dino,
        infer_present_pointmap=infer_present_pointmap,
        action_dim_mask=action_dim_mask,
        obs_train_parity=obs_train_parity,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
