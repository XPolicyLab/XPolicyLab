"""FlexWAM policy and evaluation support; use the scripts/ entrypoints."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deploy.robot import add_robot_type_argument
from deploy.robot import camera_client_key
from deploy.robot import client_field_key
from deploy.robot import derive_router_display_order as _derive_router_display_order
from deploy.robot import get_robot
from deploy.robot import lookup_client_value
from deploy.robot import require_client_keys
from deploy.robot import resolve_robot_type
from deploy.robot.flexiv import STATE_DIM as FLEXIV_POSE6D_DIM
from deploy.robot.flexiv import integrate_delta_actions as _integrate_flexiv_pose6d_delta_actions
from deploy.robot.piper import CAMERA_ORDER
from deploy.robot.piper import ROUTER_DISPLAY_ORDER as _MULTIVIEW_ROUTER_DISPLAY_ORDER
from deploy.robot.piper import STATE_DIM as PIPER_STATE_DIM
from deploy.robot.piper import integrate_delta_actions as _integrate_joint_delta_actions
from flexwam.datasets.lerobot.processors.flexwam_processor import FlexWAMProcessor
from flexwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from flexwam.datasets.lerobot.transforms.image import pil_bilinear_resize
from flexwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from flexwam.utils.camera_metrics import configure_camera_metrics
from flexwam.utils.config_resolvers import register_default_resolvers


logger = logging.getLogger(__name__)

os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = os.environ.get("FLEXWAM_SKIP_DOWNLOAD", "true")




DEFAULT_TASK = "penbox"
_ROUTER_BAR_HEIGHT = 54


def _payload_to_serializable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _payload_to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_payload_to_serializable(item) for item in value]
    return value


def _annotate_camera_image(
    image: np.ndarray,
    *,
    camera_name: str,
    router_weight: float | None,
) -> Image.Image:
    image_pil = Image.fromarray(image, mode="RGB")
    canvas = Image.new(
        "RGB",
        (image_pil.width, image_pil.height + _ROUTER_BAR_HEIGHT),
        color=(0, 0, 0),
    )
    canvas.paste(image_pil, (0, _ROUTER_BAR_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    camera_label = f"{camera_name}: n/a" if router_weight is None else f"{camera_name}: {router_weight:.4f}"
    draw.text((6, 2), camera_label, fill=(255, 255, 0), font=font)
    return canvas


def _compose_multiview_router_image(
    received_images: dict[str, np.ndarray],
    router_weights: dict[str, float],
    *,
    display_order: tuple[str, ...] | None = None,
) -> np.ndarray | None:
    order = display_order or _MULTIVIEW_ROUTER_DISPLAY_ORDER
    annotated_views: list[np.ndarray] = []
    for camera_name in order:
        if camera_name not in received_images:
            continue
        annotated = _annotate_camera_image(
            received_images[camera_name],
            camera_name=camera_name,
            router_weight=router_weights.get(camera_name),
        )
        annotated_views.append(np.asarray(annotated, dtype=np.uint8))
    if not annotated_views:
        return None
    return np.concatenate(annotated_views, axis=1)


def _format_array(array: np.ndarray) -> str:
    array = np.asarray(array)
    return np.array2string(
        array,
        precision=4,
        floatmode="fixed",
        suppress_small=False,
        separator=", ",
        max_line_width=200,
    )


def _backbone_name(cfg: DictConfig) -> str:
    target = str(cfg.model.get("_target_", "")).rsplit(".", 1)[-1]
    if target == "create_lingbot_flexwam":
        return "lingbot"
    if target == "create_flexwam":
        return "wan"
    raise ValueError(
        f"Unsupported deploy model `_target_` {cfg.model.get('_target_')!r}. "
        "Expected create_lingbot_flexwam or create_flexwam."
    )


def _model_config_for_deploy(cfg: DictConfig) -> DictConfig:
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.skip_dit_load_from_pretrain = True
    if _backbone_name(cfg) == "wan":
        text_dim = int(model_cfg.video_dit_config.text_dim)
        if int(model_cfg.get("context_input_dim") or text_dim) != text_dim:
            raise ValueError("WAN prompt deployment requires native UMT5 context; adapted context must be supplied explicitly.")
        # Training uses cached UMT5; deployment receives raw prompt strings.
        model_cfg.load_text_encoder = True
        model_cfg.tokenizer_max_len = int(cfg.data.train.context_len)
    return model_cfg


def _resolve_model_file_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _resolve_task_name(task: str | None, task_config_path: str | None) -> str:
    if task:
        return str(task)
    if task_config_path:
        return Path(task_config_path).resolve().stem
    return DEFAULT_TASK


def _compose_task_cfg(task_name: str) -> DictConfig:
    from flexwam.config import compose_task
    return compose_task(task_name)


def _resolve_existing_path(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _resolve_dataset_stats_path(
    dataset_stats_path: str | None,
    checkpoint_path: Path,
) -> Path:
    if dataset_stats_path:
        return _resolve_existing_path(dataset_stats_path, "Dataset stats")

    candidate_paths = [
        checkpoint_path.parents[2] / "dataset_stats.json",
        checkpoint_path.parent / "dataset_stats.json",
        checkpoint_path.parent.parent / "dataset_stats.json",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            logger.info("Using inferred dataset stats path: %s", candidate)
            return candidate.resolve()

    raise FileNotFoundError(
        "Dataset stats path was not provided and could not be inferred from checkpoint path. "
        "Please pass --dataset-stats-path explicitly."
    )


def _to_rgb_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 0.0
            if max_value <= 1.0:
                array = np.clip(array * 255.0, 0.0, 255.0)
        array = np.asarray(array, dtype=np.uint8)
    return np.ascontiguousarray(array)


def _is_robodojo_cfg(cfg: DictConfig) -> bool:
    """RoboDojo is the only dataset whose live frames must pass through the stored raw size."""
    transforms = OmegaConf.select(cfg, "data.train.processor.action_state_transforms", default=None)
    if transforms is None:
        return False
    for item in OmegaConf.to_container(transforms, resolve=True):
        target = item.get("_target_", "") if isinstance(item, dict) else ""
        if str(target).endswith("BimanualPose6DDeltaTransform"):
            return True
    return False


def _shared_raw_image_hw(image_specs: list[dict], video_size_hw: tuple[int, int]) -> tuple[int, int] | None:
    """Return the shared camera raw size when it differs from the model input size.

    RoboDojo stores [3, 240, 320] and trains at [192, 224]. Other datasets resize once.
    """
    raw_sizes: list[tuple[int, int]] = []
    for spec in image_specs:
        raw = spec.get("raw_shape")
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return None
        height, width = int(raw[1]), int(raw[2])
        if height <= 0 or width <= 0:
            return None
        raw_sizes.append((height, width))
    if not raw_sizes or len(set(raw_sizes)) != 1:
        return None
    raw_hw = raw_sizes[0]
    if raw_hw == tuple(video_size_hw):
        return None
    return raw_hw


def _cv2_inter_linear_resize(images: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """Stretch ``[..., 3, H, W]`` uint8 frames with ``cv2.INTER_LINEAR``.

    This is the LeRobot converter's ``--resolution`` resize
    (``scripts/transform_lerobot_v21_format.py`` / ``v30``). Live frames that
    are not already the stored raw size go through it before the training
    PIL bilinear stretch to the model input.
    """
    import cv2

    height, width = int(size_hw[0]), int(size_hw[1])
    if images.dtype != torch.uint8:
        raise TypeError(f"_cv2_inter_linear_resize expects uint8 input, got {images.dtype}.")
    if images.ndim < 3 or int(images.shape[-3]) != 3:
        raise ValueError(f"_cv2_inter_linear_resize expects [...,3,H,W], got shape {tuple(images.shape)}.")
    frames = images.detach().cpu()
    lead_shape = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:]).permute(0, 2, 3, 1).contiguous().numpy()
    resized = np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in flat],
        axis=0,
    )
    out = torch.from_numpy(np.ascontiguousarray(resized)).permute(0, 3, 1, 2).reshape(*lead_shape, 3, height, width)
    return out.to(device=images.device)


def _extract_image(obs: dict[str, Any], *keys: str) -> np.ndarray | None:
    for key in keys:
        value = lookup_client_value(obs, key)
        if value is not None:
            return _to_rgb_uint8(value)
    return None


class FlexWAMPolicy:
    """Task-specific websocket policy adapter for FlexWAM on Piper towel."""

    def reset(self) -> None:
        self._prev_state = None
        self._request_index = 0

    def __init__(
        self,
        *,
        cfg: DictConfig,
        checkpoint_path: Path,
        dataset_stats_path: Path,
        device: str,
        action_horizon: int,
        num_inference_steps: int,
        sigma_shift: float | None,
        seed: int | None,
        default_prompt: str | None,
        prompt_template: str | None,
        log_full_io: bool,
        action_representation: str,
        delta_joint_actions: bool,
        robot_type: str = "piper",
        cuda_graph: bool = False,
    ) -> None:
        self.robot = get_robot(robot_type)
        self.robot_type = self.robot.name
        self.backbone = _backbone_name(cfg)
        model_cfg = _model_config_for_deploy(cfg)
        if self.backbone == "lingbot":
            self.lingbot_video_path = _resolve_model_file_path(str(model_cfg.model_path))
            model_cfg.model_path = str(self.lingbot_video_path)
        else:
            self.lingbot_video_path = None
        if model_cfg.get("action_dit_pretrained_path") is not None:
            action_dit_path = Path(str(model_cfg.action_dit_pretrained_path)).expanduser()
            if not action_dit_path.is_absolute():
                model_cfg.action_dit_pretrained_path = str((PROJECT_ROOT / action_dit_path).resolve())

        self.model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=device)
        configure_camera_metrics(self.model, cfg.data.train)
        checkpoint_payload = self.model.load_checkpoint(str(checkpoint_path))
        self._validate_router_checkpoint_match(checkpoint_payload, checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: FlexWAMProcessor = instantiate(cfg.data.train.processor).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.num_video_frames = int(cfg.data.train.num_frames)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = 1.0
        self.negative_prompt = ""
        self.rand_device = "cpu"
        self.tiled = False
        self.default_prompt = None if default_prompt is None else str(default_prompt)
        self.prompt_template = None if prompt_template is None else str(prompt_template)
        self.image_specs = list(OmegaConf.to_container(cfg.data.train.shape_meta.images, resolve=True))
        self.concat_multi_camera = None
        if "concat_multi_camera" in cfg.data.train and cfg.data.train.concat_multi_camera is not None:
            self.concat_multi_camera = str(cfg.data.train.concat_multi_camera)
        if len(self.image_specs) > 1 and self.concat_multi_camera != "independent":
            raise ValueError(
                "Multi-camera deployment requires `data.train.concat_multi_camera='independent'`, "
                f"got {self.concat_multi_camera!r}."
            )
        self.video_size_hw = tuple(int(x) for x in OmegaConf.to_container(cfg.data.train.video_size, resolve=True))
        self.preprocess_size_hw = (
            _shared_raw_image_hw(self.image_specs, self.video_size_hw) if _is_robodojo_cfg(cfg) else None
        )
        self.log_full_io = bool(log_full_io)
        self.state_key = self.processor.shape_meta["state"][0]["key"]
        self.action_key = self.processor.shape_meta["action"][0]["key"]
        self.action_representation = self.robot.resolve_action_representation(str(action_representation))
        self.delta_joint_actions = bool(delta_joint_actions)
        if self.delta_joint_actions and self.action_representation not in {
            "joint",
            "flexiv_pose6d",
            "dual_ee_rot6d",
        }:
            raise ValueError(
                "`--delta` is only valid with Piper joint, Flexiv pose6D, or dual_ee rot6D "
                "action representation."
            )
        expected_proprio_dims = (self.robot.state_dim,)
        expected_action_dim = self.robot.action_dim
        if int(self.processor.proprio_output_dim) not in expected_proprio_dims:
            raise ValueError(
                f"Policy action_representation={self.action_representation!r} expects proprio dim "
                f"{expected_proprio_dims}, got processor.proprio_output_dim={self.processor.proprio_output_dim}."
            )
        if int(self.processor.action_output_dim) != expected_action_dim:
            raise ValueError(
                f"{self.robot_type} deploy expects denormalized policy action dim {expected_action_dim}, "
                f"got {self.processor.action_output_dim}."
            )
        self._prev_state: np.ndarray | None = None
        self._request_index = 0

        self.lingbot_text_encoder = None
        if self.lingbot_video_path is not None:
            from flexwam.models.backbone.lingbot_video.text_encoder import LingBotTextEncoderAdapter, resolve_project_relative_path

            cache_dir = getattr(cfg.data.train, "text_embedding_cache_dir", None)
            resolved_cache_dir = (
                resolve_project_relative_path(str(cache_dir), PROJECT_ROOT)
                if cache_dir is not None and str(cache_dir).strip()
                else None
            )
            encoder_id = str(getattr(cfg.data.train, "text_embedding_cache_encoder_id", "lingbotqwen3vl"))
            context_len = int(getattr(cfg.data.train, "context_len", 128))
            self.lingbot_text_encoder = LingBotTextEncoderAdapter(
                model_path=self.lingbot_video_path,
                cache_dir=resolved_cache_dir,
                context_len=context_len,
                encoder_id=encoder_id,
                device=str(device),
                dtype=torch.bfloat16,
            )

        self.camera_order = self._resolve_deploy_camera_order()
        self.router_display_order = self.robot.router_display_order(self.camera_order)
        self._enable_action_cuda_graph(bool(cuda_graph))
        self._warmup_prompt_cache()

    def _enable_action_cuda_graph(self, requested: bool) -> None:
        enabled = bool(requested) or bool(getattr(self.model, "use_cuda_graph", False))
        if not enabled:
            self.model.use_cuda_graph = False
            return
        device = getattr(self.model, "device", None)
        device_type = str(device.type) if isinstance(device, torch.device) else str(device)
        if not str(device_type).startswith("cuda") or not torch.cuda.is_available():
            logger.warning("CUDA graph requested but CUDA is unavailable; ignoring.")
            self.model.use_cuda_graph = False
            return
        self.model.use_cuda_graph = True
        logger.info("Action denoise CUDA graph enabled (captured on the first inference request).")

    def _warmup_prompt_cache(self) -> None:
        if self.default_prompt is None:
            return
        prompt = str(self.default_prompt)
        if self.prompt_template is not None:
            prompt = self.prompt_template.format(task=prompt)
        if self.lingbot_text_encoder is not None:
            self.lingbot_text_encoder.encode_prompt(prompt)
            logger.info("Warmed LingBot text embedding cache for the default prompt.")
            return
        if getattr(self.model, "text_encoder", None) is None:
            return
        self.model.encode_prompt(prompt)
        logger.info("Warmed UMT5 prompt embedding cache for the default prompt.")

    def _resolve_deploy_camera_order(self) -> tuple[str, ...]:
        return self.robot.resolve_camera_order(
            view_names=getattr(self.model, "view_names", None),
            image_specs=self.image_specs,
        )

    def _validate_router_checkpoint_match(self, checkpoint_payload: dict[str, Any], checkpoint_path: Path) -> None:
        router_expectations = (
            ("action_camera_router", "action_camera_router"),
        )
        for checkpoint_key, attr_name in router_expectations:
            has_module = getattr(self.model, attr_name, None) is not None
            has_weights = checkpoint_key in checkpoint_payload
            if has_module and not has_weights:
                raise ValueError(
                    f"Task config enables `{attr_name}`, but checkpoint has no `{checkpoint_key}` weights: "
                    f"{checkpoint_path}. Use the matching task config/checkpoint pair."
                )
            if has_weights and not has_module:
                raise ValueError(
                    f"Checkpoint contains `{checkpoint_key}` weights, but task config disables `{attr_name}`: "
                    f"{checkpoint_path}. Use the matching router-enabled task config for deployment."
                )

    def _robot_profile(self):
        robot = getattr(self, "robot", None)
        if robot is not None:
            return robot
        return get_robot(getattr(self, "robot_type", "piper"))

    def _save_multiview_router_images(
        self,
        *,
        request_index: int,
        received_images: dict[str, np.ndarray],
        router_weights: dict[str, float] | None,
    ) -> None:
        del request_index, received_images, router_weights

    def _extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        state_key = client_field_key(self._robot_profile(), "state")
        value = lookup_client_value(obs, state_key)
        if value is None:
            raise KeyError(f"Observation is missing mapped state key {state_key!r}.")
        state = np.asarray(value, dtype=np.float32).ravel()

        expected_dim = self._robot_profile().state_dim
        if state.shape != (expected_dim,):
            raise ValueError(f"Expected {expected_dim}-dim {self.robot_type} state, got shape {state.shape}")
        return state

    def _deploy_camera_order(self) -> tuple[str, ...]:
        order = getattr(self, "camera_order", None)
        if order:
            return tuple(order)
        robot = self._robot_profile()
        robot_order = getattr(robot, "camera_order", None)
        if robot_order:
            return tuple(robot_order)
        return tuple(require_client_keys(robot)["cameras"])

    def _extract_camera_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        robot = self._robot_profile()
        camera_order = self._deploy_camera_order()
        camera_images: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for camera_name in camera_order:
            image = _extract_image(obs, camera_client_key(robot, camera_name))
            if image is None:
                missing.append(f"{camera_name}->{camera_client_key(robot, camera_name)}")
            else:
                camera_images[camera_name] = image
        if missing:
            raise KeyError(
                f"Observation must provide mapped camera keys for {list(camera_order)}. Missing: {missing}"
            )
        return camera_images

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        batch = {"state": {self.state_key: torch.as_tensor(state.copy(), dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][self.state_key]

    def _denormalize_actions(self, actions: torch.Tensor) -> np.ndarray:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        normalizer = self.processor.normalizer.normalizers["action"][self.action_key]
        denorm = normalizer.backward(actions.to(device="cpu", dtype=torch.float32))
        return denorm.numpy()

    def _extract_prompt(self, obs: dict[str, Any]) -> str:
        prompt_key = client_field_key(self._robot_profile(), "prompt")
        prompt = lookup_client_value(obs, prompt_key)
        if prompt is None:
            prompt = self.default_prompt
        if prompt is None:
            raise KeyError(
                f"Observation is missing mapped prompt key {prompt_key!r} and no `--default-prompt` was provided."
            )
        prompt = str(prompt)
        if not prompt:
            raise ValueError("Prompt must not be empty.")
        if self.prompt_template is not None:
            prompt = self.prompt_template.format(task=prompt)
        return prompt

    def _build_input_image(
        self,
        obs: dict[str, Any],
    ) -> tuple[torch.Tensor, np.ndarray, dict[str, np.ndarray], str | None]:
        composite = _extract_image(obs, "observation/image", "image", "input_image")
        if composite is not None:
            raise ValueError(
                "Deployment requires independent camera images. "
                "Do not send a pre-stitched `observation/image`."
            )
        camera_images = self._extract_camera_images(obs)
        received_images = {key: value.copy() for key, value in camera_images.items()}
        camera_order = self._deploy_camera_order()
        views = np.stack([camera_images[name] for name in camera_order], axis=0)
        # RoboDojo files are 240x320 from cv2.INTER_LINEAR (--resolution). Live
        # frames that are not already that size take the same resize, then the
        # training PIL BILINEAR stretch to video_size. A step is skipped when
        # the frame is already that size. Other tasks resize once with PIL.
        frames = torch.from_numpy(views).permute(0, 3, 1, 2).contiguous()
        preprocess_size = getattr(self, "preprocess_size_hw", None)
        if preprocess_size is not None and tuple(frames.shape[-2:]) != tuple(preprocess_size):
            frames = _cv2_inter_linear_resize(frames, preprocess_size)
        if tuple(frames.shape[-2:]) != tuple(self.video_size_hw):
            frames = pil_bilinear_resize(frames, self.video_size_hw)
        preview = np.concatenate(list(frames.permute(0, 2, 3, 1).numpy()), axis=1)
        tensor = transforms_F.normalize(frames.to(torch.float32) / 255.0, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        tensor = tensor.to(device=self.model.device, dtype=self.model.torch_dtype)
        return tensor, preview, received_images, None

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        preprocess_t0 = time.perf_counter()
        prompt = self._extract_prompt(obs)
        raw_state = self._extract_state(obs)
        image_t0 = time.perf_counter()
        image_tensor, _, received_images, _ = self._build_input_image(obs)
        image_preprocess_ms = (time.perf_counter() - image_t0) * 1000.0
        proprio = self._normalize_state(raw_state)
        text_encode_ms = 0.0
        lingbot_context = None
        lingbot_context_mask = None
        if self.lingbot_text_encoder is not None:
            text_t0 = time.perf_counter()
            lingbot_context, lingbot_context_mask = self.lingbot_text_encoder.encode_prompt(prompt)
            text_encode_ms = (time.perf_counter() - text_t0) * 1000.0
        preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000.0

        infer_t0 = time.perf_counter()
        infer_kwargs = {
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
        if self.lingbot_text_encoder is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = lingbot_context.to(device=self.model.device, dtype=self.model.torch_dtype)
            infer_kwargs["context_mask"] = lingbot_context_mask.to(device=self.model.device, dtype=torch.bool)
        else:
            infer_kwargs["prompt"] = prompt
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = self.num_video_frames
        if (
            getattr(self, "collect_action_attention", False)
            and "return_action_attention_stats" in inspect.signature(self.model.infer_action).parameters
        ):
            infer_kwargs["return_action_attention_stats"] = True
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        model_ms = (time.perf_counter() - infer_t0) * 1000.0

        post_t0 = time.perf_counter()
        model_actions = self._denormalize_actions(pred["action"])[0].astype(np.float32, copy=False)
        robot = self._robot_profile()
        if self.delta_joint_actions:
            actions = robot.integrate_delta(raw_state, model_actions)
        else:
            actions = model_actions
        action_log_label = robot.action_log_label(delta=self.delta_joint_actions)
        summary = robot.action_summary(raw_state, actions)
        router_weights = None
        if "router_weights" in pred:
            router_array = torch.as_tensor(pred["router_weights"]).detach().to(device="cpu", dtype=torch.float32).numpy()
            if router_array.ndim == 2:
                router_array = router_array[0]
            if router_array.ndim == 1:
                camera_order = self._deploy_camera_order()
                router_weights = {
                    camera_name: float(router_array[idx])
                    for idx, camera_name in enumerate(camera_order[: router_array.shape[0]])
                }
                logger.info(
                    "Action camera router weights | %s",
                    " ".join(f"{name}={weight:.4f}" for name, weight in router_weights.items()),
                )
        postprocess_ms = (time.perf_counter() - post_t0) * 1000.0
        logger.info(
            "Policy IO summary | representation=%s | output_layout=%s | %s",
            self.action_representation,
            action_log_label,
            summary,
        )
        prev_state = self._prev_state
        self._prev_state = raw_state.copy()
        state_delta = None if prev_state is None else raw_state - prev_state
        if self.delta_joint_actions:
            action_delta = model_actions
        else:
            action_delta = actions - raw_state[None, :]
        if self.log_full_io:
            logger.info(
                "Policy IO full\nprompt=%s\nstate=%s\nstate_delta=%s\nactions=\n%s\nmodel_actions_or_delta=\n%s",
                prompt,
                _format_array(raw_state),
                "init" if state_delta is None else _format_array(state_delta),
                _format_array(actions),
                _format_array(action_delta),
            )

        request_index = self._request_index
        self._request_index += 1
        timings = {
            "preprocess_ms": preprocess_ms,
            "image_preprocess_ms": image_preprocess_ms,
            "text_encode_ms": text_encode_ms,
            "model_ms": model_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": preprocess_ms + model_ms + postprocess_ms,
        }
        infer_timing_ms = pred.get("infer_timing_ms")
        denoise_step_ms = None
        if isinstance(infer_timing_ms, dict):
            for key, value in infer_timing_ms.items():
                if key == "denoise_step_ms" and isinstance(value, (list, tuple)):
                    denoise_step_ms = [float(item) for item in value]
                    timings[key] = denoise_step_ms
                elif isinstance(value, (int, float)):
                    timings[str(key)] = float(value)
        logger.info(
            "Timing | total=%.1fms | preprocess=%.1fms (image=%.1f text=%.1f) | "
            "model=%.1fms (text=%.1f encode=%.1f prefill=%.1f denoise=%.1f) | post=%.1fms",
            timings["total_ms"],
            preprocess_ms,
            image_preprocess_ms,
            text_encode_ms,
            model_ms,
            float(timings.get("text_ms", 0.0)),
            float(timings.get("encode_ms", 0.0)),
            float(timings.get("prefill_ms", 0.0)),
            float(timings.get("denoise_ms", 0.0)),
            postprocess_ms,
        )
        if denoise_step_ms:
            logger.info(
                "Timing denoise steps | %s",
                " ".join(f"{idx}={ms:.1f}ms" for idx, ms in enumerate(denoise_step_ms)),
            )
        self._save_multiview_router_images(
            request_index=request_index,
            received_images=received_images,
            router_weights=router_weights,
        )

        result = {
            client_field_key(robot, "action"): actions,
            "action_representation": self.action_representation,
            "policy_timing": dict(timings),
        }
        if router_weights is not None:
            result["router_weights"] = router_weights
        if "action_attention_heatmaps" in pred:
            result["action_attention_heatmaps"] = _payload_to_serializable(pred["action_attention_heatmaps"])
        return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve FlexWAM as an OpenPI-compatible websocket policy for Piper towel deployment.",
    )
    parser.add_argument("--task", default=None, help="Hydra task name under configs/task.")
    parser.add_argument(
        "--task-config-path",
        default=str(PROJECT_ROOT / "configs" / "task" / f"{DEFAULT_TASK}.yaml"),
        help="Optional task config path; only its stem is used for Hydra compose.",
    )
    parser.add_argument("--checkpoint-path", required=True, help="Path to the FlexWAM checkpoint.")
    parser.add_argument(
        "--dataset-stats-path",
        default=None,
        help="Path to the dataset_stats.json file. "
        "If omitted, the server will try to infer it from the checkpoint run directory.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-horizon", default=None, type=int)
    parser.add_argument("--num-inference-steps", default=None, type=int)
    parser.add_argument("--sigma-shift", default=None, type=float)
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional diffusion seed for reproducible sampling. Omit to use random noise each request.",
    )
    parser.add_argument("--default-prompt", default="fold the towel")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT,
        help="Wrap the incoming task string before inference. "
        "Use an empty string to send the raw prompt unchanged.",
    )
    parser.add_argument(
        "--log-full-io",
        action="store_true",
        help="Log full prompt/state/action tensors for every inference request.",
    )
    parser.add_argument(
        "--action-representation",
        default="auto",
        choices=("auto", "joint"),
        help=(
            "Policy low-dimensional action semantics. `auto` uses the selected robot profile default."
        ),
    )
    parser.add_argument(
        "--delta",
        action="store_true",
        help=(
            "Interpret denormalized actions as deltas and integrate them with the current robot state. "
            "The selected robot profile defines how deltas are converted to absolute targets."
        ),
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help=(
            "Capture the action denoise step as a CUDA graph after warmup. "
            "Shapes must stay fixed (same cameras, horizon, and inference steps). "
            "Can also be set via model.use_cuda_graph."
        ),
    )
    add_robot_type_argument(parser)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    register_default_resolvers()

    task_name = _resolve_task_name(args.task, args.task_config_path)
    cfg = _compose_task_cfg(task_name)

    checkpoint_path = _resolve_existing_path(args.checkpoint_path, "Checkpoint")
    dataset_stats_path = _resolve_dataset_stats_path(args.dataset_stats_path, checkpoint_path)
    action_horizon = (
        int(args.action_horizon)
        if args.action_horizon is not None
        else int(cfg.data.train.num_frames) - 1
    )
    if action_horizon <= 0:
        raise ValueError(f"Invalid action horizon: {action_horizon}")
    num_inference_steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(getattr(cfg, "eval_num_inference_steps", 10))
    )
    if num_inference_steps <= 0:
        raise ValueError(f"Invalid num_inference_steps: {num_inference_steps}")
    prompt_template = None if args.prompt_template == "" else str(args.prompt_template)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to CPU.")
        args.device = "cpu"

    action_representation = str(args.action_representation)

    policy = FlexWAMPolicy(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        device=str(args.device),
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        default_prompt=args.default_prompt,
        prompt_template=prompt_template,
        log_full_io=bool(args.log_full_io),
        action_representation=action_representation,
        delta_joint_actions=bool(args.delta),
        robot_type=resolve_robot_type(args),
        cuda_graph=bool(args.cuda_graph),
    )

    state_layout, output_layout = policy.robot.layouts()

    metadata = {
        "policy_name": policy.robot.policy_name(backbone=policy.backbone),
        "robot_type": policy.robot_type,
        "backbone": policy.backbone,
        "task": task_name,
        "checkpoint_path": str(checkpoint_path),
        "dataset_stats_path": str(dataset_stats_path),
        "action_horizon": action_horizon,
        "num_video_frames": int(cfg.data.train.num_frames),
        "state_dim": int(policy.processor.proprio_output_dim),
        "action_dim": int(policy.processor.action_output_dim),
        "action_representation": policy.action_representation,
        "delta_joint_actions": bool(policy.delta_joint_actions),
        "input_image_shape": [len(policy.camera_order), 3, int(policy.video_size_hw[0]), int(policy.video_size_hw[1])],
        "camera_order": list(policy.camera_order),
        "prompt_template": prompt_template,
        "state_layout": state_layout,
        "output_layout": output_layout,
        "client_keys": require_client_keys(policy.robot),
    }

    logger.info(
        "Starting %s websocket server on %s:%d | task=%s | ckpt=%s | model_base=%s",
        policy.backbone,
        args.host,
        args.port,
        task_name,
        checkpoint_path,
        os.environ.get("FLEXWAM_MODEL_BASE_PATH")
        or os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
        or "auto(/path/to/fastwam, ./checkpoints)",
    )

    from deploy.server import TimedWebsocketPolicyServer
    server = TimedWebsocketPolicyServer(
        policy=policy,
        host=str(args.host),
        port=int(args.port),
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"flexwam_piper_server_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    main()
