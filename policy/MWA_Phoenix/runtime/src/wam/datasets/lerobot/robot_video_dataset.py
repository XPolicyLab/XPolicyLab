import hashlib
import bisect
import math
import os
from typing import Optional
import numpy as np
import traceback
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transforms_F
from torch.utils.data._utils.collate import default_collate

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from wam.utils.logging_config import get_logger
from wam.utils import misc
logger = get_logger(__name__)


def _is_main_process_without_init() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID")
    if rank is None:
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return True


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_hz: float | None = None,
        action_video_freq_ratio: int = 1,
        memory_video_anchor_size: int = 1,
        memory_recent_frame_offset: int | None = None,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "latent_horizontal", "robotwin", or None
        single_canvas: bool = False,
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        load_episode_stats: bool = True,
        check_local_files: bool = True,
        image_color_jitter: Optional[dict] = None,
        image_camera_jitter: Optional[dict] = None,
        image_sensor_noise: Optional[dict] = None,
        proprio_noise_std: float = 0.0,
        episode_selection: Optional[dict] = None,
        return_vlm_current_images: bool = False,
        sync_vlm_color_jitter: bool = False,
        sync_vlm_camera_jitter: bool = False,
    ):
        shape_meta_dict = OmegaConf.to_container(shape_meta, resolve=True) if isinstance(shape_meta, DictConfig) else shape_meta
        self.dataset_dirs = [str(path) for path in dataset_dirs]
        self.shape_meta = shape_meta_dict
        self.video_view_names = [
            str(item.get("key", f"view_{idx}"))
            for idx, item in enumerate(shape_meta_dict.get("images", []))
            if isinstance(item, dict)
        ]

        self.is_training_set = bool(is_training_set)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        current_action_size = int(num_frames) - 1
        self.memory_video_anchor_size = max(0, int(memory_video_anchor_size))
        self.memory_recent_frame_offset = max(
            1,
            int(
                current_action_size
                if memory_recent_frame_offset is None
                else memory_recent_frame_offset
            ),
        )
        self.past_action_size = 0
        action_window_size = current_action_size
        self.past_obs_size = 0
        obs_window_size = int(num_frames)
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta_dict,
            obs_size=obs_window_size,
            action_size=action_window_size,
            past_action_size=self.past_action_size,
            past_obs_size=self.past_obs_size,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            load_episode_stats=load_episode_stats,
            check_local_files=check_local_files,
            episode_selection=(
                OmegaConf.to_container(episode_selection, resolve=True)
                if isinstance(episode_selection, DictConfig)
                else episode_selection
            ),
        )
        if action_hz is not None:
            actual_hz = self.lerobot_dataset.fps / float(global_sample_stride)
            if not math.isfinite(float(action_hz)) or not math.isclose(float(action_hz), actual_hz, rel_tol=1e-6):
                raise ValueError(f"Configured action_hz={action_hz} differs from dataset cadence {actual_hz}")
        self._episode_starts = [int(x) for x in self.lerobot_dataset.episode_data_index["from"].tolist()]
        self._episode_ends = [int(x) for x in self.lerobot_dataset.episode_data_index["to"].tolist()]
        self.num_frames = num_frames
        self.action_video_freq_ratio = int(action_video_freq_ratio)

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.single_canvas = bool(single_canvas)
        if self.single_canvas and self.concat_multi_camera == "latent_horizontal":
            raise ValueError(
                "single_canvas=true requires pixel-space camera packing; "
                "use concat_multi_camera='horizontal' instead of 'latent_horizontal'."
            )
        self.override_instruction = override_instruction
        self.image_color_jitter = self._parse_image_color_jitter(image_color_jitter)
        # VLM stays on clean frames unless explicitly opted in; frozen Qwen
        # features degrade under photometric noise more than the video DiT.
        self.apply_color_jitter_to_vlm = self._parse_apply_color_jitter_to_vlm(
            image_color_jitter
        )
        self.image_camera_jitter = self._parse_image_camera_jitter(image_camera_jitter)
        self.image_sensor_noise = self._parse_image_sensor_noise(image_sensor_noise)
        self.proprio_noise_std = max(0.0, float(proprio_noise_std or 0.0))
        self.return_vlm_current_images = bool(return_vlm_current_images)
        self.sync_vlm_color_jitter = bool(sync_vlm_color_jitter)
        self.sync_vlm_camera_jitter = bool(sync_vlm_camera_jitter)
        for augmentation in (self.image_camera_jitter, self.image_sensor_noise):
            if augmentation and augmentation.get("camera_keys") is not None:
                unknown = set(augmentation["camera_keys"]) - set(self.video_view_names)
                if unknown:
                    raise ValueError(f"Unknown augmentation cameras: {sorted(unknown)}")

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if hasattr(processor, "num_obs_steps"):
                processor.num_obs_steps = obs_window_size
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        logger.info("Calculating dataset stats for normalization...")
                        dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                        work_dir = misc.get_work_dir()
                        save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                    else:
                        dataset_stats = None
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
                else:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if _is_main_process_without_init():
                    work_dir = misc.get_work_dir()
                    stats_path = os.path.join(work_dir, "dataset_stats.json")
                    if os.path.abspath(pretrained_norm_stats) != os.path.abspath(stats_path):
                        save_dataset_stats_to_json(dataset_stats, stats_path)

            processor.set_normalizer_from_stats(dataset_stats)
            if self.return_vlm_current_images:
                processor.vlm_current_image_position = self.memory_video_anchor_size + 1
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    @property
    def sample_episode_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self._episode_starts, self._episode_ends))

    @staticmethod
    def _parse_apply_color_jitter_to_vlm(config: Optional[dict]) -> bool:
        if config is None:
            return False
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        if not bool(config.get("enabled", True)):
            return False
        return bool(config.get("apply_to_vlm", False))

    @staticmethod
    def _parse_image_color_jitter(config: Optional[dict]) -> Optional[dict[str, float]]:
        if config is None:
            return None
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        enabled = bool(config.get("enabled", True))
        if not enabled:
            return None
        jitter = {
            "brightness": max(0.0, float(config.get("brightness", 0.0) or 0.0)),
            "contrast": max(0.0, float(config.get("contrast", 0.0) or 0.0)),
            "saturation": max(0.0, float(config.get("saturation", 0.0) or 0.0)),
            "gamma": max(0.0, float(config.get("gamma", 0.0) or 0.0)),
        }
        return jitter if any(value > 0.0 for value in jitter.values()) else None

    def _sample_color_jitter_params(self) -> Optional[dict[str, float]]:
        if not self.is_training_set or self.image_color_jitter is None:
            return None

        def _sample_factor(amount: float) -> float:
            if amount <= 0.0:
                return 1.0
            low = max(0.0, 1.0 - amount)
            high = 1.0 + amount
            return float(torch.empty(()).uniform_(low, high).item())

        return {
            "brightness": _sample_factor(self.image_color_jitter["brightness"]),
            "contrast": _sample_factor(self.image_color_jitter["contrast"]),
            "saturation": _sample_factor(self.image_color_jitter["saturation"]),
            "gamma": _sample_factor(self.image_color_jitter["gamma"]),
        }

    @staticmethod
    def _parse_image_camera_jitter(config: Optional[dict]) -> Optional[dict[str, float]]:
        if config is None:
            return None
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        if not bool(config.get("enabled", True)):
            return None
        scale_min = float(config.get("scale_min", 1.0) or 1.0)
        scale_max = float(config.get("scale_max", 1.0) or 1.0)
        scale_min = min(max(scale_min, 0.5), 1.0)
        scale_max = min(max(scale_max, scale_min), 1.0)
        jitter = {
            "prob": min(max(float(config.get("prob", 1.0) or 0.0), 0.0), 1.0),
            "scale_min": scale_min,
            "scale_max": scale_max,
            "translate": max(0.0, float(config.get("translate", 0.0) or 0.0)),
            "rotation_degrees": max(0.0, float(config.get("rotation_degrees", 0.0))),
            "random_crop": bool(config.get("random_crop", False)),
            "apply_to_vlm": bool(config.get("apply_to_vlm", False)),
            "camera_keys": config.get("camera_keys", config.get("views")),
        }
        if jitter["prob"] <= 0.0:
            return None
        if jitter["scale_min"] >= 1.0 and jitter["translate"] <= 0.0 and jitter["rotation_degrees"] <= 0.0:
            return None
        return jitter

    def _sample_camera_jitter_params(self) -> Optional[dict[str, float]]:
        if not self.is_training_set or self.image_camera_jitter is None:
            return None
        if float(torch.rand(()).item()) >= self.image_camera_jitter["prob"]:
            return None
        if self.image_camera_jitter["random_crop"]:
            return {
                "scale": float(torch.empty(()).uniform_(self.image_camera_jitter["scale_min"], self.image_camera_jitter["scale_max"]).item()),
                "offset_x": float(torch.rand(()).item()),
                "offset_y": float(torch.rand(()).item()),
                "camera_keys": self.image_camera_jitter["camera_keys"],
            }
        return {
            "scale": float(
                torch.empty(()).uniform_(
                    self.image_camera_jitter["scale_min"],
                    self.image_camera_jitter["scale_max"],
                ).item()
            ),
            "shift_x": float(
                torch.empty(()).uniform_(
                    -self.image_camera_jitter["translate"],
                    self.image_camera_jitter["translate"],
                ).item()
            ),
            "shift_y": float(
                torch.empty(()).uniform_(
                    -self.image_camera_jitter["translate"],
                    self.image_camera_jitter["translate"],
                ).item()
            ),
            "angle": float(torch.empty(()).uniform_(
                -self.image_camera_jitter["rotation_degrees"],
                self.image_camera_jitter["rotation_degrees"],
            ).item()) if self.image_camera_jitter["rotation_degrees"] > 0 else 0.,
            "camera_keys": self.image_camera_jitter["camera_keys"],
        }

    @staticmethod
    def _parse_image_sensor_noise(config: Optional[dict]) -> Optional[dict[str, float]]:
        if config is None:
            return None
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        if not bool(config.get("enabled", True)):
            return None
        noise = {
            "gaussian_prob": min(max(float(config.get("gaussian_prob", 0.0) or 0.0), 0.0), 1.0),
            "gaussian_std": max(0.0, float(config.get("gaussian_std", 0.0) or 0.0)),
            "blur_prob": min(max(float(config.get("blur_prob", 0.0) or 0.0), 0.0), 1.0),
            "blur_kernel_size": max(1, int(config.get("blur_kernel_size", 3) or 3)),
            "blur_sigma": max(0.0, float(config.get("blur_sigma", 0.0) or 0.0)),
            "downsample_prob": min(max(float(config.get("downsample_prob", 0.0) or 0.0), 0.0), 1.0),
            "downsample_scale_min": min(max(float(config.get("downsample_scale_min", 1.0) or 1.0), 0.25), 1.0),
            "camera_keys": config.get("camera_keys", config.get("views")),
        }
        if noise["blur_kernel_size"] % 2 == 0:
            noise["blur_kernel_size"] += 1
        has_gaussian = noise["gaussian_prob"] > 0.0 and noise["gaussian_std"] > 0.0
        has_blur = noise["blur_prob"] > 0.0 and noise["blur_sigma"] > 0.0
        has_downsample = noise["downsample_prob"] > 0.0 and noise["downsample_scale_min"] < 1.0
        return noise if has_gaussian or has_blur or has_downsample else None

    def _sample_sensor_noise_params(self) -> Optional[dict[str, float]]:
        if not self.is_training_set or self.image_sensor_noise is None:
            return None
        gaussian_std = (
            self.image_sensor_noise["gaussian_std"]
            if float(torch.rand(()).item()) < self.image_sensor_noise["gaussian_prob"]
            else 0.0
        )
        blur_sigma = (
            self.image_sensor_noise["blur_sigma"]
            if float(torch.rand(()).item()) < self.image_sensor_noise["blur_prob"]
            else 0.0
        )
        if float(torch.rand(()).item()) < self.image_sensor_noise["downsample_prob"]:
            downsample_scale = float(
                torch.empty(()).uniform_(
                    self.image_sensor_noise["downsample_scale_min"],
                    1.0,
                ).item()
            )
        else:
            downsample_scale = 1.0
        if gaussian_std <= 0.0 and blur_sigma <= 0.0 and downsample_scale >= 1.0:
            return None
        return {
            "gaussian_std": gaussian_std,
            "blur_kernel_size": self.image_sensor_noise["blur_kernel_size"],
            "blur_sigma": blur_sigma,
            "downsample_scale": downsample_scale,
            "camera_keys": self.image_sensor_noise["camera_keys"],
        }

    def _apply_color_jitter_to_video(
        self,
        video: torch.Tensor,
        params: Optional[dict[str, float]],
    ) -> torch.Tensor:
        if params is None:
            return video
        if video.ndim not in (4, 5):
            raise ValueError(f"Expected video shape [T,C,H,W] or [N,T,C,H,W], got {tuple(video.shape)}")
        if not torch.is_floating_point(video):
            raise TypeError("image_color_jitter expects float pixel_values after processor ToTensor.")
        original_shape = tuple(video.shape)
        flat = video.reshape(-1, *original_shape[-3:]).clone()
        brightness = float(params["brightness"])
        contrast = float(params["contrast"])
        saturation = float(params["saturation"])
        gamma = float(params.get("gamma", 1.0))
        if brightness != 1.0:
            flat.mul_(brightness)
        if contrast != 1.0:
            mean = flat.mean(dim=(-3, -2, -1), keepdim=True)
            flat.sub_(mean).mul_(contrast).add_(mean)
        if saturation != 1.0 and int(flat.shape[-3]) == 3:
            weights = flat.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
            gray = (flat * weights).sum(dim=1, keepdim=True)
            flat.sub_(gray).mul_(saturation).add_(gray)
        if gamma != 1.0:
            flat.clamp_(0.0, 1.0).pow_(gamma)
        flat.clamp_(0.0, 1.0)
        return flat.reshape(original_shape)

    def _apply_color_jitter_to_vlm_images(
        self,
        images: torch.Tensor,
        params: Optional[dict[str, float]],
    ) -> torch.Tensor:
        """Apply the *same* sampled color jitter to the VLM current frames.

        `vlm_current_images` is [V,C,H,W] and, unlike the video canvas, may be
        raw uint8 (the default path clones frames before ToTensor).  The Qwen
        encoder divides by 255 itself, so the dtype/range contract is kept:
        uint8 in -> uint8 out, float [0,1] in -> float [0,1] out.
        """
        if params is None:
            return images
        if images.ndim != 4:
            raise ValueError(
                f"Expected vlm_current_images shape [V,C,H,W], got {tuple(images.shape)}"
            )
        if images.dtype == torch.uint8:
            jittered = self._apply_color_jitter_to_video(
                images.to(torch.float32).div_(255.0), params
            )
            return jittered.mul_(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)
        return self._apply_color_jitter_to_video(images, params)

    def _apply_camera_jitter_to_video(
        self,
        video: torch.Tensor,
        params: Optional[dict[str, float]],
    ) -> torch.Tensor:
        if params is None:
            return video
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [N,T,C,H,W], got {tuple(video.shape)}")
        if not torch.is_floating_point(video):
            raise TypeError("image_camera_jitter expects float pixel_values after processor ToTensor.")
        num_cameras, _, _, height, width = video.shape
        scale = min(max(float(params["scale"]), 0.5), 1.0)
        crop_h = max(1, min(height, int(round(height * scale))))
        crop_w = max(1, min(width, int(round(width * scale))))
        max_top = max(0, height - crop_h)
        max_left = max(0, width - crop_w)
        center_top = max_top // 2
        center_left = max_left // 2
        if "offset_x" in params:
            top = int(round(float(params["offset_y"]) * max_top))
            left = int(round(float(params["offset_x"]) * max_left))
        else:
            shift_y = int(round(float(params["shift_y"]) * height))
            shift_x = int(round(float(params["shift_x"]) * width))
            top = max(0, min(max_top, center_top + shift_y))
            left = max(0, min(max_left, center_left + shift_x))
        angle = float(params.get("angle", 0.))
        if crop_h == height and crop_w == width and top == 0 and left == 0 and angle == 0:
            return video
        views = []
        for cam_idx in range(num_cameras):
            view = video[cam_idx]
            if self._camera_selected(cam_idx, params):
                view = transforms_F.resized_crop(
                    view,
                    top=top,
                    left=left,
                    height=crop_h,
                    width=crop_w,
                    size=[height, width],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                if angle != 0:
                    view = transforms_F.rotate(view, angle,
                        interpolation=transforms_F.InterpolationMode.BILINEAR)
                    # Keep an inscribed rectangle with the original aspect
                    # ratio. An extra pixel margin excludes interpolated fill
                    # pixels; resizing this valid crop introduces no black rim.
                    radians = math.radians(abs(angle))
                    safe_scale = 1. / (abs(math.cos(radians)) + abs(math.sin(radians))
                                       * max(width / height, height / width))
                    inner_h = max(1, math.floor(height * safe_scale) - 2)
                    inner_w = max(1, math.floor(width * safe_scale) - 2)
                    view = transforms_F.resized_crop(view,
                        top=(height - inner_h) // 2, left=(width - inner_w) // 2,
                        height=inner_h, width=inner_w, size=[height, width],
                        interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
            views.append(view)
        return torch.stack(views)

    def _apply_sensor_noise_to_video(
        self,
        video: torch.Tensor,
        params: Optional[dict[str, float]],
    ) -> torch.Tensor:
        if params is None:
            return video
        if video.ndim not in (4, 5):
            raise ValueError(f"Expected video shape [T,C,H,W] or [N,T,C,H,W], got {tuple(video.shape)}")
        if not torch.is_floating_point(video):
            raise TypeError("image_sensor_noise expects float pixel_values after processor ToTensor.")
        original_shape = tuple(video.shape)
        flat = video.reshape(-1, *original_shape[-3:]).clone()
        _, _, height, width = flat.shape
        blur_sigma = float(params["blur_sigma"])
        if blur_sigma > 0.0:
            kernel_size = max(1, int(params["blur_kernel_size"]))
            if kernel_size % 2 == 0:
                kernel_size += 1
            flat = transforms_F.gaussian_blur(
                flat,
                kernel_size=[kernel_size, kernel_size],
                sigma=[blur_sigma, blur_sigma],
            )
        downsample_scale = float(params["downsample_scale"])
        if downsample_scale < 1.0:
            small_h = max(1, int(round(height * downsample_scale)))
            small_w = max(1, int(round(width * downsample_scale)))
            flat = F.interpolate(flat, size=(small_h, small_w), mode="bilinear", align_corners=False)
            flat = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
        gaussian_std = float(params["gaussian_std"])
        if gaussian_std > 0.0:
            flat = flat + torch.randn_like(flat) * gaussian_std
        flat.clamp_(0.0, 1.0)
        return flat.reshape(original_shape)

    def _apply_proprio_noise(
        self,
        proprio: torch.Tensor,
        is_pad: Optional[torch.Tensor] = None,
        dim_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.is_training_set or self.proprio_noise_std <= 0.0:
            return proprio
        if not torch.is_floating_point(proprio):
            raise TypeError("proprio_noise_std expects normalized floating-point proprio tensors.")
        noise = torch.randn_like(proprio) * self.proprio_noise_std
        if is_pad is not None:
            valid = (~is_pad.bool()).to(device=proprio.device, dtype=proprio.dtype)
            while valid.ndim < proprio.ndim:
                valid = valid.unsqueeze(-1)
            noise = noise * valid
        if dim_is_pad is not None:
            dim_valid = (~dim_is_pad.bool()).to(
                device=proprio.device,
                dtype=proprio.dtype,
            )
            while dim_valid.ndim < proprio.ndim:
                dim_valid = dim_valid.unsqueeze(0)
            noise = noise * dim_valid
        return proprio + noise

    def _format_video_window(
        self,
        video: torch.Tensor,
        color_jitter_params: Optional[dict[str, float]] = None,
        camera_jitter_params: Optional[dict[str, float]] = None,
        sensor_noise_params: Optional[dict[str, float]] = None,
    ) -> torch.Tensor:
        num_cameras = 1
        if video.ndim == 5:
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W] or [N,T,C,H,W], got {video.shape}"
            T_video, C, H, W = video.shape
        video = video.view(num_cameras, T_video, C, H, W)
        if self.sync_vlm_color_jitter:
            video = self._apply_color_jitter_to_video(video, color_jitter_params)
            color_jitter_params = None
        video = self._apply_camera_jitter_to_video(video, camera_jitter_params)
        if sensor_noise_params is not None and sensor_noise_params.get("camera_keys") is not None:
            video = self._apply_view_sensor_noise(video, sensor_noise_params)
            sensor_noise_params = None
        if self.concat_multi_camera == "latent_horizontal":
            if num_cameras <= 1:
                video = video.squeeze(0)
            else:
                video = self.resize_transform(video)
                video = self.crop_transform(video)
                video = self._apply_color_jitter_to_video(video, color_jitter_params)
                video = self._apply_sensor_noise_to_video(video, sensor_noise_params)
                video = self.normalize_transform(video)
                return video.permute(0, 2, 1, 3, 4).contiguous()
        elif self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            target_h, target_w = (int(value) for value in self.video_size)
            top_h = (target_h * 2) // 3
            bottom_h = target_h - top_h
            left_w = target_w // 2
            right_w = target_w - left_w
            cam_top = transforms_F.resize(
                video[0],
                size=[top_h, target_w],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[bottom_h, left_w],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[bottom_h, right_w],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, latent_horizontal, robotwin."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self._apply_color_jitter_to_video(video, color_jitter_params)
        video = self._apply_sensor_noise_to_video(video, sensor_noise_params)
        video = self.normalize_transform(video)
        return video.permute(1, 0, 2, 3).contiguous()

    def _episode_start_index(self, sample_idx: int) -> int:
        episode_idx = bisect.bisect_right(self._episode_starts, int(sample_idx)) - 1
        if episode_idx < 0:
            return int(sample_idx)
        if int(sample_idx) >= self._episode_ends[episode_idx]:
            return int(sample_idx)
        return self._episode_starts[episode_idx]

    @staticmethod
    def _as_int(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

    def _build_episode_anchor(
        self,
        *,
        video_full: torch.Tensor,
        image_is_pad_full: torch.Tensor,
        proprio_full: torch.Tensor,
        proprio_is_pad_full: torch.Tensor,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        color_jitter_params: Optional[dict[str, float]] = None,
        camera_jitter_params: Optional[dict[str, float]] = None,
        sensor_noise_params: Optional[dict[str, float]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.memory_video_anchor_size <= 0:
            return None, None, None, None
        anchor_offsets = list(range(self.memory_video_anchor_size))
        if video_full.ndim == 5:
            max_idx = max(0, int(video_full.shape[1]) - 1)
            anchor_indices = [min(offset, max_idx) for offset in anchor_offsets]
            anchor_video_raw = video_full[:, anchor_indices, :, :, :].clone()
        else:
            max_idx = max(0, int(video_full.shape[0]) - 1)
            anchor_indices = [min(offset, max_idx) for offset in anchor_offsets]
            anchor_video_raw = video_full[anchor_indices, :, :, :].clone()
        max_pad_idx = max(0, int(image_is_pad_full.shape[0]) - 1)
        anchor_is_pad = torch.stack(
            [
                image_is_pad_full[min(offset, max_pad_idx)]
                if offset <= max_pad_idx
                else torch.ones((), dtype=torch.bool, device=image_is_pad_full.device)
                for offset in anchor_offsets
            ],
            dim=0,
        ).clone()
        valid_positions = (~anchor_is_pad).nonzero(as_tuple=False).flatten()
        if valid_positions.numel() > 0 and bool(anchor_is_pad.any().item()):
            first_valid = int(valid_positions[0].item())
            for pos in range(self.memory_video_anchor_size):
                if bool(anchor_is_pad[pos].item()):
                    prior_valid = valid_positions[valid_positions < pos]
                    copy_pos = int(prior_valid[-1].item()) if prior_valid.numel() > 0 else first_valid
                    if anchor_video_raw.ndim == 5:
                        anchor_video_raw[:, pos] = anchor_video_raw[:, copy_pos]
                    else:
                        anchor_video_raw[pos] = anchor_video_raw[copy_pos]
                    anchor_is_pad[pos] = False
        anchor_video = self._format_video_window(
            anchor_video_raw,
            color_jitter_params=color_jitter_params,
            camera_jitter_params=camera_jitter_params,
            sensor_noise_params=sensor_noise_params,
        )

        max_state_idx = max(0, int(proprio_full.shape[0]) - 1)
        anchor_state_indices = [min(offset, max_state_idx) for offset in anchor_offsets]
        anchor_proprio = proprio_full[anchor_state_indices].clone()
        max_state_pad_idx = max(0, int(proprio_is_pad_full.shape[0]) - 1)
        anchor_proprio_is_pad = torch.stack(
            [
                proprio_is_pad_full[min(index, max_state_pad_idx)]
                if index <= max_state_pad_idx
                else torch.ones((), dtype=torch.bool, device=proprio_is_pad_full.device)
                for index in anchor_state_indices
            ],
            dim=0,
        ).clone()
        valid_state_positions = (~anchor_proprio_is_pad).nonzero(as_tuple=False).flatten()
        if valid_state_positions.numel() > 0 and bool(anchor_proprio_is_pad.any().item()):
            first_valid = int(valid_state_positions[0].item())
            for pos in range(self.memory_video_anchor_size):
                if bool(anchor_proprio_is_pad[pos].item()):
                    prior_valid = valid_state_positions[valid_state_positions < pos]
                    copy_pos = int(prior_valid[-1].item()) if prior_valid.numel() > 0 else first_valid
                    anchor_proprio[pos] = anchor_proprio[copy_pos]
                    anchor_proprio_is_pad[pos] = False
        anchor_proprio = self._apply_proprio_noise(anchor_proprio, anchor_proprio_is_pad, dim_is_pad=proprio_dim_is_pad)
        return anchor_video, anchor_is_pad, anchor_proprio, anchor_proprio_is_pad

    def collate_fn(self, batch):
        return default_collate(batch)

    def _memory_query_layout(
        self,
        sample_idx: int,
    ) -> tuple[list[int], list[int], list[int], int, bool]:
        sample_idx = int(sample_idx)
        episode_start_idx = (
            self._episode_start_index(sample_idx)
            if self.memory_video_anchor_size > 0
            else sample_idx
        )
        episode_step_offset = max(0, sample_idx - episode_start_idx)

        known_anchor_frames = min(
            self.memory_video_anchor_size,
            max(1, episode_step_offset + 1),
        )
        anchor_indices = [
            episode_start_idx + min(offset, known_anchor_frames - 1)
            for offset in range(self.memory_video_anchor_size)
        ]

        recent_is_real = (
            episode_step_offset >= self.memory_recent_frame_offset
        )
        recent_idx = (
            sample_idx - self.memory_recent_frame_offset
            if recent_is_real
            else episode_start_idx
        )

        memory_offsets = [
            absolute_idx - sample_idx
            for absolute_idx in [*anchor_indices, recent_idx]
        ]
        image_offsets = [*memory_offsets, *self.video_sample_indices]
        state_offsets = [
            *memory_offsets,
            *range(int(self.num_frames)),
        ]
        action_offsets = list(range(int(self.num_frames) - 1))
        return (
            image_offsets,
            state_offsets,
            action_offsets,
            episode_step_offset,
            recent_is_real,
        )

    def _combined_sample_offsets(
        self,
        sample_idx: int,
    ) -> tuple[list[int], list[int], list[int]]:
        image_offsets, state_offsets, action_offsets, _, _ = (
            self._memory_query_layout(sample_idx)
        )
        return image_offsets, state_offsets, action_offsets

    def _vlm_current_sample_offsets(
        self,
        sample_idx: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """Load one current RGB frame while retaining full state/action windows."""

        _, state_offsets, action_offsets = self._combined_sample_offsets(
            sample_idx
        )
        return [0], state_offsets, action_offsets

    def _cached_video_shape(self) -> torch.Tensor:
        """Describe the decoded-video shape replaced by cached VAE latents."""

        num_frames = len(self.video_sample_indices)
        height, width = (int(value) for value in self.video_size)
        if self.concat_multi_camera == "latent_horizontal":
            processor = self.lerobot_dataset.processor
            num_cameras = int(
                getattr(processor, "num_output_cameras", len(self.video_view_names))
            )
            shape = (num_cameras, 3, num_frames, height, width)
        else:
            shape = (3, num_frames, height, width)
        return torch.tensor(shape, dtype=torch.long)

    def _get_with_vlm_current_images(self, idx: int, *, decode_images: bool = True) -> dict:
        """Build a cache-backed training sample with current RGB views only."""

        sample = self.lerobot_dataset.get_item_with_offset_factory(
            int(idx),
            offsets_factory=self._vlm_current_sample_offsets,
            processor_method=("preprocess_vlm_current" if decode_images
                              else "preprocess_without_images"),
            decode_images=decode_images,
        )
        actual_sample_idx = self._as_int(sample.get("idx", idx))
        _, _, _, _, recent_is_real = self._memory_query_layout(
            actual_sample_idx
        )

        anchor_count = self.memory_video_anchor_size
        recent_position = anchor_count
        memory_count = anchor_count + 1
        current_state_slice = slice(
            memory_count,
            memory_count + int(self.num_frames),
        )

        proprio_full = sample["proprio"]
        proprio_is_pad_full = sample["proprio_is_pad"]
        current_proprio_full = proprio_full[current_state_slice]
        current_proprio_is_pad_full = proprio_is_pad_full[current_state_slice]
        proprio_is_pad = current_proprio_is_pad_full[
            self.past_obs_size : -1
        ]
        proprio = self._apply_proprio_noise(
            current_proprio_full[self.past_obs_size : -1, :],
            proprio_is_pad,
            dim_is_pad=sample.get("proprio_dim_is_pad"),
        )
        image_is_pad = current_proprio_is_pad_full[
            self.video_sample_indices
        ].clone()

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(instruction)

        canvas_layout = self.concat_multi_camera or "single"
        canvas_view_names = "|".join(self.video_view_names)
        data = {
            "sample_id": actual_sample_idx,
            "video_shape": self._cached_video_shape(),
            "action": sample["action"],
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "video_layout": "single" if self.single_canvas else canvas_layout,
            "video_view_names": "canvas" if self.single_canvas else canvas_view_names,
            "video_canvas_layout": canvas_layout,
            "video_canvas_view_names": canvas_view_names,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "action_dim_is_pad": sample["action_dim_is_pad"],
            "proprio_is_pad": proprio_is_pad,
        }
        if decode_images:
            data["vlm_current_images"] = sample["vlm_current_images"]
        if sample.get("proprio_dim_is_pad") is not None:
            data["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]

        if anchor_count > 0:
            anchor_proprio_is_pad = proprio_is_pad_full[:anchor_count].clone()
            anchor_proprio = self._apply_proprio_noise(
                proprio_full[:anchor_count].clone(),
                anchor_proprio_is_pad,
                dim_is_pad=sample.get("proprio_dim_is_pad"),
            )
            data["memory_video_anchor_is_pad"] = anchor_proprio_is_pad.clone()
            data["memory_video_anchor_proprio"] = anchor_proprio
            data["memory_video_anchor_proprio_is_pad"] = anchor_proprio_is_pad

        recent_proprio_is_pad = torch.stack(
            [
                proprio_is_pad_full[recent_position].to(dtype=torch.bool)
                | (not recent_is_real)
            ]
        )
        recent_proprio = self._apply_proprio_noise(
            proprio_full[recent_position : recent_position + 1].clone(),
            recent_proprio_is_pad,
            dim_is_pad=sample.get("proprio_dim_is_pad"),
        )
        data["memory_video_recent_is_pad"] = recent_proprio_is_pad.clone()
        data["memory_video_recent_proprio"] = recent_proprio
        data["memory_video_recent_proprio_is_pad"] = recent_proprio_is_pad
        return data

    @staticmethod
    def _slice_video_time(
        video: torch.Tensor,
        time_slice: slice,
    ) -> torch.Tensor:
        if video.ndim == 5:
            return video[:, time_slice, :, :, :]
        if video.ndim == 4:
            return video[time_slice, :, :, :]
        raise ValueError(
            "Expected video to have shape [T,C,H,W] or [N,T,C,H,W], "
            f"got {tuple(video.shape)}"
        )

    def _get(self, idx):
        idx = int(idx)
        color_jitter_params = self._sample_color_jitter_params()
        camera_jitter_params = self._sample_camera_jitter_params()
        sensor_noise_params = self._sample_sensor_noise_params()
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset.get_item_with_offset_factory(
                sample_idx,
                offsets_factory=self._combined_sample_offsets,
            )

            memory_count = self.memory_video_anchor_size + 1
            current_image_slice = slice(
                memory_count,
                memory_count + len(self.video_sample_indices),
            )
            current_state_slice = slice(
                memory_count,
                memory_count + int(self.num_frames),
            )

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            current_action_is_pad = action_is_pad[self.past_action_size :]
            if bool(current_action_is_pad.any().item()):
                has_pad = True
            current_image_is_pad = image_is_pad[current_image_slice]
            if bool(current_image_is_pad.any().item()):
                has_pad = True
            current_proprio_is_pad = proprio_is_pad[current_state_slice]
            if bool(current_proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        actual_sample_idx = self._as_int(sample.get("idx", sample_idx))
        _, _, _, _, recent_is_real = self._memory_query_layout(
            actual_sample_idx
        )
        anchor_count = self.memory_video_anchor_size
        recent_position = anchor_count
        memory_count = anchor_count + 1
        current_image_slice = slice(
            memory_count,
            memory_count + len(self.video_sample_indices),
        )
        current_state_slice = slice(
            memory_count,
            memory_count + int(self.num_frames),
        )

        video_full = sample["pixel_values"]
        recent_video_raw = self._slice_video_time(
            video_full,
            slice(recent_position, recent_position + 1),
        ).clone()
        memory_video_recent = self._format_video_window(
            recent_video_raw,
            color_jitter_params=color_jitter_params,
            camera_jitter_params=camera_jitter_params,
            sensor_noise_params=sensor_noise_params,
        )
        recent_source_pad = sample["image_is_pad"][recent_position].to(
            dtype=torch.bool
        )
        memory_video_recent_is_pad = torch.stack(
            [recent_source_pad | (not recent_is_real)]
        )
        memory_video_recent_proprio = sample["proprio"][
            recent_position : recent_position + 1
        ].clone()
        memory_video_recent_proprio_is_pad = torch.stack(
            [
                sample["proprio_is_pad"][recent_position].to(
                    dtype=torch.bool
                )
                | (not recent_is_real)
            ]
        )
        memory_video_recent_proprio = self._apply_proprio_noise(
            memory_video_recent_proprio,
            memory_video_recent_proprio_is_pad,
            dim_is_pad=sample.get("proprio_dim_is_pad"),
        )

        video_raw = self._slice_video_time(video_full, current_image_slice)
        image_is_pad = sample["image_is_pad"][current_image_slice]
        video = self._format_video_window(
            video_raw,
            color_jitter_params=color_jitter_params,
            camera_jitter_params=camera_jitter_params,
            sensor_noise_params=sensor_noise_params,
        )  # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action_full = sample["action"] # [past_action_size + T - 1, action_dim]
        action_is_pad_full = sample["action_is_pad"]
        action = action_full[self.past_action_size :]
        action_is_pad = action_is_pad_full[self.past_action_size :]

        proprio_full = sample["proprio"][current_state_slice]
        proprio_is_pad_full = sample["proprio_is_pad"][current_state_slice]
        memory_video_anchor_proprio = None
        memory_video_anchor_proprio_is_pad = None
        proprio_is_pad = proprio_is_pad_full[self.past_obs_size : -1]
        proprio = self._apply_proprio_noise(
            proprio_full[self.past_obs_size : -1, :],
            proprio_is_pad,
            dim_is_pad=sample.get("proprio_dim_is_pad"),
        ) # [T-1, state_dim], aligned with current action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )
        if self.memory_video_anchor_size > 0:
            anchor_video_raw = self._slice_video_time(
                video_full,
                slice(0, anchor_count),
            )
            anchor_image_is_pad = sample["image_is_pad"][:anchor_count]
            anchor_proprio = sample["proprio"][:anchor_count]
            anchor_proprio_is_pad = sample["proprio_is_pad"][:anchor_count]
            (
                memory_video_anchor,
                memory_video_anchor_is_pad,
                memory_video_anchor_proprio,
                memory_video_anchor_proprio_is_pad,
            ) = self._build_episode_anchor(
                video_full=anchor_video_raw,
                image_is_pad_full=anchor_image_is_pad,
                proprio_full=anchor_proprio,
                proprio_is_pad_full=anchor_proprio_is_pad,
                color_jitter_params=color_jitter_params,
                camera_jitter_params=camera_jitter_params,
                sensor_noise_params=sensor_noise_params,
                proprio_dim_is_pad=sample.get("proprio_dim_is_pad"),
            )
        else:
            memory_video_anchor = None
            memory_video_anchor_is_pad = None

        task = sample["instruction"]
        
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        
        canvas_layout = self.concat_multi_camera or "single"
        canvas_view_names = "|".join(self.video_view_names)
        data = {
            "sample_id": actual_sample_idx,
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "video_layout": "single" if self.single_canvas else canvas_layout,
            "video_view_names": "canvas" if self.single_canvas else canvas_view_names,
            "video_canvas_layout": canvas_layout,
            "video_canvas_view_names": canvas_view_names,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": sample["action_dim_is_pad"],
            "proprio_is_pad": proprio_is_pad,
        }
        if sample.get("proprio_dim_is_pad") is not None:
            data["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]
        if "vlm_current_images" in sample:
            images = sample["vlm_current_images"]
            if self.sync_vlm_color_jitter or self.sync_vlm_camera_jitter:
                images = self._format_vlm_color(images, color_jitter_params)
                images = self._format_vlm_camera(images, camera_jitter_params, sensor_noise_params)
                data["vlm_current_images"] = images
            elif self.apply_color_jitter_to_vlm:
                if self.image_camera_jitter and self.image_camera_jitter["apply_to_vlm"]:
                    pixels = self._apply_camera_jitter_to_video(images.float().unsqueeze(1), camera_jitter_params).squeeze(1)
                    images = pixels.round().clamp(0, 255).to(torch.uint8) if images.dtype == torch.uint8 else pixels
                sample["vlm_current_images"] = images
                # Opt-in: reuse the per-sample video color-jitter factors.
                data["vlm_current_images"] = self._apply_color_jitter_to_vlm_images(
                    sample["vlm_current_images"], color_jitter_params
                )
            else:
                data["vlm_current_images"] = sample["vlm_current_images"]
        if memory_video_anchor is not None:
            data["memory_video_anchor"] = memory_video_anchor
            data["memory_video_anchor_is_pad"] = memory_video_anchor_is_pad
            if memory_video_anchor_proprio is not None:
                data["memory_video_anchor_proprio"] = memory_video_anchor_proprio
                data["memory_video_anchor_proprio_is_pad"] = memory_video_anchor_proprio_is_pad
        data["memory_video_recent"] = memory_video_recent
        data["memory_video_recent_is_pad"] = memory_video_recent_is_pad
        data["memory_video_recent_proprio"] = memory_video_recent_proprio
        data["memory_video_recent_proprio_is_pad"] = (
            memory_video_recent_proprio_is_pad
        )
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run tools/text_cache.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        context = context.clone()
        context[~context_mask] = 0.0
        return context, torch.ones_like(context_mask)

    def __getitem__(self, idx):
        try:
            return self._get(int(idx))
        except Exception as exc:
            print(
                f"Error processing sample idx {idx}: {exc}. "
                "Returning a random sample instead."
            )
            print(traceback.format_exc())
            return self._get(int(np.random.randint(len(self))))

    def get_item_without_images(self, idx: int) -> dict:
        """Build non-image training inputs for complete VAE/VLM cache coverage."""
        return self._get_with_vlm_current_images(int(idx), decode_images=False)

    def get_item_with_vlm_current_images(self, idx: int) -> dict:
        """Cache projection that decodes one current frame for each camera."""

        try:
            return self._get_with_vlm_current_images(int(idx))
        except Exception as exc:
            print(
                f"Error processing VLM-current sample idx {idx}: {exc}. "
                "Returning a random sample instead."
            )
            print(traceback.format_exc())
            return self._get_with_vlm_current_images(
                int(np.random.randint(len(self)))
            )

    def _camera_selected(self, index, params):
        keys = params.get("camera_keys")
        return keys is None or self.video_view_names[index] in keys


    def _apply_view_sensor_noise(self, video, params, value_scale=1.):
        if params is None:
            return video
        return torch.stack([
            self._apply_sensor_noise_to_video(view / value_scale, params) * value_scale
            if self._camera_selected(index, params) else view
            for index, view in enumerate(video)
        ])


    def _format_vlm_color(self, images, params):
        if not getattr(self, "sync_vlm_color_jitter", False) or params is None:
            return images
        # Preserve resolution and the encoder's 0..255 input contract. Float
        # output avoids a second uint8 quantization after augmentation.
        return self._apply_color_jitter_to_video(
            images.to(torch.float32) / 255.0, params
        ) * 255.0


    def _format_vlm_camera(self, images, params, sensor_noise_params=None):
        if not getattr(self, "sync_vlm_camera_jitter", False):
            return images
        if params is None and sensor_noise_params is None:
            return images
        # Share draws across current/memory video and VLM views. Only selected
        # cameras are modified, preserving the wrist pixels exactly.
        video = self._apply_camera_jitter_to_video(
            images.to(torch.float32).unsqueeze(1), params
        )
        # Bilinear interpolation may overshoot 255 by a few float32 ULPs.
        return self._apply_view_sensor_noise(video, sensor_noise_params, value_scale=255.).squeeze(1).clamp(0., 255.)
