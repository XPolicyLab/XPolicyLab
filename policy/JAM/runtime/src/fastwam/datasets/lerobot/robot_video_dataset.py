import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


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
        load_text_context: bool = True,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        first_frame_only: bool = False,  # decode/encode ONLY video frame 0 (pixels); pad metadata for
                                         # all video frames is rebuilt from proprio_is_pad (same time grid),
                                         # so foresight frame_valid masking is UNCHANGED. Use only with
                                         # foresight target caches (teacher needs no pixels).
        video_future_frames: int = 0,    # extend the VIDEO window this many extra frames on the latent grid
                                         # (multiple of 4 obs steps each) for foresight teacher targets;
                                         # action horizon and proprio/state windows UNCHANGED. Online only.
        ssl_clip_indices: Optional[list[int]] = None,
        use_input_latent_cache: bool = False,
    ):
        self.video_future_frames = int(video_future_frames)
        assert self.video_future_frames >= 0
        assert not (first_frame_only and self.video_future_frames > 0), \
            "first_frame_only and video_future_frames are mutually exclusive"
        image_future_steps = self.video_future_frames * action_video_freq_ratio
        self.ssl_clip_indices = (
            None
            if ssl_clip_indices is None
            else [int(index) for index in ssl_clip_indices]
        )
        if self.ssl_clip_indices is not None:
            if first_frame_only:
                raise ValueError("`ssl_clip_indices` and `first_frame_only` are mutually exclusive.")
            if not self.ssl_clip_indices:
                raise ValueError("`ssl_clip_indices` must not be empty.")
            max_image_index = num_frames + image_future_steps - 1
            if (
                min(self.ssl_clip_indices) < 0
                or max(self.ssl_clip_indices) > max_image_index
            ):
                raise ValueError(
                    "`ssl_clip_indices` must index the fetched image window "
                    f"0..{max_image_index}, got {self.ssl_clip_indices}."
                )
            if len(set(self.ssl_clip_indices)) != len(self.ssl_clip_indices):
                raise ValueError(
                    f"`ssl_clip_indices` must not contain duplicates, got {self.ssl_clip_indices}."
                )
            decoded_image_indices = sorted({0, *self.ssl_clip_indices})
        else:
            decoded_image_indices = None
        self._decoded_image_indices = decoded_image_indices
        self._decoded_position = (
            None
            if decoded_image_indices is None
            else {raw_index: pos for pos, raw_index in enumerate(decoded_image_indices)}
        )
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            image_obs_indices=(
                decoded_image_indices
                if decoded_image_indices is not None
                else ([0] if first_frame_only else None)
            ),
            image_future_steps=image_future_steps,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )

        self.num_frames = num_frames
        self.first_frame_only = bool(first_frame_only)
        self.use_input_latent_cache = bool(use_input_latent_cache)
        if self.use_input_latent_cache and not self.first_frame_only:
            raise ValueError(
                "`use_input_latent_cache=true` requires `first_frame_only=true`."
            )
        self.action_video_freq_ratio = action_video_freq_ratio

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        total_video_grid = num_frames + image_future_steps
        self.video_sample_indices = list(range(0, total_video_grid, self.action_video_freq_ratio))
        assert (len(self.video_sample_indices) - 1) % 4 == 0, \
            f"video frames-1 must be divisible by 4 for tokenization, got {len(self.video_sample_indices)}"
        # video frames covering ONLY the action window (excludes appended future frames)
        self.num_action_video_frames = len(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        # With an exact input-latent cache, querying the current RGB frame is
        # both redundant and the remaining native PyAV/AV1 crash surface.
        self.lerobot_dataset._set_return_images(not self.use_input_latent_cache)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.load_text_context = bool(load_text_context)
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

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
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        ssl_image_is_pad = None
        if self.ssl_clip_indices is not None:
            decoded_pad = sample["image_is_pad"]
            current_position = self._decoded_position[0]
            clip_positions = [
                self._decoded_position[index] for index in self.ssl_clip_indices
            ]
            image_is_pad = decoded_pad[current_position : current_position + 1]
            ssl_image_is_pad = decoded_pad[clip_positions]
            pixel_indices = [current_position, *clip_positions]
        elif self.first_frame_only:
            # Pixels: only frame 0 was decoded. Pad METADATA must still cover every video
            # frame for foresight frame_valid; proprio shares the exact obs time grid, so
            # its pad vector is bit-identical to what the image pads would have been.
            image_is_pad = sample["proprio_is_pad"]
            pixel_indices = [0]
        else:
            image_is_pad = sample["image_is_pad"]
            pixel_indices = self.video_sample_indices

        video = sample.get("pixel_values")
        num_cameras = 1
        if self.use_input_latent_cache:
            if video is not None:
                raise RuntimeError(
                    "Input-latent cache mode must not materialize RGB pixel tensors."
                )
            T_video = 0
        elif video.ndim == 5:
            video = video[:, pixel_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], got {video.shape}"
            video = video[pixel_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        if self.ssl_clip_indices is None:
            image_is_pad = image_is_pad[self.video_sample_indices]  # always the FULL video-frame grid (9), even when pixels carry 1 frame

        if not self.use_input_latent_cache:
            video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.use_input_latent_cache:
            pass
        elif self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        if not self.use_input_latent_cache:
            video = self.resize_transform(video)
            video = self.crop_transform(video)
            video = self.normalize_transform(video)  # [T_video, C, H, W]

            video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]
        ssl_video = None
        if self.ssl_clip_indices is not None:
            # Position zero is the causal current image; the remaining frames
            # are the ordered future clip requested by ssl_clip_indices.
            ssl_video = video[:, 1:]
            video = video[:, :1]

        # Proxy (from lerobot):
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if not self.first_frame_only:
            if self.ssl_clip_indices is None and video.shape[1] <= 1:
                raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
            # divisibility binds action to the ACTION-window video frames; appended
            # video_future_frames (teacher-only) must not enter this check
            if action.shape[0] % (self.num_action_video_frames - 1) != 0:
                raise ValueError(
                    f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {self.num_action_video_frames - 1}"
                )

        task = sample["instruction"]

        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        data = {
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        # Sample provenance for mixed-root objectives (success distillation keys its
        # per-episode masks off these). Taken from the returned sample, not the
        # requested idx, because __getitem__ may substitute a random sample.
        for key, fallback in (("dataset_index", 0), ("episode_index", -1), ("index", -1)):
            value = sample.get(key, None)
            if value is None:
                value = torch.tensor(fallback)
            elif not torch.is_tensor(value):
                value = torch.tensor(int(value))
            data["sample_index" if key == "index" else key] = value.reshape(()).to(dtype=torch.long)
        if video is not None:
            data["video"] = video
        if ssl_video is not None:
            data["ssl_video"] = ssl_video
            data["ssl_image_is_pad"] = ssl_image_is_pad
        if self.load_text_context:
            context, context_mask = self._get_cached_text_context(instruction)
            # NOTE: to keep consistent with wan2.2's behavior
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
            data["context"] = context
            data["context_mask"] = context_mask
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
                "Run scripts/precompute_text_embeds.py first."
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

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
