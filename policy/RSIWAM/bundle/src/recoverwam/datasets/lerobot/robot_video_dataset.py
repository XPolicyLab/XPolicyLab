import hashlib
import os
from typing import Optional
import numpy as np
import traceback
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from recoverwam.utils.logging_config import get_logger
from recoverwam.utils import misc

logger = get_logger(__name__)
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def _is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return os.environ.get("RANK", "0") in {"0", ""}


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        action_horizon: Optional[int] = None,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        text_embedding_cache_tag: str = "wan22ti2v5b",
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
        decode_only_first_video_frame: bool = False,
        decode_sampled_video_frames: bool = False,
        episode_instruction_key: Optional[str] = None,
    ):
        if concat_multi_camera != "horizontal" or camera_key is not None:
            raise ValueError("LIBERO release uses both cameras in horizontal layout")
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be positive, got {num_frames}")
        if action_horizon is None:
            action_horizon = num_frames - 1
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")
        if action_horizon > num_frames - 1:
            raise ValueError(
                f"`action_horizon` cannot exceed the observation transitions, got action_horizon={action_horizon} and num_frames={num_frames}"
            )
        if action_video_freq_ratio <= 0:
            raise ValueError(
                f"`action_video_freq_ratio` must be positive, got {action_video_freq_ratio}"
            )
        if (num_frames - 1) % action_video_freq_ratio != 0:
            raise ValueError(
                f"`num_frames - 1` must be divisible by `action_video_freq_ratio`, got {num_frames - 1} and {action_video_freq_ratio}"
            )
        video_transitions = (num_frames - 1) // action_video_freq_ratio
        if video_transitions % 4 != 0:
            raise ValueError(
                f"`video` transitions must be divisible by 4 for tokenization, got {video_transitions}"
            )
        self.decode_only_first_video_frame = bool(decode_only_first_video_frame)
        self.decode_sampled_video_frames = bool(decode_sampled_video_frames)
        if self.decode_only_first_video_frame and self.decode_sampled_video_frames:
            raise ValueError(
                "`decode_only_first_video_frame` and `decode_sampled_video_frames` are mutually exclusive"
            )
        self.logical_num_video_frames = video_transitions + 1
        self.video_sample_indices = (
            [0]
            if self.decode_only_first_video_frame
            else list(range(0, num_frames, action_video_freq_ratio))
        )
        image_obs_indices = (
            self.video_sample_indices
            if self.decode_only_first_video_frame or self.decode_sampled_video_frames
            else None
        )
        self.decoded_video_sample_indices = (
            list(range(len(self.video_sample_indices)))
            if image_obs_indices is not None
            else self.video_sample_indices
        )
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            image_obs_indices=image_obs_indices,
            action_size=action_horizon,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            episode_instruction_key=episode_instruction_key,
        )
        self.num_frames = num_frames
        self.action_horizon = action_horizon
        self.action_video_freq_ratio = action_video_freq_ratio
        self.camera_key = None
        self.lerobot_dataset._set_return_images(True)
        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.text_embedding_cache_tag = text_embedding_cache_tag
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = "horizontal"
        self.override_instruction = override_instruction
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            expected_image_steps = self.lerobot_dataset.image_obs_size
            actual_image_steps = int(
                getattr(processor, "num_image_obs_steps", processor.num_obs_steps)
            )
            if actual_image_steps != expected_image_steps:
                raise ValueError(
                    f"Processor image horizon does not match the dataset decode horizon: expected {expected_image_steps}, got {actual_image_steps}."
                )
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them."
                    )
                if _is_main_process():
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if _is_main_process():
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )
            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx, *, strict_index: bool = False):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            if strict_index:
                sample = self.lerobot_dataset.get_strict_item(sample_idx)
                actual_index = int(sample.get("idx", -1))
                if actual_index != int(idx):
                    raise ValueError(
                        f"Strict sample provenance mismatch: requested {idx}, got {actual_index}"
                    )
            else:
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
            if strict_index or not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = np.random.randint(len(self.lerobot_dataset))
        image_is_pad = sample["image_is_pad"]
        video = sample["pixel_values"]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.decoded_video_sample_indices, :, :, :]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, (
                f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            )
            video = video[self.decoded_video_sample_indices, :, :, :]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.decoded_video_sample_indices]
        video = video.view(num_cameras, T_video, C, H, W)
        if num_cameras > 1:
            video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
        else:
            video = video.squeeze(0)
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)
        action = sample["action"]
        proprio = sample["proprio"][:1, :]
        if video.shape[1] <= 1 and (not self.decode_only_first_video_frame):
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        data = {
            "video": video,
            "video_num_frames": self.logical_num_video_frames,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"][:1],
        }
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_dir, f"{hashed}.t5_len{self.context_len}.{self.text_embedding_cache_tag}.pt"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. Run scripts/precompute_text_embeds.py first."
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
        return (context, context_mask)

    def get_strict_item(self, idx):
        """Return an exact fixed-index sample together with its provenance."""
        return (int(idx), self._get(idx, strict_index=True))

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
