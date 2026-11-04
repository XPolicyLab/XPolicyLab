import hashlib
import os
import numpy as np
import traceback
import torch

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from flexwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from flexwam.datasets.lerobot.utils.normalizer import (
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
    wait_for_dataset_stats_json,
)
from flexwam.datasets.dataset_utils import Normalize
from flexwam.datasets.lerobot.transforms.image import pil_bilinear_resize
from flexwam.utils.logging_config import get_logger
from flexwam.utils import misc
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
        text_embedding_cache_encoder_id="wan22ti2v5b",
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        concat_multi_camera: str = "independent", # "independent", or None for single-camera
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
        self.global_sample_stride = global_sample_stride
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.text_embedding_cache_encoder_id = str(text_embedding_cache_encoder_id)
        self.context_len = context_len
        self.concat_multi_camera = concat_multi_camera
        if self.concat_multi_camera not in {None, "independent"}:
            raise ValueError(
                f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                "Expected 'independent' for multi-camera inputs, or None for single-camera."
            )

        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    save_dataset_stats_to_json(dataset_stats, stats_path)
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
                elif world_size > 1:
                    wait_for_dataset_stats_json(stats_path)
                    dataset_stats = load_dataset_stats_from_json(stats_path)
            else:
                if world_size > 1:
                    wait_for_dataset_stats_json(pretrained_norm_stats)
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _resize_and_normalize(self, video: torch.Tensor) -> torch.Tensor:
        if tuple(video.shape[-2:]) != (int(self.video_size[0]), int(self.video_size[1])):
            video = pil_bilinear_resize(video, self.video_size)
        return self.normalize_transform(video)

    def _get(self, idx):
        sample = self.lerobot_dataset[idx]
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        is_independent_multiview = self.concat_multi_camera == "independent"
        if is_independent_multiview:
            pass
        elif num_cameras > 1:
            raise ValueError(
                f"Multi-camera inputs require concat_multi_camera='independent', got {self.concat_multi_camera!r}."
            )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        if is_independent_multiview:
            video = video.reshape(num_cameras * T_video, C, H, W)
            video = self._resize_and_normalize(video)
            _, C, H, W = video.shape
            video = video.view(num_cameras, T_video, C, H, W)
            video = video.permute(1, 0, 2, 3, 4).contiguous() # [T_video, V, C, H, W], range [-1, 1]
        else:
            video = self._resize_and_normalize(video)  # [T_video, C, H, W]
            video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim], to align with action
        num_video_frames = video.shape[0] if is_independent_multiview else video.shape[1]
        if num_video_frames <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (num_video_frames - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {num_video_frames - 1}"
            )

        instruction = DEFAULT_PROMPT.format(task=sample["instruction"])
        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_dir,
            f"{hashed}.text_len{self.context_len}.{self.text_embedding_cache_encoder_id}.pt",
        )
        legacy_cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if self.text_embedding_cache_encoder_id == "wan22ti2v5b" and not os.path.exists(cache_path):
            cache_path = legacy_cache_path
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/preprocess/precompute_lingbot_text_embeds.py for LingBot or scripts/preprocess/precompute_text_embeds.py for WAN first."
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
