"""RoboDojo (arx_x5) dataset for EWAM post-training.

Reads XPolicyLab-standard RoboDojo HDF5 episodes directly; no offline video
conversion. Prepared layout (written by policy/EWAM/process_data.py):

    <dataset_dir>/
    ├── episodes/<task>/episode_*.hdf5   # links to the raw <task>/arx_x5/data/
    └── t5_cache/<sha1(instruction)>.pt  # UMT5-XXL embedding, unpadded (seq, 4096)

HDF5 fields used:
    state|action/{left,right}_arm_joint_states   (T, 6)
    state|action/{left,right}_ee_joint_states    (T, 1) gripper in [0, 1]
    vision/cam_{head,left_wrist,right_wrist}/colors  (T,) JPEG bytes, 640x480
    instruction                                  task instruction string

Conventions:
* action/state: 16-D raw qpos [L6, pad, LG, R6, pad, RG], unified from the 14-D
  [L6, LG, R6, RG]; no normalization. ``action_dim_is_pad`` marks dims 6/14 so
  the reweighted loss skips them.
* video: 3-camera T-shape, each camera resized once onto the (H, W) canvas
  (cam_head on top, cam_left_wrist | cam_right_wrist below), block centered
  vertically. Matches the ``robodojo`` image layout in policy/EWAM/model.py.
* sampling: frames=8, ratio=2, gds=3 -> 16-step chunk at 8.33 Hz on 25 Hz data,
  so one model action spans 3 env steps at eval (``action_stride: 3``).
* text: the raw instruction (no scene prefix) for both T5 and the VLM.
"""

import hashlib
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from XPolicyLab.utils.process_data import decode_image_bit

from data.utils.image_utils import resize_with_padding
from utils.vlm_utils import preprocess_vlm_messages

logger = logging.getLogger(__name__)

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
PAD_DIMS = (6, 14)


def instruction_key(instruction: str) -> str:
    return hashlib.sha1(instruction.encode("utf-8")).hexdigest()


def read_instruction(f: h5py.File) -> str:
    raw = f["instruction"][()]
    return raw.decode("utf-8").strip() if isinstance(raw, bytes) else str(raw).strip()


def _joints14(f: h5py.File, group: str, idxs: List[int]) -> np.ndarray:
    parts = [
        f[f"{group}/left_arm_joint_states"][()][idxs, :6],
        f[f"{group}/left_ee_joint_states"][()][idxs].reshape(len(idxs), 1),
        f[f"{group}/right_arm_joint_states"][()][idxs, :6],
        f[f"{group}/right_ee_joint_states"][()][idxs].reshape(len(idxs), 1),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def _unify14_to_16(raw14: np.ndarray) -> np.ndarray:
    pad = np.zeros((raw14.shape[0], 1), dtype=np.float32)
    return np.concatenate(
        [raw14[:, 0:6], pad, raw14[:, 6:7], raw14[:, 7:13], pad, raw14[:, 13:14]], axis=1
    ).astype(np.float32)


class RoboDojoDataset(data.Dataset):
    """RoboDojo 16-D post-training dataset (local XPolicyLab-standard HDF5)."""

    ACTION_DIM = 16
    STATE_DIM = 16

    def __init__(
        self,
        dataset_dir: str,
        *,
        num_video_frames: int = 8,
        video_size: Tuple[int, int] = (384, 320),
        global_downsample_rate: int = 3,
        video_action_freq_ratio: int = 2,
        max_episodes: Optional[int] = None,
        image_aug: bool = False,
        val: bool = False,
        vlm_checkpoint_path: Optional[str] = None,
        t5_cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.num_video_frames = int(num_video_frames)
        self.video_size = tuple(video_size)
        self.global_downsample_rate = int(global_downsample_rate)
        self.video_action_freq_ratio = int(video_action_freq_ratio)
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        self.val = bool(val)
        self.action_dim = self.ACTION_DIM
        self.state_dim = self.STATE_DIM

        self.episodes = sorted((self.dataset_dir / "episodes").glob("*/*.hdf5"))
        if max_episodes is not None:
            self.episodes = self.episodes[:max_episodes]
        if not self.episodes:
            raise RuntimeError(f"No RoboDojo episodes under {self.dataset_dir / 'episodes'}")

        self.vlm_processor = None
        if vlm_checkpoint_path is not None:
            from transformers import AutoProcessor
            self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)

        cache_dir = Path(t5_cache_dir) if t5_cache_dir else self.dataset_dir / "t5_cache"
        self._t5 = {p.stem: torch.load(p, map_location="cpu", weights_only=True).float()
                    for p in sorted(cache_dir.glob("*.pt"))}
        if not self._t5:
            raise RuntimeError(f"No T5 embeddings under {cache_dir}; run process_data.sh first")

        logger.info(
            f"RoboDojoDataset: {len(self.episodes)} episodes, {len(self._t5)} instruction embeddings "
            f"(chunk={self.action_chunk_size}, gds={self.global_downsample_rate})"
        )

    def _calculate_sampling_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        physical_chunk_size = self.action_chunk_size * self.global_downsample_rate
        max_condition_idx = total_frames - physical_chunk_size - 1
        condition_frame_idx = 0 if max_condition_idx < 0 else random.randint(0, max_condition_idx)
        action_indices = [
            min(condition_frame_idx + (i + 1) * self.global_downsample_rate, total_frames - 1)
            for i in range(self.action_chunk_size)
        ]
        video_indices = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            video_indices.append(action_indices[min(action_step, len(action_indices) - 1)])
        return condition_frame_idx, video_indices, action_indices

    def _t_shape_geometry(self, top_hw, bl_hw, br_hw) -> Dict[str, int]:
        th, tw = self.video_size
        split_w = tw // 2
        right_w = tw - split_w

        def natural_h(hw, target_w):
            h, w = hw
            return max(1, int(round(h * target_w / w)))

        bot_h = min(max(natural_h(bl_hw, split_w), natural_h(br_hw, right_w)), th - 1)
        top_h = min(natural_h(top_hw, tw), th - bot_h)
        return {"top_h": top_h, "bot_h": bot_h, "split_w": split_w, "right_w": right_w,
                "pad_top": (th - top_h - bot_h) // 2}

    def _stitch_t_shape(self, top, bl, br) -> torch.Tensor:
        """Aspect-preserving single-resize T-shape -> [T, C, H, W] float in [0, 1]."""
        th, tw = self.video_size
        geo = self._t_shape_geometry(top.shape[1:3], bl.shape[1:3], br.shape[1:3])
        top_h, bot_h, split_w, right_w, y0 = (
            geo["top_h"], geo["bot_h"], geo["split_w"], geo["right_w"], geo["pad_top"])
        out = torch.zeros((top.shape[0], 3, th, tw), dtype=torch.float32)
        for i in range(top.shape[0]):
            canvas = np.zeros((th, tw, 3), dtype=np.uint8)
            canvas[y0:y0 + top_h] = resize_with_padding(top[i], (top_h, tw))
            yb = y0 + top_h
            canvas[yb:yb + bot_h, :split_w] = resize_with_padding(bl[i], (bot_h, split_w))
            canvas[yb:yb + bot_h, split_w:] = resize_with_padding(br[i], (bot_h, right_w))
            out[i] = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        return out

    def _read_sample(self, path: Path) -> Optional[Dict[str, Any]]:
        with h5py.File(path, "r") as f:
            total = int(f["state/left_arm_joint_states"].shape[0])
            if total < self.action_chunk_size * self.global_downsample_rate + 2:
                return None
            cond, video_indices, action_indices = self._calculate_sampling_indices(total)
            state16 = _unify14_to_16(_joints14(f, "state", [cond]))[0]
            actions16 = _unify14_to_16(_joints14(f, "action", action_indices))
            frame_indices = [cond] + video_indices
            frames = [decode_image_bit(f[f"vision/{cam}/colors"][()][frame_indices]) for cam in CAMERAS]
            instruction = read_instruction(f)
        return {
            "state16": state16, "actions16": actions16, "frames": frames, "instruction": instruction,
        }

    def __len__(self) -> int:
        return len(self.episodes) * 10

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        for _ in range(8):
            path = random.choice(self.episodes)
            try:
                raw = self._read_sample(path)
                if raw is None:
                    continue
                video_all = self._stitch_t_shape(*raw["frames"])
                first_frame, video_frames = video_all[0], video_all[1:]

                instruction = raw["instruction"]
                key = instruction_key(instruction)
                if key not in self._t5:
                    raise KeyError(f"no T5 embedding for instruction {instruction!r}; rerun process_data.sh")

                vlm_inputs = None
                if self.vlm_processor is not None:
                    frame = (first_frame.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    vlm_inputs = preprocess_vlm_messages(instruction, Image.fromarray(frame), self.vlm_processor)

                pad = torch.zeros(self.ACTION_DIM, dtype=torch.bool)
                pad[list(PAD_DIMS)] = True
                return {
                    "first_frame": first_frame,
                    "video_frames": video_frames,
                    "initial_state": torch.from_numpy(raw["state16"]),
                    "action_sequence": torch.from_numpy(raw["actions16"]),
                    "language_embedding": self._t5[key],
                    "vlm_inputs": vlm_inputs,
                    "action_dim_is_pad": pad,
                }
            except Exception as e:
                logger.warning(f"RoboDojo sample retry ({path.parent.name}/{path.name}): {e}")
        return None
