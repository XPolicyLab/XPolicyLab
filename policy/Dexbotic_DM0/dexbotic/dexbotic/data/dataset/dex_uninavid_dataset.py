"""UniNaVid navigation dataset with dexbotic-style transforms/processors."""

from __future__ import annotations

import copy
import glob
import hashlib
import math
import os
import random
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from decord import VideoReader, cpu
from loguru import logger
from PIL import Image


from dexbotic.data.dataset.augmentations import NAME2AUG
from dexbotic.data.dataset.dex_dataset import DexDataset, load_jsonl


class DexUniNaVidDataset(DexDataset):
    """Navigation dataset loading stem-aligned jsonl + video with sparse prefix sampling.

    Each jsonl under ``dataset_info["annotations"]`` corresponds to one video episode.
    Rows are sorted by ``images_1.frame_idx``.  For every episode we sample a set of
    "prefix end" indices so each training sample sees a prefix of the episode's history.
    """

    default_keys = ["input_ids", "labels", "image", "prompt"]

    def __init__(
        self,
        data_args,
        tokenization_func: Callable,
        action_process_func=None,
        image_process_func=None,
        depth_process_func=None,
        **kwargs,
    ):
        # Set before super().__init__ because _build_dataset_index needs it.
        self.data_args = data_args
        self.use_nav_augment = getattr(data_args, "dex_use_nav_augment", True)
        super().__init__(
            data_args,
            tokenization_func,
            action_process_func=action_process_func,
            image_process_func=image_process_func,
            depth_process_func=depth_process_func,
            **kwargs,
        )

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info(
                "DexUniNaVidDataset: {} jsonl files, {} samples",
                len(self._jsonl_files),
                self.total_samples,
            )

    def _build_dataset_index(self):
        self._video_sample_steps: Dict[str, int] = {}
        self._episode_records: Dict[str, List[Dict[str, Any]]] = {}
        self._episode_videos: Dict[str, str] = {}
        self._jsonl_files: List[str] = []
        global_index = []
        file_name_map = {}
        dataset_map = {}
        file_id = 0

        jsonl_suffix = getattr(self.data_args, "dex_jsonl_suffix", ".jsonl")

        for dataset_id, dataset_info in enumerate(self.datasets_info):
            data_path = dataset_info["annotations"]
            video_dir = dataset_info.get("data_path_prefix", "")
            dataset_map[dataset_id] = {
                "data_path": data_path,
                "meta_data": dataset_info["meta_data"],
                "data_path_prefix": video_dir,
            }

            jsonl_files = sorted(glob.glob(os.path.join(data_path, f"*{jsonl_suffix}")))
            self._jsonl_files.extend(jsonl_files)

            for jsonl_file in jsonl_files:
                records = sorted(
                    load_jsonl(jsonl_file, parse=True),
                    key=lambda r: int(r["images_1"]["frame_idx"]),
                )
                video_name = os.path.basename(str(records[0]["images_1"]["url"]))
                video_abs = (
                    os.path.join(video_dir, video_name)
                    if video_dir
                    else os.path.join(data_path, "videos", video_name)
                )

                if video_abs not in self._video_sample_steps:
                    video_reader = VideoReader(video_abs, ctx=cpu(0))
                    self._video_sample_steps[video_abs] = round(
                        video_reader.get_avg_fps() / self.data_args.video_fps
                    )

                self._episode_records[jsonl_file] = records
                self._episode_videos[jsonl_file] = video_abs
                if jsonl_file not in file_name_map:
                    file_name_map[jsonl_file] = file_id
                    file_id += 1
                file_index = file_name_map[jsonl_file]

                episode_key = f"{dataset_id}:{os.path.relpath(jsonl_file, data_path)}"
                for line_idx in self._sparse_prefix_end_lines(
                    len(records), seed_key=episode_key
                ):
                    global_index.append((dataset_id, file_index, line_idx))

        self.global_index = global_index
        self.file_name_map = {v: k for k, v in file_name_map.items()}
        self.dataset_map = dataset_map
        self.total_samples = len(self.global_index)

    def _get_episode_sample(
        self, idx: int
    ) -> tuple[int, str, List[Dict[str, Any]], int]:
        dataset_index, file_index, line_idx = self.global_index[idx]
        jsonl_file = self.file_name_map[file_index]
        return (
            dataset_index,
            self._episode_videos[jsonl_file],
            self._episode_records[jsonl_file],
            line_idx,
        )

    @property
    def lengths(self) -> List[int]:
        lengths = []
        for idx in range(self.total_samples):
            _, video_abs, records, line_idx = self._get_episode_sample(idx)
            sample_step = self._video_sample_steps[video_abs]
            supervised_row_index = list(range(0, line_idx + 1, sample_step))[-1]
            record = records[supervised_row_index]
            prompt_text = str(record["prompt"]).strip()
            answer_text = str(record["answer"]).strip()
            lengths.append(len(prompt_text.split()) + len(answer_text.split()))
        return lengths

    @property
    def modality_lengths(self) -> List[int]:
        return [max(1, x) for x in self.lengths]

    @staticmethod
    def _build_conversation(prompt: str, answer: Any) -> List[Dict[str, str]]:
        return [
            {"from": "human", "value": str(prompt).strip()},
            {"from": "gpt", "value": str(answer).strip()},
        ]

    def _augment_history_frames(self, frames: List[Any]) -> List[Any]:
        if not self.use_nav_augment or len(frames) <= 1:
            return frames

        last = len(frames) - 1
        max_drop = math.ceil(0.1 * last)
        n_keep = last - random.randint(0, max_drop)
        sampled = sorted(random.sample(range(last), n_keep)) + [last]
        duplicated = []
        for i, idx in enumerate(sampled):
            duplicated.append(idx)
            if random.random() < 0.03 or (
                i == len(sampled) - 1 and random.random() < 0.06
            ):
                duplicated.append(idx)

        jitter = NAME2AUG["uninavid"]()
        out = []
        for idx in duplicated:
            frame = frames[idx]
            if frame is None:
                out.append(None)
            else:
                aug_frame = jitter(image=np.asarray(frame.convert("RGB")))["image"]
                out.append(Image.fromarray(aug_frame, mode="RGB"))
        return out


    def unsafe_getitem(self, idx: int) -> dict:
        dataset_index, video_abs, records, current_idx = self._get_episode_sample(idx)
        prefix_len = current_idx + 1
        sample_step = self._video_sample_steps[video_abs]

        history_row_indices = list(range(0, prefix_len, sample_step))
        supervised_row_index = history_row_indices[-1]
        dataset_info = self.dataset_map[dataset_index]
        meta_data = copy.deepcopy(dataset_info["meta_data"])
        images_keys = self.images_keys or ["images_1"]
        meta_data.update(
            dict(
                fram_indicies=history_row_indices,
                jsonl_file=self.file_name_map[self.global_index[idx][1]],
                dataset=dataset_info["data_path"],
                num_images=len(images_keys),
                images_keys=images_keys,
                depths_keys=self.depths_keys,
                load_depth=False,
                data_path_prefix=dataset_info["data_path_prefix"],
            )
        )
        episode_data_list = self.action_process_func(records, meta_data=meta_data)

        history_rgb_frames = []
        for frame_idx in history_row_indices:
            frame_data = copy.deepcopy(episode_data_list[frame_idx])
            rgb_data = frame_data.get("rgb_data", [])
            history_rgb_frames.append(rgb_data[0] if len(rgb_data) > 0 else None)
        history_rgb_frames = self._augment_history_frames(history_rgb_frames)

        image_process_func = (
            self.image_process_func[0]
            if isinstance(self.image_process_func, list)
            else self.image_process_func
        )
        pixel_values = [image_process_func(frame) for frame in history_rgb_frames]
        image = torch.stack(pixel_values, dim=0)

        sample_data = episode_data_list[supervised_row_index]
        if "conversations" not in sample_data:
            sample_data["conversations"] = self._build_conversation(
                prompt=sample_data["prompt"],
                answer=sample_data["answer"],
            )
        conversations = sample_data["conversations"]
        tokenized = self.tokenization_func(conversations, has_image=True)

        out = {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "image": image,
        }
        if tokenized.get("prompt") is not None:
            out["prompt"] = tokenized["prompt"]
        return out

    @staticmethod
    def _stable_rng(seed_key: str) -> random.Random:
        seed_digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
        return random.Random(int.from_bytes(seed_digest[:8], "big"))

    @classmethod
    def _sparse_prefix_end_lines(
        cls, n_rows: int, seed_key: str, min_step: int = 2, max_step: int = 4
    ) -> List[int]:
        """Return deterministic sparse prefix-end row indices covering [0, n_rows-1]."""
        if n_rows <= 0:
            return []
        if n_rows == 1:
            return [0]

        rng = cls._stable_rng(seed_key)
        out = [0]
        current = 0
        while current < n_rows - 1:
            current = min(current + rng.randint(min_step, max_step), n_rows - 1)
            out.append(current)

        if out[-1] != n_rows - 1:
            out.append(n_rows - 1)
        return out

