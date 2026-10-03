"""RoboTwin Clean50 loader used by the public post-training recipe."""

from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


CAMERAS = ("head_camera", "left_camera", "right_camera")
ACTION_OFFSETS = tuple(range(3, 49, 3))
VIDEO_OFFSETS = tuple(range(6, 49, 6))


@dataclass(frozen=True)
class Episode:
    task: str
    number: int
    data: Path
    instructions: Path


class TextEmbeddingCache:
    """Lazy reader for the disk-backed BF16 T5 cache."""

    def __init__(self, metadata_path: Path) -> None:
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        if metadata.get("cache_format") not in (None, "variable_length_per_prompt_disk_v2"):
            raise ValueError(f"Unsupported T5 cache format: {metadata.get('cache_format')}")
        self.prompts = tuple(metadata["prompts"])
        self.index = tuple(tuple(map(int, item)) for item in metadata["embedding_index"])
        self.prompt_to_index = {prompt: index for index, prompt in enumerate(self.prompts)}
        self.payload_path = metadata_path.with_suffix(".bin")
        if not self.payload_path.is_file():
            raise FileNotFoundError(self.payload_path)
        self._payload: np.memmap | None = None

    def get(self, prompt: str) -> torch.Tensor:
        if self._payload is None:
            self._payload = np.memmap(self.payload_path, dtype=np.uint16, mode="c")
        offset, length, width = self.index[self.prompt_to_index[prompt]]
        values = np.asarray(self._payload[offset : offset + length * width])
        if values.size != length * width:
            raise RuntimeError(f"Truncated T5 cache payload: {self.payload_path}")
        return torch.from_numpy(values.reshape(length, width)).view(torch.bfloat16)


class Clean50Dataset(Dataset):
    """Samples one Clean50 anchor with the ME-Dex temporal contract."""

    def __init__(
        self,
        root: str | Path,
        text_cache: str | Path,
        quality_manifest: str | Path,
        *,
        samples_per_episode: int = 10,
        image_size: tuple[int, int] = (384, 320),
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.image_size = image_size
        self.text_cache = TextEmbeddingCache(Path(text_cache))
        manifest = Path(quality_manifest)
        opener = gzip.open if manifest.suffix == ".gz" else open
        with opener(manifest, "rt", encoding="utf-8") as stream:
            records = json.load(stream)
        raw_quality = records.get("episodes", records)
        self.quality = {}
        for key, value in raw_quality.items():
            path = Path(key)
            self.quality[str(path)] = value
            self.quality[str(path.resolve())] = value
        episodes: list[Episode] = []
        task_root = self.root / "production" if (self.root / "production").is_dir() else self.root
        for task_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
            candidates = (
                task_dir / "tactile_replay_aloha_clean50_tfa2_full",
                task_dir / "tactile_replay_aloha_clean50_batch",
                task_dir,
            )
            data_dir = next(
                (candidate / "data" for candidate in candidates if (candidate / "data").is_dir()),
                candidates[0] / "data",
            )
            instruction_dir = data_dir.parent / "instructions"
            for path in sorted(data_dir.glob("episode*.hdf5"), key=self._number):
                number = self._number(path)
                episodes.append(Episode(task_dir.name, number, path, instruction_dir / f"episode{number}.json"))
        expected = 50 * 50
        if len(episodes) != expected:
            raise ValueError(f"Clean50 requires {expected} episodes, found {len(episodes)}")
        self.episodes = tuple(episodes)
        self.length = len(episodes) * samples_per_episode

    @staticmethod
    def _number(path: Path) -> int:
        match = re.fullmatch(r"episode(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Invalid episode name: {path.name}")
        return int(match.group(1))

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _decode_frame(dataset: h5py.Dataset, index: int) -> np.ndarray:
        value = dataset[index]
        if isinstance(value, (bytes, np.bytes_)):
            encoded = np.frombuffer(value, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Invalid camera frame")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"Unsupported camera frame: {frame.shape}/{frame.dtype}")
        return frame

    def _frame(self, handle: h5py.File, index: int) -> torch.Tensor:
        head, left, right = [self._decode_frame(handle["observation"][camera]["rgb"], index) for camera in CAMERAS]
        left = cv2.resize(left, (head.shape[1] // 2, head.shape[0] // 2))
        right = cv2.resize(right, (head.shape[1] // 2, head.shape[0] // 2))
        frame = np.concatenate((head, np.concatenate((left, right), axis=1)), axis=0)
        height, width = frame.shape[:2]
        scale = min(self.image_size[0] / height, self.image_size[1] / width)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        output = np.zeros((*self.image_size, 3), dtype=np.uint8)
        y = (self.image_size[0] - resized.shape[0]) // 2
        x = (self.image_size[1] - resized.shape[1]) // 2
        output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return torch.from_numpy(output).permute(2, 0, 1).float() / 255

    def _anchors(self, episode: Episode) -> list[int]:
        key = str(episode.data.resolve())
        record = self.quality.get(key)
        if record is None:
            record = self.quality.get(episode.data.relative_to(self.root).as_posix())
        if record is None:
            raise KeyError(f"Episode missing from quality manifest: {key}")
        return record["training_anchors"]

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        episode = random.choice(self.episodes)
        anchor = random.choice(self._anchors(episode))
        action_indices = [anchor + offset for offset in ACTION_OFFSETS]
        video_indices = [anchor + offset for offset in VIDEO_OFFSETS]
        with h5py.File(episode.data, "r") as handle:
            qpos = handle["joint_action/vector"]
            state = np.asarray(qpos[anchor], dtype=np.float32)
            actions = np.stack([np.asarray(qpos[index], dtype=np.float32) for index in action_indices])
            first_frame = self._frame(handle, anchor)
            video_frames = torch.stack([self._frame(handle, index) for index in video_indices])
            force = handle["tactile_force_field/force_canonical"]
            support = handle["tactile_force_field/support_mask"]
            observed = np.stack([np.asarray(force[index], dtype=np.float32) for index in (anchor - 1, anchor)])
            future = np.stack([np.asarray(force[index], dtype=np.float32) for index in action_indices])
            if support.ndim == 3:
                static_support = np.asarray(support, dtype=np.bool_)
                observed_support = np.broadcast_to(static_support, (2, 4, 10, 14)).copy()
                future_support = np.broadcast_to(static_support, (16, 4, 10, 14)).copy()
            elif support.ndim == 4:
                observed_support = np.stack([np.asarray(support[index], dtype=np.bool_) for index in (anchor - 1, anchor)])
                future_support = np.stack([np.asarray(support[index], dtype=np.bool_) for index in action_indices])
                static_support = np.asarray(support[0], dtype=np.bool_)
                selected_support = np.concatenate((observed_support, future_support), axis=0)
                if not np.array_equal(selected_support, np.broadcast_to(static_support, selected_support.shape)):
                    raise ValueError(f"Tactile support changes within a training window: {episode.data}")
            else:
                raise ValueError(f"Unsupported tactile support mask shape: {support.shape}")
            observed *= observed_support[:, :, None]
            future *= future_support[:, :, None]
            intervals = np.asarray(handle["tactile_force_field/interval_seconds"][: action_indices[-1] + 1], dtype=np.float64)
        absolute_time = np.cumsum(intervals)
        times = (absolute_time[np.asarray([anchor - 1, anchor, *action_indices])] - absolute_time[anchor - 1]).astype(np.float32)
        instructions = json.loads(episode.instructions.read_text(encoding="utf-8"))["seen"]
        prompt = random.choice([text for text in instructions if text in self.text_cache.prompt_to_index])
        return {
            "first_frame": first_frame,
            "video_frames": video_frames,
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "language_embeddings": self.text_cache.get(prompt),
            "tactile_observed_source": torch.from_numpy(np.moveaxis(observed, 1, 0).copy()),
            "tactile_observed_support_source": torch.from_numpy(np.moveaxis(observed_support, 1, 0).copy()),
            "tactile_future_source": torch.from_numpy(np.moveaxis(future, 1, 0).copy()),
            "tactile_future_support_source": torch.from_numpy(np.moveaxis(future_support, 1, 0).copy()),
            "tactile_observed_frame_times": torch.from_numpy(times[:2].copy()),
            "tactile_future_query_times": torch.from_numpy(times[2:].copy()),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    output = {key: torch.stack([sample[key] for sample in batch]) for key in batch[0] if key != "language_embeddings"}
    embeddings = [sample["language_embeddings"] for sample in batch]
    padded = embeddings[0].new_zeros((len(batch), 512, embeddings[0].shape[-1]))
    for index, embedding in enumerate(embeddings):
        padded[index, : min(embedding.shape[0], 512)] = embedding[:512]
    output["language_embeddings"] = padded
    return output
