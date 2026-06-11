"""
LeRobot-format dataset loader.

Expected directory structure (lerobot v2):
    <data_path>/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl          (optional)
    │   └── tasks.jsonl             (optional)
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    ├── videos/
    │   └── chunk-000/
    │       ├── observation.images.cam_high/
    │       │   ├── episode_000000.mp4
    │       │   └── ...
    │       └── ...
    └── t5_embedding/               (optional, per-episode .pt files)
        ├── episode_000000.pt
        └── ...

T5 embeddings can come from:
    1. <data_path>/t5_embedding/episode_XXXXXX.pt   (per-episode)
    2. <data_path>/meta/t5_text_embeds.pt           (dict: task_index -> tensor)
    3. <t5_embed_path>                              (explicit path)
"""

import glob
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LeRobotDataset(Dataset):
    def __init__(
        self,
        data_path,
        delta_info=None,
        delta_frames=None,
        video_backend="pyav",
        transform=None,
        t5_embed_path=None,
        robotype="aloha",
        data_size=None,
    ):
        if isinstance(data_path, (list, tuple)):
            data_path = data_path[0]
        self.root = data_path
        self.action_horizon = (delta_info or {}).get("action", 1)
        self.frame_offsets = delta_frames or {}
        self.video_backend = video_backend
        self.transform = transform
        self.robotype = robotype

        self.episodes: list[dict] = []
        self._load_episodes()

        self.t5_embeddings = self._load_t5_embeddings(t5_embed_path)
        self.per_episode_t5 = self._load_per_episode_t5()

        self._build_sample_index()

        if data_size is not None and data_size < len(self.samples):
            self.samples = self.samples[:data_size]

    def _load_episodes(self):
        data_dir = os.path.join(self.root, "data")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data dir not found: {data_dir}")

        parquet_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {data_dir}")

        for pf in parquet_files:
            df = pd.read_parquet(pf)
            ep_idx = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else len(self.episodes)

            actions = np.stack(df["action"].values).astype(np.float32)
            states = np.stack(df["observation.state"].values).astype(np.float32)

            rel = os.path.relpath(pf, data_dir)
            chunk = os.path.dirname(rel)
            ep_name = os.path.splitext(os.path.basename(pf))[0]

            task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0

            t5_embed = None
            if "t5_embedding" in df.columns:
                raw = df["t5_embedding"].iloc[0]
                if isinstance(raw, np.ndarray):
                    t5_embed = torch.from_numpy(raw).float()
                elif isinstance(raw, (list, tuple)):
                    t5_embed = torch.tensor(raw, dtype=torch.float32)

            self.episodes.append(
                {
                    "index": ep_idx,
                    "length": len(df),
                    "actions": actions,
                    "states": states,
                    "chunk": chunk,
                    "name": ep_name,
                    "task_index": task_index,
                    "t5_embed": t5_embed,
                }
            )

        if os.environ.get("RANK", "0") == "0" and os.environ.get("WA_VERBOSE_LOAD", "0") == "1":
            print(f"Loaded {len(self.episodes)} episodes from {self.root}")

    def _load_t5_embeddings(self, explicit_path=None):
        candidates = []
        if explicit_path:
            candidates.append(explicit_path)
        candidates += [
            os.path.join(self.root, "meta", "t5_text_embeds.pt"),
            os.path.join(self.root, "meta", "text_embeddings.pt"),
            os.path.join(self.root, "t5_text_embeds.pt"),
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                data = torch.load(path, map_location="cpu", weights_only=False)
                return data
        return None

    def _load_per_episode_t5(self):
        """Load per-episode T5 embeddings from <data_path>/t5_embedding/episode_XXXXXX.pt."""
        t5_dir = os.path.join(self.root, "t5_embedding")
        if not os.path.isdir(t5_dir):
            return {}

        result = {}
        for ep in self.episodes:
            pt_path = os.path.join(t5_dir, f"{ep['name']}.pt")
            if os.path.isfile(pt_path):
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                if isinstance(data, torch.Tensor):
                    result[ep["index"]] = data.float()
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, torch.Tensor):
                            result[ep["index"]] = v.float()
                            break
        return result

    def _build_sample_index(self):
        max_offset = 0
        if self.frame_offsets:
            max_offset = max(max(offsets) for offsets in self.frame_offsets.values())

        self.samples: list[tuple[int, int]] = []
        for ep_i, ep in enumerate(self.episodes):
            ep_len = ep["length"]
            need = max(self.action_horizon, max_offset + 1)
            valid = ep_len - need
            if valid <= 0:
                continue
            for start in range(valid):
                self.samples.append((ep_i, start))

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found (episodes too short for "
                f"action_horizon={self.action_horizon}, max_offset={max_offset})"
            )

    def _read_video_frames(self, episode: dict, view_key: str, frame_indices: list[int]):
        video_path = os.path.join(
            self.root, "videos", episode["chunk"], view_key, f"{episode['name']}.mp4"
        )

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        if self.video_backend == "decord":
            return self._read_frames_decord(video_path, frame_indices)
        return self._read_frames_pyav(video_path, frame_indices)

    @staticmethod
    def _read_frames_pyav(video_path: str, frame_indices: list[int]) -> torch.Tensor:
        import av

        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        min_idx = min(frame_indices)
        max_idx = max(frame_indices)
        target = set(frame_indices)
        collected: dict[int, np.ndarray] = {}

        if min_idx > 30:
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            seek_sec = max(0.0, (min_idx - 5) / fps)
            container.seek(int(seek_sec * 1_000_000), any_frame=False)

            frame_counter = None
            for frame in container.decode(stream):
                if frame_counter is None:
                    frame_counter = round(float(frame.pts * stream.time_base) * fps) if frame.pts is not None else min_idx
                if frame_counter in target:
                    collected[frame_counter] = frame.to_ndarray(format="rgb24")
                if frame_counter >= max_idx:
                    break
                frame_counter += 1
        else:
            for i, frame in enumerate(container.decode(stream)):
                if i in target:
                    collected[i] = frame.to_ndarray(format="rgb24")
                if i >= max_idx:
                    break
        container.close()

        frames = []
        for idx in frame_indices:
            if idx in collected:
                frames.append(collected[idx])
            elif frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        return torch.from_numpy(np.stack(frames))

    @staticmethod
    def _read_frames_decord(video_path: str, frame_indices: list[int]) -> torch.Tensor:
        from decord import VideoReader

        vr = VideoReader(video_path)
        safe_indices = [min(idx, len(vr) - 1) for idx in frame_indices]
        frames = vr.get_batch(safe_indices).asnumpy()
        del vr
        return torch.from_numpy(frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _attempt in range(10):
            try:
                return self._getitem_inner(idx)
            except Exception as e:
                if _attempt == 0 and os.environ.get("RANK", "0") == "0":
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Dataset error at idx={idx}, retrying with random idx: {e}"
                    )
                idx = random.randint(0, len(self.samples) - 1)
        return self._getitem_inner(idx)

    def _getitem_inner(self, idx):
        ep_i, start = self.samples[idx]
        ep = self.episodes[ep_i]

        data_dict: dict = {}

        for view_key, offsets in self.frame_offsets.items():
            frame_indices = [min(start + o, ep["length"] - 1) for o in offsets]
            data_dict[view_key] = self._read_video_frames(ep, view_key, frame_indices)

        end = start + self.action_horizon
        data_dict["action"] = torch.from_numpy(ep["actions"][start:end].copy())
        data_dict["observation.state"] = torch.from_numpy(ep["states"][start : start + 1].copy())

        t5_embed = ep.get("t5_embed")
        if t5_embed is None:
            ep_idx = ep["index"]
            if ep_idx in self.per_episode_t5:
                t5_embed = self.per_episode_t5[ep_idx]
        if t5_embed is None and self.t5_embeddings is not None:
            task_idx = ep.get("task_index", 0)
            if isinstance(self.t5_embeddings, dict):
                t5_embed = self.t5_embeddings.get(
                    task_idx, next(iter(self.t5_embeddings.values()))
                )
            elif isinstance(self.t5_embeddings, torch.Tensor):
                t5_embed = self.t5_embeddings
        if t5_embed is not None:
            data_dict["t5_embedding"] = t5_embed.clone() if isinstance(t5_embed, torch.Tensor) else torch.tensor(t5_embed)

        data_dict["robotype"] = self.robotype

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict
