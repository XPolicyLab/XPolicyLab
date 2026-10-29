import hashlib
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from .robotwin_depth_dataset import RobotWinDepthVideoDataset
from .robotwin_instructions import RobotWinRawInstructionMixin
from .robotwin_rgb_dataset import PROMPT_TEMPLATE


class RobotWinDepthJointVideoDataset(
    RobotWinRawInstructionMixin, RobotWinDepthVideoDataset
):
    """Raw RoboTwin depth windows with aligned action chunks for PrismWAM joint training."""

    def __init__(
        self,
        *args,
        action_key: str = "joint_action/vector",
        proprio_key: str = "joint_action/vector",
        action_horizon: int | None = None,
        allow_shorter_action_horizon: bool = False,
        action_dim: int = 14,
        normalize_actions: bool = True,
        action_std_floor: float = 1e-06,
        action_stats_cache_path: str | None = None,
        action_stats_cache_policy: str = "strict",
        action_start_offset: int = 0,
        proprio_start_offset: int = 0,
        prompt_template: str = PROMPT_TEMPLATE,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
        text_encoder_id: str = "wan22ti2v5b",
        humanize_task_name: bool = False,
        instruction_source: str = "task",
        instruction_type: str = "seen",
        instruction_selection: str = "episode_hash",
        task_instruction_dir: str | None = None,
        objects_description_dir: str | None = None,
        max_instructions_per_episode: int | None = None,
        dataset_uid: str | None = None,
        prompt_policy_id: str = "raw_robotwin_prompt",
        **kwargs,
    ):
        num_video_frames = int(kwargs.get("num_video_frames", 9))
        frame_stride = int(kwargs.get("frame_stride", 4))
        window_span = (num_video_frames - 1) * frame_stride + 1
        self.action_horizon = (
            int(action_horizon) if action_horizon is not None else window_span - 1
        )
        aligned_horizon = self.action_horizon == window_span - 1
        shorter_horizon = (
            allow_shorter_action_horizon and 0 < self.action_horizon < window_span - 1
        )
        if not (aligned_horizon or shorter_horizon):
            raise ValueError(
                f"Raw RoboTwin depth joint training expects continuous action steps over the video window: action_horizon must be {window_span - 1}, got {self.action_horizon}."
            )
        if self.action_horizon % (num_video_frames - 1) != 0:
            raise ValueError(
                f"action_horizon={self.action_horizon} must be divisible by video transitions {num_video_frames - 1}."
            )
        self.action_key = str(action_key)
        self.proprio_key = str(proprio_key)
        self.action_dim = int(action_dim)
        self.normalize_actions = bool(normalize_actions)
        self.action_std_floor = float(action_std_floor)
        self.action_stats_cache_path = (
            None if action_stats_cache_path is None else Path(action_stats_cache_path)
        )
        self.action_stats_cache_policy = str(action_stats_cache_policy).strip().lower()
        if self.action_stats_cache_policy not in {"strict", "fixed"}:
            raise ValueError(
                f"`action_stats_cache_policy` must be one of ['fixed', 'strict'], got {action_stats_cache_policy!r}."
            )
        if (
            self.action_stats_cache_policy == "fixed"
            and self.action_stats_cache_path is None
        ):
            raise ValueError(
                "`action_stats_cache_policy=fixed` requires `action_stats_cache_path`."
            )
        self.action_start_offset = int(action_start_offset)
        self.proprio_start_offset = int(proprio_start_offset)
        if self.action_start_offset < 0 or self.proprio_start_offset < 0:
            raise ValueError(
                f"action/proprio offsets must be non-negative, got {self.action_start_offset} and {self.proprio_start_offset}."
            )
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir)
        )
        self.context_len = int(context_len)
        self.text_encoder_id = str(text_encoder_id)
        self.prompt_policy_id = str(prompt_policy_id)
        self._init_instruction_config(
            prompt_template=prompt_template,
            humanize_task_name=humanize_task_name,
            instruction_source=instruction_source,
            instruction_type=instruction_type,
            instruction_selection=instruction_selection,
            task_instruction_dir=task_instruction_dir,
            objects_description_dir=objects_description_dir,
            max_instructions_per_episode=max_instructions_per_episode,
        )
        super().__init__(*args, **kwargs)
        self._attach_instruction_candidates_to_episodes()
        self.dataset_uid = (
            str(dataset_uid)
            if dataset_uid is not None
            else str(self.root_dir.resolve())
        )
        self.action_mean = torch.zeros(self.action_dim, dtype=torch.float32)
        self.action_std = torch.ones(self.action_dim, dtype=torch.float32)
        if self.normalize_actions:
            (mean, std) = self._load_or_compute_action_stats()
            self.action_mean = mean
            self.action_std = std

    def _num_samples_for_episode(self, length: int) -> int:
        """Keep only anchors with at least one non-padding action target."""
        video_samples = super()._num_samples_for_episode(length)
        action_start_span = max(int(length) - self.action_start_offset, 0)
        action_samples = (
            action_start_span + self.sample_stride - 1
        ) // self.sample_stride
        return min(video_samples, action_samples)

    def _read_episode_length(self, path: Path) -> int:
        depth_key = f"observation/{self.cameras[0]}/depth"
        with h5py.File(path, "r") as f:
            if depth_key not in f:
                raise KeyError(f"Missing depth dataset `{depth_key}` in {path}")
            if self.action_key not in f:
                raise KeyError(f"Missing action dataset `{self.action_key}` in {path}")
            depth_len = int(f[depth_key].shape[0])
            action_len = int(f[self.action_key].shape[0])
        return min(depth_len, action_len)

    def _compute_action_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        count = 0
        total = np.zeros((self.action_dim,), dtype=np.float64)
        total_sq = np.zeros((self.action_dim,), dtype=np.float64)
        for episode in self.episodes:
            with h5py.File(episode["path"], "r") as f:
                action = np.asarray(f[self.action_key], dtype=np.float64)
            if action.ndim != 2 or action.shape[1] != self.action_dim:
                raise ValueError(
                    f"`{self.action_key}` must have shape [T,{self.action_dim}], got {action.shape} in {episode['path']}"
                )
            total += action.sum(axis=0)
            total_sq += np.square(action).sum(axis=0)
            count += int(action.shape[0])
        if count <= 0:
            raise ValueError(
                "Cannot compute action stats from an empty RoboTwin action index."
            )
        mean = total / count
        var = np.maximum(total_sq / count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(var), self.action_std_floor)
        return (
            torch.from_numpy(mean.astype(np.float32)),
            torch.from_numpy(std.astype(np.float32)),
        )

    def _action_stats_signature(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.root_dir.resolve()).encode("utf-8"))
        digest.update(str(self.action_key).encode("utf-8"))
        digest.update(str(self.action_dim).encode("utf-8"))
        digest.update(str(self.action_std_floor).encode("utf-8"))
        for episode in self.episodes:
            digest.update(str(Path(episode["path"]).resolve()).encode("utf-8"))
            digest.update(str(episode["length"]).encode("utf-8"))
        return digest.hexdigest()

    def _load_or_compute_action_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        signature = self._action_stats_signature()
        cache_path = self.action_stats_cache_path
        if self.action_stats_cache_policy == "fixed":
            if cache_path is None or not cache_path.exists():
                raise FileNotFoundError(
                    f"Fixed action stats cache not found: {cache_path}. Fixed mode never recomputes normalization statistics."
                )
            payload = torch.load(cache_path, map_location="cpu")
            mean = payload.get("mean")
            std = payload.get("std")
            if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
                raise ValueError(
                    f"Fixed action stats cache is missing tensor mean/std: {cache_path}"
                )
            if tuple(mean.shape) != (self.action_dim,) or tuple(std.shape) != (
                self.action_dim,
            ):
                raise ValueError(
                    f"Fixed action stats shape mismatch: mean={tuple(mean.shape)} std={tuple(std.shape)} action_dim={self.action_dim}."
                )
            if not bool(torch.isfinite(mean).all()) or not bool(
                torch.isfinite(std).all()
            ):
                raise ValueError(
                    f"Fixed action stats contain non-finite values: {cache_path}"
                )
            if not bool((std > 0).all()):
                raise ValueError(
                    f"Fixed action stats must have positive std values: {cache_path}"
                )
            return (mean.float(), std.float())
        if cache_path is not None and cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            if payload.get("signature") == signature:
                return (payload["mean"].float(), payload["std"].float())
        (mean, std) = self._compute_action_stats()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
            torch.save(
                {
                    "signature": signature,
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                    "action_key": self.action_key,
                    "action_dim": self.action_dim,
                    "num_episodes": len(self.episodes),
                },
                tmp_path,
            )
            os.replace(tmp_path, cache_path)
        return (mean, std)

    def _read_vector_window(
        self,
        h5_file: h5py.File,
        key: str,
        start: int,
        length: int,
        start_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ds = h5_file[key]
        read_start = int(start) + int(start_offset)
        ds_length = min(int(length), int(ds.shape[0]))
        if ds_length <= 0:
            raise ValueError(f"Cannot read `{key}` window from an empty sequence.")
        if read_start < ds_length:
            stop = min(read_start + self.action_horizon, ds_length)
            values = np.asarray(ds[read_start:stop], dtype=np.float32)
        else:
            values = np.asarray(ds[ds_length - 1 : ds_length], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.action_dim:
            raise ValueError(
                f"`{key}` must have shape [T,{self.action_dim}], got {values.shape}"
            )
        pad_mask = torch.tensor(
            [read_start + i >= ds_length for i in range(self.action_horizon)],
            dtype=torch.bool,
        )
        if values.shape[0] < self.action_horizon:
            pad = np.repeat(values[-1:], self.action_horizon - values.shape[0], axis=0)
            values = np.concatenate([values, pad], axis=0)
        tensor = torch.from_numpy(values[: self.action_horizon]).contiguous()
        if self.normalize_actions:
            tensor = (tensor - self.action_mean) / self.action_std
        return (tensor, pad_mask)

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            return (None, None)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = (
            self.text_embedding_cache_dir
            / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. Build it with `scripts/prepare_data.py text`."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L,D], got {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context length mismatch in {cache_path}: expected {self.context_len}, got {context.shape[0]}"
            )
        context = context.clone()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        return (context, context_mask)

    def __getitem__(self, idx: int):
        (episode, start) = self._locate(idx)
        length = int(episode["length"])
        with h5py.File(episode["path"], "r") as f:
            camera_videos = [
                self._read_depth_window(f, camera=camera, start=start, length=length)
                for camera in self.cameras
            ]
            (action, action_is_pad) = self._read_vector_window(
                f,
                self.action_key,
                start=start,
                length=length,
                start_offset=self.action_start_offset,
            )
            (proprio, proprio_is_pad) = self._read_vector_window(
                f,
                self.proprio_key,
                start=start,
                length=length,
                start_offset=self.proprio_start_offset,
            )
        video_tchw = self._layout_video(camera_videos)
        image_is_pad = torch.tensor(
            [start + offset >= length for offset in self.frame_offsets],
            dtype=torch.bool,
        )
        prompt = self._prompt_for_episode(episode, start=start)
        (context, context_mask) = self._get_cached_text_context(prompt)
        sample = {
            "video": video_tchw.permute(1, 0, 2, 3).contiguous(),
            "action": action,
            "proprio": proprio,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
            "prompt": prompt,
            "idx": int(idx),
            "task": episode["task"],
            "phase": episode["phase"],
            "episode_index": int(episode["episode_index"]),
            "start_index": int(start),
        }
        if context is not None:
            sample["context"] = context
            sample["context_mask"] = context_mask
        return sample
