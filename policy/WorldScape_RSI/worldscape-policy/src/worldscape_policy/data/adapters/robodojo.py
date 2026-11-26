"""Native HDF5 adapter for RoboDojo ARX-X5 joint demonstrations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from worldscape_policy.data.adapters.common import (
    _metadata_task,
    _optional_dependency,
)
from worldscape_policy.data.adapters.hdf5 import NativeHDF5Dataset
from worldscape_policy.data.robodojo_chunks import (
    task_chunk_counts,
    chunk_policy_options,
    task_chunk_limit,
    task_id_from_path as _task_id_from_path,
)
from worldscape_policy.data.schema import EventSample
from worldscape_policy.robodojo_joint import (
    JOINT_DIM,
    joint_actions_relative_to_chunk_anchor,
    load_joint_statistics,
    normalize_joint,
)
from worldscape_policy.robodojo_velocity import (
    VELOCITY_DIM,
    arm_joint_velocity,
    load_velocity_statistics,
    normalize_velocity,
)


_CAMERA_PATHS = {
    "observations.images.head": "vision/cam_head/colors",
    "observations.images.wrist_0_right": "vision/cam_right_wrist/colors",
    "observations.images.wrist_1_left": "vision/cam_left_wrist/colors",
}
_STATE_JOINT_PATHS = (
    "state/left_arm_joint_states",
    "state/left_ee_joint_states",
    "state/right_arm_joint_states",
    "state/right_ee_joint_states",
)
_ACTION_JOINT_PATHS = (
    "action/left_arm_joint_states",
    "action/left_ee_joint_states",
    "action/right_arm_joint_states",
    "action/right_ee_joint_states",
)


def _required_dataset(handle: Any, name: str) -> Any:
    if name not in handle:
        raise KeyError(f"RoboDojo episode is missing required dataset {name!r}")
    return handle[name]


def _embodiment_id_from_path(path: Path) -> str:
    if path.parent.name == "data" and len(path.parents) >= 2:
        return path.parents[1].name
    return "arx_x5"


def _pack_joint_array(handle: Any, paths: tuple[str, ...]) -> np.ndarray:
    values: list[np.ndarray] = []
    length: int | None = None
    for path, width in zip(paths, (6, 1, 6, 1), strict=True):
        value = np.asarray(_required_dataset(handle, path)[()])
        if value.ndim == 1 and width == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[-1] != width:
            raise ValueError(
                f"RoboDojo field {path!r} must have shape [T,{width}], "
                f"got {value.shape}"
            )
        if length is not None and len(value) != length:
            raise ValueError("RoboDojo joint fields have inconsistent lengths")
        length = len(value)
        values.append(value)
    return np.concatenate(values, axis=-1).astype(np.float32, copy=False)


class RoboDojoJointHDF5Dataset(NativeHDF5Dataset):
    """Read official-format RoboDojo demonstrations and their stored actions.

    Images decode with XPolicyLab; BGR model inputs are explicit opt-in.
    Optional velocity features use only current/past arm observations.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        stats_path: str | Path,
        normalization: str | None = None,
        input_color_order: str = "RGB",
        relative_joint_action: bool = True,
        normalize_state: bool = True,
        normalize_action: bool = True,
        include_joint_velocity: bool = False,
        velocity_stats_path: str | Path | None = None,
        precomputed_index_path: str | Path | None = None,
        verify_index_files: bool = False,
        chunk_policy: str = "uniform",
        embodiment: str = "arx_x5_joint",
        file_pattern: str = "**/episode_*.hdf5",
        visual_prompt: str = "none",
        **kwargs: Any,
    ) -> None:
        if input_color_order not in {"RGB", "BGR"}:
            raise ValueError("input_color_order must be RGB or BGR")
        self.input_color_order = input_color_order
        self.chunk_policy = chunk_policy
        horizon = kwargs.get("action_horizon", 24)
        chunk_policy_options(chunk_policy, horizon)
        if chunk_policy == "task_aware":
            memory_chunks, default_chunks = task_chunk_counts(horizon)
        if chunk_policy == "task_aware" and (
            not kwargs.get("temporal_packing", False)
            or kwargs.get("max_chunk_size", 4) != memory_chunks
            or kwargs.get("action_horizon", 24) not in {24, 48}
        ):
            raise ValueError(
                "RoboDojo task_aware requires temporal_packing=True, "
                f"max_chunk_size={memory_chunks}, and action_horizon=24 or 48"
            )
        self.joint_stats = load_joint_statistics(stats_path, normalization=normalization)
        self.relative_joint_action = bool(relative_joint_action)
        self.normalize_state = bool(normalize_state)
        self.normalize_action = bool(normalize_action)
        self.include_joint_velocity = bool(include_joint_velocity)
        self.velocity_stats = (
            load_velocity_statistics(velocity_stats_path)
            if self.include_joint_velocity
            else None
        )
        stats_relative = bool(self.joint_stats["relative_joint_action"])
        if self.relative_joint_action != stats_relative:
            raise ValueError(
                "relative_joint_action does not match the statistics artifact"
            )
        if visual_prompt not in {"none", "initial", "goal", "demo"}:
            raise ValueError(
                "RoboDojo visual_prompt must be none, initial, goal, or demo"
            )
        self.robodojo_visual_prompt = visual_prompt
        self._index_file_pattern = file_pattern
        self._index_max_episodes = kwargs.get("max_episodes")
        self._precomputed_index = None
        if precomputed_index_path is not None:
            from worldscape_policy.data.robodojo_index import read_index

            self._precomputed_index = read_index(
                precomputed_index_path, verify_files=verify_index_files
            )
            if "episode_manifest" in kwargs or kwargs.get("data_roots") is not None:
                raise ValueError("precomputed RoboDojo indices require one data_root")
            kwargs["episode_manifest"] = self._precomputed_index["manifest"]
        super().__init__(
            data_root,
            embodiment=embodiment,
            file_pattern=file_pattern,
            # Initial context is a persistent single image, so it intentionally
            # reuses the native goal-image transport while retaining distinct
            # dataset/config semantics and provenance.
            visual_prompt="goal" if visual_prompt == "initial" else visual_prompt,
            **kwargs,
        )
        self._prompt_episode_indices = self._build_prompt_episode_indices()
        if self.chunk_policy == "task_aware":
            memory_episodes = sum(
                task_chunk_limit(path, self.action_horizon) == memory_chunks
                for path in self.paths
            )
            print(
                f"RoboDojo task_aware: {memory_episodes} episodes at up to {memory_chunks} chunks, "
                f"{len(self.paths) - memory_episodes} episodes at up to {default_chunks} chunks",
                flush=True,
            )

    def _decode_model_image(self, value: object) -> np.ndarray:
        """Decode through XPolicyLab; opt in to legacy BGR checkpoint inputs.

        The shared decoder owns JPEG markers and returns RGB for either stored
        format. input_color_order=BGR deliberately adapts that RGB result to the
        historical RoboDojo training contract; it is not a decode correction.
        """
        from XPolicyLab.utils.process_data import decode_image_bit

        if value is None:
            raise TypeError("unplanned encoded image value: NoneType")
        rgb = decode_image_bit(value)
        return np.ascontiguousarray(rgb[..., ::-1] if self.input_color_order == "BGR" else rgb)

    def _decode_model_sequence(self, values: np.ndarray) -> np.ndarray:
        return np.stack([self._decode_model_image(value) for value in values])

    def _episode_max_chunk_size(self, episode_index: int) -> int:
        if self.chunk_policy == "task_aware":
            return task_chunk_limit(self.paths[episode_index], self.action_horizon)
        return super()._episode_max_chunk_size(episode_index)

    def _index_cache_key(self) -> str:
        original = super()._index_cache_key()
        options = chunk_policy_options(self.chunk_policy, self.action_horizon)
        if not options:
            return original
        return hashlib.sha256(
            (original + json.dumps(options, sort_keys=True)).encode("utf-8")
        ).hexdigest()

    def _build_index(self) -> list[tuple[int, int]] | np.ndarray:
        if self._precomputed_index is None:
            return super()._build_index()
        from worldscape_policy.data.robodojo_index import index_options

        artifact = self._precomputed_index
        # These values are set by the base constructor before _build_index.
        if index_options(self) != artifact["manifest"]["options"]:
            raise ValueError(
                "RoboDojo precomputed index configuration mismatch; regenerate "
                "for this data_root/chunk/policy/stride configuration"
            )
        self._sampling_progress = artifact["sampling_progress"]
        print(
            f"RoboDojo: loaded precomputed index ({len(self.paths)} episodes, "
            f"{len(artifact['index'])} samples)", flush=True,
        )
        return artifact["index"]

    def length_signature(self, index: int) -> tuple[int, ...]:
        if self._precomputed_index is not None:
            return tuple(int(value) for value in self._precomputed_index["lengths"][index])
        return super().length_signature(index)

    def memory_task(self, index: int) -> bool:
        """Classify an indexed sample without loading its episode payload."""
        from worldscape_policy.data.robodojo_chunks import MEMORY_TASKS

        episode_index = int(self.index[index][0])
        return _task_id_from_path(self.paths[episode_index]) in MEMORY_TASKS

    def _build_prompt_episode_indices(self) -> dict[int, int]:
        groups: dict[tuple[str, str], list[int]] = {}
        for index, path in enumerate(self.paths):
            key = (_task_id_from_path(path), _embodiment_id_from_path(path))
            groups.setdefault(key, []).append(index)
        return {
            index: indices[(offset + 1) % len(indices)]
            for indices in groups.values()
            for offset, index in enumerate(indices)
        }

    def _read_visual_prompt_arrays(
        self, episode_index: int
    ) -> dict[str, np.ndarray]:
        prompt_mode = self.robodojo_visual_prompt
        if prompt_mode == "none":
            return {}
        prompt_index = (
            episode_index
            if prompt_mode == "initial"
            else self._prompt_episode_indices[episode_index]
        )
        if prompt_mode in {"goal", "demo"} and prompt_index == episode_index:
            task_id = _task_id_from_path(self.paths[episode_index])
            raise ValueError(
                "RoboDojo goal/demo training requires at least two episodes "
                f"for task {task_id!r}"
            )
        prompt_path = self.paths[prompt_index]
        h5py = _optional_dependency("h5py", "train")
        with h5py.File(prompt_path, "r") as handle:
            head_dataset = _required_dataset(
                handle, _CAMERA_PATHS["observations.images.head"]
            )
            if len(head_dataset) == 0:
                raise ValueError(
                    f"RoboDojo prompt episode has no head frames: {prompt_path}"
                )
            # Single-image modes only need one compressed frame. Avoid reading
            # and decoding the complete episode for every training sample.
            if prompt_mode == "initial":
                head_payload = np.asarray(head_dataset[0:1])
            elif prompt_mode == "goal":
                head_payload = np.asarray(head_dataset[-1:])
            else:
                head_payload = np.asarray(head_dataset[()])
            head = self._decode_model_sequence(head_payload)
        source_episode_id = str(
            self.episodes[prompt_index].get("episode_index", prompt_index)
        )
        source_session_id = str(
            self.episodes[prompt_index].get("session_id", source_episode_id)
        )
        transport_name = "goal" if prompt_mode == "initial" else prompt_mode
        metadata = {
            f"{transport_name}_task_id": np.asarray(_task_id_from_path(prompt_path)),
            f"{transport_name}_embodiment": np.asarray(self.embodiment),
            f"{transport_name}_source_episode_id": np.asarray(source_episode_id),
            f"{transport_name}_source_session_id": np.asarray(source_session_id),
        }
        if prompt_mode == "initial":
            return {"goal_image": head[0], **metadata}
        if prompt_mode == "goal":
            return {"goal_image": head[-1], **metadata}
        indices = np.linspace(0, len(head) - 1, self.context_video_len)
        indices = indices.round().astype(np.int64)
        return {"demo_video": head[indices], **metadata}

    def _read_arrays(self, episode_index: int, *, load_images: bool = True) -> dict[str, np.ndarray]:
        h5py = _optional_dependency("h5py", "train")
        path = self.paths[episode_index]
        with h5py.File(path, "r") as handle:
            arrays = {
                target: np.empty(
                    len(_required_dataset(handle, source)),
                    dtype=object,
                )
                for target, source in _CAMERA_PATHS.items()
            }
            if load_images:
                for target, source in _CAMERA_PATHS.items():
                    payload = np.asarray(_required_dataset(handle, source)[()])
                    arrays[target] = (self._decode_model_sequence(payload)
                                     if self.cache_video_decode or payload.ndim == 4 else payload)
            arrays["robot_state"] = _pack_joint_array(handle, _STATE_JOINT_PATHS)
            arrays["actions"] = _pack_joint_array(handle, _ACTION_JOINT_PATHS)
            if self.include_joint_velocity:
                # Compute on the complete raw episode, before chunk selection.
                frequency = _required_dataset(handle, "additional_info/frequency")[()]
                velocity = arm_joint_velocity(arrays["robot_state"], frequency)
                arrays["robot_state"] = np.concatenate(
                    (arrays["robot_state"], velocity), axis=-1
                )
            instruction = np.asarray(_required_dataset(handle, "instruction")[()])
        arrays.update(self._read_visual_prompt_arrays(episode_index))
        temporal_lengths = {
            key: len(value)
            for key, value in arrays.items()
            if key.startswith("observations.images.")
            or key in {"robot_state", "actions"}
        }
        if len(set(temporal_lengths.values())) != 1:
            raise ValueError(
                "RoboDojo episode has inconsistent temporal lengths: "
                f"{temporal_lengths}"
            )
        arrays["high_level_instruction"] = instruction
        arrays["task_id"] = np.asarray(_task_id_from_path(path))
        _metadata_task(arrays, self.episodes[episode_index], self.tasks)
        return arrays

    def _read_index_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        cached = self._index_arrays.get(episode_index)
        if cached is not None:
            return cached
        h5py = _optional_dependency("h5py", "train")
        path = self.paths[episode_index]
        lengths: dict[str, int] = {}
        with h5py.File(path, "r") as handle:
            for source in (
                *_CAMERA_PATHS.values(), *_STATE_JOINT_PATHS, *_ACTION_JOINT_PATHS,
            ):
                dataset = _required_dataset(handle, source)
                lengths[source] = int(dataset.shape[0])
            instruction = np.asarray(_required_dataset(handle, "instruction")[()])
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"RoboDojo episode has inconsistent temporal lengths: {lengths}"
            )
        length = next(iter(lengths.values()))
        arrays = {
            key: np.empty((length,), dtype=np.uint8) for key in _CAMERA_PATHS
        }
        state_dim = JOINT_DIM + (VELOCITY_DIM if self.include_joint_velocity else 0)
        arrays["robot_state"] = np.empty((length, state_dim), dtype=np.float32)
        arrays["actions"] = np.empty((length, JOINT_DIM), dtype=np.float32)
        arrays["high_level_instruction"] = instruction
        arrays["task_id"] = np.asarray(_task_id_from_path(path))
        _metadata_task(
            arrays,
            self.episodes[episode_index],
            self.tasks,
            length=length,
        )
        self._index_arrays[episode_index] = arrays
        return arrays

    def _build_indexed_sample(
        self,
        index: int,
        arrays: dict[str, np.ndarray],
        episode_index: int,
        anchor: int,
    ) -> EventSample:
        sample = super()._build_indexed_sample(
            index, arrays, episode_index, anchor
        )
        if self.robodojo_visual_prompt == "initial":
            if sample.goal_prompt_metadata is None:
                raise RuntimeError("RoboDojo initial prompt is missing metadata")
            sample = replace(
                sample,
                goal_prompt_metadata=replace(
                    sample.goal_prompt_metadata,
                    trusted_same_sample=True,
                ),
                provenance={
                    **sample.provenance,
                    "visual_prompt_semantics": "initial_image",
                },
            )
        actions = sample.actions
        if self.relative_joint_action:
            actions = joint_actions_relative_to_chunk_anchor(
                actions,
                sample.robot_state[:, :JOINT_DIM],
                action_horizon=self.action_horizon,
            )
        if self.normalize_action:
            actions = normalize_joint(actions, self.joint_stats, "action")
        state = sample.robot_state
        if self.normalize_state:
            positions = normalize_joint(state[:, :JOINT_DIM], self.joint_stats, "state")
            if self.include_joint_velocity:
                assert self.velocity_stats is not None
                velocity = normalize_velocity(
                    state[:, JOINT_DIM:], self.velocity_stats["velocity"]
                )
                state = np.concatenate((positions, velocity), axis=-1)
            else:
                state = positions
        transformed = replace(
            sample,
            actions=actions,
            robot_state=state,
            provenance={
                **sample.provenance,
                "action_space": "robodojo_joint_14d",
                "relative_joint_action": self.relative_joint_action,
                "joint_normalization": self.joint_stats.get("normalization", "q99"),
                "image_color_order": self.input_color_order,
                "action_label_source": "stored_action",
                **(
                    {"include_joint_velocity": True}
                    if self.include_joint_velocity else {}
                ),
            },
        )
        transformed.validate()
        return transformed


__all__ = ["RoboDojoJointHDF5Dataset"]
