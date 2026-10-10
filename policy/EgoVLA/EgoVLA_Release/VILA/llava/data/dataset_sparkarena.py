"""Lazy, provenance-bound SparkArena dataset for the EgoVLA 54-D joint path.

The target at each horizon step is read directly from the four original
``/action/*`` datasets.  No state at another timestep and no ``/mano/action``
value can participate in a target.  Images are decoded by XPolicyLab's
``decode_image_bit`` contract as RGB.  The complete 480x640 frame is resized to
384 square by the shared train/runtime helper: no crop, pad, or channel swap.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Conversion-only environments can still import helpers.
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass


POLICY_DIR = Path(__file__).resolve().parents[4]
XPL_ROOT = POLICY_DIR.parents[1]
UPSTREAM_ROOT = Path(
    os.environ.get("EGOVLA_UPSTREAM_ROOT", POLICY_DIR / "EgoVLA_Release")
).expanduser().resolve()
for import_path in (POLICY_DIR, XPL_ROOT, UPSTREAM_ROOT / "VILA", UPSTREAM_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from human_plan.dataset_preprocessing.sparkarena.process_data import (
    _as_bool,
    _get,
    _read_joint_frame,
    ACTION_DATASETS,
    CAMERA_DATASET,
    HISTORY_STRIDE,
    HISTORY_STEPS,
    JOINT_DIM,
    Joint54Normalization,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    RUNTIME_INSTRUCTIONS,
    STATE_DATASETS,
    canonical_prompt,
    extract_action_chunk,
    extract_current_state,
    load_joint54_stats,
    preprocess_vla_sparkarena_54d,
    process_head_rgb,
    verify_sparkarena_artifacts,
)


class EgoVLASparkArenaHDF5Dataset(Dataset):
    """Hydra-compatible lazy dataset for SparkArena's 54-D joint robot."""

    def __init__(
        self,
        tokenizer: Any,
        data_args: Any,
        training_args: Any = None,
        split: str = "train",
        manifest_dir: str | os.PathLike[str] | None = None,
        raw_root: str | os.PathLike[str] | None = None,
        data_skip: int = 1,
        target_stride: int = 1,
        **_: Any,
    ) -> None:
        if torch is None:
            raise RuntimeError("EgoVLASparkArenaHDF5Dataset requires PyTorch")
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"split must be train or val, got {split!r}")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args
        self.split = split
        self.data_skip = max(1, int(data_skip))
        self.target_stride = max(1, int(target_stride))
        manifest_value = manifest_dir or _get(data_args, "manifest_dir") or os.environ.get("EGOVLA_DATA_DIR")
        if not manifest_value:
            raise ValueError("SparkArena dataset requires manifest_dir or EGOVLA_DATA_DIR")
        self.manifest_dir = Path(manifest_value).expanduser().resolve()
        metadata, _ = verify_sparkarena_artifacts(self.manifest_dir)
        raw_value = raw_root or _get(data_args, "raw_root") or os.environ.get("SPARKARENA_RAW_ROOT") or metadata.get("source_root")
        self.raw_root = Path(str(raw_value)).expanduser().resolve()
        recorded_root = Path(str(metadata["source_root"])).expanduser().resolve()
        if self.raw_root != recorded_root:
            raise ValueError(f"raw root differs from provenance: {self.raw_root} != {recorded_root}")

        if str(_get(data_args, "prompt_version", "")) != "v0":
            raise ValueError("SparkArena EgoVLA requires prompt_version='v0'")
        if _as_bool(_get(data_args, "ignore_language", False)):
            raise ValueError("SparkArena canonical task instructions must not be ignored")
        self.history_steps = max(0, int(_get(data_args, "add_his_obs_step", 5)))
        self.history_stride = max(1, int(_get(data_args, "add_his_img_skip", HISTORY_STRIDE)))
        if self.history_steps and not _as_bool(_get(data_args, "add_his_imgs", True)):
            raise ValueError("history_steps > 0 requires add_his_imgs=True")
        self.predict_future_step = max(1, int(_get(data_args, "predict_future_step", 30)))
        if _get(data_args, "image_processor") is None:
            raise ValueError("data_args.image_processor is not initialized")

        manifest_path = self.manifest_dir / f"{split}.jsonl"
        self.records: list[dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                task = str(record.get("task_name", ""))
                instruction = RUNTIME_INSTRUCTIONS.get(task)
                if instruction is None or record.get("instruction") != instruction:
                    raise ValueError(f"{manifest_path}:{line_number}: non-canonical instruction")
                if record.get("prompt") != canonical_prompt(instruction):
                    raise ValueError(f"{manifest_path}:{line_number}: non-canonical v0 prompt")
                relative = Path(str(record.get("path", "")))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"{manifest_path}:{line_number}: unsafe source path")
                if int(record.get("state_dim", -1)) != JOINT_DIM or int(record.get("action_dim", -1)) != JOINT_DIM:
                    raise ValueError(f"{manifest_path}:{line_number}: not a 54-D record")
                self.records.append(record)
        if not self.records:
            raise ValueError(f"empty manifest: {manifest_path}")

        self.samples: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            self.samples.extend(
                (record_index, frame)
                for frame in range(0, int(record["frames"]), self.data_skip)
            )
        max_samples = int(os.environ.get("EGOVLA_MAX_SAMPLES", "0") or 0)
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.state_stats = load_joint54_stats(
            self.manifest_dir / "joint_state_stats.npz", "/state/*"
        )
        self.action_stats = load_joint54_stats(
            self.manifest_dir / "joint_action_stats.npz", "/action/*"
        )
        if self.state_stats.count != int(metadata["state_stats_count"]):
            raise ValueError("state stats count differs from metadata")
        if self.action_stats.count != int(metadata["action_stats_count"]):
            raise ValueError("action stats count differs from metadata")
        self._handles: dict[str, h5py.File] = {}
        self._validated: set[str] = set()

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def lengths(self) -> list[int]:
        return [self.history_steps * 128 + 64] * len(self.samples)

    @property
    def modality_lengths(self) -> list[int]:
        return self.lengths

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_validated"] = set()
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def _path(self, record: Mapping[str, Any]) -> Path:
        path = (self.raw_root / str(record["path"])).resolve()
        try:
            path.relative_to(self.raw_root)
        except ValueError as exc:
            raise ValueError(f"episode path escapes source root: {path}") from exc
        return path

    def _open(self, path: Path, expected_frames: int) -> h5py.File:
        key = str(path)
        handle = self._handles.get(key)
        if handle is None or not handle.id.valid:
            handle = h5py.File(path, "r", swmr=True)
            self._handles[key] = handle
        if key not in self._validated:
            for dataset, width in (*STATE_DATASETS, *ACTION_DATASETS):
                if dataset not in handle or handle[dataset].shape != (expected_frames, width):
                    got = handle[dataset].shape if dataset in handle else "missing"
                    raise ValueError(f"{path}:/{dataset}: expected {(expected_frames, width)}, got {got}")
            if CAMERA_DATASET not in handle or len(handle[CAMERA_DATASET]) != expected_frames:
                raise ValueError(f"{path}: invalid /{CAMERA_DATASET}")
            self._validated.add(key)
        return handle

    def _images(self, handle: h5py.File, frame: int) -> Any:
        images = []
        for history_index in range(1, self.history_steps + 1):
            query = max(0, frame - history_index * self.history_stride)
            images.append(process_head_rgb(handle[CAMERA_DATASET][query], self.data_args.image_processor))
        images.append(process_head_rgb(handle[CAMERA_DATASET][frame], self.data_args.image_processor))
        result = torch.stack(images, dim=0)
        expected = (self.history_steps + 1, 3, MODEL_HEIGHT, MODEL_WIDTH)
        if tuple(result.shape) != expected:
            raise ValueError(f"processed image history must be {expected}, got {tuple(result.shape)}")
        return result

    def _prompt(self, instruction: str) -> str:
        from human_plan.preprocessing.preprocessing import preprocess_multimodal_vla
        from human_plan.preprocessing.prompting_format import preprocess_language_instruction

        base = canonical_prompt(instruction)
        # Validate the underlying v0 template independently of history expansion.
        # Using the live data_args here is wrong when add_his_imgs=True: even a
        # zero-length history still injects the "current observation" prefix.
        base_args = copy.copy(self.data_args)
        base_args.add_his_imgs = False
        upstream_base = preprocess_language_instruction(instruction, 0, base_args)
        if upstream_base != base:
            raise ValueError("upstream v0 prompt differs from canonical SparkArena prompt")
        language = preprocess_language_instruction(instruction, self.history_steps, self.data_args)
        return preprocess_multimodal_vla(language, self.data_args)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, frame = self.samples[int(index)]
        record = self.records[record_index]
        frames = int(record["frames"])
        path = self._path(record)
        if not path.is_file():
            raise FileNotFoundError(f"raw SparkArena episode is missing: {path}")
        handle = self._open(path, frames)
        state = extract_current_state(handle, frame)
        actions, valid_steps = extract_action_chunk(
            handle,
            frame=frame,
            end=frames,
            horizon=self.predict_future_step,
            stride=self.target_stride,
        )
        normalized_state = self.state_stats.normalize(state).reshape(1, JOINT_DIM)
        normalized_actions = self.action_stats.normalize(actions)
        normalized_actions[~valid_steps] = 0.0
        action_mask = np.broadcast_to(valid_steps[:, None], normalized_actions.shape).copy()
        instruction = str(record["instruction"])
        data = preprocess_vla_sparkarena_54d(
            self._prompt(instruction),
            normalized_state,
            normalized_actions,
            action_mask,
            self.data_args.action_tokenizer,
            self.tokenizer,
            mask_input=_as_bool(_get(self.data_args, "mask_input", True)),
            mask_ignore=_as_bool(_get(self.data_args, "mask_ignore", False)),
            input_placeholder_diff_index=_as_bool(
                _get(self.data_args, "input_placeholder_diff_index", True)
            ),
            sep_query_token=_as_bool(_get(self.data_args, "sep_query_token", True)),
        )
        data["image"] = self._images(handle, frame)
        # Keep the release collator ABI while making separate EE/MANO inputs impossible.
        empty = torch.empty((1, 0), dtype=torch.float32)
        data["proprio_input_2d"] = empty.clone()
        data["proprio_input_3d"] = empty.clone()
        data["proprio_input_rot"] = empty.clone()
        data["proprio_input_handdof"] = empty.clone()
        data["proprio_input_hand_finger_tip"] = empty.clone()
        data["ee_movement_mask"] = torch.ones((1, 2), dtype=torch.float32)
        return data


def validate_sparkarena_collated_batch(
    batch: Mapping[str, Any],
    batch_size: int,
    horizon: int,
    history_steps: int = HISTORY_STEPS,
) -> None:
    """Fail closed on the shapes handed to the 54-D decoder."""
    expected = {
        "raw_action_labels": (batch_size * horizon, JOINT_DIM),
        "raw_action_masks": (batch_size * horizon, JOINT_DIM),
        "raw_proprio_inputs": (batch_size, JOINT_DIM),
        "images": (batch_size * (history_steps + 1), 3, MODEL_HEIGHT, MODEL_WIDTH),
    }
    for key, shape in expected.items():
        if key not in batch or tuple(batch[key].shape) != shape:
            got = tuple(batch[key].shape) if key in batch else "missing"
            raise ValueError(f"collated {key} must be {shape}, got {got}")


__all__ = [
    "EgoVLASparkArenaHDF5Dataset",
    "Joint54Normalization",
    "extract_action_chunk",
    "extract_current_state",
    "load_joint54_stats",
    "preprocess_vla_sparkarena_54d",
    "process_head_rgb",
    "validate_sparkarena_collated_batch",
    "verify_sparkarena_artifacts",
]
