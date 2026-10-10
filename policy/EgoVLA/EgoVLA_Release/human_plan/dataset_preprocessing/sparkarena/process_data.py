#!/usr/bin/env python3
"""Audit and prepare a lazy EgoVLA manifest for the seven-task SparkArena set.

Raw HDF5 files remain read-only.  State statistics are computed only from
``/state/*`` and action statistics only from the four original ``/action/*``
datasets.  ``/mano/action`` is deliberately forbidden because its source file
declares next-state semantics.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:  # Conversion-only environments do not need torch.
    torch = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parents[4]
RELEASE_DIR = Path(__file__).resolve().parents[3]
for _path in (SCRIPT_DIR, RELEASE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# Use XPolicyLab's canonical decoder.  It defines trajectory buffers as RGB
# and deliberately does not perform a second OpenCV-style channel swap.
POLICY_DIR = Path(__file__).resolve().parents[4]
XPL_ROOT = POLICY_DIR.parents[1]
if str(XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(XPL_ROOT))
from XPolicyLab.utils.process_data import decode_image_bit  # noqa: E402


TASK_NAMES = (
    "click_mouse",
    "collect_objects",
    "dual_bottles_pick",
    "hammer_beat",
    "put_food_in_microwave",
    "retrieve_gap",
    "stack_bowls",
)

# What is literally stored in the source HDF5 files.
SOURCE_INSTRUCTIONS = {
    "click_mouse": "Place the mouse on the mouse pad, then click the left button.",
    "collect_objects": "Put all the objects into the basket.",
    "dual_bottles_pick": "Pick up both bottles on the table.",
    "hammer_beat": "Pick up the hammer and beat the cube with the hammer head.",
    "put_food_in_microwave": "Put the bread in the microwave, then close the door.",
    "retrieve_gap": "Move the two cubes aside upright, then place the garage on the cushion.",
    "stack_bowls": "Stack the three bowls together.",
}

# Exact instructions emitted by the SparkArena evaluation tasks.  Microwave is
# the only intentional source/runtime wording correction (bread -> sandwich).
RUNTIME_INSTRUCTIONS = {
    **SOURCE_INSTRUCTIONS,
    "put_food_in_microwave": "Put the sandwich in the microwave, then close the door.",
}

STATE_DATASETS = (
    ("state/left_arm_joint_states", 7),
    ("state/left_ee_joint_states", 20),
    ("state/right_arm_joint_states", 7),
    ("state/right_ee_joint_states", 20),
)
ACTION_DATASETS = (
    ("action/left_arm_joint_states", 7),
    ("action/left_ee_joint_states", 20),
    ("action/right_arm_joint_states", 7),
    ("action/right_ee_joint_states", 20),
)
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ROBOT = "tianji_marvin_wuji"
ROBOT_KEY = ROBOT
ACTION_TYPE = "joint"
FPS = 25
JOINT_DIM = 54
LOGICAL_DIM = JOINT_DIM
SOURCE_HEIGHT = 480
SOURCE_WIDTH = 640
MODEL_HEIGHT = 384
MODEL_WIDTH = 384
PROCESSED_IMAGE_HEIGHT = MODEL_HEIGHT
PROCESSED_IMAGE_WIDTH = MODEL_WIDTH
COLOR_SPACE = "RGB"
COLOR_ORDER = COLOR_SPACE
CAMERA_MODE = "head_only"
INTERPOLATION = "PIL bicubic"
PROMPT_VERSION = "v0"
RUNTIME_CAMERA_KEY = "cam_head"
MODEL_OUTPUT_DIM = JOINT_DIM
MODEL_OUTPUT_INDICES = tuple(range(JOINT_DIM))
RAW_LABEL_DIM = JOINT_DIM
NATIVE_DIM = JOINT_DIM
PROMPT_TEMPLATE = "<image>\nWhere should i move hand to: {} ? A: "
HISTORY_PROMPT_PREFIX = "You have been given a video of history observations:"
HISTORY_PROMPT_CURRENT = "and current observation:<image>\n"
HISTORY_STEPS = 5
HISTORY_STRIDE = 5
TASK_INSTRUCTIONS = RUNTIME_INSTRUCTIONS
ACTION_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)
LOGICAL_ORDER = (
    *(f"left_arm_joint_{index}" for index in range(7)),
    *(f"left_hand_joint_{index}" for index in range(20)),
    *(f"right_arm_joint_{index}" for index in range(7)),
    *(f"right_hand_joint_{index}" for index in range(20)),
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


TASK_INSTRUCTIONS_SHA256 = canonical_json_sha256(RUNTIME_INSTRUCTIONS)


def decode_text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", errors="strict")
    text = str(value).strip()
    if not text:
        raise ValueError("empty HDF5 text value")
    return text


def decode_rgb_image(encoded: Any) -> np.ndarray:
    """Decode one buffer through XPolicyLab's official RGB decoder."""
    rgb = np.asarray(decode_image_bit(encoded))
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"decoded image is not uint8 HxWx3 RGB: {rgb.shape}/{rgb.dtype}")
    return rgb


def resize_rgb_for_egovla(rgb: np.ndarray) -> np.ndarray:
    """Shared train/runtime geometry: resize the complete RGB frame to 384².

    This deliberately preserves the left and right scene edges and changes the
    4:3 aspect ratio, matching EgoVLA's official ``image_aspect_ratio=resize``
    path.  There is no crop, letterbox/pad, or channel reversal.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3 RGB, got {rgb.shape}/{rgb.dtype}")
    resized = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize(
            (MODEL_WIDTH, MODEL_HEIGHT),
            resample=Image.Resampling.BICUBIC,
        ),
        dtype=np.uint8,
    )
    if resized.shape != (MODEL_HEIGHT, MODEL_WIDTH, 3):
        raise AssertionError(f"unexpected preprocessed shape: {resized.shape}")
    return resized


def preprocess_rgb_image(rgb: np.ndarray) -> np.ndarray:
    """Backward-compatible name for the shared full-frame resize helper."""
    return resize_rgb_for_egovla(rgb)


def canonical_prompt(instruction: str) -> str:
    """Exact EgoVLA v0 text wrapper used for both training and inference."""
    return PROMPT_TEMPLATE.format(instruction)


def instruction_for_task(task: str) -> str:
    try:
        return TASK_INSTRUCTIONS[str(task)]
    except KeyError as exc:
        raise KeyError(f"unsupported SparkArena task: {task!r}") from exc


def pack_raw_label(logical_actions: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate native 54-D targets and return the raw-label tensor and mask."""
    logical = np.asarray(logical_actions, dtype=np.float32)
    if logical.ndim < 1 or logical.shape[-1] != LOGICAL_DIM:
        raise ValueError(f"logical action must end in {LOGICAL_DIM}, got {logical.shape}")
    if not np.isfinite(logical).all():
        raise ValueError("logical action contains non-finite values")
    return logical.copy(), np.ones(logical.shape, dtype=bool)


def select_model_actions(prediction: Any) -> np.ndarray:
    """Validate and expose all 54 outputs of the native 27+27 decoder."""
    array = np.asarray(prediction)
    if array.ndim < 1 or array.shape[-1] != MODEL_OUTPUT_DIM:
        raise ValueError(f"decoder must emit exactly {MODEL_OUTPUT_DIM}, got {array.shape}")
    return np.take(array, MODEL_OUTPUT_INDICES, axis=-1)


def state_to_logical(state: Any) -> np.ndarray:
    """Pack the standard XPolicyLab Wuji state mapping into canonical 54-D order."""
    if not isinstance(state, dict) and not hasattr(state, "__getitem__"):
        raise TypeError("observation state must be a mapping")
    parts = []
    for key, size in (
        ("left_arm_joint_state", 7),
        ("left_ee_joint_state", 20),
        ("right_arm_joint_state", 7),
        ("right_ee_joint_state", 20),
    ):
        try:
            raw = state[key]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"observation is missing state.{key}") from exc
        value = np.asarray(raw, dtype=np.float32)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"state.{key} must be finite and exactly ({size},), got {value.shape}")
        parts.append(value)
    result = np.concatenate(parts).astype(np.float32, copy=False)
    if result.shape != (LOGICAL_DIM,):
        raise AssertionError("invalid SparkArena logical state contract")
    return result


def pack_logical_state(state: Any) -> np.ndarray:
    """Compatibility alias for :func:`state_to_logical`."""
    return state_to_logical(state)


def logical_to_action(logical: Any) -> dict[str, np.ndarray]:
    """Split one canonical 54-D command into XPolicyLab joint action keys."""
    value = np.asarray(logical, dtype=np.float32)
    if value.shape != (LOGICAL_DIM,) or not np.isfinite(value).all():
        raise ValueError(f"logical action must be finite and exactly ({LOGICAL_DIM},)")
    return {
        "left_arm_joint_state": value[0:7].copy(),
        "left_ee_joint_state": value[7:27].copy(),
        "right_arm_joint_state": value[27:34].copy(),
        "right_ee_joint_state": value[34:54].copy(),
    }


def preprocessing_contract() -> dict[str, Any]:
    return {
        "encoded_source": "HDF5 vision/<camera>/colors",
        "decoded_color_space": COLOR_SPACE,
        "source_resolution_hw": [SOURCE_HEIGHT, SOURCE_WIDTH],
        "geometry": "whole_frame_resize_to_square",
        "crop": None,
        "padding": None,
        "preserves_horizontal_edges": True,
        "preserves_aspect_ratio": False,
        "model_resolution_hw": [MODEL_HEIGHT, MODEL_WIDTH],
        "resize_interpolation": INTERPOLATION,
        "training_resize_owner": "shared helper before EgoVLA image normalization",
        "channel_swap_after_decode": False,
        "shared_function": "human_plan.dataset_preprocessing.sparkarena.process_data.resize_rgb_for_egovla",
    }


def prompt_contract() -> dict[str, Any]:
    return {
        "template": "<image>\\nWhere should i move hand to: {instruction} ? A: ",
        "source_instructions": SOURCE_INSTRUCTIONS,
        "runtime_instructions": RUNTIME_INSTRUCTIONS,
        "intentional_overrides": {
            "put_food_in_microwave": {
                "source": SOURCE_INSTRUCTIONS["put_food_in_microwave"],
                "runtime": RUNTIME_INSTRUCTIONS["put_food_in_microwave"],
                "reason": "match SparkArena runtime task prompt",
            }
        },
        "mapping_sha256": canonical_json_sha256(RUNTIME_INSTRUCTIONS),
    }


def contract_dict() -> dict[str, Any]:
    """Serializable single source of truth shared by conversion/train/runtime."""
    return {
        "schema": "egovla_sparkarena_joint54_contract_v1",
        "benchmark": "SparkArena",
        "robot_key": ROBOT_KEY,
        "action_type": ACTION_TYPE,
        "logical_action_dim": LOGICAL_DIM,
        "state_dim": LOGICAL_DIM,
        "model_output_dim": MODEL_OUTPUT_DIM,
        "model_output_indices": list(MODEL_OUTPUT_INDICES),
        "logical_order": list(LOGICAL_ORDER),
        "arm_dim": [7, 7],
        "hand_dim": [20, 20],
        "state_source_datasets": [f"/{key}" for key, _ in STATE_DATASETS],
        "action_source_datasets": [f"/{key}" for key, _ in ACTION_DATASETS],
        "target_alignment": "state[t] -> original /action/*[t]; future k -> /action/*[t+k]",
        "forbid_next_state_as_action": True,
        "forbidden_action_sources": ["/mano/action", "state[t+1]"],
        "prompt_version": PROMPT_VERSION,
        "prompt_template": PROMPT_TEMPLATE,
        "history_steps": HISTORY_STEPS,
        "history_stride": HISTORY_STRIDE,
        "task_instructions": dict(TASK_INSTRUCTIONS),
        "task_instructions_sha256": TASK_INSTRUCTIONS_SHA256,
        "runtime_camera_key": RUNTIME_CAMERA_KEY,
        "camera_mode": CAMERA_MODE,
        "camera_source": "/vision/cam_head/colors",
        "wrist_placeholder": None,
        "color_order": COLOR_SPACE,
        "processed_image_hw": [PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH],
        "geometry": preprocessing_contract(),
    }

def _read_joint_frame(
    handle: h5py.File,
    fields: tuple[tuple[str, int], ...],
    frame: int,
) -> np.ndarray:
    parts = []
    for key, width in fields:
        if key not in handle:
            raise KeyError(f"{handle.filename}: missing /{key}")
        dataset = handle[key]
        if dataset.ndim != 2 or dataset.shape[1] != width:
            raise ValueError(f"{handle.filename}:/{key}: expected (T,{width}), got {dataset.shape}")
        if frame < 0 or frame >= dataset.shape[0]:
            raise IndexError(f"frame {frame} outside /{key}")
        parts.append(np.asarray(dataset[frame], dtype=np.float32))
    result = np.concatenate(parts)
    if result.shape != (JOINT_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"invalid 54-D joint frame: {result.shape}")
    return result


def extract_current_state(handle: h5py.File, frame: int) -> np.ndarray:
    """Read state[t] from exactly the four canonical /state datasets."""
    return _read_joint_frame(handle, STATE_DATASETS, frame)


def extract_action_chunk(
    handle: h5py.File,
    *,
    frame: int,
    end: int,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Read original action[t+k], with zero padding only beyond episode end."""
    if horizon <= 0 or stride <= 0:
        raise ValueError("horizon and stride must be positive")
    if end <= 0:
        raise ValueError("end must be positive")
    targets = np.zeros((horizon, JOINT_DIM), dtype=np.float32)
    valid = np.zeros((horizon,), dtype=bool)
    for future_step in range(horizon):
        target_frame = frame + future_step * stride
        if target_frame >= end:
            break
        targets[future_step] = _read_joint_frame(handle, ACTION_DATASETS, target_frame)
        valid[future_step] = True
    if not valid[0]:
        raise IndexError(f"action frame {frame} is outside [0,{end})")
    return targets, valid


def process_head_rgb(encoded: Any, image_processor: Any) -> Any:
    """Official RGB decode -> shared whole-frame resize -> model normalization."""
    if torch is None:
        raise RuntimeError("image processing requires PyTorch")
    rgb = decode_rgb_image(encoded)
    if rgb.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
        raise ValueError(f"cam_head must decode to 480x640 RGB, got {rgb.shape}")
    model_rgb = resize_rgb_for_egovla(rgb)
    processed = image_processor.preprocess(model_rgb, return_tensors="pt")["pixel_values"][0]
    if tuple(processed.shape) != (3, MODEL_HEIGHT, MODEL_WIDTH):
        raise ValueError(
            f"EgoVLA image_processor must emit (3,{MODEL_HEIGHT},{MODEL_WIDTH}), "
            f"got {tuple(processed.shape)}"
        )
    return processed


def preprocess_vla_sparkarena_54d(
    language_label: str,
    raw_input: Any,
    raw_label: Any,
    label_mask: Any,
    action_tokenizer: Any,
    tokenizer: Any,
    *,
    mask_input: bool = True,
    mask_ignore: bool = False,
    input_placeholder_diff_index: bool = True,
    sep_query_token: bool = True,
) -> dict[str, Any]:
    """Self-contained EgoVLA raw-label preprocessor with a true 54-D ABI.

    This mirrors the release's conversation/token behavior but never invokes
    its hard-coded ``reshape(-1, 46)`` implementation.
    """
    if torch is None:
        raise RuntimeError("preprocess_vla_sparkarena_54d requires PyTorch")
    from llava import conversation as conversation_lib
    from llava.constants import IGNORE_INDEX
    from llava.mm_utils import tokenizer_image_token

    raw_label_tensor = torch.as_tensor(raw_label, dtype=torch.float32).reshape(-1, JOINT_DIM)
    mask_tensor = torch.as_tensor(label_mask, dtype=torch.bool).reshape(-1, JOINT_DIM)
    if tuple(raw_label_tensor.shape) != tuple(mask_tensor.shape):
        raise ValueError("raw action label and mask shapes differ")
    proprio = torch.as_tensor(raw_input, dtype=torch.float32).reshape(-1, JOINT_DIM)

    valid_steps = mask_tensor[:, 0]
    if not torch.equal(mask_tensor, valid_steps[:, None].expand_as(mask_tensor)):
        raise ValueError("all 54 dimensions at one horizon step must share validity")
    # Match the split decoder's branch-major query order: all left-branch
    # horizon steps followed by all right-branch horizon steps.
    query_mask = valid_steps.repeat(2) if sep_query_token else valid_steps
    dummy = np.full(query_mask.numel(), 0.5, dtype=np.float32)
    action_text = action_tokenizer(dummy, query_mask.cpu().numpy())
    if not isinstance(action_text, str):
        raise TypeError("action tokenizer must return one string for a flattened query mask")

    conversation = conversation_lib.default_conversation.copy()
    conversation.messages = []
    conversation.append_message(conversation.roles[0], language_label)
    conversation.append_message(conversation.roles[1], action_text)
    if conversation.sep_style != conversation_lib.SeparatorStyle.TWO:
        raise ValueError("EgoVLA SparkArena requires a TWO-separator conversation template")
    input_ids = tokenizer_image_token(
        conversation.get_prompt(), tokenizer, return_tensors="pt"
    ).reshape(-1)
    targets = input_ids.clone()
    response_length = len(tokenizer_image_token(action_text, tokenizer))
    if response_length < 1 or response_length >= len(targets):
        raise ValueError("action response tokenization has an invalid length")
    targets[:-response_length] = IGNORE_INDEX
    targets[-1] = IGNORE_INDEX

    query_count = int(query_mask.numel())
    start = -(query_count + 1)
    if query_count + 1 > len(input_ids):
        raise ValueError("prompt is too short for raw action query placeholders")
    if input_placeholder_diff_index:
        placeholders = action_tokenizer.input_placeholder_token_idx - torch.arange(
            query_count, dtype=input_ids.dtype, device=input_ids.device
        )
    else:
        placeholders = torch.full(
            (query_count,),
            int(action_tokenizer.input_placeholder_token_idx),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
    if mask_input:
        input_ids[start:-1] = placeholders
    targets[start:-1] = placeholders
    if mask_ignore:
        targets[targets == int(action_tokenizer.invalid_token_idx)] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "labels": targets,
        "raw_action_label": raw_label_tensor,
        "raw_action_mask": mask_tensor,
        "proprio_input": proprio,
    }

DEFAULT_SOURCE = Path(os.environ.get("SPARKARENA_RAW_ROOT", SCRIPT_DIR / "data" / "raw" / "spark0_bench_7tasks"))
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "SparkArena-egovla-tianji_marvin_wuji-joint"
EPISODE_RE = re.compile(r"episode_(\d{7})\.hdf5$")
EXPECTED_EPISODES_PER_TASK = 100
EXPECTED_TOTAL_EPISODES = len(TASK_NAMES) * EXPECTED_EPISODES_PER_TASK
EXPECTED_TOTAL_FRAMES = 154_251
PRETRAINED_PATH = Path(os.environ.get("EGOVLA_PRETRAINED_PATH", SCRIPT_DIR / "EgoVLA_Release" / "checkpoints"))
PRETRAINED_REPOSITORY = "rchal97/ego_vla_human_video_pretrained"
PRETRAINED_COMMIT = "52dea0ceae8754d658d7c3f5fd8aa17c309348ea"


@dataclasses.dataclass(frozen=True)
class Episode:
    task: str
    source_episode_index: int
    global_episode_index: int
    path: Path


class StreamingStats:
    def __init__(self, width: int) -> None:
        self.width = width
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.total_sq = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)

    def update(self, value: np.ndarray) -> None:
        if value.ndim != 2 or value.shape[1] != self.width:
            raise ValueError(f"expected (*,{self.width}), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError("statistics input contains NaN/Inf")
        value64 = value.astype(np.float64, copy=False)
        self.count += len(value64)
        self.total += value64.sum(axis=0)
        self.total_sq += np.square(value64).sum(axis=0)
        self.minimum = np.minimum(self.minimum, value64.min(axis=0))
        self.maximum = np.maximum(self.maximum, value64.max(axis=0))

    def finish(self) -> dict[str, np.ndarray]:
        if self.count < 1:
            raise ValueError("cannot finish empty statistics")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
            "min": self.minimum.astype(np.float32),
            "max": self.maximum.astype(np.float32),
            "count": np.asarray(self.count, dtype=np.int64),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MANIFEST_SCHEMA = "xpolicylab-egovla-lazy-hdf5-manifest-v1"
PROVENANCE_SCHEMA = "sparkarena-egovla-provenance-v1"
STATS_SCHEMA = "sparkarena_joint54_stats_v1"
NORMALIZATION = "minmax_to_unit_interval_no_clip"
CAMERA_DATASET = "vision/cam_head/colors"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Joint54Normalization:
    minimum: np.ndarray
    maximum: np.ndarray
    scale: np.ndarray
    count: int
    source: str

    def normalize(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim < 1 or array.shape[-1] != JOINT_DIM:
            raise ValueError(f"joint values must end in {JOINT_DIM}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("joint values contain NaN/Inf")
        return ((array - self.minimum) / self.scale).astype(np.float32)

    def denormalize(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim < 1 or array.shape[-1] != JOINT_DIM:
            raise ValueError(f"joint values must end in {JOINT_DIM}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("normalized joint values contain NaN/Inf")
        return (array * self.scale + self.minimum).astype(np.float32)


def load_joint54_stats(path: Path, expected_source: str) -> Joint54Normalization:
    with np.load(path, allow_pickle=False) as stats:
        required = {
            "schema", "source", "minimum", "maximum", "scale", "count",
            "normalization", "epsilon", "order",
        }
        missing = sorted(required.difference(stats.files))
        if missing:
            raise ValueError(f"{path}: missing fields: {missing}")
        schema = str(np.asarray(stats["schema"]).item())
        source = str(np.asarray(stats["source"]).item())
        normalization = str(np.asarray(stats["normalization"]).item())
        minimum = np.asarray(stats["minimum"], dtype=np.float32).reshape(-1).copy()
        maximum = np.asarray(stats["maximum"], dtype=np.float32).reshape(-1).copy()
        scale = np.asarray(stats["scale"], dtype=np.float32).reshape(-1).copy()
        count = int(np.asarray(stats["count"]).item())
        epsilon = float(np.asarray(stats["epsilon"]).item())
        order = tuple(str(value) for value in np.asarray(stats["order"]).reshape(-1))
    if schema != STATS_SCHEMA or source != expected_source:
        raise ValueError(f"{path}: unexpected stats schema/source {schema!r}/{source!r}")
    if normalization != NORMALIZATION or count < 1:
        raise ValueError(f"{path}: invalid normalization/count")
    if any(value.shape != (JOINT_DIM,) for value in (minimum, maximum, scale)):
        raise ValueError(f"{path}: stats vectors must be exactly ({JOINT_DIM},)")
    if len(order) != JOINT_DIM or not all(np.isfinite(x).all() for x in (minimum, maximum, scale)):
        raise ValueError(f"{path}: invalid order or non-finite statistics")
    expected_scale = np.maximum(maximum - minimum, epsilon)
    if np.any(maximum < minimum) or np.any(scale <= 0) or not np.allclose(
        scale, expected_scale, rtol=1e-6, atol=1e-7
    ):
        raise ValueError(f"{path}: scale is not derived from min/max")
    return Joint54Normalization(minimum, maximum, scale, count, source)


def verify_sparkarena_artifacts(manifest_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = manifest_dir / "metadata.json"
    provenance_path = manifest_dir / "sparkarena_provenance.json"
    if not metadata_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"{manifest_dir}: metadata/provenance are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if metadata.get("format") != MANIFEST_SCHEMA:
        raise ValueError(f"{metadata_path}: unsupported manifest format")
    if provenance.get("format") != PROVENANCE_SCHEMA:
        raise ValueError(f"{provenance_path}: unsupported provenance format")
    if metadata.get("benchmark") != "SparkArena" or int(metadata.get("action_dim", -1)) != JOINT_DIM:
        raise ValueError("manifest is not the SparkArena 54-D joint contract")
    expected_prompt_hash = canonical_json_sha256(RUNTIME_INSTRUCTIONS)
    if provenance.get("prompt_mapping_sha256") != expected_prompt_hash:
        raise ValueError("canonical SparkArena prompt mapping hash mismatch")
    artifacts = provenance.get("artifact_sha256")
    if not isinstance(artifacts, Mapping):
        raise ValueError("provenance artifact_sha256 must be a mapping")
    for name in (
        "train.jsonl", "val.jsonl", "joint_state_stats.npz",
        "joint_action_stats.npz", "metadata.json",
    ):
        path = manifest_dir / name
        if not path.is_file() or artifacts.get(name) != _sha256_file(path):
            raise ValueError(f"artifact missing or hash mismatch: {path}")
    return metadata, provenance


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_for(task: str, source_episode_index: int, val_percent: int) -> str:
    token = f"SparkArena-v1:{task}:{source_episode_index}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 100
    return "val" if bucket < val_percent else "train"


def inventory_status(
    *,
    limit_per_task: int | None,
    episodes: int,
    frames: int,
    task_episode_counts: dict[str, int],
) -> dict[str, Any]:
    """Describe partial/full inventory and fail closed for production scans."""
    partial = limit_per_task is not None
    if not partial:
        if episodes != EXPECTED_TOTAL_EPISODES:
            raise ValueError(
                f"full inventory requires {EXPECTED_TOTAL_EPISODES} episodes, got {episodes}"
            )
        if frames != EXPECTED_TOTAL_FRAMES:
            raise ValueError(
                f"full inventory requires {EXPECTED_TOTAL_FRAMES} frames, got {frames}"
            )
        wrong = {
            task: task_episode_counts.get(task, 0)
            for task in TASK_NAMES
            if task_episode_counts.get(task, 0) != EXPECTED_EPISODES_PER_TASK
        }
        if wrong:
            raise ValueError(f"full inventory task counts differ from 100: {wrong}")
    return {
        "partial_inventory": partial,
        "selected_limit_per_task": limit_per_task,
        "expected_total_episodes": EXPECTED_TOTAL_EPISODES,
        "expected_total_frames": EXPECTED_TOTAL_FRAMES,
        "expected_episodes_per_task": EXPECTED_EPISODES_PER_TASK,
    }


def discover(
    source_root: Path,
    limit_per_task: int | None,
    expected_per_task: int = EXPECTED_EPISODES_PER_TASK,
) -> list[Episode]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root is not a directory: {source_root}")
    if limit_per_task is not None and limit_per_task < 1:
        raise ValueError("--limit-per-task must be >= 1")
    result: list[Episode] = []
    for task in TASK_NAMES:
        data_dir = source_root / task / ROBOT / "data"
        paths = sorted(data_dir.glob("episode_*.hdf5"))
        if not paths:
            raise FileNotFoundError(f"no episodes in {data_dir}")
        indices = []
        for path in paths:
            match = EPISODE_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"unexpected episode filename: {path}")
            indices.append(int(match.group(1)))
        if indices != list(range(len(paths))):
            raise ValueError(f"episode indices are not contiguous in {data_dir}")
        if limit_per_task is None and len(paths) != expected_per_task:
            raise ValueError(
                f"{task}: expected {expected_per_task} episodes, found {len(paths)}"
            )
        selected = paths[:limit_per_task] if limit_per_task is not None else paths
        for path in selected:
            match = EPISODE_RE.fullmatch(path.name)
            assert match is not None
            result.append(
                Episode(task, int(match.group(1)), len(result), path.resolve())
            )
    return result


def read_matrix(
    handle: h5py.File, key: str, width: int, expected_length: int | None = None
) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"{handle.filename}: missing /{key}")
    value = np.asarray(handle[key], dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{handle.filename}:/{key}: expected (T,{width}), got {value.shape}")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(
            f"{handle.filename}:/{key}: expected T={expected_length}, got {len(value)}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{handle.filename}:/{key} contains NaN/Inf")
    return value


def read_joint_group(
    handle: h5py.File, fields: tuple[tuple[str, int], ...], length: int | None = None
) -> np.ndarray:
    values: list[np.ndarray] = []
    for index, (key, width) in enumerate(fields):
        value = read_matrix(handle, key, width, length if index or length is not None else None)
        if length is None:
            length = len(value)
        values.append(value)
    assert length is not None
    combined = np.concatenate(values, axis=1)
    if combined.shape != (length, JOINT_DIM):
        raise AssertionError(f"invalid combined joint shape: {combined.shape}")
    return combined


def validate_camera(
    handle: h5py.File,
    camera: str,
    length: int,
    probe: bool,
) -> dict[str, Any]:
    color_key = f"vision/{camera}/colors"
    shape_key = f"vision/{camera}/shape"
    if color_key not in handle or shape_key not in handle:
        raise KeyError(f"{handle.filename}: missing /{color_key} or /{shape_key}")
    if len(handle[color_key]) != length:
        raise ValueError(
            f"{handle.filename}:/{color_key}: T={len(handle[color_key])}, expected {length}"
        )
    shape = np.asarray(handle[shape_key], dtype=np.int64).tolist()
    expected = [SOURCE_HEIGHT, SOURCE_WIDTH, 3]
    if shape != expected:
        raise ValueError(f"{handle.filename}:/{shape_key}: expected {expected}, got {shape}")
    report: dict[str, Any] = {"dataset": f"/{color_key}", "shape": shape}
    if probe:
        probe_rows = []
        for frame_index in sorted({0, length - 1}):
            rgb = decode_rgb_image(handle[color_key][frame_index])
            if rgb.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
                raise ValueError(f"decoded {camera} shape mismatch: {rgb.shape}")
            model_rgb = preprocess_rgb_image(rgb)
            if model_rgb.shape != (MODEL_HEIGHT, MODEL_WIDTH, 3):
                raise AssertionError(f"model image shape mismatch: {model_rgb.shape}")
            probe_rows.append(
                {
                    "frame_index": frame_index,
                    "decoded_dtype": str(rgb.dtype),
                    "decoded_shape": list(rgb.shape),
                    "model_shape": list(model_rgb.shape),
                    "rgb_corner_pixel": rgb[0, 0].tolist(),
                }
            )
        report["decoded_probes"] = probe_rows
    return report


def inspect_episode(
    episode: Episode,
    probe_images: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    with h5py.File(episode.path, "r") as handle:
        if "mano/action" in handle:
            # This is expected to exist, but is never read as a target.
            mano_semantics = decode_text(handle["mano"].attrs.get("action_semantics", "unknown"))
        else:
            mano_semantics = "absent"
        source_instruction = decode_text(handle["instruction"][()])
        expected_source_instruction = SOURCE_INSTRUCTIONS[episode.task]
        if source_instruction != expected_source_instruction:
            raise ValueError(
                f"{episode.path}: instruction mismatch; {source_instruction!r} != "
                f"{expected_source_instruction!r}"
            )
        frequency = int(np.asarray(handle["additional_info/frequency"]).item())
        version = decode_text(handle["data_format_version"][()])
        if frequency != FPS or version != "v1.0":
            raise ValueError(
                f"{episode.path}: expected v1.0/{FPS}Hz, got {version}/{frequency}Hz"
            )

        # These calls are intentionally independent: action never derives from state.
        state = read_joint_group(handle, STATE_DATASETS)
        action = read_joint_group(handle, ACTION_DATASETS, len(state))
        camera_report = {
            camera: validate_camera(handle, camera, len(state), probe_images)
            for camera in CAMERAS
        }
    return state, action, camera_report, {
        "source_instruction": source_instruction,
        "mano_action_semantics": mano_semantics,
        "source_version": version,
    }


def artifact_hashes(directory: Path) -> dict[str, str]:
    names = (
        "train.jsonl",
        "val.jsonl",
        "joint_state_stats.npz",
        "joint_action_stats.npz",
        "metadata.json",
    )
    return {name: sha256_file(directory / name) for name in names}


def save_stats(path: Path, values: dict[str, np.ndarray], source: str) -> None:
    minimum = values["min"]
    maximum = values["max"]
    epsilon = np.float32(1.0e-6)
    np.savez(
        path,
        schema=np.asarray("sparkarena_joint54_stats_v1"),
        source=np.asarray(source),
        minimum=minimum,
        maximum=maximum,
        mean=values["mean"],
        std=values["std"],
        scale=np.maximum(maximum - minimum, epsilon).astype(np.float32),
        count=values["count"],
        order=np.asarray(
            [
                *(f"left_arm_joint_{index}" for index in range(7)),
                *(f"left_hand_joint_{index}" for index in range(20)),
                *(f"right_arm_joint_{index}" for index in range(7)),
                *(f"right_hand_joint_{index}" for index in range(20)),
            ]
        ),
        normalization=np.asarray("minmax_to_unit_interval_no_clip"),
        epsilon=epsilon,
    )


def convert(
    source_root: Path,
    output: Path,
    val_percent: int,
    limit_per_task: int | None,
) -> dict[str, Any]:
    if not 1 <= val_percent <= 50:
        raise ValueError("--val-percent must be between 1 and 50")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        episodes = discover(source_root, limit_per_task)
        # A bounded smoke scan can be smaller than the requested percentage
        # bucket. Keep the split deterministic while guaranteeing one
        # validation episode whenever a partial scan has at least two episodes.
        fallback_val_episode = None
        if (
            limit_per_task is not None
            and len(episodes) > 1
            and not any(
                split_for(item.task, item.source_episode_index, val_percent) == "val"
                for item in episodes
            )
        ):
            fallback_val_episode = min(
                episodes,
                key=lambda item: (
                    item.task,
                    item.source_episode_index,
                    str(item.path),
                ),
            ).global_episode_index
        train_rows: list[dict[str, Any]] = []
        val_rows: list[dict[str, Any]] = []
        state_stats = StreamingStats(JOINT_DIM)
        action_stats = StreamingStats(JOINT_DIM)
        frame_counts: dict[str, int] = {task: 0 for task in TASK_NAMES}
        episode_counts: dict[str, int] = {task: 0 for task in TASK_NAMES}
        image_probes: dict[str, Any] = {}
        source_inventory: list[dict[str, Any]] = []
        mano_semantics: set[str] = set()

        for number, episode in enumerate(episodes, 1):
            probe = episode.source_episode_index == 0
            state, action, cameras, audit = inspect_episode(episode, probe)
            mano_semantics.add(audit["mano_action_semantics"])
            split = (
                "val"
                if episode.global_episode_index == fallback_val_episode
                else split_for(episode.task, episode.source_episode_index, val_percent)
            )
            instruction = RUNTIME_INSTRUCTIONS[episode.task]
            stat = episode.path.stat()
            row = {
                "episode_index": episode.global_episode_index,
                "source_episode_index": episode.source_episode_index,
                "source_path": str(episode.path),
                "path": str(episode.path.relative_to(source_root.resolve())),
                "task_name": episode.task,
                "source_instruction": audit["source_instruction"],
                "instruction": instruction,
                "prompt": canonical_prompt(instruction),
                "split": split,
                "frames": len(state),
                "fps": FPS,
                "state_dim": JOINT_DIM,
                "action_dim": JOINT_DIM,
                "state_source_datasets": [f"/{key}" for key, _ in STATE_DATASETS],
                "action_source_datasets": [f"/{key}" for key, _ in ACTION_DATASETS],
                "action_alignment": "state[t] -> original /action/*[t] (no shift)",
                "camera_mode": "head_only",
                "model_camera": "/vision/cam_head/colors",
                "available_wrist_cameras": [
                    "/vision/cam_left_wrist/colors",
                    "/vision/cam_right_wrist/colors",
                ],
            }
            (val_rows if split == "val" else train_rows).append(row)
            if split == "train":
                state_stats.update(state)
                action_stats.update(action)
            frame_counts[episode.task] += len(state)
            episode_counts[episode.task] += 1
            if probe:
                image_probes[episode.task] = cameras
            source_inventory.append(
                {
                    "relative_path": str(episode.path.relative_to(source_root.resolve())),
                    "bytes": stat.st_size,
                    "frames": len(state),
                }
            )
            if number % 25 == 0 or number == len(episodes):
                print(f"audited {number}/{len(episodes)} episodes", flush=True)

        if not train_rows or not val_rows:
            raise ValueError(
                f"deterministic split produced train={len(train_rows)}, val={len(val_rows)}"
            )
        # Every task must be represented in both splits for a full conversion.
        if limit_per_task is None:
            for task in TASK_NAMES:
                splits = {
                    row["split"] for row in (*train_rows, *val_rows) if row["task_name"] == task
                }
                if splits != {"train", "val"}:
                    raise ValueError(f"{task}: missing train or val episode in deterministic split")

        write_jsonl(staging / "train.jsonl", train_rows)
        write_jsonl(staging / "val.jsonl", val_rows)
        state_values = state_stats.finish()
        action_values = action_stats.finish()
        save_stats(staging / "joint_state_stats.npz", state_values, "/state/*")
        save_stats(staging / "joint_action_stats.npz", action_values, "/action/*")

        total_frames = sum(frame_counts.values())
        inventory = inventory_status(
            limit_per_task=limit_per_task,
            episodes=len(episodes),
            frames=total_frames,
            task_episode_counts=episode_counts,
        )
        metadata = {
            "format": "xpolicylab-egovla-lazy-hdf5-manifest-v1",
            "benchmark": "SparkArena",
            "source_root": str(source_root.resolve()),
            "output_root": str(output.resolve()),
            "robot": ROBOT,
            "fps": FPS,
            "episodes": len(episodes),
            "frames": total_frames,
            "train_episodes": len(train_rows),
            "val_episodes": len(val_rows),
            "val_percent_rule": val_percent,
            "task_episode_counts": episode_counts,
            "task_frame_counts": frame_counts,
            **inventory,
            "state_dim": JOINT_DIM,
            "action_dim": JOINT_DIM,
            "state_stats_count": int(state_values["count"]),
            "action_stats_count": int(action_values["count"]),
            "joint_order": [
                "left_arm_joint_states[0:7]",
                "left_ee_joint_states[0:20]",
                "right_arm_joint_states[0:7]",
                "right_ee_joint_states[0:20]",
            ],
            "state_source_datasets": [f"/{key}" for key, _ in STATE_DATASETS],
            "action_source_datasets": [f"/{key}" for key, _ in ACTION_DATASETS],
            "action_policy": {
                "kind": "absolute joint command",
                "alignment": "state[t] -> original /action/*[t]",
                "temporal_shift": 0,
                "forbidden_sources": ["/mano/action", "state[t+1]"],
                "observed_mano_action_semantics": sorted(mano_semantics),
            },
            "camera_policy": {
                "model_mode": "head_only",
                "model_camera": "cam_head",
                "available_source_cameras": list(CAMERAS),
                "wrist_placeholder": None,
                "reason": "SparkArena has real wrists, but the current EgoVLA architecture is head-only",
            },
            "image_preprocessing": preprocessing_contract(),
            "prompt": prompt_contract(),
            "pretrained": {
                "path": str(PRETRAINED_PATH),
                "repository": PRETRAINED_REPOSITORY,
                "commit": PRETRAINED_COMMIT,
            },
            "model_compatibility": {
                "required_state_dim": JOINT_DIM,
                "required_action_dim": JOINT_DIM,
                "current_pretrained_decoder_action_dim": 48,
                "required_decoder_branch_dims": [27, 27],
                "training_must_not_start_before_adapter_update": True,
            },
        }
        write_json(staging / "metadata.json", metadata)

        provenance = {
            "format": "sparkarena-egovla-provenance-v1",
            "source_inventory": source_inventory,
            "source_inventory_sha256": canonical_json_sha256(source_inventory),
            "prompt_mapping_sha256": canonical_json_sha256(RUNTIME_INSTRUCTIONS),
            "image_probes": image_probes,
            "artifact_sha256": artifact_hashes(staging),
        }
        write_json(staging / "sparkarena_provenance.json", provenance)
        os.replace(staging, output)
        print(f"published {output}", flush=True)
        return metadata
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-percent", type=int, default=5)
    parser.add_argument(
        "--limit-per-task",
        type=int,
        help="test-only: select the first N episodes in each task",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = convert(
        args.source_root.resolve(),
        args.output.resolve(),
        args.val_percent,
        args.limit_per_task,
    )
    print(json.dumps({
        "episodes": metadata["episodes"],
        "frames": metadata["frames"],
        "train_episodes": metadata["train_episodes"],
        "val_episodes": metadata["val_episodes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
