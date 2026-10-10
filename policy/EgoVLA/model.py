"""XPolicyLab model adapter for the SparkArena raw-joint EgoVLA recipe."""

from __future__ import annotations

import copy
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

XPL_ROOT = Path(__file__).resolve().parents[2]
if str(XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(XPL_ROOT))

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_batch_size, get_robot_action_dim_info

from .EgoVLA_Release.human_plan.utils.sparkarena_provenance import verify_sparkarena_checkpoint


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class Model(ModelTemplate):
    """Stateful batched policy backed by the 54-D SparkArena decoder."""

    def __init__(self, model_cfg: Mapping[str, Any]):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg.get("action_type", "joint"))
        self.env_cfg_type = str(
            self.model_cfg.get("env_cfg_type", "tianji_marvin_wuji")
        )
        if self.action_type != "joint" or self.env_cfg_type != "tianji_marvin_wuji":
            raise ValueError(
                "this adapter only supports SparkArena tianji_marvin_wuji/joint"
            )

        dims = get_robot_action_dim_info(self.env_cfg_type)
        self.batch_size = int(get_batch_size(self.env_cfg_type))
        self.arm_dims = tuple(int(dim) for dim in dims.get("arm_dim", []))
        self.ee_dims = tuple(int(dim) for dim in dims.get("ee_dim", []))
        if self.arm_dims != (7, 7) or self.ee_dims != (20, 20):
            raise ValueError(
                "tianji_marvin_wuji must register arm_dim=[7,7] and ee_dim=[20,20], "
                f"got {dims}"
            )

        self.action_chunk_size = max(1, int(self.model_cfg.get("action_chunk_size", 30)))
        self.history_length = max(1, int(self.model_cfg.get("history_length", 6)))
        self.history_stride = max(1, int(self.model_cfg.get("history_stride", 5)))
        self._histories: dict[int, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.history_length * self.history_stride + 1)
        )
        self._obs: dict[str, Any] | None = None
        self._obs_batch: dict[int, dict[str, Any]] = {}
        self.runtime: EgoVLASparkArenaInference | None = None
        self.provenance_bundle: dict[str, Any] | None = None

        self.dry_run = _truthy(self.model_cfg.get("dry_run", False))
        debug_requested = os.environ.get("EVAL_ENV_TYPE", "").strip().lower() == "debug"
        if debug_requested and _truthy(self.model_cfg.get("allow_debug_stub", False)):
            self.dry_run = True

        if not self.dry_run:
            policy_dir = Path(__file__).resolve().parent
            checkpoint_root = resolve_checkpoint_root(
                self.model_cfg,
                policy_dir / "checkpoints",
                policy_dir=policy_dir,
                must_exist=True,
            )
            upstream_value = self.model_cfg.get("upstream_root") or (
                policy_dir / "EgoVLA_Release"
            )
            upstream_root = Path(upstream_value).expanduser().resolve()
            self.provenance_bundle = verify_sparkarena_checkpoint(checkpoint_root)
            self.runtime = EgoVLASparkArenaInference(
                checkpoint_root,
                upstream_root=upstream_root,
                device="cuda",
            )
            if self.runtime.history_steps != self.history_length - 1:
                raise ValueError(
                    "deploy history_length differs from checkpoint: "
                    f"{self.history_length} vs {self.runtime.history_steps + 1}"
                )
            if self.runtime.history_stride != self.history_stride:
                raise ValueError(
                    "deploy history_stride differs from checkpoint: "
                    f"{self.history_stride} vs {self.runtime.history_stride}"
                )
            self.model = self.runtime.model
            print(f"[egovla] loaded SparkArena checkpoint: {checkpoint_root}")
        else:
            print("[egovla] debug/dry-run mode: returning hold actions")

    @staticmethod
    def _snapshot(obs: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(obs)
        vision = obs.get("vision")
        if isinstance(vision, Mapping):
            result["vision"] = {}
            for name, camera in vision.items():
                if not isinstance(camera, Mapping):
                    result["vision"][name] = camera
                    continue
                payload = dict(camera)
                if "color" in payload:
                    payload["color"] = np.asarray(payload["color"]).copy()
                result["vision"][name] = payload
        state = obs.get("state")
        if isinstance(state, Mapping):
            result["state"] = {
                key: (np.asarray(value).copy() if isinstance(value, np.ndarray) else value)
                for key, value in state.items()
            }
        return result

    @staticmethod
    def _env_idx(obs: Mapping[str, Any], fallback: int = 0) -> int:
        try:
            return int(obs.get("env_idx", fallback))
        except (TypeError, ValueError):
            return fallback

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        if not isinstance(obs, Mapping):
            raise TypeError(f"update_obs expects a mapping, got {type(obs).__name__}")
        snapshot = self._snapshot(obs)
        env_idx = self._env_idx(snapshot)
        self._obs = snapshot
        self._obs_batch[env_idx] = snapshot
        self._histories[env_idx].append(snapshot)

    def update_obs_batch(self, obs_list: Sequence[Mapping[str, Any]]) -> None:
        if not isinstance(obs_list, Sequence):
            raise TypeError("update_obs_batch expects a sequence of observations")
        self._obs_batch = {}
        for position, obs in enumerate(obs_list):
            if not isinstance(obs, Mapping):
                raise TypeError(f"observation {position} is not a mapping")
            snapshot = self._snapshot(obs)
            env_idx = self._env_idx(snapshot, position)
            self._obs_batch[env_idx] = snapshot
            self._histories[env_idx].append(snapshot)
            if position == 0:
                self._obs = snapshot

    def _hold_action(self, obs: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
        state = obs.get("state", {}) if isinstance(obs, Mapping) else {}
        if not isinstance(state, Mapping):
            state = {}
        values: dict[str, np.ndarray] = {}
        for key, size in (
            ("left_arm_joint_state", 7),
            ("left_ee_joint_state", 20),
            ("right_arm_joint_state", 7),
            ("right_ee_joint_state", 20),
        ):
            value = np.asarray(
                state.get(key, np.zeros(size, dtype=np.float32)), dtype=np.float32
            ).reshape(-1)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"SparkArena hold action requires finite {key} with {size} values")
            values[key] = value.copy()
        return values

    def _actions_for(
        self, env_idx: int, obs: Mapping[str, Any] | None
    ) -> list[dict[str, np.ndarray]]:
        history = list(self._histories[env_idx])
        if obs is not None and (not history or history[-1] is not obs):
            history.append(self._snapshot(obs))
        if not history:
            raise RuntimeError(f"no observation available for environment {env_idx}")
        if self.dry_run or self.runtime is None:
            base = self._hold_action(history[-1])
            return [copy.deepcopy(base) for _ in range(self.action_chunk_size)]

        instruction = history[-1].get("instruction", history[-1].get("instructions", ""))
        predicted = self.runtime.predict(history, str(instruction or ""))
        if not predicted:
            raise RuntimeError("EgoVLA returned an empty action trajectory")
        expected_dims = {
            "left_arm_joint_state": 7,
            "left_ee_joint_state": 20,
            "right_arm_joint_state": 7,
            "right_ee_joint_state": 20,
        }
        for action in predicted:
            if set(action) != set(expected_dims):
                raise RuntimeError("EgoVLA action keys do not match SparkArena joint contract")
            for key, dim in expected_dims.items():
                value = np.asarray(action[key], dtype=np.float32).reshape(-1)
                if value.shape != (dim,) or not np.isfinite(value).all():
                    raise RuntimeError(f"EgoVLA action {key} must be finite and exactly {dim}-D")
        return predicted[: self.action_chunk_size]

    def get_action(self) -> list[dict[str, np.ndarray]]:
        env_idx = self._env_idx(self._obs or {})
        return self._actions_for(env_idx, self._obs)

    def get_action_batch(
        self,
        env_idx_list: Sequence[int] | None = None,
        obs: Any = None,
    ) -> list[list[dict[str, np.ndarray]]]:
        if isinstance(obs, Sequence) and obs and isinstance(obs[0], Mapping):
            self.update_obs_batch(obs)
        elif isinstance(env_idx_list, Sequence) and env_idx_list and isinstance(env_idx_list[0], Mapping):
            self.update_obs_batch(env_idx_list)  # type: ignore[arg-type]
            env_idx_list = list(self._obs_batch)
        if env_idx_list is None:
            env_idx_list = list(self._obs_batch) or [0]
        return [
            self._actions_for(int(env_idx), self._obs_batch.get(int(env_idx)))
            for env_idx in env_idx_list
        ]

    def reset(self) -> None:
        self._obs = None
        self._obs_batch.clear()
        self._histories.clear()


# Inlined SparkArena inference runtime.

import os

import sys

from pathlib import Path

from types import SimpleNamespace

from typing import Any, Mapping, Sequence

import numpy as np

from .EgoVLA_Release.human_plan.dataset_preprocessing.sparkarena.process_data import (
    ACTION_TYPE,
    COLOR_ORDER,
    HISTORY_PROMPT_CURRENT,
    HISTORY_PROMPT_PREFIX,
    LOGICAL_DIM,
    MODEL_OUTPUT_DIM,
    MODEL_OUTPUT_INDICES,
    PROCESSED_IMAGE_HEIGHT,
    PROCESSED_IMAGE_WIDTH,
    PROMPT_VERSION,
    ROBOT_KEY,
    RUNTIME_CAMERA_KEY,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TASK_INSTRUCTIONS,
    canonical_prompt,
    contract_dict,
    logical_to_action,
    pack_raw_label,
    resize_rgb_for_egovla,
    select_model_actions,
    state_to_logical,
)

_PROVENANCE_FILENAME = "egovla_sparkarena_provenance.json"

_STATE_STATS_FILENAME = "joint_state_stats.npz"

_ACTION_STATS_FILENAME = "joint_action_stats.npz"

_RUN_CONTRACT_SCHEMA = "egovla_sparkarena_run_contract_v1"

_RUN_GEOMETRY = "full_frame_resize_480x640_to_384x384"

def _insert_upstream(upstream_root: Path) -> None:
    for path in (upstream_root / "VILA", upstream_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

def extract_head_rgb(observation: Mapping[str, Any]) -> np.ndarray:
    """Validate RGB 480x640 ``cam_head`` and apply the shared full-frame resize."""

    if not isinstance(observation, Mapping):
        raise TypeError("SparkArena observation must be a mapping")
    vision = observation.get("vision")
    if not isinstance(vision, Mapping):
        raise KeyError("observation is missing vision")
    if RUNTIME_CAMERA_KEY not in vision:
        raise KeyError(
            f"observation is missing vision.{RUNTIME_CAMERA_KEY}; "
            "wrist/alternate camera fallback is forbidden"
        )
    camera = vision[RUNTIME_CAMERA_KEY]
    if not isinstance(camera, Mapping) or "color" not in camera:
        raise KeyError(f"observation is missing vision.{RUNTIME_CAMERA_KEY}.color")

    image = np.asarray(camera["color"])
    expected = (SOURCE_HEIGHT, SOURCE_WIDTH, 3)
    if image.shape != expected:
        raise ValueError(
            f"vision.{RUNTIME_CAMERA_KEY}.color must be exactly {expected}, got "
            f"{image.shape}; pre-resized/cropped/letterboxed input is forbidden"
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"vision.{RUNTIME_CAMERA_KEY}.color must be uint8 {COLOR_ORDER}, got {image.dtype}"
        )

    # The policy server already decoded RGB.  The same helper is used by data
    # conversion/training, with no crop, padding, or channel reversal.
    resized = np.asarray(resize_rgb_for_egovla(image))
    processed = (PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, 3)
    if resized.shape != processed or resized.dtype != np.uint8:
        raise RuntimeError(
            "shared SparkArena image helper violated its ABI: "
            f"expected uint8 {processed}, got {resized.shape}/{resized.dtype}"
        )
    return np.ascontiguousarray(resized.copy())

def validate_instruction(
    instruction: str,
    observation: Mapping[str, Any] | None = None,
) -> str:
    """Require one exact canonical SparkArena runtime instruction."""

    if not isinstance(instruction, str) or instruction not in TASK_INSTRUCTIONS.values():
        raise ValueError("instruction is not an exact canonical SparkArena task instruction")
    if observation is None:
        return instruction
    if not isinstance(observation, Mapping):
        raise TypeError("SparkArena observation must be a mapping")

    observed = [
        observation[key] for key in ("instruction", "instructions") if key in observation
    ]
    if not observed or any(value != instruction for value in observed):
        raise ValueError("observation instruction differs from the inference instruction")

    tasks: list[Any] = [
        observation[key] for key in ("task_name", "task") if key in observation
    ]
    additional = observation.get("additional_info")
    if additional is not None and not isinstance(additional, Mapping):
        raise TypeError("observation.additional_info must be a mapping when present")
    if isinstance(additional, Mapping):
        tasks.extend(additional[key] for key in ("task_name", "task") if key in additional)
    for task in tasks:
        if not isinstance(task, str) or task not in TASK_INSTRUCTIONS:
            raise KeyError(f"unsupported SparkArena benchmark task: {task!r}")
        if instruction != TASK_INSTRUCTIONS[task]:
            raise ValueError(
                f"instruction does not match canonical SparkArena mapping for task {task}"
            )
    return instruction

def logical_state_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    """Pack the four Wuji state fields into canonical 7+20+7+20 order."""

    if not isinstance(observation, Mapping):
        raise TypeError("SparkArena observation must be a mapping")
    state = observation.get("state")
    if not isinstance(state, Mapping):
        raise KeyError("observation is missing state")
    logical = np.asarray(state_to_logical(state), dtype=np.float32)
    if logical.shape != (LOGICAL_DIM,) or not np.isfinite(logical).all():
        raise ValueError(f"shared state helper must return finite ({LOGICAL_DIM},)")
    return logical.copy()

def normalize_joint_state(logical_state: Any, minimum: Any, scale: Any) -> np.ndarray:
    state = np.asarray(logical_state, dtype=np.float32)
    low = np.asarray(minimum, dtype=np.float32)
    span = np.asarray(scale, dtype=np.float32)
    expected = (LOGICAL_DIM,)
    if state.shape != expected or low.shape != expected or span.shape != expected:
        raise ValueError(f"joint state/minimum/scale must all be exactly {LOGICAL_DIM}-D")
    if not np.isfinite(state).all() or not np.isfinite(low).all() or not np.isfinite(span).all():
        raise ValueError("joint state normalization received non-finite values")
    if np.any(span <= 0.0):
        raise ValueError("joint state normalization scale must be positive")
    return ((state - low) / span).astype(np.float32)

def _validate_processed_image_tensor(image_tensor: Any, *, history_steps: int) -> Any:
    shape = getattr(image_tensor, "shape", None)
    actual = tuple(int(value) for value in shape) if shape is not None else ()
    expected = (
        int(history_steps) + 1,
        3,
        PROCESSED_IMAGE_HEIGHT,
        PROCESSED_IMAGE_WIDTH,
    )
    if actual != expected:
        raise ValueError(
            "processed head image tensor must be exactly "
            f"{expected}, got {actual}; processor geometry/channel drift is forbidden"
        )
    return image_tensor

def _preprocess_head_rgb(image_processor: Any, observation: Mapping[str, Any]) -> Any:
    """Apply the same shared resize + processor path as the training dataset."""

    model_rgb = extract_head_rgb(observation)
    processed = image_processor.preprocess(model_rgb, return_tensors="pt")
    if not isinstance(processed, Mapping) or "pixel_values" not in processed:
        raise ValueError("EgoVLA image processor did not return pixel_values")
    pixels = processed["pixel_values"]
    shape = getattr(pixels, "shape", None)
    actual = tuple(int(value) for value in shape) if shape is not None else ()
    expected = (1, 3, PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH)
    if actual != expected:
        raise ValueError(
            f"EgoVLA image processor must emit {expected}, got {actual}"
        )
    return pixels[0]

def decode_joint_prediction(
    prediction: Any, minimum: Any, scale: Any
) -> list[dict[str, np.ndarray]]:
    """Invert normalization and split every native 54-D decoder output."""

    output = np.asarray(prediction)
    if output.ndim != 2 or output.shape[-1] != MODEL_OUTPUT_DIM:
        raise ValueError(
            f"SparkArena decoder output must be [H,{MODEL_OUTPUT_DIM}], got {output.shape}"
        )
    if not np.isfinite(output).all():
        raise ValueError("SparkArena decoder output contains NaN or infinity")
    selected = np.asarray(select_model_actions(output), dtype=np.float32)
    if selected.shape != output.shape:
        raise RuntimeError("SparkArena action selector must preserve all 54 decoder slots")
    low = np.asarray(minimum, dtype=np.float32)
    span = np.asarray(scale, dtype=np.float32)
    expected = (LOGICAL_DIM,)
    if low.shape != expected or span.shape != expected:
        raise ValueError(f"joint action statistics must be exactly {LOGICAL_DIM}-D")
    if not np.isfinite(low).all() or not np.isfinite(span).all() or np.any(span <= 0.0):
        raise ValueError("joint action statistics must be finite with a positive scale")
    logical = selected * span + low  # Training/inference contract forbids clipping.
    if not np.isfinite(logical).all():
        raise ValueError("denormalized SparkArena joint action contains NaN or infinity")
    return [logical_to_action(step) for step in logical]

def _validate_decoder_abi(decoder: Any) -> Any:
    """Require a native 54-D decoder with exact 27+27 projection branches."""

    actual = {
        "out_dim": getattr(decoder, "out_dim", None),
        "proprio_size": getattr(decoder, "proprio_size", None),
        "use_proprio": getattr(decoder, "use_proprio", None),
        "sep_proprio": getattr(decoder, "sep_proprio", None),
    }
    expected = {
        "out_dim": MODEL_OUTPUT_DIM,
        "proprio_size": LOGICAL_DIM,
        "use_proprio": True,
        "sep_proprio": False,
    }
    if actual != expected:
        raise ValueError(f"loaded trajectory decoder ABI differs from SparkArena: {actual}")
    try:
        left = decoder.decoder.output_projection_left[-1]
        right = decoder.decoder.output_projection_right[-1]
        branches = (int(left.out_features), int(right.out_features))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("SparkArena trajectory decoder lacks inspectable split projections") from exc
    if branches != (27, 27):
        raise ValueError(
            "SparkArena trajectory decoder must expose exact 27+27 output branches, "
            f"got {branches[0]}+{branches[1]}"
        )
    return decoder

def _validate_stats_mapping(stats: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(stats, Mapping):
        raise TypeError(f"verified checkpoint {name}_stats must be a mapping")
    if "minimum" not in stats or "scale" not in stats:
        raise ValueError(f"verified checkpoint {name}_stats is missing minimum/scale")
    minimum = np.asarray(stats["minimum"], dtype=np.float32)
    scale = np.asarray(stats["scale"], dtype=np.float32)
    expected = (LOGICAL_DIM,)
    if minimum.shape != expected or scale.shape != expected:
        raise ValueError(f"verified checkpoint {name}_stats must be {LOGICAL_DIM}-D")
    if not np.isfinite(minimum).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError(f"verified checkpoint {name}_stats must be finite with positive scale")
    return minimum.copy(), scale.copy()

def _validate_checkpoint_bundle(
    checked: Any,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Defensively validate the verifier return ABI before CUDA imports."""

    if not isinstance(checked, Mapping):
        raise TypeError("verify_sparkarena_checkpoint must return a mapping")
    required = {"root", "contract", "run_contract", "state_stats", "action_stats"}
    missing = sorted(required.difference(checked))
    if missing:
        raise ValueError(f"verified SparkArena checkpoint result is missing: {missing}")
    root = Path(checked["root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"verified SparkArena checkpoint root is not a directory: {root}")
    for filename in (_PROVENANCE_FILENAME, _STATE_STATS_FILENAME, _ACTION_STATS_FILENAME):
        if not (root / filename).is_file():
            raise FileNotFoundError(f"verified SparkArena checkpoint is missing {filename}")

    contract = checked["contract"]
    run = checked["run_contract"]
    if not isinstance(contract, Mapping) or not isinstance(run, Mapping):
        raise TypeError("verified checkpoint contract and run_contract must be mappings")
    expected_contract = contract_dict()
    if contract != expected_contract:
        raise ValueError("verified SparkArena checkpoint contract differs from runtime")

    expected_run: dict[str, Any] = {
        "schema": _RUN_CONTRACT_SCHEMA,
        "benchmark": "SparkArena",
        "robot_key": ROBOT_KEY,
        "action_type": ACTION_TYPE,
        "prompt_version": PROMPT_VERSION,
        "runtime_camera_key": RUNTIME_CAMERA_KEY,
        "raw_input_resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
        "geometry": _RUN_GEOMETRY,
        "input_resolution": [PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH],
        "color_order": COLOR_ORDER,
        "reverse_channel_order": False,
        "state_normalization": "minmax_to_unit_interval_no_clip",
        "action_denormalization": "prediction*scale+minimum; no clipping",
    }
    bad = [key for key, value in expected_run.items() if run.get(key) != value]
    if bad:
        raise ValueError("verified SparkArena run contract differs from runtime: " + ", ".join(bad))
    for key in ("predict_future_step", "history_stride"):
        value = run.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"SparkArena run contract {key} must be a positive integer")
    history_steps = run.get("history_steps")
    if not isinstance(history_steps, int) or isinstance(history_steps, bool) or history_steps < 0:
        raise ValueError("SparkArena run contract history_steps must be a non-negative integer")
    if history_steps != contract.get("history_steps"):
        raise ValueError("checkpoint contract and run contract disagree on history_steps")
    if run["history_stride"] != contract.get("history_stride"):
        raise ValueError("checkpoint contract and run contract disagree on history_stride")
    conversation = run.get("conversation_template")
    if not isinstance(conversation, str) or not conversation:
        raise ValueError("SparkArena run contract conversation_template must be non-empty")

    model_abi = run.get("model_abi")
    expected_abi = {
        "traj_decoder_type": "transformer_split_action_v2",
        "proprio_size": LOGICAL_DIM,
        "use_proprio": True,
        "sep_proprio": False,
        "action_output_dim": MODEL_OUTPUT_DIM,
        "model_output_indices": list(MODEL_OUTPUT_INDICES),
    }
    if not isinstance(model_abi, Mapping) or any(
        model_abi.get(key) != value for key, value in expected_abi.items()
    ):
        raise ValueError("verified SparkArena run contract has an incompatible model_abi")

    state_min, state_scale = _validate_stats_mapping(checked["state_stats"], name="state")
    action_min, action_scale = _validate_stats_mapping(checked["action_stats"], name="action")
    return root, contract, run, state_min, state_scale, action_min, action_scale

def _validate_loaded_image_abi(model: Any, image_processor: Any) -> None:
    aspect = getattr(getattr(model, "config", None), "image_aspect_ratio", None)
    if aspect != "resize":
        raise ValueError(
            "loaded EgoVLA checkpoint must declare image_aspect_ratio='resize', "
            f"got {aspect!r}"
        )
    if getattr(image_processor, "do_resize", None) is not True:
        raise ValueError("loaded EgoVLA image processor must have do_resize=True")
    size = getattr(image_processor, "size", None)
    if not isinstance(size, Mapping):
        raise ValueError("loaded EgoVLA image processor must expose a height/width size mapping")
    actual = (size.get("height"), size.get("width"))
    expected = (PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH)
    if actual != expected:
        raise ValueError(f"loaded EgoVLA image processor size must be {expected}, got {actual}")

class EgoVLASparkArenaInference:
    """Load and execute one finalized SparkArena 54-D EgoVLA checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        upstream_root: str | os.PathLike[str],
        device: str = "cuda",
    ) -> None:
        from .EgoVLA_Release.human_plan.utils.sparkarena_provenance import verify_sparkarena_checkpoint

        checked = verify_sparkarena_checkpoint(checkpoint_path)
        (
            self.checkpoint_root,
            self.contract,
            self.run_contract,
            self.state_minimum,
            self.state_scale,
            self.action_minimum,
            self.action_scale,
        ) = _validate_checkpoint_bundle(checked)
        self.device = str(device)
        self.predict_future_step = int(self.run_contract["predict_future_step"])
        self.history_steps = int(self.run_contract["history_steps"])
        self.history_stride = int(self.run_contract["history_stride"])
        self.prompt_version = str(self.run_contract["prompt_version"])
        self.conversation_template = str(self.run_contract["conversation_template"])

        self.upstream_root = Path(upstream_root).expanduser().resolve()
        if not (self.upstream_root / "VILA").is_dir():
            raise FileNotFoundError(f"vendored EgoVLA VILA tree is missing: {self.upstream_root}")
        _insert_upstream(self.upstream_root)

        import torch
        from .EgoVLA_Release.human_plan.utils.compat import install_optional_compat, patch_attention_implementation

        install_optional_compat()
        patch_attention_implementation()
        from human_plan.utils.action_tokenizer import build_action_tokenizer
        from llava import conversation as conversation_lib
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        self.torch = torch
        load_device = self.device
        if self.device.strip().lower() == "cuda":
            load_device = f"cuda:{torch.cuda.current_device()}"
        tokenizer, model, image_processor, _ = load_pretrained_model(
            str(self.checkpoint_root),
            get_model_name_from_path(str(self.checkpoint_root)),
            model_base=None,
            device_map={"": load_device},
            device=load_device,
        )
        _validate_loaded_image_abi(model, image_processor)
        _validate_decoder_abi(model.get_traj_decoder())

        requested_dtype = os.environ.get("EGOVLA_INFERENCE_DTYPE", "bf16").strip().lower()
        if requested_dtype in {"bf16", "bfloat16"}:
            inference_dtype = torch.bfloat16
        elif requested_dtype in {"fp16", "float16", "half"}:
            inference_dtype = torch.float16
        else:
            raise ValueError(
                f"EGOVLA_INFERENCE_DTYPE must be bf16 or fp16, got {requested_dtype!r}"
            )
        model.to(device=load_device, dtype=inference_dtype)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        if self.conversation_template not in conversation_lib.conv_templates:
            raise ValueError(
                f"upstream does not provide conversation template {self.conversation_template!r}"
            )
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            self.conversation_template
        ]
        model_args = SimpleNamespace(
            action_tokenizer="uniform",
            min_action=0.0,
            max_action=1.0,
            num_action_bins=256,
            num_action_dims=27,
            num_action_sep_dims=27,
            sep_query_token=True,
            predict_future_step=self.predict_future_step,
        )
        action_tokenizer = build_action_tokenizer(
            model_args.action_tokenizer, tokenizer, model_args
        )
        model.config.invalid_token_idx = action_tokenizer.invalid_token_idx
        model.config.input_placeholder_token_idx = action_tokenizer.input_placeholder_token_idx
        model.config.input_placeholder_start_token_idx = action_tokenizer.input_placeholder_start_token_idx
        model.config.input_placeholder_end_token_idx = action_tokenizer.input_placeholder_end_token_idx
        model.config.image_aspect_ratio = "resize"
        model.config.mm_use_im_start_end = False
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.action_tokenizer = action_tokenizer
        self.dtype = inference_dtype
        try:
            self.model_device = next(model.parameters()).device
        except StopIteration:
            self.model_device = torch.device(load_device)
        self.data_args = SimpleNamespace(
            is_multimodal=True,
            image_processor=image_processor,
            image_aspect_ratio="resize",
            min_tiles=1,
            max_tiles=12,
            mm_use_im_start_end=False,
            with_aug=False,
            add_his_obs_step=self.history_steps,
            add_his_imgs=True,
            add_his_img_skip=self.history_stride,
            predict_future_step=self.predict_future_step,
            future_index=1,
            mask_input=True,
            mask_ignore=False,
            raw_action_label=True,
            input_placeholder_diff_index=True,
            sep_query_token=True,
            include_response=False,
            include_repeat_instruction=False,
            prompt_version=self.prompt_version,
            ignore_language=False,
            traj_action_output_dim=MODEL_OUTPUT_DIM,
            num_video_frames=8,
            fps=0.0,
            action_tokenizer=action_tokenizer,
        )

    def _prompt(self, instruction: str) -> str:
        from human_plan.preprocessing.preprocessing import preprocess_multimodal_vla
        from human_plan.preprocessing.prompting_format import preprocess_language_instruction

        text = preprocess_language_instruction(instruction, self.history_steps, self.data_args)
        expected = canonical_prompt(instruction)
        history = HISTORY_PROMPT_PREFIX + "<image>\n" * self.history_steps
        history += HISTORY_PROMPT_CURRENT
        expected = expected.replace("<image>\n", history)
        if text != expected:
            raise RuntimeError("official EgoVLA prompt drifted from the SparkArena contract")
        return preprocess_multimodal_vla(text, self.data_args)

    def _make_sample(
        self,
        history: Sequence[Mapping[str, Any]],
        instruction: str,
    ) -> dict[str, Any]:
        from .EgoVLA_Release.VILA.llava.data.dataset_sparkarena import preprocess_vla_sparkarena_54d

        if not history:
            raise ValueError("SparkArena inference requires at least one observation")
        canonical = validate_instruction(instruction, history[-1])
        for item in history:
            validate_instruction(canonical, item)

        observations = list(history)
        images = []
        for offset in range(1, self.history_steps + 1):
            index = max(0, len(observations) - 1 - offset * self.history_stride)
            images.append(_preprocess_head_rgb(self.image_processor, observations[index]))
        images.append(_preprocess_head_rgb(self.image_processor, observations[-1]))
        image_tensor = _validate_processed_image_tensor(
            self.torch.stack(images, dim=0), history_steps=self.history_steps
        )
        normalized_state = normalize_joint_state(
            logical_state_from_observation(observations[-1]),
            self.state_minimum,
            self.state_scale,
        ).reshape(1, LOGICAL_DIM)
        raw_label, raw_mask = pack_raw_label(
            np.zeros((self.predict_future_step, LOGICAL_DIM), dtype=np.float32)
        )
        # The release preprocessor hard-codes reshape(-1, 46) and is never
        # valid for this adapter.  Reuse the exact 54-D helper used in training.
        data = preprocess_vla_sparkarena_54d(
            self._prompt(canonical),
            self.torch.from_numpy(normalized_state),
            self.torch.from_numpy(raw_label),
            self.torch.from_numpy(raw_mask),
            self.action_tokenizer,
            self.tokenizer,
            mask_input=True,
            mask_ignore=False,
            input_placeholder_diff_index=True,
            sep_query_token=True,
        )
        data["image"] = image_tensor
        return data

    def predict(
        self,
        history: Sequence[Mapping[str, Any]],
        instruction: str,
    ) -> list[dict[str, np.ndarray]]:
        """Return an absolute raw 54-D action chunk in XPolicyLab key order."""

        import torch
        from torch.nn.utils.rnn import pad_sequence

        raw = self._make_sample(history, instruction)
        images = raw["image"].to(device=self.model_device, dtype=self.dtype)
        input_ids = pad_sequence(
            [raw["input_ids"]], batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.model_device)
        labels = pad_sequence([raw["labels"]], batch_first=True, padding_value=-100).to(
            self.model_device
        )
        with torch.inference_mode():
            output = self.model.forward(
                images=images,
                input_ids=input_ids,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                labels=labels,
                inference=True,
                raw_action_labels=None,
                raw_action_masks=None,
                raw_proprio_inputs=raw["proprio_input"].to(
                    self.model_device, dtype=self.dtype
                ),
            )
        if not hasattr(output, "prediction"):
            raise RuntimeError("EgoVLA inference output is missing prediction")
        prediction = output.prediction.detach().float().cpu().numpy()
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        expected = (self.predict_future_step, MODEL_OUTPUT_DIM)
        if prediction.shape != expected:
            raise RuntimeError(
                f"SparkArena trajectory decoder returned {prediction.shape}, expected {expected}"
            )
        return decode_joint_prediction(prediction, self.action_minimum, self.action_scale)

__all__ = [
    "EgoVLASparkArenaInference",
    "decode_joint_prediction",
    "extract_head_rgb",
    "logical_state_from_observation",
    "normalize_joint_state",
    "validate_instruction",
]
