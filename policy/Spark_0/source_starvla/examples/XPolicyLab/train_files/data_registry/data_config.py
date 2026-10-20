"""XPolicyLab benchmark data config for StarVLA.

Embodiment schemas (camera/state/action keys, per-key dims, which keys carry
EE poses) are defined in YAML files and loaded at import time, so a new robot
only requires a new YAML — not a code change.

Search order for the YAML folder:
  1. ``SPARK_0_EMBODIMENT_DIR`` environment variable
  2. ``<adapter root>/model_configs/embodiments/`` (the directory containing
     source_starvla, then its model_configs/embodiments)
  3. ``<this directory>/embodiments/``

Each YAML file registers a robot_type named after the file stem. Two built-in
schemas (``robodojo_arx_x5_h50_q99``, ``xpolicylab_arx_x5``) are always
available as fallbacks even when no YAML folder exists.

A schema describes what the robot IS (including the camera *catalog*); what a
TRAINING RUN decides about it (active cameras, chunk length, per-modality
normalization, observation indices, pose formats) comes from the train config
and reaches this module as environment variables that ``_apply_run_overrides``
layers on top of every spec — see ``SPARK_0_XPOLICY_SPEC_OVERRIDES`` and
``SPARK_0_XPOLICY_*_POSE_FORMAT``.

The XPolicyLab dual-arm joint layout is:
    [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)

# Pose-key encodings for the ``pose_representation`` block; must stay in sync
# with ee_delta_dataset.py (raw stored layout is 7-d xyz_wxyz, converted
# encodings are 9-d xyz+rot6d). Keyed as format-name -> model dim.
_POSE_RAW_FORMAT = "xyz_wxyz"
_POSE_RAW_DIM = 7
_POSE_FORMAT_DIMS = {
    "raw": _POSE_RAW_DIM,
    "abs_xyz_rot6d": 9,
    "cam_abs_xyz_rot6d": 9,
    "delta_xyz_rot6d": 9,
}
_STATE_POSE_FORMATS = ("raw", "abs_xyz_rot6d", "cam_abs_xyz_rot6d")
_ACTION_POSE_FORMATS = ("raw", "abs_xyz_rot6d", "cam_abs_xyz_rot6d", "delta_xyz_rot6d")

# Normalization of the CONVERTED pose-action values (datasets.vla_data.
# pose_normalization). `q99` is the legacy affine map, whose normalized 0 is
# the midpoint (q01+q99)/2 — a real motion, not "hold". `hold_centered`
# rescales about the representation's hold vector instead, so 0 decodes to
# "do not move". `hold_centered_perrow` keeps that origin and additionally
# takes the span from each horizon row's own quantiles, which equalizes the
# target amplitude across the chunk. Both hold modes need a constant hold, so
# they are limited to the formats in _POSE_HOLD_VALUES.
_POSE_NORMALIZATIONS = ("q99", "hold_centered", "hold_centered_perrow")
_HOLD_CENTERED_NORMALIZATIONS = ("hold_centered", "hold_centered_perrow")
_POSE_HOLD_VALUES = {
    # delta_xyz_rot6d: zero translation + rot6d of the identity rotation
    # (first two ROWS of I, row-major).
    "delta_xyz_rot6d": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
}


def _cumulative_ranges(key_dims: dict) -> dict:
    ranges = {}
    cursor = 0
    for key, dim in key_dims.items():
        ranges[key] = (cursor, cursor + int(dim))
        cursor += int(dim)
    return ranges


def _load_ee_delta_dataset_module():
    """Import the sibling ee_delta_dataset.py.

    This module is loaded standalone by the registry (importlib by path), so
    relative imports are unavailable; load the sibling the same way and cache
    it in sys.modules.
    """
    module_name = "_data_registry_XPolicyLab.ee_delta_dataset"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = Path(__file__).resolve().parent / "ee_delta_dataset.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class YamlEmbodimentDataConfig:
    """DataConfig built from an embodiment YAML spec.

    Exposes the same interface as the previous hand-written config classes
    (video/state/action keys, per-key dims, modality_key_ranges,
    modality_config(), transform(), embodiment_tag).
    """

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.embodiment_tag = EmbodimentTag(spec.get("embodiment_tag", "new_embodiment"))

        self.video_keys = list(spec["video_keys"])
        self.language_keys = list(
            spec.get("language_keys", ["annotation.human.action.task_description"])
        )

        state_items = dict(spec["state_keys"])
        action_items = dict(spec["action_keys"])
        self.state_keys = [f"state.{key}" for key in state_items]
        self.action_keys = [f"action.{key}" for key in action_items]

        # Optional pose re-encoding (see ee_delta_dataset.py). Listed keys are
        # stored RAW as 7-d xyz_wxyz in the LeRobot data; per-modality formats
        # declare what the MODEL sees (e.g. 9-d abs_xyz_rot6d for state,
        # 9-d delta_xyz_rot6d chunk-wise deltas for action).
        pose_repr = spec.get("pose_representation") or {}
        self.pose_keys = [str(key) for key in pose_repr.get("keys", [])]
        raw_format = pose_repr.get("raw_format", _POSE_RAW_FORMAT)
        self.state_pose_format = pose_repr.get("state", "abs_xyz_rot6d")
        self.action_pose_format = pose_repr.get("action", "delta_xyz_rot6d")
        # Per-run switch frozen by train.sh; "q99" reproduces the legacy path.
        self.pose_normalization = str(pose_repr.get("normalization", "q99"))
        # Optional train-time reference-pose perturbation (closed-loop
        # robustness augmentation; see EEPoseReprLeRobotDataset.ref_noise).
        # {prob, rot_deg, trans_m}; absent/empty -> disabled (exact old
        # behaviour). Only consumed by make_dataset, so eval-side users of
        # this config (norm processor, deployment codec) are unaffected.
        self.pose_ref_noise = dict(pose_repr.get("ref_noise") or {})
        if self.pose_ref_noise:
            prob = float(self.pose_ref_noise.get("prob", 0.0))
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"[{name}] pose_representation.ref_noise.prob={prob} not in [0, 1]")
        if self.pose_keys:
            if raw_format != _POSE_RAW_FORMAT:
                raise ValueError(
                    f"[{name}] pose_representation.raw_format={raw_format!r} is not "
                    f"supported; the stored layout must be {_POSE_RAW_FORMAT!r} "
                    f"({_POSE_RAW_DIM}-d [xyz + quat_wxyz])"
                )
            if self.state_pose_format not in _STATE_POSE_FORMATS:
                raise ValueError(
                    f"[{name}] pose_representation.state={self.state_pose_format!r}; "
                    f"expected one of {_STATE_POSE_FORMATS}"
                )
            if self.action_pose_format not in _ACTION_POSE_FORMATS:
                raise ValueError(
                    f"[{name}] pose_representation.action={self.action_pose_format!r}; "
                    f"expected one of {_ACTION_POSE_FORMATS}"
                )
            if self.pose_normalization not in _POSE_NORMALIZATIONS:
                raise ValueError(
                    f"[{name}] pose_representation.normalization="
                    f"{self.pose_normalization!r}; expected one of {_POSE_NORMALIZATIONS}"
                )
            if (
                self.pose_normalization in _HOLD_CENTERED_NORMALIZATIONS
                and self.action_pose_format not in _POSE_HOLD_VALUES
            ):
                raise ValueError(
                    f"[{name}] pose_representation.normalization="
                    f"{self.pose_normalization!r} needs an action format with a "
                    f"constant hold value ({sorted(_POSE_HOLD_VALUES)}), got "
                    f"{self.action_pose_format!r}. Absolute formats hold at the "
                    "current state, which is not a constant."
                )
            for key in self.pose_keys:
                for items, block in ((state_items, "state_keys"), (action_items, "action_keys")):
                    if key not in items:
                        raise KeyError(
                            f"[{name}] pose_representation key {key!r} not found in {block}"
                        )
                    if int(items[key]) != _POSE_RAW_DIM:
                        raise ValueError(
                            f"[{name}] pose_representation key {key!r} must be stored as "
                            f"{_POSE_RAW_DIM}-d {_POSE_RAW_FORMAT} in the data, got "
                            f"{items[key]}-d; the model-side dim "
                            f"({_POSE_FORMAT_DIMS[self.state_pose_format]}/"
                            f"{_POSE_FORMAT_DIMS[self.action_pose_format]}) is derived, "
                            "do not write it here"
                        )

        def _model_dims(items: dict, pose_format: str) -> dict:
            return {
                key: (_POSE_FORMAT_DIMS[pose_format] if key in self.pose_keys else int(dim))
                for key, dim in items.items()
            }

        # Downstream consumers (framework dims, deployment norm processor)
        # see MODEL-side dims; raw dims are only used to slice the parquet
        # columns, which modality_key_ranges below controls.
        self.state_key_dims = {
            f"state.{key}": dim
            for key, dim in _model_dims(state_items, self.state_pose_format).items()
        }
        self.action_key_dims = {
            f"action.{key}": dim
            for key, dim in _model_dims(action_items, self.action_pose_format).items()
        }
        self.modality_key_ranges = {
            "state": _cumulative_ranges(state_items),
            "action": _cumulative_ranges(action_items),
        }

        # Run-level knobs (train config -> SPARK_0_XPOLICY_SPEC_OVERRIDES ->
        # _apply_run_overrides). `action_horizon` has no safe default: a wrong
        # chunk length silently trains a different contract, so resolve it
        # lazily and fail where it is actually needed.
        self.observation_indices = list(spec.get("observation_indices", [0]))
        self._action_horizon = spec.get("action_horizon")
        self._state_norm = spec.get("state_normalization", "q99")
        self._action_norm = spec.get("action_normalization", "q99")

    @property
    def action_indices(self) -> list:
        if self._action_horizon is None:
            raise ValueError(
                f"[{self.name}] no action horizon resolved for this embodiment. "
                "Set framework.action_model.action_horizon in the train config "
                "(train.sh forwards it as SPARK_0_XPOLICY_SPEC_OVERRIDES)."
            )
        return list(range(int(self._action_horizon)))

    @property
    def state_dim(self) -> int:
        return sum(self.state_key_dims.values())

    @property
    def action_dim(self) -> int:
        return sum(self.action_key_dims.values())

    def _normalization_modes(self, keys: list, norm) -> dict:
        if isinstance(norm, str):
            return {key: norm for key in keys}
        # Per-key mapping uses short key names (without the modality prefix).
        return {key: norm[key.split(".", 1)[1]] for key in keys}

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def _action_pose_normalization(self) -> tuple[dict, dict]:
        """``(normalization_modes, hold_values)`` for the ACTION modality.

        Only the pose keys move to a ``hold_centered*`` mode; hand/gripper
        blocks keep the embodiment's mode. The state modality is never
        hold-centered: its formats are absolute, so "hold" is the current
        state, not a constant.
        """
        modes = self._normalization_modes(self.action_keys, self._action_norm)
        if (
            not self.pose_keys
            or self.pose_normalization not in _HOLD_CENTERED_NORMALIZATIONS
        ):
            return modes, {}
        hold = _POSE_HOLD_VALUES[self.action_pose_format]
        hold_values = {}
        for key in self.pose_keys:
            action_key = f"action.{key}"
            modes[action_key] = self.pose_normalization
            hold_values[action_key] = list(hold)
        return modes, hold_values

    def transform(self):
        action_modes, action_hold_values = self._action_pose_normalization()
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=self._normalization_modes(self.state_keys, self._state_norm),
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                    hold_values=action_hold_values,
                    # Per-row statistics carry one span per chunk row, so bind
                    # the run's chunk length: a disagreement then fails when
                    # the transform is built (train AND eval) instead of
                    # silently rescaling every commanded row. None (horizon
                    # unresolved) skips the check.
                    per_row_horizon=(
                        None if self._action_horizon is None else int(self._action_horizon)
                    ),
                ),
            ]
        )

    def make_dataset(self, dataset_name=None, **dataset_kwargs):
        """Factory hook consumed by ``make_LeRobotSingleDataset``.

        Swaps in EEPoseReprLeRobotDataset when the embodiment declares a
        ``pose_representation`` block; otherwise builds the default dataset.
        """
        # Optional state history (datasets.vla_data.state_history_frames,
        # default 1 == legacy single-frame). Only the STATE modality gets the
        # extra frames [1-N .. 0] (current frame last); video/language keep
        # datasets.vla_data.observation_indices. Chunk-boundary frames are
        # first/last-padded upstream, matching the deploy-side seeding of the
        # history buffer with the first observation.
        frames = int((dataset_kwargs.get("data_cfg") or {}).get("state_history_frames", 1) or 1)
        if frames < 1:
            raise ValueError(
                f"[{self.name}] datasets.vla_data.state_history_frames must be >= 1, got {frames}"
            )
        if frames > 1:
            modality_configs = dataset_kwargs.get("modality_configs")
            if modality_configs is None or "state" not in modality_configs:
                raise ValueError(
                    f"[{self.name}] state_history_frames={frames} requires a state modality"
                )
            state_cfg = modality_configs["state"]
            if list(state_cfg.delta_indices) != [0]:
                raise ValueError(
                    f"[{self.name}] state_history_frames={frames} requires "
                    "datasets.vla_data.observation_indices == [0], got "
                    f"{list(state_cfg.delta_indices)}"
                )
            modality_configs = dict(modality_configs)
            modality_configs["state"] = ModalityConfig(
                delta_indices=list(range(1 - frames, 1)),
                modality_keys=list(state_cfg.modality_keys),
            )
            dataset_kwargs["modality_configs"] = modality_configs

        if not self.pose_keys:
            return LeRobotSingleDataset(**dataset_kwargs)

        # ref_noise resolution: the TRAIN CONFIG (datasets.vla_data.ref_noise,
        # arriving here via data_cfg) takes precedence when the key is present
        # — including an explicit disable ({} or prob: 0). The embodiment
        # YAML's pose_representation.ref_noise is only the fallback default.
        ref_noise = self.pose_ref_noise or None
        data_cfg = dataset_kwargs.get("data_cfg")
        if data_cfg is not None:
            train_ref_noise = data_cfg.get("ref_noise", None)
            if train_ref_noise is not None:
                ref_noise = dict(train_ref_noise)

        # Required from datasets.vla_data (shared across co-train mixture).
        if data_cfg is None or "ee_in_camera_frame" not in data_cfg:
            raise ValueError(
                f"[{self.name}] datasets.vla_data.ee_in_camera_frame is required "
                "for embodiments with pose_representation."
            )
        ee_in_camera_frame = bool(data_cfg["ee_in_camera_frame"])
        # Optional overrides; default from meta/modality.json.
        camera_extrinsics_key = None
        if data_cfg.get("camera_extrinsics_key"):
            camera_extrinsics_key = str(data_cfg["camera_extrinsics_key"])
        camera_extrinsics_semantics = None
        if data_cfg.get("camera_extrinsics_semantics"):
            camera_extrinsics_semantics = str(data_cfg["camera_extrinsics_semantics"])

        ee_delta_module = _load_ee_delta_dataset_module()
        kwargs = dict(
            pose_keys=self.pose_keys,
            state_pose_format=self.state_pose_format,
            action_pose_format=self.action_pose_format,
            ref_noise=ref_noise,
            ee_in_camera_frame=ee_in_camera_frame,
            **dataset_kwargs,
            # The converted STATE statistics rebuild the sampler's history
            # block, so the dataset needs the run's frame count.
            state_history_frames=frames,
        )
        if camera_extrinsics_key is not None:
            kwargs["camera_extrinsics_key"] = camera_extrinsics_key
        if camera_extrinsics_semantics is not None:
            kwargs["camera_extrinsics_semantics"] = camera_extrinsics_semantics
        return ee_delta_module.EEPoseReprLeRobotDataset(**kwargs)


_ARX_X5_BASE_SPEC = {
    "embodiment_tag": "new_embodiment",
    "video_keys": ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"],
    "language_keys": ["annotation.human.action.task_description"],
    "state_keys": {"left_joints": 6, "left_gripper": 1, "right_joints": 6, "right_gripper": 1},
    "action_keys": {"left_joints": 6, "left_gripper": 1, "right_joints": 6, "right_gripper": 1},
    "state_normalization": "q99",
    "action_normalization": "q99",
    "observation_indices": [0],
}

# Always-available fallbacks; YAML files with the same stem take precedence,
# and a run's own values (_apply_run_overrides) take precedence over both.
# Neither ships as a YAML file any more — they exist so pre-EE joint
# checkpoints still resolve their robot_type at eval, which makes them two
# names for the same layout. Like every schema they pin NO run knob (the
# chunk length they used to default to is now the run's to declare), so a
# launch without SPARK_0_XPOLICY_SPEC_OVERRIDES fails loudly instead of
# guessing a horizon.
_BUILTIN_SPECS = {
    "robodojo_arx_x5_h50_q99": dict(_ARX_X5_BASE_SPEC),
    "xpolicylab_arx_x5": dict(_ARX_X5_BASE_SPEC),
}


def _embodiment_dirs() -> list:
    here = Path(__file__).resolve()
    candidates = []
    env_dir = os.environ.get("SPARK_0_EMBODIMENT_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    # data_registry -> train_files -> XPolicyLab -> examples -> source_starvla -> adapter root
    if len(here.parents) > 5:
        candidates.append(here.parents[5] / "model_configs" / "embodiments")
    candidates.append(here.parent / "embodiments")
    return [d for d in candidates if d.is_dir()]


_SPEC_OVERRIDE_KEYS = (
    "action_horizon",
    "state_normalization",
    "action_normalization",
    "observation_indices",
    # Active cameras for this run (subset of the embodiment catalog). train.sh
    # derives it from datasets.vla_data.video_keys / obs_image_size and freezes
    # the resolved full keys here so the registry matches the checkpoint.
    "video_keys",
)


def _apply_run_overrides(spec: dict) -> dict:
    """Layer the TRAIN CONFIG's per-run decisions on top of an embodiment spec.

    Two environment payloads, both exported by train.sh (and replayed by
    setup_eval_policy_server.sh) and both shared across a co-train mixture:

    * ``SPARK_0_XPOLICY_SPEC_OVERRIDES`` — JSON with the top-level run knobs
      (``action_horizon`` from ``framework.action_model``, plus
      ``state_normalization`` / ``action_normalization`` /
      ``observation_indices`` / ``video_keys`` from ``datasets.vla_data``;
      ``video_keys`` is the active-camera subset resolved by train.sh).
    * ``SPARK_0_XPOLICY_{STATE,ACTION}_POSE_FORMAT`` and
      ``SPARK_0_XPOLICY_POSE_NORMALIZATION`` — ``datasets.vla_data.
      {state_pose_format,action_pose_format,pose_normalization}``, applied to
      ``pose_representation.{state,action,normalization}`` for every
      embodiment that declares pose keys.

    When a payload is unset (e.g. eval, where setup_eval_policy_server
    materializes the checkpoint's frozen embodiment_spec, which train.sh
    already wrote these values into), the spec values apply as-is.
    """
    spec_overrides = os.environ.get("SPARK_0_XPOLICY_SPEC_OVERRIDES")
    if spec_overrides:
        parsed = json.loads(spec_overrides)
        spec = {**spec, **{k: v for k, v in parsed.items() if k in _SPEC_OVERRIDE_KEYS}}

    pose_overrides = {
        "state": os.environ.get("SPARK_0_XPOLICY_STATE_POSE_FORMAT"),
        "action": os.environ.get("SPARK_0_XPOLICY_ACTION_POSE_FORMAT"),
        "normalization": os.environ.get("SPARK_0_XPOLICY_POSE_NORMALIZATION"),
    }
    pose_overrides = {field: value for field, value in pose_overrides.items() if value}
    pose_repr = spec.get("pose_representation") or {}
    if pose_overrides and pose_repr.get("keys"):
        spec = dict(spec)
        spec["pose_representation"] = {**pose_repr, **pose_overrides}
    return spec


def _load_embodiment_configs() -> dict:
    configs = {
        name: YamlEmbodimentDataConfig(name, _apply_run_overrides(spec))
        for name, spec in _BUILTIN_SPECS.items()
    }
    # Later directories in the search order have lower priority; iterate in
    # reverse so the highest-priority directory wins on name collisions.
    for directory in reversed(_embodiment_dirs()):
        for yaml_path in sorted(directory.glob("*.yaml")):
            with open(yaml_path, "r", encoding="utf-8") as stream:
                spec = yaml.safe_load(stream)
            if not isinstance(spec, dict):
                continue
            spec = _apply_run_overrides(spec)
            configs[yaml_path.stem] = YamlEmbodimentDataConfig(yaml_path.stem, spec)
    return configs


ROBOT_TYPE_CONFIG_MAP = _load_embodiment_configs()

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "xpolicylab_robodojo_sim_arx_x5_v30": [
        ("RoboDojo_sim_arx-x5_v30", 1.0, "xpolicylab_arx_x5"),
    ],
    "xpolicylab_robodojo_real_piper_x_v30": [
        ("RoboDojo_real_piper-x_lerobot_v30_video", 1.0, "xpolicylab_arx_x5"),
    ],
    "xpolicylab_robodojo_real_piper_lerobot_v30": [
        ("RoboDojo_real_piper_lerobot_v30", 1.0, "xpolicylab_arx_x5"),
    ],
    "xpolicylab_robodojo_real_arx_lerobot_v30": [
        ("RoboDojo_real_arx_lerobot_v30", 1.0, "xpolicylab_arx_x5"),
    ],
    "xpolicylab_stack_bowls_arx_x5_50": [
        ("arx_x5", 1.0, "xpolicylab_arx_x5"),
    ],
    "robodojo_arx_x5_h50_q99": [
        ("RoboDojo_lerobot_v21_video", 1.0, "robodojo_arx_x5_h50_q99"),
    ],
    "robodojo_v21_all_h50_q99": [
        ("RoboDojo_lerobot_v21_video", 1.0, "robodojo_arx_x5_h50_q99"),
    ],
}

# ---------------------------------------------------------------------------
# Runtime mixture registration (wired by train.sh through environment vars).
#
# Single dataset (the default): SPARK_0_XPOLICY_DATASET_NAME names the LeRobot
# dataset dir and SPARK_0_XPOLICY_ROBOT_TYPE the embodiment schema; one entry
# is registered under SPARK_0_XPOLICY_DATA_MIX (default xpolicylab_runtime).
#
# Multi-dataset co-training: a training YAML's `data_mixture` list arrives as
# SPARK_0_XPOLICY_MIXTURE_JSON ([{dataset, weight, embodiment}, ...]) and
# REPLACES the single-dataset entry under the same mixture name.
# ---------------------------------------------------------------------------
_runtime_mix_name = os.environ.get("SPARK_0_XPOLICY_DATA_MIX", "xpolicylab_runtime")


def _require_robot_type(robot_type: str) -> str:
    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        raise KeyError(
            f"robot_type/embodiment {robot_type!r} is not registered; "
            f"choose one of {sorted(ROBOT_TYPE_CONFIG_MAP)}"
        )
    return robot_type


_runtime_dataset_name = os.environ.get("SPARK_0_XPOLICY_DATASET_NAME")
if _runtime_dataset_name:
    _runtime_robot_type = _require_robot_type(
        os.environ.get("SPARK_0_XPOLICY_ROBOT_TYPE", "robodojo_arx_x5_h50_q99")
    )
    DATASET_NAMED_MIXTURES[_runtime_mix_name] = [
        (_runtime_dataset_name, 1.0, _runtime_robot_type),
    ]

_runtime_mixture_json = os.environ.get("SPARK_0_XPOLICY_MIXTURE_JSON")
if _runtime_mixture_json:
    DATASET_NAMED_MIXTURES[_runtime_mix_name] = [
        (
            str(entry["dataset"]),
            float(entry.get("weight", 1.0)),
            _require_robot_type(str(entry["embodiment"])),
        )
        for entry in json.loads(_runtime_mixture_json)
    ]
