from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


IMAGE_KEYS = {
    "exterior_rgb": ("cam_high", "cam_head", "cam_third_view"),
    "left_wrist_rgb": ("cam_left_wrist",),
    "right_wrist_rgb": ("cam_right_wrist",),
}
EMBODIMENT_BY_ROBOT = {
    "arx_x5": "robodojo_arx_x5",
    "piper": "robodojo_piper",
    "piper_x": "robodojo_piper_x",
}


class Model(ModelTemplate):
    """RoboDojo-real adapter for checkpoints produced by the current G05 runtime."""

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = str(model_cfg.get("action_type") or "")
        if self.action_type != "joint":
            raise ValueError(f"RoboDojo real requires action_type=joint, got {self.action_type!r}")

        self.env_cfg_type = str(model_cfg.get("env_cfg_type") or "")
        if self.env_cfg_type not in EMBODIMENT_BY_ROBOT:
            raise ValueError(
                f"unsupported RoboDojo real robot {self.env_cfg_type!r}; "
                f"expected one of {sorted(EMBODIMENT_BY_ROBOT)}"
            )
        self.embodiment = str(model_cfg.get("eval_embodiment") or "")
        expected = EMBODIMENT_BY_ROBOT[self.env_cfg_type]
        if self.embodiment != expected:
            raise ValueError(
                f"env_cfg_type={self.env_cfg_type!r} requires "
                f"eval_embodiment={expected!r}, got {self.embodiment!r}"
            )

        self.frequency = float(model_cfg.get("frequency"))
        if self.frequency != 25.0:
            raise ValueError(f"RoboDojo real is trained and served at 25 Hz, got {self.frequency:g}")
        self.action_steps = int(model_cfg.get("action_steps"))
        if not 1 <= self.action_steps <= 32:
            raise ValueError(f"action_steps must be in [1, 32], got {self.action_steps}")
        if str(model_cfg.get("action_source")) != "fm":
            raise ValueError("this real submission is validated only with action_source=fm")

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.inference_batch_size = max(1, int(model_cfg.get("inference_batch_size") or 1))
        self.batch_size = self.inference_batch_size
        self._obs: dict[str, Any] | None = None
        self._obs_batch: list[dict[str, Any]] = []

        policy_dir = Path(__file__).resolve().parent
        checkpoint_root = resolve_checkpoint_root(
            model_cfg,
            policy_dir / "checkpoints",
            policy_dir=policy_dir,
        )
        runtime_root = checkpoint_root / "inference_runtime"
        if not (runtime_root / "src" / "g05").is_dir():
            raise FileNotFoundError(
                "G05 real bundle is missing inference_runtime/src/g05: "
                f"{checkpoint_root}"
            )
        for path in (
            runtime_root / "src",
            runtime_root / "third_party" / "galaxea_dataset" / "src",
            runtime_root / "third_party" / "galaxea_tokenizer" / "src",
            runtime_root,
        ):
            sys.path.insert(0, str(path))
        os.chdir(runtime_root)

        checkpoint_path = _resolve_checkpoint_file(checkpoint_root)

        from g05.models.g05.inferencer import PolicyInferencer
        from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
        from scripts.serve_policy import build_obs_dict, setup

        run_dir = find_run_dir(str(checkpoint_path))
        cfg = load_config_from_run_dir(
            run_dir,
            str(checkpoint_path),
            [
                "model.model_arch.discrete_action=false",
                "model.model_arch.continuous_action=true",
                "model.model_arch.return_continuous_action=true",
                "model.processor.discrete_action=false",
                f"eval_embodiment={self.embodiment}",
            ],
        )
        _validate_checkpoint_contract(cfg, run_dir, self.embodiment)
        self.policy, self.processor = setup(cfg, device="cuda")
        if bool(self.policy.discrete_action) or not bool(self.policy.continuous_action):
            raise ValueError("deployment must expose FM only: discrete=false, continuous=true")
        self.inferencer = PolicyInferencer(self.policy, self.processor, device="cuda")
        self._build_obs_dict = build_obs_dict
        print(
            f"[G05 real] robot={self.env_cfg_type} embodiment={self.embodiment} "
            f"frequency={self.frequency:g}Hz action_steps={self.action_steps} "
            f"checkpoint={checkpoint_path}"
        )

    def update_obs(self, obs):
        self._obs = obs
        self._obs_batch = []

    def update_obs_batch(self, obs_list):
        self._obs = None
        self._obs_batch = list(obs_list)

    def get_action(self):
        if self._obs is None:
            raise RuntimeError("update_obs must be called before get_action")
        return self._predict([self._obs])[0]

    def get_action_batch(self, env_idx_list=None):
        if not self._obs_batch:
            raise RuntimeError("update_obs_batch must be called before get_action_batch")
        observations = self._obs_batch
        if env_idx_list is not None and len(env_idx_list) != len(observations):
            raise ValueError(
                f"env_idx_list length={len(env_idx_list)} does not match "
                f"observation batch={len(observations)}"
            )
        return self._predict(observations)

    def reset(self):
        self._obs = None
        self._obs_batch = []

    def _predict(self, observations: list[dict[str, Any]]):
        prepared = [self._build_obs_dict(self._to_g05_obs(obs), self.processor) for obs in observations]
        outputs = []
        for start in range(0, len(prepared), self.inference_batch_size):
            outputs.extend(self.inferencer.infer(prepared[start : start + self.inference_batch_size]))
        return [self._format_action_chunk(output) for output in outputs]

    def _to_g05_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        packed = pack_robot_state(
            obs,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32)
        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("RoboDojo real observation requires a non-empty instruction string")
        additional_info = obs.get("additional_info") or {}
        observed_frequency = float(additional_info.get("frequency", self.frequency))
        if observed_frequency != self.frequency:
            raise ValueError(
                f"observation frequency={observed_frequency:g} does not match "
                f"checkpoint frequency={self.frequency:g}"
            )
        return {
            "images": {key: self._extract_image(obs, candidates) for key, candidates in IMAGE_KEYS.items()},
            "state": {
                "left_control": packed[0:6],
                "left_gripper": packed[6:7],
                "right_control": packed[7:13],
                "right_gripper": packed[13:14],
            },
            "task": instruction,
            "frequency": self.frequency,
            "embodiment_type": self.embodiment,
        }

    @staticmethod
    def _extract_image(obs: dict[str, Any], candidates: tuple[str, ...]) -> np.ndarray:
        vision = obs.get("vision")
        if not isinstance(vision, dict):
            raise KeyError("observation is missing vision")
        for name in candidates:
            view = vision.get(name)
            if isinstance(view, dict) and "color" in view:
                return _as_chw_uint8(view["color"])
        raise KeyError(f"missing camera image; tried {candidates}")

    def _format_action_chunk(self, action: dict[str, Any]):
        action = {key: value for key, value in action.items() if not key.startswith("_")}
        required = ("left_control", "left_gripper", "right_control", "right_gripper")
        missing = [key for key in required if key not in action]
        if missing:
            raise KeyError(f"G05 output missing {missing}; available={sorted(action)}")
        parts = {key: _as_horizon(action[key]) for key in required}
        horizon = min(self.action_steps, *(parts[key].shape[0] for key in required))
        result = []
        for index in range(horizon):
            packed = np.concatenate([parts[key][index] for key in required]).astype(np.float32)
            result.append(
                unpack_robot_state(
                    packed,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            )
        return result


def _resolve_checkpoint_file(root: Path) -> Path:
    if root.is_file():
        return root
    candidates = [root / "checkpoints" / "checkpoint.pt", root / "checkpoint.pt"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"checkpoint bundle must contain exactly one checkpoint.pt; checked {candidates}"
        )
    return existing[0]


def _validate_checkpoint_contract(cfg, run_dir: Path, embodiment: str) -> None:
    datasets = cfg.data.get("datasets")
    if datasets is None or list(datasets) != [embodiment]:
        raise ValueError(
            f"checkpoint must contain exactly data.datasets.{embodiment}; "
            f"got {list(datasets) if datasets is not None else None}"
        )
    if str(datasets[embodiment].embodiment) != embodiment:
        raise ValueError("checkpoint dataset embodiment does not match the selected robot")
    stats_path = run_dir / "dataset_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"checkpoint bundle is missing {stats_path.name}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if set(stats) != {embodiment}:
        raise ValueError(f"dataset_stats.json keys must be [{embodiment!r}], got {sorted(stats)}")
    if int(datasets[embodiment].action_size) != 32:
        raise ValueError("checkpoint action horizon must be 32")


def _as_chw_uint8(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"camera image must be 3D, got {array.shape}")
    if array.shape[0] != 3 and array.shape[-1] == 3:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != 3:
        raise ValueError(f"camera image must have three RGB channels, got {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"camera image must be uint8 RGB, got {array.dtype}")
    return np.ascontiguousarray(array)


def _as_horizon(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"action part must be [T,D], got {array.shape}")
    return array
