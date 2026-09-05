from __future__ import annotations

import collections
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_batch_size,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


IMAGE_CANDIDATES = {
    "cam_high": ("cam_high", "cam_head", "cam_third_view"),
    "cam_left_wrist": ("cam_left_wrist",),
    "cam_right_wrist": ("cam_right_wrist",),
}
BATCH_OBSERVATIONS_KEY = "__xpolicylab_batch_observations__"
BATCH_ENV_INDICES_KEY = "__xpolicylab_batch_env_indices__"
G05_HISTORY_OBSERVATIONS_KEY = "__g05_history_observations__"


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = str(model_cfg.get("action_type", "joint"))
        if self.action_type != "joint":
            raise ValueError(f"G0.5 RoboDojo adapter supports action_type=joint, got {self.action_type}")

        self.env_cfg_type = str(model_cfg.get("env_cfg_type", "arx_x5"))
        self.embodiment_type = str(model_cfg.get("eval_embodiment") or "robodojo")
        self.action_steps = int(model_cfg.get("action_steps") or 16)
        self.default_frequency = float(model_cfg.get("frequency") or 25)
        self.inference_batch_size = max(
            1,
            int(
                model_cfg.get("inference_batch_size")
                or os.environ.get("ROBODOJO_G05_INFERENCE_BATCH_SIZE", "8")
            ),
        )

        self.robot_action_dim_info = self._resolve_robot_action_dim_info()
        self.batch_size = self._resolve_batch_size()
        self._obs: dict[str, Any] | None = None
        self._obs_batch: list[dict[str, Any]] = []
        self._obs_batch_env_keys: list[Any] | None = None
        self._history_buffers: dict[Any, dict[str, Any]] = {}

        raw_g05_root = (
            model_cfg.get("g05_root")
            or os.environ.get("G05_ROOT")
            or (Path(__file__).resolve().parent / "G05")
        )
        g05_root = Path(str(raw_g05_root)).expanduser().resolve()
        if not g05_root.exists():
            raise FileNotFoundError(f"G0.5 repo not found: {g05_root}")
        for path in (g05_root / "src", g05_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        os.chdir(g05_root)

        ckpt_path = self._resolve_ckpt_path(model_cfg)
        requested_action_source = str(
            model_cfg.get("action_source")
            or os.environ.get("ROBODOJO_G05_ACTION_SOURCE")
            or "auto"
        )
        hydra_overrides = [
            str(item)
            for item in (model_cfg.get("hydra_overrides") or [])
            if not str(item).startswith(
                (
                    "model.model_arch.discrete_action=",
                    "model.model_arch.continuous_action=",
                    "model.model_arch.return_continuous_action=",
                    "model.processor.discrete_action=",
                )
            )
        ]
        if requested_action_source == "ar":
            hydra_overrides.extend(
                [
                    "model.model_arch.discrete_action=true",
                    "model.model_arch.continuous_action=false",
                    "model.model_arch.return_continuous_action=true",
                    "model.processor.discrete_action=true",
                ]
            )
        elif requested_action_source == "fm":
            hydra_overrides.extend(
                [
                    "model.model_arch.discrete_action=false",
                    "model.model_arch.continuous_action=true",
                    "model.model_arch.return_continuous_action=true",
                    "model.processor.discrete_action=false",
                ]
            )
        hydra_overrides.append(f"eval_embodiment={self.embodiment_type}")

        from g05.models.g05.inferencer import PolicyInferencer
        from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
        from g05.utils.eval.eval_utils import filter_embodiment
        from scripts.serve_policy import build_obs_dict, setup

        run_dir = find_run_dir(str(ckpt_path))
        cfg = load_config_from_run_dir(run_dir, str(ckpt_path), hydra_overrides)
        if self.embodiment_type and "embodiment_datasets" in cfg.data:
            filter_embodiment(cfg, self.embodiment_type)

        self._build_obs_dict = build_obs_dict
        self._build_obs_accepts_buffers = _call_accepts_history_buffers(build_obs_dict)
        device = "cuda"
        self.policy, self.processor = setup(cfg, device=device)
        self.discrete_action = bool(
            getattr(self.policy, "discrete_action", cfg.model.model_arch.discrete_action)
        )
        self.continuous_action = bool(
            getattr(self.policy, "continuous_action", cfg.model.model_arch.continuous_action)
        )
        if requested_action_source == "auto":
            # Preserve existing joint/FM behavior while making AR-only explicit.
            requested_action_source = "fm" if self.continuous_action else "ar"
        if requested_action_source == "fm" and not self.continuous_action:
            raise ValueError("G05 action_source=fm requested but checkpoint has continuous_action=false")
        if requested_action_source == "ar" and not self.discrete_action:
            raise ValueError("G05 action_source=ar requested but checkpoint has discrete_action=false")
        if requested_action_source not in {"ar", "fm", "both"}:
            raise ValueError(f"Unsupported G05 action_source={requested_action_source!r}")
        self.action_source = requested_action_source
        self.inferencer = PolicyInferencer(
            self.policy,
            self.processor,
            device=device,
            action_source=self.action_source,
        )
        print(
            f"[G05] loaded ckpt={ckpt_path} run_dir={run_dir} "
            f"embodiment={self.embodiment_type} "
            f"inference_batch_size={self.inference_batch_size} "
            f"num_obs_steps={getattr(self.processor, 'num_obs_steps', 1)} "
            f"action_source={self.action_source} "
            f"discrete_action={self.discrete_action} "
            f"continuous_action={self.continuous_action}"
        )

    def _resolve_robot_action_dim_info(self) -> dict[str, list[int]]:
        try:
            return get_robot_action_dim_info(self.env_cfg_type)
        except Exception as exc:
            if self.env_cfg_type in {"arx_x5", "piper", "piper_x"}:
                print(f"[G05] using built-in {self.env_cfg_type} dims because env_cfg lookup failed: {exc}")
                return {"arm_dim": [6, 6], "ee_dim": [1, 1]}
            raise

    def _resolve_batch_size(self) -> int:
        try:
            return int(get_batch_size(self.env_cfg_type))
        except Exception:
            return 1

    def _resolve_ckpt_path(self, model_cfg) -> Path:
        raw = model_cfg.get("ckpt_path") or os.environ.get("G05_CKPT_PATH")
        if not raw:
            raise FileNotFoundError("Set ckpt_path in deploy.yml/overrides or export G05_CKPT_PATH")

        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        path = path.resolve()

        if path.is_file():
            return path
        if not path.is_dir():
            raise FileNotFoundError(f"G0.5 checkpoint path not found: {path}")

        candidates = [path / "last.pt"]
        candidates.extend(sorted((path / "checkpoints").glob("step_*.pt"), key=_step_sort_key))
        candidates.extend(sorted(path.glob("**/step_*.pt"), key=_step_sort_key))
        candidates.extend(path.glob("model_state_dict.pt"))
        for candidate in reversed(candidates):
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"No G0.5 .pt checkpoint found under {path}")

    def update_obs(self, obs):
        if isinstance(obs, dict) and BATCH_OBSERVATIONS_KEY in obs:
            observations = obs[BATCH_OBSERVATIONS_KEY]
            if not isinstance(observations, (list, tuple)):
                raise TypeError(f"{BATCH_OBSERVATIONS_KEY} must contain a list")
            self._obs = None
            self._obs_batch = list(observations)
            env_keys = obs.get(BATCH_ENV_INDICES_KEY)
            if env_keys is not None:
                if not isinstance(env_keys, (list, tuple)):
                    raise TypeError(f"{BATCH_ENV_INDICES_KEY} must contain a list")
                if len(env_keys) != len(self._obs_batch):
                    raise ValueError(
                        f"{BATCH_ENV_INDICES_KEY} length={len(env_keys)} does not match "
                        f"batch size={len(self._obs_batch)}"
                    )
                self._obs_batch_env_keys = list(env_keys)
            else:
                self._obs_batch_env_keys = None
            return
        self._obs = obs
        self._obs_batch = []
        self._obs_batch_env_keys = None

    def update_obs_batch(self, obs_list):
        self._obs_batch = list(obs_list)
        self._obs_batch_env_keys = [
            self._env_key_from_obs(obs, fallback_idx=i) for i, obs in enumerate(self._obs_batch)
        ]

    def get_action(self):
        if self._obs_batch:
            return self.get_action_batch()
        if self._obs is None:
            raise RuntimeError("update_obs must be called before get_action")
        return self._predict_chunk(self._obs)

    def get_action_batch(self, env_idx_list=None):
        obs_list = self._obs_batch
        if env_idx_list is not None:
            obs_list = obs_list[: len(env_idx_list)]
        if not obs_list:
            size = len(env_idx_list) if env_idx_list is not None else self.batch_size
            return [self.get_action() for _ in range(size)]
        env_keys = (
            list(env_idx_list)
            if env_idx_list is not None
            else self._obs_batch_env_keys
        )
        if env_keys is None:
            env_keys = [
                self._env_key_from_obs(obs, fallback_idx=i) for i, obs in enumerate(obs_list)
            ]
        else:
            env_keys = list(env_keys)[: len(obs_list)]
        return self._predict_chunks(obs_list, env_keys=env_keys)

    def reset(self):
        self._obs = None
        self._obs_batch = []
        self._obs_batch_env_keys = None
        self._history_buffers.clear()

    def _predict_chunk(self, obs: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        return self._predict_chunks([obs], env_keys=[self._env_key_from_obs(obs, fallback_idx=0)])[0]

    def _predict_chunks(
        self, observations: list[dict[str, Any]], env_keys: list[Any] | None = None
    ) -> list[list[dict[str, np.ndarray]]]:
        if env_keys is None:
            env_keys = [
                self._env_key_from_obs(obs, fallback_idx=i) for i, obs in enumerate(observations)
            ]
        obs_dicts = []
        for i, obs in enumerate(observations):
            raw_obs = self._to_g05_raw_obs(obs)
            explicit_history = self._explicit_history_buffers_from_obs(obs, raw_obs)
            if explicit_history is None:
                frame_buffers, state_buffers = self._update_history_buffers(env_keys[i], raw_obs)
            else:
                frame_buffers, state_buffers = explicit_history
            if self._build_obs_accepts_buffers:
                obs_dicts.append(
                    self._build_obs_dict(
                        raw_obs,
                        self.processor,
                        frame_buffers=frame_buffers,
                        state_buffers=state_buffers,
                    )
                )
            else:
                if frame_buffers is not None or state_buffers is not None:
                    raise RuntimeError(
                        "G05 mem/history evaluation requires a G05_ROOT whose "
                        "scripts.serve_policy.build_obs_dict accepts frame_buffers/state_buffers; "
                        "use a G0.5 checkout with history-buffer support."
                    )
                obs_dicts.append(self._build_obs_dict(raw_obs, self.processor))
        batch_size = max(1, int(getattr(self, "inference_batch_size", len(obs_dicts))))
        actions = []
        for start in range(0, len(obs_dicts), batch_size):
            actions.extend(self.inferencer.infer(obs_dicts[start : start + batch_size]))
        return [self._format_action_chunk(action) for action in actions]

    def _explicit_history_buffers_from_obs(
        self, obs: dict[str, Any], current_raw_obs: dict[str, Any]
    ) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[torch.Tensor]]] | None:
        history = obs.get(G05_HISTORY_OBSERVATIONS_KEY) if isinstance(obs, dict) else None
        num_obs_steps = int(getattr(self.processor, "num_obs_steps", 1))
        if num_obs_steps <= 1 or not history:
            return None

        shape_meta = getattr(self.processor, "shape_meta", {}) or {}
        expected_img_keys = {m["key"] for m in shape_meta.get("images", [])}
        expected_state_keys = {m["key"] for m in shape_meta.get("state", [])}
        if not expected_img_keys or not expected_state_keys:
            return None

        raw_history = [self._to_g05_raw_obs(item) for item in list(history)[-num_obs_steps:]]
        if not raw_history:
            raw_history = [current_raw_obs]
        while len(raw_history) < num_obs_steps:
            raw_history.insert(0, raw_history[0])

        frames = {k: [] for k in expected_img_keys}
        states = {k: [] for k in expected_state_keys}
        for raw in raw_history[-num_obs_steps:]:
            for k in expected_img_keys:
                if k in raw["images"]:
                    frames[k].append(torch.from_numpy(np.array(raw["images"][k], copy=True)))
            for k in expected_state_keys:
                if k in raw["state"]:
                    states[k].append(torch.from_numpy(np.array(raw["state"][k], copy=True)).float())

        if any(len(v) != num_obs_steps for v in frames.values()) or any(
            len(v) != num_obs_steps for v in states.values()
        ):
            return None
        return frames, states

    def _env_key_from_obs(self, obs: dict[str, Any], fallback_idx: int | None = None) -> Any:
        for key in ("env_idx", "env_id", "env_index"):
            if isinstance(obs, dict) and key in obs:
                return obs[key]
        additional_info = obs.get("additional_info") if isinstance(obs, dict) else None
        if isinstance(additional_info, dict):
            for key in ("env_idx", "env_id", "env_index"):
                if key in additional_info:
                    return additional_info[key]
        return "single" if fallback_idx is None else f"batch:{fallback_idx}"

    def _update_history_buffers(
        self, env_key: Any, raw_obs: dict[str, Any]
    ) -> tuple[dict[str, collections.deque] | None, dict[str, collections.deque] | None]:
        num_obs_steps = int(getattr(self.processor, "num_obs_steps", 1))
        if num_obs_steps <= 1:
            return None, None

        shape_meta = getattr(self.processor, "shape_meta", {}) or {}
        expected_img_keys = {m["key"] for m in shape_meta.get("images", [])}
        expected_state_keys = {m["key"] for m in shape_meta.get("state", [])}
        if not expected_img_keys or not expected_state_keys:
            return None, None

        entry = self._history_buffers.get(env_key)
        if entry is None:
            entry = {
                "frames": {
                    k: collections.deque(maxlen=num_obs_steps) for k in expected_img_keys
                },
                "states": {
                    k: collections.deque(maxlen=num_obs_steps) for k in expected_state_keys
                },
                "initialized": False,
                "num_obs_steps": num_obs_steps,
            }
            self._history_buffers[env_key] = entry
        elif entry.get("num_obs_steps") != num_obs_steps:
            entry["frames"] = {
                k: collections.deque(maxlen=num_obs_steps) for k in expected_img_keys
            }
            entry["states"] = {
                k: collections.deque(maxlen=num_obs_steps) for k in expected_state_keys
            }
            entry["initialized"] = False
            entry["num_obs_steps"] = num_obs_steps

        img_tensors = {
            k: torch.from_numpy(np.array(v, copy=True))
            for k, v in raw_obs["images"].items()
            if k in expected_img_keys
        }
        state_tensors = {
            k: torch.from_numpy(np.array(v, copy=True)).float()
            for k, v in raw_obs["state"].items()
            if k in expected_state_keys
        }

        if not entry["initialized"]:
            for _ in range(num_obs_steps):
                for k, t in img_tensors.items():
                    entry["frames"][k].append(t.clone())
                for k, t in state_tensors.items():
                    entry["states"][k].append(t.clone())
            entry["initialized"] = True
        else:
            for k, t in img_tensors.items():
                entry["frames"][k].append(t)
            for k, t in state_tensors.items():
                entry["states"][k].append(t)

        return entry["frames"], entry["states"]

    def _format_action_chunk(self, action) -> list[dict[str, np.ndarray]]:
        action.pop("_cot_text", None)
        action.pop("_absent_keys", None)

        parts = {key: _to_horizon_array(value) for key, value in action.items()}
        left_key = "left_arm" if "left_arm" in parts else "left_control"
        right_key = "right_arm" if "right_arm" in parts else "right_control"
        required = [left_key, "left_gripper", right_key, "right_gripper"]
        missing = [key for key in required if key not in parts]
        if missing:
            raise KeyError(f"G05 action output missing keys {missing}; available={sorted(parts)}")
        horizon = min(self.action_steps, min(parts[key].shape[0] for key in required))
        result = []
        for t in range(horizon):
            packed = np.concatenate(
                [
                    parts[left_key][t],
                    parts["left_gripper"][t],
                    parts[right_key][t],
                    parts["right_gripper"][t],
                ],
                axis=-1,
            ).astype(np.float32)
            result.append(
                unpack_robot_state(
                    packed,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            )
        return result

    def _to_g05_raw_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        packed_state = pack_robot_state(
            obs,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32)

        instruction = obs.get("instruction", obs.get("instructions", self.model_cfg.get("task_name", "")))
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""

        additional_info = obs.get("additional_info") or {}
        images = self._build_raw_images(obs)
        state = self._build_raw_state(packed_state)
        return {
            "images": images,
            "state": state,
            "task": str(instruction),
            "frequency": float(additional_info.get("frequency", self.default_frequency)),
            "embodiment_type": self.embodiment_type,
        }

    def _build_raw_state(self, packed_state: np.ndarray) -> dict[str, np.ndarray]:
        """Map RoboDojo joint-state names to the state keys expected by the active G05 processor."""
        shape_meta = getattr(self.processor, "shape_meta", {}) or {}
        expected = {m.get("key") for m in shape_meta.get("state", []) if isinstance(m, dict)}
        if {"left_control", "left_gripper", "right_control", "right_gripper"}.issubset(expected):
            return {
                "left_control": packed_state[0:6],
                "left_gripper": packed_state[6:7],
                "right_control": packed_state[7:13],
                "right_gripper": packed_state[13:14],
            }
        return {
            "left_arm": packed_state[0:6],
            "left_gripper": packed_state[6:7],
            "right_arm": packed_state[7:13],
            "right_gripper": packed_state[13:14],
        }

    def _build_raw_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Map RoboDojo camera names to the image keys expected by the active G05 processor.

        Older G05 runs used RoboDojo-native keys:
          cam_high, cam_left_wrist, cam_right_wrist
        G05 checkpoints use canonical schema keys:
          exterior_rgb, left_wrist_rgb, right_wrist_rgb
        """
        shape_meta = getattr(self.processor, "shape_meta", {}) or {}
        expected = {m.get("key") for m in shape_meta.get("images", []) if isinstance(m, dict)}
        if {"exterior_rgb", "left_wrist_rgb", "right_wrist_rgb"}.issubset(expected):
            return {
                "exterior_rgb": self._extract_image(obs, "cam_high"),
                "left_wrist_rgb": self._extract_image(obs, "cam_left_wrist"),
                "right_wrist_rgb": self._extract_image(obs, "cam_right_wrist"),
            }
        return {
            "cam_high": self._extract_image(obs, "cam_high"),
            "cam_left_wrist": self._extract_image(obs, "cam_left_wrist"),
            "cam_right_wrist": self._extract_image(obs, "cam_right_wrist"),
        }

    def _extract_image(self, obs: dict[str, Any], g05_key: str) -> np.ndarray:
        vision = obs.get("vision") or {}
        for candidate in IMAGE_CANDIDATES[g05_key]:
            view = vision.get(candidate)
            if isinstance(view, dict):
                if "color" in view:
                    return _as_chw_uint8(view["color"])
                if "colors" in view:
                    return _as_chw_uint8(view["colors"])
            elif view is not None:
                return _as_chw_uint8(view)
        raise KeyError(f"Missing image for {g05_key}; tried {IMAGE_CANDIDATES[g05_key]}")


def _step_sort_key(path: Path) -> int:
    stem = path.stem
    if stem.startswith("step_"):
        try:
            return int(stem.split("_", 1)[1])
        except ValueError:
            return -1
    return -1


def _call_accepts_history_buffers(fn) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "frame_buffers" in params and "state_buffers" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _as_chw_uint8(value) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        value = decode_image_bit(value)
    arr = np.asarray(value)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"image must be 3D, got shape={arr.shape}")
    if arr.shape[0] == 3:
        chw = arr
    elif arr.shape[-1] == 3:
        chw = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"image must have 3 channels, got shape={arr.shape}")
    return np.ascontiguousarray(chw.astype(np.uint8, copy=False))


def _to_horizon_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"action part must be [H,D], got shape={arr.shape}")
    return arr
