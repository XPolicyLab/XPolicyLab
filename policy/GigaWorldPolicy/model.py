from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


_CUR_DIR = Path(__file__).resolve().parent
_XPL_ROOT = _CUR_DIR.parents[2]
_INNER_ROOT = _CUR_DIR / "giga_world_policy"
_SRC_ROOT = _INNER_ROOT / "src"
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"
_ARX_X5_DELTA_MASK = np.array(
    [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
    dtype=bool,
)

for _path in (str(_XPL_ROOT), str(_CUR_DIR), str(_INNER_ROOT), str(_SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.batch_indices import selected_env_ids, validate_observed_ids
from XPolicyLab.utils.checkpoint_resolver import (
    build_run_dir_name,
    candidate_checkpoint_roots,
    ckpt_name_is_path,
)
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _as_path(value: str | None, base: Path = _CUR_DIR) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _pad_or_trim_np(value: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-1] == dim:
        return arr
    if arr.shape[-1] > dim:
        return arr[..., :dim]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, dim - arr.shape[-1])
    return np.pad(arr, pad_width, mode="constant").astype(np.float32)


def _checkpoint_step(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint-"):
        tail = name.split("checkpoint-", 1)[1]
        if tail.isdigit():
            return int(tail)
    return -1


def _choose_checkpoint_file(path: Path, preferred_file: str) -> Path | None:
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        return None
    for name in (preferred_file, "model_ema.pt", "model.pt"):
        candidate = path / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _image_to_uint8_hwc(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(arr)) <= 1.5 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return np.ascontiguousarray(arr)


def _manual_unpack_ee(packed_state: np.ndarray, robot_action_dim_info: dict[str, list[int]]) -> dict[str, np.ndarray]:
    packed = np.asarray(packed_state, dtype=np.float32)
    ee_dims = robot_action_dim_info["ee_dim"]
    if len(ee_dims) == 1:
        return {
            "ee_pose": packed[:7],
            "ee_joint_state": packed[7 : 7 + ee_dims[0]],
        }
    if len(ee_dims) == 2:
        left_end = 7 + ee_dims[0]
        right_pose_end = left_end + 7
        return {
            "left_ee_pose": packed[:7],
            "left_ee_joint_state": packed[7:left_end],
            "right_ee_pose": packed[left_end:right_pose_end],
            "right_ee_joint_state": packed[right_pose_end : right_pose_end + ee_dims[1]],
        }
    raise ValueError(f"Unsupported arm count: {len(ee_dims)}")


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name") or ""
        self.action_type = self.model_cfg.get("action_type") or "joint"
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if not self.env_cfg_type:
            raise ValueError("env_cfg_type is required for GigaWorldPolicy.")

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.xpolicylab_action_dim = self._packed_dim()
        self.model_state_dim = int(self.model_cfg.get("model_state_dim") or self.model_cfg.get("state_dim") or self.xpolicylab_action_dim)
        self.model_action_dim = int(self.model_cfg.get("model_action_dim") or self.model_cfg.get("action_dim") or self.xpolicylab_action_dim)
        self.action_chunk = int(self.model_cfg.get("action_chunk") or 24)
        execute_action_chunk = self.model_cfg.get("execute_action_chunk", self.model_cfg.get("exec_action_chunk"))
        self.execute_action_chunk = (
            self.action_chunk
            if execute_action_chunk in (None, "")
            else max(1, min(int(execute_action_chunk), self.action_chunk))
        )
        self.delta_to_absolute = _parse_bool(self.model_cfg.get("delta_to_absolute"), True)
        self.num_frames = int(self.model_cfg.get("num_frames") or max(self.action_chunk, 24))
        self.load_model = _parse_bool(self.model_cfg.get("load_model"), True)
        self.action_request_count = 0

        self.view_candidates = {
            "left": self.model_cfg.get("left_view_candidates")
            or ["cam_left_wrist", "cam_left", "agentview_left", "left_camera", "cam_head"],
            "wrist": self.model_cfg.get("wrist_view_candidates")
            or ["cam_right_wrist", "cam_wrist", "eye_in_hand", "right_camera", "cam_head"],
            "right": self.model_cfg.get("right_view_candidates")
            or ["cam_head", "cam_right", "agentview_right", "head_camera", "cam_third_view"],
        }

        self._latest_env_idx_list = [0]
        self._latest_observations: dict[int, dict[str, Any]] = {}
        self.policy = self._load_policy() if self.load_model else None

        print(
            "[GigaWorldPolicy] initialized",
            f"load_model={self.load_model}",
            f"xpolicylab_action_dim={self.xpolicylab_action_dim}",
            f"model_state_dim={self.model_state_dim}",
            f"model_action_dim={self.model_action_dim}",
            f"action_chunk={self.action_chunk}",
            f"execute_action_chunk={self.execute_action_chunk}",
            f"delta_to_absolute={self.delta_to_absolute}",
        )

    def _packed_dim(self) -> int:
        if self.action_type == "ee":
            return 7 * len(self.robot_action_dim_info["ee_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        return sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])

    def _resolve_checkpoint_path(self) -> Path:
        preferred_file = str(self.model_cfg.get("checkpoint_file") or "model_ema.pt")
        checkpoint_num = self.model_cfg.get("checkpoint_num") or "latest"

        candidates = candidate_checkpoint_roots(
            self.model_cfg,
            _CHECKPOINTS_DIR,
            policy_dir=_CUR_DIR,
            explicit_keys=("checkpoint_path", "model_path"),
        )
        for root in candidates:
            if not root.exists():
                continue
            if str(checkpoint_num).lower() not in {"", "none", "latest"}:
                names = [str(checkpoint_num)]
                digits = "".join(ch for ch in str(checkpoint_num) if ch.isdigit())
                if digits:
                    names.append(f"checkpoint-{digits}")
                for name in names:
                    chosen = _choose_checkpoint_file(root / name, preferred_file)
                    if chosen is not None:
                        return chosen
            chosen = _choose_checkpoint_file(root, preferred_file)
            if chosen is not None:
                return chosen
            if root.is_dir():
                ckpt_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
                for ckpt_dir in sorted(ckpt_dirs, key=_checkpoint_step, reverse=True):
                    chosen = _choose_checkpoint_file(ckpt_dir, preferred_file)
                    if chosen is not None:
                        return chosen

        raise FileNotFoundError(
            "Could not resolve GigaWorldPolicy checkpoint. Set checkpoint_path/model_path, "
            "or place model_ema.pt/model.pt under checkpoints/<ckpt_name>/checkpoint-<step>."
        )

    def _load_policy(self):
        import torch

        inference_server = importlib.import_module("experiment.xpolicylab.inference_server")
        checkpoint_path = self._resolve_checkpoint_path()
        model_id = _as_path(
            self.model_cfg.get("base_model_path") or self.model_cfg.get("model_id")
            or os.environ.get("GIGAWORLD_PRETRAINED_PATH")
            or os.environ.get("WAN22_DIFFUSERS_PATH")
        )
        stats_path = _as_path(self.model_cfg.get("stats_path") or os.environ.get("GIGAWORLD_NORM_PATH"))
        if stats_path is None:
            # data-setting folder is named by the run-dir name; fall back to a
            # non-path ckpt_name for backward compatibility.
            data_settings: list[str] = []
            run_dir_name = build_run_dir_name(self.model_cfg)
            if run_dir_name:
                data_settings.append(run_dir_name)
            ckpt_name = self.model_cfg.get("ckpt_name")
            if ckpt_name and not ckpt_name_is_path(ckpt_name):
                data_settings.append(str(ckpt_name))
            for data_setting in data_settings:
                candidate = _CUR_DIR / "data" / data_setting / "norm_stats_delta.json"
                if candidate.is_file():
                    stats_path = candidate.resolve()
                    break
        if model_id is None or stats_path is None:
            raise ValueError("base_model_path/model_id and stats_path are required when load_model=true.")

        dtype_name = str(self.model_cfg.get("dtype") or "bf16").lower()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        device = str(self.model_cfg.get("device") or "cuda")
        seed = int(self.model_cfg.get("sample_seed") or self.model_cfg.get("seed") or 42)

        model_dict = inference_server.build_model(
            pretrained_path=str(model_id),
            checkpoint_path=str(checkpoint_path),
            action_dim=self.model_action_dim,
            state_dim=self.model_state_dim,
            flow_shift=float(self.model_cfg.get("flow_shift") or 5.0),
            device=device,
            dtype=dtype,
        )
        if device.startswith("cuda"):
            model_dict["rng"] = torch.Generator(device=model_dict["device"]).manual_seed(seed)
        else:
            model_dict["rng"] = torch.Generator().manual_seed(seed)

        norm_stats = inference_server.load_norm_stats(str(stats_path), self.model_state_dim, self.model_action_dim, device)
        t5_embedding = self._load_t5_embedding(inference_server, dtype=torch.float32)
        tokenizer, text_encoder = self._load_prompt_encoder(str(model_id), torch)

        return inference_server.GWPPolicy(
            model_dict=model_dict,
            norm_stats=norm_stats,
            t5_embedding=t5_embedding,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt_max_length=int(self.model_cfg.get("prompt_max_length") or 512),
            prompt_cache_size=int(self.model_cfg.get("prompt_cache_size") or 256),
            dst_size=tuple(self.model_cfg.get("dst_size") or [320, 256]),
            num_steps=int(self.model_cfg.get("num_steps") or self.model_cfg.get("num_inference_steps") or 10),
            num_frames=self.num_frames,
            action_chunk=self.action_chunk,
            action_only=_parse_bool(self.model_cfg.get("action_only"), True),
            zero_action_dims=_parse_int_list(self.model_cfg.get("zero_action_dims")),
            ctrl_mode_dim=(None if self.model_cfg.get("ctrl_mode_dim") is None else int(self.model_cfg.get("ctrl_mode_dim"))),
            ctrl_mode_threshold=float(self.model_cfg.get("ctrl_mode_threshold") or 0.0),
            skip_action_denorm=_parse_bool(self.model_cfg.get("skip_action_denorm"), False),
            tshape=_parse_bool(self.model_cfg.get("tshape"), True),
            tshape_head_index=(
                2
                if self.model_cfg.get("tshape_head_index") in (None, "")
                else int(self.model_cfg.get("tshape_head_index"))
            ),
        )

    def _load_t5_embedding(self, inference_server: Any, dtype: Any):
        import torch

        path = _as_path(self.model_cfg.get("t5_embedding_path") or self.model_cfg.get("t5_embedding_pkl"))
        if path is not None and path.is_file():
            tensor = torch.load(path, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            tensor = tensor[:64]
            if tensor.shape[0] < 64:
                tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 64 - tensor.shape[0]))
            return tensor.to(dtype=dtype)
        print("[GigaWorldPolicy] no t5_embedding_path provided; using zeros unless dynamic prompt is enabled")
        return torch.zeros(64, 4096, dtype=dtype)

    def _load_prompt_encoder(self, model_id: str, torch_module: Any):
        if _parse_bool(self.model_cfg.get("disable_dynamic_prompt"), False):
            return None, None
        try:
            from transformers import AutoTokenizer, UMT5EncoderModel

            tok_path = os.path.join(model_id, "tokenizer")
            te_path = os.path.join(model_id, "text_encoder")
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            text_encoder = UMT5EncoderModel.from_pretrained(te_path, torch_dtype=torch_module.float16).to(
                self.model_cfg.get("device") or "cuda"
            )
            text_encoder.eval()
            print(f"[GigaWorldPolicy] dynamic prompt encoder enabled: {te_path}")
            return tokenizer, text_encoder
        except Exception as exc:
            print(f"[GigaWorldPolicy] dynamic prompt encoder unavailable, using static embedding: {exc}")
            return None, None

    def _extract_image(self, obs: dict[str, Any], candidates: list[str]) -> np.ndarray:
        vision = obs.get("vision", {})
        for name in candidates:
            if name in vision:
                item = vision[name]
                if isinstance(item, dict):
                    for key in ("color", "rgb", "image"):
                        if key in item:
                            return _image_to_uint8_hwc(item[key])
                return _image_to_uint8_hwc(item)
            if name in obs:
                return _image_to_uint8_hwc(obs[name])
        raise KeyError(f"Could not find image for candidates: {candidates}")

    def _extract_prompt(self, obs: dict[str, Any]) -> str:
        value = obs.get("instruction", obs.get("instructions", None))
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return str(value or self.task_name or "")

    def _encode_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        state = pack_robot_state(obs, self.action_type, self.robot_action_dim_info, source_type="obs").astype(np.float32)
        state = _pad_or_trim_np(state, self.model_state_dim)
        return {
            "observation/image": self._extract_image(obs, list(self.view_candidates["left"])),
            "observation/wrist_image": self._extract_image(obs, list(self.view_candidates["wrist"])),
            "observation/right_image": self._extract_image(obs, list(self.view_candidates["right"])),
            "observation/state": state,
            "prompt": self._extract_prompt(obs),
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        env_ids = [int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)]
        validate_observed_ids(env_ids)
        self._latest_observations = {
            env_idx: self._encode_observation(obs) for env_idx, obs in zip(env_ids, obs_list)
        }
        self._latest_env_idx_list = env_ids

    def get_action(self, env_idx_list=None):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        elif isinstance(env_idx_list, np.ndarray):
            env_idx_list = env_idx_list.reshape(-1).tolist()
        elif isinstance(env_idx_list, (int, np.integer)):
            env_idx_list = [int(env_idx_list)]
        else:
            env_idx_list = list(env_idx_list)
        ids = selected_env_ids(self._latest_env_idx_list, [int(i) for i in env_idx_list])
        observations = [self._latest_observations[i] for i in ids]
        return [self._predict_action_sequence(obs) for obs in observations]

    def reset(self):
        self._latest_env_idx_list = [0]
        self._latest_observations = {}

    def _predict_action_sequence(self, encoded_obs: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        if self.policy is None:
            packed_actions = np.zeros((self.action_chunk, self.model_action_dim), dtype=np.float32)
        else:
            result = self.policy.infer(encoded_obs)
            packed_actions = np.asarray(result["actions"], dtype=np.float32)
            if packed_actions.ndim == 1:
                packed_actions = packed_actions[None, :]

        current_state = _pad_or_trim_np(
            np.asarray(encoded_obs["observation/state"], dtype=np.float32),
            self.model_action_dim,
        )
        if self.delta_to_absolute:
            packed_actions = self._delta_to_absolute_actions(packed_actions, current_state)
        packed_actions = packed_actions[: self.execute_action_chunk]

        self.action_request_count += 1
        print(
            "[GigaWorldPolicy] action request",
            f"#{self.action_request_count}",
            f"delta_to_absolute={self.delta_to_absolute}",
            f"state_first6={np.array2string(current_state.reshape(-1)[:6], precision=4, separator=',')}",
            f"packed0_first6={np.array2string(packed_actions[0, :6], precision=4, separator=',')}",
            flush=True,
        )

        action_list = []
        for packed_action in packed_actions:
            packed_action = _pad_or_trim_np(packed_action, self.xpolicylab_action_dim)
            if self.action_type == "ee":
                action_list.append(_manual_unpack_ee(packed_action, self.robot_action_dim_info))
            else:
                action_list.append(
                    unpack_robot_state(
                        packed_action,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )
        return action_list

    def _delta_to_absolute_actions(self, packed_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        actions = np.asarray(packed_actions, dtype=np.float32).copy()
        state = np.asarray(current_state, dtype=np.float32).reshape(-1, self.model_action_dim)[0]
        d = min(actions.shape[-1], state.shape[-1], len(_ARX_X5_DELTA_MASK))
        if d <= 0:
            return actions
        mask = _ARX_X5_DELTA_MASK[:d]
        prefix = actions[..., :d]
        prefix[..., mask] = prefix[..., mask] + state[:d][mask]
        actions[..., :d] = prefix
        return actions
