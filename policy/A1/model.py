import copy
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
# Allow pointing at an external A1 repo (e.g. the current A1_cvpr working tree) via the
# A1_REPO_DIR env var, so the server runs the exact inference code the checkpoint was
# trained/served with. Defaults to the embedded copy at policy/A1/A1.
_A1_DIR = Path(os.environ.get("A1_REPO_DIR", str(_SCRIPT_DIR / "A1")))
if str(_A1_DIR) not in sys.path:
    sys.path.insert(0, str(_A1_DIR))

_INFER_VLA_PATH = _A1_DIR / "deploy" / "infer_vla.py"
_INFER_VLA_SPEC = importlib.util.spec_from_file_location("a1_deploy_infer_vla", _INFER_VLA_PATH)
if _INFER_VLA_SPEC is None or _INFER_VLA_SPEC.loader is None:
    raise ImportError(f"Unable to load A1 infer_vla from {_INFER_VLA_PATH}")
_INFER_VLA_MODULE = importlib.util.module_from_spec(_INFER_VLA_SPEC)
_INFER_VLA_SPEC.loader.exec_module(_INFER_VLA_MODULE)
run_inference = _INFER_VLA_MODULE.run_inference
from a1.config import TrainConfig  # noqa: E402
from a1.data.vla.utils import NormalizationType  # noqa: E402
from a1.torch_util import get_local_rank, seed_all  # noqa: E402
from a1.util import resource_path  # noqa: E402
from a1.vla.affordvla import AffordVLA  # noqa: E402

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


DEFAULT_MODEL_PATH = str((_SCRIPT_DIR / "../../../../models/a1-pretrain").resolve())
DEFAULT_CAMERA_GROUPS = (
    ("cam_head", "cam_high"),
    ("cam_left_wrist", "cam_hand_left"),
    ("cam_right_wrist", "cam_hand_right"),
)


def _quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (np.pi / 2.0), np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def _pose7_to_pose6(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] != 7:
        return pose
    return np.concatenate([pose[..., :3], _quat_wxyz_to_rpy(pose[..., 3:7])], axis=-1).astype(np.float32)


def _rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    rpy = np.asarray(rpy, dtype=np.float64)
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.stack([w, x, y, z], axis=-1).astype(np.float32)


def _pose6_to_pose7(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] != 6:
        return pose
    return np.concatenate([pose[..., :3], _rpy_to_quat_wxyz(pose[..., 3:6])], axis=-1).astype(np.float32)


def _prepare_ee_obs_schema(obs: dict) -> dict:
    obs = copy.deepcopy(obs)
    state = obs.get("state", {})
    for key in ("ee_pose", "left_ee_pose", "right_ee_pose"):
        if key in state:
            state[key] = _pose7_to_pose6(state[key])
    return obs


def _prepare_ee_action_schema(action: dict) -> dict:
    action = copy.deepcopy(action)
    for key in ("ee_pose", "left_ee_pose", "right_ee_pose"):
        if key in action:
            action[key] = _pose6_to_pose7(action[key])
    return action


def _make_bool_mask(*dims: int) -> np.ndarray:
    """Same semantics as a1.data.vla.maniparena_datasets.make_bool_mask.

    make_bool_mask(6, -1, 6, -1) -> [T*6, F, T*6, F]
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * dim)
        else:
            result.extend([False] * (-dim))
    return np.asarray(result, dtype=bool)


def _parse_delta_mask(delta_mask):
    """Accept either a list/tuple of ints or a comma-separated string '6,-1,6,-1'."""
    if delta_mask is None:
        return None
    if isinstance(delta_mask, str):
        delta_mask = [int(x) for x in delta_mask.replace(" ", "").split(",") if x != ""]
    return _make_bool_mask(*[int(x) for x in delta_mask])


def _parse_image_resize(image_resize):
    """None / [] / '' -> no resize (keep original resolution). [w,h] or 'w,h' -> (w,h)."""
    if image_resize is None:
        return None
    if isinstance(image_resize, str):
        image_resize = image_resize.strip()
        if not image_resize or image_resize.lower() in ("none", "null"):
            return None
        image_resize = [int(x) for x in image_resize.replace(" ", "").split(",") if x != ""]
    if not image_resize:
        return None
    if len(image_resize) != 2:
        raise ValueError(f"image_resize must be [w, h], got {image_resize}")
    return (int(image_resize[0]), int(image_resize[1]))


def _apply_delta_postprocess(actions: np.ndarray, state: np.ndarray, delta_mask: np.ndarray | None) -> np.ndarray:
    """Convert model-predicted delta-action to absolute action by adding the raw state back.

    Mirrors A1 deploy/infer_vla.py:_apply_delta_postprocess so that the XPolicyLab
    server reproduces exactly what the A1 HTTP api_server does when launched with
    --delta --delta_mask. `state` is the RAW (un-normalized) proprio vector.
    """
    out = np.array(actions, dtype=np.float32)  # copy: predicted_actions may be a read-only inference-mode array
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if delta_mask is None:
        dims = min(out.shape[-1], state.shape[-1])
        out[..., :dims] = out[..., :dims] + state[:dims]
        return out
    dims = min(out.shape[-1], state.shape[-1], delta_mask.shape[-1])
    state_form = np.where(delta_mask[:dims], state[:dims], 0.0).astype(np.float32)
    out[..., :dims] = out[..., :dims] + state_form
    return out


def _find_latest_unsharded(run_dir: str | os.PathLike | None) -> str | None:
    if not run_dir or not os.path.isdir(run_dir):
        return None
    latest = os.path.join(run_dir, "latest-unsharded")
    if os.path.isdir(latest) and os.path.isfile(os.path.join(latest, "model.pt")):
        return latest
    candidates = []
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if name.endswith("-unsharded") and os.path.isdir(path) and os.path.isfile(os.path.join(path, "model.pt")):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _find_config_path(checkpoint_path: str | os.PathLike) -> Path:
    checkpoint_path = Path(checkpoint_path)
    candidates = [checkpoint_path / "config.yaml", checkpoint_path.parent / "config.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"config.yaml not found in {checkpoint_path} or {checkpoint_path.parent}")


def _load_json(path: str | os.PathLike):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_a1_model(checkpoint_path: str, seed: int):
    config = TrainConfig.load(_find_config_path(checkpoint_path), validate_paths=False)
    model_cfg = config.model

    if model_cfg.vit_load_path:
        model_cfg.vit_load_path = os.path.join(
            os.environ.get("DATA_DIR", ""),
            "pretrained_image_encoders",
            os.path.basename(model_cfg.vit_load_path),
        )
    if model_cfg.llm_load_path:
        model_cfg.llm_load_path = os.path.join(
            os.environ.get("DATA_DIR", ""),
            "pretrained_llms",
            os.path.basename(model_cfg.llm_load_path),
        )
    model_cfg.tokenizer.tokenizer_dir = os.environ.get("HF_HOME", "")

    if not torch.cuda.is_available():
        raise RuntimeError("A1 inference requires CUDA in the current implementation.")
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")

    model = AffordVLA(model_cfg)
    model_state_dict_path = resource_path(checkpoint_path, "model.pt")
    if not os.path.exists(model_state_dict_path):
        raise FileNotFoundError(f"A1 model.pt not found at {model_state_dict_path}")
    model_state_dict = torch.load(model_state_dict_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[A1 Model] Loaded finetuned checkpoint with strict=False | "
            f"missing={len(missing)} {missing[:5]} | unexpected={len(unexpected)} {unexpected[:5]}"
        )
    del model_state_dict
    model = model.to(device)
    model.eval()
    seed_all(seed)
    return model, model_cfg


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.task_name = model_cfg.get("task_name", "")
        self.default_prompt = model_cfg.get("prompt") or self.task_name or "Do your job."

        # Prefer an explicit robot_action_dim_info from the config (so the server can
        # run without the env_cfg/ infra); otherwise resolve it from env_cfg_type.
        radi = model_cfg.get("robot_action_dim_info")
        if radi:
            self.robot_action_dim_info = {
                "arm_dim": [int(x) for x in radi["arm_dim"]],
                "ee_dim": [int(x) for x in radi["ee_dim"]],
            }
        else:
            self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        assert len(self.robot_action_dim_info["arm_dim"]) == len(self.robot_action_dim_info["ee_dim"]), (
            "Arm and EE action dimensions must match"
        )

        self.action_dim = int(model_cfg.get("action_dim") or (
            sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        ))
        self.action_chunk_size = int(model_cfg.get("action_chunk_size", 50))
        self.sequence_length = int(model_cfg.get("sequence_length", 600))
        self.no_norm = bool(model_cfg.get("no_norm", False))
        self.normalization_type = NormalizationType(model_cfg.get("normalization_type", "bounds"))
        self.use_wrist_image = bool(model_cfg.get("use_wrist_image", True))
        self.seed = int(model_cfg.get("seed") or 6198)

        # Image resize before feeding the model. None/empty -> keep the caller's original
        # resolution (matches A1 training/HTTP api_server, which hand the full-res image to
        # the model's own preprocessor). Set to [w, h] to force a fixed size.
        self.image_resize = _parse_image_resize(model_cfg.get("image_resize"))

        # Delta-action post-processing. The embedded A1 run_inference does NOT support
        # delta restoration, so we replicate it here using the raw buffered proprio.
        self.delta = bool(model_cfg.get("delta", False))
        self.delta_mask = _parse_delta_mask(model_cfg.get("delta_mask")) if self.delta else None

        # Optional request logging: dump each caller's images + raw state + output action
        # to disk (mirrors A1 deploy/api_server.py:save_request_log) so the owner can verify
        # what external clients send. Only active when request_log_dir is set.
        self.request_log_dir = model_cfg.get("request_log_dir") or None
        self._request_counter = 0
        if self.request_log_dir:
            os.makedirs(self.request_log_dir, exist_ok=True)
            print(f"[A1 Model] request logging ON -> {self.request_log_dir}")

        self.model_path = self._resolve_model_path(model_cfg)
        self.norm_stats = self._load_norm_stats(model_cfg)
        self.model, self.a1_model_cfg = _load_a1_model(self.model_path, self.seed)

        self._obs_buffer_batch = {}
        self._latest_env_idx_list = [0]

        print(
            f"[A1 Model] Initialized | model_path={self.model_path} | action_type={self.action_type} | "
            f"action_dim={self.action_dim} | action_chunk_size={self.action_chunk_size} | "
            f"norm={self.normalization_type.value} | no_norm={self.no_norm} | "
            f"delta={self.delta} | delta_mask={None if self.delta_mask is None else self.delta_mask.astype(int).tolist()}"
        )

    def _resolve_model_path(self, model_cfg):
        model_path = model_cfg.get("model_path") or None
        if model_path:
            latest = _find_latest_unsharded(model_path)
            return latest or model_path

        dataset_name = model_cfg.get("dataset_name", "")
        ckpt_name = model_cfg.get("ckpt_name") or model_cfg.get("task_name", "")
        env_cfg_type = model_cfg.get("env_cfg_type", "")
        expert_data_num = model_cfg.get("expert_data_num", "")
        seed = model_cfg.get("seed", "")
        run_base = f"{dataset_name}-{ckpt_name}-{env_cfg_type}-{expert_data_num}-{self.action_type}-{seed}"
        latest_file = _SCRIPT_DIR / "checkpoints" / f"{run_base}.latest"
        if latest_file.is_file():
            latest = _find_latest_unsharded(latest_file.read_text().strip())
            if latest:
                return latest

        checkpoints_dir = _SCRIPT_DIR / "checkpoints"
        if checkpoints_dir.is_dir():
            matches = sorted(checkpoints_dir.glob(f"{run_base}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for match in matches:
                latest = _find_latest_unsharded(match)
                if latest:
                    return latest

            seed_agnostic_base = (
                f"{dataset_name}-{ckpt_name}-{env_cfg_type}-{expert_data_num}-{self.action_type}"
            )
            matches = sorted(checkpoints_dir.glob(f"{seed_agnostic_base}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for match in matches:
                latest = _find_latest_unsharded(match)
                if latest:
                    return latest

        return DEFAULT_MODEL_PATH

    def _load_norm_stats(self, model_cfg):
        if self.no_norm:
            return None
        stats_path = model_cfg.get("norm_stats_json_path") or model_cfg.get("data_stats_path")
        if not stats_path:
            for candidate in (
                Path(self.model_path) / "dataset_stats.json",
                Path(self.model_path) / "dataset_statistics.json",
                Path(self.model_path).parent / "dataset_stats.json",
                Path(self.model_path).parent / "dataset_statistics.json",
            ):
                if candidate.is_file():
                    stats_path = str(candidate)
                    break
        if not stats_path:
            raise ValueError("A1 normalization is enabled, but no norm stats json path was provided.")
        return _load_json(stats_path)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", i) for i, obs in enumerate(obs_list)]
        for obs in obs_list:
            env_idx = obs.get("env_idx", 0)
            self._obs_buffer_batch[env_idx] = _encode_obs(
                obs,
                self.action_type,
                self.robot_action_dim_info,
                self.default_prompt,
                self.image_resize,
            )

    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list

        action_batch = []
        for env_idx in env_idx_list:
            if env_idx not in self._obs_buffer_batch:
                raise RuntimeError(f"No observation buffered for env_idx={env_idx}. Call update_obs first.")

            input_data = self._obs_buffer_batch[env_idx]
            _t0 = time.monotonic()
            with torch.inference_mode():
                results = run_inference(
                    self.model,
                    input_data,
                    self.sequence_length,
                    self.norm_stats,
                    self.normalization_type,
                    use_proprio=True,
                    use_wrist_image=self.use_wrist_image,
                    no_norm=self.no_norm,
                )
            _proc_ms = (time.monotonic() - _t0) * 1000.0

            actions = np.asarray(results["predicted_actions"], dtype=np.float32).squeeze()
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            # Restore absolute action from predicted delta (run_inference returns
            # un-normalized but still delta-relative actions; add the raw state back).
            if self.delta:
                raw_state = np.asarray(input_data["proprio"], dtype=np.float32).reshape(-1)
                actions = _apply_delta_postprocess(actions, raw_state, self.delta_mask)

            actions = actions[: self.action_chunk_size, : self.action_dim]

            # Log this caller's input (images + raw state) and output (absolute action).
            self._save_request_log(input_data, actions, env_idx, _proc_ms)

            action_steps = []
            for step_action in actions:
                action = unpack_robot_state(
                    step_action,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
                if self.action_type == "ee":
                    action = _prepare_ee_action_schema(action)
                action_steps.append(action)
            action_batch.append(action_steps)

        return action_batch

    def _save_request_log(self, input_data, actions, env_idx, processing_time_ms):
        """Dump one caller's input (images + raw state) and output (absolute action).

        Mirrors A1 deploy/api_server.py:save_request_log. The images saved here are
        exactly what the model received (original resolution unless image_resize is set),
        and the state is the raw, un-normalized proprio packed by pack_robot_state.
        Only active when request_log_dir is set.
        """
        if not self.request_log_dir:
            return
        try:
            self._request_counter += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_dir = os.path.join(self.request_log_dir, f"{ts}_{self._request_counter:06d}_env{env_idx}")
            os.makedirs(out_dir, exist_ok=True)

            # 1) images actually fed to the model (HWC uint8 RGB)
            images = input_data.get("images") or []
            image_shapes = []
            for i, img in enumerate(images):
                arr = np.asarray(img)
                image_shapes.append(list(arr.shape))
                try:
                    Image.fromarray(arr.astype(np.uint8)).save(os.path.join(out_dir, f"cam{i}.png"))
                except Exception as e:  # noqa: BLE001
                    print(f"[request-log] save image {i} failed: {e}", flush=True)

            # 2) raw state + output action + config -> meta.json
            raw_state = np.asarray(input_data.get("proprio"), dtype=np.float32).reshape(-1)
            acts = np.asarray(actions, dtype=np.float32)
            meta = {
                "timestamp": ts,
                "env_idx": int(env_idx),
                "instruction": input_data.get("instruction"),
                "num_images": len(images),
                "image_shapes": image_shapes,
                "proprio_state_raw": raw_state.tolist(),  # un-normalized state from the caller
                "proprio_dim": int(raw_state.shape[-1]),
                "predicted_actions": acts.tolist(),        # absolute action (delta restored)
                "predicted_actions_shape": list(acts.shape),
                "processing_time_ms": round(processing_time_ms, 2),
                "action_type": self.action_type,
                "normalization_type": self.normalization_type.value,
                "no_norm": self.no_norm,
                "delta": self.delta,
                "delta_mask": None if self.delta_mask is None else self.delta_mask.astype(int).tolist(),
            }
            with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            np.save(os.path.join(out_dir, "predicted_actions.npy"), acts)
            print(
                f"[request-log] saved -> {out_dir} "
                f"(images={len(images)}, state_dim={meta['proprio_dim']}, {processing_time_ms:.0f}ms)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[request-log] failed: {e}", flush=True)

    def reset(self):
        self._obs_buffer_batch = {}
        self._latest_env_idx_list = [0]
        print("[A1 Model] Reset", flush=True)


def _encode_obs(observation, action_type, robot_action_dim_info, default_prompt, image_resize=None):
    if action_type == "ee":
        observation = _prepare_ee_obs_schema(observation)

    vision = observation.get("vision", {})
    images = []
    for camera_names in DEFAULT_CAMERA_GROUPS:
        img = _extract_rgb_image(vision, camera_names, image_resize)
        if img is not None:
            images.append(img)

    if not images:
        raise ValueError("A1 requires at least one RGB camera image in observation['vision'].")

    instruction = observation.get("instruction", observation.get("instructions", default_prompt))
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else default_prompt

    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs")
    return {
        "images": images,
        "instruction": str(instruction),
        "proprio": torch.tensor(state.reshape(1, -1), dtype=torch.float32),
    }


def _extract_rgb_image(vision, camera_names, image_resize=None):
    if isinstance(camera_names, str):
        camera_names = (camera_names,)
    camera_data = None
    for camera_name in camera_names:
        camera_data = vision.get(camera_name)
        if camera_data is not None:
            break
    if camera_data is None:
        return None
    if isinstance(camera_data, dict):
        img = camera_data.get("color", camera_data.get("rgb"))
    else:
        img = camera_data
    if img is None:
        return None

    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim != 3 or img.shape[-1] != 3:
        return None

    # By default keep the caller's original resolution (matches A1 training / HTTP server,
    # which let the model's own preprocessor crop/resize). Only resize when explicitly asked.
    if image_resize is not None:
        img = cv2.resize(img, (int(image_resize[0]), int(image_resize[1])), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)
