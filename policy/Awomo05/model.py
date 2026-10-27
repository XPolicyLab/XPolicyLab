# Copyright (c) 2026 Awomo-WAM Team. Apache-2.0.
"""XPolicy observation/action adapter for the public prepared checkpoint format."""

from __future__ import annotations

from collections import deque
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state
from .runtime import ACTIVE, InferenceRuntime, bundled_module, validate_contract

LOG = logging.getLogger(__name__)

CAMERA_ALIASES = (
    ("cam_head", "cam_high", "head_camera"),
    ("cam_left_wrist", "cam_hand_left", "left_camera"),
    ("cam_right_wrist", "cam_hand_right", "right_camera"),
)
RAW_FPS = 25.0
PLUGIN_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = os.environ.get("AWOMO05_WEIGHTS", str(PLUGIN_DIR / "weights"))


def _is_none_like(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _decode_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    elif isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tobytes()
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
    text = str(value).strip().strip("\x00").strip()
    return text or fallback


def _extract_rgb(vision: dict, aliases: tuple[str, ...]) -> np.ndarray:
    """HWC uint8 RGB images are already decoded by the policy server."""
    image = None
    for key in aliases:
        cam = vision.get(key)
        if cam is None:
            continue
        image = cam.get("color", cam.get("rgb", cam.get("colors"))) if isinstance(cam, dict) else cam
        if image is not None:
            break
    if image is None:
        raise KeyError(f"None of {aliases} present in obs['vision'] (keys: {sorted(vision)})")
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"camera {aliases[0]} gave an unusable array of shape {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image.astype(np.uint8))


class HistorySlotGrid:
    """Fixed negative-time slots ordered oldest-to-newest; mask unavailable history."""

    def __init__(self, offset_bins_sec, fps: float = RAW_FPS):
        bins = []
        for lo, hi in offset_bins_sec:
            lo_f = max(1, int(round(float(lo) * fps)))
            hi_f = max(lo_f, int(round(float(hi) * fps)))
            bins.append((lo_f, hi_f))
        order = sorted(range(len(bins)), key=lambda i: bins[i][0], reverse=True)
        self.bins_frames = [bins[i] for i in order]
        self.num_slots = len(self.bins_frames)
        self.max_lookback = max(hi for _, hi in self.bins_frames)

    def offsets(self, current_t: int) -> list[Optional[int]]:
        """-> per-slot frame distance back from `current_t`, None for an invalid slot."""
        out: list[Optional[int]] = []
        for lo_f, hi_f in self.bins_frames:
            hi_eff = min(hi_f, current_t)
            if hi_eff < lo_f:
                out.append(None)
                continue
            out.append(min((lo_f + hi_f) // 2, hi_eff))
        return out


class _EnvState:
    """Per-environment rollout state."""

    def __init__(self, history_capacity: int, vl_history_capacity: int):
        self.head_frames: deque[torch.Tensor] = deque(maxlen=history_capacity)
        self.obs: Optional[dict] = None
        self.views: Optional[torch.Tensor] = None
        self.vl_pil: Optional[Image.Image] = None
        self.vl_hist_frames: deque[Image.Image] = deque(maxlen=vl_history_capacity)
        self.pending: deque[np.ndarray] = deque()
        self.step = -1


def checkpoint_root(cfg):
    roots = candidate_checkpoint_roots(dict(cfg), PLUGIN_DIR / "checkpoints", policy_dir=PLUGIN_DIR)
    weights = Path(cfg.get("weights_dir") or os.environ.get("AWOMO05_WEIGHTS") or DEFAULT_WEIGHTS_DIR).expanduser()
    explicit = cfg.get("ckpt")
    if not _is_none_like(explicit):
        direct = Path(str(explicit)).expanduser()
        roots = [direct, weights / str(explicit), weights / (str(explicit) + ".pt")]
    elif not _is_none_like(cfg.get("ckpt_name")):
        roots += [weights / str(cfg["ckpt_name"]), weights / (str(cfg["ckpt_name"]) + ".pt")]
    if _is_none_like(explicit) and _is_none_like(cfg.get("ckpt_name")):
        roots += [weights]
    for root in roots:
        directory = root.parent if root.is_file() else root
        if (directory / "awomo_config.json").is_file():
            if root.is_file() and root.name != "model.pt":
                raise ValueError("A prepared checkpoint file must be the bundle's model.pt")
            return directory.resolve()
    raise FileNotFoundError("Checkpoint requires model.pt, awomo_config.json, dataset_stats.json and SHA256SUMS")


def stage_bundle(source, local_dir, enabled=True):
    """Cache an entire bundle by content, verifying every byte rather than only file size."""
    source = Path(source)
    manifest = (source / "SHA256SUMS").read_bytes()
    files = {}
    for line in manifest.decode().splitlines():
        digest, name = line.split(maxsplit=1)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid checkpoint checksum")
        if name != Path(name).name or name in files:
            raise ValueError("Unsafe or duplicate checkpoint member")
        files[name] = digest
    if set(files) != {"model.pt", "awomo_config.json", "dataset_stats.json"}:
        raise ValueError("Expected the three runtime files in SHA256SUMS")
    if not enabled:
        return source
    cache = Path(local_dir) / "prepared"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(manifest).hexdigest()
    target = cache / key
    with (cache / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return target
        temporary = Path(tempfile.mkdtemp(prefix=key + ".", dir=cache))
        try:
            for name, expected in files.items():
                if (source / name).is_symlink():
                    raise ValueError("Indirect checkpoint files are not supported")
                digest = hashlib.sha256()
                with (source / name).open("rb") as reader, (temporary / name).open("xb") as writer:
                    for block in iter(lambda: reader.read(8 << 20), b""):
                        digest.update(block)
                        writer.write(block)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Checkpoint checksum mismatch: {name}")
            (temporary / "SHA256SUMS").write_bytes(manifest)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target


class Model(ModelTemplate):
    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def _build_history(self, state: _EnvState):
        if self.history_grid is None:
            return None, None
        n = self.history_grid.num_slots
        w, h = self.view_size_wh
        frames = torch.zeros(n, 3, h, w)
        valid = torch.zeros(n, dtype=torch.bool)
        buf = state.head_frames
        for slot, d in enumerate(self.history_grid.offsets(state.step)):
            if d is None or d >= len(buf):
                continue
            frames[slot] = buf[len(buf) - 1 - d]
            valid[slot] = True
        frames = frames * 2.0 - 1.0
        return frames.unsqueeze(0), valid.unsqueeze(0)

    def _build_vl_history(self, state: _EnvState) -> list:
        if not self.vl_history_offsets:
            return []
        buf = state.vl_hist_frames
        out = []
        for d in self.vl_history_offsets:
            if d > state.step or d >= len(buf):
                continue
            out.append(buf[len(buf) - 1 - d])
        return out

    def __init__(self, model_cfg):
        if str(model_cfg.get("checkpoint_format") or "prepared") != "prepared":
            raise ValueError("Only the published three-entry checkpoint format is supported")
        self.model_cfg = dict(model_cfg)
        if model_cfg.get("action_type") != "joint":
            raise ValueError("Prepared Awomo supports absolute joint control only")
        self.action_type = "joint"
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.robot_info = get_robot_action_dim_info(self.env_cfg_type)
        self.expected_action_dim = sum(self.robot_info["arm_dim"]) + sum(self.robot_info["ee_dim"])
        if self.expected_action_dim != 14 or self.env_cfg_type != "arx_x5":
            raise ValueError("This checkpoint's joint mapping supports arx_x5 only")
        configured_dim = model_cfg.get("action_dim")
        if configured_dim is not None and int(configured_dim) != self.expected_action_dim:
            raise ValueError("Configured robot action dimension does not match robot metadata")
        self.ckpt_path = checkpoint_root(model_cfg)
        self.local_dir = Path(model_cfg.get("local_dir") or "/tmp/Awomo05")
        self.ckpt_load_path = stage_bundle(self.ckpt_path, self.local_dir,
                                           _as_bool(model_cfg.get("ckpt_local_copy"), True))
        loader = bundled_module("factory")
        config = loader.read_checkpoint_config(self.ckpt_load_path, verify_checksums=False)
        validate_contract(config)
        self.input_color_order = str(model_cfg.get("input_color_order") or "rgb").lower()
        if self.input_color_order != config["image_color_order"]:
            raise ValueError("Set input_color_order to the checkpoint's model color order; observations stay RGB")
        self.action_horizon = int(config["action_horizon"])
        self.replan_steps = int(model_cfg.get("replan_steps") or 8)
        if not 0 < self.replan_steps <= self.action_horizon:
            raise ValueError("replan_steps must lie in [1, action_horizon]")
        if model_cfg.get("action_horizon") is not None and int(model_cfg["action_horizon"]) != self.action_horizon:
            raise ValueError("Do not override the checkpoint action horizon")
        self.num_inference_steps = int(model_cfg.get("num_inference_steps") or config["scheduler"]["num_inference_steps"])
        self.sigma_shift = float(config["scheduler"]["shift"])
        if model_cfg.get("sigma_shift") is not None and float(model_cfg["sigma_shift"]) != self.sigma_shift:
            raise ValueError("Do not override the checkpoint shift")
        self.seed = int(model_cfg.get("seed") or 0)
        self.rand_device = str(model_cfg.get("rand_device") or "cpu")
        self.max_env_batch_size = int(model_cfg.get("max_env_batch_size") or 10)
        if self.max_env_batch_size <= 0 or self.num_inference_steps <= 0:
            raise ValueError("Inference steps and batch capacity must be positive")
        self.dit_serial = _as_bool(model_cfg.get("dit_serial"), True)
        dtype_name = str(model_cfg.get("mixed_precision") or "bf16")
        if dtype_name not in {"bf16", "fp32"}:
            raise ValueError("Prepared inference supports bf16 or fp32 DiT compute")
        qwen = model_cfg.get("qwen_vl_path") or os.environ.get("AWOMO05_QWEN_VL") or os.environ.get("AWOMO_QWEN3_VL")
        vae = model_cfg.get("vae_path") or os.environ.get("AWOMO05_FLUX2_VAE") or os.environ.get("AWOMO_FLUX2_VAE")
        if not qwen or not vae or not Path(qwen).is_dir() or not Path(vae).is_dir():
            raise FileNotFoundError("Prepared inference requires local Qwen and Diffusers VAE directories")
        self.model = InferenceRuntime(self.ckpt_load_path, qwen_path=qwen, vae_path=vae,
                                       device=str(model_cfg.get("device") or "cuda"),
                                       dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name])
        normalizer = bundled_module("datasets.lerobot.utils.normalizer")
        stats = normalizer.load_dataset_stats_from_json(str(self.ckpt_load_path / "dataset_stats.json"))
        if model_cfg.get("dataset_stats_path"):
            override = Path(str(model_cfg["dataset_stats_path"]))
            supplied = self.ckpt_load_path / "dataset_stats.json"
            if override.read_bytes() != supplied.read_bytes():
                raise ValueError("Normalization override differs from the checkpoint's bundled statistics")
        shape = {field: [{"key": "default", "raw_shape": 14, "shape": 14}] for field in ("action", "state")}
        self.normalizer = normalizer.LinearNormalizer(shape, False, "z-score", None, stats)
        self.history_grid = HistorySlotGrid([(k + .6, k + 1.4) for k in range(20)], fps=25.)
        self._history_capacity = self.history_grid.max_lookback + 1
        self.vl_history_offsets = self.model.vl_encoder.history_offsets()
        self._vl_history_capacity = max(self.vl_history_offsets) + 1
        self.view_size_wh = tuple(reversed(config["video_size"]))
        self.vl_image_size_wh = self.vl_history_size_wh = tuple(config["vl_image_size"])
        mosaic = bundled_module("model")
        self._compose_rgb_mosaic, self._resize_mosaic = mosaic.compose_robodojo_rgb_mosaic, mosaic.resize_mosaic_for_vae
        self.default_instruction = str(model_cfg.get("default_instruction") or "")
        self._envs, self._latest_env_idx_list = {}, []
        LOG.info("Prepared Awomo ready: source=%s color=%s horizon=%d replan=%d steps=%d encoding=batch DiT=%s",
                 self.ckpt_path, self.input_color_order, self.action_horizon, self.replan_steps,
                 self.num_inference_steps, "serial" if self.dit_serial else "batch")

    def reset(self, env_idx=None):
        if env_idx is None:
            self._envs.clear()
            self._latest_env_idx_list = []
        else:
            self._envs.pop(int(env_idx), None)
            self._latest_env_idx_list = [i for i in self._latest_env_idx_list if i != int(env_idx)]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        ids = [int(obs.get("env_idx", i)) for i, obs in enumerate(obs_list)]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate environment indices in observation batch")
        self._latest_env_idx_list = ids
        for idx, obs in zip(ids, obs_list):
            state = self._envs.setdefault(idx, _EnvState(self._history_capacity, self._vl_history_capacity))
            frames = [_extract_rgb(obs["vision"], aliases) for aliases in CAMERA_ALIASES]
            mosaic = self._compose_rgb_mosaic(frames, color_order=self.input_color_order)
            unit = self._resize_mosaic(mosaic, self.model.video_size)
            pil = Image.fromarray(mosaic)
            state.obs, state.views = obs, unit.unsqueeze(0)
            state.vl_pil = pil.resize(self.vl_image_size_wh, Image.BILINEAR)
            state.vl_hist_frames.append(pil.resize(self.vl_history_size_wh, Image.BILINEAR))
            state.head_frames.append(unit)
            state.step += 1

    def _proprio(self, obs):
        raw = np.asarray(pack_robot_state(obs, "joint", self.robot_info, source_type="obs"), dtype=np.float32)
        if raw.shape != (14,) or not np.isfinite(raw).all():
            raise ValueError("Expected fourteen finite robot state values")
        compact = self.normalizer.forward({"state": {"default": torch.from_numpy(raw).unsqueeze(0)}})["state"]["default"]
        compact = compact.clone()
        compact[:, [6, 13]] *= -1
        dense = compact.new_zeros(1, 80)
        dense[:, list(ACTIVE)] = compact
        return dense

    def _denormalize(self, action):
        compact = action[..., list(ACTIVE)].detach().to(device="cpu", dtype=torch.float32).clone()
        compact[..., [6, 13]] *= -1
        return self.normalizer.normalizers["action"]["default"].backward(compact).numpy()

    def _request(self, state):
        instruction = _decode_text(state.obs.get("instruction", state.obs.get("instructions", state.obs.get("task_instruction"))),
                                   self.default_instruction)
        if not instruction:
            raise ValueError("Observation has no instruction")
        history, valid = self._build_history(state)
        # Preserve the evaluated arithmetic order: current image casts before scaling;
        # historical frames scale in float32 before casting.
        views = state.views.unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)
        views = views * 2 - 1
        history = history.to(device=self.model.device, dtype=self.model.torch_dtype)
        valid = valid.to(self.model.device)
        return dict(prompt=instruction, input_image=views,
                    proprio=self._proprio(state.obs), history_frames=history, history_valid_mask=valid,
                    vl_images=[state.vl_pil], vl_history=[self._build_vl_history(state)],
                    action_horizon=self.action_horizon, num_inference_steps=self.num_inference_steps,
                    sigma_shift=self.sigma_shift, seed=self.seed, rand_device=self.rand_device)

    def get_action_batch(self, env_idx_list=None):
        indices = self._latest_env_idx_list if env_idx_list is None else [int(i) for i in env_idx_list]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate environment indices in action request")
        results = []
        for start in range(0, len(indices), self.max_env_batch_size):
            rows = indices[start:start + self.max_env_batch_size]
            if any(i not in self._envs or self._envs[i].obs is None for i in rows):
                raise RuntimeError("Call update_obs before requesting actions")
            requests = [self._request(self._envs[i]) for i in rows]
            outputs = self.model.infer_action_flux2_batch(requests, dit_serial=self.dit_serial)
            for output in outputs:
                actions = self._denormalize(output["action"])
                if actions.shape != (self.action_horizon, 14) or not np.isfinite(actions).all():
                    raise FloatingPointError("Invalid physical action output")
                results.append([unpack_robot_state(np.asarray(a, dtype=np.float32), "joint", self.robot_info,
                                                   source_type="obs") for a in actions[:self.replan_steps]])
        return results
