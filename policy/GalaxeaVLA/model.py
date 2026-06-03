"""XPolicyLab adapter for GalaxeaVLA (galaxea_fm).

Variant-agnostic: defaults to G0Plus_3B but the upstream model/processor are
fully driven by a Hydra task config (``task_config_name`` in ``deploy.yml``),
so G0Tiny / pi0 / pi0fast can be selected without code changes.

The inference path mirrors ``scripts/eval_libero.py``:
    model = instantiate(cfg.model.model_arch)
    model, dataset_stats = <load checkpoint>
    processor = instantiate(cfg.data.processor)
    processor.set_normalizer_from_stats(dataset_stats)
    sample = processor.preprocess({images, state, task, *_is_pad, idx})
    batch  = policy.predict_action(collate(sample))
    batch  = processor.postprocess(batch)   # -> per-key action dict
"""

import os
from typing import Any

import cv2
import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.join(CURRENT_DIR, "GalaxeaVLA")
CONFIG_DIR = os.path.join(UPSTREAM_DIR, "configs")

# XPolicyLab image standard (see references/xpolicylab-policy-contract.md).
STD_W, STD_H = 320, 240

CAM_NAME_CANDIDATES = {
    "head_rgb": ["cam_head", "head_camera", "cam_high", "head", "top_camera"],
    "head_condition": ["cam_head", "head_camera", "cam_high", "head"],
    "left_wrist_rgb": ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
    "right_wrist_rgb": ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
    "cam_high": ["cam_head", "head_camera", "cam_high", "head", "top_camera"],
    "cam_left_wrist": ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
    "cam_right_wrist": ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
    "image": ["cam_head", "head_camera", "cam_high", "image"],
    "wrist_image": ["cam_left_wrist", "left_camera", "wrist_image"],
}


def _normalize_wxyz_pose(pose7: np.ndarray) -> np.ndarray:
    """Unit-normalize quaternion in XPolicyLab wxyz pose (matches X-VLA quat_to_rotate6d)."""
    p = np.asarray(pose7, dtype=np.float32).reshape(-1)
    if p.shape[-1] != 7:
        raise ValueError(f"Expected 7-dim pose, got shape {p.shape}.")
    quat = p[3:7]
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        quat = (quat / norm).astype(np.float32)
    return np.concatenate([p[:3], quat], axis=-1).astype(np.float32)


def _xpolicylab_pose_to_upstream(pose7: np.ndarray) -> np.ndarray:
    """XPolicyLab [x,y,z,qw,qx,qy,qz] -> galaxea_fm [x,y,z,qx,qy,qz,qw]."""
    p = _normalize_wxyz_pose(pose7)
    return np.concatenate([p[:3], p[4:7], p[3:4]])


def _upstream_pose_to_xpolicylab(pose7: np.ndarray) -> np.ndarray:
    """galaxea_fm [x,y,z,qx,qy,qz,qw] -> XPolicyLab [x,y,z,qw,qx,qy,qz]."""
    p = np.asarray(pose7, dtype=np.float32).reshape(-1)
    return _normalize_wxyz_pose(np.concatenate([p[:3], p[6:7], p[3:6]], axis=-1))


# Galaxea ee shape_meta uses left_gripper; XPolicyLab sim obs uses left_ee_joint_state.
_EE_GRIPPER_TO_XPL = {
    "left_gripper": "left_ee_joint_state",
    "right_gripper": "right_ee_joint_state",
}


def _read_upstream_state_from_obs(observation: dict, state_shape_meta: list) -> dict:
    """Build upstream state tensors from XPolicyLab ee-mode obs (7-dim poses + grippers).

    Obs/action poses follow the same wxyz layout as X-VLA ``build_xvla_proprio`` /
    ``RoboDojoHandler`` (HDF5 ``left_ee_poses``); only the internal galaxea layout
    uses scalar-last xyzw before ``RelativePoseTransform``.
    """
    state_dict = observation.get("state", {})
    state = {}
    for meta in state_shape_meta:
        key = meta["key"]
        src_key = _EE_GRIPPER_TO_XPL.get(key, key)
        if src_key not in state_dict:
            raise KeyError(f"Missing obs['state']['{src_key}'] for upstream key '{key}'.")
        chunk = np.asarray(state_dict[src_key], dtype=np.float32).reshape(-1)
        n = int(meta["raw_shape"])
        if chunk.shape[-1] != n:
            raise ValueError(
                f"State field '{src_key}' last dim mismatch: expected {n}, got {chunk.shape[-1]}."
            )
        if "ee_pose" in key:
            chunk = _xpolicylab_pose_to_upstream(chunk)
        state[key] = torch.from_numpy(chunk).unsqueeze(0)
    return state


def _upstream_action_to_xpolicylab_steps(
    action_dict: dict, action_shape_meta: list, batch_index: int
) -> list[dict]:
    """Convert upstream per-key ee actions to XPolicyLab step dicts."""
    horizon = action_dict[action_shape_meta[0]["key"]][batch_index].shape[0]
    steps = []
    for t in range(horizon):
        step = {}
        for meta in action_shape_meta:
            key = meta["key"]
            val = action_dict[key][batch_index][t].astype(np.float32)
            if "ee_pose" in key:
                val = _upstream_pose_to_xpolicylab(val)
                out_key = key
            elif "gripper" in key:
                out_key = _EE_GRIPPER_TO_XPL[key]
                # X-VLA passes gripper scalar straight to left_ee_joint_state; clip to sim range.
                val = np.clip(val, 0.0, 1.0)
            else:
                out_key = key
            step[out_key] = val
        steps.append(step)
    return steps


def _resolve_ckpt_path(ckpt_path: str) -> str:
    """Resolve checkpoints/<6-tuple>[/timestamp] to a deployable step dir."""
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isfile(ckpt_path):
        return ckpt_path

    def _is_run_root(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "model.pt")) or os.path.isdir(
            os.path.join(path, "checkpoints")
        )

    if not _is_run_root(ckpt_path) and os.path.isdir(ckpt_path):
        run_dirs = sorted(
            (
                os.path.join(ckpt_path, name)
                for name in os.listdir(ckpt_path)
                if os.path.isdir(os.path.join(ckpt_path, name))
            ),
            key=os.path.getmtime,
        )
        for run_dir in reversed(run_dirs):
            if _is_run_root(run_dir):
                ckpt_path = run_dir
                break

    if os.path.isfile(os.path.join(ckpt_path, "model.pt")):
        return ckpt_path

    steps_root = os.path.join(ckpt_path, "checkpoints")
    if os.path.isdir(steps_root):
        step_dirs = sorted(
            (
                name
                for name in os.listdir(steps_root)
                if name.startswith("step_")
                and os.path.isdir(os.path.join(steps_root, name))
                and os.path.isfile(os.path.join(steps_root, name, "model.pt"))
            ),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        if step_dirs:
            return os.path.join(steps_root, step_dirs[-1])

    raise FileNotFoundError(
        f"No deployable checkpoint under {ckpt_path}. "
        "Expected checkpoints/<dataset>-<ckpt_name>-<env>-<num>-<action>-<seed>[/timestamp]/checkpoints/step_*/model.pt."
    )


def _extract_image(observation: dict, key: str):
    """Return an HWC uint8 image for the given upstream image key, or None."""
    vision = observation.get("vision", {})
    candidates = CAM_NAME_CANDIDATES.get(key, []) + [key]
    for name in candidates:
        if name not in vision:
            continue
        image = vision[name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    return None


def _standardize_rgb(image, width: int, height: int) -> np.ndarray:
    """Standardize to RGB HWC (height, width, 3) uint8.

    XPolicyLab obs images are already RGB-ordered, so do NOT insert BGR2RGB
    here (see the channel-order convention in the policy contract).

    The target size comes from the active Hydra shape_meta raw_shape so joint
    configs keep (240, 320) while ee configs match upstream (256, 256).
    """
    image = np.asarray(image)
    assert image.ndim == 3 and image.shape[-1] == 3, f"expected HWC RGB, got {image.shape}"
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    assert image.shape == (height, width, 3), f"got {image.shape}, expected {(height, width, 3)}"
    return image


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.action_type = model_cfg.get("action_type", "joint")
        self.env_cfg_type = model_cfg.get("env_cfg_type")
        self.default_prompt = model_cfg.get("prompt") or model_cfg.get("task_name") or "Do your job."
        self.task_config_name = model_cfg.get(
            "task_config_name", "real/g0plus_xpolicylab_finetune"
        )
        self.ckpt_path = _resolve_ckpt_path(model_cfg["ckpt_path"])

        self.robot_action_dim_info = (
            get_robot_action_dim_info(self.env_cfg_type) if self.env_cfg_type is not None else None
        )

        cfg = self._compose_config(model_cfg)
        self.image_shape_meta = list(cfg.data.dataset.shape_meta.images)
        self.state_shape_meta = list(cfg.data.dataset.shape_meta.state)
        self.action_shape_meta = list(cfg.data.dataset.shape_meta.action)

        raw_shape = self.image_shape_meta[0]["raw_shape"]
        self._img_h, self._img_w = int(raw_shape[1]), int(raw_shape[2])

        self.policy, self.processor = self._build(cfg)
        self.model = self.policy
        self._device = self.policy.device

        self._sample_batch: dict | None = None
        self._latest_env_idx_list: list[int] = [0]

    # ------------------------------------------------------------------ setup
    def _compose_config(self, model_cfg: dict[str, Any]):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from galaxea_fm.utils.config_resolvers import register_default_resolvers

        register_default_resolvers()
        # The upstream train.yaml interpolates a few env vars that are
        # irrelevant for inference; set harmless defaults so compose succeeds.
        os.environ.setdefault("GALAXEA_FM_OUTPUT_DIR", "/tmp/galaxea_fm_output")
        os.environ.setdefault("GALAXEA_FM_DATASET_STATS_CACHE_DIR", "/tmp/galaxea_fm_stats")

        overrides = [f"task={self.task_config_name}"]
        paligemma_path = model_cfg.get("paligemma_path")
        if paligemma_path:
            overrides += [
                f"model.model_arch.pretrained_model_path={paligemma_path}",
                f"model.tokenizer.tokenizer_params.pretrained_model_name_or_path={paligemma_path}",
            ]
        if model_cfg.get("num_inference_steps") is not None:
            overrides.append(
                f"model.model_arch.num_inference_steps={int(model_cfg['num_inference_steps'])}"
            )
        for extra in model_cfg.get("hydra_overrides", []) or []:
            overrides.append(str(extra))

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
            cfg = compose(config_name="train", overrides=overrides)
        return cfg

    def _build(self, cfg):
        from accelerate import PartialState
        from hydra.utils import instantiate
        from galaxea_fm.models.base_policy import BasePolicy
        from galaxea_fm.processors.base_processor import BaseProcessor

        # Upstream model code uses accelerate.logging; mirror eval_open_loop.py.
        PartialState()

        model: BasePolicy = instantiate(cfg.model.model_arch)
        model, dataset_stats = self._load_checkpoint(self.ckpt_path, model)

        dtype = torch.bfloat16 if bool(getattr(cfg.model, "enable_bf16_training", False)) else torch.float32
        policy = model.to(dtype=dtype).cuda().eval()

        processor: BaseProcessor = instantiate(cfg.data.processor)
        processor.set_normalizer_from_stats(dataset_stats)
        processor.eval()

        # Autoregressive variants (pi0fast) need the tokenizer for decoding.
        if hasattr(policy, "set_tokenizer") and hasattr(processor, "tokenizer"):
            policy.set_tokenizer(processor.tokenizer)

        return policy, processor

    def _load_checkpoint(self, ckpt_path: str, model):
        """Load weights + dataset_stats via the upstream eval helper."""
        from galaxea_fm.utils.load_pretrained_resumed import load_checkpoint_for_eval

        ckpt_path = _resolve_ckpt_path(ckpt_path)
        return load_checkpoint_for_eval(ckpt_path, model, device="cpu")

    # --------------------------------------------------------------- obs / act
    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", idx) for idx, obs in enumerate(obs_list)]
        samples = [self._encode_obs(obs) for obs in obs_list]
        self._sample_batch = self._collate(samples)

    def _encode_obs(self, observation) -> dict:
        # images: one (num_obs_steps=1, C, H, W) uint8 tensor per upstream key
        images = {}
        for meta in self.image_shape_meta:
            key = meta["key"]
            raw = _extract_image(observation, key)
            if raw is None:
                std = np.zeros((self._img_h, self._img_w, 3), dtype=np.uint8)
            else:
                std = _standardize_rgb(raw, self._img_w, self._img_h)
            chw = np.transpose(std, (2, 0, 1))  # (3, H, W)
            images[key] = torch.from_numpy(chw).unsqueeze(0).contiguous()  # (1, 3, H, W) uint8

        # state: ee mode reads obs keys directly (7-dim pose + gripper); joint mode
        # uses the canonical XPolicyLab pack/unpack helpers.
        if self.action_type == "ee":
            state = _read_upstream_state_from_obs(observation, self.state_shape_meta)
        else:
            packed = pack_robot_state(
                observation, self.action_type, self.robot_action_dim_info, source_type="obs"
            ).astype(np.float32)
            state = {}
            offset = 0
            for meta in self.state_shape_meta:
                n = int(meta["raw_shape"])
                chunk = packed[offset:offset + n]
                if "ee_pose" in meta["key"]:
                    chunk = _xpolicylab_pose_to_upstream(chunk)
                state[meta["key"]] = torch.from_numpy(chunk).unsqueeze(0)  # (1, n)
                offset += n

        instruction = observation.get("instruction", observation.get("instructions", ""))
        if isinstance(instruction, (list, tuple)):
            instruction = str(instruction[0]) if instruction else ""
        prompt = observation.get("prompt") or (str(instruction) if instruction else self.default_prompt)
        sample = {
            "images": images,
            "state": state,
            "task": str(prompt),
            "state_is_pad": torch.tensor([False]),
            "image_is_pad": torch.tensor([False]),
            "idx": torch.tensor(0),
        }
        return self.processor.preprocess(sample)

    def _collate(self, samples: list[dict]) -> dict:
        from galaxea_fm.utils.pytorch_utils import dict_apply

        keys = samples[0].keys()
        batch = {}
        for key in keys:
            values = [s[key] for s in samples]
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values, dim=0)
            else:
                batch[key] = values
        return dict_apply(
            batch, lambda x: x.to(self._device) if isinstance(x, torch.Tensor) else x
        )

    def get_action(self):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        from galaxea_fm.utils.pytorch_utils import dict_apply

        if self._sample_batch is None:
            raise AssertionError(self._error_msg("update_obs or update_obs_batch first!"))
        env_idx_list = env_idx_list if env_idx_list is not None else self._latest_env_idx_list

        with torch.no_grad():
            param_dtype = next(self.policy.parameters()).dtype
            use_bf16 = param_dtype == torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                batch = self.policy.predict_action(self._sample_batch)
        batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        batch = self.processor.postprocess(batch)
        action_dict = dict_apply(batch["action"], lambda x: x.cpu().numpy())  # {key: (B, T, n)}

        result = []
        for batch_index in range(len(env_idx_list)):
            if self.action_type == "ee":
                steps = _upstream_action_to_xpolicylab_steps(
                    action_dict, self.action_shape_meta, batch_index
                )
            else:
                parts = [action_dict[meta["key"]][batch_index] for meta in self.action_shape_meta]
                flat = np.concatenate(parts, axis=-1).astype(np.float32)  # (T, action_dim)
                steps = unpack_robot_state(
                    flat, self.action_type, self.robot_action_dim_info, source_type="obs"
                )
                if isinstance(steps, dict):
                    steps = [steps]
            result.append(steps)
        return result

    def reset(self):
        self._sample_batch = None
        self._latest_env_idx_list = [0]
