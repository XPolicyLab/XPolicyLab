"""Hy-Embodied-0.5-VLA policy for XPolicyLab / RoboDojo.

Wraps the released dual-arm flow-matching checkpoint (Hy-Embodied-0.5-VLA)
so it can be evaluated closed-loop in the RoboDojo Isaac Sim env
(env_cfg_type=arx_x5 -> dual_x5, dual-arm).

The heavy model deps (torch 2.7, the HunYuanVLMoT transformers fork,
flash_attn) live in the Hy-Embodied repo's uv venv; this process is the policy
*server* (see XPolicyLab/setup_policy_server.py). The Isaac Sim env client
runs in a separate conda env and talks to us over a socket.

The Hy-Embodied source tree (providing the ``hy_vla`` package and the
``robotwin_eval`` adapter) is located via ``hy_root`` in deploy.yml, the
``HY_VLA_ROOT`` env var, or -- by default -- a ``Hy-Embodied-0.5-VLA`` checkout
inside this policy directory. Clone it from:
    https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA

Data path per inference (mirrors Hy-VLA's own robotwin_eval adapter):
  RoboDojo obs (3 cams RGB + dual-arm EEF pose/gripper + instruction)
    -> encode_obs (16-d dual-arm state, wxyz; HWC->CHW float images)
    -> wxyz->xyzw -> apply_umi_coord_transform (-> UMI)
    -> PosRotMat6d -> normalize -> model forward -> denormalize
    -> RT-relative -> absolute UMI PosQuat -> inverse_umi_transform (-> RoboDojo)
    -> xyzw->wxyz -> per-step {left,right}_ee_pose (wxyz) + {left,right}_ee_joint_state dicts

Batched inference: ``update_obs_batch`` / ``get_action_batch`` maintain
*per-env* observation and MEM history buffers keyed by ``env_idx`` so that
parallel rollouts never contaminate each other's temporal context. Each env's
chunk is decoded independently, then the results are assembled in the same
order as the requested ``env_idx`` list.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

# Hy-Embodied source tree resolution order:
#   1. model_cfg["hy_root"] (deploy.yml)
#   2. $HY_VLA_ROOT
#   3. "Hy-Embodied-0.5-VLA" inside this policy dir.
_POLICY_DIR = Path(__file__).resolve().parent
_DEFAULT_HY_VLA_ROOT = str(_POLICY_DIR / "Hy-Embodied-0.5-VLA")

# 16-d dual-arm EEF layout (RoboTwin/Hy-VLA convention, quat is wxyz):
#   [L_xyz(3), L_quat_wxyz(4), L_grip(1), R_xyz(3), R_quat_wxyz(4), R_grip(1)]
L_POS, L_QUAT, L_GRIP = slice(0, 3), slice(3, 7), 7
R_POS, R_QUAT, R_GRIP = slice(8, 11), slice(11, 15), 15

# Camera-name candidates in the RoboDojo obs (head + dual wrist).
CAM_HEAD = ["cam_head", "head_camera", "cam_high", "top_camera"]
CAM_LEFT = ["cam_left_wrist", "left_camera", "left_wrist"]
CAM_RIGHT = ["cam_right_wrist", "right_camera", "right_wrist"]


def _extract_image(obs: dict, candidates: list[str]) -> np.ndarray:
    vision = obs.get("vision", {})
    for name in candidates:
        if name in vision:
            entry = vision[name]
            if isinstance(entry, dict):
                for k in ("color", "rgb"):
                    if k in entry:
                        return np.asarray(entry[k])
            else:
                return np.asarray(entry)
    raise KeyError(f"No image for any of {candidates}; have {list(vision.keys())}")


def _to_hwc_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce to (H, W, 3) uint8 RGB. RoboDojo/Isaac renders RGB already
    (no BGR flip), matching Hy-VLA's training-time decode."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    if img.shape[-1] == 4:
        img = img[..., :3]
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _to_chw_float(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (1, 3, H, W) float32 in [0, 1]."""
    return img_hwc_uint8.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0


def _pad_state(state: np.ndarray, max_state_dim: int = 32) -> np.ndarray:
    if state.shape[-1] == max_state_dim:
        return state
    shape = list(state.shape)
    cur = shape[-1]
    shape[-1] = max_state_dim
    out = np.zeros(shape, dtype=state.dtype)
    out[..., :cur] = state
    return out

def _convert_pose_robo_dojo(
    eepose16_wxyz: np.ndarray,
    qpos_mean: np.ndarray,
    qpos_std: np.ndarray,
    umi_coord_frame: bool,
    umi_gripper_space: bool,
) -> np.ndarray:
    """Encode a 16-d dual-arm EE state for the network (RoboDojo variant).

    Applies the same optional RoboDojo/RoboTwin -> UMI conversion that the
    latest Hy-VLA RoboDojo training dataset uses before PosQuat->PosRotMat6d.

    Input layout (quaternion is wxyz):
      [left_xyz(3), left_quat_wxyz(4), left_gripper(1),
       right_xyz(3), right_quat_wxyz(4), right_gripper(1)]
    Output: ``(1, 20)`` float, normalized in the configured model frame.
    """
    # Lazy import: hy_vla repo is only on sys.path after Model.__init__.
    from hy_vla.utils.transform_utils import (
        convert_PosQuat2PosRotationMatrix_batch,
        convert_frame_robo_to_umi,
    )

    e = eepose16_wxyz.copy()
    e[3:7] = eepose16_wxyz[[4, 5, 6, 3]]    # wxyz -> xyzw
    e[11:15] = eepose16_wxyz[[12, 13, 14, 11]]
    if umi_coord_frame:
        e = convert_frame_robo_to_umi(
            e[None, :], convert_gripper=umi_gripper_space,
        )[0]
    ee_prop = convert_PosQuat2PosRotationMatrix_batch(e[None, :], quat_order="xyzw")[0]
    ee_prop = (ee_prop - qpos_mean) / (qpos_std + 1e-8)
    return ee_prop[None, ...]


def _resolve_hy_root(model_cfg: dict[str, Any]) -> str:
    """Locate the Hy-Embodied source tree (provides hy_vla + robotwin_eval)."""
    candidates = [model_cfg.get("hy_root"), os.environ.get("HY_VLA_ROOT"), _DEFAULT_HY_VLA_ROOT]
    fallback = None
    for hy_root in candidates:
        if not hy_root:
            continue
        path = Path(hy_root).expanduser()
        if not path.is_absolute():
            path = _POLICY_DIR / path
        path = path.resolve()
        fallback = path
        if path.is_dir():
            return str(path)
    assert fallback is not None
    return str(fallback)


def _resolve_path(path_like: str | Path, base: str | Path) -> str:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return str(path.resolve())


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.cfg = model_cfg
        self.action_type = model_cfg.get("action_type", "ee")
        self.env_cfg_type = model_cfg.get("env_cfg_type")
        self.default_prompt = model_cfg.get("prompt") or model_cfg.get("task_name") or ""
        # Must match the training dataset / norm-stats generation. The
        # RoboDojo Hy-VLA training config uses UMI coordinates by default.
        self.umi_coord_frame = bool(model_cfg.get("umi_coord_frame", True))
        self.umi_gripper_space = bool(model_cfg.get("umi_gripper_space", False))
        if self.umi_gripper_space and not self.umi_coord_frame:
            raise ValueError("umi_gripper_space=true requires umi_coord_frame=true")

        # Dual-arm check: arx_x5 -> dual_x5 -> arm_dim [6,6], ee_dim [1,1].
        if self.env_cfg_type is not None:
            self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
            n_arms = len(self.robot_action_dim_info["arm_dim"])
            assert n_arms == 2, (
                f"Hy-VLA is a dual-arm policy; env_cfg_type={self.env_cfg_type!r} "
                f"resolves to {n_arms} arm(s). Use a dual-arm embodiment (e.g. arx_x5)."
            )

        hy_root = _resolve_hy_root(model_cfg)
        if not os.path.isdir(hy_root):
            raise FileNotFoundError(
                f"Hy-Embodied source tree not found at {hy_root!r}. Set 'hy_root' in "
                f"deploy.yml or $HY_VLA_ROOT, or clone "
                f"https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA into "
                f"{_DEFAULT_HY_VLA_ROOT!r}."
            )
        if hy_root not in sys.path:
            sys.path.insert(0, hy_root)

        ckpt_path = _resolve_path(model_cfg["ckpt_path"], hy_root)

        # Fallback ckpt selection via ckpt_name: setup_eval_policy_server.sh
        # already resolves ckpt_name -> a ckpt_path override, but when this
        # server is launched without that override (e.g. setup_policy_server.py
        # run directly) let a non-placeholder ckpt_name pick the checkpoint too.
        # Only kicks in when the configured ckpt_path is missing, so existing
        # absolute / relative ckpt_path behaviour is preserved.
        ckpt_name = (model_cfg.get("ckpt_name") or "").strip()
        _ckpt_placeholders = {"", "null", "none", "default", "ckpt", "ckpt_name", "-"}
        if ckpt_name.lower() not in _ckpt_placeholders and not os.path.isdir(ckpt_path):
            # Shared precedence for the ckpt_name fallback: ckpt_name-as-path >
            # 5-tuple concat under checkpoints/ > checkpoints/<ckpt_name>
            # verbatim (from the helper), followed by the Hy-specific hy_root
            # layouts. The hy_root-relative ckpt_path above stays the explicit
            # override, so those candidates are only tried when it is missing.
            from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots

            candidates = candidate_checkpoint_roots(
                model_cfg,
                _POLICY_DIR / "checkpoints",
                policy_dir=_POLICY_DIR,
                explicit_keys=(),
            )
            candidates += [
                Path(_resolve_path(ckpt_name, _POLICY_DIR)),
                Path(_resolve_path(Path("checkpoints") / ckpt_name, hy_root)),
                Path(_resolve_path(Path("Hy-VLA-RoboDojo-v3") / ckpt_name, hy_root)),
                Path(_resolve_path(Path("checkpoints") / ckpt_name, _POLICY_DIR)),
            ]
            for cand in candidates:
                if os.path.isdir(cand):
                    ckpt_path = str(cand)
                    break

        norm_cfg = model_cfg.get("norm_path") or os.environ.get("HY_VLA_NORM_PATH")
        norm_path = (
            _resolve_path(norm_cfg, hy_root)
            if norm_cfg
            else os.path.join(ckpt_path, "norm_stats.pkl")
        )

        # Decode / cadence knobs (defaults track robotwin_eval/deploy_policy.yml).
        blend_mode = model_cfg.get("blend_mode", "rel_only")
        self.exc_action_size = int(model_cfg.get("exc_action_size", 25))
        self.exc_action_interval = int(model_cfg.get("exc_action_interval", 1))
        self.img_history_size = int(model_cfg.get("img_history_size", 6))
        self.img_history_interval = int(model_cfg.get("img_history_interval", 5))

        assert self.exc_action_interval >= 1, "exc_action_interval must be >= 1"
        assert (
            self.img_history_interval % self.exc_action_interval == 0
        ), (
            f"img_history_interval ({self.img_history_interval}) must be divisible "
            f"by exc_action_interval ({self.exc_action_interval}) for strict "
            f"temporal alignment"
        )

        self.weight_dtype = torch.bfloat16

        # --- Reuse Hy-VLA's own RoboTwin transforms + config/model loader. ---
        from hy_vla import HyVLA, HyVLAConfig
        from robotwin_eval.transforms import (
            get_norm_data,
            pos_rotation_matrix_to_pos_quat,
            relative_to_dual_arm_poses,
        )
        from hy_vla.utils.transform_utils import (
            convert_frame_robo_to_umi,
            convert_frame_umi_to_robo,
        )

        self._relative_to_dual_arm_poses = relative_to_dual_arm_poses
        self._pos_rotation_matrix_to_pos_quat = pos_rotation_matrix_to_pos_quat
        self._convert_frame_robo_to_umi = convert_frame_robo_to_umi
        self._convert_frame_umi_to_robo = convert_frame_umi_to_robo

        print(f"[hy_vla] loading config + model from {ckpt_path} ...", flush=True)
        self.config = HyVLAConfig.from_pretrained(ckpt_path)
        self.policy = HyVLA.from_pretrained(ckpt_path, config=self.config)
        self.policy.enable_video_encoder_if_needed()
        self.policy.cuda().eval()
        self.policy = self.policy.to(self.weight_dtype)

        self.norm_data = get_norm_data(norm_path)
        _pkl_has_abs_keys = (
            self.norm_data.get("act_mean_abs") is not None
            and self.norm_data.get("act_std_abs") is not None
        )
        if blend_mode not in ("rel_abs", "rel_only", "abs_only"):
            raise ValueError(f"bad blend_mode {blend_mode!r}")
        if not _pkl_has_abs_keys and blend_mode != "rel_only":
            raise ValueError(
                f"blend_mode={blend_mode!r} needs abs stats in {norm_path!r}"
            )
        self.blend_mode = blend_mode

        # with_absolute: explicit flag (not inferable from norm pkl).
        self._with_abs = bool(model_cfg.get("with_absolute", False))
        if self._with_abs and not _pkl_has_abs_keys:
            raise ValueError(f"with_absolute=true requires abs stats in {norm_path!r}")

        n_act = int(self.config.n_action_steps)
        effective_chunk = n_act // 2 if self._with_abs else n_act
        print(f"[hy_vla] decode mode: with_absolute={self._with_abs}, chunk={effective_chunk}", flush=True)

        for key in ("act_mean", "act_std", "act_mean_abs", "act_std_abs"):
            val = self.norm_data.get(key)
            if val is not None and val.shape[0] != effective_chunk:
                assert effective_chunk <= val.shape[0]
                self.norm_data[key] = val[:effective_chunk].copy()

        self.use_video_encoder = bool(self.config.use_video_encoder)

        # --- Per-episode state, keyed by env_idx for batched rollouts. ---
        # Each parallel env keeps its own latest encoded observation and its own
        # MEM video-encoder frame history, so envs never share temporal context.
        self.action_cache: deque[np.ndarray] = deque()
        self._obs_by_env: dict[int, dict] = {}
        self._top_imgs_by_env: dict[int, list[np.ndarray]] = {}
        self._left_imgs_by_env: dict[int, list[np.ndarray]] = {}
        self._right_imgs_by_env: dict[int, list[np.ndarray]] = {}
        self._latest_env_idx_list: list[int] = [0]
        print(f"[hy_vla] model ready (video_encoder={self.use_video_encoder}, "
              f"blend={self.blend_mode}, exc={self.exc_action_size}, "
              f"exc_interval={self.exc_action_interval}, "
              f"umi_coord_frame={self.umi_coord_frame}, "
              f"umi_gripper_space={self.umi_gripper_space}).", flush=True)

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def encode_obs(self, obs: dict) -> dict:
        """RoboDojo v1.0 obs -> Hy-VLA batch dict (single env)."""
        head = _to_hwc_uint8_rgb(_extract_image(obs, CAM_HEAD))
        left = _to_hwc_uint8_rgb(_extract_image(obs, CAM_LEFT))
        right = _to_hwc_uint8_rgb(_extract_image(obs, CAM_RIGHT))

        state = obs.get("state", {})
        lpose = np.asarray(state["left_ee_pose"], dtype=np.float32)    # (7,) xyz+wxyz
        rpose = np.asarray(state["right_ee_pose"], dtype=np.float32)
        lgrip = float(np.asarray(state["left_ee_joint_state"]).reshape(-1)[0])
        rgrip = float(np.asarray(state["right_ee_joint_state"]).reshape(-1)[0])
        state16 = np.concatenate([
            lpose[:3], lpose[3:7], [lgrip],
            rpose[:3], rpose[3:7], [rgrip],
        ]).astype(np.float32)                                          # (16,) wxyz

        instruction = obs.get("instruction") or obs.get("prompt") or self.default_prompt

        return {
            "observation.images.top_head": _to_chw_float(head),
            "observation.images.hand_left": _to_chw_float(left),
            "observation.images.hand_right": _to_chw_float(right),
            "observation.state": _pad_state(state16[None, :], max_state_dim=32),
            "task": [instruction],
            # Raw uint8 HWC frames for the MEM video-encoder history buffer.
            "raw_images.top_head": head,
            "raw_images.hand_left": left,
            "raw_images.hand_right": right,
        }

    def update_obs(self, obs):
        # Single-env path: env_idx defaults to 0 if the env didn't tag it.
        if "env_idx" not in obs:
            obs = {**obs, "env_idx": 0}
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        """Encode and store the latest observation for each env, and append the
        raw frames to that env's own MEM history buffer.

        Per-env keying is what makes batched inference correct: with one shared
        buffer the video-encoder history of every env would be dominated by
        whichever env happened to be updated last.
        """
        self._latest_env_idx_list = [obs.get("env_idx", i) for i, obs in enumerate(obs_list)]
        for env_idx, obs in zip(self._latest_env_idx_list, obs_list):
            batch = self.encode_obs(obs)
            self._obs_by_env[env_idx] = batch
            if self.use_video_encoder:
                self._top_imgs_by_env.setdefault(env_idx, []).append(batch["raw_images.top_head"])
                self._left_imgs_by_env.setdefault(env_idx, []).append(batch["raw_images.hand_left"])
                self._right_imgs_by_env.setdefault(env_idx, []).append(batch["raw_images.hand_right"])

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------
    def get_action(self, **kwargs):
        if not self._obs_by_env:
            raise AssertionError("call update_obs first")
        env_idx = self._latest_env_idx_list[0]
        chunk16 = self._infer_chunk_wxyz(env_idx)          # (T, 16) wxyz
        return self._chunk_to_action_dicts(chunk16)

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._obs_by_env:
            raise AssertionError("call update_obs_batch first")
        # Decode each env independently against its own obs + history buffer.
        # Default to the env order from the most recent update_obs_batch.
        if env_idx_list is None:
            env_idx_list = kwargs.get("obs")
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        elif isinstance(env_idx_list, np.ndarray):
            env_idx_list = env_idx_list.reshape(-1).tolist()
        elif isinstance(env_idx_list, (int, np.integer)):
            env_idx_list = [int(env_idx_list)]
        else:
            env_idx_list = list(env_idx_list)
        return [self._chunk_to_action_dicts(self._infer_chunk_wxyz(env_idx))
                for env_idx in env_idx_list]

    def _chunk_to_action_dicts(self, chunk16_wxyz: np.ndarray) -> list[dict]:
        steps = []
        for i in range(chunk16_wxyz.shape[0]):
            row = chunk16_wxyz[i]
            steps.append({
                "left_ee_pose": np.concatenate([row[L_POS], row[L_QUAT]]).astype(np.float32),
                "right_ee_pose": np.concatenate([row[R_POS], row[R_QUAT]]).astype(np.float32),
                "left_ee_joint_state": np.array([row[L_GRIP]], dtype=np.float32),
                "right_ee_joint_state": np.array([row[R_GRIP]], dtype=np.float32),
            })
        return steps

    @torch.no_grad()
    def _infer_chunk_wxyz(self, env_idx: int) -> np.ndarray:
        """Run one flow-matching forward for a single env; return a
        (exc_action_size, 16) dual-arm PosQuat chunk in RoboTwin wxyz layout."""
        batch = self._obs_by_env[env_idx]

        # Initial EE pose: wxyz -> xyzw -> optional UMI frame, matching the
        # coordinate frame used by the training norm stats.
        initial_wxyz = batch["observation.state"][0, :16].copy()
        initial_xyzw_rd = initial_wxyz.copy()
        initial_xyzw_rd[3:7] = initial_wxyz[[4, 5, 6, 3]]
        initial_xyzw_rd[11:15] = initial_wxyz[[12, 13, 14, 11]]
        if self.umi_coord_frame:
            initial_xyzw_model = self._convert_frame_robo_to_umi(
                initial_xyzw_rd[None, :],
                convert_gripper=self.umi_gripper_space,
            )[0]
        else:
            initial_xyzw_model = initial_xyzw_rd

        # Normalize the state into the network's 20-d PosRotMat space.
        net_batch = dict(batch)
        net_batch["observation.state"] = _convert_pose_robo_dojo(
            batch["observation.state"][0],
            self.norm_data["qpos_mean"], self.norm_data["qpos_std"],
            self.umi_coord_frame, self.umi_gripper_space,
        )

        if self.use_video_encoder:
            self._inject_history_stacks(net_batch, env_idx)

        # numpy -> cuda tensors (skip raw_images.* and task).
        feed = {}
        for k, v in net_batch.items():
            if k.startswith("raw_images.") or k == "task":
                continue
            if isinstance(v, np.ndarray):
                feed[k] = torch.from_numpy(v).to(self.weight_dtype).cuda()
            elif isinstance(v, torch.Tensor):
                feed[k] = v.to(self.weight_dtype).cuda()
            else:
                feed[k] = v
        feed["task"] = net_batch["task"]

        self.policy.reset()
        action0 = self.policy.select_action(feed)
        actions = [action0]
        for _ in range(len(self.policy._action_queue)):
            actions.append(self.policy._action_queue.popleft())
        actions = torch.cat(actions, dim=0).float().cpu().numpy()   # (chunk, 20 or 40)

        actions_xyzw = self._decode_actions(actions, initial_xyzw_model)  # (T, 16) model-frame xyzw

        # Optional UMI -> RoboDojo coordinate frame.
        if self.umi_coord_frame:
            actions_rd_xyzw = self._convert_frame_umi_to_robo(
                actions_xyzw,
                convert_gripper=self.umi_gripper_space,
            )
        else:
            actions_rd_xyzw = actions_xyzw

        # xyzw -> wxyz for the env.
        actions_wxyz = actions_rd_xyzw.copy()
        actions_wxyz[:, 3:7] = actions_rd_xyzw[:, [6, 3, 4, 5]]
        actions_wxyz[:, 11:15] = actions_rd_xyzw[:, [14, 11, 12, 13]]

        # Subsample: execute every exc_action_interval-th action,
        # for a total of exc_action_size executed steps.
        # Slot 0 is the identity frame (current->current rel=zero), skip it.
        if self.exc_action_interval > 1:
            needed = self.exc_action_size * self.exc_action_interval
            actions_wxyz = actions_wxyz[1 : needed + 1 : self.exc_action_interval]
        else:
            actions_wxyz = actions_wxyz[1 : self.exc_action_size + 1]
        return actions_wxyz

    def _decode_actions(self, actions: np.ndarray, initial_xyzw: np.ndarray) -> np.ndarray:
        if not self._with_abs:
            actions = actions * self.norm_data["act_std"] + self.norm_data["act_mean"]
            return self._relative_to_dual_arm_poses(actions, initial_xyzw)

        # rel_only branch is the common case for the released ckpt.
        if self.blend_mode == "rel_only":
            half = actions.shape[0] // 2 if actions.shape[0] % 2 == 0 else actions.shape[0]
            rel = actions[:half, :20] * self.norm_data["act_std"] + self.norm_data["act_mean"]
            return self._relative_to_dual_arm_poses(rel, initial_xyzw)

        assert actions.shape[0] % 2 == 0, "rel_abs/abs need even token count"
        half = actions.shape[0] // 2
        if self.blend_mode == "abs_only":
            abs_ = actions[half:, :20] * self.norm_data["act_std_abs"] + self.norm_data["act_mean_abs"]
            out = np.zeros((abs_.shape[0], 16), dtype=abs_.dtype)
            for i in range(abs_.shape[0]):
                out[i] = np.concatenate([
                    self._pos_rotation_matrix_to_pos_quat(abs_[i, :10]),
                    self._pos_rotation_matrix_to_pos_quat(abs_[i, 10:20]),
                ])
            return out
        # rel_abs: defer to the bundled wrapper's blend (slerp) for fidelity.
        from robotwin_eval.policy_wrapper import _blend_dual_arm_pose_quat
        rel = actions[:half, :20] * self.norm_data["act_std"] + self.norm_data["act_mean"]
        p1 = self._relative_to_dual_arm_poses(rel, initial_xyzw)
        abs_ = actions[half:, :20] * self.norm_data["act_std_abs"] + self.norm_data["act_mean_abs"]
        p2 = np.zeros((abs_.shape[0], 16), dtype=abs_.dtype)
        for i in range(abs_.shape[0]):
            p2[i] = np.concatenate([
                self._pos_rotation_matrix_to_pos_quat(abs_[i, :10]),
                self._pos_rotation_matrix_to_pos_quat(abs_[i, 10:20]),
            ])
        return _blend_dual_arm_pose_quat(p1, p2)

    # --- MEM video-encoder history helpers (mirror robotwin_eval) -------
    @staticmethod
    def _eval_history_indices(step_id: int, K: int, S: int) -> list[int]:
        out = [max(step_id - (K - 1 - k) * S, 0) for k in range(K)]
        out[-1] = step_id
        return out

    def _inject_history_stacks(self, batch: dict, env_idx: int) -> None:
        K = self.img_history_size
        S_raw = self.img_history_interval
        N = self.exc_action_interval
        # Each buffer step = N raw env steps. Scale S to buffer-index units
        # so the absolute temporal coverage stays close to training.
        S_buf = max(1, S_raw // N)
        top_buf = self._top_imgs_by_env[env_idx]
        left_buf = self._left_imgs_by_env[env_idx]
        right_buf = self._right_imgs_by_env[env_idx]
        step_id = len(top_buf) - 1
        idx_list = self._eval_history_indices(step_id, K, S_buf)
        valid = [(step_id - (K - 1 - k) * S_buf) >= 0 for k in range(K)]

        def _stack(buf: list[np.ndarray]) -> torch.Tensor:
            frames = [buf[i] for i in idx_list]
            arr = torch.from_numpy(np.stack(frames, 0)).permute(0, 3, 1, 2).float() / 255.0
            for k, ok in enumerate(valid):
                if not ok:
                    arr[k].zero_()
            return arr.unsqueeze(0)  # (1, K, C, H, W)

        batch["observation.images.top_head"] = _stack(top_buf)
        batch["observation.images.hand_left"] = _stack(left_buf)
        batch["observation.images.hand_right"] = _stack(right_buf)

    # ------------------------------------------------------------------
    def reset(self):
        self.policy.reset()
        self.action_cache.clear()
        self._obs_by_env.clear()
        self._top_imgs_by_env.clear()
        self._left_imgs_by_env.clear()
        self._right_imgs_by_env.clear()
        self._latest_env_idx_list = [0]
        print("[hy_vla] reset", flush=True)


__all__ = ["Model"]
