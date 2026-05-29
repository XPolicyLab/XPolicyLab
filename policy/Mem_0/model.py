"""
XPolicyLab adapter for Mem_0 (MemoryMatters) execution module.

Wraps the upstream ``MemoryMattersExecutor`` behind the XPolicyLab
``ModelTemplate`` server/client contract:

- ``update_obs``  : encode one env observation, push it through Qwen3-VL +
                    MemoryBank, and cache the fused feature (no action yet).
- ``get_action``  : run the flow-matching action head on the cached feature,
                    denormalize, remap to the env action layout, and return a
                    list of per-step action dicts.
- ``reset``       : reset the MemoryBank between episodes.

Only the execution module is wired here. The planning module (Qwen3-VL-8B via
vLLM, used for multi-stage "Mn" tasks to produce sub-task instructions) is an
external server; for the debug/sim client the instruction is taken from the
observation (or ``global_task`` in deploy.yml). See INSTALLATION.md.

Action / state mapping (dual-arm joint, robot ``dual_x5``: arm_dim=[6,6],
ee_dim=[1,1] -> 14-dim packed state):

    XPolicyLab packed (14): [LA(6), LGrip, RA(6), RGrip]
    Mem_0 model layout (16): [LA(6),pad, RA(6),pad, LGrip, RGrip]

The 14<->16 bridge reuses the upstream layout/normalization helpers so training
and deployment share one convention.
"""

import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.join(CURRENT_DIR, "Mem_0")
if UPSTREAM_DIR not in sys.path:
    sys.path.insert(0, UPSTREAM_DIR)

# Upstream helpers (lightweight: numpy / PIL only) and the heavy executor.
from scripts.tools_for_deploy.image_utils import to_pil  # noqa: E402
from scripts.tools_for_deploy.layout_utils import (  # noqa: E402
    env_to_model_layout,
    model_to_env_layout,
)
from scripts.tools_for_deploy.normlization import (  # noqa: E402
    denormalize_arms,
    load_stats,
    normalize_arms,
)
from source.models.execution_module.memorymatters_executor import (  # noqa: E402
    MemoryMattersExecutor,
    resize_images,
)

# Standardized camera frame before model preprocessing (skill image rule).
STD_W, STD_H = 320, 240
# Arm dims that carry continuous normalization (grippers 14,15 are left raw).
ARM_NORM_DIMS = 14


def _resolve_path(path, base=CURRENT_DIR):
    """Resolve a possibly-relative path against the policy folder."""
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def _standardize_image(color: np.ndarray) -> np.ndarray:
    """env color image -> RGB HWC (240, 320, 3) uint8, ready for Qwen preprocessing."""
    img = np.asarray(color)
    assert img.ndim == 3 and img.shape[-1] == 3, f"Expected HxWx3, got {img.shape}"
    img = cv2.resize(img, (STD_W, STD_H), interpolation=cv2.INTER_AREA)
    assert img.shape == (STD_H, STD_W, 3), f"Expected {(STD_H, STD_W, 3)}, got {img.shape}"
    return img


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)

        # Mem_0 is dual-arm joint-space (16-dim model layout). Fail loudly otherwise.
        assert len(self.robot_action_dim_info["arm_dim"]) == 2, (
            "Mem_0 expects a dual-arm robot (e.g. env_cfg_type=arx_x5 -> dual_x5); "
            f"got arm_dim={self.robot_action_dim_info['arm_dim']}."
        )
        if self.action_type != "joint":
            cprint(
                f"[Mem_0] action_type={self.action_type!r}; Mem_0 was trained joint-space. "
                "Proceeding, but joint is expected.",
                "yellow",
            )

        device_str = model_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.norm_way = model_cfg.get("norm_way", "minmax")
        self.image_size = tuple(model_cfg.get("image_size", (224, 224)))
        self.instruction = ""

        # --- Build executor config (resolve checkpoint paths to absolute) ---
        cfg = OmegaConf.create(dict(model_cfg))
        qwen_path = _resolve_path(cfg.execution_module.qwen_vl.get("model_path", ""))
        cfg.execution_module.qwen_vl.model_path = qwen_path
        if not os.path.isdir(qwen_path):
            cprint(
                f"[Mem_0] Qwen3-VL backbone not found at {qwen_path}. "
                "Download it first (see INSTALLATION.md / Mem_0/checkpoints/_download.py).",
                "red",
            )

        cprint(f"[Mem_0] building executor on {self.device}", "cyan")
        self.model = MemoryMattersExecutor(cfg, device=self.device).to(self.device)
        self.model.eval()
        self._load_ckpt(_resolve_path(model_cfg.get("execution_ckpt", "")))

        # --- Normalization stats (state/action min-max etc.) ---
        self._stats = {}
        stats_path = _resolve_path(model_cfg.get("state_stats_path", ""))
        if stats_path and os.path.isfile(stats_path):
            self._stats = load_stats(stats_path)
            cprint(f"[Mem_0] loaded norm stats from {stats_path}", "cyan")
        else:
            cprint(
                f"[Mem_0] no norm stats ({stats_path or 'unset'}); running un-normalized "
                "(fine for the debug-client wiring check, not for real rollouts).",
                "yellow",
            )

        # --- Episode/rollout state ---
        self.episode_id = 0
        self._last_fused = None
        self._last_state = None
        cprint("[Mem_0] Model initialized", "green")

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def _load_ckpt(self, ckpt_path: str) -> None:
        if not ckpt_path:
            cprint("[Mem_0] no execution_ckpt; using randomly initialized action head.", "red")
            return
        if not os.path.isfile(ckpt_path):
            cprint(f"[Mem_0] execution_ckpt not found: {ckpt_path}", "red")
            return
        try:
            payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:  # PyTorch < 2.6
            payload = torch.load(ckpt_path, map_location=self.device)
        state_dict = payload.get("model_state_dict", payload)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        cprint(f"[Mem_0] checkpoint loaded: {ckpt_path}", "green")
        if missing:
            cprint(f"[Mem_0] missing keys: {len(missing)}", "yellow")
        if unexpected:
            cprint(f"[Mem_0] unexpected keys: {len(unexpected)}", "yellow")

    # ------------------------------------------------------------------ #
    # Normalization helpers (state in, action out; only arm dims 0..13)
    # ------------------------------------------------------------------ #
    def _normalize_state(self, state_vec: np.ndarray) -> np.ndarray:
        if self.norm_way == "minmax" and self._stats.get("state_min") is not None:
            return normalize_arms(
                state_vec, None, None,
                self._stats["state_min"], self._stats["state_max"], arm_dims=ARM_NORM_DIMS,
            )
        if self.norm_way == "meanstd" and self._stats.get("state_mean") is not None:
            return normalize_arms(
                state_vec, self._stats["state_mean"], self._stats["state_std"],
                None, None, arm_dims=ARM_NORM_DIMS,
            )
        if self.norm_way == "quantile" and self._stats.get("state_q01") is not None:
            return normalize_arms(
                state_vec, None, None, None, None, arm_dims=state_vec.shape[-1],
                quantile=True, q01=self._stats["state_q01"], q99=self._stats["state_q99"],
            )
        return state_vec

    def _denormalize_action(self, action_vec: np.ndarray) -> np.ndarray:
        if self.norm_way == "minmax" and self._stats.get("action_min") is not None:
            return denormalize_arms(
                action_vec, None, None,
                self._stats["action_min"], self._stats["action_max"], arm_dims=ARM_NORM_DIMS,
            )
        if self.norm_way == "meanstd" and self._stats.get("action_mean") is not None:
            return denormalize_arms(
                action_vec, self._stats["action_mean"], self._stats["action_std"],
                None, None, arm_dims=ARM_NORM_DIMS,
            )
        if self.norm_way == "quantile" and self._stats.get("action_q01") is not None:
            return denormalize_arms(
                action_vec, None, None, None, None, arm_dims=action_vec.shape[-1],
                quantile=True, q01=self._stats["action_q01"], q99=self._stats["action_q99"],
            )
        return action_vec

    # ------------------------------------------------------------------ #
    # Observation encoding + 14<->16 layout bridge
    # ------------------------------------------------------------------ #
    def _packed14_to_env16(self, packed14: np.ndarray) -> np.ndarray:
        """[LA(6),LGrip,RA(6),RGrip] -> [LA(6),pad,LGrip,RA(6),pad,RGrip] (env layout for layout_utils)."""
        env16 = np.zeros(16, dtype=np.float32)
        env16[0:6] = packed14[0:6]      # left arm
        env16[6] = 0.0                  # left arm 7th-joint pad
        env16[7] = packed14[6]          # left gripper
        env16[8:14] = packed14[7:13]    # right arm
        env16[14] = 0.0                 # right arm 7th-joint pad
        env16[15] = packed14[13]        # right gripper
        return env16

    def _env16_to_packed14(self, env16: np.ndarray) -> np.ndarray:
        """Inverse of _packed14_to_env16 (drop the two arm pads)."""
        return np.concatenate(
            [env16[0:6], env16[7:8], env16[8:14], env16[15:16]], axis=0
        ).astype(np.float32)

    def encode_obs(self, observation: dict) -> dict:
        """env observation -> {image: PIL(224), state: (1,16) normalized model layout, instruction}."""
        color = observation["vision"]["cam_head"]["color"]
        std_img = _standardize_image(color)  # (240,320,3) RGB uint8
        pil_image = to_pil(std_img, self.image_size)

        # XPolicyLab packed state (14) -> Mem_0 env layout (16, with arm pads) -> model layout (16)
        packed14 = pack_robot_state(
            observation, self.action_type, self.robot_action_dim_info, source_type="obs"
        ).reshape(-1)
        env16 = self._packed14_to_env16(packed14)
        model16 = env_to_model_layout(env16)
        norm_state = self._normalize_state(model16)

        instruction = observation.get("instruction") or self.instruction or self.model_cfg.get("global_task", "")
        return {"image": pil_image, "state": norm_state.reshape(1, -1), "instruction": instruction}

    # ------------------------------------------------------------------ #
    # XPolicyLab Model interface
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def update_obs(self, obs):
        """Encode one observation and update the MemoryBank (no action prediction)."""
        payload = self.encode_obs(obs)

        images = resize_images([[payload["image"]]], target_size=self.image_size)
        instruction = [payload["instruction"]]
        qwen_inputs = self.model.qwen_model.build_qwenvl_inputs(
            images, instruction, system_prompt=None,
            add_summary_token=False, add_generation_prompt=False, max_length=128,
        )
        qwen_out = self.model.qwen_model(
            **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True,
        )
        last_hidden = qwen_out.hidden_states[-1]
        image_feature, text_feature = self.model.qwen_model.extract_features(
            qwen_inputs.input_ids, last_hidden
        )
        memory_fusion, anchor, _sub_end = self.model.memory_bank.update_on_eval(
            image_feature, text_feature, self.model.classifier, episode_id=self.episode_id
        )
        self._last_fused = torch.cat([memory_fusion, anchor, text_feature], dim=1)  # (1,3,H)

        state = np.asarray(payload["state"], dtype=np.float32)
        state_t = torch.from_numpy(state).to(self._last_fused.device, dtype=self._last_fused.dtype)
        if state_t.dim() == 2:
            state_t = state_t.unsqueeze(1)  # (1,1,16)
        self._last_state = state_t

    def update_obs_batch(self, obs_list):
        raise NotImplementedError(
            self._error_msg(
                "Mem_0 MemoryBank tracks per-episode temporal state; batch (multi-env) "
                "inference is not supported. Run the debug/sim client with eval_batch=false."
            )
        )

    @torch.inference_mode()
    def get_action(self):
        """Predict a normalized action chunk, denormalize, remap to env layout, return per-step dicts."""
        if self._last_fused is None:
            raise RuntimeError(self._error_msg("get_action called before update_obs."))

        with torch.autocast("cuda", dtype=torch.float32, enabled=self.device.type == "cuda"):
            pred = self.model.action_model.predict_action(self._last_fused, self._last_state)
        chunk = pred.detach().cpu().numpy()
        if chunk.ndim == 3:
            chunk = chunk[0]  # (T, 16) drop batch
        chunk = np.atleast_2d(chunk)

        action_dicts = []
        for step in chunk:
            denorm_model16 = self._denormalize_action(step.astype(np.float32))
            env16 = model_to_env_layout(denorm_model16)
            packed14 = self._env16_to_packed14(env16)
            action_dicts.append(
                unpack_robot_state(
                    packed14, self.action_type, self.robot_action_dim_info, source_type="obs"
                )
            )
        return action_dicts

    def get_action_batch(self, env_idx_list):
        raise NotImplementedError(
            self._error_msg("Mem_0 does not support batch inference (see update_obs_batch).")
        )

    def reset(self):
        """Reset MemoryBank and rollout cache between episodes."""
        self.episode_id += 1
        self.model.memory_bank.reset()
        self._last_fused = None
        self._last_state = None
        cprint(f"[Mem_0] reset (episode {self.episode_id})", "cyan")
