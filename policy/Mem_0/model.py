"""
XPolicyLab adapter for Mem_0 (MemoryMatters) execution + planning agent.

Wraps upstream ``MemoryMattersAgent`` behind the XPolicyLab server/client contract.
Deploy loop uses ``begin_episode`` / ``step`` / ``reset`` (see deploy.py), which
ports the upstream chunk-based eval with action smoothing and Mn subtask switching.

Action / state mapping (dual-arm joint, robot ``dual_x5``):

    XPolicyLab packed (14): [LA(6), LGrip, RA(6), RGrip]
    Mem_0 model layout (16): [LA(6),pad, RA(6),pad, LGrip, RGrip]
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
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
from source.agent.memorymatters_agent import MemoryMattersAgent  # noqa: E402

STD_W, STD_H = 320, 240
ARM_NORM_DIMS = 14
TMP_VISUAL = "./_tmp_visual"


def _resolve_path(path, base=CURRENT_DIR):
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def _resolve_task_type(task_name: str, override: str | None) -> str:
    if override in ("M1", "Mn"):
        return override
    cfg_path = os.path.join(UPSTREAM_DIR, "xpolicylab_adapter", "task_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if task_name in (cfg.get("Mn") or []):
            return "Mn"
        if task_name in (cfg.get("M1") or []):
            return "M1"
    return "M1"


def _standardize_image(color: np.ndarray) -> np.ndarray:
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

        assert len(self.robot_action_dim_info["arm_dim"]) == 2, (
            "Mem_0 expects a dual-arm robot (e.g. env_cfg_type=arx_x5 -> dual_x5); "
            f"got arm_dim={self.robot_action_dim_info['arm_dim']}."
        )
        if self.action_type != "joint":
            cprint(
                f"[Mem_0] action_type={self.action_type!r}; Mem_0 was trained joint-space.",
                "yellow",
            )

        device_str = model_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.norm_way = model_cfg.get("norm_way", "minmax")
        self.image_size = tuple(model_cfg.get("image_size", (224, 224)))
        self.task_type = _resolve_task_type(
            model_cfg.get("task_name") or "",
            model_cfg.get("task_type"),
        )

        cfg = OmegaConf.create(dict(model_cfg))
        qwen_path = _resolve_path(cfg.execution_module.qwen_vl.get("model_path", ""))
        cfg.execution_module.qwen_vl.model_path = qwen_path
        if not os.path.isdir(qwen_path):
            cprint(
                f"[Mem_0] Qwen3-VL backbone not found at {qwen_path}. "
                "Download it first (see INSTALLATION.md).",
                "red",
            )

        planning_cfg = _resolve_path(cfg.get("planning_module_config_path", ""))
        if planning_cfg:
            cfg.planning_module_config_path = planning_cfg

        ckpt_path = _resolve_path(model_cfg.get("execution_ckpt", ""))
        self._stats = {}
        stats_path = _resolve_path(model_cfg.get("state_stats_path", ""))
        if stats_path and os.path.isfile(stats_path):
            self._stats = load_stats(stats_path)
            cprint(f"[Mem_0] loaded norm stats from {stats_path}", "cyan")
        else:
            cprint(
                f"[Mem_0] no norm stats ({stats_path or 'unset'}); "
                "running un-normalized (debug wiring only).",
                "yellow",
            )

        cprint(f"[Mem_0] building MemoryMattersAgent on {self.device} task_type={self.task_type}", "cyan")
        self.agent = MemoryMattersAgent(cfg, ckpt_path=ckpt_path, device=self.device)
        self.agent.task_type = self.task_type
        self.agent.action_horizon = int(model_cfg.get("action_horizon", self.agent.action_horizon))
        self.agent.action_strip = self.agent.action_horizon
        self.agent.threshold = int(model_cfg.get("threshold", self.agent.threshold))

        self._chunk_queue: list[dict] = []
        self._need_obs_update = False
        self._macro_started = False
        self._last_rgb: np.ndarray | None = None

        os.makedirs(TMP_VISUAL, exist_ok=True)
        cprint("[Mem_0] Model initialized", "green")

    # ------------------------------------------------------------------ #
    # Normalization + layout
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

    def _packed14_to_env16(self, packed14: np.ndarray) -> np.ndarray:
        env16 = np.zeros(16, dtype=np.float32)
        env16[0:6] = packed14[0:6]
        env16[6] = 0.0
        env16[7] = packed14[6]
        env16[8:14] = packed14[7:13]
        env16[14] = 0.0
        env16[15] = packed14[13]
        return env16

    def _env16_to_packed14(self, env16: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [env16[0:6], env16[7:8], env16[8:14], env16[15:16]], axis=0
        ).astype(np.float32)

    def encode_obs(self, observation: dict) -> dict:
        color = observation["vision"]["cam_head"]["color"]
        std_img = _standardize_image(color)
        pil_image = to_pil(std_img, self.image_size)

        packed14 = pack_robot_state(
            observation, self.action_type, self.robot_action_dim_info, source_type="obs"
        ).reshape(-1)
        env16 = self._packed14_to_env16(packed14)
        model16 = env_to_model_layout(env16)
        norm_state = self._normalize_state(model16)

        instruction = (
            observation.get("instruction")
            or self.agent.instruction
            or self.model_cfg.get("global_task")
            or ""
        )
        return {
            "image": pil_image,
            "state": norm_state.reshape(1, -1),
            "instruction": instruction,
            "_rgb": std_img,
        }

    def _postprocess_chunk(self, actions_model: np.ndarray) -> list[dict]:
        chunk = np.asarray(actions_model, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        flat = chunk.reshape(-1, chunk.shape[-1])
        out = []
        for step in flat:
            denorm = self._denormalize_action(step.astype(np.float32))
            env16 = model_to_env_layout(denorm)
            packed14 = self._env16_to_packed14(env16)
            out.append(
                unpack_robot_state(
                    packed14, self.action_type, self.robot_action_dim_info, source_type="obs"
                )
            )
        return out

    def _write_ffmpeg(self, rgb: np.ndarray) -> None:
        if getattr(self.agent, "ffmpeg", None) is not None and self.agent.is_init == 1:
            self.agent.ffmpeg.stdin.write(np.asarray(rgb, dtype=np.uint8).tobytes())

    def _save_stage_image(self, rgb: np.ndarray, path: str) -> None:
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)

    def _start_macro_chunk(self, encoded: dict) -> None:
        if self.agent.action_count == 0:
            payload = {k: v for k, v in encoded.items() if not k.startswith("_")}
            self.agent.update_obs(payload)

        result = self.agent.get_action()
        if result is None or len(result) == 0:
            raise RuntimeError(self._error_msg("Empty actions from model; aborting eval."))

        self.agent.accumulate_actions_chunk(result["normalized_actions"])
        smoothed = self.agent.get_smoothed_actions(self.agent.iter, self.agent.action_strip)
        steps_to_run = min(self.agent.action_strip, smoothed.shape[0])
        env_actions = self._postprocess_chunk(smoothed[:steps_to_run])
        self._chunk_queue = list(env_actions)
        self._macro_started = True
        self._steps_in_chunk = steps_to_run
        self._need_obs_update = False

    def _advance_macro_iter(self) -> None:
        if self._macro_started:
            self.agent.iter += self._steps_in_chunk
            self._macro_started = False

    def _handle_subtask_transition(self, rgb: np.ndarray) -> None:
        self._write_ffmpeg(rgb)
        self._save_stage_image(rgb, f"{TMP_VISUAL}/image_{self.agent.stage}.png")
        if getattr(self.agent, "ffmpeg", None) is not None:
            self.agent._del_video_ffmpeg()
        cprint(f"[Mem_0] subtask end; moving to stage {self.agent.stage + 1}", "green")
        self.agent.update_high_observation()
        self.agent.end_signal_count = 0
        self.agent.action_count = 0
        self.agent.executor.memory_bank.reset()
        self._chunk_queue = []
        self._write_ffmpeg(rgb)

    # ------------------------------------------------------------------ #
    # Deploy RPC (primary eval path)
    # ------------------------------------------------------------------ #
    def begin_episode(self, obs):
        os.makedirs(TMP_VISUAL, exist_ok=True)
        encoded = self.encode_obs(obs)
        rgb = encoded["_rgb"]
        self._last_rgb = rgb
        self._save_stage_image(rgb, f"{TMP_VISUAL}/init.png")

        if self.task_type == "Mn":
            self.agent.init_high_with_image()
        else:
            self.agent.instruction = (
                self.model_cfg.get("global_task")
                or obs.get("instruction")
                or encoded["instruction"]
                or self.model_cfg.get("task_name")
                or ""
            )
            self.agent._set_video_ffmpeg()

        self.agent.is_init = 1
        self._chunk_queue = []
        self._macro_started = False
        self._need_obs_update = False
        cprint(f"[Mem_0] begin_episode type={self.task_type} instruction={self.agent.instruction!r}", "cyan")

    def step(self, obs):
        """Return one env action dict, or None between macro chunks / after subtask switch."""
        encoded = self.encode_obs(obs)
        rgb = encoded["_rgb"]
        self._last_rgb = rgb

        if self._need_obs_update:
            payload = {k: v for k, v in encoded.items() if not k.startswith("_")}
            sub_end = self.agent.update_obs(payload)
            self._need_obs_update = False
            if self.task_type == "Mn" and sub_end == 1:
                cprint(f"[Mem_0] subtask end signal on iter={self.agent.iter}", "yellow")

            if self._macro_started and not self._chunk_queue:
                self._advance_macro_iter()

            if self.task_type == "Mn" and self.agent.end_signal_count >= self.agent.threshold:
                self._handle_subtask_transition(rgb)
                return None

        if not self._chunk_queue:
            self._start_macro_chunk(encoded)
            self._write_ffmpeg(rgb)

        if not self._chunk_queue:
            return None

        action = self._chunk_queue.pop(0)
        self._need_obs_update = True
        return action

    # ------------------------------------------------------------------ #
    # ModelTemplate compatibility (debug / legacy)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def update_obs(self, obs):
        encoded = self.encode_obs(obs)
        payload = {k: v for k, v in encoded.items() if not k.startswith("_")}
        self.agent.update_obs(payload)

    @torch.inference_mode()
    def get_action(self):
        result = self.agent.get_action()
        if result is None:
            raise RuntimeError(self._error_msg("get_action called before update_obs."))
        chunk = result["normalized_actions"]
        if chunk.ndim == 3:
            chunk = chunk[0]
        return self._postprocess_chunk(chunk)

    def update_obs_batch(self, obs_list):
        raise NotImplementedError(self._error_msg("Mem_0 does not support batch inference."))

    def get_action_batch(self, env_idx_list):
        raise NotImplementedError(self._error_msg("Mem_0 does not support batch inference."))

    def reset(self):
        self.agent.reset()
        self.agent.task_type = self.task_type
        self._chunk_queue = []
        self._macro_started = False
        self._need_obs_update = False
        self._last_rgb = None
        cprint(f"[Mem_0] reset (episode {self.agent.episode_id})", "cyan")
