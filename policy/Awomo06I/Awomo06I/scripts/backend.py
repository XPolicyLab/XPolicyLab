# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RoboCasa 闭环评测的 AP394 policy backend（VLA Harness plugin，推理合同 `ap394-history-vl-v1`）。

AP394 = aap337 Action-as-Patch + 9 帧 2 s 间隔历史（VAE 与 Qwen 共用）+ Qwen3-VL LoRA 条件。合同冻结于
model_project/docs/robocasa-sft-aap-20261003/15-eval-adapter-contract.md，与训练逐项对齐：

- 观测：Harness `history_window_nokv` 模式每次 replan 送上次以来的新帧（5 Hz，观测步为 20 Hz 控制步号），并带
  `camera_frames_block` uint8 [T, 3, 256, 256, 3]（相机顺序 agentview_left / agentview_right / eye_in_hand）。
  插件按 session 缓存三路原始相机帧；新 session（replan 0）清空。不使用 Harness 画布（384×320 裁剪版与训练不同）。
- 单一前处理路径：当前帧 = 本次最后一个观测步 s；历史帧号与有效性直接调用训练的 `history_raw_indices(s, 9, 40)`；
  画布调用训练的 `compose_eye_in_hand_top_square_canvas` -> `aap_canvas_to_model_video`；prompt 调用
  `AP394PromptCodec.encode`；State 调用 `state16_to_canonical80`；样本经训练的 `ap394_collate_fn` 组 batch。
- 模型：直接实例化训练 recipe（`TrainActionPatchRoboCasaAP394Recipe.setup_for_inference`），调用 recipe 的
  `_encode_batch`（Qwen3-VL + LoRA 条件、reference latent、历史 VAE token）与 `_model_forward`（带历史的 DiT 前向）；
  去噪循环与父模型 AAP 离线推理 `ActionPatchOfflineActionAdapter._denoise_group` 相同（图像与动作共用 t，
  FlowMatchScheduler shift 5，16 步，seed = 请求 seed），只是改走带历史的 `_model_forward`；无效动作槽位按
  `inactive_action_slots` 处理（默认 noise_path：固定在训练输入分布 σ·ε₀；free = cf03efb9 旧行为；zero = DreamZero
  式全程置 0），结束后置零。
- 子任务：可选（`subtask_decode_every`，默认每次 replan）在同一次 Qwen prefill 的 KV cache 上贪心解码子任务句子，
  只写回执；DiT 条件只取 prompt 内位置，子任务 token 在 prompt 之后、因果不可见，不改变动作。
- 后处理：16 个 20 Hz Canonical80 -> `aap_action12_to_canonical80` 的逆 -> [16, 12] 原始 Action12（训练顺序，
  Engine 置换由 Harness 完成），不做任何时间重采样；无效槽位置零并记录置零前最大绝对值；夹爪 / control_mode
  默认二值化为 ±1（`binarize_discrete_actions`），二值化前的连续值写入回执。
- 权重：raw = `model/` + `vl_lora/adapter.safetensors`；EMA = `ema/`（主 Power EMA，σ_rel 0.10）+
  `vl_lora/ema.safetensors`；ema_secondary = `ema_secondary/`（第二套 Power EMA，σ_rel 0.05）+
  `vl_lora/ema_secondary.safetensors`；DiT 与 LoRA 总是一起切换，任何缺失/多余 key 直接拒绝。
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from transformers import AutoProcessor

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.checkpoint.checkpointing import _is_safetensors_checkpoint
from nemo_automodel.components.datasets.diffusion.robocasa_lerobot.aap import (
    AAP_ACTION_TARGETS,
    AAP_ADAPTER_ID,
    AAP_DEFAULT_MODEL_VIDEO_SIZE,
    AAP_VIDEO_OFFSETS,
    AAP_WINDOW_PROFILE,
    aap_canvas_to_model_video,
    compose_eye_in_hand_top_square_canvas,
)
from nemo_automodel.components.datasets.diffusion.robocasa_lerobot.adapter import (
    CANONICAL_DIM,
    RAW_ACTION_CONTROL_MODE,
    RAW_ACTION_DIM,
    RAW_ACTION_EEF_POSITION_DELTA,
    RAW_ACTION_EEF_ROTATION_DELTA,
    RAW_ACTION_GRIPPER,
    RAW_ACTION_TORSO,
    RAW_STATE_DIM,
    SLOT_BASE_XY,
    SLOT_BASE_YAW,
    SLOT_CONTROL_MODE,
    SLOT_EEF_POSITION,
    SLOT_EEF_ROTATION_DELTA,
    SLOT_GRIPPER,
    STATE_MASK80,
    state16_to_canonical80,
    validate_raw_action12,
)
from nemo_automodel.components.datasets.diffusion.robocasa_lerobot.aap_rotation6d import ACTION_ROTATION6D_MASK80 as ACTION_MASK80
from nemo_automodel.components.datasets.diffusion.command_action_normalization import CommandActionNormalizer
from nemo_automodel.components.datasets.diffusion.canonical_state_normalization import CanonicalStateNormalizer
from nemo_automodel.components.datasets.diffusion.robocasa_lerobot.ap394_dataset import (
    AP394_HISTORY_PERIOD_RAW,
    AP394_HISTORY_SLOTS,
    ap394_collate_fn,
)
from nemo_automodel.components.datasets.diffusion.robocasa_lerobot.ap394_vl import (
    PROMPT_VERSION,
    AP394PromptCodec,
    history_raw_indices,
)
from nemo_automodel.components.datasets.diffusion.robocasa_lerobot.harness_backend import (
    REQUEST_ID_PATTERN,
    _sha256_file,
    validate_observation_increment,
)
from nemo_automodel.components.datasets.diffusion.robocasa_lerobot.video import CAMERA_KEYS
from nemo_automodel.components.models.action_patch.history import HISTORY_POOL_GRID, history_image_ids
from nemo_automodel.components.models.action_patch.vl_history import CONDITION_LAYERS, SUMMARY_LAYERS, TEXT_LAYOUT
from nemo_automodel.components.models.dreamzero.flow_match_scheduler import FlowMatchScheduler
from nemo_automodel.components.training.ema import PowerFunctionEMAManager, power_function_gamma
from nemo_automodel.components.training.rng import init_all_rng

BACKEND_NAME = "ap394-history-vl"
INFERENCE_CONTRACT_ID = "ap394-five-macro2-qnorm-v1"
INFERENCE_MODE = "history_window_nokv"
DEFAULT_MODEL_CONFIG = "Automodel/examples/diffusion/finetune/robocasa_aap337_ap394/train_ap394.yaml"

# 时序合同：20 Hz 控制、5 Hz 观测（每 4 个控制步一帧）；每次 replan 预测并执行 16 个原生 20 Hz 动作（0.8 s），
# 因此每次 replan 恰好新到 4 帧。历史间隔 40 步是观测步长 4 的整数倍，且 replan 锚点都在 16 的倍数上，
# 所以训练规则下有效（s - 40j >= 0）的历史帧一定已在缓存里。
CONTROL_HZ = 20.0
OBSERVATION_HZ = 5.0
OBSERVATION_STRIDE = 4
ACTION_HORIZON = AAP_ACTION_TARGETS
REPLAN_STEPS = AAP_ACTION_TARGETS
NUM_INFERENCE_STEPS = 16
# 子任务解码 token 上限（含 `<|im_end|>`）。训练 `AP394PromptCodec.subtask_ids` 不截断子任务句子，训练代码里
# 没有长度上限；子任务是“一句简短祈使句”，64 个 token 足够，到上限仍未见结束符时记为 max_new_tokens。
SUBTASK_MAX_NEW_TOKENS = 64
# 预测图像 = 训练 target 帧：reference 之后 16 个 20 Hz 控制步（+0.8 s）。
PREDICTED_IMAGE_STEP_OFFSET = int(AAP_VIDEO_OFFSETS[-1])
# 缓存只需覆盖最远一张历史帧：s - 40 * 9 = s - 360。
HISTORY_REACH_STEPS = AP394_HISTORY_SLOTS * AP394_HISTORY_PERIOD_RAW

# Harness `CAMERA_FRAMES_BLOCK_KEYS`（= robocasa_closeloop.contracts.CAMERA_KEYS）的相机轴顺序；与训练
# `video.CAMERA_KEYS` 同序（agentview_left, agentview_right, eye_in_hand），模块加载时逐项核对。
CAMERA_FRAMES_BLOCK_FIELD = "camera_frames_block"
HARNESS_CAMERA_ORDER = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
for _harness_name, _training_key in zip(HARNESS_CAMERA_ORDER, CAMERA_KEYS, strict=True):
    if not _training_key.endswith(f".{_harness_name}"):
        raise RuntimeError(f"camera order drift: Harness {_harness_name!r} vs training {_training_key!r}")
CAMERA_FRAME_SHAPE = (len(CAMERA_KEYS), 256, 256, 3)

WEIGHTS_RAW = "raw"
WEIGHTS_EMA = "ema"
WEIGHTS_EMA_SECONDARY = "ema_secondary"
INACTIVE_SLOTS_NOISE_PATH = "noise_path"
INACTIVE_SLOTS_FREE = "free"
INACTIVE_SLOTS_ZERO = "zero"
# 诊断：训练视频是 libx264 CRF 23 / yuv420p / 256² / 20 fps（重新编码码率对齐核对，10-04），闭环送来的是无损渲染。
# "h264_crf23"：每次 replan 的新帧逐路以同参数（5 Hz 片段）编码再解码后再进缓存与画布，用来检验压缩域差。
INPUT_ROUNDTRIP_NONE = "none"
INPUT_ROUNDTRIP_H264_CRF23 = "h264_crf23"
# 冻结方案 §6：EMA 评测只用主 Power EMA（σ_rel 0.10，目录 ema/），DiT 与 LoRA 一起换。
EMA_SIGMA_REL = 0.10
EMA_DIRECTORY = "ema"
# 10-05 用户要求在最终 checkpoint 上评第二套 Power EMA（训练同时保存，σ_rel 0.05）：weights="ema_secondary"。
EMA_SECONDARY_SIGMA_REL = 0.05
EMA_SECONDARY_DIRECTORY = "ema_secondary"
# weights → (DiT EMA 目录, σ_rel, LoRA EMA 文件)
EMA_VARIANTS = {
    WEIGHTS_EMA: (EMA_DIRECTORY, EMA_SIGMA_REL, "ema.safetensors"),
    WEIGHTS_EMA_SECONDARY: (EMA_SECONDARY_DIRECTORY, EMA_SECONDARY_SIGMA_REL, "ema_secondary.safetensors"),
}
LORA_DIRECTORY = "vl_lora"
CHECKPOINT_WRAPPER_PREFIX = "_checkpoint_wrapped_module."
POWER_EMA_METADATA_KEYS = (
    PowerFunctionEMAManager._NUM_UPDATES_KEY,
    PowerFunctionEMAManager._SIGMA_REL_KEY,
    PowerFunctionEMAManager._GAMMA_KEY,
)


@dataclass(frozen=True)
class AP394BackendOptions:
    """`--backend-options` JSON；未知键直接报错。时序字段只用于声明并核对合同，不能改成别的值。"""

    weights: str
    action_horizon: int = ACTION_HORIZON
    replan_steps: int = REPLAN_STEPS
    control_hz: float = CONTROL_HZ
    observation_sample_freq: float = OBSERVATION_HZ
    expected_training_commit: str | None = None
    # 夹爪 / control_mode 是否二值化为 ±1（训练数据里两者恒为 ±1）；关掉时输出 clip 到 [-1, 1] 的连续值。
    # 二值化前的连续值总是写入回执。
    binarize_discrete_actions: bool = True
    # 子任务文本 AR 解码频率：每 N 次 replan 解码一次（0 = 关）；只进回执，不改变 DiT 条件。
    subtask_decode_every: int = 1
    # 无效动作槽位（不进 loss 的 69 维）在去噪中的处理。训练时这些槽位真值恒为 0，模型在 σ 处看到的输入恰为 σ·ε，
    # 但对它们的预测不受监督。"noise_path"：每步把这些槽位的速度固定为 ε₀（= 训练 target ε − 0），使其始终严格
    # 等于 σ·ε₀，与训练输入分布一致，σ = 0 时归零；"free"：整块积分模型预测（cf03efb9 及之前的行为，仅作对照）；
    # "zero"：与 DreamZero 推理相同，初始噪声与每步速度都乘 mask，这些槽位全程为 0（与训练的 σ·ε 不一致，作对照）。
    inactive_action_slots: str = INACTIVE_SLOTS_NOISE_PATH
    # 诊断选项（默认关闭）：输入压缩往返；前 N 次 replan 把模型输入（画布、历史、原始相机帧）存进请求目录。
    # 10-04 闭环诊断：训练视频是 libx264 CRF 23，闭环无损画面导致动作幅度只有训练的约 0.5×、AS 成功率约 25%；
    # 同 checkpoint 加压缩往返后约 85%。AP394 默认开启；"none" 仅作对照。
    input_roundtrip: str = INPUT_ROUNDTRIP_H264_CRF23
    dump_model_input_replans: int = 0
    # 按模型实际输入 / 输出落盘（训练布局：腕部在上 320×224，压缩往返之后）：每次 replan 的当前帧画布、K 张有效历史画布、
    # 预测画布（PNG，登记进回执），以及每个 episode 一条 5 Hz 的模型输入画布 MP4。默认关闭。
    save_model_io: bool = False

    @classmethod
    def from_mapping(cls, options: dict[str, Any] | None) -> AP394BackendOptions:
        options = dict(options or {})
        unknown = sorted(set(options) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown backend options: {unknown}")
        # 权重选择决定评测身份（raw / EMA），不给默认值。
        if "weights" not in options:
            raise ValueError(f"backend option 'weights' is required: {WEIGHTS_RAW!r}, {WEIGHTS_EMA!r} or {WEIGHTS_EMA_SECONDARY!r}")
        parsed = cls(**options)
        if parsed.weights not in (WEIGHTS_RAW, *EMA_VARIANTS):
            raise ValueError(f"weights must be {WEIGHTS_RAW!r}, {WEIGHTS_EMA!r} or {WEIGHTS_EMA_SECONDARY!r}, got {parsed.weights!r}")
        if isinstance(parsed.action_horizon, bool) or parsed.action_horizon != ACTION_HORIZON:
            raise ValueError(f"AP394 action_horizon is frozen at {ACTION_HORIZON}, got {parsed.action_horizon!r}")
        if isinstance(parsed.replan_steps, bool) or parsed.replan_steps != REPLAN_STEPS:
            raise ValueError(f"AP394 replan_steps is frozen at {REPLAN_STEPS}, got {parsed.replan_steps!r}")
        if not math.isfinite(parsed.control_hz) or not math.isfinite(parsed.observation_sample_freq):
            raise ValueError("control_hz and observation_sample_freq must be finite")
        if parsed.control_hz != CONTROL_HZ or parsed.observation_sample_freq != OBSERVATION_HZ:
            raise ValueError(
                f"AP394 backend requires control_hz={CONTROL_HZ:g} and observation_sample_freq={OBSERVATION_HZ:g}"
            )
        if not isinstance(parsed.binarize_discrete_actions, bool):
            raise ValueError(f"binarize_discrete_actions must be a bool, got {parsed.binarize_discrete_actions!r}")
        decode_every = parsed.subtask_decode_every
        if isinstance(decode_every, bool) or not isinstance(decode_every, int) or decode_every < 0:
            raise ValueError(f"subtask_decode_every must be a non-negative int, got {decode_every!r}")
        if parsed.input_roundtrip not in (INPUT_ROUNDTRIP_NONE, INPUT_ROUNDTRIP_H264_CRF23):
            raise ValueError(f"input_roundtrip must be none or h264_crf23, got {parsed.input_roundtrip!r}")
        if not isinstance(parsed.save_model_io, bool):
            raise ValueError(f"save_model_io must be a bool, got {parsed.save_model_io!r}")
        dump = parsed.dump_model_input_replans
        if isinstance(dump, bool) or not isinstance(dump, int) or dump < 0:
            raise ValueError(f"dump_model_input_replans must be a non-negative int, got {dump!r}")
        modes = (INACTIVE_SLOTS_NOISE_PATH, INACTIVE_SLOTS_FREE, INACTIVE_SLOTS_ZERO)
        if parsed.inactive_action_slots not in modes:
            raise ValueError(f"inactive_action_slots must be one of {modes}, got {parsed.inactive_action_slots!r}")
        return parsed

    @property
    def observation_frames_per_replan(self) -> int:
        return int(self.replan_steps * self.observation_sample_freq / self.control_hz)


def zero_inactive_action_slots(denoised: np.ndarray) -> tuple[np.ndarray, float]:
    """[16, 80] 模型输出 -> 非 ACTION_MASK80 槽位置零；返回 (置零后, 置零前非有效槽位的最大绝对值)。

    模型 80 维都有输出，训练 loss 只监督 ACTION_MASK80 的 11 维；其余槽位没有语义，置零后再解码，最大值记入回执。
    """
    denoised = np.asarray(denoised, dtype=np.float32)
    if denoised.shape != (ACTION_HORIZON, CANONICAL_DIM):
        raise ValueError(f"denoised chunk must have shape [{ACTION_HORIZON}, {CANONICAL_DIM}], got {denoised.shape}")
    if not np.isfinite(denoised).all():
        raise ValueError("denoised chunk contains NaN or Inf")
    inactive_max = float(np.abs(denoised[:, ~ACTION_MASK80]).max())
    masked = denoised.copy()
    masked[:, ~ACTION_MASK80] = 0.0
    return masked, inactive_max


def canonical80_to_aap_action12(action80: np.ndarray, binarize_discrete_actions: bool) -> np.ndarray:
    """16 个 20 Hz Canonical80（无效槽位已置零）-> [16, 12] 原始 Action12（训练顺序）float32。

    `aap_action12_to_canonical80` 的逆，逐行映射、不做时间重采样（AP394 原生 20 Hz、16 步）。数值口径与 DZ377 /
    Plan A 的 raw 命令 delta 解码一致：连续通道原值、clip 到 [-1, 1]；torso 恒 0。夹爪按训练映射 g = (cmd + 1) / 2
    取逆 2g - 1。binarize_discrete_actions=True（默认，同 DZ377 / Plan A）时夹爪取符号（等价于 closedness >= 0.5
    -> +1）、control_mode 按符号还原 ±1（0 归手臂模式 -1）；False 时两者都只 clip 到 [-1, 1]。
    """
    action80 = np.asarray(action80, dtype=np.float64)
    if action80.shape != (ACTION_HORIZON, CANONICAL_DIM):
        raise ValueError(f"action chunk must have shape [{ACTION_HORIZON}, {CANONICAL_DIM}], got {action80.shape}")
    if not np.isfinite(action80).all():
        raise ValueError("action chunk contains NaN or Inf")
    if np.any(action80[:, ~ACTION_MASK80] != 0.0):
        raise ValueError("inactive Canonical80 action slots must be zeroed before decoding")
    action12 = np.zeros((ACTION_HORIZON, RAW_ACTION_DIM), dtype=np.float64)
    action12[:, RAW_ACTION_EEF_POSITION_DELTA] = np.clip(action80[:, SLOT_EEF_POSITION], -1.0, 1.0)
    action12[:, RAW_ACTION_EEF_ROTATION_DELTA] = np.clip(action80[:, SLOT_EEF_ROTATION_DELTA], -1.0, 1.0)
    action12[:, 0:2] = np.clip(action80[:, SLOT_BASE_XY], -1.0, 1.0)
    action12[:, 2] = np.clip(action80[:, SLOT_BASE_YAW], -1.0, 1.0)
    action12[:, RAW_ACTION_TORSO] = 0.0
    gripper_command = 2.0 * action80[:, SLOT_GRIPPER] - 1.0
    if binarize_discrete_actions:
        action12[:, RAW_ACTION_GRIPPER] = np.where(gripper_command >= 0.0, 1.0, -1.0)
        action12[:, RAW_ACTION_CONTROL_MODE] = np.where(action80[:, SLOT_CONTROL_MODE] > 0.0, 1.0, -1.0)
        # 与训练读取原始命令时同一组校验（±1 离散量、[-1, 1] 连续量、torso 0）。
        validate_raw_action12(action12)
    else:
        action12[:, RAW_ACTION_GRIPPER] = np.clip(gripper_command, -1.0, 1.0)
        action12[:, RAW_ACTION_CONTROL_MODE] = np.clip(action80[:, SLOT_CONTROL_MODE], -1.0, 1.0)
    return np.ascontiguousarray(action12.astype(np.float32))


def validate_vl_lora_meta(meta: dict[str, Any], lora_rank: int) -> None:
    """checkpoint `vl_lora/meta.json` 必须与本代码的 Qwen 条件布局逐项一致；任何不一致直接拒绝（不降级）。"""
    expected = {
        "text_layout": TEXT_LAYOUT,
        "prompt_version": PROMPT_VERSION,
        "condition_layers": list(CONDITION_LAYERS),
        "summary_layers": list(SUMMARY_LAYERS),
        "history_slots": AP394_HISTORY_SLOTS,
        "history_pool": list(HISTORY_POOL_GRID),
        "lora_rank": int(lora_rank),
    }
    mismatched = {key: (meta.get(key), value) for key, value in expected.items() if meta.get(key) != value}
    if mismatched:
        raise ValueError(f"checkpoint {LORA_DIRECTORY}/meta.json does not match this AP394 code: {mismatched}")


def _as_numpy(value: Any) -> np.ndarray:
    """batch 字段（tensor / tensor 列表 / ndarray）-> ndarray，仅用于诊断落盘。"""
    if isinstance(value, (list, tuple)):
        return np.stack([_as_numpy(item) for item in value])
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy()
    return np.asarray(value)


def _model_canvas_uint8(frame: np.ndarray) -> np.ndarray:
    """模型视频帧 [H, W, 3] float（0–1）-> uint8，与送进 VAE / Qwen 的画布逐像素对应。"""
    return np.ascontiguousarray((np.asarray(frame, dtype=np.float32).clip(0.0, 1.0) * 255.0).round().astype(np.uint8))


def _save_png(path: Path, image: np.ndarray) -> dict[str, str]:
    from PIL import Image

    Image.fromarray(image).save(path)
    return {"path": path.name, "sha256": _sha256_file(path)}


class ModelInputVideoWriter:
    """每个 session 一条 5 Hz MP4：每个新到的观测帧都按训练函数拼成模型输入画布（压缩往返之后）。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.session_id: str | None = None
        self._writer = None
        self.path: Path | None = None

    def write(self, session_id: str, frames: np.ndarray) -> None:
        import imageio_ffmpeg

        if session_id != self.session_id:
            self.close()
            self.root.mkdir(parents=True, exist_ok=True)
            self.path = self.root / f"{session_id}_model_input.mp4"
            height, width = int(frames.shape[1]), int(frames.shape[2])
            self._writer = imageio_ffmpeg.write_frames(str(self.path), (width, height), fps=5, codec="libx264",
                                                       pix_fmt_out="yuv420p", macro_block_size=1, quality=None,
                                                       output_params=["-crf", "18"])
            self._writer.send(None)
            self.session_id = session_id
        for frame in frames:
            self._writer.send(np.ascontiguousarray(frame))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self._writer = None
        self.session_id = None


def h264_crf23_roundtrip(camera_frames: np.ndarray) -> np.ndarray:
    """[T, 3, 256, 256, 3] uint8 -> 每路相机单独以 libx264 CRF 23 / preset medium / yuv420p（5 Hz）编码再解码。"""
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    count, cameras, height, width, _ = camera_frames.shape
    out = np.empty_like(camera_frames)
    for camera in range(cameras):
        stream = np.ascontiguousarray(camera_frames[:, camera])
        encoded = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", "5", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-f", "h264", "-"],
            input=stream.tobytes(), capture_output=True, check=True).stdout
        decoded = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-f", "h264", "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            input=encoded, capture_output=True, check=True).stdout
        frames = np.frombuffer(decoded, dtype=np.uint8)
        if frames.size != count * height * width * 3:
            raise RuntimeError(f"h264 round-trip returned {frames.size // (height * width * 3)} frames, expected {count}")
        out[:, camera] = frames.reshape(count, height, width, 3)
    return out


class AP394RequestPreprocessor:
    """Harness 请求 -> 与训练 `RoboCasaAP394Dataset.__getitem__` 同构的单样本 batch。

    每个 policy server 只服务一个 worker：按 session 缓存三路原始相机帧（key = 20 Hz 观测步号），
    replan 0 清空；历史帧号、画布、prompt、State 全部调用训练函数，不另写一份。
    """

    def __init__(self, codec: AP394PromptCodec, video_size: tuple[int, int], input_roundtrip: str = INPUT_ROUNDTRIP_NONE) -> None:
        self.codec = codec
        self.input_roundtrip = input_roundtrip
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.observation_frames_per_replan = int(REPLAN_STEPS // OBSERVATION_STRIDE)
        self._session: dict[str, Any] | None = None

    def update(
        self,
        session_id: str,
        replan_index: int,
        observation_steps: list[int],
        camera_frames: np.ndarray,
        instruction: str,
    ) -> dict[str, Any]:
        """把本次 replan 的新帧并入 session 缓存；时间线不连续、session 不一致都直接拒绝。"""
        frame_count = int(camera_frames.shape[0])
        if self.input_roundtrip == INPUT_ROUNDTRIP_H264_CRF23:
            camera_frames = h264_crf23_roundtrip(np.asarray(camera_frames, dtype=np.uint8))
        if replan_index == 0:
            # 训练的历史有效性按 episode 绝对帧号（s - 40j >= 0）判定，所以首帧必须是第 0 个控制步。
            if frame_count != 1 or list(observation_steps) != [0]:
                raise ValueError(f"replan 0 requires exactly the step-0 observation, got steps={observation_steps}")
            self._session = {
                "session_id": session_id,
                "instruction": instruction,
                "replan_index": 0,
                "frames": {0: camera_frames[0].copy()},
                "last_step": 0,
            }
            return self._session
        session = self._session
        if session is None or session["session_id"] != session_id:
            raise ValueError(f"Unknown history-window session {session_id!r} at replan {replan_index}")
        if replan_index != int(session["replan_index"]) + 1:
            raise ValueError(f"Non-consecutive replan: expected={int(session['replan_index']) + 1}, got={replan_index}")
        if instruction != session["instruction"]:
            raise ValueError("Instruction changed within one history-window session")
        validate_observation_increment(
            frame_count, list(observation_steps), int(session["last_step"]), self.observation_frames_per_replan
        )
        for index, step in enumerate(observation_steps):
            session["frames"][int(step)] = camera_frames[index].copy()
        current = int(observation_steps[-1])
        # 只保留最远历史（s - 360）及之后的帧，缓存上限 91 帧。
        for step in [step for step in session["frames"] if step < current - HISTORY_REACH_STEPS]:
            del session["frames"][step]
        session["last_step"] = current
        session["replan_index"] = replan_index
        return session

    def build_sample(self, session: dict[str, Any], state16: np.ndarray, instruction: str) -> tuple[dict, dict]:
        """当前帧 s = 最新观测步；按训练规则取历史并构造样本（字段与 `RoboCasaAP394Dataset.__getitem__` 一致）。"""
        current = int(session["last_step"])
        history_indices, history_valid = history_raw_indices(current, AP394_HISTORY_SLOTS, AP394_HISTORY_PERIOD_RAW)
        valid_history = history_indices[history_valid]
        missing = [int(step) for step in valid_history if int(step) not in session["frames"]]
        if missing:
            raise ValueError(f"history frames {missing} required by the training rule are not in the session cache")
        # 与训练 `_read_canvases` 同一组函数：三路相机 -> hand-top 画布 -> 320×224；每帧独立处理，顺序为最旧 -> 当前。
        steps = [int(step) for step in valid_history] + [current]
        views = {
            camera_key: np.stack([session["frames"][step][camera_index] for step in steps])
            for camera_index, camera_key in enumerate(CAMERA_KEYS)
        }
        frames = aap_canvas_to_model_video(compose_eye_in_hand_top_square_canvas(views), self.video_size)
        video = frames[-1:]
        history = np.zeros((AP394_HISTORY_SLOTS, *video.shape[1:]), dtype=np.float32)
        history[history_valid] = frames[:-1]
        task = str(instruction).strip()
        state = state16_to_canonical80(np.asarray(state16, dtype=np.float64)[None]).astype(np.float32)
        # 推理没有动作真值：action 只是占位（形状与训练一致），不进入去噪。
        action = np.zeros((ACTION_HORIZON, CANONICAL_DIM), dtype=np.float32)
        sample = {
            "action": action,
            "action_mask": np.broadcast_to(ACTION_MASK80, action.shape).copy(),
            "state": state,
            "state_mask": np.broadcast_to(STATE_MASK80, state.shape).copy(),
            "text": task,
            "task_instruction": task,
            "subtask": "",
            "has_subtask": False,
            "history_valid": history_valid,
            "history_raw_indices": history_indices,
            "has_real_action": True,
            "dataset_name": "robocasa_ap394",
            "task": task,
            "prompt_version": PROMPT_VERSION,
            "video": np.ascontiguousarray(video),
            "history": history,
            # Qwen 与 VAE 看同一组有效历史帧（最旧 -> 最新）；推理不带子任务答案。
            "vl": self.codec.encode(task, video[0], list(frames[:-1]), subtask=None),
        }
        info = {
            "current_observation_step": current,
            "history_raw_indices": [int(step) for step in history_indices],
            "history_valid": [bool(value) for value in history_valid],
            "history_frames_used": int(len(valid_history)),
            "qwen_images": int(len(valid_history)) + 1,
            "prompt_tokens": int(sample["vl"].prompt_len),
            "condition_text_tokens": int(sample["vl"].condition_pos.numel()),
            "cached_observation_steps": len(session["frames"]),
        }
        return sample, info

    def prepare(self, request: dict[str, Any]) -> tuple[dict, dict]:
        """校验请求 -> 更新缓存 -> 训练同构 batch（经训练的 `ap394_collate_fn`）。"""
        session_id = str(request["session_id"])
        replan_index = int(request["replan_index"])
        observation_steps = [int(value) for value in request["observation_steps"]]
        instruction = str(request["instruction"])
        if CAMERA_FRAMES_BLOCK_FIELD not in request or request[CAMERA_FRAMES_BLOCK_FIELD] is None:
            raise ValueError(
                f"AP394 requires {CAMERA_FRAMES_BLOCK_FIELD!r} (set model.send_raw_camera_frames: true); "
                "the Harness canvas is not the training canvas"
            )
        camera_frames = np.asarray(request[CAMERA_FRAMES_BLOCK_FIELD])
        if camera_frames.dtype != np.uint8 or camera_frames.shape[1:] != CAMERA_FRAME_SHAPE:
            raise ValueError(
                f"{CAMERA_FRAMES_BLOCK_FIELD} must be uint8 [T, {', '.join(map(str, CAMERA_FRAME_SHAPE))}], "
                f"got {camera_frames.dtype} {camera_frames.shape}"
            )
        if int(camera_frames.shape[0]) != len(observation_steps) or not observation_steps:
            raise ValueError(
                f"camera frames and observation steps disagree: frames={camera_frames.shape[0]} "
                f"steps={len(observation_steps)}"
            )
        state16 = np.asarray(request["state"], dtype=np.float32)
        if state16.shape != (RAW_STATE_DIM,) or not np.isfinite(state16).all():
            raise ValueError(f"Expected finite raw state [{RAW_STATE_DIM}], got {state16.shape}")
        session = self.update(session_id, replan_index, observation_steps, camera_frames, instruction)
        sample, info = self.build_sample(session, state16, instruction)
        batch = ap394_collate_fn([sample], pad_id=self.codec.pad_id)
        info["observation_steps_received"] = observation_steps
        return batch, info


def validate_dcp_tensor_coverage(
    state_dict_metadata: dict[str, Any], expected_shapes: dict[str, tuple[int, ...]], source: str
) -> None:
    """DCP 存档（FSDP 分片写出：每个文件 / 每个 rank 只存张量的一块）逐 key 核对：0 缺失、0 多余，全局形状与模型一致，
    且各块元素数之和恰为整个张量（训练写出的分片互不重叠），否则拒绝。

    DCP 读入时只覆盖存档里有的块，没覆盖到的区域会悄悄保留模型原值（预训练权重），所以必须在读之前核对覆盖完整。
    """
    missing = sorted(set(expected_shapes) - set(state_dict_metadata))
    unexpected = sorted(set(state_dict_metadata) - set(expected_shapes))
    if missing or unexpected:
        raise ValueError(
            f"{source} key mismatch: missing={len(missing)} {missing[:5]} unexpected={len(unexpected)} {unexpected[:5]}"
        )
    for key, shape in expected_shapes.items():
        item = state_dict_metadata[key]
        if not isinstance(item, TensorStorageMetadata):
            raise ValueError(f"{source} entry {key!r} is not a tensor")
        if tuple(item.size) != tuple(shape):
            raise ValueError(f"{source} {key} has global shape {tuple(item.size)}, model expects {tuple(shape)}")
        covered = sum(math.prod(chunk.sizes) for chunk in item.chunks)
        if covered != math.prod(shape):
            raise ValueError(f"{source} {key} chunks cover {covered} of {math.prod(shape)} elements")


def load_dit_power_ema(model: torch.nn.Module, ema_dir: Path, sigma_rel: float) -> int:
    """把 `ema/`（DCP 原生 .distcp、FP32、键为训练时 FSDP 模型的 named_parameters）拷进 model；返回 EMA 更新次数。

    键去掉激活检查点包装段 `_checkpoint_wrapped_module.` 后必须与 model 的参数名一一对应（0 缺失、0 多余），各分片
    必须完整覆盖每个张量；σ_rel / γ 必须是请求的主 EMA，`ema/metadata.json` 的更新次数必须与 DCP 内一致。
    """
    reader = FileSystemReader(str(ema_dir))
    state_dict_metadata = reader.read_metadata().state_dict_metadata
    missing_metadata = [key for key in POWER_EMA_METADATA_KEYS if key not in state_dict_metadata]
    if missing_metadata:
        raise ValueError(f"{ema_dir} is not a Power EMA checkpoint: missing {missing_metadata}")
    # 训练保存的 EMA 键 = FSDP 包装后模型的参数名；去掉激活检查点包装段后对应推理模型参数名。
    stored_by_name = {
        key.replace(CHECKPOINT_WRAPPER_PREFIX, ""): key
        for key in state_dict_metadata
        if key not in POWER_EMA_METADATA_KEYS
    }
    if len(stored_by_name) != len(state_dict_metadata) - len(POWER_EMA_METADATA_KEYS):
        raise ValueError("EMA keys collide after stripping the activation-checkpoint wrapper prefix")
    parameters = dict(model.named_parameters())
    validate_dcp_tensor_coverage(
        {name: state_dict_metadata[key] for name, key in stored_by_name.items()},
        {name: tuple(parameter.shape) for name, parameter in parameters.items()},
        f"DiT EMA {ema_dir}",
    )
    state: dict[str, torch.Tensor] = {}
    for key, item in state_dict_metadata.items():
        state[key] = torch.empty(tuple(item.size), dtype=item.properties.dtype)
    dcp.load(state, storage_reader=reader, no_dist=True)
    num_updates = int(state[PowerFunctionEMAManager._NUM_UPDATES_KEY].item())
    loaded_sigma = float(state[PowerFunctionEMAManager._SIGMA_REL_KEY].item())
    loaded_gamma = float(state[PowerFunctionEMAManager._GAMMA_KEY].item())
    if not math.isclose(loaded_sigma, sigma_rel, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{ema_dir} sigma_rel={loaded_sigma}, expected the primary EMA sigma_rel={sigma_rel}")
    if not math.isclose(loaded_gamma, power_function_gamma(sigma_rel), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{ema_dir} gamma={loaded_gamma} is inconsistent with sigma_rel={sigma_rel}")
    if num_updates < 1:
        raise ValueError(f"{ema_dir} has num_updates={num_updates}; an un-updated EMA equals the init weights")
    metadata = json.loads((ema_dir / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata["num_updates"]) != num_updates or not math.isclose(float(metadata["sigma_rel"]), sigma_rel):
        raise ValueError(f"{ema_dir}/metadata.json {metadata} disagrees with the DCP state (num_updates={num_updates})")
    with torch.no_grad():
        for name, parameter in parameters.items():
            shadow = state[stored_by_name[name]]
            if shadow.dtype != torch.float32:
                raise ValueError(f"EMA {name} must be fp32, got {shadow.dtype}")
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))
    return num_updates


class AP394HistoryVLBackend:
    """AP394 checkpoint 的 history-window（无 KV）backend。"""

    # Harness 在 load_backend / PolicyServer 启动时核对这三个声明（docs/BACKEND_PLUGIN.md）。
    history_window = True
    action_horizon = ACTION_HORIZON
    requires_camera_frames_block = True

    def __init__(
        self,
        *,
        runtime_root: Path,
        model_config: Path,
        checkpoint: Path,
        output_root: Path,
        options: AP394BackendOptions,
    ) -> None:
        runtime_root = Path(runtime_root).resolve()
        automodel_root = runtime_root / "Automodel"
        if not automodel_root.is_dir():
            raise FileNotFoundError(f"NeMo_Wan source must contain Automodel: {runtime_root}")
        model_config = Path(model_config).resolve()
        if not model_config.is_file():
            raise FileNotFoundError(f"AP394 model config does not exist: {model_config}")
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_dir() or not (checkpoint / ".complete").is_file():
            raise FileNotFoundError(f"AP394 checkpoint is incomplete: {checkpoint}")
        output_root = Path(output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        self.runtime_root = runtime_root
        self.model_config = model_config
        self.checkpoint = checkpoint
        self.output_root = output_root
        self.options = options
        self.weights = options.weights
        self.action_normalizer, self.state_normalizer = self.domain_normalizers("robocasa")
        self.active_action_mask = ACTION_MASK80.copy()

        # 分配模型前先核对 checkpoint 文件齐全与 Qwen 条件布局戳，避免浪费 GPU 加载时间。
        self._validate_checkpoint_files()
        self.lora_meta = json.loads((checkpoint / LORA_DIRECTORY / "meta.json").read_text(encoding="utf-8"))
        self.recipe, self.cfg = self._build_recipe(automodel_root)
        self._validate_model_contract()
        self.weight_info = self._load_weights()
        dataloader_cfg = self.cfg.data.dataloader
        self.preprocessor = AP394RequestPreprocessor(
            AP394PromptCodec(AutoProcessor.from_pretrained(str(dataloader_cfg.processor_path))),
            tuple(int(value) for value in dataloader_cfg.video_size),
            input_roundtrip=options.input_roundtrip,
        )
        self.model_input_video = ModelInputVideoWriter(self.output_root / "model_input_videos") if options.save_model_io else None

    def domain_normalizers(self, domain: str) -> tuple[CommandActionNormalizer, CanonicalStateNormalizer]:
        """严格按checkpoint归档合同加载各域统计，禁止回退到单域旧统计。"""
        contract = json.loads((self.checkpoint / "dataset_statistics/five_domain/adapter_contract.json").read_text())
        if contract["schema_version"] != 18 or domain not in ("libero", "robocasa"):
            raise ValueError("Five-domain checkpoint schema/domain mismatch")
        # AP395的RC统计缺失于checkpoint；显式使用已冻结且留有来源SHA的评测合同。
        if domain == "robocasa":
            contract["sources"][domain] = json.loads(Path(os.environ["AWOMO_ADDITIONAL_STATS_CONTRACT"]).read_text())[domain]
        sources = contract["sources"][domain]
        if len(sources) != 2:
            raise ValueError("Expected exactly action/state statistics for closed-loop domain")
        paths = []
        for item in sources:
            path = self.checkpoint / item["path"]
            if _sha256_file(path) != item["sha256"]:
                raise ValueError(f"Five-domain {domain} statistics SHA mismatch")
            paths.append(path)
        action_doc, state_doc = [json.loads(path.read_text()) for path in paths]
        if action_doc["action_codec"] != "command-rotation6d-signed/v2":
            raise ValueError("Five-domain action codec mismatch")
        if state_doc["state_codec"] != "canonical-absolute-rotation6d/v1":
            raise ValueError("Five-domain state codec mismatch")
        if action_doc["domain"] != domain or state_doc["domain"] != domain:
            raise ValueError("Five-domain statistics domain mismatch")
        action = CommandActionNormalizer(str(paths[0]), tuple(action_doc["continuous_slots"]),
                                         tuple(action_doc["discrete_slots"]))
        state = CanonicalStateNormalizer(str(paths[1]), tuple(state_doc["active_dimensions"]))
        return action, state

    # ---- 构建 ---------------------------------------------------------------------------------

    def _validate_checkpoint_files(self) -> None:
        """raw 需要 model/ + vl_lora/adapter；EMA 需要 model/（buffer 与 key 核对）+ ema/ + vl_lora/ema。"""
        required = [
            self.checkpoint / "model",
            self.checkpoint / LORA_DIRECTORY / "meta.json",
            self.checkpoint / LORA_DIRECTORY / "adapter.safetensors",
        ]
        if self.weights in EMA_VARIANTS:
            ema_directory, _, lora_file = EMA_VARIANTS[self.weights]
            required += [
                self.checkpoint / ema_directory / ".metadata",
                self.checkpoint / ema_directory / "metadata.json",
                self.checkpoint / LORA_DIRECTORY / lora_file,
            ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"AP394 checkpoint is missing {missing} for weights={self.weights!r}")

    def _build_recipe(self, automodel_root: Path) -> tuple[Any, Any]:
        """实例化训练 recipe 本身（同一个 `_build_models` / `_encode_batch` / `_model_forward`），只做推理 setup。"""
        if str(automodel_root) not in sys.path:
            sys.path.insert(0, str(automodel_root))
        # 训练 YAML 的输出目录等字段读环境变量；推理只需要能解析，指向本次评测输出目录。
        os.environ.setdefault("AP394_RUN_ID", f"robocasa-closeloop-{BACKEND_NAME}")
        os.environ.setdefault("AP394_CHECKPOINT_DIR", str(self.checkpoint.parent))
        os.environ.setdefault("AP394_EVAL_DIR", str(self.output_root))
        os.environ.setdefault("AP394_LOG_DIR", str(self.output_root / "model-logs"))
        # 续训 checkpoint 的 config.yaml 还引用 AP394_RESUME_FROM 等变量（r05 / r06 起）；YAML 加载时就要解析，早于下面的
        # --checkpoint.restore_from None 覆盖。凡是 config 里没有默认值的 ${oc.env:VAR} 一律补占位值（推理不使用）。
        for name in sorted(set(re.findall(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}", Path(self.model_config).read_text()))):
            os.environ.setdefault(name, f"unused-at-inference:{name}")
        # 训练镜像没有 Transformer Engine；推理侧强制同一路径（与 DZ377 插件相同），因此 recipe 在此之后才 import。
        sys.modules["transformer_engine"] = None

        from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
        from nemo_automodel.recipes.diffusion.train_action_patch_one_domain_robotwin_dp32 import (
            TrainActionPatchOneDomainRobotWinDp32Recipe,
        )

        # restore_from 置空：setup_for_inference 不加载权重，权重由 _load_weights 按 raw / EMA 显式加载并核对 key。
        overrides = [
            "--checkpoint.restore_from",
            "None",
            "--fsdp.enabled",
            "false",
            "--fsdp.activation_checkpointing",
            "false",
            "--action_patch.vl_history.history_vae_micro_batch",
            "8",
        ]
        cfg = parse_args_and_load_config(str(self.model_config), argv=overrides)
        recipe = TrainActionPatchOneDomainRobotWinDp32Recipe(cfg)
        # 与已验证Molmo入口一致：CUDA搬运时直接使用BF16，避免5090临时FP32副本。
        from types import MethodType
        def setup_single_gpu(instance, fsdp_cfg):
            instance.device_mesh = None
            instance.__dict__["model"] = instance.model.to(device=instance.device, dtype=torch.bfloat16)
        recipe._setup_fsdp = MethodType(setup_single_gpu, recipe)
        recipe.setup_for_inference(build_dataloader=False)
        return recipe, cfg

    def _validate_model_contract(self) -> None:
        """aap338 三域 fused-VAE 合同：action_dim 80、history_slots 9、16 步去噪、LoRA 元数据一致。"""
        cfg = self.cfg
        if int(cfg.inference.num_inference_steps) != NUM_INFERENCE_STEPS:
            raise ValueError(f"inference.num_inference_steps must be {NUM_INFERENCE_STEPS}")
        if int(cfg.action_patch.dit.action_dim) != CANONICAL_DIM or int(self.recipe.model_action_dim) != CANONICAL_DIM:
            raise ValueError(f"AP394 action_dim must be {CANONICAL_DIM}")
        if int(self.recipe.history_slots) != AP394_HISTORY_SLOTS:
            raise ValueError(f"AP394 history_slots must be {AP394_HISTORY_SLOTS}")
        conditioner = self.recipe.vl_conditioner
        validate_vl_lora_meta(self.lora_meta, conditioner.lora_rank)
        if int(self.lora_meta["lora_tensors"]) != len(conditioner.lora_parameters()):
            raise ValueError(
                f"checkpoint LoRA has {self.lora_meta['lora_tensors']} tensors, model has "
                f"{len(conditioner.lora_parameters())}"
            )

    def _load_weights(self) -> dict[str, Any]:
        """raw：model/ + vl_lora/adapter；EMA：在 model/ 之上覆盖 ema/（全部参数）+ vl_lora/ema。DiT 与 LoRA 一起切换。"""
        recipe = self.recipe
        model_path = self.checkpoint / "model"
        # 与训练加载器同一分支：DCP 分片 safetensors 用 HF reader，否则用默认 DCP reader。
        if _is_safetensors_checkpoint(str(model_path)):
            reader = _HuggingFaceStorageReader(path=str(model_path))
        else:
            reader = FileSystemReader(str(model_path))
        # 训练 `Checkpointer.save_model` 用 `_HuggingFaceStorageWriter(save_sharded=True)` 写出：每个
        # shard-*.safetensors 只含各 rank 的一块，块的位置在文件头 DCP_SHARDING_INFO 的 saved_offsets 里；
        # reader 把同一 key 的各块汇成全局张量的 chunk 列表。读前核对 key 与覆盖完整，读时由 DCP 按块拼回。
        model_shapes = {key: tuple(value.shape) for key, value in recipe.model.state_dict().items()}
        validate_dcp_tensor_coverage(reader.read_metadata().state_dict_metadata, model_shapes, f"DiT {model_path}")
        recipe._load_model_weights_only(str(self.checkpoint), recipe.cfg.checkpoint, no_dist=True)
        conditioner = recipe.vl_conditioner
        info: dict[str, Any] = {
            "weights": self.weights,
            "dit_source": "model/",
            "dit_keys": len(model_shapes),
            "lora_tensors": len(conditioner.lora_parameters()),
        }
        if self.weights == WEIGHTS_RAW:
            conditioner.load_lora_state_dict(load_file(str(self.checkpoint / LORA_DIRECTORY / "adapter.safetensors")))
            info["lora_source"] = f"{LORA_DIRECTORY}/adapter.safetensors"
            return info
        ema_directory, sigma_rel, lora_file = EMA_VARIANTS[self.weights]
        dit_updates = load_dit_power_ema(recipe.model, self.checkpoint / ema_directory, sigma_rel)
        # LoRA EMA 用训练同一个 PowerFunctionEMAManager 读取：严格 key、σ_rel / γ 核对，再拷进 LoRA 参数。
        lora_ema = PowerFunctionEMAManager(conditioner, sigma_rel=sigma_rel)
        lora_ema.load_state_dict(load_file(str(self.checkpoint / LORA_DIRECTORY / lora_file)), strict=True)
        if lora_ema.num_updates != dit_updates:
            raise ValueError(f"LoRA EMA num_updates={lora_ema.num_updates} != DiT EMA num_updates={dit_updates}")
        lora_ema.copy_to_model(conditioner)
        info.update(
            {
                "dit_source": f"{ema_directory}/ (Power EMA sigma_rel {sigma_rel})",
                "lora_source": f"{LORA_DIRECTORY}/{lora_file}",
                "ema_num_updates": dit_updates,
            }
        )
        return info

    # ---- 模型 -------------------------------------------------------------------------------

    @torch.no_grad()
    def _encode_with_prefill_cache(self, batch: dict) -> tuple[dict, Any, torch.Tensor]:
        """与 `recipe._encode_batch` 同一组成，只是 Qwen prefill 打开 KV cache，供之后的子任务解码复用。

        Qwen 前向参数与 `HistoryVLConditioner.run_language_model` 相同（仅 use_cache=True，不改变 prefill 数值）；
        DiT 条件仍由训练的 `gather_condition` 取，图像 / 历史编码直接调用 recipe 的 `_encode_video_groups` /
        `_encode_history`。返回 (encoded, prefill KV cache, prompt 末位第 36 层隐状态)。单测逐元素核对它与
        `_encode_batch` 的输出相同。
        """
        recipe = self.recipe
        conditioner = recipe.vl_conditioner
        recipe._ensure_encoders_on_device()
        vb = {key: value.to(recipe.device, non_blocking=True) for key, value in batch["vl"].items()}
        outputs = conditioner.qwen.model(
            input_ids=vb["input_ids"],
            attention_mask=vb["attention_mask"],
            pixel_values=vb["pixel_values"],
            image_grid_thw=vb["image_grid_thw"],
            mm_token_type_ids=vb["mm_token_type_ids"],
            output_hidden_states=True,
            use_cache=True,
        )
        recorded = {
            layer: outputs.hidden_states[layer] for layer in sorted(set(CONDITION_LAYERS) | set(SUMMARY_LAYERS))
        }
        context, context_mask = conditioner.gather_condition(recorded, vb)
        encoded = recipe._encode_video_groups(batch)
        history_tokens, history_valid = recipe._encode_history(batch)
        encoded["context"] = context.to(torch.bfloat16)
        encoded["text_mask"] = context_mask
        encoded["history"] = history_tokens
        encoded["history_valid"] = history_valid
        encoded["history_ids"] = history_image_ids(recipe.history_slots, HISTORY_POOL_GRID, recipe.device)
        return encoded, outputs.past_key_values, recorded[conditioner.final_layer][:, -1]

    @torch.no_grad()
    def _decode_subtask(self, cache: Any, last_hidden: torch.Tensor) -> dict[str, Any]:
        """在 prefill KV cache 上贪心解码子任务句子，直到训练的结束符 `<|im_end|>` 或 token 上限。

        与训练子任务 CE 同一读出：冻结 lm_head 读第 36 层（最终 RMSNorm 之后）；LoRA 权重即当前加载的 raw / EMA。
        解码只发生在 DiT 条件取完之后，结果只写回执。
        """
        codec = self.preprocessor.codec
        qwen = self.recipe.vl_conditioner.qwen
        started = time.perf_counter()
        logits = qwen.lm_head(last_hidden)
        token_ids: list[int] = []
        stop_reason = "max_new_tokens"
        for _ in range(SUBTASK_MAX_NEW_TOKENS):
            token = int(logits.argmax(dim=-1)[0])
            token_ids.append(token)
            if token == codec.im_end_id:
                stop_reason = "im_end"
                break
            # 单 token 增量前向：无图像输入时 Qwen3-VL 用 prefill 记下的 rope_deltas 续接 M-RoPE 位置。
            step = qwen.model(
                input_ids=torch.tensor([[token]], device=last_hidden.device),
                past_key_values=cache,
                use_cache=True,
            )
            logits = qwen.lm_head(step.last_hidden_state[:, -1])
        text_ids = [token for token in token_ids if token != codec.im_end_id]
        return {
            "subtask_text": codec.tokenizer.decode(text_ids, skip_special_tokens=False).strip(),
            "subtask_token_ids": token_ids,
            "subtask_stop_reason": stop_reason,
            "subtask_decode_seconds": time.perf_counter() - started,
        }

    @torch.no_grad()
    def _predict(
        self, batch: dict, seed: int, decode_subtask: bool
    ) -> tuple[np.ndarray, torch.Tensor, tuple[int, int], dict[str, Any]]:
        """训练同一编码与 DiT 前向；从纯噪声联合去噪图像与动作。

        返回 (动作 [16, 80] 未置零, 图像 token, 网格, 子任务解码记录)。subtask_decode_every=0 时直接走训练的
        `recipe._encode_batch`；否则整次运行都走带 KV cache 的同组成 prefill（单测证明两者 DiT 条件与动作逐元素相同）。
        """
        batch = dict(batch)
        batch.setdefault("domain", "robocasa")
        if batch["domain"] not in ("libero", "robocasa", "robotwin"):
            raise ValueError("This evaluation adapter supports LIBERO and RoboCasa")
        recipe = self.recipe
        recipe.model.eval()
        recipe.vl_conditioner.eval()
        vl = batch["vl"]
        prompt_len = int(vl["prompt_len"][0])
        # DiT 条件只取 prompt 内的位置（布局句 + 任务文本、summary = prompt_len - 1），推理序列只含 prompt；
        # 因果注意力下之后解码的子任务 token 对这些位置不可见，解码不会改变 DiT 条件。
        if int(vl["input_ids"].shape[1]) != prompt_len:
            raise RuntimeError("inference Qwen input must be exactly the prompt (no subtask tokens)")
        if int(vl["condition_pos"].max()) >= prompt_len or int(vl["summary_pos"][0]) != prompt_len - 1:
            raise RuntimeError("DiT condition positions must lie inside the prompt, summary at prompt_len - 1")
        encode_started = time.perf_counter()
        if self.options.subtask_decode_every == 0:
            encoded = recipe._encode_batch(batch)
            cache = None
            last_hidden = None
        else:
            encoded, cache, last_hidden = self._encode_with_prefill_cache(batch)
        timing = {"encode_seconds": time.perf_counter() - encode_started}
        if encoded["sample_groups"] != [[0]]:
            raise RuntimeError(f"unexpected encode grouping {encoded['sample_groups']}")
        target = encoded["target"][0]
        image_grid = encoded["image_grid"][0]
        indices = torch.tensor([0], device=recipe.device)
        states = torch.stack([batch["state"][0]]).to(recipe.device, torch.bfloat16)[:, 0]
        # 与父模型 AAP 离线推理 `_denoise_group` 同一采样器与 RNG 次序（先图像噪声、后动作噪声）。
        # 无效动作槽位：inactive_action_slots="noise_path" 时每步把其速度固定为初始噪声 ε₀（训练 target = ε − 0），
        # 使输入始终是训练分布 σ·ε₀；"free" 时整块积分模型预测（父路径 action_patch_offline_action.py 的行为）；
        # "zero" 时初始值与每步速度都置 0（DreamZero 推理做法）。RNG 次序三种模式相同。
        # 两种模式最终都按 ACTION_MASK80 置零后解码。
        denoise_started = time.perf_counter()
        init_all_rng(int(seed))
        scheduler_cfg = recipe.cfg.action_patch.scheduler
        scheduler = FlowMatchScheduler(
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_train_timesteps=int(scheduler_cfg.num_train_timesteps),
            shift=float(scheduler_cfg.shift),
        )
        scheduler.set_timesteps(num_inference_steps=NUM_INFERENCE_STEPS, training=False)
        timesteps = scheduler.timesteps.to(recipe.device)
        noisy_image = torch.randn_like(target)
        noisy_action = torch.randn((1, ACTION_HORIZON, CANONICAL_DIM), device=recipe.device, dtype=target.dtype)
        inactive = torch.from_numpy(~self.active_action_mask).to(recipe.device)
        mode = self.options.inactive_action_slots
        if mode == INACTIVE_SLOTS_ZERO:
            noisy_action = torch.where(inactive, torch.zeros_like(noisy_action), noisy_action)
        initial_action_noise = noisy_action.clone()
        for timestep_index, timestep in enumerate(timesteps):
            expanded = timestep.to(dtype=target.dtype).expand(1)
            image_prediction, action_prediction = recipe._model_forward(
                encoded, indices, 0, noisy_image, noisy_action, states, expanded
            )
            if mode == INACTIVE_SLOTS_NOISE_PATH:
                action_prediction = torch.where(inactive, initial_action_noise.to(action_prediction.dtype), action_prediction)
            elif mode == INACTIVE_SLOTS_ZERO:
                action_prediction = torch.where(inactive, torch.zeros_like(action_prediction), action_prediction)
            to_final = timestep_index == len(timesteps) - 1
            noisy_image = scheduler.step(image_prediction, timestep, noisy_image, to_final=to_final)
            noisy_action = scheduler.step(action_prediction, timestep, noisy_action, to_final=to_final)
        action = noisy_action[0].float().cpu().numpy()
        timing["denoise_seconds"] = time.perf_counter() - denoise_started
        subtask = {"subtask_decoded": False}
        if decode_subtask:
            subtask = {"subtask_decoded": True, **self._decode_subtask(cache, last_hidden)}
        del cache
        return action, noisy_image, image_grid, {**timing, **subtask}

    @torch.no_grad()
    def _decode_predicted_image(self, image_tokens: torch.Tensor, image_grid: tuple[int, int]) -> np.ndarray:
        """干净的 target 图像 token [1, h*w, 128] -> FLUX VAE 解码 -> uint8 [1, 320, 224, 3]（训练画布，模型分辨率）。"""
        height, width = int(image_grid[0]), int(image_grid[1])
        latent = image_tokens.transpose(1, 2).reshape(1, 128, height, width).unsqueeze(2)
        decoded = self.recipe.vae.decode(latent)
        if decoded.ndim != 5 or int(decoded.shape[2]) != 1 or not torch.isfinite(decoded).all():
            raise ValueError(f"predicted image decode must be finite [1,3,1,H,W], got {tuple(decoded.shape)}")
        frame = ((decoded[0, :, 0].float() + 1.0) * 127.5).clamp(0, 255).round()
        return np.ascontiguousarray(frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)[None])

    # ---- Harness 接口 -------------------------------------------------------------------------

    def infer_history_window(self, request: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        request_id = str(request["request_id"])
        session_id = str(request["session_id"])
        if not REQUEST_ID_PATTERN.fullmatch(request_id) or not REQUEST_ID_PATTERN.fullmatch(session_id):
            raise ValueError("Unsafe request or session ID")
        replan_index = int(request["replan_index"])
        seed = int(request["seed"])
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        capture_image = bool(request.get("capture_predicted_video", False))
        request_root = self.output_root / "requests" / request_id
        request_root.mkdir(parents=True, exist_ok=False)

        started_at = time.perf_counter()
        batch, input_info = self.preprocessor.prepare(request)
        anchor_step = int(input_info["current_observation_step"])
        model_io: dict[str, Any] = {}
        if self.options.save_model_io:
            # 模型实际输入：当前帧画布 + 有效历史画布（与 batch 逐像素一致）；文件名带槽位号与 20 Hz 帧号。
            model_io["input_current"] = _save_png(request_root / f"model_input_current_f{anchor_step}.png",
                                                  _model_canvas_uint8(_as_numpy(batch["video"])[0][0]))
            history = _as_numpy(batch["history"])[0]
            valid = _as_numpy(batch["history_valid"])[0].astype(bool)
            for slot, (frame_step, ok) in enumerate(zip(input_info["history_raw_indices"], valid)):
                if ok:
                    model_io[f"input_history_slot{slot}"] = _save_png(
                        request_root / f"model_input_history_slot{slot}_f{int(frame_step)}.png", _model_canvas_uint8(history[slot]))
            session = self.preprocessor._session
            new_steps = [int(step) for step in request["observation_steps"]]
            views = {camera_key: np.stack([session["frames"][step][index] for step in new_steps])
                     for index, camera_key in enumerate(CAMERA_KEYS)}
            canvases = aap_canvas_to_model_video(compose_eye_in_hand_top_square_canvas(views), self.preprocessor.video_size)
            self.model_input_video.write(session_id, np.stack([_model_canvas_uint8(frame) for frame in canvases]))
        if replan_index < self.options.dump_model_input_replans:
            # 诊断：存下模型实际看到的输入（训练同构 batch 的画布 / 历史），以及 Harness 送来的原始相机帧。
            np.savez_compressed(
                request_root / "model_input.npz",
                video=_as_numpy(batch["video"]).astype(np.float16),
                history=_as_numpy(batch["history"]).astype(np.float16),
                history_valid=_as_numpy(batch["history_valid"]),
                camera_frames_block=np.asarray(request[CAMERA_FRAMES_BLOCK_FIELD], dtype=np.uint8),
                state=np.asarray(request["state"], dtype=np.float32),
            )
        decode_every = self.options.subtask_decode_every
        decode_subtask = decode_every > 0 and replan_index % decode_every == 0
        # 原始State80已由训练同一映射产生，再按该域q01/q99处理。
        batch["state"] = [torch.from_numpy(self.state_normalizer.normalize(value.cpu().numpy())) for value in batch["state"]]
        raw_output, image_tokens, image_grid, model_info = self._predict(batch, seed, decode_subtask)
        denoised, inactive_max = zero_inactive_action_slots(raw_output)
        actions = self.action_normalizer.decode(denoised, "robocasa")
        if actions.shape != (self.action_horizon, RAW_ACTION_DIM):
            raise RuntimeError(f"decoded action must be [{self.action_horizon}, {RAW_ACTION_DIM}], got {actions.shape}")
        predicted_artifacts = None
        predicted_timestamps: list[float] = []
        if capture_image:
            predicted_timestamps = [(anchor_step + PREDICTED_IMAGE_STEP_OFFSET) / CONTROL_HZ]
            # 模型侧 sidecar（XScope 按 request_id 合并）；Harness 的 PredictedVideo 合同固定 8 帧，单帧不上线。
            image_path = request_root / "predicted_image.npy"
            timestamp_path = request_root / "predicted_image_timestamps.npy"
            np.save(image_path, self._decode_predicted_image(image_tokens, image_grid), allow_pickle=False)
            np.save(timestamp_path, np.asarray(predicted_timestamps, dtype=np.float64), allow_pickle=False)
            predicted_artifacts = {
                path.stem: {"path": path.name, "sha256": _sha256_file(path)} for path in (image_path, timestamp_path)
            }
        if self.options.save_model_io:
            predicted = self._decode_predicted_image(image_tokens, image_grid)[0]
            model_io["output_predicted"] = _save_png(request_root / f"model_output_predicted_f{anchor_step + PREDICTED_IMAGE_STEP_OFFSET}.png", predicted)
            if self.model_input_video is not None and self.model_input_video.path is not None:
                model_io["session_model_input_video"] = {"path": str(self.model_input_video.path.relative_to(self.output_root))}
        latency_seconds = time.perf_counter() - started_at

        raw_path = request_root / f"model_action_raw_output_{ACTION_HORIZON}x{CANONICAL_DIM}.npy"
        denoised_path = request_root / f"model_action_denoised_slots_{ACTION_HORIZON}x{CANONICAL_DIM}.npy"
        native_path = request_root / "native_action.npy"
        np.save(raw_path, raw_output.astype(np.float32), allow_pickle=False)
        np.save(denoised_path, denoised, allow_pickle=False)
        np.save(native_path, actions, allow_pickle=False)
        diagnostics = {
            "backend": BACKEND_NAME,
            "inference_contract_id": INFERENCE_CONTRACT_ID,
            "inference_mode": INFERENCE_MODE,
            "training_adapter_id": AAP_ADAPTER_ID,
            "text_layout": TEXT_LAYOUT,
            "prompt_version": PROMPT_VERSION,
            "request_id": request_id,
            "session_id": session_id,
            "replan_index": replan_index,
            "seed": seed,
            "instruction": str(request["instruction"]),
            "state": np.asarray(request["state"], dtype=np.float32).tolist(),
            "checkpoint": str(self.checkpoint),
            "expected_training_commit": self.options.expected_training_commit,
            **self.weight_info,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "scheduler_shift": float(self.cfg.action_patch.scheduler.shift),
            "latency_seconds": latency_seconds,
            "camera_input": f"{CAMERA_FRAMES_BLOCK_FIELD} {list(HARNESS_CAMERA_ORDER)}; Harness canvas unused",
            "model_input_canvas": "eye_in_hand_top_square 384x256 -> 320x224 (training functions)",
            **input_info,
            **model_info,
            "kv_reused_from_previous_replan": False,
            "subtask_decode_every": decode_every,
            "inactive_action_slots": self.options.inactive_action_slots,
            "input_roundtrip": self.options.input_roundtrip,
            "binarize_discrete_actions": self.options.binarize_discrete_actions,
            # 统一训练直接预测signed夹爪；记录阈值前值，禁止再次做2g-1。
            "gripper_command_pre_binarization": denoised[:, SLOT_GRIPPER].tolist(),
            "control_mode_pre_binarization": denoised[:, SLOT_CONTROL_MODE].tolist(),
            "normalization_contract": "checkpoint per-domain action/state q01q99 after Rotation6D; gripper/control_mode signed",
            "prediction_shape": list(actions.shape),
            "native_action_order": "raw_training_action12",
            "action_sample_freq": CONTROL_HZ,
            "temporal_resampling": "none (native 20 Hz)",
            "prediction_horizon_steps": ACTION_HORIZON,
            "prediction_horizon_sec": ACTION_HORIZON / CONTROL_HZ,
            "replan_steps": REPLAN_STEPS,
            "observation_sample_freq": OBSERVATION_HZ,
            "observation_stride_steps": OBSERVATION_STRIDE,
            "action_start_observation_step": anchor_step,
            "action_engine_steps": [anchor_step + index for index in range(ACTION_HORIZON)],
            "model_action_active_slots": np.flatnonzero(ACTION_MASK80).tolist(),
            "model_action_inactive_slots_max_abs_before_zeroing": inactive_max,
            "model_action_artifacts": {
                "model_action_raw_output": {"path": raw_path.name, "sha256": _sha256_file(raw_path)},
                "model_action_denoised_slots": {"path": denoised_path.name, "sha256": _sha256_file(denoised_path)},
                "native_action": {"path": native_path.name, "sha256": _sha256_file(native_path)},
            },
            "predicted_video_capture": capture_image,
            "predicted_video_on_wire": False,
            "predicted_image_semantics": "training target frame: reference + 16 control steps (+0.8 s), 320x224",
            "predicted_image_timestamps_sec": predicted_timestamps,
            "predicted_image_artifacts": predicted_artifacts,
            "model_io_artifacts": model_io,
            "model_io_layout": "training canvas eye_in_hand_top 384x256 -> 320x224, after input_roundtrip" if model_io else None,
        }
        (request_root / "receipt.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return actions, diagnostics

    def close(self) -> None:
        writer = getattr(self, "model_input_video", None)
        if writer is not None:
            writer.close()
        recipe = getattr(self, "recipe", None)
        if recipe is not None:
            for name in ("model", "vae", "vl_conditioner", "text_encoder"):
                if name in recipe.__dict__:
                    recipe.__dict__[name] = None
            self.recipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create(context: Any) -> AP394HistoryVLBackend:
    """VLA Harness `--backend-factory` / entry point `ap394-history-vl` 入口；`context` 是 Harness 的 BackendContext。"""
    runtime_root = context.runtime_root
    checkpoint = context.checkpoint
    output_root = context.output_root
    if runtime_root is None or checkpoint is None or output_root is None:
        raise ValueError("AP394 backend requires --runtime-root, --checkpoint and --output-root")
    model_config = context.model_config
    if model_config is None:
        model_config = Path(runtime_root) / DEFAULT_MODEL_CONFIG
    options = AP394BackendOptions.from_mapping(context.options)
    return AP394HistoryVLBackend(
        runtime_root=Path(runtime_root),
        model_config=Path(model_config),
        checkpoint=Path(checkpoint),
        output_root=Path(output_root),
        options=options,
    )


__all__ = [
    "BACKEND_NAME",
    "INFERENCE_CONTRACT_ID",
    "INFERENCE_MODE",
    "AP394BackendOptions",
    "AP394HistoryVLBackend",
    "AP394RequestPreprocessor",
    "canonical80_to_aap_action12",
    "create",
    "load_dit_power_ema",
    "validate_dcp_tensor_coverage",
    "validate_vl_lora_meta",
    "zero_inactive_action_slots",
]
