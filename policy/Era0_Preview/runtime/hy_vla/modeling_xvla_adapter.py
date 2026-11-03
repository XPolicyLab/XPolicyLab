"""X-VLA implementation behind the established Hy-VLA batch contract."""

from __future__ import annotations

import gc
import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from XPolicyLab.policy.Era0_Preview.runtime.models.configuration_xvla import XVLAConfig
from XPolicyLab.policy.Era0_Preview.runtime.models.architecture_contract import is_vlm_fast_action_bridge_version
from XPolicyLab.policy.Era0_Preview.runtime.models.modeling_xvla import XVLA
from XPolicyLab.policy.Era0_Preview.runtime.models.processing_xvla import XVLAProcessor, apply_action_language_max_length


_IMAGE_KEYS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)
_IMAGE_HISTORY_MASK_KEY = "observation.images.history_mask"
_IMAGE_HISTORY_SIZE_KEY = "observation.images.history_size"
_STATE_KEY = "observation.state"
_ACTION_KEY = "action"
_ADAPTER_MANIFEST = "hy_xvla_adapter.json"
_EE6D_ACTION_DIM = 20
_HY_ACTION_DIM = 32
_CROSS_EMB_EE_INDICES = tuple(range(0, 10)) + tuple(range(30, 40))
_FLORENCE_XVLA_POLICY_IMAGE_SIZE = (224, 224)
_HY_ROBODOJO_ACTION_CONTRACT = (
    "hy_robodojo_ee6d20_rt_relative_rows_zscore_flow_matching_v1"
)
_CROSS_EMB_ACTION_TRANSFER_MODES = frozenset(
    {"auto", "reset", "copy_compatible", "preserve", "pretrained"}
)
_ACTION_LOSS_PROFILES = frozenset(
    {"hy_normalized_mse", "official_ee6d_regression"}
)
_ACTION_LOSS_COMPONENTS = (
    "position_loss",
    "rotate6D_loss",
    "gripper_loss",
    "regression_loss",
    "flow_loss",
)
_QWEN_TIMESTAMP_CAMERA_NAMES = {
    0: "head",
    1: "left_wrist",
    2: "right_wrist",
}
_QWEN_TIMESTAMP_MIN_RELATIVE_SLOT = -64
_QWEN_TIMESTAMP_MAX_RELATIVE_SLOT = 0
_logger = logging.getLogger(__name__)


def _install_qwen_timestamp_tokens(config: XVLAConfig, tokenizer) -> None:
    """Cache camera-qualified relative-time text ids for multimodal packing.

    The backend packs timestamps immediately before each image block. Keeping
    tokenization here avoids coupling the backend to a processor/tokenizer
    object while still allowing old checkpoints to gain the new runtime-only
    metadata when loaded through the adapter.
    """
    if str(getattr(config, "vlm_backend", "")).strip().lower() != "qwen3_5":
        return
    token_ids: dict[str, list[int]] = {}
    for raw_view_idx in range(8):
        camera = _QWEN_TIMESTAMP_CAMERA_NAMES.get(
            raw_view_idx, f"view_{raw_view_idx}"
        )
        for relative_slot in range(
            _QWEN_TIMESTAMP_MIN_RELATIVE_SLOT,
            _QWEN_TIMESTAMP_MAX_RELATIVE_SLOT + 1,
        ):
            text = f"[camera={camera}] [time={relative_slot}] "
            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
            token_ids[f"{raw_view_idx}:{relative_slot}"] = list(encoded)
    config.qwen_timestamp_token_ids = token_ids


def _is_raw_qwen35_checkpoint(source: str) -> bool:
    """Return whether ``source`` is a native Qwen3.5 (not XVLA) directory."""
    config_path = Path(source) / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    return str(payload.get("model_type", "")).strip().lower() == "qwen3_5"


def _is_raw_dm05_checkpoint(source: str) -> bool:
    """Return whether ``source`` is a native DM0.5 checkpoint directory."""
    config_path = Path(source) / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    return str(payload.get("model_type", "")).strip().lower() in {"dm05", "dm0.5", "dm_05"}


def _coerce_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _normalize_cross_emb_action_transfer(value: Any) -> str:
    if value is None:
        return "auto"
    normalized = str(value).strip().lower()
    aliases = {
        "": "auto",
        "auto": "auto",
        "reset": "reset",
        "random": "reset",
        "random_init": "reset",
        "copy_compatible": "copy_compatible",
        "copy-compatible": "copy_compatible",
        # Explicitly retain a source CrossEmb action head and its norm state.
        # This is intended for mixed-pretraining checkpoints whose native
        # 64-wide CrossEmb layout already contains the RoboDojo EE slots.
        "preserve": "preserve",
        "native": "preserve",
        # Strictly reuse the complete pretrained CrossEmb action head.  This
        # name is intentionally explicit for launchers that want to guarantee
        # no action-I/O random initialization; it is equivalent to preserve.
        "pretrained": "pretrained",
        "pretrained_weights": "pretrained",
        "use_pretrained": "pretrained",
        # Saved non-CrossEmb checkpoints record this informational value.
        # It has no transfer behavior and should resolve to the default.
        "not_applicable": "auto",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _CROSS_EMB_ACTION_TRANSFER_MODES:
        allowed = ", ".join(sorted(_CROSS_EMB_ACTION_TRANSFER_MODES))
        raise ValueError(
            "cross_emb_action_transfer must be one of "
            f"{allowed}, got {value!r}"
        )
    return normalized


def _normalize_action_loss_profile(value: Any) -> str:
    normalized = str(value or "hy_normalized_mse").strip().lower()
    aliases = {
        "": "hy_normalized_mse",
        "default": "hy_normalized_mse",
        "hy": "hy_normalized_mse",
        "hy_normalized_mse": "hy_normalized_mse",
        "official": "official_ee6d_regression",
        "official_ee6d_regression": "official_ee6d_regression",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _ACTION_LOSS_PROFILES:
        allowed = ", ".join(sorted(_ACTION_LOSS_PROFILES))
        raise ValueError(
            f"action_loss_profile must be one of {allowed}, got {value!r}"
        )
    return normalized


def _validate_action_loss_profile_contract(
    *,
    profile: str,
    action_loss_mode: str,
    action_normalization: str,
    action_representation: str,
) -> None:
    if profile != "official_ee6d_regression":
        return
    if action_loss_mode != "regression":
        raise ValueError(
            "official_ee6d_regression requires action_loss_mode=regression."
        )
    if action_normalization != "none":
        raise ValueError(
            "official_ee6d_regression requires raw physical actions: "
            "dataset.action_normalization=none."
        )
    if action_representation != "absolute_chunk_ee_RT":
        raise ValueError(
            "official_ee6d_regression requires "
            "dataset.act_type=absolute_chunk_ee_RT."
        )


def _log_from_pretrained_summary(
    policy: "HyXVLA",
    *,
    source: str,
    branch: str,
    source_action_mode: str | None = None,
    source_action_dim: int | None = None,
    action_shape_mismatch: bool | None = None,
) -> None:
    """Emit one compact, post-load contract summary for reproducibility."""
    model = policy.xvla
    config = model.config
    action_space = getattr(model, "action_space", None)
    norm_enabled = getattr(action_space, "_norm_enabled", None)
    clip_enabled = getattr(action_space, "_clip_enabled", None)
    norm_mode = None
    if action_space is not None and hasattr(action_space, "_norm_mode"):
        norm_mode = action_space._norm_mode()
    norm_shape = getattr(norm_enabled, "shape", None)
    bank_shape = getattr(getattr(action_space, "_bank_mean", None), "shape", None)
    trainable = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in policy.parameters())
    vlm = getattr(model, "vlm", None)
    expert = getattr(model, "transformer", None)
    vlm_trainable = sum(
        parameter.numel() for parameter in vlm.parameters() if parameter.requires_grad
    ) if vlm is not None else 0
    vlm_total = (
        sum(parameter.numel() for parameter in vlm.parameters())
        if vlm is not None
        else 0
    )
    expert_trainable = sum(
        parameter.numel() for parameter in expert.parameters() if parameter.requires_grad
    ) if expert is not None else 0
    expert_total = (
        sum(parameter.numel() for parameter in expert.parameters())
        if expert is not None
        else 0
    )
    _logger.info(
        "[from_pretrained] branch=%s source=%s source_contract=(mode=%s dim=%s) "
        "target_contract=(backend=%s mode=%s dim=%s real_dim=%s actions=%s hidden=%s depth=%s) "
        "loss=(mode=%s profile=%s) qwen_temporal=%s qwen_spacetime_stride=%s "
        "transfer=%s shape_mismatch=%s "
        "norm=(class=%s dim=%s enabled=%s mode=%s clip=%s banks=%s) "
        "modules=(xvla=%s vlm=%s dtype=%s) params=(trainable=%d total=%d "
        "vlm=%d/%d expert=%d/%d)",
        branch,
        source,
        source_action_mode,
        source_action_dim,
        getattr(config, "vlm_backend", None),
        getattr(config, "action_mode", None),
        getattr(config, "max_action_dim", None),
        getattr(config, "real_action_dim", None),
        getattr(config, "num_actions", None),
        getattr(config, "hidden_size", None),
        getattr(config, "depth", None),
        getattr(config, "action_loss_mode", None),
        getattr(config, "action_loss_profile", None),
        getattr(config, "qwen_temporal_embedding", "off"),
        getattr(config, "qwen_spacetime_stride", 4),
        policy.cross_emb_action_transfer,
        action_shape_mismatch,
        None if action_space is None else type(action_space).__name__,
        None if norm_shape is None else tuple(norm_shape),
        None if norm_enabled is None else int(norm_enabled.item()),
        norm_mode,
        None if clip_enabled is None else int(clip_enabled.item()),
        None if bank_shape is None else tuple(bank_shape),
        type(model).__name__,
        type(getattr(model, "vlm", None)).__name__,
        getattr(next(model.parameters(), None), "dtype", None),
        trainable,
        total,
        vlm_trainable,
        vlm_total,
        expert_trainable,
        expert_total,
    )


class HyXVLA(nn.Module):
    """Expose X-VLA through the existing Hy-VLA model API.

    The data pipeline emits 32-wide state/action tensors. Historical RoboDojo
    EEF runs use the first 20 channels; the dedicated DM05 joint source uses
    the first 14 OpenDM joint channels. State/action values are normalized by
    the dataset statistics, and native DM05 masks the padded tail. An explicit
    CrossEmb-preserve path expands the historical 20-D EEF layout only.
    """

    name = "xvla"

    def __init__(
        self,
        xvla: XVLA,
        processor: XVLAProcessor,
        *,
        inference_steps: int = 10,
        domain_id: int = 0,
        action_normalization: str = "zscore",
        action_representation: str = "relative_chunk_ee_RT",
        action_loss_profile: str = "hy_normalized_mse",
        random_init_action_head: bool = False,
        cross_emb_action_transfer: str = "not_applicable",
        detach_vlm_to_expert: bool = False,
        action_policy_init_source: str = "",
        action_prompt_init_source_domain: int | None = None,
        action_io_init_seed: int | None = None,
        action_expert_source_depth: int | None = None,
        action_expert_parameter_count: int | None = None,
        action_expert_block_parameter_count: int | None = None,
    ):
        super().__init__()
        self.xvla = xvla
        self.processor = processor
        self.config = xvla.config
        _install_qwen_timestamp_tokens(self.config, processor.tokenizer)
        self.inference_steps = int(inference_steps)
        self.domain_id = int(domain_id)
        self.action_normalization = str(action_normalization)
        self.action_representation = str(action_representation)
        self.action_loss_profile = _normalize_action_loss_profile(action_loss_profile)
        self.random_init_action_head = bool(random_init_action_head)
        self.cross_emb_action_transfer = str(cross_emb_action_transfer)
        self.detach_vlm_to_expert = bool(detach_vlm_to_expert)
        self.action_policy_init_source = str(action_policy_init_source)
        self.action_prompt_init_source_domain = action_prompt_init_source_domain
        self.action_io_init_seed = action_io_init_seed
        self.action_expert_source_depth = action_expert_source_depth
        self.action_expert_parameter_count = action_expert_parameter_count
        self.action_expert_block_parameter_count = (
            action_expert_block_parameter_count
        )
        # Training masks the 12-coordinate public action tail from both the
        # loss and the noised action condition. Evaluation keeps that behavior
        # unless an explicit diagnostic opts in to unmasked tail noise.
        self.preserve_padding_noise_inference = False
        self._action_queue: deque[Tensor] = deque([], maxlen=int(xvla.num_actions))
        self.register_buffer(
            "_image_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_image_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 1, 1, 3, 1, 1),
            persistent=False,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        *,
        num_actions: int | None = None,
        local_files_only: bool = False,
        inference_steps: int = 10,
        action_loss_mode: str | None = None,
        action_loss_profile: str | None = None,
        domain_id: int | None = None,
        action_normalization: str | None = None,
        action_representation: str | None = None,
        random_init_action_head: bool | str | None = None,
        qwen_temporal_embedding: str | None = None,
        qwen_image_attention_mode: str | None = None,
        qwen_far_aux: dict | None = None,
        qwen_history_time_rope: dict | None = None,
        qwen_spacetime_stride: int | None = None,
        qwen_history_value_interval: int | None = None,
        qwen_history_value_rope_theta: float | None = None,
        qwen_history_memory_enabled: bool | str | None = None,
        qwen_history_memory_pool_size: int | None = None,
        qwen_history_memory_gate_init: float | None = None,
        qwen_native_video_history_enabled: bool | str | None = None,
        qwen_native_video_history_views: tuple[int, ...] | list[int] | None = None,
        qwen_action_adalayernorm_enabled: bool | str | None = None,
        dm05_state_dim: int | None = None,
        cross_emb_action_transfer: str | None = None,
        detach_vlm_to_expert: bool | str | None = None,
        keep_vlm_decoder: bool | str | None = None,
        action_policy_init_path: str | None = None,
        cross_emb_prompt_source_domain: int | None = None,
        action_io_init_seed: int | None = None,
        action_expert_depth: int | str | None = None,
        time_embedding_max_period: int | None = None,
        time_embedding_input_scale: float | None = None,
        flow_time_distribution: str | None = None,
        beta_a: float | None = None,
        beta_b: float | None = None,
        logit_normal_m: float | None = None,
        logit_normal_s: float | None = None,
    ) -> "HyXVLA":
        source = str(pretrained_name_or_path)
        _logger.info(
            "[from_pretrained] start source=%s requested=(loss_mode=%s "
            "profile=%s domain=%s action_norm=%s action_rep=%s transfer=%s "
            "random_head=%s expert_depth=%s)",
            source,
            action_loss_mode,
            action_loss_profile,
            domain_id,
            action_normalization,
            action_representation,
            cross_emb_action_transfer,
            random_init_action_head,
            action_expert_depth,
        )
        manifest: dict[str, Any] = {}
        manifest_path = Path(source) / _ADAPTER_MANIFEST
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if action_normalization is None:
            action_normalization = str(manifest.get("action_normalization", "zscore"))
        if action_representation is None:
            action_representation = str(
                manifest.get("action_representation", "relative_chunk_ee_RT")
            )
        if domain_id is None:
            domain_id = int(manifest.get("domain_id", 0))
        else:
            domain_id = int(domain_id)
        saved_prompt_source_domain = manifest.get(
            "action_prompt_init_source_domain"
        )
        prompt_source_domain = (
            int(saved_prompt_source_domain)
            if saved_prompt_source_domain is not None
            else None
        )
        if action_normalization not in {"zscore", "none"}:
            raise ValueError(
                "HyXVLA action_normalization must be zscore or none, got "
                f"{action_normalization!r}"
            )
        reset_action_head = (
            _coerce_bool(random_init_action_head, name="random_init_action_head")
            if random_init_action_head is not None
            else False
        )
        record_random_init_action_head = (
            reset_action_head
            if random_init_action_head is not None
            else _coerce_bool(manifest.get("random_init_action_head", False), name="random_init_action_head")
        )
        requested_cross_emb_transfer = _normalize_cross_emb_action_transfer(
            cross_emb_action_transfer
            if cross_emb_action_transfer is not None
            else manifest.get("cross_emb_action_transfer", "auto")
        )
        if requested_cross_emb_transfer == "pretrained" and reset_action_head:
            raise ValueError(
                "cross_emb_action_transfer=pretrained cannot be combined with "
                "random_init_action_head=True. Set RANDOM_INIT_ACTION_HEAD=false "
                "to retain the checkpoint action I/O."
            )
        if (
            requested_cross_emb_transfer == "pretrained"
            and str(action_policy_init_path or "").strip()
        ):
            raise ValueError(
                "cross_emb_action_transfer=pretrained cannot be combined with "
                "action_policy_init_path. Clear ACTION_POLICY_INIT_PATH to use "
                "the checkpoint's complete pretrained policy."
            )
        if _is_raw_qwen35_checkpoint(source) and requested_cross_emb_transfer == "pretrained":
            raise ValueError(
                "cross_emb_action_transfer=pretrained requires a complete "
                "X-VLA checkpoint with pretrained action-policy weights; "
                f"{source!r} is a native Qwen3.5 VLM-only checkpoint. "
                "Set PRETRAIN to the pretraining pipeline's ckpt-<step> "
                "model directory (for example ckpt-50000)."
            )
        if qwen_temporal_embedding is not None:
            qwen_temporal_embedding = str(qwen_temporal_embedding).strip().lower()
            if qwen_temporal_embedding not in {
                "off",
                "fixed_sinusoidal_v1",
                "spacetime_mem_v1",
                "vlm_history_current_v1",
                "vlm_history_value_v1",
            }:
                raise ValueError(
                    "qwen_temporal_embedding must be one of "
                    "off / fixed_sinusoidal_v1 / spacetime_mem_v1 / vlm_history_current_v1 / vlm_history_value_v1, got "
                    f"{qwen_temporal_embedding!r}"
                )
        if qwen_image_attention_mode is not None:
            qwen_image_attention_mode = str(qwen_image_attention_mode).strip().lower()
            from XPolicyLab.policy.Era0_Preview.runtime.models.qwen_image_attention import IMAGE_ATTENTION_MODES
            if qwen_image_attention_mode not in IMAGE_ATTENTION_MODES:
                raise ValueError(f"Invalid qwen_image_attention_mode: {qwen_image_attention_mode!r}")
        if qwen_spacetime_stride is not None:
            qwen_spacetime_stride = int(qwen_spacetime_stride)
            if qwen_spacetime_stride < 1:
                raise ValueError(
                    "qwen_spacetime_stride must be >= 1, got "
                    f"{qwen_spacetime_stride!r}"
                )
        if qwen_history_memory_enabled is not None:
            qwen_history_memory_enabled = _coerce_bool(
                qwen_history_memory_enabled,
                name="qwen_history_memory_enabled",
            )
        if qwen_history_memory_pool_size is not None:
            qwen_history_memory_pool_size = int(qwen_history_memory_pool_size)
            if qwen_history_memory_pool_size < 1:
                raise ValueError(
                    "qwen_history_memory_pool_size must be >= 1, got "
                    f"{qwen_history_memory_pool_size!r}"
                )
        if qwen_history_memory_gate_init is not None:
            qwen_history_memory_gate_init = float(qwen_history_memory_gate_init)
            if not 0.0 <= qwen_history_memory_gate_init < 1.0:
                raise ValueError(
                    "qwen_history_memory_gate_init must be in [0,1), got "
                    f"{qwen_history_memory_gate_init!r}"
                )
        if qwen_native_video_history_enabled is not None:
            qwen_native_video_history_enabled = _coerce_bool(
                qwen_native_video_history_enabled,
                name="qwen_native_video_history_enabled",
            )
        if qwen_native_video_history_views is not None:
            qwen_native_video_history_views = tuple(
                int(v) for v in qwen_native_video_history_views
            )
        if qwen_action_adalayernorm_enabled is not None:
            qwen_action_adalayernorm_enabled = _coerce_bool(
                qwen_action_adalayernorm_enabled,
                name="qwen_action_adalayernorm_enabled",
            )
        if _is_raw_qwen35_checkpoint(source) or _is_raw_dm05_checkpoint(source):
            raise ValueError("Evaluation requires a complete Era0-Preview policy checkpoint.")
        source_cfg = XVLAConfig.from_pretrained(source, local_files_only=local_files_only)
        source_has_far_aux = bool(getattr(source_cfg, "qwen_far_aux", None)
                                  and source_cfg.qwen_far_aux.get("enabled", False))
        if qwen_temporal_embedding is not None:
            source_cfg.qwen_temporal_embedding = str(
                qwen_temporal_embedding
            ).strip().lower()
        if qwen_image_attention_mode is not None:
            source_cfg.qwen_image_attention_mode = str(qwen_image_attention_mode).strip().lower()
        if qwen_far_aux is not None:
            source_cfg.qwen_far_aux = dict(qwen_far_aux)
        if qwen_spacetime_stride is not None:
            source_cfg.qwen_spacetime_stride = int(qwen_spacetime_stride)
        if qwen_history_value_interval is not None:
            if int(qwen_history_value_interval) < 1:
                raise ValueError("qwen_history_value_interval must be positive.")
            source_cfg.qwen_history_value_interval = int(qwen_history_value_interval)
        if qwen_history_value_rope_theta is not None:
            if float(qwen_history_value_rope_theta) <= 1.0:
                raise ValueError("qwen_history_value_rope_theta must exceed 1.")
            source_cfg.qwen_history_value_rope_theta = float(qwen_history_value_rope_theta)
        if qwen_history_memory_enabled is not None:
            source_cfg.qwen_history_memory_enabled = bool(
                qwen_history_memory_enabled
            )
        if qwen_history_memory_pool_size is not None:
            source_cfg.qwen_history_memory_pool_size = int(
                qwen_history_memory_pool_size
            )
        if qwen_history_memory_gate_init is not None:
            source_cfg.qwen_history_memory_gate_init = float(
                qwen_history_memory_gate_init
            )
        if qwen_native_video_history_enabled is not None:
            source_cfg.qwen_native_video_history_enabled = bool(
                qwen_native_video_history_enabled
            )
        if qwen_native_video_history_views is not None:
            source_cfg.qwen_native_video_history_views = tuple(
                int(v) for v in qwen_native_video_history_views
            )
        if qwen_action_adalayernorm_enabled is not None:
            source_cfg.qwen_action_adalayernorm_enabled = bool(
                qwen_action_adalayernorm_enabled
            )
        num_domains = int(getattr(source_cfg, "num_domains", 0) or 0)
        if domain_id < 0 or domain_id >= num_domains:
            raise ValueError(
                f"domain_id={domain_id} is outside checkpoint range [0, {num_domains})."
            )
        source_cfg_payload: dict[str, Any] = {}
        source_cfg_path = Path(source) / "config.json"
        if source_cfg_path.is_file():
            source_cfg_payload = json.loads(source_cfg_path.read_text(encoding="utf-8"))
        requested_action_loss_profile = _normalize_action_loss_profile(
            action_loss_profile
            if action_loss_profile is not None
            else manifest.get(
                "action_loss_profile",
                source_cfg_payload.get(
                    "action_loss_profile",
                    getattr(source_cfg, "action_loss_profile", "hy_normalized_mse"),
                ),
            )
        )
        source_action_mode = str(getattr(source_cfg, "action_mode", "")).strip().lower()
        source_action_dim = int(getattr(source_cfg, "max_action_dim", 0) or 0)
        _logger.info(
            "[from_pretrained] source_config backend=%s mode=%s dim=%s "
            "real_dim=%s actions=%s hidden=%s depth=%s loss_mode=%s",
            getattr(source_cfg, "vlm_backend", None),
            source_action_mode,
            source_action_dim,
            getattr(source_cfg, "real_action_dim", None),
            getattr(source_cfg, "num_actions", None),
            getattr(source_cfg, "hidden_size", None),
            getattr(source_cfg, "depth", None),
            getattr(source_cfg, "action_loss_mode", None),
        )
        preserve_cross_emb = (
            source_action_mode == "cross_emb"
            and requested_cross_emb_transfer in {"preserve", "pretrained"}
        )
        if preserve_cross_emb and source_action_dim < max(_CROSS_EMB_EE_INDICES) + 1:
            raise ValueError(
                "CROSS_EMB_ACTION_TRANSFER=pretrained/preserve requires a native CrossEmb "
                f"action width >= 40, got {source_action_dim}."
            )
        if (
            requested_cross_emb_transfer == "pretrained"
            and source_action_mode != "cross_emb"
        ):
            raise ValueError(
                "cross_emb_action_transfer=pretrained requires a CrossEmb "
                f"source checkpoint, got action_mode={source_action_mode!r}."
            )
        target_cfg_dict = source_cfg.to_dict()
        if (target_cfg_dict.get("qwen_far_aux", None)
                and target_cfg_dict["qwen_far_aux"].get("enabled", False)
                and detach_vlm_to_expert is not None):
            # mixed11 pretraining saved detach=True. Apply the requested joint
            # SFT setting before validating the auxiliary configuration.
            target_cfg_dict["ki_detach_vlm_to_expert"] = _coerce_bool(
                detach_vlm_to_expert, name="detach_vlm_to_expert")
        target_cfg = XVLAConfig.from_dict(target_cfg_dict)
        source_action_expert_depth = int(getattr(source_cfg, "depth", 0) or 0)
        target_cfg.depth = cls._resolve_action_expert_depth(
            source_depth=source_action_expert_depth,
            requested_depth=action_expert_depth,
        )
        if (
            is_vlm_fast_action_bridge_version(
                getattr(source_cfg, "architecture_version", "legacy_cross_emb_v5")
            )
            and target_cfg.depth != source_action_expert_depth
        ):
            # The prefix-KV expert runs one block per VLM decoder layer and
            # requires exactly one prefix-K/V entry per block. Deepening it
            # would leave the extra blocks without a matching VLM layer.
            raise ValueError(
                "vlm_fast_action_bridge pairs one expert block with each VLM "
                "decoder layer, so ACTION_EXPERT_DEPTH cannot change it "
                f"(checkpoint depth={source_action_expert_depth})."
            )
        if (
            requested_cross_emb_transfer == "pretrained"
            and target_cfg.depth != source_action_expert_depth
        ):
            raise ValueError(
                "cross_emb_action_transfer=pretrained requires the checkpoint "
                "action-expert depth. Unset ACTION_EXPERT_DEPTH."
            )
        if preserve_cross_emb:
            # The mixed11 checkpoint uses the canonical 64-wide CrossEmb
            # layout. Keep it intact so its action head and norm buffers load
            # without shape/key conversion or uninitialized reinitialization.
            target_cfg.action_mode = "cross_emb"
            target_cfg.max_action_dim = source_action_dim
            target_cfg.real_action_dim = int(
                getattr(source_cfg, "real_action_dim", 60) or 60
            )
            target_cfg.action_norm_bank_count = int(
                getattr(source_cfg, "action_norm_bank_count", 1) or 1
            )
            target_cfg.cross_emb_ee_equivalent = bool(
                getattr(source_cfg, "cross_emb_ee_equivalent", False)
            )
        else:
            target_cfg.action_mode = "ee6d_padded"
            target_cfg.max_action_dim = _HY_ACTION_DIM
            target_cfg.real_action_dim = _EE6D_ACTION_DIM
        target_cfg.keep_gripper_proprio = True
        target_cfg.ee6d_regression_normalization = False
        target_cfg.ki_detach_vlm_to_expert = (
            _coerce_bool(
                detach_vlm_to_expert,
                name="detach_vlm_to_expert",
            )
            if detach_vlm_to_expert is not None
            else bool(getattr(source_cfg, "ki_detach_vlm_to_expert", False))
        )
        target_cfg.keep_vlm_decoder = (
            _coerce_bool(keep_vlm_decoder, name="keep_vlm_decoder")
            if keep_vlm_decoder is not None
            else bool(getattr(source_cfg, "keep_vlm_decoder", False))
        )
        # Hy-VLA's action training objective is flow matching.  The padded
        # action space owns its fp32, first-20-dimension scalar reduction so
        # the public Hy batch/data contract remains unchanged.
        target_cfg.action_loss_mode = (
            str(action_loss_mode)
            if action_loss_mode is not None
            else str(
                manifest.get(
                    "action_loss_mode",
                    source_cfg_payload.get("action_loss_mode", "flow_matching"),
                )
            )
        )
        target_cfg.action_loss_mode = str(
            target_cfg.action_loss_mode
        ).strip().lower().replace("-", "_")
        if target_cfg.action_loss_mode in {"flow", "fm"}:
            target_cfg.action_loss_mode = "flow_matching"
        if target_cfg.action_loss_mode not in {"flow_matching", "regression"}:
            raise ValueError(
                "action_loss_mode must be flow_matching or regression, got "
                f"{target_cfg.action_loss_mode!r}"
            )
        _validate_action_loss_profile_contract(
            profile=requested_action_loss_profile,
            action_loss_mode=target_cfg.action_loss_mode,
            action_normalization=action_normalization,
            action_representation=action_representation,
        )
        target_cfg.action_loss_profile = requested_action_loss_profile
        target_cfg.hy_action_contract = (
            _HY_ROBODOJO_ACTION_CONTRACT
            if (
                str(target_cfg.action_loss_mode).strip().lower()
                == "flow_matching"
                and action_normalization == "zscore"
                and action_representation == "relative_chunk_ee_RT"
            )
            else ""
        )
        if num_actions is not None:
            target_cfg.num_actions = int(num_actions)
        # Flow-matching timestep conditioning. Unspecified keeps whatever the
        # source checkpoint recorded (100 / 1.0 for every pre-2026-09-08
        # checkpoint), so warm starts and evaluation stay bit-exact unless the
        # launcher explicitly opts in.
        if qwen_history_time_rope is not None:
            from XPolicyLab.policy.Era0_Preview.runtime.models.qwen_history_time_rope import validate_config as validate_time_rope
            target_cfg.qwen_history_time_rope = qwen_history_time_rope
            validate_time_rope(target_cfg)
        if time_embedding_max_period is not None:
            target_cfg.time_embedding_max_period = int(time_embedding_max_period)
        if time_embedding_input_scale is not None:
            target_cfg.time_embedding_input_scale = float(time_embedding_input_scale)
        # Flow-matching t sampling. Unspecified keeps whatever the source
        # checkpoint recorded (Beta(1.5, 1.0) for every pre-2026-09-11
        # checkpoint) so warm starts and evaluation stay bit-exact unless the
        # launcher opts in explicitly.
        if flow_time_distribution is not None:
            distribution_norm = str(flow_time_distribution).strip().lower()
            if distribution_norm not in {"beta", "logit_normal"}:
                raise ValueError(
                    "flow_time_distribution must be 'beta' or 'logit_normal'; "
                    f"got {flow_time_distribution!r}"
                )
            target_cfg.flow_time_distribution = distribution_norm
        if (beta_a is None) != (beta_b is None):
            raise ValueError("beta_a and beta_b must be set together.")
        if beta_a is not None:
            target_cfg.beta_a = float(beta_a)
            target_cfg.beta_b = float(beta_b)
            if target_cfg.beta_a <= 0.0 or target_cfg.beta_b <= 0.0:
                raise ValueError(
                    "beta_a and beta_b must be strictly positive; got "
                    f"beta_a={beta_a!r}, beta_b={beta_b!r}"
                )
        if logit_normal_m is not None:
            target_cfg.logit_normal_m = float(logit_normal_m)
        if logit_normal_s is not None:
            s_value = float(logit_normal_s)
            if s_value <= 0.0:
                raise ValueError(
                    "logit_normal_s must be strictly positive; got "
                    f"{logit_normal_s!r}"
                )
            target_cfg.logit_normal_s = s_value
        if reset_action_head:
            target_cfg.action_initialization_contract = "hy_random_action_io_v1"

        effective_cross_emb_transfer = "not_applicable"
        if source_action_mode == "cross_emb":
            if preserve_cross_emb:
                effective_cross_emb_transfer = requested_cross_emb_transfer
            else:
                effective_cross_emb_transfer = cls._resolve_cross_emb_action_transfer(
                    requested_cross_emb_transfer,
                    source_cfg_payload,
                )
            if reset_action_head:
                effective_cross_emb_transfer = "reset"
            if effective_cross_emb_transfer == "reset":
                target_cfg.action_initialization_contract = (
                    "hy_cross_emb_incompatible_action_io_reset_v2"
                )
        action_shape_mismatch = (
            target_cfg.num_actions != source_cfg.num_actions
            or source_action_mode != target_cfg.action_mode
            or source_action_dim != target_cfg.max_action_dim
            or target_cfg.depth != source_action_expert_depth
        )
        _logger.info(
            "[from_pretrained] target_config mode=%s dim=%s real_dim=%s "
            "actions=%s hidden=%s depth=%s loss=(%s,%s) transfer=%s "
            "ignore_mismatched_sizes=%s",
            getattr(target_cfg, "action_mode", None),
            getattr(target_cfg, "max_action_dim", None),
            getattr(target_cfg, "real_action_dim", None),
            getattr(target_cfg, "num_actions", None),
            getattr(target_cfg, "hidden_size", None),
            getattr(target_cfg, "depth", None),
            getattr(target_cfg, "action_loss_mode", None),
            getattr(target_cfg, "action_loss_profile", None),
            requested_cross_emb_transfer,
            action_shape_mismatch,
        )

        xvla = XVLA.from_pretrained(
            source,
            config=target_cfg,
            ignore_mismatched_sizes=action_shape_mismatch,
            local_files_only=local_files_only,
        )
        cls._reinit_history_memory_after_pretrained(xvla, source=source)
        if getattr(xvla, "far_aux_head", None) is not None and not source_has_far_aux:
            # Missing modules loaded by HF may have been constructed on meta.
            # Reset only a fresh head; restored auxiliary weights stay intact.
            for module in xvla.far_aux_head.modules():
                if isinstance(module, (nn.Linear, nn.LayerNorm)):
                    module.reset_parameters()
        if source_action_mode == "cross_emb" and not preserve_cross_emb:
            # CrossEmb and EE6D action spaces have different dimensions and
            # normalization-buffer layouts. Same-shaped source flags can still
            # be restored while the 20-D statistics are skipped, accidentally
            # enabling invalid normalization for already z-scored RoboDojo data.
            cls._reset_converted_ee6d_action_space(xvla)
        if preserve_cross_emb:
            action_space = xvla.action_space
            # RoboDojo already supplies dataset-side z-score actions/proprio.
            # The mixed11 CrossEmb buffers describe a different raw-action
            # distribution; applying them here would normalize (and possibly
            # clip) the Hy batch a second time.  Preserve the native 64-wide
            # head, but make its action-space transform an identity so the
            # Hy loss/data contract remains valid.
            source_norm_enabled = bool(action_space._norm_enabled.item())
            source_clip_enabled = bool(action_space._clip_enabled.item())
            if action_normalization == "zscore":
                action_space._norm_enabled.fill_(0)
                action_space._norm_mode_id.fill_(0)
                action_space._clip_enabled.fill_(0)
                _logger.info(
                    "[cross-emb preserve] disabled incompatible source action "
                    "norm/clip for Hy dataset z-score contract "
                    "(source_norm=%s source_clip=%s); native 64-D head retained",
                    source_norm_enabled,
                    source_clip_enabled,
                )
            _logger.info(
                "[cross-emb %s] loaded native action space: mode=%s dim=%d "
                "real_dim=%d norm_enabled=%d norm_mode=%s banks=%d",
                requested_cross_emb_transfer,
                target_cfg.action_mode,
                int(action_space.dim_action),
                int(getattr(action_space, "base_dim", target_cfg.real_action_dim)),
                int(action_space._norm_enabled.item()),
                action_space._norm_mode(),
                int(getattr(action_space, "_bank_low", torch.empty(1)).shape[0]),
            )
        if len(xvla.transformer.blocks) != int(target_cfg.depth):
            raise RuntimeError(
                "Loaded action expert depth does not match requested architecture: "
                f"expected {target_cfg.depth}, got {len(xvla.transformer.blocks)}."
            )
        action_expert_parameter_count = sum(
            parameter.numel() for parameter in xvla.transformer.parameters()
        )
        action_expert_block_parameter_count = sum(
            parameter.numel() for parameter in xvla.transformer.blocks.parameters()
        )
        _logger.info(
            "[from_pretrained] action_expert depth=%d source_depth=%d "
            "params=%d block_params=%d",
            len(xvla.transformer.blocks),
            source_action_expert_depth,
            action_expert_parameter_count,
            action_expert_block_parameter_count,
        )
        target_cfg.action_expert_source_depth = source_action_expert_depth
        target_cfg.action_expert_parameter_count = action_expert_parameter_count
        target_cfg.action_expert_block_parameter_count = (
            action_expert_block_parameter_count
        )
        policy_init_source = str(action_policy_init_path or "").strip()
        position_source_steps = int(source_cfg.num_actions)
        if policy_init_source:
            reference_cfg = cls._load_reference_action_policy(
                xvla,
                policy_init_source,
            )
            effective_cross_emb_transfer = "reference_policy"
            position_source_steps = int(reference_cfg.num_actions)
            target_cfg.action_initialization_contract = (
                "hy_reference_action_policy_v1"
            )
        elif reset_action_head:
            cls._random_init_action_io_modules(xvla, seed=action_io_init_seed)
        elif source_action_mode == "cross_emb":
            if effective_cross_emb_transfer == "reset":
                cls._random_init_action_io_modules(xvla, seed=action_io_init_seed)
            elif effective_cross_emb_transfer not in {"preserve", "pretrained"}:
                cls._warm_start_cross_emb_padded_action_modules(xvla, source)
        elif source_cfg.max_action_dim < target_cfg.max_action_dim:
            cls._warm_start_padded_action_modules(xvla, source)
        if (
            source_action_mode == "cross_emb"
            and not policy_init_source
            and cross_emb_prompt_source_domain is not None
        ):
            prompt_source_domain = int(cross_emb_prompt_source_domain)
            cls._copy_domain_soft_prompt(
                xvla,
                source_domain=prompt_source_domain,
                target_domain=domain_id,
            )
        if target_cfg.num_actions != position_source_steps:
            # X-VLA's learned position table is laid out as
            # [action tokens][vision/text/aux tokens].  Loading a shorter
            # action chunk succeeds shape-wise, but would otherwise make the
            # visual prefix read positions trained for action tokens.
            pos_emb = xvla.transformer.pos_emb
            source_steps = position_source_steps
            target_steps = int(target_cfg.num_actions)
            original = pos_emb.data.clone()
            usable = max(0, int(original.shape[1]) - source_steps)
            pos_emb.data[:, target_steps : target_steps + usable] = original[
                :, source_steps : source_steps + usable
            ]
            pos_emb.data[:, target_steps + usable :] = 0.0
        processor = XVLAProcessor.from_pretrained(source, local_files_only=local_files_only)
        apply_action_language_max_length(processor, target_cfg)
        policy = cls(
            xvla,
            processor,
            inference_steps=inference_steps,
            domain_id=domain_id,
            action_normalization=action_normalization,
            action_representation=action_representation,
            action_loss_profile=requested_action_loss_profile,
            random_init_action_head=(
                record_random_init_action_head
                or effective_cross_emb_transfer == "reset"
            ),
            cross_emb_action_transfer=effective_cross_emb_transfer,
            detach_vlm_to_expert=bool(target_cfg.ki_detach_vlm_to_expert),
            action_policy_init_source=policy_init_source,
            action_prompt_init_source_domain=prompt_source_domain,
            action_io_init_seed=action_io_init_seed,
            action_expert_source_depth=source_action_expert_depth,
            action_expert_parameter_count=action_expert_parameter_count,
            action_expert_block_parameter_count=action_expert_block_parameter_count,
        )
        _log_from_pretrained_summary(
            policy,
            source=source,
            branch="xvla_checkpoint",
            source_action_mode=source_action_mode,
            source_action_dim=source_action_dim,
            action_shape_mismatch=action_shape_mismatch,
        )
        return policy



    @staticmethod
    def _resolve_action_expert_depth(
        *,
        source_depth: int,
        requested_depth: int | str | None,
    ) -> int:
        """Validate an action-expert expansion while preserving source blocks."""
        if source_depth <= 0:
            raise ValueError(
                f"Checkpoint action expert depth must be positive, got {source_depth}."
            )
        if requested_depth is None or str(requested_depth).strip().lower() in {
            "",
            "none",
            "null",
        }:
            return source_depth
        try:
            target_depth = int(requested_depth)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "action_expert_depth must be a positive integer or null, got "
                f"{requested_depth!r}."
            ) from error
        if target_depth <= 0:
            raise ValueError(
                "action_expert_depth must be a positive integer or null, got "
                f"{requested_depth!r}."
            )
        if target_depth < source_depth:
            raise ValueError(
                "action_expert_depth cannot shrink the checkpoint policy: "
                f"source={source_depth}, requested={target_depth}."
            )
        return target_depth

    @staticmethod
    def _resolve_cross_emb_action_transfer(
        requested: str,
        source_cfg_payload: dict[str, Any],
    ) -> str:
        """Resolve whether a CrossEmb action head can be reused safely.

        Shape and EE-slot alignment alone are insufficient: source targets can
        differ in horizon reference, rotation encoding, normalization, and
        regression objective.
        Direct copying is allowed only after an explicit source-contract
        attestation for this exact RoboDojo target representation.
        """
        source_contract = str(
            source_cfg_payload.get("hy_action_contract", "")
        ).strip()
        compatible = source_contract == _HY_ROBODOJO_ACTION_CONTRACT
        if requested == "reset":
            return "reset"
        if requested == "copy_compatible":
            if not compatible:
                raise ValueError(
                    "cross_emb_action_transfer=copy_compatible requires "
                    f"config.json hy_action_contract={_HY_ROBODOJO_ACTION_CONTRACT!r}; "
                    f"got {source_contract or '<missing>'!r}. Use reset for "
                    "mixed CrossEmb checkpoints without an explicit conversion."
                )
            return "copy_compatible"
        return "copy_compatible" if compatible else "reset"

    @staticmethod
    def _random_init_domain_linear(module: nn.Module, *, name: str) -> None:
        fc = getattr(module, "fc", None)
        bias = getattr(module, "bias", None)
        if isinstance(fc, nn.Embedding) and isinstance(bias, nn.Embedding):
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(bias.weight)
            return
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()
            return
        raise TypeError(f"Cannot random-initialize {name}: unsupported module {type(module)!r}")

    @classmethod
    def _random_init_action_io_modules(
        cls,
        xvla: XVLA,
        *,
        seed: int | None = None,
    ) -> None:
        """Reset only action I/O while preserving the learned policy backbone."""
        def reset() -> None:
            cls._random_init_domain_linear(
                xvla.transformer.action_encoder,
                name="action_encoder",
            )
            cls._random_init_domain_linear(
                xvla.transformer.action_decoder,
                name="action_decoder",
            )

        if seed is None:
            reset()
            return

        cuda_devices = sorted({
            parameter.device.index
            for module in (
                xvla.transformer.action_encoder,
                xvla.transformer.action_decoder,
            )
            for parameter in module.parameters()
            if parameter.device.type == "cuda" and parameter.device.index is not None
        })
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            reset()

    @staticmethod
    def _reinit_history_memory_after_pretrained(
        xvla: XVLA, *, source: str
    ) -> None:
        """Restore deterministic init for the opt-in history memory branch.

        Root cause: ``XVLA.from_pretrained`` delegates missing-key allocation
        to Hugging Face, which reserves uninitialized storage for parameters
        absent from the source checkpoint and then relies on
        ``_init_weights`` to overwrite it. ``ReadOnlyContextCrossAttention``
        keeps ``raw_gate`` as a bespoke ``nn.Parameter`` that the default
        HF rule does not touch, so on a fresh Qwen X-VLA start the new
        ``history_memory_cross_attn.*.raw_gate`` can materialize with
        garbage (observed |raw_gate| ~ 4e30, driving ``tanh`` to +/-1 and
        blowing the memory branch open before optimization begins). The
        same risk applies to ``history_memory_stream_embedding`` and
        ``history_memory_view_embeddings``.

        This helper unconditionally rewrites the history-memory branch
        parameters via ``SoftPromptedTransformer.reset_history_memory_parameters``
        after loading, then asserts the invariants ``raw_gate == atanh(gate_init)``
        and finite storage everywhere. Any deviation stops training with a
        precise diagnostic rather than silently poisoning the run.
        """
        transformer = getattr(xvla, "transformer", None)
        if transformer is None:
            return
        if not bool(getattr(transformer, "history_memory_enabled", False)):
            return
        reset = getattr(transformer, "reset_history_memory_parameters", None)
        if reset is None:
            raise RuntimeError(
                "SoftPromptedTransformer is missing reset_history_memory_parameters(); "
                "cannot recover deterministic history-memory init."
            )
        reset()
        expected_gate = float(transformer.history_memory_gate_init)
        atanh_expected = math.atanh(expected_gate) if expected_gate < 1.0 else float("inf")
        readers = transformer.history_memory_cross_attn
        gates: dict[str, float] = {}
        for layer_key, reader in readers.items():
            raw = reader.raw_gate.detach().float().cpu()
            if not torch.isfinite(raw).all().item():
                raise RuntimeError(
                    "history_memory_cross_attn."
                    f"{layer_key}.raw_gate contains non-finite values after "
                    f"from_pretrained({source!r}); expected atanh({expected_gate})."
                )
            value = float(raw.reshape(-1)[0].item())
            if not math.isclose(value, atanh_expected, abs_tol=1e-6):
                raise RuntimeError(
                    "history_memory_cross_attn."
                    f"{layer_key}.raw_gate={value} != atanh({expected_gate})="
                    f"{atanh_expected} after re-initialization."
                )
            gates[str(layer_key)] = value
        for name in (
            "history_memory_stream_embedding",
            "history_memory_view_embeddings",
            "history_memory_proj",
        ):
            module_or_param = getattr(transformer, name, None)
            if module_or_param is None:
                continue
            tensors: list[Tensor]
            if isinstance(module_or_param, Tensor):
                tensors = [module_or_param.detach()]
            else:
                tensors = [p.detach() for p in module_or_param.parameters()]
            for t in tensors:
                if not torch.isfinite(t).all().item():
                    raise RuntimeError(
                        f"transformer.{name} holds non-finite values after "
                        f"history-memory re-initialization."
                    )
        _logger.info(
            "[history-memory init] reset history_memory branch after "
            "from_pretrained(%s); gate_init=%s raw_gate=%s",
            source,
            expected_gate,
            gates,
        )

    @staticmethod
    def _reset_converted_ee6d_action_space(xvla: XVLA) -> None:
        """Clear source-only norm state after CrossEmb -> EE6D conversion."""
        action_space = getattr(xvla, "action_space", None)
        if action_space is None:
            raise RuntimeError("Converted XVLA checkpoint has no action space")
        defaults = {
            "_norm_enabled": 0,
            "_norm_mode_id": 0,
            "_norm_mean": 0.0,
            "_norm_std": 1.0,
            "_norm_q_low": -1.0,
            "_norm_q_high": 1.0,
            "_regression_norm_requested": 0,
            "_regression_norm_stats_loaded": 0,
            "_regression_norm_mode_id": 0,
        }
        for name, value in defaults.items():
            buffer = getattr(action_space, name, None)
            if isinstance(buffer, Tensor):
                buffer.fill_(value)
        _logger.info("[cross-emb conversion] reset target EE6D normalization state")

    @staticmethod
    def _copy_domain_soft_prompt(
        xvla: XVLA,
        *,
        source_domain: int,
        target_domain: int,
    ) -> None:
        """Initialize a downstream domain from a compatible learned embodiment prompt."""
        prompt_hub = xvla.transformer.soft_prompt_hub
        if not isinstance(prompt_hub, nn.Embedding):
            raise TypeError(
                "CrossEmb prompt transfer requires transformer.soft_prompt_hub "
                f"to be nn.Embedding, got {type(prompt_hub)!r}."
            )
        num_domains = int(prompt_hub.num_embeddings)
        for name, value in (
            ("source_domain", source_domain),
            ("target_domain", target_domain),
        ):
            if value < 0 or value >= num_domains:
                raise ValueError(
                    f"{name}={value} is outside soft-prompt range [0, {num_domains})."
                )
        if source_domain == target_domain:
            return
        with torch.no_grad():
            prompt_hub.weight[target_domain].copy_(prompt_hub.weight[source_domain])

    @classmethod
    def _load_reference_action_policy(
        cls,
        xvla: XVLA,
        reference_source: str,
    ) -> XVLAConfig:
        """Diagnostic ablation replacing the policy while retaining the VLM."""
        reference = Path(reference_source)
        checkpoint = reference / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Reference action policy requires a local model.safetensors: "
                f"{checkpoint}"
            )
        reference_cfg = XVLAConfig.from_pretrained(
            str(reference),
            local_files_only=True,
        )
        from safetensors import safe_open

        target_state = xvla.transformer.state_dict()
        replacement: dict[str, Tensor] = {}
        skipped_action_io = {
            "action_encoder.fc.weight",
            "action_encoder.bias.weight",
            "action_decoder.fc.weight",
            "action_decoder.bias.weight",
        }
        with safe_open(str(checkpoint), framework="pt", device="cpu") as state:
            source_keys = set(state.keys())
            for target_key, target_value in target_state.items():
                if target_key in skipped_action_io:
                    continue
                source_key = f"transformer.{target_key}"
                if source_key not in source_keys:
                    raise KeyError(
                        f"Reference action policy is missing {source_key!r}."
                    )
                source_value = state.get_tensor(source_key)
                if source_value.shape != target_value.shape:
                    raise ValueError(
                        "Reference action-policy tensor shape mismatch for "
                        f"{source_key}: source={tuple(source_value.shape)} "
                        f"target={tuple(target_value.shape)}."
                    )
                replacement[target_key] = source_value

        incompatible = xvla.transformer.load_state_dict(
            replacement,
            strict=False,
        )
        missing = set(incompatible.missing_keys)
        if missing != skipped_action_io or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected reference action-policy load result: "
                f"missing={sorted(missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )

        reference_action_mode = str(reference_cfg.action_mode).strip().lower()
        reference_action_dim = int(
            getattr(reference_cfg, "max_action_dim", 0) or 0
        )
        if (
            reference_action_mode != "ee6d"
            or reference_action_dim != _EE6D_ACTION_DIM
        ):
            raise ValueError(
                "Reference action-policy initialization currently requires "
                "an ee6d 20-D checkpoint; got "
                f"mode={reference_action_mode!r}, dim={reference_action_dim}."
            )
        cls._warm_start_padded_action_modules(xvla, str(reference))
        return reference_cfg

    @staticmethod
    def _warm_start_padded_action_modules(xvla: XVLA, source: str) -> None:
        """Expand the 20-D action modules while retaining all pretrained rows."""
        checkpoint = Path(source) / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "32-D Hy-XVLA initialization requires the local X-VLA "
                f"safetensors checkpoint at {checkpoint}."
            )
        from safetensors import safe_open

        with safe_open(str(checkpoint), framework="pt", device="cpu") as state:
            decoder_fc = state.get_tensor("transformer.action_decoder.fc.weight")
            decoder_bias = state.get_tensor("transformer.action_decoder.bias.weight")
            encoder_fc = state.get_tensor("transformer.action_encoder.fc.weight")

        target_decoder_fc = xvla.transformer.action_decoder.fc.weight
        target_decoder_bias = xvla.transformer.action_decoder.bias.weight
        target_encoder_fc = xvla.transformer.action_encoder.fc.weight
        num_domains = target_decoder_fc.shape[0]
        hidden_size = xvla.transformer.hidden_size
        source_action_dim = decoder_bias.shape[1]
        target_action_dim = target_decoder_bias.shape[1]
        source_encoder_inputs = encoder_fc.shape[1] // hidden_size
        target_encoder_inputs = target_encoder_fc.shape[1] // hidden_size

        if (
            decoder_fc.shape[0] != num_domains
            or encoder_fc.shape[0] != num_domains
            or source_action_dim != _EE6D_ACTION_DIM
            or target_action_dim != _HY_ACTION_DIM
            or source_encoder_inputs != _EE6D_ACTION_DIM * 2 + xvla.transformer.dim_time
            or target_encoder_inputs != _HY_ACTION_DIM * 2 + xvla.transformer.dim_time
        ):
            raise ValueError(
                "Unexpected action-head layout while expanding X-VLA checkpoint: "
                f"decoder={tuple(decoder_fc.shape)}, encoder={tuple(encoder_fc.shape)}, "
                f"target_decoder={tuple(target_decoder_fc.shape)}, "
                f"target_encoder={tuple(target_encoder_fc.shape)}."
            )

        with torch.no_grad():
            # DomainAwareLinear stores action-decoder kernels as
            # [domain, hidden_size, action_dim]. New padded outputs are zero.
            target_decoder_fc.zero_()
            target_decoder_fc.view(num_domains, hidden_size, target_action_dim)[..., :source_action_dim].copy_(
                decoder_fc.view(num_domains, hidden_size, source_action_dim)
            )
            target_decoder_bias.zero_()
            target_decoder_bias[:, :source_action_dim].copy_(decoder_bias)

            # Action encoder input layout is [action | proprio | timestep].
            target_encoder_fc.zero_()
            source_encoder = encoder_fc.view(num_domains, source_encoder_inputs, hidden_size)
            target_encoder = target_encoder_fc.view(num_domains, target_encoder_inputs, hidden_size)
            source_proprio_start = source_action_dim
            source_time_start = source_proprio_start + source_action_dim
            target_proprio_start = target_action_dim
            target_time_start = target_proprio_start + target_action_dim
            target_encoder[:, :source_action_dim].copy_(source_encoder[:, :source_action_dim])
            target_encoder[:, target_proprio_start:target_proprio_start + source_action_dim].copy_(
                source_encoder[:, source_proprio_start:source_proprio_start + source_action_dim]
            )
            target_encoder[:, target_time_start:target_time_start + xvla.transformer.dim_time].copy_(
                source_encoder[:, source_time_start:source_time_start + xvla.transformer.dim_time]
            )

    @staticmethod
    def _warm_start_cross_emb_padded_action_modules(xvla: XVLA, source: str) -> None:
        """Map cross_emb64 EE slots onto Hy's padded 32-D EE6D contract."""
        checkpoint = Path(source) / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "cross_emb -> ee6d_padded initialization requires the local X-VLA "
                f"safetensors checkpoint at {checkpoint}."
            )
        from safetensors import safe_open

        with safe_open(str(checkpoint), framework="pt", device="cpu") as state:
            decoder_fc = state.get_tensor("transformer.action_decoder.fc.weight")
            decoder_bias = state.get_tensor("transformer.action_decoder.bias.weight")
            encoder_fc = state.get_tensor("transformer.action_encoder.fc.weight")

        target_decoder_fc = xvla.transformer.action_decoder.fc.weight
        target_decoder_bias = xvla.transformer.action_decoder.bias.weight
        target_encoder_fc = xvla.transformer.action_encoder.fc.weight
        num_domains = target_decoder_fc.shape[0]
        hidden_size = xvla.transformer.hidden_size
        source_action_dim = decoder_bias.shape[1]
        target_action_dim = target_decoder_bias.shape[1]
        source_encoder_inputs = encoder_fc.shape[1] // hidden_size
        target_encoder_inputs = target_encoder_fc.shape[1] // hidden_size
        ee_indices = torch.tensor(_CROSS_EMB_EE_INDICES, dtype=torch.long)

        if (
            decoder_fc.shape[0] != num_domains
            or encoder_fc.shape[0] != num_domains
            or source_action_dim < max(_CROSS_EMB_EE_INDICES) + 1
            or target_action_dim != _HY_ACTION_DIM
            or source_encoder_inputs != source_action_dim * 2 + xvla.transformer.dim_time
            or target_encoder_inputs != _HY_ACTION_DIM * 2 + xvla.transformer.dim_time
        ):
            raise ValueError(
                "Unexpected cross_emb action-head layout while adapting Qwen X-VLA checkpoint: "
                f"decoder={tuple(decoder_fc.shape)}, encoder={tuple(encoder_fc.shape)}, "
                f"target_decoder={tuple(target_decoder_fc.shape)}, "
                f"target_encoder={tuple(target_encoder_fc.shape)}."
            )

        with torch.no_grad():
            target_decoder_fc.zero_()
            source_decoder = decoder_fc.view(num_domains, hidden_size, source_action_dim)
            target_decoder = target_decoder_fc.view(num_domains, hidden_size, target_action_dim)
            target_decoder[..., :_EE6D_ACTION_DIM].copy_(
                source_decoder.index_select(-1, ee_indices)
            )
            target_decoder_bias.zero_()
            target_decoder_bias[:, :_EE6D_ACTION_DIM].copy_(
                decoder_bias.index_select(-1, ee_indices)
            )

            target_encoder_fc.zero_()
            source_encoder = encoder_fc.view(num_domains, source_encoder_inputs, hidden_size)
            target_encoder = target_encoder_fc.view(num_domains, target_encoder_inputs, hidden_size)
            source_proprio_start = source_action_dim
            source_time_start = source_proprio_start + source_action_dim
            target_proprio_start = _HY_ACTION_DIM
            target_time_start = target_proprio_start + _HY_ACTION_DIM
            target_encoder[:, :_EE6D_ACTION_DIM].copy_(
                source_encoder[:, _CROSS_EMB_EE_INDICES]
            )
            target_encoder[:, target_proprio_start:target_proprio_start + _EE6D_ACTION_DIM].copy_(
                source_encoder[:, [source_proprio_start + i for i in _CROSS_EMB_EE_INDICES]]
            )
            target_encoder[:, target_time_start:target_time_start + xvla.transformer.dim_time].copy_(
                source_encoder[:, source_time_start:source_time_start + xvla.transformer.dim_time]
            )


    def reset(self) -> None:
        self._action_queue.clear()

    @staticmethod
    def _first_tensor(batch: dict[str, Any]) -> Tensor:
        for value in batch.values():
            if isinstance(value, Tensor):
                return value
        raise ValueError("HyXVLA batch has no tensor values.")

    def _xvla_image_size(self) -> tuple[int, int]:
        """Return the image size used by the X-VLA action-policy path.

        Cross-emb Florence-large pretraining serializes Florence's native
        768px processor (577 visual tokens), but its action dataset is always
        resized to 224px. Feeding the serialized 768px size to a 512-token
        action transformer cannot fit even one view. Keep that historical
        train/deploy convention here without changing checkpoints whose
        recorded processor size already fits the action transformer.
        """
        size = getattr(self.processor.image_processor, "size", None)
        if hasattr(size, "get"):
            height = size.get("height", size.get("shortest_edge"))
            width = size.get("width", height)
        elif getattr(size, "height", None) is not None:
            height = size.height
            width = getattr(size, "width", height)
        else:
            height = width = size
        try:
            height = int(height)
            width = int(width)
        except (TypeError, ValueError):
            height = width = 0
        if height <= 0 or width <= 0:
            raise ValueError(
                "X-VLA image processor must define a positive target image size, "
                f"got {size!r}."
            )
        backend = str(getattr(self.config, "vlm_backend", "florence2")).strip().lower()
        if backend == "florence2":
            image_seq_length = getattr(self.processor.image_processor, "image_seq_length", None)
            transformer = getattr(self.xvla, "transformer", None)
            pos_emb = getattr(transformer, "pos_emb", None)
            if (
                image_seq_length is not None
                and pos_emb is not None
                and int(image_seq_length) >= int(pos_emb.shape[1])
            ):
                return _FLORENCE_XVLA_POLICY_IMAGE_SIZE
        return height, width

    def _resize_for_xvla(self, image_input: Tensor) -> Tensor:
        """Apply X-VLA's square processor resize without changing Hy-VLA batches.

        The shared Hy-VLA loader deliberately uses ``Resize(224)`` and thus
        preserves the camera aspect ratio.  Its native model pads those tensors
        internally.  Florence instead reshapes the DaViT tokens to a square
        grid, so the model adapter must apply the X-VLA checkpoint's recorded
        ``(height, width)`` resize after the unchanged Hy preprocessing.
        """
        target_size = self._xvla_image_size()
        if tuple(image_input.shape[-2:]) == target_size:
            return image_input
        original_shape = image_input.shape
        flattened = image_input.reshape(-1, *original_shape[-3:])
        resized = F.interpolate(
            flattened.float(),
            size=target_size,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return resized.to(dtype=image_input.dtype).reshape(*original_shape[:-2], *target_size)

    def _letterbox_for_xvla(self, image_input: Tensor) -> Tensor:
        """Aspect-preserving tensor letterbox for Qwen checkpoints."""
        target_h, target_w = self._xvla_image_size()
        source_h, source_w = map(int, image_input.shape[-2:])
        if (source_h, source_w) == (target_h, target_w):
            return image_input
        if source_h <= 0 or source_w <= 0:
            raise ValueError(f"Invalid image tensor size {(source_h, source_w)}.")

        scale = min(target_w / source_w, target_h / source_h)
        resized_w = min(target_w, max(1, int(round(source_w * scale))))
        resized_h = min(target_h, max(1, int(round(source_h * scale))))
        original_shape = image_input.shape
        flattened = image_input.reshape(-1, *original_shape[-3:])
        resized = F.interpolate(
            flattened.float(),
            size=(resized_h, resized_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=image_input.dtype)

        pad_rgb = getattr(self.processor, "image_pad_rgb", (128, 128, 128))
        fill = flattened.new_tensor([float(v) / 255.0 for v in pad_rgb]).view(1, -1, 1, 1)
        canvas = fill.expand(flattened.shape[0], flattened.shape[1], target_h, target_w).clone()
        top = (target_h - resized_h) // 2
        left = (target_w - resized_w) // 2
        canvas[:, :, top:top + resized_h, left:left + resized_w] = resized
        return canvas.reshape(*original_shape[:-2], target_h, target_w)

    def _prepare_pixels_for_xvla(self, image_input: Tensor) -> Tensor:
        resize_mode = str(getattr(self.processor, "image_resize_mode", "stretch")).strip().lower()
        if resize_mode == "letterbox":
            return self._letterbox_for_xvla(image_input)
        return self._resize_for_xvla(image_input)

    def _prepare_per_view_history_pixels(
        self,
        views: list[Tensor],
        history_mask: Tensor,
        image_hist_size: Tensor,
        *,
        history_size: int,
        image_hist_size_device: Tensor | None = None,
        history_mask_device: Tensor | None = None,
    ) -> Tensor:
        target_h, target_w = self._xvla_image_size()
        batch_size = int(views[0].shape[0])
        prepared_views: list[Tensor] = []
        for view_idx, image in enumerate(views):
            view_history = int(image.shape[1])
            channels = int(image.shape[2])
            view_output = image.new_zeros(
                (batch_size, history_size, channels, target_h, target_w)
            )
            # Metadata stays on the CPU. It controls grouping only; the
            # device copies are used solely for indexing image tensors.
            hist_for_view_host = image_hist_size[:, view_idx]
            if bool((hist_for_view_host > view_history).any()):
                raise ValueError(
                    "Per-view image history requests more frames than the "
                    f"provided tensor for view {view_idx}: max requested "
                    f"{int(hist_for_view_host.max().item())}, tensor K={view_history}."
                )
            view_valid_host = history_mask[:, view_idx].any(dim=1)
            hist_for_view = (
                image_hist_size_device[:, view_idx]
                if image_hist_size_device is not None
                else image_hist_size.to(
                    device=image.device, dtype=torch.long, non_blocking=True
                )[:, view_idx]
            )
            view_valid = (
                history_mask_device[:, view_idx].any(dim=1)
                if history_mask_device is not None
                else view_valid_host.to(device=image.device, non_blocking=True)
            )
            if not bool(view_valid_host.any()):
                prepared_views.append(view_output)
                continue
            unique_k = sorted(
                int(v)
                for v in torch.unique(hist_for_view_host[view_valid_host]).tolist()
            )
            for k_v in unique_k:
                group_host = view_valid_host & (hist_for_view_host == k_v)
                if not bool(group_host.any()):
                    continue
                group = view_valid & (hist_for_view == k_v)
                resized = self._prepare_pixels_for_xvla(image[group, -k_v:])
                view_output[group, -k_v:] = resized
            prepared_views.append(view_output)
        return torch.stack(prepared_views, dim=1)

    def _prepare_images(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        views: list[Tensor] = []
        history_sizes: list[int] = []
        batch_size: int | None = None
        for key in _IMAGE_KEYS:
            if key not in batch:
                raise KeyError(f"HyXVLA requires batch[{key!r}].")
            image = batch[key]
            if image.ndim == 4:
                image = image.unsqueeze(1)
            if image.ndim != 5:
                raise ValueError(f"{key} must be [B,C,H,W] or [B,K,C,H,W], got {tuple(image.shape)}")
            if batch_size is None:
                batch_size = int(image.shape[0])
            elif batch_size != int(image.shape[0]):
                raise ValueError("All Hy-VLA camera views must have the same batch size.")
            if image.dtype == torch.uint8:
                # Compact transport keeps CPU-side image tensors in uint8.
                # Restore the legacy [0, 1] float range before resize and
                # policy normalization.
                image = image.to(dtype=torch.float32).div_(255.0).to(
                    dtype=self._image_mean.dtype
                )
            history_sizes.append(int(image.shape[1]))
            views.append(image)

        assert batch_size is not None
        history_size = max(history_sizes)
        history_mask = batch.get(_IMAGE_HISTORY_MASK_KEY)
        image_hist_size = batch.get(_IMAGE_HISTORY_SIZE_KEY)
        if (history_mask is None) != (image_hist_size is None):
            raise ValueError(
                "Per-view image history requires both "
                f"{_IMAGE_HISTORY_MASK_KEY!r} and {_IMAGE_HISTORY_SIZE_KEY!r}."
            )
        if history_mask is not None:
            if not isinstance(history_mask, Tensor) or not isinstance(image_hist_size, Tensor):
                raise ValueError("Per-view image-history metadata must be tensors.")
            expected_mask_shape = (batch_size, len(views), history_size)
            if tuple(history_mask.shape) != expected_mask_shape:
                raise ValueError(
                    f"{_IMAGE_HISTORY_MASK_KEY} must be {expected_mask_shape}, "
                    f"got {tuple(history_mask.shape)}."
                )
            expected_size_shape = (batch_size, len(views))
            if tuple(image_hist_size.shape) != expected_size_shape:
                raise ValueError(
                    f"{_IMAGE_HISTORY_SIZE_KEY} must be {expected_size_shape}, "
                    f"got {tuple(image_hist_size.shape)}."
                )
            image_hist_size_device = image_hist_size.to(
                device=views[0].device, dtype=torch.long
            )
            history_mask_device = history_mask.to(
                device=views[0].device, dtype=torch.bool, non_blocking=True
            )
            if bool((image_hist_size < 1).any()) or bool(
                (image_hist_size > history_size).any()
            ):
                raise ValueError(
                    f"{_IMAGE_HISTORY_SIZE_KEY} entries must be in [1, {history_size}]."
                )
            image_input = self._prepare_per_view_history_pixels(
                views,
                history_mask,
                image_hist_size,
                history_size=history_size,
                image_hist_size_device=image_hist_size_device,
                history_mask_device=history_mask_device,
            )
            # Keep the host copies for backend-side control-plane grouping.
            image_mask = history_mask
        else:
            if len(set(history_sizes)) != 1:
                raise ValueError(
                    "Camera views with different history lengths require "
                    f"{_IMAGE_HISTORY_MASK_KEY!r} and {_IMAGE_HISTORY_SIZE_KEY!r}."
                )
            image_input = torch.stack(views, dim=1)
            image_input = self._prepare_pixels_for_xvla(image_input)
            image_mask = None

        image_mean = self._image_mean.to(device=image_input.device)
        image_std = self._image_std.to(device=image_input.device)
        image_input = image_input.to(dtype=image_mean.dtype)
        image_input = (image_input - image_mean) / image_std

        if history_size == 1 and image_hist_size is None:
            image_input = image_input[:, :, 0]
            return (
                image_input,
                torch.ones(
                    (batch_size, len(views)), dtype=torch.bool, device=image_input.device
                ),
                None,
            )
        if image_mask is None:
            image_mask = torch.ones(
                (batch_size, len(views), history_size),
                dtype=torch.bool,
                device=image_input.device,
            )
        return image_input, image_mask, image_hist_size

    def _prepare_inputs(self, batch: dict[str, Any], *, include_action: bool) -> dict[str, Tensor]:
        reference = self._first_tensor(batch)
        image_input, image_mask, image_hist_size = self._prepare_images(batch)
        state = batch[_STATE_KEY]
        if state.ndim != 2 or state.shape[-1] < _HY_ACTION_DIM:
            raise ValueError(f"{_STATE_KEY} must be [B,>={_HY_ACTION_DIM}], got {tuple(state.shape)}")
        task = batch.get("task")
        if not isinstance(task, (list, tuple)) or len(task) != state.shape[0]:
            raise ValueError("HyXVLA requires batch['task'] as one instruction per sample.")
        dm05_native = self.xvla.config.vlm_backend == "dm05"
        if dm05_native:
            dm05_state_dim = int(getattr(self.xvla.config, "dm05_state_dim", _EE6D_ACTION_DIM))
            language_inputs = self.processor.encode_dm05_multimodal(
                list(task),
                state,
                num_images=int(image_input.shape[1]),
                state_dim=dm05_state_dim,
                control_mode="joint position" if dm05_state_dim == 14 else "end effector",
            )
            input_ids = language_inputs["input_ids"].to(device=reference.device)
            token_type_ids = language_inputs["token_type_ids"].to(device=reference.device)
            attention_mask = language_inputs["attention_mask"].to(device=reference.device)
        else:
            input_ids = self.processor.encode_language(list(task))["input_ids"].to(device=reference.device)
            token_type_ids = None
            pad_token_id = int(self.processor.tokenizer.pad_token_id)
            attention_mask = (input_ids != pad_token_id).to(dtype=torch.long)
        pad_token_id = int(self.processor.tokenizer.pad_token_id)
        action_mode = str(getattr(self.xvla.config, "action_mode", "ee6d_padded")).strip().lower()
        dm05_native_actions = dm05_native
        if dm05_native_actions:
            action_dim = int(getattr(self.xvla.config, "max_action_dim", 32) or 32)
            if action_dim != 32:
                raise ValueError(
                    "DM05 native action head requires max_action_dim=32, "
                    f"got {action_dim}."
                )
            proprio = state[:, :action_dim]
            # Native DM05 consumes only the configured state/action prefix;
            # the remaining head channels are padding and must not contribute
            # to conditioning or flow-matching loss.
            proprio_valid_mask = state.new_zeros((state.shape[0], action_dim))
            proprio_valid_mask[:, :dm05_state_dim] = 1.0
            action_valid_mask = state.new_zeros(
                (state.shape[0], self.xvla.num_actions, action_dim),
            )
            action_valid_mask[:, :, :dm05_state_dim] = 1.0
        elif action_mode == "cross_emb":
            action_dim = int(getattr(self.xvla.config, "max_action_dim", 64) or 64)
            if action_dim < max(_CROSS_EMB_EE_INDICES) + 1:
                raise ValueError(
                    "CrossEmb action space is too narrow for the canonical EE layout: "
                    f"max_action_dim={action_dim}."
                )
            ee_indices = torch.tensor(
                _CROSS_EMB_EE_INDICES, dtype=torch.long, device=state.device
            )
            # The public Hy batch remains 32-wide with EE6D in [0:20].
            # CrossEmb keeps the two EE slots separated by the hand-action
            # region: left EE [0:10], right EE [30:40].
            proprio = state.new_zeros((state.shape[0], action_dim))
            proprio[:, ee_indices] = state[:, :_EE6D_ACTION_DIM]
            proprio_valid_mask = state.new_zeros((state.shape[0], action_dim))
            proprio_valid_mask[:, ee_indices] = 1.0
            action_valid_mask = state.new_zeros(
                (state.shape[0], self.xvla.num_actions, action_dim)
            )
            action_valid_mask[:, :, ee_indices] = 1.0
        else:
            action_dim = _HY_ACTION_DIM
            proprio = state[:, :action_dim]
            proprio_valid_mask = torch.cat(
                [
                    torch.ones_like(state[:, :_EE6D_ACTION_DIM]),
                    torch.zeros_like(state[:, _EE6D_ACTION_DIM:action_dim]),
                ],
                dim=-1,
            )
            action_valid_mask = torch.cat(
                [
                    torch.ones(
                        (state.shape[0], self.xvla.num_actions, _EE6D_ACTION_DIM),
                        dtype=state.dtype,
                        device=state.device,
                    ),
                    torch.zeros(
                        (state.shape[0], self.xvla.num_actions, action_dim - _EE6D_ACTION_DIM),
                        dtype=state.dtype,
                        device=state.device,
                    ),
                ],
                dim=-1,
            )

        result: dict[str, Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_input": image_input.to(device=reference.device, dtype=reference.dtype),
            # Per-view history masks are CPU control-plane metadata. The Qwen
            # backend copies them to the vision device once when needed and
            # uses the host copy for Python grouping and packing.
            "image_mask": image_mask,
            "domain_id": torch.full(
                (state.shape[0],),
                self.domain_id,
                dtype=torch.long,
                device=reference.device,
            ),
            "proprio": proprio,
            "proprio_valid_mask": proprio_valid_mask,
            "action_valid_mask": action_valid_mask,
        }
        if token_type_ids is not None:
            result["token_type_ids"] = token_type_ids
        if image_hist_size is not None:
            result["image_hist_size"] = image_hist_size
        # All three stacks come directly from the Hy-Embodied collator. The
        # legacy symmetric path omits ``image_hist_size``; opt-in per-view
        # history forwards it so X-VLA can sparsify short wrist stacks.
        if include_action:
            action = batch.get(_ACTION_KEY)
            if not isinstance(action, Tensor) or action.ndim != 3 or action.shape[-1] < _HY_ACTION_DIM:
                raise ValueError(f"{_ACTION_KEY} must be [B,T,>={_HY_ACTION_DIM}], got {getattr(action, 'shape', None)}")
            if action.shape[1] != self.xvla.num_actions:
                raise ValueError(
                    f"action chunk length {action.shape[1]} does not match X-VLA num_actions {self.xvla.num_actions}."
                )
            if action_mode == "cross_emb":
                action_wide = action.new_zeros(
                    (action.shape[0], action.shape[1], action_dim)
                )
                action_wide[:, :, ee_indices] = action[:, :, :_EE6D_ACTION_DIM]
                result["action"] = action_wide
            else:
                result["action"] = action[:, :, :_HY_ACTION_DIM]
        return result



    @torch.no_grad()
    def forward_evaluate(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        inputs = self._prepare_inputs(batch, include_action=False)
        if self.preserve_padding_noise_inference:
            # Diagnostic-only override: allow the padding tail to retain its
            # Gaussian sampler state. This intentionally differs from training.
            inputs.pop("action_valid_mask")
        prediction = self.xvla.generate_actions(
            **inputs,
            steps=self.inference_steps,
            # Keep the Hy flow/regression output in model space here.  The
            # normal CrossEmb-preserve path has its incompatible source norm
            # disabled at load time; the guarded inverse below is only a
            # compatibility fallback for older/manual objects.
            apply_postprocess=(
                self.action_loss_profile == "official_ee6d_regression"
            ),
        )
        if str(getattr(self.xvla.config, "action_mode", "")).strip().lower() == "cross_emb":
            if (
                self.cross_emb_action_transfer in {"preserve", "pretrained"}
                and self.action_loss_profile != "official_ee6d_regression"
            ):
                # ``generate_actions`` returns CrossEmb model space when
                # postprocess is disabled.  Undo only an accidentally active
                # source norm; do not call ``postprocess`` because it clamps
                # velocity grippers to [0, 1], while Hy expects z-scored values.
                if bool(self.xvla.action_space._norm_enabled.item()):
                    prediction = self.xvla.action_space._denormalize(prediction)
            ee_indices = torch.tensor(
                _CROSS_EMB_EE_INDICES, dtype=torch.long, device=prediction.device
            )
            prediction = prediction.index_select(-1, ee_indices)
        result = {"pred": prediction}
        action = batch.get(_ACTION_KEY)
        if isinstance(action, Tensor):
            # Keep prediction and GT in the same public space for validation.
            # CrossEmb predictions are gathered to 20-D above; preserve the
            # historical 32-D GT for the ee6d_padded path.
            if str(getattr(self.xvla.config, "action_mode", "")).strip().lower() == "cross_emb":
                result["gt"] = action[:, :, :_EE6D_ACTION_DIM]
            else:
                result["gt"] = action[:, :, :_HY_ACTION_DIM]
        return result

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        del noise
        self.eval()
        if not self._action_queue:
            actions = self.forward_evaluate(batch)["pred"]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
