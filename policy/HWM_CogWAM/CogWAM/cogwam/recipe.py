"""The frozen CogWAM reproduction contract.

Everything that defines *what was trained* is pinned here and checked before a
single parameter is allocated, so a config drift costs seconds rather than a
GPU-hour. The check is a flat key/value comparison plus a SHA-256 fingerprint
over the canonicalised payload; the fingerprint is written next to each run so a
checkpoint can always be traced back to the exact configuration that produced it.

What is deliberately *not* frozen:

* Filesystem locations. Upstream froze absolute storage paths into the contract,
  which made the recipe unusable outside one cluster. Here the backbone and data
  locations come from ``COGWAM_BASE_VLM`` / ``COGWAM_DINO_MODEL`` /
  ``COGWAM_DATA_ROOT`` / ``COGWAM_RUN_ROOT`` and are validated for *identity*
  (``model_type``, hidden sizes) rather than for *path*.
* ``trainer.is_resume`` / ``trainer.resume_state_path``. These describe how one
  process restores an already-defined run, and legitimately differ between an
  initial launch, a resumed launch, and checkpoint evaluation. They must not
  affect the fingerprint stored with the weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

RECIPE_PROFILE = "cogwam_robodojo_causal_dino_mot_h25_eventmem_dino_multilayer_50k_v1"

# Identifies the state-dict layout the released weights use: one shared context
# projection feeding the state and world streams. Bump this string if the module
# tree changes in a way that invalidates published checkpoints.
WEIGHT_CONTRACT = "shared_context_state_world_v1"

GLOBAL_BATCH_SIZE = 768
MAX_TRAIN_STEPS = 50000

# Environment variables that stand in for the absolute paths a run needs.
ENV_BASE_VLM = "COGWAM_BASE_VLM"
ENV_DINO_MODEL = "COGWAM_DINO_MODEL"
ENV_DATA_ROOT = "COGWAM_DATA_ROOT"
ENV_RUN_ROOT = "COGWAM_RUN_ROOT"

_MISSING = object()


def _get_child(value: Any, key: str, default: Any = _MISSING) -> Any:
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            result = getter(key, _MISSING)
        except TypeError:
            result = _MISSING
        if result is not _MISSING:
            return result
    if isinstance(value, dict) and key in value:
        return value[key]
    if hasattr(value, key):
        return getattr(value, key)
    if default is not _MISSING:
        return default
    raise KeyError(key)


def _select(config: Any, path: str, default: Any = _MISSING) -> Any:
    value = config
    for key in path.split("."):
        try:
            value = _get_child(value, key)
        except KeyError:
            if default is not _MISSING:
                return default
            raise KeyError(path) from None
    return value


def _plain(value: Any) -> Any:
    """Reduce OmegaConf / namespace containers to plain JSON-comparable values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "items"):
        try:
            return {str(key): _plain(item) for key, item in value.items()}
        except TypeError:
            pass
    if isinstance(value, (list, tuple)) or (hasattr(value, "__iter__") and not isinstance(value, (str, bytes))):
        return [_plain(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

EXPECTED: dict[str, Any] = {
    "seed": 42,
    "version_id": "0.21",
    # --- framework -----------------------------------------------------------
    "framework.name": "CogWAM",
    "framework.enable_world_action_mot": True,
    "framework.reproduction_profile": RECIPE_PROFILE,
    "framework.qwenvl.attn_implementation": "sdpa",
    "framework.qwenvl.require_attn_implementation": True,
    "framework.qwenvl.enable_gradient_checkpointing": True,
    "framework.qwenvl.enable_thinking": False,
    "framework.qwenvl.vl_hidden_dim": 2048,
    "framework.qwenvl.truncate_vlm_layers": 0,
    # --- planner -------------------------------------------------------------
    "framework.planner.num_world_queries": 16,
    # Must equal action_horizon: the query bank's RoPE table is sliced to the
    # horizon, so the two cannot drift apart.
    "framework.planner.num_action_queries": 25,
    "framework.planner.world_placeholder_token": "<WORLD_PLAN>",
    "framework.planner.action_placeholder_token": "<ACTION_PLAN>",
    # --- event-driven semantic memory ----------------------------------------
    "framework.planner.text_supervision.enabled": True,
    "framework.planner.text_supervision.mode": "event_driven_memory_ntp",
    "framework.planner.text_supervision.keep_token": "<KEEP>",
    "framework.planner.text_supervision.update_token": "<UPDATE>",
    "framework.planner.text_supervision.subtask_field": "subtask_text",
    "framework.planner.text_supervision.prompt_template": (
        "Task instruction: {instruction}\n"
        "Semantic Memory: {semantic_memory}\n"
        "Cached Current Subtask: {cached_current_subtask}\n"
        "Decide whether the cached subtask is still valid. Reply with <KEEP> "
        "only when it remains valid. Otherwise begin with <UPDATE>, then provide "
        "only `Memory Add:` and `Current Subtask:`."
    ),
    "framework.planner.text_supervision.update_response_template": (
        "{update_token}\nMemory Add: {memory_add}\nCurrent Subtask: {subtask_text}"
    ),
    "framework.planner.text_supervision.max_new_tokens": 96,
    "framework.planner.text_supervision.do_sample": False,
    "framework.planner.text_supervision.scheduled_sampling.enabled": True,
    # p(model's own semantic state is used as physical conditioning), linearly
    # interpolated by optimizer step. Sampled text never supervises itself.
    "framework.planner.text_supervision.scheduled_sampling.points": [
        [0, 0.0],
        [30000, 0.0],
        [40000, 0.2],
        [50000, 0.5],
    ],
    # Event memory is the no-visual-history path; the two are mutually exclusive.
    "framework.planner.text_supervision.history.enabled": False,
    # --- world/action mixture-of-transformers --------------------------------
    "framework.world_action_mot.architecture": "causal_dino_mot",
    "framework.world_action_mot.interaction_mode": "base",
    "framework.world_action_mot.world_attention_mask_mode": "first_frame_causal",
    # `inherit` makes the action shell follow the launcher dtype (BF16 under the
    # released DeepSpeed and policy-server entrypoints) instead of forcing an
    # explicit FP32 boundary.
    "framework.world_action_mot.action_precision_mode": "inherit",
    "framework.world_action_mot.world_hidden_size": 512,
    "framework.world_action_mot.action_hidden_size": 1024,
    "framework.world_action_mot.world_ffn_dim": 2048,
    "framework.world_action_mot.action_ffn_dim": 4096,
    "framework.world_action_mot.num_layers": 30,
    "framework.world_action_mot.num_attention_heads": 24,
    "framework.world_action_mot.attention_head_dim": 128,
    "framework.world_action_mot.layerwise_planner_coupling": False,
    "framework.world_action_mot.norm_eps": 1.0e-6,
    "framework.world_action_mot.time_frequency_dim": 256,
    "framework.world_action_mot.world_grid_height": 12,
    "framework.world_action_mot.world_grid_width": 10,
    "framework.world_action_mot.max_world_tokens": 120,
    "framework.world_action_mot.enable_gradient_checkpointing": True,
    "framework.world_action_mot.world_train_shift": 5.0,
    "framework.world_action_mot.world_infer_shift": 5.0,
    "framework.world_action_mot.world_num_train_timesteps": 1000,
    "framework.world_action_mot.action_train_shift": 5.0,
    "framework.world_action_mot.action_infer_shift": 5.0,
    "framework.world_action_mot.action_num_train_timesteps": 1000,
    "framework.world_action_mot.num_inference_timesteps": 20,
    "framework.world_action_mot.action_prediction_type": "velocity",
    "framework.world_action_mot.action_velocity_target": "noise_minus_clean",
    "framework.world_action_mot.jit_t_eps": 0.05,
    "framework.world_action_mot.repeated_diffusion_steps": 1,
    "framework.world_action_mot.action_loss_weight": 1.0,
    "framework.world_action_mot.world_loss_weight": 1.0,
    # Semantic objective = 0.005 * (decision CE + UPDATE-body CE).
    "framework.world_action_mot.text_loss_weight": 0.005,
    # --- DINO ----------------------------------------------------------------
    "framework.dino.name": "dinov3_vitb16",
    "framework.dino.model_size": "base",
    "framework.dino.repo_or_dir": "facebookresearch/dinov3",
    "framework.dino.loader": "hf",
    "framework.dino.image_size": [384, 320],
    "framework.dino.patch_size": 16,
    "framework.dino.embed_dim": 768,
    "framework.dino.stats_path": None,
    "framework.dino.load_live_backbone": True,
    "framework.dino.force_online": True,
    "framework.dino.dino_pool": 2,
    "framework.dino.current_dino_pool": None,
    # The clean current prefix and the future denoising target both mean-fuse
    # normalized patch tokens from four depths instead of the last layer alone.
    # They share one `world_input` projection, so both sides move together and
    # every weight shape is identical to the single-layer control.
    "framework.dino.dino_layers": [2, 5, 8, 11],
    # --- action space --------------------------------------------------------
    "framework.action_model.action_dim": 14,
    "framework.action_model.state_dim": 14,
    "framework.action_model.action_horizon": 25,
    # --- data ----------------------------------------------------------------
    "datasets.vla_data.dataset_py": "jointflow",
    "datasets.vla_data.data_mix": "robodojo_v21_language",
    "datasets.vla_data.lerobot_version": "v2.1",
    "datasets.vla_data.image_layout": "tri_view_composite",
    "datasets.vla_data.composite_source_view_keys": [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ],
    "datasets.vla_data.composite_view_key": "video.tri_view_composite",
    "datasets.vla_data.include_state": True,
    "datasets.vla_data.action_type": "abs_qpos",
    "datasets.vla_data.action_mode": "abs",
    "datasets.vla_data.action_horizon": 25,
    "datasets.vla_data.world_model.future_stride": 16,
    "datasets.vla_data.future_valid_requires_full_stride": True,
    "datasets.vla_data.decode_future_video": True,
    "datasets.vla_data.online_dino": True,
    "datasets.vla_data.dino_target_latents": False,
    "datasets.vla_data.optional_text_annotations.enabled": False,
    "datasets.vla_data.optional_text_annotations.require_columns": False,
    "datasets.vla_data.text_annotations.enabled": True,
    "datasets.vla_data.text_annotations.fields.subtask_text": "subtask_text",
    "datasets.vla_data.text_annotations.fields.completed_subtask_text": "complete_text",
    "datasets.vla_data.text_annotations.history.enabled": False,
    "datasets.vla_data.text_annotations.event_memory.enabled": True,
    "datasets.vla_data.text_annotations.event_memory.semantic_offset": -10,
    "datasets.vla_data.text_annotations.event_memory.replan_interval": 10,
    "datasets.vla_data.text_annotations.event_memory.replan_phase": 0,
    "datasets.vla_data.text_annotations.event_memory.normalization": "n1",
    "datasets.vla_data.text_annotations.event_memory.empty_memory": "None.",
    "datasets.vla_data.text_annotations.event_memory.empty_cached_subtask": "None.",
    "datasets.vla_data.text_annotations.event_memory.semantic_memory_field": "semantic_memory",
    "datasets.vla_data.text_annotations.event_memory.cached_subtask_field": "cached_current_subtask",
    "datasets.vla_data.text_annotations.event_memory.decision_field": "semantic_decision",
    "datasets.vla_data.text_annotations.event_memory.memory_add_field": "memory_add",
    "datasets.vla_data.text_annotations.event_memory.cache_valid_field": "semantic_cache_valid",
    "datasets.vla_data.text_annotations.event_memory.index_cache_name": "semantic_index_v1.npz",
    "datasets.vla_data.text_annotations.event_memory.expected_phase_update_ratio": 0.064,
    "datasets.vla_data.text_annotations.event_memory.expected_phase_update_tolerance": 0.005,
    "datasets.vla_data.text_annotations.event_memory.sampler.per_device_batch_size": 6,
    "datasets.vla_data.text_annotations.event_memory.sampler.update_per_batch": 2,
    "datasets.vla_data.text_annotations.event_memory.sampler.hard_keep_per_batch": 2,
    "datasets.vla_data.text_annotations.event_memory.sampler.random_keep_per_batch": 2,
    "datasets.vla_data.text_annotations.event_memory.sampler.seed": 42,
    "datasets.vla_data.text_annotations.event_memory.sampler.num_workers": 4,
    "datasets.vla_data.text_annotations.event_memory.sampler.prefetch_factor": 2,
    "datasets.vla_data.text_annotations.event_memory.sampler.pin_memory": True,
    "datasets.vla_data.text_annotations.event_memory.sampler.persistent_workers": False,
    "datasets.vla_data.sequential_step_sampling": False,
    "datasets.vla_data.balance_dataset_weights": False,
    "datasets.vla_data.balance_trajectory_weights": False,
    "datasets.vla_data.per_device_batch_size": 12,
    "datasets.vla_data.load_all_data_for_training": True,
    "datasets.vla_data.obs_image_size": [320, 384],
    "datasets.vla_data.video_backend": "pyav",
    "datasets.vla_data.num_workers": 8,
    "datasets.vla_data.prefetch_factor": 2,
    "datasets.vla_data.pin_memory": True,
    "datasets.vla_data.persistent_workers": False,
    # --- optimisation --------------------------------------------------------
    "trainer.seed_before_model_init": True,
    "trainer.expected_global_batch_size": GLOBAL_BATCH_SIZE,
    "trainer.max_train_steps": MAX_TRAIN_STEPS,
    "trainer.num_warmup_steps": 2000,
    "trainer.save_interval": 10000,
    "trainer.eval_interval": 1000,
    # RoboDojo evaluation maintains persistent semantic state across a rollout.
    # The trainer's stateless action-MSE probe would measure the wrong thing.
    "trainer.action_eval_enabled": False,
    "trainer.world_validation.enabled": False,
    "trainer.pretrained_checkpoint": None,
    "trainer.save_full_training_state": True,
    "trainer.learning_rate.base": 1.0e-5,
    "trainer.learning_rate.qwen_vl_interface": 1.0e-5,
    "trainer.learning_rate.action_plan_queries": 1.0e-4,
    "trainer.learning_rate.world_plan_queries": 1.0e-4,
    "trainer.learning_rate.action_model": 1.0e-4,
    "trainer.lr_scheduler_type": "cosine_with_min_lr",
    "trainer.scheduler_specific_kwargs.min_lr": 5.0e-7,
    "trainer.freeze_modules": "",
    "trainer.loss_scale.vla": 1.0,
    "trainer.max_grad_norm": 1.0,
    "trainer.weight_decay": 0.0,
    "trainer.logging_frequency": 200,
    "trainer.log_grad_norms": False,
    "trainer.log_global_grad_norm": True,
    "trainer.deepspeed_skip_unused_param_anchors": True,
    "trainer.gradient_clipping": 1.0,
    "trainer.gradient_accumulation_steps": 1,
    "trainer.optimizer.name": "AdamW",
    "trainer.optimizer.betas": [0.9, 0.95],
    "trainer.optimizer.eps": 1.0e-8,
    "trainer.optimizer.weight_decay": 1.0e-8,
}


def contract_payload(config: Any) -> dict[str, Any]:
    """Canonical key/value payload the fingerprint is computed over."""
    return {path: _plain(_select(config, path, "<MISSING>")) for path in EXPECTED}


def config_fingerprint(config: Any) -> str:
    encoded = json.dumps(
        contract_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_recipe(config: Any) -> dict[str, Any]:
    """Fail fast unless *config* is the frozen CogWAM contract.

    Returns a small audit record worth writing next to the run.

    ``COGWAM_SKIP_RECIPE_VALIDATION=1`` downgrades the failure to a warning so a
    deliberately-shrunk smoke test can run. The returned record then carries
    ``validated: False`` -- a run recorded that way is not a reproduction of this
    recipe, and the audit file says so.
    """
    mismatches: dict[str, dict[str, Any]] = {}
    for path, expected_value in EXPECTED.items():
        try:
            actual = _plain(_select(config, path))
        except KeyError:
            actual = "<MISSING>"
        if actual != expected_value:
            mismatches[path] = {"actual": actual, "expected": expected_value}

    record = {
        "profile": RECIPE_PROFILE,
        "weight_contract": WEIGHT_CONTRACT,
        "fingerprint": config_fingerprint(config),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "max_train_steps": MAX_TRAIN_STEPS,
        "validated": True,
    }
    if not mismatches:
        return record

    details = "; ".join(
        f"{path}={values['actual']!r} (expected {values['expected']!r})" for path, values in sorted(mismatches.items())
    )
    message = f"Invalid {RECIPE_PROFILE} reproduction contract: {details}"
    if os.environ.get("COGWAM_SKIP_RECIPE_VALIDATION", "") not in {"", "0", "false", "False"}:
        warnings.warn(f"COGWAM_SKIP_RECIPE_VALIDATION is set; NOT a reproduction run. {message}", stacklevel=2)
        record["validated"] = False
        record["mismatches"] = mismatches
        return record
    raise ValueError(message)


# ---------------------------------------------------------------------------
# Runtime assets
# ---------------------------------------------------------------------------

# What the two backbone directories must contain for the released weights to
# load. Checked by identity rather than by path so the recipe travels.
_VLM_EXPECTED = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}
_DINO_EXPECTED = {"model_type": "dinov3_vit", "hidden_size": 768, "num_hidden_layers": 12, "patch_size": 16}


def _read_hf_config(directory: Path) -> dict[str, Any]:
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{directory} is not a Hugging Face model directory (no config.json)")
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_asset_dir(env_var: str, configured: str | None = None) -> Path:
    """Resolve a backbone directory from config, falling back to its env var."""
    raw = configured or os.environ.get(env_var)
    if not raw:
        raise EnvironmentError(f"Set {env_var} to a local Hugging Face model directory (see README.md)")
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{env_var}={raw} is not a directory")
    return path


def validate_runtime_assets(config: Any) -> dict[str, Any]:
    """Verify the two backbones are the ones this recipe was trained against.

    Shape mismatches would surface later as an unreadable checkpoint; this
    catches "pointed at the wrong Qwen" in milliseconds instead.
    """
    vlm_dir = resolve_asset_dir(ENV_BASE_VLM, _select(config, "framework.qwenvl.base_vlm", None))
    dino_dir = resolve_asset_dir(ENV_DINO_MODEL, _select(config, "framework.dino.hf_model_id", None))

    vlm_config = _read_hf_config(vlm_dir)
    for key, expected_value in _VLM_EXPECTED.items():
        actual = vlm_config.get(key)
        if actual != expected_value:
            raise ValueError(f"{ENV_BASE_VLM}: config.json {key}={actual!r}, expected {expected_value!r}")
    text_config = vlm_config.get("text_config", vlm_config)
    hidden_size = text_config.get("hidden_size")
    expected_hidden = int(_select(config, "framework.qwenvl.vl_hidden_dim", 2048))
    if hidden_size != expected_hidden:
        raise ValueError(f"{ENV_BASE_VLM}: text hidden_size={hidden_size}, expected {expected_hidden}")

    dino_config = _read_hf_config(dino_dir)
    for key, expected_value in _DINO_EXPECTED.items():
        actual = dino_config.get(key)
        if actual != expected_value:
            raise ValueError(f"{ENV_DINO_MODEL}: config.json {key}={actual!r}, expected {expected_value!r}")

    return {
        "vlm_dir": str(vlm_dir),
        "vlm_model_type": vlm_config.get("model_type"),
        "vlm_hidden_size": hidden_size,
        "dino_dir": str(dino_dir),
        "dino_model_type": dino_config.get("model_type"),
    }
