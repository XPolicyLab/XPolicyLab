"""XPolicyLab server entry point for VAP."""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from XPolicyLab.model_template import ModelTemplate

from .profile import LEGACY_PROFILE, resolve_evaluation_profile
from .state import VAPPolicyState


def _value(
    config: Mapping[str, Any], key: str, env_key: str, default: Any = None
) -> Any:
    value = os.environ.get(env_key)
    if value is not None and value != "":
        return value
    return config.get(key, default)


def _bool_value(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _gripper_hysteresis_value(config: Mapping[str, Any]) -> bool:
    """Resolve the canonical knob and its former launch-script alias safely."""

    canonical = os.environ.get("VAP_GRIPPER_HYSTERESIS")
    legacy = os.environ.get("VAP_GRIPPER_POSTPROCESS")
    if canonical not in (None, "") and legacy not in (None, ""):
        canonical_value = _bool_value(canonical, name="VAP_GRIPPER_HYSTERESIS")
        legacy_value = _bool_value(legacy, name="VAP_GRIPPER_POSTPROCESS")
        if canonical_value != legacy_value:
            raise ValueError(
                "VAP_GRIPPER_HYSTERESIS and deprecated "
                "VAP_GRIPPER_POSTPROCESS disagree"
            )
        return canonical_value
    if canonical not in (None, ""):
        return _bool_value(canonical, name="VAP_GRIPPER_HYSTERESIS")
    if legacy not in (None, ""):
        print(
            "[VAP_CONFIG_DEPRECATION] VAP_GRIPPER_POSTPROCESS is deprecated; "
            "use VAP_GRIPPER_HYSTERESIS",
            flush=True,
        )
        return _bool_value(legacy, name="VAP_GRIPPER_POSTPROCESS")
    return _bool_value(
        config.get("gripper_hysteresis", False), name="gripper_hysteresis"
    )


def _rgb_gain_value(value: Any) -> tuple[float, float, float] | None:
    if value in (None, "", "null", "none"):
        return None
    fields = str(value).replace(",", " ").split()
    if len(fields) != 3:
        raise ValueError(
            "VAP_IMAGE_RGB_GAIN must contain exactly three RGB gains, " f"got {value!r}"
        )
    gain = tuple(float(field) for field in fields)
    if any(not (0.0 < channel < float("inf")) for channel in gain):
        raise ValueError(
            f"VAP_IMAGE_RGB_GAIN values must be finite and positive, got {gain!r}"
        )
    return gain


class Model(ModelTemplate):
    """One stateful VAP runtime per XPolicyLab model process."""

    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        self.model_cfg = dict(model_cfg)
        repo = _value(model_cfg, "vap_repo", "VAP_REPO")
        config = _value(model_cfg, "vap_config", "VAP_CONFIG")
        checkpoint = _value(model_cfg, "vap_checkpoint", "VAP_CHECKPOINT")
        if not repo or not config or not checkpoint:
            raise ValueError("VAP_REPO, VAP_CONFIG and VAP_CHECKPOINT are required")
        repo_path = Path(str(repo)).expanduser().resolve()
        if not (repo_path / "wall_x" / "model" / "vap").is_dir():
            raise FileNotFoundError(f"invalid VAP_REPO: {repo_path}")
        sys.path.insert(0, str(repo_path))

        from wall_x.config.loader import load_config

        evaluation_profile = resolve_evaluation_profile(
            _value(
                model_cfg,
                "evaluation_profile",
                "VAP_ROBODOJO_PROFILE",
                LEGACY_PROFILE,
            )
        )
        checkpoint_config = load_config(str(Path(str(config)).expanduser().resolve()))
        # Fail before loading the 5B weights when a client profile and the
        # materialized training config disagree.
        evaluation_profile.validate_config(checkpoint_config)
        if evaluation_profile.name != LEGACY_PROFILE:
            import yaml

            raw_checkpoint_config = yaml.safe_load(
                Path(str(config)).expanduser().resolve().read_text()
            )
            evaluation_profile.validate_raw_config(raw_checkpoint_config)

        inference_steps = _value(
            model_cfg,
            "action_inference_steps",
            "VAP_ACTION_INFERENCE_STEPS",
            16,
        )
        if inference_steps in (None, "", "null"):
            inference_steps = None
        else:
            inference_steps = int(inference_steps)
        inference_shift = _value(
            model_cfg,
            "action_inference_shift",
            "VAP_ACTION_INFERENCE_SHIFT",
        )
        if inference_shift in (None, "", "null"):
            inference_shift = None
        else:
            inference_shift = float(inference_shift)
        normalizer_key = _value(model_cfg, "normalizer_key", "VAP_NORMALIZER_KEY")
        if normalizer_key in (None, "", "null"):
            normalizer_key = evaluation_profile.normalizer_key
        else:
            normalizer_key = str(normalizer_key)
        if (
            evaluation_profile.normalizer_key is not None
            and normalizer_key != evaluation_profile.normalizer_key
        ):
            raise ValueError(
                f"profile {evaluation_profile.name!r} requires "
                f"VAP_NORMALIZER_KEY={evaluation_profile.normalizer_key!r}, "
                f"got {normalizer_key!r}"
            )
        state_conditioning = _bool_value(
            _value(
                model_cfg,
                "state_conditioning",
                "VAP_STATE_CONDITIONING",
                (
                    evaluation_profile.state_conditioning_required
                    if evaluation_profile.state_conditioning_required is not None
                    else False
                ),
            ),
            name="VAP_STATE_CONDITIONING",
        )
        if (
            evaluation_profile.state_conditioning_required is not None
            and state_conditioning != evaluation_profile.state_conditioning_required
        ):
            raise ValueError(
                f"profile {evaluation_profile.name!r} requires "
                f"VAP_STATE_CONDITIONING="
                f"{evaluation_profile.state_conditioning_required}"
            )
        normalizer_stats_path = _value(
            model_cfg,
            "normalizer_stats_path",
            "VAP_NORMALIZER_STATS",
        )
        if normalizer_stats_path in (None, "", "null", "none"):
            normalizer_stats_path = None
        elif not evaluation_profile.allow_config_stats_without_checkpoint_state:
            raise ValueError(
                "VAP_NORMALIZER_STATS override is only allowed by an explicit "
                f"evaluation profile; {evaluation_profile.name!r} forbids it"
            )
        else:
            normalizer_stats_path = str(normalizer_stats_path)
            evaluation_profile.validate_stats_file(normalizer_stats_path)
        text_feature_cache_path = _value(
            model_cfg,
            "text_feature_cache_path",
            "VAP_TEXT_FEATURE_CACHE",
        )
        if text_feature_cache_path in (None, "", "null", "none"):
            text_feature_cache_path = None
        else:
            text_feature_cache_path = str(text_feature_cache_path)
        text_feature_cache_online_fallback = _bool_value(
            _value(
                model_cfg,
                "text_feature_cache_online_fallback",
                "VAP_TEXT_FEATURE_CACHE_ONLINE_FALLBACK",
                False,
            ),
            name="VAP_TEXT_FEATURE_CACHE_ONLINE_FALLBACK",
        )
        if text_feature_cache_online_fallback and text_feature_cache_path is None:
            raise ValueError(
                "VAP_TEXT_FEATURE_CACHE_ONLINE_FALLBACK requires "
                "VAP_TEXT_FEATURE_CACHE"
            )
        use_kv_cache = _bool_value(
            _value(model_cfg, "use_kv_cache", "VAP_USE_KV_CACHE", True),
            name="use_kv_cache",
        )
        vae_tiled = _bool_value(
            _value(model_cfg, "vae_tiled", "VAP_VAE_TILED", False),
            name="vae_tiled",
        )
        device = str(_value(model_cfg, "device", "VAP_DEVICE", "cuda:0"))
        if checkpoint_config.model_type == "vapv2":
            if (
                checkpoint_config.model.architecture_version
                != "vapv2.qwen_joint_ivava.v2"
            ):
                raise ValueError(
                    "RoboDojo supports only the joint-v2 VAPv2 architecture"
                )
            if text_feature_cache_online_fallback:
                raise ValueError("VAPv2 does not support legacy T5 cache fallback")
            if state_conditioning != bool(checkpoint_config.model.state_film_enabled):
                raise ValueError(
                    "VAP_STATE_CONDITIONING differs from the VAPv2 checkpoint"
                )
            qwen_online_from_cache = _bool_value(
                _value(
                    model_cfg,
                    "qwen_online_from_cache",
                    "VAP_QWEN_ONLINE_FROM_CACHE",
                    False,
                ),
                name="qwen_online_from_cache",
            )
            qwen_model_path = _value(model_cfg, "qwen_model_path", "VAP_QWEN_MODEL")
            if not qwen_model_path:
                qwen_model_path = None
            from wall_x.model.vapv2.serving import VAPv2ServingRuntime

            # Evaluation-only default; V1 and generic V2 serving stay unchanged.
            action_acceleration = _value(
                model_cfg, "action_acceleration", "VAP_V2_ACTION_ACCELERATION",
                "graph_rope" if use_kv_cache else "off",
            )
            if action_acceleration not in {"off", "rope", "graph_rope"}:
                raise ValueError("VAP_V2_ACTION_ACCELERATION must be off/rope/graph_rope")
            if action_acceleration != "off" and not use_kv_cache:
                raise ValueError("V2 action acceleration requires KV-cache")
            runtime = VAPv2ServingRuntime.from_paths(
                checkpoint_path=str(checkpoint),
                config_path=str(config),
                device=device,
                normalizer_key=normalizer_key,
                use_kv_cache=use_kv_cache,
                action_acceleration=action_acceleration,
                vae_tiled=vae_tiled,
                action_inference_steps=inference_steps,
                action_inference_shift=inference_shift,
                normalizer_stats_path=normalizer_stats_path,
                text_feature_cache_path=text_feature_cache_path,
                qwen_model_path=qwen_model_path,
                qwen_online_from_cache=qwen_online_from_cache,
            )
        elif checkpoint_config.model_type == "vap":
            from wall_x.model.vap.serving import VAPServingRuntime

            runtime = VAPServingRuntime.from_paths(
                config_path=str(config),
                checkpoint_path=str(checkpoint),
                device=device,
                normalizer_key=normalizer_key,
                use_kv_cache=use_kv_cache,
                vae_tiled=vae_tiled,
                action_inference_steps=inference_steps,
                action_inference_shift=inference_shift,
                state_conditioning=state_conditioning,
                normalizer_stats_path=normalizer_stats_path,
                text_feature_cache_path=text_feature_cache_path,
                text_feature_cache_online_fallback=(text_feature_cache_online_fallback),
                require_checkpoint_normalizer_state=(
                    not evaluation_profile.allow_config_stats_without_checkpoint_state
                ),
                allow_checkpoint_proprio_normalizer=(
                    evaluation_profile.allow_checkpoint_proprio_normalizer
                ),
            )
        else:
            raise ValueError(
                f"unsupported RoboDojo model_type={checkpoint_config.model_type!r}"
            )
        # from_paths reconstructs the normalizer from config JSON and requires
        # exact tensor equality with checkpoint/normalizer_action.pth before
        # loading the model. This final check pins the profile's key semantics.
        print(f"[VAP_KV_CACHE] enabled={use_kv_cache} scope=action_chunk "
              "rebuild_each_window=true", flush=True)
        evaluation_profile.validate_runtime(runtime)
        image_rgb_gain = _rgb_gain_value(
            _value(model_cfg, "image_rgb_gain", "VAP_IMAGE_RGB_GAIN")
        )
        height_value = _value(model_cfg, "height", "VAP_IMAGE_HEIGHT")
        width_value = _value(model_cfg, "width", "VAP_IMAGE_WIDTH")
        height = (
            evaluation_profile.height
            if height_value in (None, "", "null", "none")
            else int(height_value)
        )
        width = (
            evaluation_profile.width
            if width_value in (None, "", "null", "none")
            else int(width_value)
        )
        if (height, width) != (
            evaluation_profile.height,
            evaluation_profile.width,
        ):
            raise ValueError(
                f"profile {evaluation_profile.name!r} requires image "
                f"{evaluation_profile.width}x{evaluation_profile.height}, got "
                f"{width}x{height}"
            )
        fixed_action_chunk_seed_value = _value(
            model_cfg,
            "fixed_action_chunk_seed",
            "VAP_FIXED_ACTION_CHUNK_SEED",
        )
        fixed_action_chunk_seed = (
            None
            if fixed_action_chunk_seed_value in (None, "", "null", "none")
            else int(fixed_action_chunk_seed_value)
        )
        self.policy = VAPPolicyState(
            runtime,
            height=height,
            width=width,
            base_seed=int(_value(model_cfg, "base_seed", "VAP_BASE_SEED", 0)),
            fixed_action_chunk_seed=fixed_action_chunk_seed,
            inference_mode=str(
                _value(model_cfg, "inference_mode", "VAP_INFERENCE_MODE", "ar")
            ),
            ar_prefix_action_source=str(
                _value(
                    model_cfg,
                    "ar_prefix_action_source",
                    "VAP_AR_PREFIX_ACTION_SOURCE",
                    "issued",
                )
            ),
            allow_instruction_refresh=_bool_value(
                _value(
                    model_cfg,
                    "allow_instruction_refresh",
                    "VAP_ALLOW_INSTRUCTION_REFRESH",
                    False,
                ),
                name="VAP_ALLOW_INSTRUCTION_REFRESH",
            ),
            image_rgb_gain=image_rgb_gain,
            evaluation_profile=evaluation_profile.name,
            gripper_hysteresis=_gripper_hysteresis_value(model_cfg),
            gripper_close_threshold=float(
                _value(
                    model_cfg,
                    "gripper_close_threshold",
                    "VAP_GRIPPER_CLOSE_THRESHOLD",
                    0.25,
                )
            ),
            gripper_open_threshold=float(
                _value(
                    model_cfg,
                    "gripper_open_threshold",
                    "VAP_GRIPPER_OPEN_THRESHOLD",
                    0.75,
                )
            ),
            gripper_open_confirm_steps=int(
                _value(
                    model_cfg,
                    "gripper_open_confirm_steps",
                    "VAP_GRIPPER_OPEN_CONFIRM_STEPS",
                    4,
                )
            ),
            se3_smoothing=_bool_value(
                _value(model_cfg, "se3_smoothing", "VAP_SE3_SMOOTHING", False),
                name="VAP_SE3_SMOOTHING",
            ),
            se3_smoothing_mode=str(
                _value(
                    model_cfg,
                    "se3_smoothing_mode",
                    "VAP_SE3_SMOOTHING_MODE",
                    "fir",
                )
            ),
            se3_smoothing_neighbor_weight=float(
                _value(
                    model_cfg,
                    "se3_smoothing_neighbor_weight",
                    "VAP_SE3_SMOOTHING_NEIGHBOR_WEIGHT",
                    0.25,
                )
            ),
            se3_smoothing_acceleration_weight=float(
                _value(
                    model_cfg,
                    "se3_smoothing_acceleration_weight",
                    "VAP_SE3_SMOOTHING_ACCELERATION_WEIGHT",
                    0.25,
                )
            ),
            se3_smoothing_jerk_weight=float(
                _value(
                    model_cfg,
                    "se3_smoothing_jerk_weight",
                    "VAP_SE3_SMOOTHING_JERK_WEIGHT",
                    1.0,
                )
            ),
            se3_smoothing_position_max_deviation_m=float(
                _value(
                    model_cfg,
                    "se3_smoothing_position_max_deviation_m",
                    "VAP_SE3_SMOOTHING_POSITION_MAX_DEVIATION_M",
                    0.00075,
                )
            ),
            se3_smoothing_rotation_max_deviation_deg=float(
                _value(
                    model_cfg,
                    "se3_smoothing_rotation_max_deviation_deg",
                    "VAP_SE3_SMOOTHING_ROTATION_MAX_DEVIATION_DEG",
                    0.15,
                )
            ),
        )
        stats_path = Path(runtime.normalizer_stats_file)
        evaluation_profile.validate_stats_file(stats_path)
        stats_sha256 = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        print(
            "[VAP_EVALUATION_PROFILE] "
            f"name={evaluation_profile.name} "
            f"protocol={evaluation_profile.protocol_version} "
            f"actions={evaluation_profile.action_steps} "
            f"video_frames={evaluation_profile.video_frames} "
            f"image={evaluation_profile.width}x{evaluation_profile.height}",
            flush=True,
        )
        print(
            f"[VAP_POLICY_MODE] inference_mode={self.policy.inference_mode}",
            flush=True,
        )
        print(
            "[VAP_AR_PREFIX_ACTION] "
            f"source={self.policy.ar_prefix_action_source} "
            + (
                "domain=normalized_lfu_master controller_postprocess_excluded=true"
                if self.policy.ar_prefix_action_source == "raw_model"
                else "domain=control_or_measured"
            ),
            flush=True,
        )
        print(
            "[VAP_ACTION_DENOISE] "
            f"inference_steps={runtime.cfg.model.action_flow['inference_steps']} "
            f"inference_shift={runtime.cfg.model.action_flow['shift']}",
            flush=True,
        )
        print(
            "[VAP_TEXT_FEATURE_CACHE] "
            f"path={runtime.cfg.model.text_feature_cache_path or 'disabled'} "
            "online_fallback="
            f"{runtime.cfg.model.text_feature_cache_online_fallback}",
            flush=True,
        )
        print(
            "[VAP_STATE_CONDITIONING] "
            f"enabled={runtime.state_conditioning_enabled} "
            f"state_dim={runtime.state_dim} "
            f"normalizer_key={runtime.normalizer_key} "
            "anchor=window_S0",
            flush=True,
        )
        print(
            "[VAP_FIRST_FRAME_MEMORY] "
            f"enabled={runtime.first_frame_memory_enabled} "
            "anchor=resolved_episode_local_index_0 "
            "lifetime=episode "
            "encoded_once=true "
            f"view_keys={runtime.first_frame_memory_view_keys or 'all'} "
            f"position_mode={runtime.first_frame_memory_position_mode} "
            f"history_offsets={tuple(getattr(runtime, 'history_offsets', ()))} "
            f"grid={tuple(getattr(runtime.cfg.model, 'first_frame_memory_grid', ()))} "
            "reader_layers="
            f"{tuple(getattr(runtime.cfg.model, 'first_frame_memory_reader_layers', ()))}",
            flush=True,
        )
        print(
            f"[VAP_IMAGE_CALIBRATION] rgb_gain={image_rgb_gain or 'disabled'}",
            flush=True,
        )
        gripper_config = self.policy.gripper_hysteresis
        print(
            "[VAP_GRIPPER_POSTPROCESS] "
            f"enabled={gripper_config is not None} "
            f"close_threshold={getattr(gripper_config, 'close_threshold', None)} "
            f"open_threshold={getattr(gripper_config, 'open_threshold', None)} "
            f"open_confirm_steps={getattr(gripper_config, 'open_confirm_steps', None)}",
            flush=True,
        )
        smoothing_config = self.policy.se3_smoothing
        print(
            "[VAP_SE3_SMOOTHING] "
            f"enabled={smoothing_config is not None} "
            f"mode={getattr(smoothing_config, 'mode', None)} "
            f"neighbor_weight={getattr(smoothing_config, 'neighbor_weight', None)} "
            f"center_weight={getattr(smoothing_config, 'center_weight', None)} "
            "acceleration_weight="
            f"{getattr(smoothing_config, 'acceleration_weight', None)} "
            f"jerk_weight={getattr(smoothing_config, 'jerk_weight', None)} "
            "position_max_deviation_m="
            f"{getattr(smoothing_config, 'position_max_deviation_m', None)} "
            "rotation_max_deviation_deg="
            f"{getattr(smoothing_config, 'rotation_max_deviation_deg', None)} "
            "horizon_preserved=True endpoint_preserved=True gripper=unchanged",
            flush=True,
        )
        print(
            "[VAP_ACTION_NORMALIZER] "
            f"key={runtime.normalizer_key} "
            f"stats={runtime.normalizer_stats_file} "
            f"min_key={runtime.normalizer_min_key} "
            f"delta_key={runtime.normalizer_delta_key} "
            f"clip={runtime.normalizer_clip} "
            f"sha256={stats_sha256} "
            "source="
            + (
                "config_json_hash_verified"
                if evaluation_profile.allow_config_stats_without_checkpoint_state
                else "checkpoint_state_selected_key_exact_match"
            ),
            flush=True,
        )
        self._observation: Mapping[str, Any] | None = None

    def update_obs(self, observation: Mapping[str, Any]) -> None:
        if not isinstance(observation, Mapping):
            raise TypeError("VAP observation must be a mapping")
        self._observation = observation

    def get_action(self) -> list[dict[str, Any]]:
        if self._observation is None:
            raise RuntimeError("get_action called before update_obs")
        if os.environ.get("VAP_POLICY_TIMING", "0") == "1":
            import json
            import time
            runtime_model = self.policy.runtime.model
            core = getattr(runtime_model, "core", None)
            accelerator = getattr(core, "_v2_action_accelerator", None)
            before = 0 if accelerator is None else accelerator.stats["captures"]
            started = time.perf_counter()
            result = self.policy.infer(self._observation)
            # Returned physical NumPy actions already synchronize GPU output.
            elapsed = time.perf_counter() - started
            after = 0 if accelerator is None else accelerator.stats["captures"]
            print("[VAP_POLICY_TIMING] " + json.dumps(dict(
                seconds=elapsed, action_count=len(result),
                mode="off" if accelerator is None else accelerator.mode,
                captures=after-before,
                acceleration=None if accelerator is None else accelerator.metadata(),
            )), flush=True)
            return result
        return self.policy.infer(self._observation)

    def reset(self) -> None:
        self.policy.reset()
        self._observation = None

    def update_obs_batch(self, observation: Any) -> None:
        raise NotImplementedError(
            "VAP RoboDojo adapter is stateful and not batch-enabled"
        )

    def get_action_batch(self, env_idx_list: Any = None) -> Any:
        del env_idx_list
        raise NotImplementedError(
            "VAP RoboDojo adapter is stateful and not batch-enabled"
        )


__all__ = ["Model"]
