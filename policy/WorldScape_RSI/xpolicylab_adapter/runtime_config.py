"""Profile validation and source selection, shared by Python and shell entrypoints."""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from numbers import Integral
import os
from pathlib import Path
import sys

if __package__:
    from .gripper_discretization import GripperDiscretizationConfig, resolve_gripper_config
else:
    from gripper_discretization import GripperDiscretizationConfig, resolve_gripper_config

POLICY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = POLICY_DIR / "worldscape-policy"


def validate_config(config):
    cfg = dict(config)
    if cfg.get("bench_name") not in (None, "RoboDojo"):
        raise ValueError("This policy supports only bench_name=RoboDojo")
    profile = cfg.setdefault("runtime_profile", "robodojo")
    if profile != "robodojo":
        raise ValueError("This policy supports only runtime_profile=robodojo")
    multi = cfg.get("multi_gpu")
    if multi is None:
        multi = {}
    if not isinstance(multi, Mapping) or set(multi) - {"enabled", "max_envs_per_gpu", "worker_timeout_s"}:
        raise ValueError("multi_gpu must contain only enabled, max_envs_per_gpu and worker_timeout_s")
    if not isinstance(multi.get("enabled", False), bool):
        raise ValueError("multi_gpu.enabled must be a boolean")
    capacity = multi.get("max_envs_per_gpu")
    if capacity is not None and (type(capacity) is not int or capacity < 1):
        raise ValueError("multi_gpu.max_envs_per_gpu must be a positive integer or null")
    timeout = multi.get("worker_timeout_s", 1200)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("multi_gpu.worker_timeout_s must be finite and positive")
    if multi.get("enabled", False) and (profile != "robodojo" or not cfg.get("eval_batch", False)):
        raise ValueError("multi_gpu requires runtime_profile=robodojo and eval_batch=true")
    if multi.get("enabled", False) and cfg.get("protocol", "ws") != "ws":
        raise ValueError("multi_gpu requires protocol=ws")
    for key in ("joint_action_filter", "table_color_normalization",
                "diagnostic_trace_path", "upstream_root", "action_steps",
                "kv_cache_fifo", "observation_history_stride", "diffusion_concat_profile", "diffusion_concat_height", "diffusion_concat_width",
                "diffusion_concat_view_order", "visual_prompt"):
        if key in cfg:
            raise ValueError(f"{key} is not supported; use the WorldScape_RSI profile configuration")
    for key in ("action_horizon", "replan_steps", "observation_interval", "memory_window_chunks",
                "ordinary_window_chunks", "observation_history_segments"):
        value = cfg.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{key} must be a positive integer or null")
            cfg[key] = int(value)
    if cfg.get("diffusion_view_layout", "mosaic_2x2") != "mosaic_2x2":
        raise ValueError("RoboDojo requires diffusion_view_layout=mosaic_2x2")
    for key in ("eval_batch", "constrain_action_padding", "use_observation_history",
                "use_causal_memory", "reset_independent_blocks", "short_window_for_non_memory_tasks",
                "allow_dummy_policy", "validate_checkpoint_artifacts", "log_inference"):
        if cfg.get(key) is not None and not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be a boolean")
    # Transport is RGB; BGR-trained checkpoints must opt in explicitly.
    if cfg.get("input_color_order") is None:
        cfg["input_color_order"] = "rgb"
    if cfg.get("input_color_order", "rgb") not in {"rgb", "bgr"}:
        raise ValueError("input_color_order must be rgb or bgr")
    interpolation = cfg.get("policy_image_interpolation", "area")
    if interpolation not in {"area", "bilinear_antialias"}:
        raise ValueError("policy_image_interpolation must be area or bilinear_antialias")
    GripperDiscretizationConfig.from_mapping(cfg.get("gripper_discretization"))
    if cfg.get("allow_dummy_policy", False):
        raise ValueError("RoboDojo requires real model weights")
    if cfg.get("memory_reset_chunks", 0):
        raise ValueError("RoboDojo uses wam_history_mode instead of memory_reset_chunks")
    if not cfg.get("short_window_for_non_memory_tasks", False) and any(
        cfg.get(key) is not None for key in ("memory_window_chunks", "ordinary_window_chunks")
    ):
        raise ValueError("Window overrides require short_window_for_non_memory_tasks=true")
    if profile == "robodojo":
        for name, default in {
            "constrain_action_padding": True, "short_window_for_non_memory_tasks": False,
            "use_observation_history": True, "use_causal_memory": True,
            "reset_independent_blocks": False, "validate_checkpoint_artifacts": True,
            "observation_history_segments": 8,
        }.items():
            if cfg.get(name) is None:
                cfg[name] = default
    return cfg


# These preprocessing choices are not implied by model architecture or normalization.
CHECKPOINT_DEPLOYMENT_KEYS = frozenset({
    "input_color_order", "policy_image_interpolation",
    "t5_prompt_template", "short_window_for_non_memory_tasks",
})


def resolve_checkpoint_deployment_config(config, checkpoint):
    """Checkpoint provenance overrides YAML defaults before preprocessing is built."""
    path = Path(checkpoint) / "provenance.json"
    if not path.is_file():
        return validate_config(config)
    provenance = json.loads(path.read_text())
    if not isinstance(provenance, Mapping):
        raise ValueError("Checkpoint provenance must be a mapping")
    options = provenance.get("deployment_config", {})
    if not isinstance(options, Mapping):
        raise ValueError("Checkpoint deployment_config must be a mapping")
    unknown = set(options) - CHECKPOINT_DEPLOYMENT_KEYS
    if unknown:
        raise ValueError(f"Unknown checkpoint deployment_config fields: {sorted(unknown)}")
    if "t5_prompt_template" in options and options["t5_prompt_template"] is not None:
        if not isinstance(options["t5_prompt_template"], str):
            raise ValueError("Checkpoint t5_prompt_template must be a string or null")
    return validate_config({**config, **options})


def resolve_rollout_config(config, checkpoint_model, provenance):
    """Prefer checkpoint rollout metadata to deployment fallback values."""
    resolved = dict(config)
    resolved["gripper_discretization"] = resolve_gripper_config(config, provenance)
    horizon = getattr(checkpoint_model.shape, "action_horizon", None)
    resolved["action_horizon"] = horizon if horizon is not None else config.get("action_horizon")
    keys = ["action_horizon"]
    if config.get("short_window_for_non_memory_tasks", False):
        capacity = checkpoint_model.wam.core.parameters.get("max_chunk_size")
        memory = capacity if capacity is not None else config.get("memory_window_chunks")
        resolved["memory_window_chunks"] = provenance.get("memory_window_chunks", memory)
        resolved["ordinary_window_chunks"] = provenance.get(
            "ordinary_window_chunks", config.get("ordinary_window_chunks"),
        )
        keys.extend(("memory_window_chunks", "ordinary_window_chunks"))
    for key in keys:
        value = resolved.get(key)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"Resolved {key} must be a positive integer")
        resolved[key] = int(value)
    return resolved


def prepare_source(config):
    raw = config.get("worldscape_root") or os.environ.get("WORLDSCAPE_POLICY_ROOT")
    root = Path(raw).expanduser().resolve() if raw else DEFAULT_SOURCE.resolve()
    if not (root / "src/worldscape_policy").is_dir() or not (root / "evals").is_dir():
        raise FileNotFoundError(f"Invalid WorldScape source: {root}")
    # Do not silently mix two editable installs in one interpreter.
    for name, folder in (("worldscape_policy", root / "src/worldscape_policy"), ("evals", root / "evals")):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(folder):
                raise RuntimeError(f"{name} is already imported from another checkout; use a fresh policy process")
    for path in (root, root / "src"):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    return root


# Legacy explicit environment overrides. Unset variables never replace YAML.
ENV_KEYS = {
    "WORLDSCAPE_POLICY_ROOT": "worldscape_root", "WORLDSCAPE_CHECKPOINT": "checkpoint_path",
    "WORLDSCAPE_DEVICE": "device", "WSP_MODE": "mode",
    "WSP_VALIDATE_CHECKPOINT_ARTIFACTS": "validate_checkpoint_artifacts",
    "WSP_LOG_INFERENCE": "log_inference",
}
BOOL_KEYS = {"validate_checkpoint_artifacts", "log_inference"}
STRING_KEYS = {"worldscape_root", "checkpoint_path", "device", "mode", "diffusion_view_layout",
               "vlm_cot_prompt", "t5_prompt_template"}


def launch_config(path):
    import yaml
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    overrides = {}
    for env, key in ENV_KEYS.items():
        if env in os.environ:
            raw = os.environ[env]
            if key in BOOL_KEYS and raw.lower() in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
                overrides[key] = raw.lower() in {"1", "true", "yes", "on"}
            else:
                overrides[key] = (raw or None) if key in STRING_KEYS else yaml.safe_load(raw)
            if key == "checkpoint_path" and overrides[key]:
                overrides[key] = str(Path(overrides[key]).expanduser().resolve())
    cfg.update(overrides)
    cfg = validate_config(cfg)
    root = prepare_source(cfg)
    return cfg, root, overrides


if __name__ == "__main__":
    cfg, root, overrides = launch_config(sys.argv[1])
    if len(sys.argv) > 2:
        gpu_ids = sys.argv[2].split(",")
        enabled = (cfg.get("multi_gpu") or {}).get("enabled", False)
        if enabled:
            if (len(gpu_ids) < 2 or any(not gpu.isdecimal() for gpu in gpu_ids)
                    or len({int(gpu) for gpu in gpu_ids}) != len(gpu_ids)):
                raise ValueError("multi_gpu requires at least two distinct numeric policy GPU IDs")
        elif len(gpu_ids) > 1:
            raise ValueError("Multiple policy GPUs require multi_gpu.enabled=true in the selected YAML")
    print(str(root))
    print(json.dumps({"profile": cfg["runtime_profile"], "source": str(root),
                      "input_color_order": cfg.get("input_color_order", "rgb"),
                      "action_horizon": cfg.get("action_horizon"),
                      "wam_history_mode": cfg.get("wam_history_mode", "periodic")}))
    for key, value in overrides.items():
        print(f"{key}={value!r}")
