"""Profile validation and source selection, shared by Python and shell entrypoints."""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from numbers import Integral
import os
from pathlib import Path
import sys

POLICY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = POLICY_DIR / "worldscape-policy"


def validate_config(config):
    cfg = dict(config)
    profile = cfg.setdefault("runtime_profile", "robotwin2")
    if profile not in {"robotwin2", "robodojo"}:
        raise ValueError("runtime_profile must be robotwin2 or robodojo")
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
    for key in ("joint_action_filter", "gripper_discretization", "table_color_normalization",
                "diagnostic_trace_path", "upstream_root", "action_steps",
                "kv_cache_fifo", "observation_history_stride", "diffusion_concat_profile", "visual_prompt"):
        if key in cfg:
            raise ValueError(f"{key} is not supported; use the WorldScape_Policy_2 profile configuration")
    for key in ("action_horizon", "replan_steps", "observation_interval", "memory_window_chunks",
                "ordinary_window_chunks", "observation_history_segments"):
        value = cfg.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{key} must be a positive integer or null")
            cfg[key] = int(value)
    height = cfg.get("diffusion_concat_height", 384)
    width = cfg.get("diffusion_concat_width", 320)
    if type(height) is not int or height < 3:
        raise ValueError("diffusion_concat_height must be an integer >= 3")
    if type(width) is not int or width < 2 or width % 2:
        raise ValueError("diffusion_concat_width must be an even integer >= 2")
    for key in ("eval_batch", "constrain_action_padding", "use_observation_history",
                "use_causal_memory", "reset_independent_blocks", "short_window_for_non_memory_tasks",
                "allow_dummy_policy", "validate_checkpoint_artifacts", "log_inference"):
        if cfg.get(key) is not None and not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be a boolean")
    if cfg.get("input_color_order", "rgb") not in {"rgb", "bgr"}:
        raise ValueError("input_color_order must be rgb or bgr")
    order = cfg.get("diffusion_concat_view_order", [0, 1, 2])
    if (not isinstance(order, (list, tuple)) or len(order) != 3
            or any(type(index) is not int for index in order) or sorted(order) != [0, 1, 2]):
        raise ValueError("diffusion_concat_view_order must be a permutation of [0, 1, 2]")
    if cfg.get("single_window_reference", "latest") not in {"first", "latest"}:
        raise ValueError("single_window_reference must be first or latest")
    validate_qwen_config(cfg)
    qwen = cfg.get("interactive_qwen") or {}
    if profile == "robotwin2":
        forbidden = {
            "eval_batch": cfg.get("eval_batch", False),
            "input_color_order": cfg.get("input_color_order", "rgb") != "rgb",
            "wam_history_mode": cfg.get("wam_history_mode", "periodic") != "periodic",
            "short_window_for_non_memory_tasks": cfg.get("short_window_for_non_memory_tasks", False),
            "press_by_number_prompt": (cfg.get("press_by_number_prompt") or {}).get("enabled", False),
            "interactive_qwen": qwen.get("enabled", False),
            "use_observation_history": cfg.get("use_observation_history", True) is False,
            "use_causal_memory": cfg.get("use_causal_memory", True) is False,
            "reset_independent_blocks": cfg.get("reset_independent_blocks", False),
            "memory_window_chunks": cfg.get("memory_window_chunks") is not None,
            "ordinary_window_chunks": cfg.get("ordinary_window_chunks") is not None,
            "observation_history_segments": cfg.get("observation_history_segments", 8) != 8,
        }
        if any(forbidden.values()):
            raise ValueError(f"RoboDojo-only options enabled for robotwin2: {[k for k, v in forbidden.items() if v]}")
    else:
        if cfg.get("allow_dummy_policy", False):
            raise ValueError("allow_dummy_policy is only supported by robotwin2")
        if cfg.get("memory_reset_chunks", 0):
            raise ValueError("robodojo uses wam_history_mode instead of memory_reset_chunks")
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


def resolve_rollout_config(config, checkpoint_model, provenance):
    """Prefer checkpoint rollout metadata to deployment fallback values."""
    resolved = dict(config)
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


def configure_policy_inputs(policy, config):
    """Apply deployment geometry without putting adapter identities in model code."""
    codec = policy.visual_memory.codec
    codec.diffusion_concat_view_order = tuple(config.get("diffusion_concat_view_order", (0, 1, 2)))
    codec.diffusion_concat_size = (
        config.get("diffusion_concat_height", 384), config.get("diffusion_concat_width", 320),
    )
    policy.single_window_reference = config.get("single_window_reference", "latest")


def validate_qwen_config(cfg, *, resolved=False):
    for name in ("interactive_qwen", "press_by_number_prompt"):
        if cfg.get(name) is not None and not isinstance(cfg[name], Mapping):
            raise ValueError(f"{name} must be a mapping")
    qwen = cfg.get("interactive_qwen") or {}
    number = cfg.get("press_by_number_prompt") or {}
    for name, options in (("interactive_qwen", qwen), ("press_by_number_prompt", number)):
        if not isinstance(options.get("enabled", False), bool):
            raise ValueError(f"{name}.enabled must be a boolean")
    for name, options, allowed in (
        ("interactive_qwen", qwen, {"enabled"}),
        ("press_by_number_prompt", number, {"enabled", "batch_size"}),
    ):
        if set(options) - allowed:
            raise ValueError(f"Unknown {name} fields: {sorted(set(options) - allowed)}")
    batch_size = number.get("batch_size", 4)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("press_by_number_prompt.batch_size must be a positive integer")
    if cfg.get("runtime_profile") == "robodojo" and not resolved:
        return
    if number.get("enabled", False) and not qwen.get("enabled", False):
        raise ValueError("press_by_number_prompt requires interactive_qwen.enabled=true")
    if number.get("enabled", False) and qwen.get("enabled", False) and cfg.get("mode") not in {None, "interactive"}:
        raise ValueError("interactive_qwen requires mode=interactive (or null for an Interactive checkpoint)")
    if number.get("enabled", False) and cfg.get("t5_prompt_template") is not None:
        raise ValueError("Explicit count prompts require t5_prompt_template=null")


def resolve_number_prompt_config(config, provenance):
    """Resolve checkpoint prompt semantics before validating runtime dependencies."""
    number = dict(config.get("press_by_number_prompt") or {})
    checkpoint_number = provenance.get("press_by_number_prompt", {})
    if not isinstance(checkpoint_number, Mapping):
        raise ValueError("Checkpoint press_by_number_prompt must be a mapping")
    if "enabled" in checkpoint_number and not isinstance(checkpoint_number["enabled"], bool):
        raise ValueError("Checkpoint press_by_number_prompt.enabled must be a boolean")
    number["enabled"] = checkpoint_number.get("enabled", number.get("enabled", False))
    resolved = {**config, "press_by_number_prompt": number}
    validate_qwen_config(resolved, resolved=True)
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
    "WORLDSCAPE_DEVICE": "device", "WSP_MODE": "mode", "ROBOTWIN_ACTION_HORIZON": "action_horizon",
    "ROBOTWIN_REPLAN_STEPS": "replan_steps", "ROBOTWIN_OBSERVATION_INTERVAL": "observation_interval",
    "ROBOTWIN_MEMORY_RESET_CHUNKS": "memory_reset_chunks",
    "ROBOTWIN_VLM_HISTORY_NUM_FRAMES": "vlm_history_num_frames",
    "ROBOTWIN_DIFFUSION_VIEW_LAYOUT": "diffusion_view_layout",
    "ROBOTWIN_VLM_COT_PROMPT": "vlm_cot_prompt", "ROBOTWIN_T5_PROMPT_TEMPLATE": "t5_prompt_template",
    "WSP_VALIDATE_CHECKPOINT_ARTIFACTS": "validate_checkpoint_artifacts",
    "WSP_LOG_INFERENCE": "log_inference", "WORLDSCAPE_ALLOW_DUMMY_POLICY": "allow_dummy_policy",
}
BOOL_KEYS = {"validate_checkpoint_artifacts", "log_inference", "allow_dummy_policy"}
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
