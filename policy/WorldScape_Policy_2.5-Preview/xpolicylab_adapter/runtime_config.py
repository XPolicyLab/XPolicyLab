"""Deployment configuration and runtime paths."""
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
    profile = cfg.setdefault("runtime_profile", "robodojo")
    if profile != "robodojo" or cfg.get("bench_name") not in (None, "RoboDojo", "robodojo"):
        raise ValueError("WorldScape Policy 2.5 Preview supports RoboDojo only")
    if cfg.get("history_memory") is not True or cfg.get("vlm_memory") is not True:
        raise ValueError("Preview requires history_memory=true and vlm_memory=true")
    if (cfg.get("checkpoint_format") != "lerobot_v21"
            or cfg.get("vlm_conditioning") != "joint_t5_vlm"
            or cfg.get("mode") != "auto"):
        raise ValueError("Preview requires the Auto LeRobot joint T5 + VLM checkpoint")
    if cfg.get("history_capture_mode", "dense") != "dense":
        raise ValueError("Preview requires history_capture_mode=dense")
    if cfg.get("t5_prompt_template") is not None:
        raise ValueError("T5 + VLM requires raw task instructions, without a T5 template")
    if cfg.get("vlm_input_color_order", "rgb") != "bgr":
        raise ValueError("This VLM checkpoint requires the explicit vlm_input_color_order=bgr opt-in")
    generation = cfg.get("generation_overrides")
    if generation is not None:
        if not isinstance(generation, Mapping) or set(generation) - {
            "num_inference_steps", "num_dit_steps", "dynamic_cache_schedule",
        }:
            raise ValueError("generation_overrides only supports sampling step counts and dynamic_cache_schedule")
        for key in ("num_inference_steps", "num_dit_steps"):
            if key in generation and (type(generation[key]) is not int or generation[key] < 1):
                raise ValueError(f"generation_overrides.{key} must be a positive integer")
        if "dynamic_cache_schedule" in generation and type(generation["dynamic_cache_schedule"]) is not bool:
            raise ValueError("generation_overrides.dynamic_cache_schedule must be a boolean")
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
    if multi.get("enabled", False):
        raise ValueError("Use one policy server per GPU for this inference adapter")
    for key in ("joint_action_filter", "gripper_discretization", "table_color_normalization",
                "diagnostic_trace_path", "upstream_root", "action_steps",
                "kv_cache_fifo", "observation_history_stride", "diffusion_concat_profile", "visual_prompt"):
        if key in cfg:
            raise ValueError(f"{key} is not supported; use the WorldScape_Policy_2.5-Preview profile configuration")
    for key in ("action_horizon", "replan_steps", "observation_interval",
                "observation_history_segments"):
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
                "use_causal_memory", "reset_independent_blocks",
                "allow_dummy_policy", "log_inference"):
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
    if cfg.get("allow_dummy_policy", False):
        raise ValueError("Preview does not support dummy policies")
    reset_chunks = cfg.get("memory_reset_chunks", 0)
    if reset_chunks is None:
        reset_chunks = 0
    if isinstance(reset_chunks, bool) or not isinstance(reset_chunks, Integral) or reset_chunks != 0:
        raise ValueError("Preview requires memory_reset_chunks=0")
    cfg["memory_reset_chunks"] = 0
    # Each prediction uses an independent chunk; visual history is supplied
    # explicitly by the episode archive, not by carrying KV across chunks.
    for name, expected in {
        "use_observation_history": False,
        "use_causal_memory": False,
        "reset_independent_blocks": True,
        "wam_history_mode": "periodic",
    }.items():
        if cfg.get(name) != expected:
            raise ValueError(f"Preview requires {name}={expected!r}")
    for name, default in {
        "constrain_action_padding": True,
        "observation_history_segments": 8,
    }.items():
        if cfg.get(name) is None:
            cfg[name] = default
    return cfg


def resolve_rollout_config(config, checkpoint_model):
    """Prefer checkpoint rollout metadata to deployment fallback values."""
    resolved = dict(config)
    horizon = getattr(checkpoint_model.shape, "action_horizon", None)
    resolved["action_horizon"] = horizon if horizon is not None else config.get("action_horizon")
    value = resolved["action_horizon"]
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("Resolved action_horizon must be a positive integer")
    resolved["action_horizon"] = int(value)
    return resolved


def configure_policy_inputs(policy, config):
    """Apply deployment geometry without putting adapter identities in model code."""
    codec = policy.visual_memory.codec
    codec.diffusion_concat_view_order = tuple(config.get("diffusion_concat_view_order", (0, 1, 2)))
    codec.diffusion_concat_size = (
        config.get("diffusion_concat_height", 384), config.get("diffusion_concat_width", 320),
    )
    policy.single_window_reference = config.get("single_window_reference", "latest")


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
    "ROBOTWIN_DIFFUSION_VIEW_LAYOUT": "diffusion_view_layout",
    "ROBOTWIN_T5_PROMPT_TEMPLATE": "t5_prompt_template",
    "WSP_LOG_INFERENCE": "log_inference", "WORLDSCAPE_ALLOW_DUMMY_POLICY": "allow_dummy_policy",
}
BOOL_KEYS = {"log_inference", "allow_dummy_policy"}
STRING_KEYS = {"worldscape_root", "checkpoint_path", "device", "mode", "diffusion_view_layout",
               "t5_prompt_template"}


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
        if len(gpu_ids) != 1:
            raise ValueError("Use one policy server per GPU for this inference adapter")
    print(str(root))
    print(json.dumps({"profile": cfg["runtime_profile"], "source": str(root),
                      "input_color_order": cfg.get("input_color_order", "rgb"),
                      "action_horizon": cfg.get("action_horizon"),
                      "wam_history_mode": cfg.get("wam_history_mode", "periodic")}))
    for key, value in overrides.items():
        print(f"{key}={value!r}")
