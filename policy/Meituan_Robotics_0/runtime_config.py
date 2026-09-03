from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROFILE = "robodojo_qwen35_mmdit_abs_q99_640x480"


def checkpoint_run_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser()
    if path.is_file() and path.parent.name in {"checkpoints", "final_model"}:
        return path.parent.parent
    if path.is_dir() and path.name in {"checkpoints", "final_model"}:
        return path.parent
    return path.parent if path.is_file() else path


def resolve_checkpoint_file(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file():
        return path
    candidates: list[Path] = []
    for directory in (path / "final_model", path / "checkpoints", path):
        if directory.is_dir():
            candidates.extend(directory.glob("*.pt"))
            candidates.extend(directory.glob("*.safetensors"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one checkpoint under {path}, found {len(candidates)}. "
            "Set checkpoint_path to the exact .pt/.safetensors file."
        )
    return candidates[0]


def load_and_validate_profile(
    checkpoint_path: str | Path,
    model_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = model_cfg.get("checkpoint_profile")
    if profile != PROFILE:
        raise ValueError(f"checkpoint_profile must be {PROFILE!r}, got {profile!r}.")

    run_dir = checkpoint_run_dir(checkpoint_path)
    config_path = run_dir / "config.full.yaml"
    statistics_path = run_dir / "dataset_statistics.json"
    if not config_path.is_file() or not statistics_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint sidecars config.full.yaml and dataset_statistics.json are required in {run_dir}."
        )

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    with statistics_path.open("r", encoding="utf-8") as stream:
        statistics = json.load(stream)

    framework = config.get("framework", {})
    action_model = framework.get("action_model", {})
    vla_data = config.get("datasets", {}).get("vla_data", {})
    checks = {
        "framework.name": (framework.get("name"), "QwenMMDiT"),
        "framework.qwenvl.base_vlm": (
            "Qwen3.5-4B-Action" in str(framework.get("qwenvl", {}).get("base_vlm", "")),
            True,
        ),
        "framework.obs_image_size": (framework.get("obs_image_size"), None),
        "action_model.action_model_type": (action_model.get("action_model_type"), "MMDiT-Psi0"),
        "action_model.action_horizon": (action_model.get("action_horizon"), 50),
        "action_model.future_action_window_size": (
            action_model.get("future_action_window_size"),
            49,
        ),
        "action_model.action_dim": (action_model.get("action_dim"), 14),
        "action_model.state_dim": (action_model.get("state_dim"), 14),
        "action_model.num_inference_timesteps": (
            action_model.get("num_inference_timesteps"),
            10,
        ),
        "action_model.use_correlated_noise": (
            bool(action_model.get("use_correlated_noise", False)),
            False,
        ),
        "vla_data.data_mix": (vla_data.get("data_mix"), "robodojo_v21_all_h50_q99"),
        "vla_data.action_mode": (vla_data.get("action_mode"), "abs"),
        "vla_data.normalization_mode": (vla_data.get("normalization_mode"), "q99"),
        "vla_data.include_state": (vla_data.get("include_state"), True),
        "vla_data.min_pixels": (vla_data.get("min_pixels"), 307200),
        "vla_data.max_pixels": (vla_data.get("max_pixels"), 307200),
        "deploy.env_cfg_type": (model_cfg.get("env_cfg_type"), "arx_x5"),
        "deploy.action_type": (model_cfg.get("action_type"), "joint"),
        "deploy.execute_horizon": (int(model_cfg.get("execute_horizon", -1)), 16),
        "deploy.image_size": (list(model_cfg.get("image_size", [])), [640, 480]),
    }
    embodiment = statistics.get("new_embodiment", {})
    checks["statistics.only_new_embodiment"] = (list(statistics), ["new_embodiment"])
    for modality in ("state", "action"):
        modality_stats = embodiment.get(modality, {})
        checks[f"statistics.{modality}.q01_dim"] = (len(modality_stats.get("q01", [])), 14)
        checks[f"statistics.{modality}.q99_dim"] = (len(modality_stats.get("q99", [])), 14)
        checks[f"statistics.{modality}.mask"] = (
            modality_stats.get("mask", [True] * 14),
            [True] * 14,
        )

    mismatches = [
        f"{name}: got {actual!r}, expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            f"Checkpoint does not match profile {PROFILE!r}:\n  " + "\n  ".join(mismatches)
        )
    return config, statistics
