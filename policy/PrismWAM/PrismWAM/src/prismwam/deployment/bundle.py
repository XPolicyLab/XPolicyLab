"""Exported models: one folder with the config, the weights and the files they need.

    <model>/
      config.yaml            model, observation geometry and inference settings
      policy.pt              Stage-2 weights
      rgb_vae.*, depth_vae.*
      action_stats.*         joint-action statistics (or LIBERO norm_stats.json)
      text_cache*            optional instruction features

Paths in config.yaml are relative to the folder, so it can be moved or uploaded as is.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from .policy import (
    ActionStatistics,
    apply_evaluation_config,
    prepare_model_config,
    read_config,
    read_evaluation_settings,
)

CONFIG_NAME = "config.yaml"
WEIGHTS_NAME = "policy.pt"
ASSET_KEYS = {
    "rgb_vae_checkpoint_path": "rgb_vae",
    "depth_vae_checkpoint_path": "depth_vae",
    "text_encoder_checkpoint_path": "text_encoder",
    "tokenizer_path": "tokenizer",
}
GEOMETRY_KEYS = (
    "video_size",
    "rgb_video_size",
    "num_video_frames",
    "frame_stride",
    "rgb_history_num_frames",
    "rgb_history_frame_stride",
    "cameras",
    "layout",
    "depth_min",
    "depth_max",
    "depth_max_meters",
    "action_horizon",
    "action_dim",
    "action_start_offset",
    "proprio_start_offset",
    "context_len",
    "text_encoder_id",
    "prompt_template",
    "task_names",
    "rgb_target_mode",
    "sparse_recent_action_offset",
    "sparse_frame_encoding",
    "action_source",
    "processor",
)


def sanitize_checkpoint_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove nonstructural local paths while keeping parameter keys untouched."""
    opaque = {"mot", "state_dict", "proprio_encoder", "optimizer"}

    def visit(value, parent=""):
        if parent in opaque or isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Mapping):
            return {
                key: visit(item, str(key))
                for key, item in value.items()
                if key not in {"initialization_provenance", "training_provenance"}
            }
        if isinstance(value, (list, tuple)):
            return type(value)(visit(item, parent) for item in value)
        if isinstance(value, str) and (value.startswith(("/", "~/")) or ":\\" in value):
            return "external_asset"
        return value

    return visit(payload)


def export_bundle(
    config: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    evaluation: str | Path | Mapping[str, Any] | None = None,
    statistics: str | Path | None = None,
    text_cache: str | Path | None = None,
    task_name: str | None = None,
    hardlink: bool = False,
    sanitize_checkpoint: bool = False,
    overrides: Mapping[str, str] | None = None,
) -> Path:
    """Write a model folder for ``checkpoint`` trained with ``config`` into ``output``.

    Inference settings come from the config's ``deployment`` section unless
    ``evaluation`` is given. Files are copied, or hard-linked with ``hardlink``.
    """
    directory = Path(output).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"Refusing to write into a nonempty directory: {directory}"
        )
    source = read_config(config)
    train = source.get("data", {}).get("train", {})
    if task_name is not None:
        sources = train.get("sources")
        if not sources:
            raise ValueError("task_name is only valid for a mixed real-world recipe.")
        matches = [s for s in sources if task_name in s.get("task_names", [])]
        if len(matches) != 1:
            raise ValueError(
                "task_name must select exactly one configured data source."
            )
        train = dict(matches[0])
        if statistics is None:
            statistics = train.get("action_stats_cache_path")
        if statistics is None:
            raise ValueError("Exporting a robot task requires its action statistics.")
        native_dimension = train.get("action_dim")
        if native_dimension not in (7, 14):
            raise ValueError("Robot task export requires native action_dim 7 or 14.")
        ActionStatistics.load(statistics, native_dimension, expected_task=task_name)
    settings = read_evaluation_settings(source, evaluation)
    source = apply_evaluation_config(source, settings)
    model = prepare_model_config(source)
    for key, value in (overrides or {}).items():
        key = key.removeprefix("model.")
        if key not in ASSET_KEYS:
            raise ValueError(f"Unsupported asset override: {key}")
        model[key] = value

    directory.mkdir(parents=True, exist_ok=True)

    def place(src: str | Path, name: str) -> str:
        src = Path(src).expanduser().resolve(strict=True)
        destination = directory / name
        if src.is_dir():
            shutil.copytree(
                src, destination, copy_function=os.link if hardlink else shutil.copy2
            )
        elif hardlink:
            os.link(src, destination)
        else:
            shutil.copy2(src, destination)
        return name

    def suffixed(path: str | Path, role: str) -> str:
        path = Path(path)
        return role if path.is_dir() else role + path.suffix

    for key, role in ASSET_KEYS.items():
        value = model.get(key)
        if key in {"text_encoder_checkpoint_path", "tokenizer_path"} and not model.get(
            "load_text_encoder", False
        ):
            model.pop(key, None)
            continue
        if value:
            model[key] = place(value, suffixed(value, role))

    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    if sanitize_checkpoint:
        payload = sanitize_checkpoint_metadata(
            torch.load(
                checkpoint_path, map_location="cpu", weights_only=True, mmap=True
            )
        )
        torch.save(payload, directory / WEIGHTS_NAME)
        del payload
    else:
        place(checkpoint_path, WEIGHTS_NAME)

    source_deployment = source.get("deployment", {})
    deployment = {
        key: source_deployment[key]
        for key in (
            "action_horizon",
            "replan_steps",
            "num_inference_steps",
            "inference_shifts",
        )
        if key in source_deployment
    }
    if settings.get("benchmark") is not None:
        deployment["benchmark"] = settings["benchmark"]
    deployment["portable_bundle"] = True
    if task_name is not None:
        deployment["task_name"] = task_name
    if statistics is None:
        # Joint-action recipes z-score with the training statistics.
        statistics = source_deployment.get("statistics_path") or train.get(
            "action_stats_cache_path"
        )
    if statistics is not None:
        deployment["statistics_path"] = place(
            statistics, suffixed(statistics, "action_stats")
        )
    norm_stats = source_deployment.get("norm_stats_path") or train.get(
        "pretrained_norm_stats"
    )
    if norm_stats is not None:
        # LIBERO normalizes with its dataset processor's min/max statistics.
        deployment["norm_stats_path"] = place(
            norm_stats, suffixed(norm_stats, "norm_stats")
        )
    if text_cache is None:
        text_cache = source_deployment.get("text_cache_path")
    if text_cache is not None:
        deployment["text_cache_path"] = place(
            text_cache, suffixed(text_cache, "text_cache")
        )

    bundle_config = {
        "model": model,
        "data": {"train": {key: train[key] for key in GEOMETRY_KEYS if key in train}},
        "deployment": deployment,
    }
    OmegaConf.save(OmegaConf.create(bundle_config), directory / CONFIG_NAME)
    return directory
