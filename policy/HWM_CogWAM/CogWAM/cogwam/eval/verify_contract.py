#!/usr/bin/env python3
"""Fail-fast validation for a CogWAM RoboDojo checkpoint and its run artifacts.

Loading a 2B-parameter policy on eight GPUs takes minutes; a geometry or
contract mismatch discovered afterwards costs an entire evaluation. This
preflight reads only tensor *metadata* (``map_location="meta"`` / safetensors
slices) plus the saved YAML, so it answers in seconds.

Two sources are supported:

``--artifact-dir``
    A released artifact from ``tools/convert_checkpoint.py``. The manifest's
    sha256 over ``model.safetensors`` and ``dataset_statistics.json`` is
    verified first, then the same geometry checks run against
    ``inference_config.yaml``.

``--checkpoint``
    A raw training checkpoint, validated against the ``config.yaml`` and
    ``dataset_statistics.json`` saved beside it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from cogwam.data.composite import (
    TRI_VIEW_COMPOSITE_LAYOUT,
    TRI_VIEW_COMPOSITE_SIZE,
    TRI_VIEW_COMPOSITE_SOURCE_VIEW_KEYS,
    TRI_VIEW_COMPOSITE_VIEW_KEY,
)

# The released CogWAM physical model. These are not defaults to fall back on:
# every one of them is fixed by the published checkpoint.
RELEASED_ARCHITECTURE = {
    "world_hidden_size": 512,
    "action_hidden_size": 1024,
    "world_ffn_dim": 2048,
    "action_ffn_dim": 4096,
    "num_layers": 30,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
}

RELEASED_ACTION_DIM = 14
RELEASED_STATE_DIM = 14
RELEASED_OBS_IMAGE_SIZE = list(TRI_VIEW_COMPOSITE_SIZE)
RELEASED_DATA_MIX = "robodojo_v21_language"
# The event-memory recipe labels the decision at t-10 and replans every 10
# steps. The eval client asserts the same two numbers against the server
# handshake, so a drift here would silently change the reproduction.
RELEASED_SEMANTIC_OFFSET = -10
RELEASED_REPLAN_INTERVAL = 10
# <UPDATE> + "Memory Add:" + "Current Subtask:" bodies fit in 66 tokens at the
# audited maximum; a shorter budget truncates the semantic state mid-sentence.
MIN_EVENT_MAX_NEW_TOKENS = 66


def _run_dir(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        if (parent / "config.yaml").is_file():
            return parent
    raise FileNotFoundError(f"Could not find config.yaml above checkpoint: {checkpoint}")


def _expect(actual, expected, field: str) -> None:
    if actual != expected:
        raise ValueError(f"RoboDojo checkpoint {field}={actual!r}; expected {expected!r}.")


def _check_modality_stats(stats: dict, modality: str) -> None:
    modality_stats = stats.get(modality)
    if not isinstance(modality_stats, dict):
        raise ValueError(f"dataset_statistics new_embodiment.{modality} is missing or invalid.")
    for name in ("mean", "std"):
        values = np.asarray(modality_stats.get(name), dtype=np.float64)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError(
                f"dataset_statistics new_embodiment.{modality}.{name} must be finite 14-D, got shape={values.shape}."
            )


def _checkpoint_tensor_shapes(checkpoint: Path) -> dict[str, tuple[int, ...]]:
    """Read checkpoint tensor metadata without materializing model weights."""

    if checkpoint.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required to preflight this checkpoint") from exc
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            return {key: tuple(int(value) for value in handle.get_slice(key).get_shape()) for key in handle.keys()}

    try:
        state = torch.load(checkpoint, map_location="meta", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        state = torch.load(checkpoint, map_location="meta", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must contain a raw state_dict mapping, got {type(state).__name__}")
    shapes = {
        str(key): tuple(int(value) for value in tensor.shape)
        for key, tensor in state.items()
        if torch.is_tensor(tensor)
    }
    if not shapes:
        raise ValueError(f"Checkpoint contains no tensors: {checkpoint}")
    return shapes


def _required_shape(shapes: dict[str, tuple[int, ...]], key: str) -> tuple[int, ...]:
    if key not in shapes:
        raise ValueError(f"Checkpoint is missing required causal-MoT tensor {key!r}")
    return shapes[key]


def infer_weight_contract(shapes: dict[str, tuple[int, ...]]) -> dict:
    """Infer the physical architecture encoded by the actual state_dict."""

    layer_pattern = re.compile(r"^action_model\.layers\.(\d+)\.")
    layer_indices = sorted({int(match.group(1)) for key in shapes if (match := layer_pattern.match(key)) is not None})
    if not layer_indices or layer_indices != list(range(layer_indices[-1] + 1)):
        raise ValueError(
            "Checkpoint causal-MoT layer indices must be contiguous from zero, "
            f"got {layer_indices[:8]}{'...' if len(layer_indices) > 8 else ''}"
        )

    layerwise_world_key = "action_model.world_context.0.0.weight"
    legacy_world_key = "action_model.world_context.0.weight"
    has_layerwise_context = layerwise_world_key in shapes
    has_legacy_context = legacy_world_key in shapes
    if has_layerwise_context == has_legacy_context:
        raise ValueError("Checkpoint must contain exactly one causal-MoT context layout: per-layer or legacy shared")
    if has_layerwise_context:
        raise ValueError(
            "This repository reproduces the released 30-layer shared-context CogWAM recipe; "
            "the checkpoint carries a per-layer planner-coupled context stack."
        )
    _required_shape(shapes, "action_model.action_context.0.weight")

    world_input = _required_shape(shapes, "action_model.world_input.weight")
    action_input = _required_shape(shapes, "action_model.action_input.weight")
    world_ffn = _required_shape(shapes, "action_model.layers.0.world.ffn.0.weight")
    action_ffn = _required_shape(shapes, "action_model.layers.0.action.ffn.0.weight")
    world_query = _required_shape(shapes, "action_model.layers.0.world.self_attn.q.weight")
    # The released policy is state-conditioned, so this projection must exist.
    state_projection = _required_shape(shapes, "action_model.state_to_planner.weight")
    action_queries = _required_shape(shapes, "action_plan_queries.embedding")
    world_queries = _required_shape(shapes, "world_plan_queries.embedding")
    matrix_shapes = [world_input, action_input, world_ffn, action_ffn, world_query, state_projection]
    if not all(len(shape) == 2 for shape in matrix_shapes):
        raise ValueError("Checkpoint causal-MoT projection tensors must be matrices")
    if len(action_queries) != 3 or len(world_queries) != 3:
        raise ValueError("Checkpoint planner query banks must have shape [1,N,H]")

    obsolete_action_memory = sorted(key for key in shapes if key.startswith("action_model.action_dino_"))
    if obsolete_action_memory:
        raise ValueError(
            "Checkpoint contains the obsolete action-only DINO memory branch. "
            "The supported contract places full-resolution current DINO directly in the clean world prefix; "
            f"obsolete keys={obsolete_action_memory}"
        )
    return {
        "version": "legacy_shared_context_state_world_v1",
        "layerwise_planner_coupling": False,
        "multires_world_input": False,
        "world_hidden_size": world_input[0],
        "world_dim": world_input[1],
        "action_hidden_size": action_input[0],
        "action_dim": action_input[1],
        "state_dim": state_projection[1],
        "num_action_queries": action_queries[1],
        "num_world_queries": world_queries[1],
        "world_ffn_dim": world_ffn[0],
        "action_ffn_dim": action_ffn[0],
        "attention_inner_dim": world_query[0],
        "num_layers": len(layer_indices),
    }


def _validate_construction_config(config: dict, weight_contract: dict) -> int:
    """Ensure the saved config will reconstruct the state_dict that was inspected.

    Returns the validated action horizon.
    """

    framework = config.get("framework") or {}
    _expect(str(framework.get("name", "")), "CogWAM", "framework.name")
    _expect(bool(framework.get("enable_world_action_mot", False)), True, "framework.enable_world_action_mot")
    mot = framework.get("world_action_mot") or {}
    _expect(str(mot.get("architecture", "")).lower(), "causal_dino_mot", "framework.world_action_mot.architecture")
    if str(mot.get("interaction_mode", "")).lower() not in {"base", "joint"}:
        raise ValueError("framework.world_action_mot.interaction_mode must be 'base' or 'joint'")
    _expect(
        mot.get("world_attention_mask_mode"),
        "first_frame_causal",
        "framework.world_action_mot.world_attention_mask_mode",
    )
    _expect(
        bool(mot.get("layerwise_planner_coupling", False)),
        False,
        "framework.world_action_mot.layerwise_planner_coupling",
    )

    for field, expected in RELEASED_ARCHITECTURE.items():
        _expect(int(mot.get(field, -1)), expected, f"framework.world_action_mot.{field}")
    for field in ("world_hidden_size", "action_hidden_size", "world_ffn_dim", "action_ffn_dim", "num_layers"):
        _expect(int(mot[field]), int(weight_contract[field]), f"config {field} vs checkpoint")
    _expect(
        int(mot["num_attention_heads"]) * int(mot["attention_head_dim"]),
        int(weight_contract["attention_inner_dim"]),
        "config causal-MoT attention inner dimension vs checkpoint",
    )
    for field, expected in (
        ("time_frequency_dim", 256),
        ("world_grid_height", 12),
        ("world_grid_width", 10),
        ("world_num_train_timesteps", 1000),
        ("action_num_train_timesteps", 1000),
    ):
        _expect(int(mot.get(field, -1)), expected, f"framework.world_action_mot.{field}")
    for field, expected in (
        ("norm_eps", 1.0e-6),
        ("world_train_shift", 5.0),
        ("world_infer_shift", 5.0),
        ("action_train_shift", 5.0),
        ("action_infer_shift", 5.0),
    ):
        _expect(float(mot.get(field, -1.0)), expected, f"framework.world_action_mot.{field}")

    action_prediction_type = str(mot.get("action_prediction_type", "")).lower()
    if action_prediction_type not in {"velocity", "jit_x"}:
        raise ValueError(
            "framework.world_action_mot.action_prediction_type must be 'velocity' or 'jit_x', "
            f"got {action_prediction_type!r}"
        )
    action_velocity_target = str(mot.get("action_velocity_target", "")).lower()
    if action_velocity_target not in {"clean_minus_noise", "noise_minus_clean"}:
        raise ValueError(
            "framework.world_action_mot.action_velocity_target must be 'clean_minus_noise' or 'noise_minus_clean', "
            f"got {action_velocity_target!r}"
        )
    if int(mot.get("repeated_diffusion_steps", 1)) <= 0:
        raise ValueError("framework.world_action_mot.repeated_diffusion_steps must be positive")
    saved_steps = int(mot.get("num_inference_timesteps", -1))
    if saved_steps not in {10, 20}:
        # RoboDojo evaluation explicitly overrides to 10 flow steps; the saved
        # value only has to be one of the two audited training settings.
        raise ValueError(f"Released CogWAM checkpoints save 10 or 20 inference steps, got {saved_steps}")

    action = framework.get("action_model") or {}
    _expect(int(action.get("action_dim", -1)), RELEASED_ACTION_DIM, "framework.action_model.action_dim")
    _expect(int(action.get("action_dim", -1)), int(weight_contract["action_dim"]), "config action_dim vs checkpoint")
    _expect(int(action.get("state_dim", 0) or 0), RELEASED_STATE_DIM, "framework.action_model.state_dim")
    _expect(int(action.get("state_dim", 0) or 0), int(weight_contract["state_dim"]), "config state_dim vs checkpoint")

    planner = framework.get("planner") or {}
    configured_action_queries = int(planner.get("num_action_queries", -1))
    _expect(
        configured_action_queries,
        int(weight_contract["num_action_queries"]),
        "planner.num_action_queries vs checkpoint",
    )
    action_horizon = int(action.get("action_horizon", -1))
    _expect(action_horizon, configured_action_queries, "action_horizon vs action planner queries")
    _expect(
        int(planner.get("num_world_queries", -1)),
        int(weight_contract["num_world_queries"]),
        "planner.num_world_queries vs checkpoint",
    )

    dino = framework.get("dino") or {}
    _expect(int(dino.get("embed_dim", -1)), int(weight_contract["world_dim"]), "framework.dino.embed_dim vs checkpoint")
    image_size = list(dino.get("image_size", []))
    if len(image_size) != 2:
        raise ValueError("framework.dino.image_size must be [H,W]")
    patch_size = int(dino.get("patch_size", 16))
    if patch_size <= 0 or int(image_size[0]) % patch_size or int(image_size[1]) % patch_size:
        raise ValueError("DINO image size must be divisible by patch_size")
    rows = int(image_size[0]) // patch_size
    columns = int(image_size[1]) // patch_size
    pool = int(dino.get("dino_pool", 1))
    if pool <= 0 or rows % pool or columns % pool:
        raise ValueError("framework.dino.dino_pool must divide the DINO grid")
    _expect(
        (rows // pool, columns // pool),
        (int(mot["world_grid_height"]), int(mot["world_grid_width"])),
        "pooled DINO grid vs physical world grid",
    )
    if dino.get("current_dino_pool", None) is not None:
        raise ValueError(
            "framework.dino.current_dino_pool is set; the released recipe feeds the clean current prefix at the "
            "same resolution as the future target (multi-resolution checkpoints are not reproduced here)."
        )
    return action_horizon


def _validate_event_memory(config: dict) -> dict:
    framework = config.get("framework") or {}
    mot = framework.get("world_action_mot") or {}
    planner = framework.get("planner") or {}
    text_cfg = planner.get("text_supervision") or {}
    data = (config.get("datasets") or {}).get("vla_data") or {}
    annotations = data.get("text_annotations") or {}

    text_planning_enabled = bool(text_cfg.get("enabled", False))
    text_loss_weight = float(mot.get("text_loss_weight", 0.0))
    if text_planning_enabled != (text_loss_weight > 0.0):
        raise ValueError(
            "CogWAM text contract requires planner.text_supervision.enabled exactly when "
            "world_action_mot.text_loss_weight > 0"
        )
    if not text_planning_enabled:
        raise ValueError("The released CogWAM recipe is text-supervised; planner.text_supervision.enabled must be true")
    if not bool(annotations.get("enabled", False)):
        raise ValueError("CogWAM text checkpoint requires datasets.vla_data.text_annotations.enabled=true")

    event_cfg = annotations.get("event_memory") or {}
    event_enabled = bool(event_cfg.get("enabled", False))
    event_mode = str(text_cfg.get("mode", "")).lower() == "event_driven_memory_ntp"
    if event_enabled != event_mode:
        raise ValueError("Event-memory checkpoint requires matching planner mode and dataset event_memory.enabled")
    if not event_enabled:
        raise ValueError("The released CogWAM recipe uses event_driven_memory_ntp semantic supervision")
    if bool((text_cfg.get("history") or {}).get("enabled", False)):
        raise ValueError("Event-memory checkpoint must not use RGB history")
    if int(event_cfg.get("semantic_offset", 0)) != RELEASED_SEMANTIC_OFFSET or (
        int(event_cfg.get("replan_interval", 0)) != RELEASED_REPLAN_INTERVAL
    ):
        raise ValueError("CogWAM event-memory checkpoint requires t-10 labels and 10-step action replanning")
    if int(text_cfg.get("max_new_tokens", 0)) < MIN_EVENT_MAX_NEW_TOKENS:
        raise ValueError(f"Event-memory max_new_tokens is below the audited {MIN_EVENT_MAX_NEW_TOKENS}-token maximum")
    return {
        "semantic_offset": int(event_cfg["semantic_offset"]),
        "replan_interval": int(event_cfg["replan_interval"]),
        "keep_token": str(text_cfg.get("keep_token", "<KEEP>")),
        "update_token": str(text_cfg.get("update_token", "<UPDATE>")),
        "max_new_tokens": int(text_cfg["max_new_tokens"]),
        "rgb_history": False,
    }


def _validate_data_contract(config: dict, action_horizon: int) -> None:
    data = (config.get("datasets") or {}).get("vla_data") or {}
    _expect(int(data.get("action_horizon", -1)), action_horizon, "datasets.vla_data.action_horizon vs action_horizon")
    _expect(data.get("data_mix"), RELEASED_DATA_MIX, "datasets.vla_data.data_mix")
    _expect(data.get("image_layout"), TRI_VIEW_COMPOSITE_LAYOUT, "datasets.vla_data.image_layout")
    _expect(data.get("composite_view_key"), TRI_VIEW_COMPOSITE_VIEW_KEY, "datasets.vla_data.composite_view_key")
    _expect(
        list(data.get("composite_source_view_keys") or []),
        TRI_VIEW_COMPOSITE_SOURCE_VIEW_KEYS,
        "datasets.vla_data.composite_source_view_keys",
    )
    _expect(list(data.get("obs_image_size") or []), RELEASED_OBS_IMAGE_SIZE, "datasets.vla_data.obs_image_size")
    _expect(data.get("action_type"), "abs_qpos", "datasets.vla_data.action_type")
    _expect(bool(data.get("include_state", False)), True, "datasets.vla_data.include_state")
    _expect(bool(data.get("online_dino", False)), True, "datasets.vla_data.online_dino")
    _expect(bool(data.get("decode_future_video", False)), True, "datasets.vla_data.decode_future_video")


def verify(*, checkpoint: Path | None = None, artifact_dir: Path | None = None) -> dict:
    if (checkpoint is None) == (artifact_dir is None):
        raise ValueError("Pass exactly one of --checkpoint or --artifact-dir")

    artifact_manifest = None
    if artifact_dir is not None:
        from cogwam.serve.policy_wrapper import validate_artifact_directory

        source = Path(artifact_dir).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Artifact directory does not exist: {source}")
        artifact_manifest = validate_artifact_directory(source)
        weights_path = source / "model.safetensors"
        config_path = source / "inference_config.yaml"
        stats_path = source / "dataset_statistics.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Artifact directory is missing inference_config.yaml: {source}")
    else:
        source = Path(checkpoint).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"RoboDojo checkpoint does not exist: {source}")
        if source.suffix not in {".pt", ".safetensors"}:
            raise ValueError(f"Unsupported checkpoint suffix: {source.suffix}")
        run_dir = _run_dir(source)
        weights_path = source
        config_path = run_dir / "config.yaml"
        stats_path = run_dir / "dataset_statistics.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"Missing dataset statistics beside checkpoint: {stats_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    all_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Checkpoint config is not a mapping: {config_path}")
    if not isinstance(all_stats, dict):
        raise TypeError(f"Checkpoint dataset statistics are not a mapping: {stats_path}")

    weight_contract = infer_weight_contract(_checkpoint_tensor_shapes(weights_path))
    action_horizon = _validate_construction_config(config, weight_contract)
    event_memory_contract = _validate_event_memory(config)
    _validate_data_contract(config, action_horizon)

    embodiment_stats = all_stats.get("new_embodiment")
    if not isinstance(embodiment_stats, dict):
        raise ValueError(
            f"dataset_statistics.json must contain top-level key 'new_embodiment'; available={sorted(all_stats)}."
        )
    _check_modality_stats(embodiment_stats, "state")
    _check_modality_stats(embodiment_stats, "action")

    summary = {
        "source": str(source),
        "contract_config": str(config_path),
        "framework": "CogWAM",
        "include_state": True,
        "data_mix": config["datasets"]["vla_data"]["data_mix"],
        "state_action_normalization": "state+action z-score via new_embodiment statistics",
        "image": "head+left_wrist+right_wrist -> tri-view 320x384 composite",
        "action_chunk": [action_horizon, RELEASED_ACTION_DIM],
        "weight_contract": weight_contract,
        "event_memory_contract": event_memory_contract,
        "artifact_manifest_sha256": (artifact_manifest or {}).get("manifest_sha256"),
        "dataset_statistics_sha256": (
            ((artifact_manifest or {}).get("files") or {}).get("dataset_statistics.json", {}).get("sha256")
        ),
    }
    print("[RoboDojo] checkpoint contract PASS")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-dir", type=Path, default=None)
    source.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    verify(checkpoint=args.checkpoint, artifact_dir=args.artifact_dir)


if __name__ == "__main__":
    main()
