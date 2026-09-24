from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import Tensor, nn

from worldscape_policy.checkpoint import (
    load_checkpoint_state_dict,
    validate_native_checkpoint_artifacts,
)
from worldscape_policy.checkpoint.adapter_rows import (
    select_pretrained_adapter_row,
)
from worldscape_policy.memory.event import EventMemoryFusion
from worldscape_policy.memory.visual.normalization import VisualInputRange
from worldscape_policy.model_config import GenerationConfig, ModelConfig
from worldscape_policy.policy import WorldScapePolicy
from worldscape_policy.registry import (
    Wan22PolicyBuildConfig,
    build_wan22_policy,
)
from worldscape_policy.types import InteractionMode
from worldscape_policy.wam.registry import DEFAULT_WAM_REGISTRY, WAMRegistry
from worldscape_policy.wam.wan22 import Wan22DistributedContext, Wan22KernelConfig

LOGGER = logging.getLogger(__name__)

_STAGE2_NEW_PREFIXES = (
    "condition_router.auto.vlm.",
    "condition_router.auto.projector.",
    "condition_router.auto.event_memory.",
    "condition_router.auto.output_norm.",
)


@dataclass(frozen=True)
class InitializationReport:
    """Compact, machine-readable account of every parameter's final source."""

    checkpoint: tuple[str, ...]
    raw_components: tuple[str, ...]
    random_init: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "checkpoint": list(self.checkpoint),
            "raw_components": list(self.raw_components),
            "random_init": list(self.random_init),
        }


class _DisabledPlanningEncoder(nn.Module):
    def encode_planning(self, **_: Any) -> None:
        raise RuntimeError("Auto conditioning is disabled by this checkpoint")


class _PassthroughEventMemory(nn.Module):
    def forward(self, current_tokens: Tensor, **_: Any) -> tuple[Tensor, dict]:
        return current_tokens, {}


def build_wan22_policy_from_checkpoint(
    checkpoint_dir: str | Path | None = None,
    *,
    visual_input_range: VisualInputRange,
    diffusion_view_layout: Literal[
        "mosaic_2x2", "robotwin_concat"
    ] | None = None,
    device: str | torch.device = "cpu",
    expected_mode: InteractionMode | str | None = None,
    expected_num_frames: int | None = None,
    expected_action_horizon: int | None = None,
    expected_semantic_gate_only: bool | None = None,
    expected_semantic_grad_clip_norm: float | None = None,
    training: bool = False,
    diffusion_model_pretrained_path: str | None = None,
    text_encoder_pretrained_path: str | None = None,
    image_encoder_pretrained_path: str | None = None,
    vae_pretrained_path: str | None = None,
    vlm_pretrained_path: str | None = None,
    vlm_cot_prompt: str | None = None,
    vlm_token_dim: int | None = None,
    vlm_context_dim: int | None = None,
    vlm_attn_implementation: str | None = None,
    tokenizer_path: str | None = None,
    pretrained_action_adapter_index: int | None = None,
    validate_checkpoint_artifacts: bool = True,
    model_config: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    initialization: Literal[
        "auto", "components", "checkpoint", "checkpoint_overlay"
    ] = "auto",
    wam_registry: WAMRegistry = DEFAULT_WAM_REGISTRY,
    distributed_context: Wan22DistributedContext | None = None,
) -> WorldScapePolicy:
    """Build from raw components, a WSP checkpoint, or a strict Stage-1 overlay.

    Raw component tensors are loaded first and checkpoint tensors second.  This
    makes the precedence unambiguous: checkpoint > raw component > random init.
    """

    directory = Path(checkpoint_dir) if checkpoint_dir else None
    if initialization not in {
        "auto",
        "components",
        "checkpoint",
        "checkpoint_overlay",
    }:
        raise ValueError(f"Unsupported initialization mode: {initialization!r}")
    if initialization == "auto":
        initialization = "checkpoint" if directory is not None else "components"
    if initialization != "components" and directory is None:
        raise ValueError(f"initialization={initialization!r} requires checkpoint_dir")
    is_native_bundle = bool(
        directory is not None
        and (directory / "checkpoint_manifest.json").is_file()
    )
    if is_native_bundle:
        if validate_checkpoint_artifacts:
            validate_native_checkpoint_artifacts(directory)
        normalization = json.loads((directory / "normalization.json").read_text())
        checkpoint_range = normalization["visual"]["input_range"]
        if (
            visual_input_range != checkpoint_range
            and initialization != "checkpoint_overlay"
        ):
            raise ValueError(
                "visual_input_range does not match native normalization artifact: "
                f"requested {visual_input_range!r}, checkpoint requires {checkpoint_range!r}"
            )
    supplied_model_config = model_config is not None
    if supplied_model_config:
        native_config = ModelConfig.from_dict(dict(model_config))
        if generation_config is None:
            raise ValueError("model_config requires generation_config")
        native_generation = GenerationConfig.from_dict(dict(generation_config))
        raw, action = _native_build_inputs(native_config, native_generation)
        mode = InteractionMode.parse(native_config.model.mode)
    elif is_native_bundle:
        native_config = _load_native_model_config(directory)
        native_generation = _load_generation_config(directory)
        raw, action = _native_build_inputs(native_config, native_generation)
        mode = InteractionMode.parse(native_config.model.mode)
    else:
        raise ValueError(
            "Native checkpoint loading requires a complete native bundle with "
            "checkpoint_manifest.json or an explicit model_config"
        )
    if vlm_token_dim is not None:
        if int(vlm_token_dim) <= 0:
            raise ValueError("vlm_token_dim must be positive")
        action["vlm_token_dim"] = int(vlm_token_dim)
    if vlm_context_dim is not None:
        if int(vlm_context_dim) <= 0:
            raise ValueError("vlm_context_dim must be positive")
        action["vlm_context_dim"] = int(vlm_context_dim)
    if diffusion_view_layout is not None:
        if diffusion_view_layout not in {"mosaic_2x2", "robotwin_concat"}:
            raise ValueError(
                "diffusion_view_layout must be 'mosaic_2x2' or "
                "'robotwin_concat'"
            )
        action["diffusion_view_layout"] = diffusion_view_layout
    resolved_diffusion_view_layout = str(
        action.get("diffusion_view_layout", "mosaic_2x2")
    )
    if resolved_diffusion_view_layout not in {
        "mosaic_2x2",
        "robotwin_concat",
    }:
        raise ValueError(
            "checkpoint diffusion_view_layout must be 'mosaic_2x2' or "
            "'robotwin_concat'"
        )
    registration = wam_registry.get("wan22")
    if "sampling" not in registration.metadata.capabilities:
        raise NotImplementedError("Registered Wan2.2 plugin does not support sampling")
    _reject_legacy_runtime_environment()
    if expected_mode is not None and not checkpoint_supports_mode(
        mode, expected_mode
    ):
        raise ValueError(
            f"Checkpoint mode is {mode.value!r}; requested "
            f"{InteractionMode.parse(expected_mode).value!r}"
        )
    _validate_recipe_expectations(
        action,
        expected_num_frames=expected_num_frames,
        expected_action_horizon=expected_action_horizon,
        expected_semantic_gate_only=expected_semantic_gate_only,
        expected_semantic_grad_clip_norm=expected_semantic_grad_clip_norm,
    )

    resolved_tokenizer_path = tokenizer_path or _resolve_tokenizer_path(
        directory, action
    )
    t5_config = _required_mapping(action, "text_encoder_cfg")
    t5_overrides: dict[str, Any] = {"tokenizer_path": resolved_tokenizer_path}
    if text_encoder_pretrained_path is not None:
        t5_overrides["text_encoder_pretrained_path"] = (
            text_encoder_pretrained_path
        )
    t5 = _instantiate(
        t5_config,
        target=(
            "worldscape_policy.conditioning.text.t5."
            "T5InstructionEncoder"
        ),
        **t5_overrides,
    )
    vae = _instantiate(
        _required_mapping(action, "vae_cfg"),
        **(
            {"vae_pretrained_path": vae_pretrained_path}
            if vae_pretrained_path is not None
            else {}
        ),
    )
    image_encoder = _instantiate(
        _required_mapping(action, "image_encoder_cfg"),
        **(
            {"image_encoder_pretrained_path": image_encoder_pretrained_path}
            if image_encoder_pretrained_path is not None
            else {}
        ),
    )
    core = _instantiate(
        _required_mapping(action, "diffusion_model_cfg"),
        **(
            {"diffusion_model_pretrained_path": diffusion_model_pretrained_path}
            if diffusion_model_pretrained_path is not None
            else {}
        ),
    )
    _validate_block_geometry(action, core)

    if mode is InteractionMode.AUTO:
        backbone = _required_mapping(raw, "backbone_cfg")
        vlm_overrides: dict[str, Any] = {}
        if vlm_pretrained_path is not None:
            vlm_overrides["vlm_base"] = vlm_pretrained_path
        if vlm_cot_prompt is not None:
            vlm_overrides["vlm_cot_prompt"] = vlm_cot_prompt
        if vlm_attn_implementation is not None:
            vlm_overrides["attn_implementation"] = vlm_attn_implementation
        for key in ("enable_planning_branch", "planning_num_tokens"):
            if key in action:
                vlm_overrides[key] = action[key]
        vlm = _instantiate(
            backbone,
            target=(
                "worldscape_policy.conditioning.vlm.qwen3vl."
                "QwenPlanningEncoder"
            ),
            **vlm_overrides,
        )
        projector = _build_projector(action)
        output_norm: nn.Module | None = (
            nn.LayerNorm(int(_required_value(action, "vlm_context_dim")))
            if bool(action.get("output_norm", True))
            else None
        )
        event_memory = _build_event_memory(action)
    else:
        vlm = _DisabledPlanningEncoder()
        projector = nn.Identity()
        output_norm = None
        event_memory = _PassthroughEventMemory()

    policy = build_wan22_policy(
        config=Wan22PolicyBuildConfig(
            num_frames=int(_required_value(action, "num_frames")),
            persistent_prompt=str(
                action.get("persistent_visual_prompt", "goal_or_demo")
            ),
            semantic_gate_only=bool(
                action.get("align_projector_gate_only", False)
            ),
            semantic_grad_clip_norm=float(
                action.get("align_proj_gate_grad_clip_norm", 0.5)
            ),
            max_history_steps=int(action.get("infer_memory_fifo_max_steps", 8)),
            view_index=int(action.get("view_index", 0)),
            tiled=bool(action.get("tiled", False)),
            tile_size=(
                int(action.get("tile_size_height", 34)),
                int(action.get("tile_size_width", 34)),
            ),
            tile_stride=(
                int(action.get("tile_stride_height", 18)),
                int(action.get("tile_stride_width", 16)),
            ),
            visual_input_range=visual_input_range,
            diffusion_view_layout=cast(
                Literal["mosaic_2x2", "robotwin_concat"],
                resolved_diffusion_view_layout,
            ),
        ),
        vlm=vlm,
        token_pooler=nn.Identity(),
        projector=projector,
        event_memory=event_memory,
        t5=t5,
        vae=vae,
        core=core,
        image_encoder=image_encoder,
        kernel_config=_kernel_config(raw, action),
        output_norm=output_norm,
        configured_mode=mode,
        wam_registry=wam_registry,
        distributed_context=distributed_context,
    )
    raw_loaded = _load_raw_components(
        policy,
        core=core,
        t5=t5,
        vae=vae,
        image_encoder=image_encoder,
        vlm=vlm,
        diffusion_model_pretrained_path=diffusion_model_pretrained_path,
        text_encoder_pretrained_path=text_encoder_pretrained_path,
        vae_pretrained_path=vae_pretrained_path,
        image_encoder_pretrained_path=image_encoder_pretrained_path,
        vlm_pretrained_path=vlm_pretrained_path,
        pretrained_action_adapter_index=pretrained_action_adapter_index,
    )
    checkpoint_loaded: set[str] = set()
    if initialization in {"checkpoint", "checkpoint_overlay"}:
        state_dict = _load_wsp_policy_state(
            directory,
            validate_artifacts=not is_native_bundle,
        )
        checkpoint_loaded = _load_policy_state(
            policy,
            state_dict,
            allow_stage2_missing=initialization == "checkpoint_overlay",
            pretrained_action_adapter_index=pretrained_action_adapter_index,
        )
    report = _initialization_report(policy, raw_loaded, checkpoint_loaded)
    policy.initialization_report = report.as_dict()
    LOGGER.info("WSP initialization sources: %s", json.dumps(report.as_dict()))
    policy.train(training).requires_grad_(training)
    return policy.to(device)


def _load_raw_components(
    policy: nn.Module,
    *,
    core: nn.Module,
    t5: nn.Module,
    vae: nn.Module,
    image_encoder: nn.Module,
    vlm: nn.Module,
    diffusion_model_pretrained_path: str | None,
    text_encoder_pretrained_path: str | None,
    vae_pretrained_path: str | None,
    image_encoder_pretrained_path: str | None,
    vlm_pretrained_path: str | None,
    pretrained_action_adapter_index: int | None,
) -> set[str]:
    """Load explicitly supplied raw weights and return affected policy keys."""

    loaded: set[str] = set()
    wan22_optional = tuple(
        key
        for key in core.state_dict()
        if key.startswith(
            ("action_encoder.", "action_decoder.", "state_encoder.", "img_emb.")
        )
        or any(
            fragment in key
            for fragment in (
                ".cross_attn.k_img.",
                ".cross_attn.v_img.",
                ".cross_attn.norm_k_img.",
            )
        )
    )
    specifications = (
        (
            "wan22_dit",
            core,
            diffusion_model_pretrained_path,
            core,
            wan22_optional,
            (),
            (),
            True,
        ),
        ("t5", t5, text_encoder_pretrained_path, t5, (), (), (), False),
        (
            "wan22_vae",
            vae,
            vae_pretrained_path,
            getattr(vae, "model", vae),
            (),
            (),
            (),
            False,
        ),
        (
            "wan21_clip",
            image_encoder,
            image_encoder_pretrained_path,
            getattr(image_encoder, "model", image_encoder),
            ("log_scale", "logit_bias"),
            (),
            ("textual.",),
            False,
        ),
    )
    for (
        component,
        owner,
        path,
        target,
        allowed_missing,
        allowed_missing_prefixes,
        allowed_unexpected,
        safetensors,
    ) in specifications:
        if path is None:
            continue
        state = (
            _load_wan_safetensors(path)
            if safetensors
            else _load_pth_state_dict(path)
        )
        if component == "wan22_dit":
            state = select_pretrained_adapter_row(
                state,
                target.state_dict(),
                source_row=pretrained_action_adapter_index,
            )
        accepted = _validated_module_load(
            target,
            state,
            component=component,
            allowed_missing=allowed_missing,
            allowed_missing_prefixes=allowed_missing_prefixes,
            allowed_unexpected_prefixes=allowed_unexpected,
        )
        del owner
        loaded.update(_policy_state_aliases(policy, target, accepted))
    if vlm_pretrained_path is not None:
        vlm_keys = {
            key
            for key in vlm.state_dict()
            if not key.startswith("qformer.")
        }
        loaded.update(_policy_state_aliases(policy, vlm, vlm_keys))
    return loaded


def _load_wan_safetensors(path: str | Path) -> dict[str, Tensor]:
    """Load Wan single/indexed safetensors with complete index validation."""

    source = Path(path)
    if source.is_file() and source.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(source), device="cpu"))
    if not source.is_dir():
        raise FileNotFoundError(f"Wan checkpoint path does not exist: {source}")
    index_path = source / "diffusion_pytorch_model.safetensors.index.json"
    single_path = source / "diffusion_pytorch_model.safetensors"
    if index_path.is_file() and single_path.is_file():
        raise ValueError("Wan checkpoint contains both single and indexed weights")
    if single_path.is_file():
        from safetensors.torch import load_file

        return dict(load_file(str(single_path), device="cpu"))
    if not index_path.is_file():
        raise FileNotFoundError(
            "Wan checkpoint requires diffusion_pytorch_model.safetensors "
            "or diffusion_pytorch_model.safetensors.index.json"
        )
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Wan safetensors index has an invalid weight_map")
    from safetensors.torch import load_file

    merged: dict[str, Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
            raise ValueError(f"Unsafe Wan shard name: {shard_name!r}")
        shard_path = source / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Wan checkpoint shard is missing: {shard_path}")
        shard = dict(load_file(str(shard_path), device="cpu"))
        collisions = set(merged) & set(shard)
        if collisions:
            raise ValueError(
                "Duplicate Wan tensor keys across shards: "
                + ", ".join(sorted(collisions)[:5])
            )
        merged.update(shard)
    indexed = set(weight_map)
    if set(merged) != indexed:
        missing = sorted(indexed - set(merged))
        extra = sorted(set(merged) - indexed)
        raise ValueError(
            f"Wan shard/index key mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    return merged


def _load_pth_state_dict(path: str | Path) -> dict[str, Tensor]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Component checkpoint does not exist: {source}")
    value = torch.load(source, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for wrapper in ("state_dict", "model_state"):
            nested = value.get(wrapper)
            if isinstance(nested, dict):
                value = nested
                break
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(tensor, Tensor)
        for key, tensor in value.items()
    ):
        raise TypeError(f"Component checkpoint {source} is not a tensor state dict")
    return dict(value)


def _validated_module_load(
    module: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    component: str,
    allowed_missing: tuple[str, ...] = (),
    allowed_missing_prefixes: tuple[str, ...] = (),
    allowed_unexpected_prefixes: tuple[str, ...] = (),
) -> set[str]:
    target = module.state_dict()
    missing = sorted(
        key
        for key in set(target) - set(state_dict)
        if key not in allowed_missing
        and not key.startswith(allowed_missing_prefixes)
    )
    unexpected = sorted(
        key
        for key in set(state_dict) - set(target)
        if not key.startswith(allowed_unexpected_prefixes)
    )
    shape_mismatches = sorted(
        key
        for key in set(target) & set(state_dict)
        if tuple(target[key].shape) != tuple(state_dict[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            f"{component} raw weight validation failed: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatches={shape_mismatches[:5]}"
        )
    accepted = {key: value for key, value in state_dict.items() if key in target}
    module.load_state_dict(accepted, strict=False)
    return set(accepted)


def _policy_state_aliases(
    policy: nn.Module,
    module: nn.Module,
    local_keys: set[str],
) -> set[str]:
    """Return every policy state key sharing a loaded tensor object."""

    module_state = module.state_dict(keep_vars=True)
    unknown = local_keys - set(module_state)
    if unknown:
        raise ValueError(
            "Loaded component keys are absent from its module state: "
            + ", ".join(sorted(unknown)[:5])
        )
    loaded_ids = {id(module_state[key]) for key in local_keys}
    return {
        key
        for key, tensor in policy.state_dict(keep_vars=True).items()
        if id(tensor) in loaded_ids
    }


def _load_wsp_policy_state(
    path: Path | None,
    *,
    validate_artifacts: bool = True,
) -> dict[str, Tensor]:
    if path is None:
        raise ValueError("checkpoint path is required")
    if path.is_dir() and (path / "checkpoint_manifest.json").is_file():
        if validate_artifacts:
            validate_native_checkpoint_artifacts(path)
        return load_checkpoint_state_dict(path)
    if path.is_dir() and (
        (path / "model.safetensors").is_file()
        or (path / "model.safetensors.index.json").is_file()
    ):
        return load_checkpoint_state_dict(path)
    if path.is_dir() and (path / "policy.pt").is_file():
        return _load_pth_state_dict(path / "policy.pt")
    if path.is_file():
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict) and isinstance(value.get("model"), dict):
            value = value["model"]
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(tensor, Tensor)
            for key, tensor in value.items()
        ):
            raise TypeError(f"WSP checkpoint {path} has no policy tensor state")
        return dict(value)
    raise FileNotFoundError(f"Unsupported WSP checkpoint artifact: {path}")


def _load_policy_state(
    policy: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    allow_stage2_missing: bool,
    pretrained_action_adapter_index: int | None = None,
) -> set[str]:
    target = policy.state_dict()
    qformer_prefix = "condition_router.auto.vlm.qformer."
    if not any(key.startswith(qformer_prefix) for key in target):
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(qformer_prefix)
        }
    state_dict = select_pretrained_adapter_row(
        state_dict,
        target,
        source_row=pretrained_action_adapter_index,
    )
    source_keys = set(state_dict)
    target_keys = set(target)
    unexpected = sorted(source_keys - target_keys)
    mismatched = sorted(
        key
        for key in source_keys & target_keys
        if tuple(state_dict[key].shape) != tuple(target[key].shape)
    )
    missing = sorted(target_keys - source_keys)
    disallowed_missing = (
        [
            key
            for key in missing
            if not key.startswith(_STAGE2_NEW_PREFIXES)
        ]
        if allow_stage2_missing
        else missing
    )
    if unexpected or mismatched or disallowed_missing:
        raise ValueError(
            "WSP checkpoint validation failed: "
            f"missing={disallowed_missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatches={mismatched[:5]}"
        )
    policy.load_state_dict(dict(state_dict), strict=not allow_stage2_missing)
    return source_keys


def _initialization_report(
    policy: nn.Module,
    raw_loaded: set[str],
    checkpoint_loaded: set[str],
) -> InitializationReport:
    keys = set(policy.state_dict())
    raw_final = raw_loaded - checkpoint_loaded
    return InitializationReport(
        checkpoint=tuple(sorted(checkpoint_loaded)),
        raw_components=tuple(sorted(raw_final)),
        random_init=tuple(sorted(keys - checkpoint_loaded - raw_final)),
    )


def checkpoint_mode(
    checkpoint_dir: str | Path,
    *,
    validate_artifacts: bool = True,
) -> InteractionMode:
    if validate_artifacts:
        validate_native_checkpoint_artifacts(checkpoint_dir)
    directory = Path(checkpoint_dir)
    return InteractionMode.parse(_load_native_model_config(directory).model.mode)


def checkpoint_supports_mode(
    checkpoint: InteractionMode | str,
    requested: InteractionMode | str,
) -> bool:
    """Auto checkpoints contain both Auto/VLM and Interactive/T5 routes."""

    checkpoint_mode_value = InteractionMode.parse(checkpoint)
    requested_mode = InteractionMode.parse(requested)
    return (
        checkpoint_mode_value is InteractionMode.AUTO
        or requested_mode is checkpoint_mode_value
    )


def _load_native_model_config(directory: Path) -> ModelConfig:
    path = directory / "model_config.yaml"
    if not path.is_file():
        raise FileNotFoundError("Native checkpoint is missing model_config.yaml")
    return ModelConfig.from_dict(
        OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    )


def _load_generation_config(directory: Path) -> GenerationConfig:
    path = directory / "generation_config.yaml"
    if not path.is_file():
        raise FileNotFoundError("Native checkpoint is missing generation_config.yaml")
    return GenerationConfig.from_dict(
        OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    )


def _native_build_inputs(
    config: ModelConfig,
    generation: GenerationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt typed native fields to the existing numerical construction API."""

    model = config.model
    shape = model.shape
    memory = model.event_memory
    visual = model.visual_memory
    wam = model.wam
    auto = model.condition_router.auto
    action = {
        "use_vlm_tokens": model.mode == "auto",
        "num_frames": shape.num_frames,
        "num_frame_per_block": shape.frame_block_size,
        "num_action_per_block": shape.actions_per_block,
        "num_state_per_block": shape.states_per_block,
        "action_horizon": shape.action_horizon,
        "action_dim": shape.action_dim,
        "max_state_dim": shape.max_state_dim,
        "vlm_token_dim": shape.vlm_token_dim,
        "vlm_context_dim": shape.condition_dim,
        "vlm_projector": auto.projector.kind if auto is not None else "linear",
        "output_norm": auto.output_norm if auto is not None else False,
        "align_projector_gate_only": (
            auto.semantic_gate_only if auto is not None else False
        ),
        "align_proj_gate_grad_clip_norm": (
            auto.semantic_grad_clip_norm if auto is not None else 0.5
        ),
        "text_encoder_cfg": model.condition_router.interactive.t5.instantiate_dict(),
        "vae_cfg": visual.vae.instantiate_dict(),
        "image_encoder_cfg": visual.image_encoder.instantiate_dict(),
        "diffusion_model_cfg": wam.core.instantiate_dict(),
        "enable_latent_cot_memory": memory.enabled,
        "infer_memory_fifo_max_steps": memory.history_steps,
        "latent_memory_goal_slots": memory.global_slots,
        "latent_memory_active_slots": memory.local_steps,
        "latent_memory_done_slots": memory.boundary_steps,
        "latent_memory_done_min_gap": memory.boundary_min_gap,
        "latent_memory_perception_gist_tokens": memory.perception_gist_tokens,
        "latent_memory_residual_scale": memory.residual_scale,
        "latent_memory_dropout": memory.dropout,
        "diffusion_view_layout": visual.diffusion_view_layout,
        "view_index": visual.view_index,
        "tiled": visual.tiled,
        "tile_size_height": visual.tile_size[0],
        "tile_size_width": visual.tile_size[1],
        "tile_stride_height": visual.tile_stride[0],
        "tile_stride_width": visual.tile_stride[1],
        "persistent_visual_prompt": visual.persistent_prompt,
        "num_timestep_buckets": wam.num_timestep_buckets,
        "train_architecture": wam.train_architecture,
        "decouple_inference_noise": wam.decouple_inference_noise,
        "video_inference_final_noise": wam.video_inference_final_noise,
        "decouple_video_action_noise": wam.decouple_video_action_noise,
        "video_noise_beta_alpha": wam.video_noise_beta_alpha,
        "video_noise_beta_beta": wam.video_noise_beta_beta,
        "use_high_noise_emphasis": wam.use_high_noise_emphasis,
        "high_noise_beta_alpha": wam.high_noise_beta_alpha,
        "high_noise_beta_beta": wam.high_noise_beta_beta,
        "num_dit_steps": generation.num_dit_steps,
        "native_num_inference_steps": generation.num_inference_steps,
        "dynamic_cache_schedule": generation.dynamic_cache_schedule,
        "kv_cache_fifo": generation.kv_cache_fifo,
        "cfg_scale": generation.cfg_scale,
        "sigma_shift": generation.sigma_shift,
    }
    return {
        "action_horizon": shape.action_horizon,
        "backbone_cfg": (
            auto.vlm.instantiate_dict()
            if auto is not None
            else {}
        ),
    }, action


def _required_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint config is missing mapping {key!r}")
    return value


def _required_value(parent: dict[str, Any], key: str) -> Any:
    if key not in parent or parent[key] is None:
        raise ValueError(f"Checkpoint config is missing value {key!r}")
    return parent[key]


def _instantiate(
    config: dict[str, Any],
    *,
    target: str | None = None,
    **overrides: Any,
) -> nn.Module:
    component = deepcopy(config)
    configured_target = target if target is not None else component.get("_target_")
    if not isinstance(configured_target, str) or not configured_target:
        raise ValueError("Native component config requires a non-empty _target_")
    if not configured_target.startswith("worldscape_policy."):
        raise ValueError(
            f"Unsupported native component _target_: {configured_target!r}"
        )
    component["_target_"] = configured_target
    if component["_target_"].endswith(".QwenPlanningEncoder"):
        component.pop("freeze_vlm", None)
        component.pop("freeze_qformer", None)
    component.update(overrides)
    result = instantiate(OmegaConf.create(component))
    if not isinstance(result, nn.Module):
        raise TypeError(f"Component {component.get('_target_')} is not an nn.Module")
    return result


def _resolve_tokenizer_path(
    directory: Path | None,
    action: dict[str, Any],
) -> str:
    if directory is not None and directory.is_dir():
        bundled = directory / "tokenizer"
        if bundled.is_dir():
            return str(bundled)
        reference_path = directory / "tokenizer_reference.json"
        if reference_path.is_file():
            reference = json.loads(reference_path.read_text())
            identifier = reference.get("identifier")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(
                    "tokenizer_reference.json requires a non-empty identifier"
                )
            return identifier
    text = _required_mapping(action, "text_encoder_cfg")
    direct = text.get("tokenizer_path") or action.get("tokenizer_path")
    if direct:
        return str(direct)
    experiment = (
        directory / "experiment_cfg" / "conf.yaml"
        if directory is not None and directory.is_dir()
        else None
    )
    if experiment is not None and experiment.exists():
        config = OmegaConf.load(experiment)
        value = OmegaConf.select(config, "tokenizer_path")
        if value:
            return str(value)
    raise ValueError(
        "Checkpoint config does not define a tokenizer_path; "
        "text_encoder_pretrained_path is a weights file, not a tokenizer"
    )


def _build_projector(action: dict[str, Any]) -> nn.Module:
    input_dim = int(_required_value(action, "vlm_token_dim"))
    output_dim = int(_required_value(action, "vlm_context_dim"))
    kind = str(action.get("vlm_projector", "linear"))
    if kind == "linear":
        return nn.Linear(input_dim, output_dim, bias=False)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
    if kind == "identity":
        if input_dim != output_dim:
            raise ValueError("Identity VLM projector requires equal dimensions")
        return nn.Identity()
    raise ValueError(f"Unsupported VLM projector: {kind!r}")


def _build_event_memory(action: dict[str, Any]) -> nn.Module:
    if not bool(action.get("enable_latent_cot_memory", False)):
        return _PassthroughEventMemory()
    return EventMemoryFusion(
        context_dim=int(_required_value(action, "vlm_context_dim")),
        goal_slots=int(action.get("latent_memory_goal_slots", 1)),
        active_slots=int(action.get("latent_memory_active_slots", 4)),
        done_slots=int(action.get("latent_memory_done_slots", 8)),
        done_min_gap=int(action.get("latent_memory_done_min_gap", 1)),
        perception_gist_tokens=int(
            action.get("latent_memory_perception_gist_tokens", 8)
        ),
        residual_scale=float(action.get("latent_memory_residual_scale", 0.1)),
        dropout=float(action.get("latent_memory_dropout", 0.0)),
    )


def _kernel_config(
    raw: dict[str, Any],
    action: dict[str, Any],
) -> Wan22KernelConfig:
    num_steps = int(action.get("native_num_inference_steps", 16))
    masks = {
        5: (True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False),
        6: (True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True),
        7: (True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True),
        8: (True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True),
    }
    compute_steps = int(action.get("num_dit_steps", 8))
    dynamic_schedule = bool(action.get("dynamic_cache_schedule", False))
    if not dynamic_schedule and compute_steps not in masks:
        raise ValueError(
            "Static DiT scheduling supports num_dit_steps values 5, 6, 7, or 8; "
            f"got {compute_steps}"
        )
    if (
        not dynamic_schedule
        and compute_steps in masks
        and num_steps != len(masks[compute_steps])
    ):
        raise ValueError(
            "Legacy sparse DiT masks require exactly 16 inference steps"
        )
    return Wan22KernelConfig(
        num_train_timesteps=int(action.get("num_timestep_buckets", 1000)),
        num_inference_steps=num_steps,
        num_frame_per_block=int(_required_value(action, "num_frame_per_block")),
        num_action_per_block=int(
            _required_value(action, "num_action_per_block")
        ),
        num_state_per_block=int(
            _required_value(action, "num_state_per_block")
        ),
        action_horizon=int(
            action.get("action_horizon", raw.get("action_horizon", 0))
        ),
        cfg_scale=float(action.get("cfg_scale", 1.0)),
        sigma_shift=float(action.get("sigma_shift", 5.0)),
        decouple_inference_noise=bool(
            action.get("decouple_inference_noise", False)
        ),
        video_inference_final_noise=float(
            action.get("video_inference_final_noise", 0.8)
        ),
        dynamic_cache_schedule=dynamic_schedule,
        dit_step_mask=masks.get(compute_steps, (True,) * num_steps),
        decouple_video_action_noise=bool(
            action.get("decouple_video_action_noise", False)
        ),
        video_noise_beta_alpha=float(
            action.get("video_noise_beta_alpha", 3.0)
        ),
        video_noise_beta_beta=float(action.get("video_noise_beta_beta", 1.0)),
        use_high_noise_emphasis=bool(
            action.get("use_high_noise_emphasis", False)
        ),
        high_noise_beta_alpha=float(
            action.get("high_noise_beta_alpha", 3.0)
        ),
        kv_cache_fifo=bool(action.get("kv_cache_fifo", False)),
    )


def _validate_block_geometry(action: dict[str, Any], core: nn.Module) -> None:
    fixed = {
        "num_frame_per_block": 2,
        "num_state_per_block": 1,
    }
    for name, required in fixed.items():
        configured = int(action.get(name, required))
        actual = int(getattr(core, name, configured))
        if configured != required or actual != required:
            raise ValueError(
                f"Native Wan2.2 requires {name}={required}; "
                f"config={configured}, core={actual}"
            )
    configured_actions = int(action.get("num_action_per_block", 24))
    actual_actions = int(
        getattr(core, "num_action_per_block", configured_actions)
    )
    action_horizon = int(action.get("action_horizon", configured_actions))
    if configured_actions not in {24, 48}:
        raise ValueError(
            "Native Wan2.2 num_action_per_block must be 24 or 48"
        )
    if (
        actual_actions != configured_actions
        or action_horizon != configured_actions
    ):
        raise ValueError(
            "Native Wan2.2 action_horizon, num_action_per_block, and core "
            "geometry must match; "
            f"horizon={action_horizon}, config={configured_actions}, "
            f"core={actual_actions}"
        )


def _validate_recipe_expectations(
    action: dict[str, Any],
    *,
    expected_num_frames: int | None,
    expected_action_horizon: int | None,
    expected_semantic_gate_only: bool | None,
    expected_semantic_grad_clip_norm: float | None,
) -> None:
    expectations = {
        "num_frames": expected_num_frames,
        "action_horizon": expected_action_horizon,
        "align_projector_gate_only": expected_semantic_gate_only,
        "align_proj_gate_grad_clip_norm": expected_semantic_grad_clip_norm,
    }
    for name, expected in expectations.items():
        if expected is None:
            continue
        actual = _required_value(action, name)
        if isinstance(expected, float):
            matches = float(actual) == expected
        elif isinstance(expected, bool):
            matches = bool(actual) is expected
        else:
            matches = int(actual) == expected
        if not matches:
            raise ValueError(
                f"Native checkpoint manifest {name}={actual!r} does not match "
                f"recipe requirement {expected!r}"
            )


def _reject_legacy_runtime_environment() -> None:
    leaked = [
        name
        for name in (
            "NUM_DIT_STEPS",
            "DYNAMIC_CACHE_SCHEDULE",
            "KV_CACHE_FIFO",
        )
        if os.environ.get(name) is not None
    ]
    if leaked:
        raise ValueError(
            "Native inference does not read legacy runtime environment variables "
            f"({', '.join(leaked)}); persist these settings in generation_config.yaml"
        )
