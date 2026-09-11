from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

MODEL_CONFIG_SCHEMA_VERSION = "1"
GENERATION_CONFIG_SCHEMA_VERSION = "1"


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _strict(value: object, path: str, fields: set[str]) -> dict[str, Any]:
    result = _mapping(value, path)
    unknown = sorted(set(result) - fields)
    missing = sorted(fields - set(result))
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{path} is missing field(s): {', '.join(missing)}")
    return result


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number")
    return float(value)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ComponentConfig:
    target: str
    parameters: dict[str, Any]

    @classmethod
    def from_dict(cls, value: object, path: str) -> ComponentConfig:
        raw = _strict(value, path, {"target", "parameters"})
        return cls(
            target=_text(raw["target"], f"{path}.target"),
            parameters=_mapping(raw["parameters"], f"{path}.parameters"),
        )

    def instantiate_dict(self) -> dict[str, Any]:
        return {"_target_": self.target, **self.parameters}


@dataclass(frozen=True)
class ShapeConfig:
    num_frames: int
    frame_block_size: int
    actions_per_block: int
    states_per_block: int
    action_horizon: int
    action_dim: int
    max_state_dim: int
    vlm_token_dim: int
    condition_dim: int

    @classmethod
    def from_dict(cls, value: object) -> ShapeConfig:
        path = "model.shape"
        fields = {
            "num_frames",
            "frame_block_size",
            "actions_per_block",
            "states_per_block",
            "action_horizon",
            "action_dim",
            "max_state_dim",
            "vlm_token_dim",
            "condition_dim",
        }
        raw = _strict(value, path, fields)
        result = cls(
            **{key: _integer(raw[key], f"{path}.{key}", minimum=1) for key in fields}
        )
        expected = {
            "frame_block_size": 2,
            "states_per_block": 1,
        }
        for name, required in expected.items():
            if getattr(result, name) != required:
                raise ValueError(
                    f"{path}.{name} must be {required} for native Wan2.2"
                )
        if result.actions_per_block not in {24, 48}:
            raise ValueError(
                "model.shape.actions_per_block must be 24 or 48 for native Wan2.2"
            )
        if (result.num_frames - 1) % result.frame_block_size:
            raise ValueError(
                "model.shape.num_frames must be one anchor plus complete "
                "2-frame causal blocks"
            )
        if result.action_horizon != result.actions_per_block:
            raise ValueError(
                "model.shape.action_horizon must match actions_per_block"
            )
        return result


@dataclass(frozen=True)
class ProjectorConfig:
    kind: Literal["linear", "mlp", "identity"]
    input_dim: int
    output_dim: int

    @classmethod
    def from_dict(cls, value: object) -> ProjectorConfig:
        path = "model.condition_router.auto.projector"
        raw = _strict(value, path, {"kind", "input_dim", "output_dim"})
        kind = raw["kind"]
        if kind not in {"linear", "mlp", "identity"}:
            raise ValueError(f"{path}.kind is unsupported: {kind!r}")
        result = cls(
            kind=kind,
            input_dim=_integer(raw["input_dim"], f"{path}.input_dim", minimum=1),
            output_dim=_integer(raw["output_dim"], f"{path}.output_dim", minimum=1),
        )
        if result.kind == "identity" and result.input_dim != result.output_dim:
            raise ValueError(
                "identity projector requires equal input/output dimensions"
            )
        return result


@dataclass(frozen=True)
class AutoConditionConfig:
    vlm: ComponentConfig
    projector: ProjectorConfig
    output_norm: bool
    semantic_gate_only: bool
    semantic_grad_clip_norm: float

    @classmethod
    def from_dict(cls, value: object) -> AutoConditionConfig:
        path = "model.condition_router.auto"
        raw = _strict(
            value,
            path,
            {
                "vlm",
                "projector",
                "output_norm",
                "semantic_gate_only",
                "semantic_grad_clip_norm",
            },
        )
        result = cls(
            vlm=ComponentConfig.from_dict(raw["vlm"], f"{path}.vlm"),
            projector=ProjectorConfig.from_dict(raw["projector"]),
            output_norm=_boolean(raw["output_norm"], f"{path}.output_norm"),
            semantic_gate_only=_boolean(
                raw["semantic_gate_only"], f"{path}.semantic_gate_only"
            ),
            semantic_grad_clip_norm=_number(
                raw["semantic_grad_clip_norm"],
                f"{path}.semantic_grad_clip_norm",
            ),
        )
        if result.semantic_grad_clip_norm < 0:
            raise ValueError(
                f"{path}.semantic_grad_clip_norm must be non-negative"
            )
        return result


@dataclass(frozen=True)
class InteractiveConditionConfig:
    t5: ComponentConfig

    @classmethod
    def from_dict(cls, value: object) -> InteractiveConditionConfig:
        path = "model.condition_router.interactive"
        raw = _strict(value, path, {"t5"})
        return cls(t5=ComponentConfig.from_dict(raw["t5"], f"{path}.t5"))


@dataclass(frozen=True)
class ConditionRouterConfig:
    auto: AutoConditionConfig | None
    interactive: InteractiveConditionConfig

    @classmethod
    def from_dict(cls, value: object) -> ConditionRouterConfig:
        raw = _mapping(value, "model.condition_router")
        unknown = sorted(set(raw) - {"auto", "interactive"})
        if unknown:
            raise ValueError(
                "model.condition_router has unknown field(s): "
                + ", ".join(unknown)
            )
        if "interactive" not in raw:
            raise ValueError(
                "model.condition_router is missing field(s): interactive"
            )
        return cls(
            auto=(
                AutoConditionConfig.from_dict(raw["auto"])
                if "auto" in raw
                else None
            ),
            interactive=InteractiveConditionConfig.from_dict(raw["interactive"]),
        )


@dataclass(frozen=True)
class EventMemoryConfig:
    enabled: bool
    history_steps: int
    global_slots: int
    local_steps: int
    boundary_steps: int
    boundary_min_gap: int
    perception_gist_tokens: int
    residual_scale: float
    dropout: float

    @classmethod
    def from_dict(cls, value: object) -> EventMemoryConfig:
        path = "model.event_memory"
        fields = {
            "enabled",
            "history_steps",
            "global_slots",
            "local_steps",
            "boundary_steps",
            "perception_gist_tokens",
            "boundary_min_gap",
            "residual_scale",
            "dropout",
        }
        # Accepted only for compatibility with older native manifests.
        compatible = _mapping(value, path)
        compatible.pop("history_stride", None)
        raw = _strict(compatible, path, fields)
        return cls(
            enabled=_boolean(raw["enabled"], f"{path}.enabled"),
            history_steps=_integer(
                raw["history_steps"], f"{path}.history_steps", minimum=1
            ),
            global_slots=_integer(
                raw["global_slots"], f"{path}.global_slots", minimum=1
            ),
            local_steps=_integer(raw["local_steps"], f"{path}.local_steps", minimum=1),
            boundary_steps=_integer(
                raw["boundary_steps"], f"{path}.boundary_steps", minimum=1
            ),
            boundary_min_gap=_integer(
                raw["boundary_min_gap"], f"{path}.boundary_min_gap", minimum=1
            ),
            perception_gist_tokens=_integer(
                raw["perception_gist_tokens"],
                f"{path}.perception_gist_tokens",
                minimum=1,
            ),
            residual_scale=_number(raw["residual_scale"], f"{path}.residual_scale"),
            dropout=_number(raw["dropout"], f"{path}.dropout"),
        )


@dataclass(frozen=True)
class VisualMemoryConfig:
    vae: ComponentConfig
    image_encoder: ComponentConfig
    persistent_prompt: Literal["none", "goal_or_demo"]
    diffusion_view_layout: Literal["mosaic_2x2", "robotwin_concat"]
    view_index: int
    tiled: bool
    tile_size: tuple[int, int]
    tile_stride: tuple[int, int]

    @classmethod
    def from_dict(cls, value: object) -> VisualMemoryConfig:
        path = "model.visual_memory"
        fields = {
            "vae",
            "image_encoder",
            "persistent_prompt",
            "diffusion_view_layout",
            "view_index",
            "tiled",
            "tile_size",
            "tile_stride",
        }
        # Accepted only for compatibility with older native manifests.
        compatible = _mapping(value, path)
        compatible.pop("recent_chunks", None)
        compatible.setdefault("diffusion_view_layout", "mosaic_2x2")
        raw = _strict(compatible, path, fields)
        if raw["persistent_prompt"] not in {"none", "goal_or_demo"}:
            raise ValueError(
                f"{path}.persistent_prompt must be 'none' or 'goal_or_demo'"
            )
        if raw["diffusion_view_layout"] not in {
            "mosaic_2x2",
            "robotwin_concat",
        }:
            raise ValueError(
                f"{path}.diffusion_view_layout must be 'mosaic_2x2' or "
                "'robotwin_concat'"
            )

        def pair(key: str) -> tuple[int, int]:
            item = raw[key]
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) != 2
            ):
                raise ValueError(f"{path}.{key} must contain two integers")
            return tuple(_integer(v, f"{path}.{key}", minimum=1) for v in item)  # type: ignore[return-value]

        return cls(
            vae=ComponentConfig.from_dict(raw["vae"], f"{path}.vae"),
            image_encoder=ComponentConfig.from_dict(
                raw["image_encoder"], f"{path}.image_encoder"
            ),
            persistent_prompt=raw["persistent_prompt"],
            diffusion_view_layout=raw["diffusion_view_layout"],
            view_index=_integer(raw["view_index"], f"{path}.view_index"),
            tiled=_boolean(raw["tiled"], f"{path}.tiled"),
            tile_size=pair("tile_size"),
            tile_stride=pair("tile_stride"),
        )


@dataclass(frozen=True)
class WAMConfig:
    plugin: Literal["wan22"]
    variant: Literal["ti2v-5b"]
    core: ComponentConfig
    num_timestep_buckets: int
    train_architecture: Literal["full"]
    decouple_inference_noise: bool
    video_inference_final_noise: float
    decouple_video_action_noise: bool
    video_noise_beta_alpha: float
    video_noise_beta_beta: float
    use_high_noise_emphasis: bool
    high_noise_beta_alpha: float
    high_noise_beta_beta: float

    @classmethod
    def from_dict(cls, value: object) -> WAMConfig:
        path = "model.wam"
        fields = {
            "plugin",
            "variant",
            "core",
            "num_timestep_buckets",
            "train_architecture",
            "decouple_inference_noise",
            "video_inference_final_noise",
            "decouple_video_action_noise",
            "video_noise_beta_alpha",
            "video_noise_beta_beta",
            "use_high_noise_emphasis",
            "high_noise_beta_alpha",
            "high_noise_beta_beta",
        }
        raw = _strict(value, path, fields)
        if raw["plugin"] != "wan22" or raw["variant"] != "ti2v-5b":
            raise ValueError(
                "model.wam supports only plugin='wan22', variant='ti2v-5b'"
            )
        if raw["train_architecture"] != "full":
            raise ValueError("native inference requires train_architecture='full'")
        return cls(
            plugin="wan22",
            variant="ti2v-5b",
            core=ComponentConfig.from_dict(raw["core"], f"{path}.core"),
            num_timestep_buckets=_integer(
                raw["num_timestep_buckets"], f"{path}.num_timestep_buckets", minimum=1
            ),
            train_architecture="full",
            decouple_inference_noise=_boolean(
                raw["decouple_inference_noise"], f"{path}.decouple_inference_noise"
            ),
            video_inference_final_noise=_number(
                raw["video_inference_final_noise"],
                f"{path}.video_inference_final_noise",
            ),
            decouple_video_action_noise=_boolean(
                raw["decouple_video_action_noise"],
                f"{path}.decouple_video_action_noise",
            ),
            video_noise_beta_alpha=_number(
                raw["video_noise_beta_alpha"], f"{path}.video_noise_beta_alpha"
            ),
            video_noise_beta_beta=_number(
                raw["video_noise_beta_beta"], f"{path}.video_noise_beta_beta"
            ),
            use_high_noise_emphasis=_boolean(
                raw["use_high_noise_emphasis"], f"{path}.use_high_noise_emphasis"
            ),
            high_noise_beta_alpha=_number(
                raw["high_noise_beta_alpha"], f"{path}.high_noise_beta_alpha"
            ),
            high_noise_beta_beta=_number(
                raw["high_noise_beta_beta"], f"{path}.high_noise_beta_beta"
            ),
        )


@dataclass(frozen=True)
class NativeModel:
    mode: Literal["auto", "interactive"]
    shape: ShapeConfig
    condition_router: ConditionRouterConfig
    event_memory: EventMemoryConfig
    visual_memory: VisualMemoryConfig
    wam: WAMConfig

    @classmethod
    def from_dict(cls, value: object) -> NativeModel:
        raw = _strict(
            value,
            "model",
            {
                "mode",
                "shape",
                "condition_router",
                "event_memory",
                "visual_memory",
                "wam",
            },
        )
        if raw["mode"] not in {"auto", "interactive"}:
            raise ValueError("model.mode must be 'auto' or 'interactive'")
        result = cls(
            mode=raw["mode"],
            shape=ShapeConfig.from_dict(raw["shape"]),
            condition_router=ConditionRouterConfig.from_dict(raw["condition_router"]),
            event_memory=EventMemoryConfig.from_dict(raw["event_memory"]),
            visual_memory=VisualMemoryConfig.from_dict(raw["visual_memory"]),
            wam=WAMConfig.from_dict(raw["wam"]),
        )
        auto = result.condition_router.auto
        if result.mode == "auto" and auto is None:
            raise ValueError("Auto model mode requires model.condition_router.auto")
        if auto is not None:
            if auto.projector.input_dim != result.shape.vlm_token_dim:
                raise ValueError(
                    "projector input_dim must equal model.shape.vlm_token_dim"
                )
            if auto.projector.output_dim != result.shape.condition_dim:
                raise ValueError(
                    "projector output_dim must equal model.shape.condition_dim"
                )
            token_mode = str(auto.vlm.parameters.get("vlm_token_mode", "last"))
            if token_mode not in {"last", "qformer"}:
                raise ValueError("vlm_token_mode must be 'last' or 'qformer'")
            if (
                token_mode == "qformer"
                and int(
                    auto.vlm.parameters.get(
                        "qformer_output_dim", result.shape.vlm_token_dim
                    )
                )
                != result.shape.vlm_token_dim
            ):
                raise ValueError(
                    "qformer_output_dim must equal model.shape.vlm_token_dim; "
                    "the shared projector owns the WAM condition_dim mapping"
                )
        if (
            result.mode == "auto"
            and auto is not None
            and auto.vlm.parameters.get(
                "enable_planning_branch", True
            )
            is not True
        ):
            raise ValueError(
                "Auto native parity requires enable_planning_branch=true; "
                "planning CE weight is configured independently"
            )
        return result


@dataclass(frozen=True)
class ModelConfig:
    schema_version: Literal["1"]
    model: NativeModel

    @classmethod
    def from_dict(cls, value: object) -> ModelConfig:
        raw = _strict(value, "model_config", {"schema_version", "model"})
        if raw["schema_version"] != MODEL_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported model_config schema_version: {raw['schema_version']!r}"
            )
        return cls(schema_version="1", model=NativeModel.from_dict(raw["model"]))


@dataclass(frozen=True)
class GenerationConfig:
    num_dit_steps: int
    num_inference_steps: int
    dynamic_cache_schedule: bool
    kv_cache_fifo: bool
    cfg_scale: float
    sigma_shift: float

    @classmethod
    def from_dict(cls, value: object) -> GenerationConfig:
        path = "generation_config"
        fields = {
            "schema_version",
            "num_dit_steps",
            "num_inference_steps",
            "dynamic_cache_schedule",
            "kv_cache_fifo",
            "cfg_scale",
            "sigma_shift",
        }
        raw = _strict(value, path, fields)
        if raw["schema_version"] != GENERATION_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported generation_config schema_version: {raw['schema_version']!r}"
            )
        return cls(
            num_dit_steps=_integer(
                raw["num_dit_steps"], f"{path}.num_dit_steps", minimum=1
            ),
            num_inference_steps=_integer(
                raw["num_inference_steps"], f"{path}.num_inference_steps", minimum=1
            ),
            dynamic_cache_schedule=_boolean(
                raw["dynamic_cache_schedule"], f"{path}.dynamic_cache_schedule"
            ),
            kv_cache_fifo=_boolean(raw["kv_cache_fifo"], f"{path}.kv_cache_fifo"),
            cfg_scale=_number(raw["cfg_scale"], f"{path}.cfg_scale"),
            sigma_shift=_number(raw["sigma_shift"], f"{path}.sigma_shift"),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Process-local settings that are deliberately absent from ModelConfig."""

    device: str = "cpu"
    batch_size: int = 1
    log_level: str = "INFO"
