"""Configuration objects for GeoRefiner.

Values not specified by the paper are implementation assumptions and are kept
configurable so later stages can tune them without changing public interfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class GeoRefinerConfig:
    """Top-level configuration for mock and pretrained GeoRefiner forward paths."""

    action_dim: int = 7
    state_dim: int = 16  # Implementation assumption: benchmark-specific.
    hidden_dim: int = 128  # Implementation assumption.
    visual_feature_dim: int = 128  # Implementation assumption for adapters/mock.
    text_feature_dim: int = 128  # Implementation assumption for adapters/mock.
    num_heads: int = 4
    geometry_layers: int = 2
    region_layers: int = 2
    temporal_layers: int = 2
    refiner_layers: int = 2
    num_region_queries: int = 16
    max_action_horizon: int = 16
    residual_scale: float | Sequence[float] = 0.05
    dropout: float = 0.0
    action_mlp_layers: int = 4
    state_mlp_layers: int = 4
    head_hidden_dim: int | None = None  # Implementation assumption for gate/residual MLPs.
    temporal_residual_kernel_size: int = 3  # Implementation assumption.
    action_loss_position_weight: float = 1.0  # Implementation assumption.
    action_loss_rotation_weight: float = 1.0  # Implementation assumption.
    action_loss_gripper_weight: float = 1.0  # Implementation assumption.
    action_loss_dimension_weights: Sequence[float] | None = None
    lambda_gate: float = 0.01  # Implementation assumption for stage-2 loss.
    lambda_smooth: float = 0.01  # Implementation assumption for stage-2 loss.
    smooth_loss_source: str = "raw_residual"
    return_intermediates: bool = True
    freeze_visual_backbone: bool = True
    freeze_text_backbone: bool = True
    profile: str = "custom"
    backbone_mode: str = "mock"
    dinov2_model_id: str = "facebook/dinov2-large"
    dinov2_revision: str = "main"
    dinov2_local_path: str | None = None
    clip_model_id: str = "openai/clip-vit-base-patch32"
    clip_revision: str = "main"
    clip_local_path: str | None = None
    pretrained_cache_dir: str | None = None
    local_files_only: bool = False
    dinov2_use_cls_token: bool = False
    initialization_seed: int = 42

    def __post_init__(self) -> None:
        self._validate_positive_int("action_dim", self.action_dim)
        self._validate_positive_int("state_dim", self.state_dim)
        self._validate_positive_int("hidden_dim", self.hidden_dim)
        self._validate_positive_int("visual_feature_dim", self.visual_feature_dim)
        self._validate_positive_int("text_feature_dim", self.text_feature_dim)
        self._validate_positive_int("num_heads", self.num_heads)
        self._validate_positive_int("num_region_queries", self.num_region_queries)
        self._validate_positive_int("max_action_horizon", self.max_action_horizon)
        self._validate_positive_int("action_mlp_layers", self.action_mlp_layers)
        self._validate_positive_int("state_mlp_layers", self.state_mlp_layers)
        self._validate_positive_int(
            "temporal_residual_kernel_size", self.temporal_residual_kernel_size
        )
        if self.head_hidden_dim is not None:
            self._validate_positive_int("head_hidden_dim", self.head_hidden_dim)

        for name in (
            "geometry_layers",
            "region_layers",
            "temporal_layers",
            "refiner_layers",
        ):
            self._validate_non_negative_int(name, getattr(self, name))

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads; "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )

        self._validate_residual_scale()

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout}.")

        if self.temporal_residual_kernel_size % 2 == 0:
            raise ValueError(
                "temporal_residual_kernel_size must be odd to preserve time length; "
                f"got {self.temporal_residual_kernel_size}."
            )
        self._validate_loss_config()
        if self.backbone_mode not in ("mock", "pretrained"):
            raise ValueError(
                "backbone_mode must be 'mock' or 'pretrained'; "
                f"got {self.backbone_mode!r}."
            )
        if not isinstance(self.initialization_seed, int) or self.initialization_seed < 0:
            raise ValueError(
                "initialization_seed must be a non-negative integer; "
                f"got {self.initialization_seed!r}."
            )

    @classmethod
    def paper_aligned(cls, **overrides: Any) -> "GeoRefinerConfig":
        """Return the documented forward-reproduction profile.

        Backbone feature dimensions are expected values here. The pretrained
        factory replaces them with values read from the actual model configs.
        """

        values: dict[str, Any] = {
            "profile": "paper_aligned",
            "backbone_mode": "pretrained",
            "action_dim": 7,
            "state_dim": 16,
            "hidden_dim": 768,
            "visual_feature_dim": 1024,
            "text_feature_dim": 512,
            "num_heads": 12,
            "geometry_layers": 2,
            "region_layers": 2,
            "temporal_layers": 2,
            "refiner_layers": 2,
            "action_mlp_layers": 4,
            "state_mlp_layers": 4,
            "num_region_queries": 16,
            "dropout": 0.0,
            "dinov2_use_cls_token": False,
            "initialization_seed": 42,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GeoRefinerConfig":
        """Restore a config from checkpoint-safe JSON data."""

        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        values = asdict(self)
        for name in ("residual_scale", "action_loss_dimension_weights"):
            value = values[name]
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, list)):
                values[name] = list(value)
        return values

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}.")

    @staticmethod
    def _validate_non_negative_int(name: str, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer; got {value!r}.")

    def _validate_residual_scale(self) -> None:
        if isinstance(self.residual_scale, (int, float)):
            if self.residual_scale <= 0:
                raise ValueError(
                    f"residual_scale must be positive; got {self.residual_scale}."
                )
            return

        if isinstance(self.residual_scale, Sequence) and not isinstance(
            self.residual_scale, (str, bytes)
        ):
            if len(self.residual_scale) != self.action_dim:
                raise ValueError(
                    "residual_scale sequence length must match action_dim; "
                    f"got {len(self.residual_scale)} and action_dim={self.action_dim}."
                )
            if any(scale <= 0 for scale in self.residual_scale):
                raise ValueError(
                    f"residual_scale values must be positive; got {self.residual_scale}."
                )
            return

        raise ValueError(
            "residual_scale must be a positive scalar or a sequence with length "
            f"action_dim={self.action_dim}; got {self.residual_scale!r}."
        )

    def _validate_loss_config(self) -> None:
        for name in (
            "action_loss_position_weight",
            "action_loss_rotation_weight",
            "action_loss_gripper_weight",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}.")

        if self.action_loss_dimension_weights is not None:
            if len(self.action_loss_dimension_weights) != self.action_dim:
                raise ValueError(
                    "action_loss_dimension_weights length must match action_dim; "
                    f"got {len(self.action_loss_dimension_weights)} and "
                    f"action_dim={self.action_dim}."
                )
            if any(weight <= 0 for weight in self.action_loss_dimension_weights):
                raise ValueError(
                    "action_loss_dimension_weights values must be positive; "
                    f"got {self.action_loss_dimension_weights}."
                )

        if self.lambda_gate < 0:
            raise ValueError(f"lambda_gate must be non-negative; got {self.lambda_gate}.")
        if self.lambda_smooth < 0:
            raise ValueError(
                f"lambda_smooth must be non-negative; got {self.lambda_smooth}."
            )
        if self.smooth_loss_source not in ("raw_residual", "applied_correction"):
            raise ValueError(
                "smooth_loss_source must be 'raw_residual' or 'applied_correction'; "
                f"got {self.smooth_loss_source!r}."
            )
