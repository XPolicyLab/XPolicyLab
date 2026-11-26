"""Top-level GeoRefiner core model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from georefiner.adapters import BenchmarkAdapter, ViewAdapter
from georefiner.backbones.base import TextBackboneBase, VisualBackboneBase
from georefiner.backbones.clip_text import CLIPTextBackbone
from georefiner.backbones.dinov2 import DINOv2Backbone
from georefiner.config import GeoRefinerConfig
from georefiner.modules import (
    GeometryConditionedResidualRefiner,
    LanguageGeometryActionEncoder,
)
from georefiner.types import GeoRefinerInput, GeoRefinerOutput


class GeoRefiner(nn.Module):
    """Complete forward-only GeoRefiner core model.

    The model consumes a base VLA's nominal action as input. It does not call a
    real VLA, benchmark, robot, or dataset.
    """

    def __init__(
        self,
        config: GeoRefinerConfig | None = None,
        view_adapter: ViewAdapter | None = None,
        benchmark_adapter: BenchmarkAdapter | None = None,
        visual_backbone: VisualBackboneBase | None = None,
        text_backbone: TextBackboneBase | None = None,
        initialize_core: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or GeoRefinerConfig()
        visual_backbone, text_backbone = self._resolve_backbones(
            visual_backbone, text_backbone
        )
        self.view_adapter = view_adapter or ViewAdapter()
        if self.config.backbone_mode == "pretrained" and (
            self.view_adapter.normalize or self.view_adapter.image_size is not None
        ):
            raise ValueError(
                "Pretrained DINOv2 owns resize and normalization; ViewAdapter must "
                "use image_size=None and normalize=False to avoid preprocessing twice."
            )
        self.benchmark_adapter = benchmark_adapter or BenchmarkAdapter(self.config)
        self.encoder = LanguageGeometryActionEncoder(
            self.config,
            visual_backbone=visual_backbone,
            text_backbone=text_backbone,
        )
        self.residual_refiner = GeometryConditionedResidualRefiner(self.config)
        if initialize_core:
            self.initialize_core_parameters(self.config.initialization_seed)
        else:
            self._initialize_conservative_refinement()

    def forward(self, inputs: GeoRefinerInput) -> GeoRefinerOutput:
        inputs.validate(self.config)

        adapted_inputs = self._adapt_views(inputs)
        adapter_output = self.benchmark_adapter(
            native_nominal_action=adapted_inputs.nominal_action,
            native_state=adapted_inputs.state,
        )
        encoder_output = self.encoder(
            adapted_inputs,
            canonical_nominal_action=adapter_output.canonical_nominal_action,
            action_feature=adapter_output.action_feature,
            state_feature=adapter_output.state_feature,
        )
        refiner_output = self.residual_refiner(
            canonical_nominal_action=adapter_output.canonical_nominal_action,
            trajectory_tokens=encoder_output.trajectory_tokens,
            geometry_tokens=encoder_output.geometry_tokens,
            region_tokens=encoder_output.region_tokens,
            action_padding_mask=adapted_inputs.action_padding_mask,
            geometry_padding_mask=encoder_output.geometry_padding_mask,
        )
        native_refined_action = self.benchmark_adapter.inverse_action(
            refiner_output.refined_action
        )
        intermediates = {}
        if self.config.return_intermediates:
            intermediates = {
                "language_tokens": encoder_output.language_tokens,
                "geometry_tokens": encoder_output.geometry_tokens,
                "region_tokens": encoder_output.region_tokens,
                "trajectory_tokens": encoder_output.trajectory_tokens,
                "fused_action_tokens": refiner_output.fused_action_tokens,
                "action_feature": adapter_output.action_feature,
                "state_feature": adapter_output.state_feature,
            }
        output = GeoRefinerOutput(
            refined_action=native_refined_action,
            canonical_refined_action=refiner_output.refined_action,
            canonical_nominal_action=adapter_output.canonical_nominal_action,
            gate=refiner_output.gate,
            residual=refiner_output.residual,
            applied_correction=refiner_output.applied_correction,
            intermediates=intermediates,
        )
        output.validate(self.config)
        return output

    def set_train_stage(self, stage: str) -> None:
        """Set `requires_grad` according to the paper's three-stage schedule."""

        normalized_stage = stage.lower()
        if normalized_stage == "stage1":
            self.train()
            self._set_requires_grad(False)
            self._set_module_trainable(self.encoder.geometry_encoder, True)
            self._set_module_trainable(self.encoder.region_qformer, True)
            return

        if normalized_stage == "stage2":
            self.train()
            self._set_requires_grad(True)
            if self.config.freeze_visual_backbone:
                self._set_module_trainable(
                    self.encoder.geometry_encoder.visual_backbone,
                    False,
                )
            if self.config.freeze_text_backbone:
                self._set_module_trainable(self.encoder.language_encoder.text_backbone, False)
            return

        if normalized_stage == "stage3":
            self.train()
            self._set_requires_grad(False)
            self._set_module_trainable(
                self.benchmark_adapter.action_adapter.calibration,
                True,
            )
            self._set_module_trainable(
                self.benchmark_adapter.state_adapter.calibration,
                True,
            )
            return

        if normalized_stage == "inference":
            self.eval()
            self._set_requires_grad(False)
            return

        raise ValueError(
            "stage must be one of 'stage1', 'stage2', 'stage3', or 'inference'; "
            f"got {stage!r}."
        )

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def count_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def count_all_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_counts(self) -> dict[str, int]:
        """Return total, backbone, core, trainable, and frozen parameter counts."""

        visual = sum(
            parameter.numel()
            for parameter in self.encoder.geometry_encoder.visual_backbone.parameters()
        )
        text = sum(
            parameter.numel()
            for parameter in self.encoder.language_encoder.text_backbone.parameters()
        )
        total = self.count_all_parameters()
        trainable = self.count_trainable_parameters()
        return {
            "total": total,
            "dinov2": visual,
            "clip_text": text,
            "georefiner_core": total - visual - text,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    @classmethod
    def from_pretrained(
        cls,
        config: GeoRefinerConfig | None = None,
        *,
        dinov2_path: str | None = None,
        clip_path: str | None = None,
    ) -> "GeoRefiner":
        """Build GeoRefiner with real pretrained DINOv2 and CLIP backbones."""

        resolved = config or GeoRefinerConfig.paper_aligned()
        resolved.backbone_mode = "pretrained"
        if dinov2_path is not None:
            resolved.dinov2_local_path = dinov2_path
        if clip_path is not None:
            resolved.clip_local_path = clip_path
        return cls(resolved)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        artifact_dir: str | Path | None = None,
        map_location: str | torch.device = "cpu",
        local_files_only: bool = True,
    ) -> "GeoRefiner":
        """Strictly restore every weight without loading backbone weight files.

        Only the small tokenizer and image-processor artifacts are read from
        ``artifact_dir``. DINOv2 and CLIP structures are created directly from
        Hugging Face config dictionaries embedded in the unified checkpoint.
        """

        if not local_files_only:
            raise ValueError("Unified checkpoint restore requires local_files_only=True.")
        checkpoint_path = Path(checkpoint_path)
        artifact_root = Path(artifact_dir) if artifact_dir is not None else checkpoint_path.parent
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        if checkpoint.get("format_version") != 1:
            raise ValueError(
                "Unsupported checkpoint format_version: "
                f"{checkpoint.get('format_version')!r}."
            )
        config = GeoRefinerConfig.from_dict(dict(checkpoint["config"]))
        config.local_files_only = True
        config.backbone_mode = "pretrained"
        metadata = checkpoint["backbones"]
        visual = DINOv2Backbone.from_config(
            metadata["dinov2_config"],
            artifact_root / "dinov2_processor",
            model_id=metadata["dinov2_model_id"],
            revision=metadata["dinov2_revision"],
            use_cls_token=config.dinov2_use_cls_token,
            frozen=config.freeze_visual_backbone,
        )
        text = CLIPTextBackbone.from_config(
            metadata["clip_text_config"],
            artifact_root / "clip_tokenizer",
            model_id=metadata["clip_model_id"],
            revision=metadata["clip_revision"],
            frozen=config.freeze_text_backbone,
        )
        model = cls(
            config,
            visual_backbone=visual,
            text_backbone=text,
            initialize_core=False,
        )
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model._checkpoint_load_info = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "checkpoint_path": str(checkpoint_path.resolve()),
        }
        model.set_train_stage("inference")
        return model

    def backbone_metadata(self) -> dict[str, Any]:
        """Return serializable config metadata for unified checkpoint restore."""

        visual = self.encoder.geometry_encoder.visual_backbone
        text = self.encoder.language_encoder.text_backbone
        if not isinstance(visual, DINOv2Backbone) or not isinstance(
            text, CLIPTextBackbone
        ):
            raise TypeError("Backbone metadata requires real DINOv2 and CLIP backbones.")
        return {
            "dinov2_model_id": visual.model_id,
            "dinov2_revision": visual.revision,
            "dinov2_config": visual.model.config.to_dict(),
            "dinov2_preprocessing": dict(visual.preprocessing_config),
            "clip_model_id": text.model_id,
            "clip_revision": text.revision,
            "clip_text_config": text.model.config.to_dict(),
        }

    def initialize_core_parameters(self, seed: int = 42) -> None:
        """Deterministically initialize only non-pretrained GeoRefiner modules."""

        backbone_prefixes = (
            "encoder.geometry_encoder.visual_backbone",
            "encoder.language_encoder.text_backbone",
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for name, module in self.named_modules():
                if name.startswith(backbone_prefixes):
                    continue
                if isinstance(module, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(module.in_proj_weight)
                    if module.in_proj_bias is not None:
                        nn.init.zeros_(module.in_proj_bias)
                    if module.bias_k is not None:
                        nn.init.zeros_(module.bias_k)
                    if module.bias_v is not None:
                        nn.init.zeros_(module.bias_v)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    if module.elementwise_affine:
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

            nn.init.normal_(
                self.encoder.geometry_encoder.view_embedding, mean=0.0, std=0.02
            )
            nn.init.normal_(
                self.encoder.region_qformer.region_queries, mean=0.0, std=0.02
            )
            self._initialize_conservative_refinement()

    def _adapt_views(self, inputs: GeoRefinerInput) -> GeoRefinerInput:
        if inputs.global_rgb is None and inputs.wrist_rgb is None:
            return inputs
        if inputs.global_rgb is None or inputs.wrist_rgb is None:
            return inputs
        view_output = self.view_adapter(inputs.global_rgb, inputs.wrist_rgb)
        return replace(
            inputs,
            global_rgb=view_output.global_rgb,
            wrist_rgb=view_output.wrist_rgb,
        )

    def _resolve_backbones(
        self,
        visual_backbone: VisualBackboneBase | None,
        text_backbone: TextBackboneBase | None,
    ) -> tuple[VisualBackboneBase | None, TextBackboneBase | None]:
        if self.config.backbone_mode == "mock":
            return visual_backbone, text_backbone

        if visual_backbone is None:
            source = self.config.dinov2_local_path or self.config.dinov2_model_id
            visual_backbone = DINOv2Backbone.from_pretrained(
                source,
                model_id=self.config.dinov2_model_id,
                revision=self.config.dinov2_revision,
                cache_dir=self.config.pretrained_cache_dir,
                local_files_only=self.config.local_files_only,
                use_cls_token=self.config.dinov2_use_cls_token,
                frozen=self.config.freeze_visual_backbone,
            )
        if text_backbone is None:
            source = self.config.clip_local_path or self.config.clip_model_id
            text_backbone = CLIPTextBackbone.from_pretrained(
                source,
                model_id=self.config.clip_model_id,
                revision=self.config.clip_revision,
                cache_dir=self.config.pretrained_cache_dir,
                local_files_only=self.config.local_files_only,
                frozen=self.config.freeze_text_backbone,
            )
        visual_dim = getattr(visual_backbone, "output_dim", None)
        text_dim = getattr(text_backbone, "output_dim", None)
        if not isinstance(visual_dim, int) or not isinstance(text_dim, int):
            raise ValueError("Pretrained backbones must expose integer output_dim values.")
        self.config.visual_feature_dim = visual_dim
        self.config.text_feature_dim = text_dim
        return visual_backbone, text_backbone

    def _initialize_conservative_refinement(self) -> None:
        """Bias the randomly initialized refiner toward small initial corrections.

        This supports conservative residual refinement at initialization. It is
        an implementation assumption, not a paper-specified initialization.
        """

        final_gate_layer = self.residual_refiner.gate_head.mlp[-1]
        if isinstance(final_gate_layer, nn.Linear):
            nn.init.zeros_(final_gate_layer.weight)
            nn.init.constant_(final_gate_layer.bias, -4.0)

        final_residual_layer = self.residual_refiner.residual_head.mlp[-1]
        if isinstance(final_residual_layer, nn.Linear):
            nn.init.normal_(final_residual_layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final_residual_layer.bias)

    def _set_requires_grad(self, value: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(value)

    @staticmethod
    def _set_module_trainable(module: nn.Module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(value)
