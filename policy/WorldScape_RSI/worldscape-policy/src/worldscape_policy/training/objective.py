"""Composable training objectives for native WorldScape Policy training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from worldscape_policy.conditioning.semantic_forcing import (
    LEGACY_SEMANTIC_TARGET_BUILDER,
    SemanticTargetBuilder,
)


@dataclass(frozen=True)
class LossResult:
    """The differentiable value and structured diagnostics for one objective."""

    loss: Tensor
    metrics: Mapping[str, Tensor] = field(default_factory=dict)
    skipped: bool = False


@dataclass(frozen=True)
class ObjectiveInputs:
    """Predictions, targets, and masks consumed by :class:`CompositeObjective`."""

    video_prediction: Tensor
    video_target: Tensor
    action_prediction: Tensor
    action_target: Tensor
    video_weight: Tensor | None = None
    action_weight: Tensor | None = None
    action_mask: Tensor | None = None
    action_dim_mask: Tensor | None = None
    has_real_action: Tensor | None = None
    semantic_prediction: Tensor | None = None
    semantic_target: Tensor | None = None
    semantic_mask: Tensor | None = None
    semantic_active: bool = True
    planning_logits: Tensor | None = None
    planning_labels: Tensor | None = None


@dataclass(frozen=True)
class ObjectiveResult:
    """Composite loss with both per-term results and flat logging metrics."""

    loss: Tensor
    terms: Mapping[str, LossResult]
    metrics: Mapping[str, Tensor]


def _scalar(value: float | bool, reference: Tensor) -> Tensor:
    return reference.new_tensor(float(value))


def _expand_trailing(value: Tensor, ndim: int) -> Tensor:
    while value.ndim < ndim:
        value = value.unsqueeze(-1)
    return value


def _validate_same_shape(prediction: Tensor, target: Tensor, name: str) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"{name} prediction and target shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )


class VideoFlowLoss(nn.Module):
    """Legacy-compatible weighted velocity-field MSE.

    For video tensors shaped ``[B, C, T, H, W]``, channel and spatial axes are
    reduced first, timestep weights are then applied to ``[B, T]``, and the
    final result is averaged.
    """

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
    ) -> LossResult:
        _validate_same_shape(prediction, target, "video")
        if prediction.ndim < 3:
            raise ValueError("video tensors must have at least [B, C, T] dimensions")
        squared_error = F.mse_loss(
            prediction.float(), target.float(), reduction="none"
        )
        reduce_dims = (1, *range(3, squared_error.ndim))
        per_timestep = squared_error.mean(dim=reduce_dims)
        if weight is not None:
            try:
                per_timestep = per_timestep * weight.to(
                    device=per_timestep.device, dtype=per_timestep.dtype
                )
            except RuntimeError as exc:
                raise ValueError(
                    "video weight must broadcast to [B, T]"
                ) from exc
        loss = per_timestep.mean()
        return LossResult(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "unweighted_mse": squared_error.mean().detach(),
            },
        )


class ActionFlowLoss(nn.Module):
    """Masked action velocity MSE with the legacy non-finite/outlier guard."""

    def __init__(
        self,
        *,
        guard_enabled: bool = True,
        guard_threshold: float = 1_000.0,
        sanitize_nonfinite: bool = True,
    ) -> None:
        super().__init__()
        if guard_threshold <= 0:
            raise ValueError("guard_threshold must be positive")
        self.guard_enabled = guard_enabled
        self.guard_threshold = float(guard_threshold)
        self.sanitize_nonfinite = sanitize_nonfinite

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        weight: Tensor | None = None,
        mask: Tensor | None = None,
        dim_mask: Tensor | None = None,
        has_real_action: Tensor | None = None,
    ) -> LossResult:
        _validate_same_shape(prediction, target, "action")
        if prediction.ndim < 2:
            raise ValueError("action tensors must have at least [B, T] dimensions")
        error = F.mse_loss(prediction.float(), target.float(), reduction="none")
        finite = torch.isfinite(error)
        nonfinite_fraction = 1.0 - finite.float().mean()
        if not bool(finite.all()):
            if not self.sanitize_nonfinite:
                raise FloatingPointError("action loss contains non-finite values")
            error = torch.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)

        combined_mask = torch.ones_like(error)
        if mask is not None:
            combined_mask = combined_mask * self._broadcast_mask(mask, error, "mask")
        if dim_mask is not None:
            combined_mask = combined_mask * self._broadcast_dim_mask(dim_mask, error)
        if has_real_action is not None:
            if has_real_action.shape[0] != error.shape[0]:
                raise ValueError("has_real_action must have one value per batch item")
            combined_mask = combined_mask * _expand_trailing(
                has_real_action.to(device=error.device, dtype=error.dtype), error.ndim
            )

        masked = error * combined_mask
        if error.ndim >= 3:
            reduced = masked.mean(dim=-1)
        else:
            reduced = masked
        if weight is not None:
            try:
                reduced = reduced * weight.to(
                    device=reduced.device, dtype=reduced.dtype
                )
            except RuntimeError as exc:
                raise ValueError(
                    "action weight must broadcast to the reduced action loss"
                ) from exc
        unguarded = reduced.mean()
        guard_hit = bool(
            self.guard_enabled
            and (
                not bool(torch.isfinite(unguarded))
                or bool((unguarded > self.guard_threshold).item())
            )
        )
        loss = torch.zeros_like(unguarded) if guard_hit else unguarded
        return LossResult(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "unguarded_loss": torch.nan_to_num(unguarded.detach()),
                "guard_hit": _scalar(guard_hit, loss),
                "nonfinite_fraction": nonfinite_fraction.detach(),
                "valid_fraction": (combined_mask != 0).float().mean().detach(),
            },
            skipped=guard_hit,
        )

    @staticmethod
    def _broadcast_mask(mask: Tensor, error: Tensor, name: str) -> Tensor:
        value = mask.to(device=error.device, dtype=error.dtype)
        if value.ndim == error.ndim - 1:
            value = value.unsqueeze(-1)
        try:
            return torch.broadcast_to(value, error.shape)
        except RuntimeError as exc:
            raise ValueError(f"action {name} does not broadcast to action shape") from exc

    @staticmethod
    def _broadcast_dim_mask(dim_mask: Tensor, error: Tensor) -> Tensor:
        value = dim_mask.to(device=error.device, dtype=error.dtype)
        if value.ndim == 1 and error.ndim >= 3:
            if value.numel() < error.shape[-1]:
                value = F.pad(value, (0, error.shape[-1] - value.numel()), value=1.0)
            elif value.numel() > error.shape[-1]:
                value = value[: error.shape[-1]]
        if value.ndim == 2 and error.ndim >= 3:
            if value.shape != (error.shape[0], error.shape[-1]):
                raise ValueError(
                    "batched action dim_mask must have shape [B,D]"
                )
            value = value.unsqueeze(1)
        return ActionFlowLoss._broadcast_mask(value, error, "dim_mask")


class AlignmentLoss(nn.Module):
    """Cosine and MSE feature alignment with optional masked token pooling."""

    def __init__(
        self,
        *,
        cosine_weight: float = 1.0,
        mse_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if cosine_weight < 0 or mse_weight < 0:
            raise ValueError("alignment weights must be non-negative")
        if cosine_weight == 0 and mse_weight == 0:
            raise ValueError("at least one alignment weight must be positive")
        self.cosine_weight = float(cosine_weight)
        self.mse_weight = float(mse_weight)

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        prediction_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> LossResult:
        prediction = self._pool(prediction, prediction_mask, "prediction")
        target = self._pool(target, target_mask, "target")
        _validate_same_shape(prediction, target, "alignment")
        prediction = prediction.float()
        target = target.float()
        mse = F.mse_loss(prediction, target)
        cosine = 1.0 - F.cosine_similarity(
            prediction, target, dim=-1, eps=1e-8
        ).mean()
        loss = self.cosine_weight * cosine + self.mse_weight * mse
        return LossResult(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "cosine": cosine.detach(),
                "mse": mse.detach(),
            },
        )

    @staticmethod
    def _pool(value: Tensor, mask: Tensor | None, name: str) -> Tensor:
        if value.ndim == 2:
            return value
        if value.ndim != 3:
            raise ValueError(f"{name} features must have shape [B, D] or [B, L, D]")
        if mask is None:
            return value.mean(dim=1)
        if mask.shape != value.shape[:2]:
            raise ValueError(f"{name}_mask must have shape [B, L]")
        weights = mask.to(device=value.device, dtype=value.dtype).unsqueeze(-1)
        return (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class PlanningCELoss(nn.Module):
    """Token-level planning cross entropy with legacy sequence truncation."""

    def __init__(self, *, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits: Tensor, labels: Tensor) -> LossResult:
        if logits.ndim != 3:
            raise ValueError("planning logits must have shape [B, T, V]")
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if labels.ndim != 2 or labels.shape[0] != logits.shape[0]:
            raise ValueError("planning labels must have shape [B, T]")
        length = min(logits.shape[1], labels.shape[1])
        if length == 0:
            loss = logits.sum() * 0.0
            return LossResult(
                loss=loss,
                metrics={"loss": loss.detach(), "valid_tokens": _scalar(0, loss)},
                skipped=True,
            )
        labels = labels[:, :length].to(device=logits.device, dtype=torch.long)
        valid_tokens = (labels != self.ignore_index).sum()
        if int(valid_tokens.item()) == 0:
            raise ValueError(
                "planning labels contain no supervised tokens; planning CE cannot "
                "be enabled without planning targets"
            )
        loss = F.cross_entropy(
            logits[:, :length].reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )
        return LossResult(
            loss=loss,
            metrics={"loss": loss.detach(), "valid_tokens": valid_tokens.detach()},
        )


class SemanticForcingLoss(nn.Module):
    """Distill a training-only semantic target into condition features."""

    def __init__(
        self,
        *,
        cosine_weight: float = 1.0,
        mse_weight: float = 0.0,
        detach_target: bool = True,
        target_builder: SemanticTargetBuilder = LEGACY_SEMANTIC_TARGET_BUILDER,
    ) -> None:
        super().__init__()
        self.alignment = AlignmentLoss(
            cosine_weight=cosine_weight,
            mse_weight=mse_weight,
        )
        self.detach_target = detach_target
        self.target_builder = target_builder

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        mask: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> LossResult:
        target = self.target_builder(
            training=True,
            explicit_target=target,
        )
        if target is None:
            raise ValueError("semantic forcing requires a semantic target")
        if self.detach_target:
            target = target.detach()
        result = self.alignment(
            prediction,
            target,
            prediction_mask=mask,
            target_mask=target_mask,
        )
        return LossResult(
            loss=result.loss,
            metrics={f"alignment_{key}": value for key, value in result.metrics.items()},
        )


class CompositeObjective(nn.Module):
    """Plan-defined action + world + semantic + optional planning objective."""

    def __init__(
        self,
        *,
        world_flow_weight: float = 1.0,
        semantic_forcing_weight: float = 0.0,
        planning_ce_weight: float = 0.0,
        video_loss: VideoFlowLoss | None = None,
        action_loss: ActionFlowLoss | None = None,
        semantic_loss: SemanticForcingLoss | None = None,
        planning_loss: PlanningCELoss | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("world_flow_weight", world_flow_weight),
            ("semantic_forcing_weight", semantic_forcing_weight),
            ("planning_ce_weight", planning_ce_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.world_flow_weight = float(world_flow_weight)
        self.semantic_forcing_weight = float(semantic_forcing_weight)
        self.planning_ce_weight = float(planning_ce_weight)
        self.video_loss = video_loss or VideoFlowLoss()
        self.action_loss = action_loss or ActionFlowLoss()
        self.semantic_loss = semantic_loss or SemanticForcingLoss()
        self.planning_loss = planning_loss or PlanningCELoss()

    def forward(self, inputs: ObjectiveInputs) -> ObjectiveResult:
        video = self.video_loss(
            inputs.video_prediction, inputs.video_target, inputs.video_weight
        )
        action = self.action_loss(
            inputs.action_prediction,
            inputs.action_target,
            weight=inputs.action_weight,
            mask=inputs.action_mask,
            dim_mask=inputs.action_dim_mask,
            has_real_action=inputs.has_real_action,
        )
        zero = action.loss.new_zeros(())
        semantic = LossResult(zero, {"loss": zero.detach()}, skipped=True)
        if self.semantic_forcing_weight > 0 and inputs.semantic_active:
            if inputs.semantic_prediction is None or inputs.semantic_target is None:
                raise ValueError(
                    "semantic prediction and target are required when semantic forcing is enabled"
                )
            semantic = self.semantic_loss(
                inputs.semantic_prediction,
                inputs.semantic_target,
                mask=inputs.semantic_mask,
            )
        planning = LossResult(zero, {"loss": zero.detach()}, skipped=True)
        if self.planning_ce_weight > 0:
            if inputs.planning_logits is None or inputs.planning_labels is None:
                raise ValueError(
                    "planning logits and labels are required when planning CE is enabled"
                )
            planning = self.planning_loss(
                inputs.planning_logits, inputs.planning_labels
            )

        terms = {
            "action_flow": action,
            "video_flow": video,
            "semantic_forcing": semantic,
            "planning_ce": planning,
        }
        total = (
            action.loss
            + self.world_flow_weight * video.loss
            + self.semantic_forcing_weight * semantic.loss
            + self.planning_ce_weight * planning.loss
        )
        metrics: dict[str, Tensor] = {"loss": total.detach()}
        weights = {
            "action_flow": 1.0,
            "video_flow": self.world_flow_weight,
            "semantic_forcing": self.semantic_forcing_weight,
            "planning_ce": self.planning_ce_weight,
        }
        for name, result in terms.items():
            metrics[f"{name}/weighted_loss"] = (
                result.loss.detach() * weights[name]
            )
            metrics[f"{name}/skipped"] = _scalar(result.skipped, total)
            for metric_name, value in result.metrics.items():
                metrics[f"{name}/{metric_name}"] = value
        return ObjectiveResult(loss=total, terms=terms, metrics=metrics)


WorldScapeObjective = CompositeObjective


__all__ = [
    "ActionFlowLoss",
    "AlignmentLoss",
    "CompositeObjective",
    "LossResult",
    "ObjectiveInputs",
    "ObjectiveResult",
    "PlanningCELoss",
    "SemanticForcingLoss",
    "VideoFlowLoss",
    "WorldScapeObjective",
]
