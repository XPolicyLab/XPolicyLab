"""VGGT-Ω online teacher and its training-only distillation objective."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import AdapterStates, StudentOutput, VGGTDiagnosticTargets, VGGTTargets


def _fp32_context(tensor: torch.Tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


class LocalVGGTBackend(nn.Module):
    """Online adapter around the frozen official aggregator and heads."""

    def __init__(self, teacher: nn.Module, frames_chunk_size: int = 8) -> None:
        super().__init__()
        self.aggregator = teacher.aggregator
        self.dense_head = teacher.dense_head
        self.camera_head = teacher.camera_head
        self.frames_chunk_size = int(frames_chunk_size)
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def infer(self, frames: torch.Tensor) -> VGGTTargets:
        with torch.inference_mode():
            caches, patch_start = self.aggregator(frames)
            with _fp32_context(frames):
                depth, confidence = self.dense_head(
                    caches,
                    images=frames.float(),
                    patch_token_start=patch_start,
                    frames_chunk_size=self.frames_chunk_size,
                )
        caches = tuple(cache.detach().clone() if cache is not None else None for cache in caches)
        return VGGTTargets(caches, depth.detach().clone(), confidence.detach().clone())

    def infer_with_pose(self, frames: torch.Tensor) -> VGGTDiagnosticTargets:
        """Run the camera head only for offline checkpoint diagnostics."""

        with torch.inference_mode():
            caches, patch_start = self.aggregator(frames)
            with _fp32_context(frames):
                depth, confidence = self.dense_head(
                    caches,
                    images=frames.float(),
                    patch_token_start=patch_start,
                    frames_chunk_size=self.frames_chunk_size,
                )
                pose = self.camera_head(caches, patch_token_start=patch_start)
        caches = tuple(cache.detach().clone() if cache is not None else None for cache in caches)
        return VGGTDiagnosticTargets(
            caches,
            depth.detach().clone(),
            confidence.detach().clone(),
            pose.detach().clone(),
        )

    def decode_student_depth(
        self,
        caches: list[torch.Tensor | None],
        frames: torch.Tensor,
    ) -> torch.Tensor:
        with _fp32_context(frames):
            depth, _ = self.dense_head(
                caches,
                images=frames.float(),
                patch_token_start=17,
                frames_chunk_size=self.frames_chunk_size,
            )
        return depth

    def decode_student_with_pose(
        self,
        caches: list[torch.Tensor | None],
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _fp32_context(frames):
            depth, _ = self.dense_head(
                caches,
                images=frames.float(),
                patch_token_start=17,
                frames_chunk_size=self.frames_chunk_size,
            )
            pose = self.camera_head(caches, patch_token_start=17)
        return depth, pose


class CacheProjector(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 1024) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(tokens))


class VGGTObjective(nn.Module):
    cache_layers = (4, 11, 17, 23)

    def __init__(self, adapter_dim: int = 1024) -> None:
        super().__init__()
        self.local_projectors = nn.ModuleList([CacheProjector(adapter_dim) for _ in self.cache_layers])
        self.inter_projectors = nn.ModuleList([CacheProjector(adapter_dim) for _ in self.cache_layers])

    def predicted_caches(self, states: AdapterStates) -> list[torch.Tensor | None]:
        if len(states.local) != 4 or len(states.inter) != 4:
            raise ValueError("VGGT distillation requires four tapped local/inter adapter states")
        caches: list[torch.Tensor | None] = [None] * 24
        for slot, layer in enumerate(self.cache_layers):
            local = self.local_projectors[slot](states.local[slot])
            inter = self.inter_projectors[slot](states.inter[slot])
            caches[layer] = torch.cat([local, inter], dim=-1)
        return caches

    @staticmethod
    def _normalized_smooth_l1(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        student = F.layer_norm(student.float(), (student.shape[-1],))
        teacher = F.layer_norm(teacher.detach().float(), (teacher.shape[-1],))
        return F.smooth_l1_loss(student, teacher)

    def cache_losses(
        self,
        caches: list[torch.Tensor | None],
        targets: VGGTTargets,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        patch_losses = []
        special_losses = []
        metrics = {}
        for layer in self.cache_layers:
            student = caches[layer]
            teacher = targets.caches[layer]
            if student is None or teacher is None:
                raise ValueError(f"Missing VGGT cache at layer {layer}")
            student_local, student_inter = student.float().chunk(2, dim=-1)
            teacher_local, teacher_inter = teacher.detach().float().chunk(2, dim=-1)

            patch_local = self._normalized_smooth_l1(student_local[:, :, 17:], teacher_local[:, :, 17:])
            patch_inter = self._normalized_smooth_l1(student_inter[:, :, 17:], teacher_inter[:, :, 17:])
            patch = patch_local + 0.5 * patch_inter

            cam_local = F.smooth_l1_loss(student_local[:, :, :1], teacher_local[:, :, :1])
            cam_inter = F.smooth_l1_loss(student_inter[:, :, :1], teacher_inter[:, :, :1])
            reg_local = F.smooth_l1_loss(student_local[:, :, 1:17], teacher_local[:, :, 1:17])
            reg_inter = F.smooth_l1_loss(student_inter[:, :, 1:17], teacher_inter[:, :, 1:17])
            camera = 0.5 * (cam_local + cam_inter)
            registers = 0.5 * (reg_local + reg_inter)
            special = 0.5 * camera + 0.5 * registers

            patch_losses.append(patch)
            special_losses.append(special)
            metrics[f"loss_vggt_patch_l{layer}"] = patch
            metrics[f"loss_vggt_special_l{layer}"] = special
        return torch.stack(patch_losses).mean(), torch.stack(special_losses).mean(), metrics

    @staticmethod
    def _confidence_weights(
        confidence: torch.Tensor,
        valid: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        weights = torch.zeros_like(confidence, dtype=torch.float32)
        for batch_index in range(confidence.shape[0]):
            values = confidence[batch_index][valid[batch_index]].float()
            low, high = torch.quantile(values, torch.tensor([0.1, 0.9], device=values.device))
            if high - low <= epsilon:
                weights[batch_index][valid[batch_index]] = 1.0
            else:
                scaled = ((confidence[batch_index].float() - low) / (high - low + epsilon)).clamp(0.0, 1.0)
                weights[batch_index] = torch.where(valid[batch_index], 0.05 + 0.95 * scaled, 0.0)
        return weights

    @classmethod
    def depth_loss(
        cls,
        student: torch.Tensor,
        teacher: torch.Tensor,
        confidence: torch.Tensor,
        epsilon: float = 1e-6,
        charbonnier_epsilon: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        student = student.float().squeeze(-1)
        teacher = teacher.detach().float().squeeze(-1)
        confidence = confidence.detach().float()
        valid = torch.isfinite(teacher) & torch.isfinite(confidence) & (teacher > 0)
        weights = cls._confidence_weights(confidence, valid, epsilon)
        log_student = torch.log(student.clamp_min(epsilon))
        log_teacher = torch.log(teacher.clamp_min(epsilon))
        error = log_student - log_teacher
        log_depth = (weights * torch.sqrt(error.square() + charbonnier_epsilon**2)).sum() / weights.sum().clamp_min(1)

        horizontal = error[..., :, 1:] - error[..., :, :-1]
        horizontal_weight = torch.minimum(weights[..., :, 1:], weights[..., :, :-1])
        vertical = error[..., 1:, :] - error[..., :-1, :]
        vertical_weight = torch.minimum(weights[..., 1:, :], weights[..., :-1, :])
        gradient = ((horizontal_weight * horizontal.abs()).sum() + (vertical_weight * vertical.abs()).sum()) / (
            horizontal_weight.sum() + vertical_weight.sum()
        ).clamp_min(1)
        return log_depth + 0.2 * gradient, gradient

    @classmethod
    def final_relation_loss(
        cls,
        student_tokens: torch.Tensor,
        targets: VGGTTargets,
    ) -> torch.Tensor:
        teacher_cache = targets.caches[cls.cache_layers[-1]]
        if teacher_cache is None:
            raise ValueError(f"Missing VGGT cache at layer {cls.cache_layers[-1]}")
        teacher_inter = teacher_cache.detach().float().chunk(2, dim=-1)[1][:, :, 17:]
        student = F.layer_norm(student_tokens.float(), (student_tokens.shape[-1],))
        teacher = F.layer_norm(teacher_inter, (teacher_inter.shape[-1],))
        student = F.normalize(student, dim=-1, eps=1e-6)
        teacher = F.normalize(teacher, dim=-1, eps=1e-6)
        student_affinity = student[:, :-1] @ student[:, 1:].transpose(-1, -2)
        teacher_affinity = teacher[:, :-1] @ teacher[:, 1:].transpose(-1, -2)
        return F.smooth_l1_loss(student_affinity, teacher_affinity)

    def forward(
        self,
        student: StudentOutput,
        frames: torch.Tensor,
        targets: VGGTTargets,
        backend: LocalVGGTBackend,
        final_relation_weight: float,
    ) -> dict[str, torch.Tensor]:
        caches = self.predicted_caches(student.states)
        student_depth = backend.decode_student_depth(caches, frames)
        patch, special, cache_metrics = self.cache_losses(caches, targets)
        depth, depth_gradient = self.depth_loss(student_depth, targets.depth, targets.confidence)
        final_relation = self.final_relation_loss(student.tokens, targets)
        loss = patch + 0.25 * special + depth + float(final_relation_weight) * final_relation
        return {
            "loss": loss,
            "loss_vggt_patch": patch,
            "loss_vggt_special": special,
            "loss_depth": depth,
            "loss_depth_gradient": depth_gradient,
            "loss_vggt_final_relation": final_relation,
            "vggt_final_relation_weight": loss.new_tensor(float(final_relation_weight)),
            **cache_metrics,
        }
