"""Alignment-only Track4World students reading clean VideoDiT tokens."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .query_decoder import TrackQueryDecoder


class _TrackStudent(nn.Module):
    """Query student mapping clean VideoDiT tokens to teacher-space features."""

    def __init__(
        self,
        *,
        video_dim: int,
        hidden_dim: int,
        teacher_dim: int,
        query_decoder: dict[str, Any],
    ) -> None:
        super().__init__()
        self.video_to_track = nn.Linear(video_dim, hidden_dim)
        self.track_queries = nn.Parameter(
            torch.empty(1, int(query_decoder.get("num_queries", 16)), hidden_dim)
        )
        nn.init.normal_(self.track_queries, std=0.02)
        self.decoder = TrackQueryDecoder(
            hidden_dim=hidden_dim,
            num_layers=int(query_decoder.get("num_layers", 2)),
            num_heads=int(query_decoder.get("num_heads", 16)),
            ffn_dim=int(query_decoder.get("ffn_dim", 4 * hidden_dim)),
            dropout=float(query_decoder.get("dropout", 0.0)),
        )
        self.student_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, teacher_dim),
        )

    def forward(
        self,
        clean_video_tokens: torch.Tensor,
        *,
        memory_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        memory = self.video_to_track(clean_video_tokens)
        queries = self.track_queries.expand(int(memory.shape[0]), -1, -1)
        query_tokens = self.decoder(queries, memory, memory_valid=memory_valid)
        return self.student_projector(query_tokens.mean(dim=1))


class TrackBridge(nn.Module):
    """Mean-pooled query student aligned to offline Track4World features."""

    def __init__(
        self,
        *,
        video_dim: int,
        action_dim: int,
        query_decoder: dict[str, Any],
        teacher: dict[str, Any],
        alignment: dict[str, Any],
    ) -> None:
        super().__init__()
        self.video_dim = int(video_dim)
        self.hidden_dim = int(query_decoder.get("hidden_dim", action_dim))
        self.num_queries = int(query_decoder.get("num_queries", 16))
        self.source_layer = int(query_decoder.get("source_layer", 14))
        if self.source_layer < 0:
            raise ValueError("source_layer must be a zero-based transformer layer.")
        self.clean_scope = str(query_decoder.get("clean_scope", "all")).lower()
        if self.clean_scope not in {"current", "local", "all"}:
            raise ValueError(f"Unsupported track clean_scope {self.clean_scope!r}.")
        self.teacher_feature_dim = int(teacher.get("feature_dim", 1430))
        self.loss_weight = float(alignment.get("loss_weight", 0.1))
        self.warmup_steps = int(alignment.get("warmup_steps", 2000))
        if self.loss_weight < 0 or self.warmup_steps < 0:
            raise ValueError("Alignment weight and warmup must be non-negative.")
        self.students = nn.ModuleList(
            [
                _TrackStudent(
                    video_dim=self.video_dim,
                    hidden_dim=self.hidden_dim,
                    teacher_dim=self.teacher_feature_dim,
                    query_decoder=query_decoder,
                )
            ]
        )

    def _clean_slice(self, tokens_per_frame: int, num_condition_frames: int) -> slice:
        scope_frames = {"current": 1, "local": 2, "all": 3}[self.clean_scope]
        if num_condition_frames < scope_frames:
            raise ValueError(
                f"clean_scope={self.clean_scope!r} requires {scope_frames} condition frames."
            )
        clean_end = int(num_condition_frames) * int(tokens_per_frame)
        clean_start = clean_end - scope_frames * int(tokens_per_frame)
        return slice(clean_start, clean_end)

    def encode_video_tokens(
        self,
        video_tokens: torch.Tensor,
        *,
        tokens_per_frame: int,
        num_condition_frames: int = 3,
        memory_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if video_tokens.ndim != 3 or int(video_tokens.shape[-1]) != self.video_dim:
            raise ValueError(
                f"video_tokens must be [B,S,{self.video_dim}], got {tuple(video_tokens.shape)}."
            )
        clean_slice = self._clean_slice(tokens_per_frame, num_condition_frames)
        clean = video_tokens[:, clean_slice]
        clean_valid = None if memory_valid is None else memory_valid[:, clean_slice]
        student = self.students[0](clean, memory_valid=clean_valid)
        # Teachers ship one window per sample, so the batch contract is [B,1,D].
        return student.unsqueeze(1)

    def alignment_loss(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
        teacher_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        if student_feature.ndim != 3 or int(student_feature.shape[1]) != 1:
            raise ValueError(
                f"student_feature must be [B,1,D], got {tuple(student_feature.shape)}."
            )
        teacher = teacher_feature.detach().to(
            device=student_feature.device, dtype=student_feature.dtype
        )
        if tuple(student_feature.shape) != tuple(teacher.shape):
            raise ValueError(
                "Track teacher/student feature shape mismatch: "
                f"{tuple(teacher.shape)} vs {tuple(student_feature.shape)}."
            )
        per_sample = F.mse_loss(
            student_feature.float(), teacher.float(), reduction="none"
        ).mean(dim=(1, 2))
        if teacher_valid is None:
            return per_sample.mean()
        valid = teacher_valid.to(device=per_sample.device, dtype=per_sample.dtype)
        if valid.ndim == 2 and int(valid.shape[1]) == 1:
            valid = valid[:, 0]
        if tuple(valid.shape) != tuple(per_sample.shape):
            raise ValueError(
                "track_teacher_valid must be [B] or [B,1], got "
                f"{tuple(teacher_valid.shape)}."
            )
        return (per_sample * valid).sum() / valid.sum().clamp(min=1.0)

    def alignment_weight(self, global_step: int | None) -> float:
        step = max(0, int(global_step or 0))
        if self.warmup_steps > 0:
            weight = self.loss_weight * min(1.0, step / float(self.warmup_steps))
        else:
            weight = self.loss_weight
        return weight
