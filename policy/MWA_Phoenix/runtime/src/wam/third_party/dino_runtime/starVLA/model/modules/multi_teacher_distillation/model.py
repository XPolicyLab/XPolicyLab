"""Deep training module that hides both online teacher implementations."""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn

from .student import DINOOnlyVideoEncoder, StudentVideoEncoder
from .types import VideoBatch
from .vggt import LocalVGGTBackend, VGGTObjective
from .vjepa import LocalVJEPABackend, SpatialBlockMaskGenerator, VJEPAObjective

TeacherType = Literal["vggt", "vjepa"]


class MultiTeacherDistillationModel(nn.Module):
    valid_modes = ("dino_only", "vggt_only", "vjepa_only", "joint")

    def __init__(
        self,
        student: StudentVideoEncoder | DINOOnlyVideoEncoder,
        mode: str,
        max_steps: int,
        vggt_backend: LocalVGGTBackend | None = None,
        vggt_objective: VGGTObjective | None = None,
        vjepa_backend: LocalVJEPABackend | None = None,
        vjepa_objective: VJEPAObjective | None = None,
        mask_generator: SpatialBlockMaskGenerator | None = None,
        final_relation_weight_max: float = 0.0,
        final_relation_warmup_fraction: float = 0.05,
        context_warmup_fraction: float = 0.20,
        context_weight_max: float = 0.5,
        ema_momentum: float = 0.9999,
        provenance: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        if mode not in self.valid_modes:
            raise ValueError(f"Unknown distillation mode {mode!r}; expected one of {self.valid_modes}")
        self.student = student
        self.mode = mode
        self.max_steps = int(max_steps)
        self.final_relation_weight_max = float(final_relation_weight_max)
        self.final_relation_warmup_fraction = float(final_relation_warmup_fraction)
        self.context_warmup_fraction = float(context_warmup_fraction)
        self.context_weight_max = float(context_weight_max)
        self.ema_momentum = float(ema_momentum)
        self.vggt_backend = vggt_backend
        self.vggt_objective = vggt_objective
        self.vjepa_backend = vjepa_backend
        self.vjepa_objective = vjepa_objective
        self.mask_generator = mask_generator
        self.provenance = dict(provenance or {})
        self.ema_student = copy.deepcopy(student).requires_grad_(False).eval()
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        if self.mode in {"vggt_only", "joint"} and (self.vggt_backend is None or self.vggt_objective is None):
            raise ValueError(f"Mode {self.mode!r} requires the VGGT backend and objective")
        if self.mode in {"vjepa_only", "joint"} and (
            self.vjepa_backend is None or self.vjepa_objective is None or self.mask_generator is None
        ):
            raise ValueError(f"Mode {self.mode!r} requires the V-JEPA backend, objective, and mask generator")

    def train(self, mode: bool = True):
        super().train(mode)
        self.ema_student.eval()
        if self.vggt_backend is not None:
            self.vggt_backend.eval()
        if self.vjepa_backend is not None:
            self.vjepa_backend.eval()
        return self

    @staticmethod
    def teacher_for_step(global_step: int, first_teacher: str = "vggt") -> TeacherType:
        if first_teacher not in {"vggt", "vjepa"}:
            raise ValueError(f"first_teacher must be `vggt` or `vjepa`, got {first_teacher!r}")
        if int(global_step) % 2 == 0:
            return first_teacher  # type: ignore[return-value]
        return "vjepa" if first_teacher == "vggt" else "vggt"

    def _context_weight(self, global_step: int) -> float:
        warmup_steps = max(1, int(self.max_steps * self.context_warmup_fraction))
        return self.context_weight_max * min(max(int(global_step), 0) / warmup_steps, 1.0)

    def _final_relation_weight(self, global_step: int) -> float:
        warmup_steps = max(1, int(self.max_steps * self.final_relation_warmup_fraction))
        return self.final_relation_weight_max * min(max(int(global_step), 0) / warmup_steps, 1.0)

    def compute_teacher_step(
        self,
        batch: VideoBatch,
        teacher_type: TeacherType,
        global_step: int,
    ) -> dict[str, torch.Tensor]:
        if self.mode == "dino_only":
            raise RuntimeError("dino_only is an inference/export ablation and has no distillation loss")
        if self.mode == "vggt_only" and teacher_type != "vggt":
            raise ValueError("vggt_only only accepts VGGT steps")
        if self.mode == "vjepa_only" and teacher_type != "vjepa":
            raise ValueError("vjepa_only only accepts V-JEPA steps")
        if not isinstance(self.student, StudentVideoEncoder):
            raise TypeError("Distillation modes require StudentVideoEncoder")

        if teacher_type == "vggt":
            if self.vggt_backend is None or self.vggt_objective is None:
                raise RuntimeError("VGGT step requested without a VGGT branch")
            targets = self.vggt_backend.infer(batch.frames)
            student = self.student._forward_with_states(
                batch.frames,
                batch.frame_positions,
            )
            return self.vggt_objective(
                student,
                batch.frames,
                targets,
                self.vggt_backend,
                final_relation_weight=self._final_relation_weight(global_step),
            )

        if self.vjepa_backend is None or self.vjepa_objective is None or self.mask_generator is None:
            raise RuntimeError("V-JEPA step requested without a V-JEPA branch")
        vjepa_update_step = int(global_step) if self.mode == "vjepa_only" else int(global_step) // 2
        patch_mask = self.mask_generator(
            batch.frames.shape[0],
            batch.frames.shape[1],
            batch.frames.device,
            vjepa_update_step,
        )
        targets = self.vjepa_backend.infer(batch.frames)
        student_tokens = self.student(
            batch.frames,
            batch.frame_positions,
            patch_mask,
        )
        outputs = self.vjepa_objective(
            student_tokens,
            patch_mask,
            targets,
            context_weight=self._context_weight(global_step),
        )
        outputs["mask_curriculum_progress"] = patch_mask.new_tensor(
            self.mask_generator.curriculum_progress(vjepa_update_step),
            dtype=torch.float32,
        )
        return outputs

    def forward(
        self,
        batch: VideoBatch,
        teacher_type: TeacherType,
        global_step: int,
    ) -> dict[str, torch.Tensor]:
        return self.compute_teacher_step(batch, teacher_type, global_step)

    @torch.no_grad()
    def update_ema(self) -> None:
        momentum = self.ema_momentum
        for ema_parameter, online_parameter in zip(
            self.ema_student.parameters(), self.student.parameters(), strict=True
        ):
            ema_parameter.mul_(momentum).add_(online_parameter.detach(), alpha=1.0 - momentum)
        for ema_buffer, online_buffer in zip(self.ema_student.buffers(), self.student.buffers(), strict=True):
            if ema_buffer.is_floating_point():
                ema_buffer.mul_(momentum).add_(online_buffer.detach(), alpha=1.0 - momentum)
            else:
                ema_buffer.copy_(online_buffer)

    def training_state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "student": self.student.state_dict(),
            "ema_student": self.ema_student.state_dict(),
        }
        if self.vggt_objective is not None:
            state["vggt_objective"] = self.vggt_objective.state_dict()
        if self.vjepa_objective is not None:
            state["vjepa_objective"] = self.vjepa_objective.state_dict()
        return state

    def load_training_state_dict(self, state: dict[str, object]) -> None:
        self.student.load_state_dict(state["student"], strict=True)
        self.ema_student.load_state_dict(state["ema_student"], strict=True)
        if self.vggt_objective is not None:
            self.vggt_objective.load_state_dict(state["vggt_objective"], strict=True)
        if self.vjepa_objective is not None:
            self.vjepa_objective.load_state_dict(state["vjepa_objective"], strict=True)

    def export_ema_student(self) -> dict[str, object]:
        metadata = {
            "image_size": 256,
            "patch_size": 16,
            "patch_grid": [16, 16],
            "feature_dim": 1024,
            "valid_lengths": [4, 8, 12, 16],
            **self.provenance,
        }
        if isinstance(self.student, StudentVideoEncoder):
            metadata.update(
                {
                    "adapter_width": self.student.adapter.width,
                    "adapter_depth": self.student.adapter.depth,
                    "adapter_supervision_taps": list(self.student.adapter.supervision_taps),
                    "adapter_layer_scale_init": self.student.adapter.layer_scale_init,
                }
            )
        return {
            "format": "starvla.multi_teacher_distillation.student.v1",
            "mode": self.mode,
            "student": self.ema_student.state_dict(),
            "metadata": metadata,
        }
