"""VGGT-Ω and V-JEPA 2.1 distillation into a deployable video encoder."""

from .model import MultiTeacherDistillationModel
from .student import DINOOnlyVideoEncoder, SharedSpatiotemporalAdapter, StudentVideoEncoder
from .types import AdapterStates, StudentOutput, VGGTDiagnosticTargets, VGGTTargets, VideoBatch, VJEPATargets
from .vggt import LocalVGGTBackend, VGGTObjective
from .vjepa import LocalVJEPABackend, SpatialBlockMaskGenerator, VJEPAObjective, VJEPAPredictor

__all__ = [
    "AdapterStates",
    "DINOOnlyVideoEncoder",
    "LocalVGGTBackend",
    "LocalVJEPABackend",
    "MultiTeacherDistillationModel",
    "SharedSpatiotemporalAdapter",
    "SpatialBlockMaskGenerator",
    "StudentOutput",
    "StudentVideoEncoder",
    "VGGTDiagnosticTargets",
    "VGGTObjective",
    "VGGTTargets",
    "VJEPAObjective",
    "VJEPAPredictor",
    "VJEPATargets",
    "VideoBatch",
]
