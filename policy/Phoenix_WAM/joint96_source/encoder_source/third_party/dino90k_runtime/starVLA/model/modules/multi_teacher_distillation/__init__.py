"""Frozen inference exports; teacher objectives and distillation trainer omitted."""
from .student import SharedSpatiotemporalAdapter, StudentVideoEncoder

__all__ = ["SharedSpatiotemporalAdapter", "StudentVideoEncoder"]
