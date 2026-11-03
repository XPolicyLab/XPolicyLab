"""Pluggable semantic VLM backends for XVLA."""

from .registry import build_vlm_backend, get_vlm_backend_class, register_vlm_backend

__all__ = [
    "build_vlm_backend",
    "get_vlm_backend_class",
    "register_vlm_backend",
]
