"""Real pretrained and offline mock backbone implementations."""

from georefiner.backbones.base import (
    TextBackboneBase,
    TextBackboneOutput,
    VisualBackboneBase,
)
from georefiner.backbones.clip_text import CLIPTextBackbone
from georefiner.backbones.dinov2 import DINOv2Backbone
from georefiner.backbones.mock import MockTextBackbone, MockVisualBackbone

__all__ = [
    "MockTextBackbone",
    "MockVisualBackbone",
    "TextBackboneBase",
    "TextBackboneOutput",
    "VisualBackboneBase",
    "CLIPTextBackbone",
    "DINOv2Backbone",
]
