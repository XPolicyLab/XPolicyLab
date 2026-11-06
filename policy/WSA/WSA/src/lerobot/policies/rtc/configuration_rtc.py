from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.types import RTCAttentionSchedule


@dataclass
class RTCConfig:
    """Runtime-only configuration for Real-Time Chunking guidance."""

    enabled: bool = False
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

    def __post_init__(self) -> None:
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be positive, got {self.execution_horizon}")
