# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Protocol, runtime_checkable


@runtime_checkable
class ResumableBatchSampler(Protocol):
    """Batch sampler protocol for deterministic optimizer-step resume."""

    global_batch_size: int
    local_batch_size: int

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state_dict: dict) -> None: ...

    def set_data_step(self, data_step: int) -> None: ...
