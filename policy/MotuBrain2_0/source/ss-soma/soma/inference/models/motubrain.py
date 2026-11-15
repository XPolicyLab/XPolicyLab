"""Falcon Motubrain models with distinct inference execution modes."""

from __future__ import annotations

from falcon.config import TrainerConfig
from falcon.models.motubrain import MotubrainModel

import soma.inference.models.patching as patching


class MotubrainInferModel(MotubrainModel):
    """Motubrain deployment model for inference GPUs such as RTX 5090.

    Falcon's architecture and computation stay unchanged. Architecture-specific
    high-level operators are the portable ones in
    ``soma.inference.models.operators``, which Falcon's model code imports
    directly, so ``torch.compile`` can capture and optimize complete blocks.
    """

    def __init__(self, cfg: TrainerConfig):
        super().__init__(cfg)
        if self.cp_size != 1:
            raise NotImplementedError(
                "MotubrainInferModel requires context_parallel_size=1",
            )
        patching.prepare_motubrain_model(self)
