"""Unchanged canonical80 numeric representation with a dedicated GR1 prompt slot."""
from .robocasa_delta import (
    CONTRACT as PARENT_CONTRACT, RoboCasaNormalizer as ParentNormalizer,
    masks, project, decode, observation_state, prompt_for, validate_modality,
)

CONTRACT = {**PARENT_CONTRACT, "version": "robocasa_gr1_canonical80_native_delta_slot18_v1",
            "embodiment_id": 18}


class RoboCasaNormalizer(ParentNormalizer):
    contract = CONTRACT
