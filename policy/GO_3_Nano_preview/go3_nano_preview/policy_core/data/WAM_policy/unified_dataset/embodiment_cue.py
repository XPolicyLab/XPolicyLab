"""Shared Robotwin caption conditioning for training and inference."""
import random
from typing import NamedTuple, Optional

class EmbodimentFactors(NamedTuple):
    body: str
    is_sim: bool
    gripper: Optional[str]

EMBODIMENT_FACTORS = {
    "robotwin": EmbodimentFactors("Agilex robot", is_sim=False, gripper=None),
}

def compose_embodiment_cue(body_type: Optional[str], rng: random.Random = None, sim_drop_prob: float = 0.0) -> str:
    if str(body_type).strip().lower() != 'robotwin':
        raise ValueError('Only robotwin robot type is supported')
    return EMBODIMENT_FACTORS['robotwin'].body + '.'

def condition_caption(caption: str, body_type: str = 'robotwin') -> str:
    cue = compose_embodiment_cue(body_type)
    text = str(caption).strip()
    if text == cue or text.startswith(cue + ' '):
        return text
    return cue + (' ' + text if text else '')
