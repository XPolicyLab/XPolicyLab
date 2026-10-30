import random
from typing import NamedTuple, Optional


class EmbodimentFactors(NamedTuple):
    body: str
    is_sim: bool
    gripper: Optional[str]


EMBODIMENT_FACTORS = {
    "g1_real": EmbodimentFactors("G1 robot", is_sim=False, gripper="Omnipicker"),
    "g2_real": EmbodimentFactors("G2 robot", is_sim=False, gripper="Omnipicker"),
    "g2_90d": EmbodimentFactors("G2 robot", is_sim=False, gripper="90D"),
    "g2_sim": EmbodimentFactors("G2 robot", is_sim=True, gripper="Omnipicker"),
    "sl_agilex": EmbodimentFactors("Agilex robot", is_sim=False, gripper=None),
    "sl_acone": EmbodimentFactors("ACone robot", is_sim=False, gripper=None),
    "sl_arxvr": EmbodimentFactors("ARX robot", is_sim=False, gripper=None),
    "libero": EmbodimentFactors("Libero robot", is_sim=True, gripper=None),
    "droid": EmbodimentFactors("Droid robot", is_sim=False, gripper=None),
    "ego_hand": EmbodimentFactors("Human hands", is_sim=False, gripper=None),
    "ego_gripper": EmbodimentFactors("Human gripper", is_sim=False, gripper=None),
    "robodojo": EmbodimentFactors("RoboDojo robot", is_sim=False, gripper=None),
    "robodojo_real": EmbodimentFactors("RoboDojo real robot", is_sim=False, gripper=None),
}


def compose_embodiment_cue(
    body_type: Optional[str],
    rng: random.Random,
    sim_drop_prob: float = 0.0,
) -> str:
    if not body_type:
        return ""
    factors = EMBODIMENT_FACTORS.get(str(body_type).strip().lower().replace("-", "_"))
    if factors is None:
        return ""

    keep_sim = factors.is_sim and not (
        sim_drop_prob > 0.0 and rng.random() < sim_drop_prob)
    cue = ("Simulated " + factors.body) if keep_sim else factors.body
    if factors.gripper is not None:
        cue += f", {factors.gripper} gripper"
    return cue + "."
