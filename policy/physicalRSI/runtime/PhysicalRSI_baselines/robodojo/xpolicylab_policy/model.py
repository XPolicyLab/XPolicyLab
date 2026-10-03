"""XPolicyLab model contract backed by the PhysicalRSI skill runtime."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "runtime"))

from XPolicyLab.model_template import ModelTemplate
from PhysicalRSI_baselines.robodojo.skill_model import Model as SkillModel


class Model(SkillModel, ModelTemplate):
    """Expose the official template while retaining the skill lifecycle methods."""

    def __init__(self, model_cfg):
        ModelTemplate.__init__(self)
        SkillModel.__init__(self, model_cfg)


__all__ = ["Model"]
