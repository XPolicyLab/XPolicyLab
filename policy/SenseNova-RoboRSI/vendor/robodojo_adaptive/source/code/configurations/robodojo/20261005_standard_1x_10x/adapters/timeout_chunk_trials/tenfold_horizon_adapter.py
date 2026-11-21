"""RoboDojo adapter with a 10× physical horizon for review checkpoints."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trial_adapter import Adapter as TrialAdapter


class Adapter(TrialAdapter):
    def __init__(self, environment_dir, options):
        super().__init__(environment_dir, options)
        self.env.step_lim *= 10
