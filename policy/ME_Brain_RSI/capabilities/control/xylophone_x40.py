"""X40: X39 minus the lowered hexashell gap; fallback hits raised to the hexashell height.

The
lowered-hover retry of X39 is kept.
"""
from .xylophone_x39 import XylophonePolicy as X39
from .xylophone_x38 import XylophonePolicy as X38


class XylophonePolicy(X39):
    FALLBACK_HIT_Z=.809

    def reset(self):
        super().reset()
        self.program_notes.update(candidate='X40_original_gap_raised_fallback_hit')

    def _choose_strokes(self,side,rotation,anchor,prefix=(),reserve=60):
        return X38._choose_strokes(self,side,rotation,anchor,prefix,reserve)
