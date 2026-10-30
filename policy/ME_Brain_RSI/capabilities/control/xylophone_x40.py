"""X40: X39 minus the lowered hexashell gap; fallback hits raised to the hexashell height.

Lowering the hexashell head target from .810 to .804 m made native13 fail with
contact plateaus on keys 4-6 while grasp, route and measured anchor were identical
to the X38 success: the keys are cylinders, a head that touches one is pushed
sideways and the beat leaves the key polygon. Contact is therefore to be avoided,
not tolerated: the imported .023 gap is restored and the endpoint-checked vertical
fallback route now also places the head at .809 m instead of .803 m. The
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
