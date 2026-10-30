"""ANS18: ANS17 with the across-height pinch for every digit 2.

ANS16 eval15 (seed1 layout 3, seed2 layout 7): the top-arc offset pinch, used
when an outer pad left enough reach, came up empty once and unconfirmed once,
while the across-height pinch carried 6/6 digits 2 in dev5 runs (one release
spin, now covered by the ANS17 spin repair). The arc offset is withdrawn; the
7 keeps its reach-checked top-bar offset.
"""
from .ans17_policy import Policy as ANS17


class Policy(ANS17):
    def _grasp_offset(self,side,digit,xy,goal,yaw):
        if digit==2:return None
        return super()._grasp_offset(side,digit,xy,goal,yaw)

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS18',digit_two_grasp='y_pinch_always_unless_tilted')
        return result
