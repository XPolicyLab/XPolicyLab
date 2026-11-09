"""ANS18: ANS17 with the across-height pinch for every digit 2.

The arc offset is withdrawn; the
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
