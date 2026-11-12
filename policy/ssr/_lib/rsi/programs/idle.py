from __future__ import annotations

from ..core import actions
from . import Program, register


@register
class Idle(Program):
    name = "idle"

    def match(self, instruction, first_obs) -> float:
        return 1e-06

    def run(self, ctx):
        while True:
            yield actions.hold(ctx.obs, n=10, left_grip=actions.OPEN, right_grip=actions.OPEN)
