"""Markers for reviewed whole-task abandonment; no robot action or retry changes."""
from ...runtime.session import SwitchReason


def abandoned(ctx, reason, *, detail, source):
    # One task may propagate the same terminal fault through a helper and caller.
    if ctx.switch_request is None:
        yield ctx.switch_to_wam(SwitchReason[reason], detail=str(detail), source=source)
