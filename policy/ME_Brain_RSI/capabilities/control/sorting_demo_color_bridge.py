"""Preserve demonstration refinement after the actual parent's color redetection."""
from types import FunctionType


def finish_core(ctx, formal_core):
    """Force the existing refinement option without capturing a live robot instance."""
    original = formal_core.__globals__.get('sorting_color_initial')
    if original is None:
        return formal_core

    def refine(robot, objects, refine_demonstrations=False):
        return original(robot, objects, refine_demonstrations=True)

    namespace = dict(formal_core.__globals__, sorting_color_initial=refine)
    result = FunctionType(formal_core.__code__, namespace, formal_core.__name__,
                          formal_core.__defaults__, formal_core.__closure__)
    result.__kwdefaults__ = formal_core.__kwdefaults__
    return result
