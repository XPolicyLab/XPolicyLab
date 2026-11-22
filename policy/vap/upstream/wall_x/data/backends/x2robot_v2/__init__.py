"""Config-only x2robot_v2 backend for complete-checkpoint serving.

The action normalizer parses the explicit statistics file directly; no dataset
package or training loader is imported by serving.
"""

import sys

from wall_x.data._registry import register_module


def build(cfg, ctx):
    del cfg, ctx
    raise NotImplementedError(
        "The eval-only VAP adapter does not include dataset construction."
    )


register_module("x2robot_v2", sys.modules[__name__])

__all__ = ["build"]
