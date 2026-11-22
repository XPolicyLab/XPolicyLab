"""Config-only x2robot_v1 backend for complete-checkpoint serving."""

import sys

from wall_x.data._registry import register_module


def build(cfg, ctx):
    del cfg, ctx
    raise NotImplementedError(
        "The eval-only VAP adapter does not include dataset construction."
    )


def load_x2r_data_config(cfg):
    """Retain the legacy adapter's inference-side config compatibility hook."""
    from wall_x.trainer.adapters.base_adapter import load_x2r_data_config as load

    return load(cfg)


register_module("x2robot_v1", sys.modules[__name__])

__all__ = ["build", "load_x2r_data_config"]
