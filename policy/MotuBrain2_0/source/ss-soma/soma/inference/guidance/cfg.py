"""Sequential classifier-free guidance.

Usage:
    v_vel, a_vel = apply_cfg(predict_fn, cfg_scale)

where predict_fn(uncond: bool) -> (v_vel | None, a_vel).
"""

from soma.inference.types import Velocity
from soma.inference.types import VelocityForward


def apply_cfg(
    predict_fn: VelocityForward,
    cfg_scale: float | None,
) -> Velocity:
    """Apply classifier-free guidance around a prediction function.

    Args:
        predict_fn: callable(uncond: bool) -> (v_vel | None, a_vel)
        cfg_scale: float or None. None = no CFG, return cond prediction directly.
    Returns:
        (v_vel | None, a_vel) with CFG applied
    """
    if cfg_scale is None:
        return predict_fn(False)

    return _cfg_sequential(predict_fn, cfg_scale)


def _cfg_sequential(
    predict_fn: VelocityForward,
    cfg_scale: float,
) -> Velocity:
    v_vel, a_vel = predict_fn(False)
    # Clone to prevent CUDA Graph buffer reuse when torch.compile
    # uses reduce-overhead mode (the uncond forward would overwrite
    # the cond output otherwise).
    if v_vel is not None:
        v_vel = v_vel.clone()
    a_vel = a_vel.clone()
    v_vel_u, a_vel_u = predict_fn(True)
    if v_vel is not None:
        if v_vel_u is None:
            raise RuntimeError("CFG video predictions must have matching shapes")
        v_vel = v_vel_u + cfg_scale * (v_vel - v_vel_u)
    a_vel = a_vel_u + cfg_scale * (a_vel - a_vel_u)
    return v_vel, a_vel
