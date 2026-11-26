#!/usr/bin/env python3
"""Compatibility fixes for the pinned HarnessVLA LIBERO-PRO runtime.

The released ``rpent-liberopro==0.2.0`` robot model still uses the pre-1.5
robosuite convention ``default_base = None``.  Robosuite 1.5 routes every
robot through ``robot_base_factory`` and represents a robot without a mounted
base as ``NullBase`` instead.  Apply that narrow API adaptation before any
LIBERO-PRO environment is constructed.
"""

from __future__ import annotations


def patch_on_the_ground_panda_null_base() -> bool:
    """Map the legacy no-base sentinel to robosuite 1.5's ``NullBase``.

    Returns ``True`` when the legacy descriptor was replaced and ``False``
    when the runtime was already compatible.  Any unexpected third value is
    rejected so this cannot silently override a future upstream choice.
    """

    from liberopro.liberopro.envs.robots.on_the_ground_panda import (
        OnTheGroundPanda,
    )
    from robosuite.models.bases import BASE_MAPPING

    if "NullBase" not in BASE_MAPPING:
        raise RuntimeError("robosuite runtime does not register NullBase")
    descriptor = OnTheGroundPanda.__dict__.get("default_base")
    if not isinstance(descriptor, property) or descriptor.fget is None:
        raise RuntimeError("OnTheGroundPanda.default_base is not a property")
    current = descriptor.fget(object())
    if current == "NullBase":
        return False
    if current is not None:
        raise RuntimeError(f"Refusing to replace unexpected OnTheGroundPanda base: {current!r}")
    OnTheGroundPanda.default_base = property(lambda self: "NullBase")
    return True


def configure_harnessvla_liberopro_runtime() -> None:
    """Apply all required compatibility fixes to the imported runtime."""

    patch_on_the_ground_panda_null_base()
