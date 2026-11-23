# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.


from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

ROBOTS: tuple[str, ...] = (
    "robodojo",
    "robodojo_xpl",
    "robodojo_xpl_rgb",
)


@dataclass(frozen=True)
class RobotAdapter:
    """What the host and the prompt need to know about one robot."""

    name: str
    robot_cls: type
    host: str
    knowledge: str
    blurb: str
    guides_subdir: str = ""
    guide_names: tuple[str, ...] = ()
    show_examples: str = "`robo.show(<camera>)`"
    services: str = ""

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(getattr(self.robot_cls, "CAMERAS", ()))

    @property
    def live_success(self) -> bool:
        """Whether the benchmark reports task success while the episode runs.

        True when the robot class has ``done`` (and ``terminated`` flags on its
        results).  False for a robot whose success the host reads only after its
        planner has finished (RoboDojo declares ``done`` absent): the contract
        then says that the final scene is judged and nothing reports success.
        """
        return "done" not in getattr(self.robot_cls, "ABSENT_MEMBERS", ())

    def host_module(self) -> Any:
        return importlib.import_module(self.host)


def get_adapter(name: str) -> RobotAdapter:
    """Return the adapter of robot ``name`` (one of :data:`ROBOTS`)."""

    if name not in ROBOTS:
        raise ValueError(f"unknown robot {name!r}; known: {ROBOTS}")
    module = importlib.import_module(f"martingale.{name}")
    adapter = getattr(module, "ADAPTER", None)
    if not isinstance(adapter, RobotAdapter):
        raise ImportError(f"martingale.{name} defines no ADAPTER")
    return adapter


__all__ = ["ROBOTS", "RobotAdapter", "get_adapter"]
