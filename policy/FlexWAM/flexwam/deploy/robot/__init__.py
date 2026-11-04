"""Robot-specific FlexWAM deployment profiles."""
from __future__ import annotations

import argparse
from typing import Any

from deploy.robot.common import (
    camera_client_key,
    client_field_key,
    derive_router_display_order,
    lookup_client_value,
    require_client_keys,
)
from deploy.robot.dual_ee import DualEeRobot
from deploy.robot.flexiv import FlexivRobot
from deploy.robot.piper import PiperRobot

DEFAULT_ROBOT_TYPE = "piper"

ROBOTS = {
    PiperRobot.name: PiperRobot,
    FlexivRobot.name: FlexivRobot,
    DualEeRobot.name: DualEeRobot,
}

__all__ = [
    "DEFAULT_ROBOT_TYPE",
    "DualEeRobot",
    "FlexivRobot",
    "PiperRobot",
    "ROBOTS",
    "add_robot_type_argument",
    "available_robot_types",
    "camera_client_key",
    "client_field_key",
    "derive_router_display_order",
    "get_robot",
    "lookup_client_value",
    "require_client_keys",
    "resolve_robot_type",
]


def available_robot_types() -> tuple[str, ...]:
    return tuple(sorted(ROBOTS))


def get_robot(robot_type: str):
    key = str(robot_type).strip().lower()
    cls = ROBOTS.get(key)
    if cls is None:
        expected = ", ".join(available_robot_types()) or "<none>"
        raise ValueError(f"Invalid robot_type={robot_type!r}. Expected one of: {expected}.")
    return cls()


def add_robot_type_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_ROBOT_TYPE,
) -> argparse.ArgumentParser:
    names = available_robot_types()
    parser.add_argument(
        "--robot-type",
        default=default,
        choices=names,
        help=(
            "Deployment robot profile registered in deploy.robot.ROBOTS. "
            f"Available: {', '.join(names)}."
        ),
    )
    parser.add_argument(
        "--flexiv",
        action="store_true",
        help="Deprecated alias for `--robot-type flexiv`.",
    )
    return parser


def resolve_robot_type(args: Any, *, default: str = DEFAULT_ROBOT_TYPE) -> str:
    requested = str(getattr(args, "robot_type", default) or default).strip().lower()
    if bool(getattr(args, "flexiv", False)):
        if requested not in {default, "flexiv"}:
            raise ValueError(f"Cannot combine --flexiv with --robot-type {requested!r}.")
        requested = "flexiv"
    return get_robot(requested).name
