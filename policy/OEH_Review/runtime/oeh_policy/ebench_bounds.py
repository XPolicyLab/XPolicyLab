"""Workspace boxes for EBench's review condition, derived from the pinned GenManip sources and EBench assets.

A gross-error gate for the rows review mode builds itself (holds and the teacher's corrections; OpenWAM's own rows stay
unbounded). GenManip's arm poses, measured and commanded, are in each arm's own base frame: the cuRobo ``base_link``
of that arm's planner, at the pose its forward kinematics reports and its IK targets (the ``ee_link``). Both of
lift2's planners load one cuRobo configuration (``genmanip/extensions/robots/default/r5a/lift2.py`` ``set_planner``),
so the two arms get the same box in their own frames.

The box is the all-links-straight reach around the frame's origin: the sum of the URDF joint-origin offsets on the
chain from ``base_link`` to ``ee_link`` (plus any prismatic travel), which bounds every joint configuration (rotations
keep lengths; the triangle inequality adds them), plus a slack, as [-reach, reach] on each axis. It knows no scene
(GenManip publishes no pose of the arm mounts above the floor or the tables), so it is looser than RoboTwin's box, which
cuts at the table. Every number comes from the files; ``bounds_source`` names them with their sha256.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import yaml

PLACEHOLDER = "READ_FROM_PINNED"
SLACK = 0.01  # metres: URDF float rounding
KEYS = ("left_xyz_bounds", "right_xyz_bounds", "bounds_source")
MOVING = ("revolute", "continuous", "prismatic")
LIFT2 = "genmanip/extensions/robots/default/r5a/lift2.py"
CUROBO = "saved/assets/miscs/curobo/R5a/r5a_left_arm.yml"
# Pinned statements the derivation relies on; a checkout that says otherwise is not this derivation's.
LIFT2_STATEMENTS = {
    "planner config": r'"saved/assets/miscs/curobo/R5a/r5a_left_arm\.yml"',
    "urdf path": r'\["urdf_path"\]\.replace\("\$\{ASSETS_DIR\}", current_dir \+ "/saved/assets"\)',
    "left planner": r"left_planner = CuroboPlanner\(lift2_cfg, self\.robot\.prim_path\)",
    "right planner": r"right_planner = CuroboPlanner\(lift2_cfg, self\.robot\.prim_path\)",
}


def _down(value: float) -> float:
    return math.floor(value * 1000 + 1e-6) / 1000


def _up(value: float) -> float:
    return math.ceil(value * 1000 - 1e-6) / 1000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned(path: Path) -> Path:
    """A file the derivation reads; its absence (no checkout, assets not downloaded) stops the freeze."""
    if not path.is_file():
        raise ValueError(f"workspace bounds come from the pinned sources, and {path} is missing")
    return path


def _floats(element, attribute: str, default: str) -> list[float]:
    return [float(value) for value in (element.get(attribute, default) if element is not None else default).split()]


def chain_reach(urdf: ET.Element, base_link: str, ee_link: str) -> dict:
    """The joints from ``base_link`` down to ``ee_link`` and their all-links-straight length."""
    by_child = {joint.find("child").get("link"): joint for joint in urdf.findall("joint")}
    chain, link = [], ee_link
    while link != base_link:
        joint = by_child.get(link)
        if joint is None:
            raise ValueError(f"URDF has no chain from {base_link} to {ee_link}")
        chain.append(joint)
        link = joint.find("parent").get("link")
        if len(chain) > len(by_child):
            raise ValueError("URDF joints form a cycle")
    chain.reverse()
    if not any(joint.get("type") in MOVING for joint in chain):
        raise ValueError(f"no moving joint between {base_link} and {ee_link}")
    if any(joint.get("type") not in MOVING + ("fixed",) for joint in chain):
        raise ValueError(f"unsupported joint type between {base_link} and {ee_link}")
    offsets = []
    for joint in chain:
        length = float(np.linalg.norm(_floats(joint.find("origin"), "xyz", "0 0 0")))
        if joint.get("type") == "prismatic":
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"prismatic joint {joint.get('name')} has no limit")
            length += max(abs(float(limit.get("lower", 0))), abs(float(limit.get("upper", 0))))
        offsets.append((joint.get("name"), length))
    return {"joints": [joint.get("name") for joint in chain], "offsets": offsets,
            "reach": sum(length for _, length in offsets)}


def derive_bounds(root, *, slack: float = SLACK) -> dict:
    """Both arms' boxes from the pinned GenManip checkout and its EBench assets; also the files they come from."""
    root = Path(root).resolve()
    files = {"lift2": _pinned(root / LIFT2), "curobo": _pinned(root / CUROBO)}
    text = files["lift2"].read_text(encoding="utf-8")
    lines = {}
    for what, pattern in LIFT2_STATEMENTS.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"pinned lift2.py no longer has the {what} statement; review the bounds derivation")
        lines[what] = text.count("\n", 0, match.start()) + 1
    config = yaml.safe_load(files["curobo"].read_text(encoding="utf-8")) or {}
    kinematics = ((config.get("robot_cfg") or {}).get("kinematics") or {}) if isinstance(config, dict) else {}
    base_link, ee_link, urdf_path = (kinematics.get(key) for key in ("base_link", "ee_link", "urdf_path"))
    if not all(isinstance(value, str) and value for value in (base_link, ee_link, urdf_path)):
        raise ValueError("the cuRobo configuration names no kinematics base_link, ee_link and urdf_path")
    if "${ASSETS_DIR}" not in urdf_path and not Path(urdf_path).is_absolute():
        # lift2.py substitutes only ${ASSETS_DIR}; cuRobo would resolve any other relative path itself.
        raise ValueError(f"the cuRobo urdf_path {urdf_path!r} is relative without ${{ASSETS_DIR}}; review the bounds "
                         "derivation")
    files["urdf"] = _pinned(Path(urdf_path.replace("${ASSETS_DIR}", str(root / "saved/assets"))).resolve())
    data = chain_reach(ET.parse(files["urdf"]).getroot(), base_link, ee_link)
    reach = data["reach"] + slack
    box = [[_down(-reach), _up(reach)] for _ in range(3)]

    def cite(name):
        path = Path(files[name])
        shown = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        return f"{shown} sha256={_sha(path)}"

    offsets = ", ".join(f"{name} {length:.4f}" for name, length in data["offsets"])
    source = "; ".join([
        "oeh_policy.ebench_bounds v1: each arm's own base frame (cuRobo base_link of its planner), at the pose its FK "
        "reports and its IK targets (ee_link); both planners load one configuration "
        f"({cite('lift2')} lines {lines['planner config']}, {lines['left planner']}-{lines['right planner']}; URDF "
        f"path at line {lines['urdf path']})",
        f"{cite('curobo')}: base_link {base_link}, ee_link {ee_link}; {cite('urdf')}",
        f"reach {data['reach']:.4f} m = joint-origin offsets ({offsets}) + slack {slack}; box {box} per arm "
        "(all-links-straight reach around the frame origin; no scene bound)"])
    return {"left_xyz_bounds": box, "right_xyz_bounds": [list(axis) for axis in box], "bounds_source": source,
            "files": [Path(path) for path in files.values()], "chain": data}


def apply_pinned_bounds(cfg: dict) -> list[Path]:
    """Fill a template's review workspace from the pinned files; typed values must equal the derivation. Returns the
    files the boxes come from, for the frozen identity."""
    block = cfg.get("review_workspace")
    if block is None:
        return []
    if not isinstance(block, dict):
        raise ValueError("review_workspace must be an object")
    derived = derive_bounds(cfg["ebench"]["root"])
    for key in KEYS:
        value = block.get(key)
        if isinstance(value, str) and PLACEHOLDER in value:
            block[key] = derived[key]
        elif value != derived[key]:  # never typed by hand: a frozen value must be the derivation itself
            raise ValueError(f"review_workspace.{key} differs from the pinned derivation; leave it {PLACEHOLDER}")
    return derived["files"]
