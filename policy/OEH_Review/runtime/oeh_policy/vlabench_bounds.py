"""Workspace bounds for VLABench review mode, derived from the pinned VLABench sources and assets.

A gross-error gate for the rows review mode builds itself (holds and the teacher's corrections; OpenWAM's own rows
stay unbounded), in VLABench's world frame at the point the native rows target and ``ee_state`` reports: the Panda's
``end_effector`` site (robots/single_arm/base.py ``get_qpos_from_ee_pos`` and ``get_end_effector_pos``).

The centre is the shoulder, the origin of ``link1``: joint1 turns about z through it, so it never moves. ``link0`` sits
where robot_config.json's ``franka`` entry puts it (``position``, rotated by ``euler``; robots/base.py ``_build`` sets
them on ``link_base``, which robots/single_arm/franka.py names ``link0``), and the robot is attached at the arena
origin (tasks/dm_task.py ``attach_entity``). The reach is the sum of the body offsets from ``link2`` to ``link7`` and
the site's offset in ``link7`` (panda.xml): the all-links-straight distance, which bounds every joint configuration
(rotations keep lengths; the triangle inequality adds them), plus ``SLACK``. The box is centre +- reach on every axis,
rounded outward to the millimetre: it refuses only targets no joint configuration reaches. VLABench's table is not a
bound here (its height and placement are per scene). ``load_env`` lets a task series override the robot config
(envs/__init__.py); the derivation refuses a series or default config that does, since the box would then be another
robot's. Every number comes from the files; ``bounds_source`` names them with their sha256.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

PLACEHOLDER = "READ_FROM_PINNED"
SLACK = 0.01  # metres: float rounding of the panda.xml offsets and the rounded base yaw (1.57)
KEYS = ("base_position", "left_xyz_bounds", "bounds_source")
PACKAGE = "VLABench"  # the package directory inside the VLABench checkout (VLABENCH_ROOT)
CHAIN = ("link1", "link2", "link3", "link4", "link5", "link6", "link7")
SITE = "end_effector"

# Pinned statements the derivation relies on; a checkout that says otherwise is not this derivation's.
BASE = {"base position on link_base": r"self\.link_base\.pos = pos",
        "base orientation on link_base": r"self\.link_base\.euler = ori",
        "ik at the end_effector site": r"site_name=self\.end_effector_site\.full_identifier",
        "ee position at the site": r"return physics\.bind\(self\.end_effector_site\)\.xpos"}
ROBOT = {"position from the robot config": r'if kwargs\.get\("position", None\) is not None:\n\s+self\.set_base_position',
         "euler from the robot config": r'if kwargs\.get\("euler", None\) is not None:\n\s+self\.set_base_orientation'}
FRANKA = {"link_base is link0": r'return self\._mjcf_model\.find\("body", "link0"\)',
          "end_effector site": r'return self\.mjcf_model\.find\("site", "end_effector"\)'}
LOAD_ENV = {"series robot override": r'robot_config_overide = default_config\.get\("robot", \{\}\)'}
TASK = {"robot attached at the arena origin": r"self\._arena\.attach\(entity\)"}


def _down(value: float) -> float:
    return math.floor(value * 1000 + 1e-6) / 1000


def _up(value: float) -> float:
    return math.ceil(value * 1000 - 1e-6) / 1000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"workspace bounds come from the pinned sources, and {path} is missing")
    return path


def _find(text: str, pattern: str, what: str, path: Path) -> int:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"pinned {path.name} no longer has the {what} statement; review the bounds derivation")
    return text.count("\n", 0, match.start()) + 1


def _euler_matrix(angles) -> np.ndarray:
    """MJCF ``euler`` with the default ``eulerseq="xyz"`` (rotating axes): Rx(a) Ry(b) Rz(c)."""
    a, b, c = (float(v) for v in angles)

    def rx(t):
        return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])

    def ry(t):
        return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])

    def rz(t):
        return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    return rx(a) @ ry(b) @ rz(c)


def name2config(path: Path) -> dict:
    """The ``name2config`` literal of the pinned configs/__init__.py (series -> task names), read without importing."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "name2config"
                                                 for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return value
    raise ValueError(f"{path} defines no name2config literal")


def series_of(task: str, mapping: dict) -> str:
    """``find_key_by_value(name2config, task)`` (utils/utils.py): the series listing the task, else the task itself."""
    for key, value in mapping.items():
        if isinstance(value, list) and task in value:
            return key
        if not isinstance(value, list) and value == task:
            return key
    return task


def _body(root: ET.Element, name: str) -> ET.Element:
    found = [body for body in root.iter("body") if body.get("name") == name]
    if len(found) != 1:
        raise ValueError(f"panda.xml has {len(found)} bodies named {name}")
    return found[0]


def _offset(element: ET.Element) -> float:
    return float(np.linalg.norm([float(v) for v in element.get("pos", "0 0 0").split()]))


def derive_bounds(root, task: str, *, slack: float = SLACK) -> dict:
    """Bounds for one VLABench task (the shared Franka of every primitive task); also the files they come from."""
    root = Path(root).resolve()
    package = root / PACKAGE
    files = {name: _pinned(package / name) for name in (
        "configs/robot_config.json", "configs/task_config.json", "configs/__init__.py",
        "assets/robots/franka_emika_panda/panda.xml", "robots/base.py", "robots/single_arm/base.py",
        "robots/single_arm/franka.py", "envs/__init__.py", "tasks/dm_task.py")}
    lines = {}
    for name, table in (("robots/single_arm/base.py", BASE), ("robots/base.py", ROBOT),
                        ("robots/single_arm/franka.py", FRANKA), ("envs/__init__.py", LOAD_ENV),
                        ("tasks/dm_task.py", TASK)):
        text = files[name].read_text(encoding="utf-8")
        for what, pattern in table.items():
            lines[what] = (name, _find(text, pattern, what, files[name]))
    robots = json.loads(files["configs/robot_config.json"].read_text(encoding="utf-8"))
    franka = robots.get("franka") if isinstance(robots, dict) else None
    if not isinstance(franka, dict) or len(franka.get("position") or []) != 3 or len(franka.get("euler") or []) != 3:
        raise ValueError("robot_config.json has no franka position and euler")
    if franka.get("xml_path") != "franka_emika_panda/panda.xml":
        raise ValueError(f"robot_config.json franka xml_path is {franka.get('xml_path')!r}, not the derived panda.xml")
    tasks = json.loads(files["configs/task_config.json"].read_text(encoding="utf-8"))
    series = series_of(task, name2config(files["configs/__init__.py"]))
    for key in ("default", series):
        if "robot" in (tasks.get(key) or {}):
            raise ValueError(f"task_config.json {key} overrides the robot config; the derived box would be another "
                             "robot's")
    xml = ET.parse(files["assets/robots/franka_emika_panda/panda.xml"]).getroot()
    bodies = {name: _body(xml, name) for name in ("link0",) + CHAIN}
    for parent, child in zip(("link0",) + CHAIN, CHAIN):
        if bodies[child] not in list(bodies[parent].findall("body")):
            raise ValueError(f"panda.xml: {child} is not a child body of {parent}")
    sites = [site for site in bodies["link7"].findall("site") if site.get("name") == SITE]
    if len(sites) != 1:
        raise ValueError(f"panda.xml: link7 has no single {SITE} site")
    if _offset(bodies["link0"]) != 0:
        raise ValueError("panda.xml: link0 has its own offset; robot_config places it")
    shoulder = [float(v) for v in bodies["link1"].get("pos", "0 0 0").split()]
    base = np.asarray(franka["position"], dtype=float)
    centre = base + _euler_matrix(franka["euler"]) @ np.asarray(shoulder)
    offsets = [(name, _offset(bodies[name])) for name in CHAIN[1:]] + [(SITE, _offset(sites[0]))]
    reach = sum(length for _, length in offsets) + slack
    box = [[_down(centre[axis] - reach), _up(centre[axis] + reach)] for axis in range(3)]

    def cite(name):
        return f"{PACKAGE}/{name} sha256={_sha(files[name])}"

    def line(what):
        name, number = lines[what]
        return f"{name}:{number}"

    chain = ", ".join(f"{name} {length:.6f}" for name, length in offsets)
    source = "; ".join([
        "oeh_policy.vlabench_bounds v1: VLABench world frame at the end_effector site, the point IK targets and "
        f"ee_state reports ({line('ik at the end_effector site')}, {line('ee position at the site')}; "
        f"{cite('robots/single_arm/base.py')})",
        f"link0 at robot_config franka position {franka['position']} rotated by euler {franka['euler']} "
        f"({cite('configs/robot_config.json')}; set on link_base {line('base position on link_base')}, "
        f"{line('base orientation on link_base')}, from the config {line('position from the robot config')}, "
        f"{line('euler from the robot config')} ({cite('robots/base.py')}); link_base is link0 "
        f"{line('link_base is link0')}, site {line('end_effector site')} ({cite('robots/single_arm/franka.py')}); "
        f"attached at the arena origin {line('robot attached at the arena origin')} ({cite('tasks/dm_task.py')}))",
        f"no robot override in task_config.json default or {series} ({cite('configs/task_config.json')}; "
        f"{cite('configs/__init__.py')}; load_env {line('series robot override')}, {cite('envs/__init__.py')})",
        f"shoulder (link1 origin, the joint1 axis point) at [{centre[0]:.6f}, {centre[1]:.6f}, {centre[2]:.6f}]; "
        f"reach {reach:.6f} m = body offsets ({chain}) + slack {slack} "
        f"({cite('assets/robots/franka_emika_panda/panda.xml')}); box centre +- reach, rounded outward to mm: {box}",
    ])
    return {"base_position": [float(v) for v in base], "left_xyz_bounds": box, "bounds_source": source,
            "files": list(files.values()), "centre": [float(v) for v in centre], "reach": reach, "series": series}


def apply_pinned_bounds(cfg: dict) -> list[Path]:
    """Fill the review workspace's base position, box and source from the pinned files; typed values must equal the
    derivation. Returns the files they come from, for the frozen identity. A config without a review workspace is
    left alone."""
    block = cfg.get("review_workspace")
    if block is None:
        return []
    if not isinstance(block, dict):
        raise ValueError("review_workspace must be an object")
    native = cfg["vlabench"]
    derived = derive_bounds(native["root"], native["task_name"])
    for key in KEYS:
        value = block.get(key)
        if isinstance(value, str) and PLACEHOLDER in value:
            block[key] = derived[key]
        elif value != derived[key]:  # never typed by hand: a frozen value must be the derivation itself
            raise ValueError(f"review_workspace.{key} differs from the pinned derivation; leave it {PLACEHOLDER}")
    return derived["files"]
