"""Workspace bounds for RoboCasa GR1 review mode, derived from the pinned robosuite GR1 model.

A gross-error gate for the rows review mode builds itself (holds and the teacher's corrections; OpenWAM's own rows
stay unbounded). The bounds are in the frame and at the point of the EEF33 poses they gate: the GR1 robot base
(robosuite's ``robot0_base`` body, the root of ``robots/gr1/robot.xml``, which robocasa's
``GR1ArmsAndWaistFourierHands`` fixes in the scene), at each Fourier hand's ``grip_site`` (the robot's
``eef_site_id``): OpenWAM's ``gr1_kinematics`` reads and solves every EEF33 pose there.

Per arm the centre is the waist yaw joint's origin, the first joint on the chain from the base to the hand; the reach
is the sum of the body offsets from there to the grip site: the all-links-straight distance, which bounds every
configuration of the waist and the arm (each joint on the chain is a hinge at its body's origin, so rotations keep
the offsets' lengths and the triangle inequality adds them). The box is centre ± (reach + ``SLACK``) on every axis.
It is kinematic only: the counter's height and placement vary with the RoboCasa scene, so no table bound is taken
(RoboDojo's and RoboTwin's z floors come from a fixed table the pinned files give). Every number comes from the
files; ``bounds_source`` names them with their sha256.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

PLACEHOLDER = "READ_FROM_PINNED"
SLACK = 0.01  # metres: float rounding in the model files and the site lookups
FRAME, QUATERNION = "gr1_robot_base", "wxyz"
KEYS = ("left_xyz_bounds", "right_xyz_bounds", "bounds_source")
BASE_BODY = "base"  # the robot model's root body; robosuite prefixes it robot0_
ARMS = ("left", "right")

# Pinned statements the derivation relies on; a checkout that says otherwise is not this derivation's.
ROBOSUITE = {  # paths relative to the installed robosuite package
    "__init__.py": {"version": r'__version__ = "1\.5\.1"'},
    "models/robots/manipulators/gr1_robot.py": {
        "robot xml": r'super\(\)\.__init__\(xml_path_completion\("robots/gr1/robot\.xml"\), idn=idn\)',
        "default gripper": r'return \{"right": "FourierRightHand", "left": "FourierLeftHand"\}',
        "eef names": r'return (\{"right": "right_eef", "left": "left_eef"\})',
    },
    "models/grippers/fourier_hands.py": {
        "left hand xml": r'super\(\)\.__init__\(xml_path_completion\("grippers/fourier_left_hand\.xml"\), idn=idn\)',
        "right hand xml": r'super\(\)\.__init__\(xml_path_completion\("grippers/fourier_right_hand\.xml"\), idn=idn\)',
    },
    "models/grippers/gripper_model.py": {"grip site": r'"grip_site": "grip_site",'},
    "robots/robot.py": {
        "gripper at the eef body": r"self\.robot_model\.add_gripper\(self\.gripper\[arm\], self\.robot_model\.eef_name\[arm\]\)"},
    "models/robots/manipulators/manipulator_model.py": {"merge": r"self\.merge\(gripper, merge_body=arm_name\)"},
    "robots/mobile_robot.py": {
        "eef site": r'self\.eef_site_id\[arm\] = self\.sim\.model\.site_name2id\(self\.gripper\[arm\]\.important_sites\["grip_site"\]\)'},
    "robots/legged_robot.py": {"legged robot": r"class LeggedRobot\(MobileRobot\):"},
}
ROBOCASA = {  # relative to the robocasa-gr1-tabletop-tasks checkout
    "robocasa/models/robots/manipulators/gr1_robot.py": {
        "arms and waist": r"class GR1ArmsAndWaist\(GR1\):",
        "fixed base": r"self\._remove_free_joint\(\)",
        "fourier hands": r"class GR1ArmsAndWaistFourierHands\(GR1ArmsAndWaist\):",
    },
}
OPENWAM = {  # relative to the OpenWAM root
    "openwam/dataloader/utils/gr1_kinematics.py": {
        "pose site": r'"left": int\(self\.robot\.eef_site_id\["left"\]\),',
        "base body": r'self\.base_body_id = int\(self\.model\.body\(f"\{self\.robot\.robot_model\.naming_prefix\}base"\)\.id\)',
        "base frame": r"base_from_world = base_rot\.T",
    },
}
ROBOT_XML = "models/assets/robots/gr1/robot.xml"
HAND_XML = {arm: f"models/assets/grippers/fourier_{arm}_hand.xml" for arm in ARMS}


def _down(value: float) -> float:
    """Round outward to the millimetre (float noise aside)."""
    return math.floor(value * 1000 + 1e-6) / 1000


def _up(value: float) -> float:
    return math.ceil(value * 1000 - 1e-6) / 1000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned(path: Path) -> Path:
    """A file the derivation reads; its absence (no checkout, robosuite not installed) stops the freeze."""
    if not path.is_file():
        raise ValueError(f"workspace bounds come from the pinned sources, and {path} is missing")
    return path


def _find(text: str, pattern: str, what: str, path: Path) -> tuple[re.Match, int]:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"pinned {path.name} no longer has the {what} statement; review the bounds derivation")
    return match, text.count("\n", 0, match.start()) + 1


def _vector(element, attribute: str = "pos") -> np.ndarray:
    values = [float(value) for value in element.get(attribute, "0 0 0").split()]
    if len(values) != 3:
        raise ValueError(f"{element.tag} {element.get('name')!r} has no 3-vector {attribute}")
    return np.asarray(values)


def _rotation(element) -> np.ndarray:
    """A body's fixed orientation: its wxyz ``quat`` (MuJoCo normalizes it); other orientation forms are refused."""
    other = [key for key in ("axisangle", "euler", "xyaxes", "zaxis") if element.get(key) is not None]
    if other:
        raise ValueError(f"body {element.get('name')!r} sets its orientation with {other}; only quat is supported")
    quat = np.asarray([float(value) for value in element.get("quat", "1 0 0 0").split()])
    if quat.shape != (4,) or not np.linalg.norm(quat):
        raise ValueError(f"body {element.get('name')!r} has no wxyz quat")
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _path_to(root: ET.Element, tag: str, name: str) -> list[ET.Element]:
    """The bodies from ``root``'s worldbody down to the element ``tag`` named ``name`` (that element last)."""
    def walk(element, trail):
        for child in element:
            if child.tag == tag and child.get("name") == name:
                return trail + [child]
            if child.tag == "body":
                found = walk(child, trail + [child])
                if found:
                    return found
        return None

    worldbody = root.find("worldbody")
    found = walk(worldbody, []) if worldbody is not None else None
    if not found:
        raise ValueError(f"no {tag} named {name!r}")
    return found


def _joints(body: ET.Element) -> list[ET.Element]:
    return [child for child in body if child.tag in ("joint", "freejoint")]


def _check_defaults(root: ET.Element, path: Path) -> None:
    """Joints without a type are hinges at their body's origin unless a default class says otherwise."""
    for joint in root.iter("joint"):
        if joint.get("class") is not None:
            raise ValueError(f"{path.name}: joint {joint.get('name')!r} names a default class; review the derivation")
    for default in root.iter("default"):
        for joint in default.findall("joint"):
            if joint.get("type", "hinge") != "hinge" or np.any(_vector(joint) != 0):
                raise ValueError(f"{path.name}: a default joint is not a hinge at the body origin")
    if any(body.get("childclass") is not None for body in root.iter("body")):
        raise ValueError(f"{path.name}: a body sets a childclass; review the derivation")


def _hinge_at_origin(body: ET.Element, path: Path) -> None:
    for joint in _joints(body):
        if joint.tag == "freejoint" or joint.get("type", "hinge") != "hinge" or np.any(_vector(joint) != 0):
            raise ValueError(f"{path.name}: joint {joint.get('name')!r} on the chain is not a hinge at its body's origin")


def arm_reach(robot: ET.Element, hand: ET.Element, eef_body: str, robot_path: Path, hand_path: Path) -> dict:
    """The waist yaw joint's origin in the base frame and the all-links-straight reach from it to the grip site."""
    chain = _path_to(robot, "body", eef_body)
    names = [body.get("name") for body in chain]
    if BASE_BODY not in names:
        raise ValueError(f"{robot_path.name}: {eef_body} does not hang from the {BASE_BODY} body")
    chain = chain[names.index(BASE_BODY) + 1:]  # below the base: the base frame is the pose frame
    moving = [index for index, body in enumerate(chain) if _joints(body)]
    if not moving:
        raise ValueError(f"{robot_path.name}: no joint between {BASE_BODY} and {eef_body}")
    rotation, position = np.eye(3), np.zeros(3)
    for body in chain[:moving[0]]:  # fixed bodies place the first joint
        position = position + rotation @ _vector(body)
        rotation = rotation @ _rotation(body)
    first = chain[moving[0]]
    centre = position + rotation @ _vector(first)
    offsets = []
    for body in chain[moving[0]:]:
        _hinge_at_origin(body, robot_path)
        if body is not first:
            offsets.append((body.get("name"), float(np.linalg.norm(_vector(body)))))
    hand_chain = _path_to(hand, "site", "grip_site")
    for body in hand_chain[:-1]:
        if _joints(body):
            raise ValueError(f"{hand_path.name}: body {body.get('name')!r} above the grip site has a joint")
        offsets.append((f"{hand_path.stem}:{body.get('name')}", float(np.linalg.norm(_vector(body)))))
    offsets.append((f"{hand_path.stem}:grip_site", float(np.linalg.norm(_vector(hand_chain[-1])))))
    return {"centre": centre, "first_joint": _joints(first)[0].get("name"), "offsets": offsets,
            "reach": sum(length for _, length in offsets)}


def derive_bounds(robosuite_root, robocasa_root, openwam_root, *, slack: float = SLACK) -> dict:
    """Both arms' boxes and the files they come from."""
    roots = {"robosuite": Path(robosuite_root).resolve(), "robocasa": Path(robocasa_root).resolve(),
             "openwam": Path(openwam_root).resolve()}
    statements = {"robosuite": ROBOSUITE, "robocasa": ROBOCASA, "openwam": OPENWAM}
    files, lines = {}, {}
    for key, table in statements.items():
        for name, patterns in table.items():
            path = files[(key, name)] = _pinned(roots[key] / name)
            text = path.read_text(encoding="utf-8")
            for what, pattern in patterns.items():
                lines[what] = (*_find(text, pattern, what, path), path)
    eef = ast.literal_eval(lines["eef names"][0].group(1))
    robot_path = files[("robosuite", ROBOT_XML)] = _pinned(roots["robosuite"] / ROBOT_XML)
    robot = ET.parse(robot_path).getroot()
    _check_defaults(robot, robot_path)
    arms = {}
    for arm in ARMS:
        hand_path = files[("robosuite", HAND_XML[arm])] = _pinned(roots["robosuite"] / HAND_XML[arm])
        hand = ET.parse(hand_path).getroot()
        _check_defaults(hand, hand_path)
        data = arm_reach(robot, hand, eef[arm], robot_path, hand_path)
        reach, centre = data["reach"] + slack, data["centre"]
        box = [[_down(centre[axis] - reach), _up(centre[axis] + reach)] for axis in range(3)]
        arms[arm] = {**data, "eef_body": eef[arm], "box": box}

    def cite(key, name):
        return f"{name} sha256={_sha(files[(key, name)])}"

    def at(what):
        return f"{lines[what][2].name}:{lines[what][1]}"

    parts = [
        f"oeh_policy.robocasa_gr1_bounds v1: GR1 robot-base frame at each Fourier hand's grip_site, where OpenWAM's "
        f"gr1_kinematics reads and solves EEF33 poses ({cite('openwam', 'openwam/dataloader/utils/gr1_kinematics.py')}: "
        f"pose site {at('pose site')}, base body {at('base body')}, base frame {at('base frame')})",
        f"robosuite 1.5.1 ({cite('robosuite', '__init__.py')} {at('version')}): GR1 model "
        f"({cite('robosuite', 'models/robots/manipulators/gr1_robot.py')}: robot.xml {at('robot xml')}, Fourier hands "
        f"{at('default gripper')}, eef bodies {eef} {at('eef names')}); hands "
        f"({cite('robosuite', 'models/grippers/fourier_hands.py')} {at('left hand xml')}, {at('right hand xml')}) "
        f"merged into the eef bodies ({cite('robosuite', 'robots/robot.py')} {at('gripper at the eef body')}; "
        f"{cite('robosuite', 'models/robots/manipulators/manipulator_model.py')} {at('merge')}); grip site "
        f"({cite('robosuite', 'models/grippers/gripper_model.py')} {at('grip site')}; "
        f"{cite('robosuite', 'robots/mobile_robot.py')} {at('eef site')}; "
        f"{cite('robosuite', 'robots/legged_robot.py')} {at('legged robot')})",
        f"robocasa {cite('robocasa', 'robocasa/models/robots/manipulators/gr1_robot.py')}: "
        f"{at('fourier hands')} on {at('arms and waist')}, base fixed {at('fixed base')}",
        f"model files {cite('robosuite', ROBOT_XML)}, {cite('robosuite', HAND_XML['left'])}, "
        f"{cite('robosuite', HAND_XML['right'])}",
    ]
    for arm, data in arms.items():
        centre = ", ".join(f"{value:.4f}" for value in data["centre"])
        offsets = ", ".join(f"{name} {length:.6f}" for name, length in data["offsets"])
        parts.append(f"{arm}: centre {data['first_joint']} origin [{centre}], reach {data['reach']:.6f} m = body "
                     f"offsets to {data['eef_body']} and its grip site ({offsets}) + slack {slack}; box {data['box']}")
    parts.append("kinematic only (centre ± reach on every axis): no table or scene bound")
    return {"left_xyz_bounds": arms["left"]["box"], "right_xyz_bounds": arms["right"]["box"],
            "bounds_source": "; ".join(parts), "files": sorted(set(files.values())), "arms": arms}


def apply_pinned_bounds(cfg: dict) -> list[Path]:
    """Fill a template's ``review_workspace`` from the pinned files; typed values must equal the derivation.

    Returns the files the bounds come from, for the frozen identity. A configuration without a review workspace (a
    direct condition never freezes one) is left alone."""
    block = cfg.get("review_workspace")
    if block is None:
        return []
    if not isinstance(block, dict):
        raise ValueError("review_workspace must be an object")
    if block.get("frame") != FRAME or block.get("quaternion") != QUATERNION:
        raise ValueError(f"review_workspace must be in the {FRAME} frame with {QUATERNION} quaternions")
    native = cfg["robocasa_gr1"]
    derived = derive_bounds(native["robosuite_root"], native["root"], native["openwam_root"])
    for key in KEYS:
        value = block.get(key)
        if isinstance(value, str) and PLACEHOLDER in value:
            block[key] = derived[key]
        elif value != derived[key]:  # never typed by hand: a frozen value must be the derivation itself
            raise ValueError(f"review_workspace.{key} differs from the pinned derivation; leave it {PLACEHOLDER}")
    return derived["files"]
