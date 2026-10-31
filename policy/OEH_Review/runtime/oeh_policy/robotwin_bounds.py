"""Workspace bounds for the RoboTwin OEH conditions, derived from the pinned RoboTwin sources and assets.

A gross-error gate for slow-primitive rows and for the rows review mode builds itself (holds and the
teacher's corrections; OpenWAM's own rows stay unbounded). The bounds are in the frame and at the point of the EE16
actions they gate: RoboTwin's world frame, at the pose ``get_arm_pose`` reports and
``take_action(..., action_type="ee")`` targets, i.e. the embodiment's ``ee_joints`` origin (the
``move_group`` link) moved ``gripper_bias - 0.12`` along the gripper x axis (``envs/robot/robot.py``
``_trans_endpose`` and ``_trans_from_gripper_to_endlink``).

Per arm the centre is the arm base link (the child of the last fixed joint on the URDF chain from the
root to the ee link) placed by the embodiment's ``robot_pose``; the reach is the sum of the joint-origin
offsets from there to the ee link (plus any prismatic travel) plus ``|gripper_bias - 0.12|``: the
all-links-straight distance, which bounds every joint configuration (rotations keep lengths; the
triangle inequality adds them). x and y span centre ± reach, z runs from ``TABLE_MARGIN`` below the
lowest table top the task and task config can produce (a scene bound, as for RoboDojo) to centre z +
reach. Every number comes from the files; ``bounds_source`` names them with their sha256.
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
TABLE_MARGIN = 0.05  # metres below the lowest table top, as the RoboDojo bounds (plan decision log)
SLACK = 0.01  # metres: quaternion normalisation (robot_pose is 0.707-rounded) and URDF float rounding
KEYS = ("left_xyz_bounds", "right_xyz_bounds", "bounds_source")
# The config blocks whose bounds the freeze derives: the hybrid's slow primitives, review mode's workspace.
BOUNDED = ("primitives", "review_workspace")
MOVING = ("revolute", "continuous", "prismatic")

# Pinned statements the derivation relies on; a checkout that says otherwise is not this derivation's.
BASE_TASK = {
    "table_height_bias default": r"def _init_task_env_\(self, table_xy_bias=\[0, 0\], table_height_bias=(-?[0-9.]+), \*\*kwags\):",
    "random_table_height": r'self\.random_table_height = random_setting\.get\("random_table_height", 0\)',
    "table_z_bias": r"self\.table_z_bias = \(np\.random\.uniform\(low=-self\.random_table_height, high=0\) \+ table_height_bias\)",
    "table height": r"self\.create_table_and_wall\(table_xy_bias=table_xy_bias, table_height=([0-9.]+)\)",
    "table height bias applied": r"table_height \+= self\.table_z_bias",
    "arm pose": r"return self\.robot\.get_left_ee_pose\(\)",
}
CREATE_ACTOR = {"table top at the pose z": r"tabletop_pose = sapien\.Pose\(\[0\.0, 0\.0, -thickness / 2\]\)"}
ROBOT = {
    "left robot_pose": r'_entity_origion_pose = left_embodiment_args\.get\("robot_pose", .*\)\[0\]',
    "right robot_pose": r"_entity_origion_pose = _entity_origion_pose\[0 if len\(_entity_origion_pose\) == 1 else 1\]",
    "pose slicing": r"_entity_origion_pose = sapien\.Pose\(_entity_origion_pose\[:3\], _entity_origion_pose\[-4:\]\)",
    "ee joint": r"self\.left_ee = self\.left_entity\.find_joint_by_name\(self\.left_ee_name\)",
    "observed point": r"endpose_arr\[:3, 3\] = ee_pose\.p \+ endpose_arr\[:3, :3\] @ np\.array\(\[dis, 0, 0\]\)\.T",
    "observed offset": r"dis = gripper_bias\n\s+if is_endpose == False:\n\s+dis -= 0\.12",
    "action offset": r"gripper_pose_pos \+= gripper_pose_mat @ np\.array\(\[0\.12 - gripper_bias, 0, 0\]\)\.T",
}


def _down(value: float) -> float:
    """Round outward to the millimetre (float noise aside)."""
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


def _find(text: str, pattern: str, what: str, path: Path) -> tuple[re.Match, int]:
    matches = list(re.finditer(pattern, text))
    if not matches:
        raise ValueError(f"pinned {path.name} no longer has the {what} statement; review the bounds derivation")
    return matches[0], text.count("\n", 0, matches[0].start()) + 1


def _rotation(wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(np.asarray(wxyz, dtype=float))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _floats(element, attribute: str, default: str) -> list[float]:
    return [float(value) for value in (element.get(attribute, default) if element is not None else default).split()]


def _chain(urdf: ET.Element, ee_joint: str) -> list[ET.Element]:
    """Joints from the URDF root down to `ee_joint`, in order."""
    by_child = {joint.find("child").get("link"): joint for joint in urdf.findall("joint")}
    joint = next((item for item in urdf.findall("joint") if item.get("name") == ee_joint), None)
    if joint is None:
        raise ValueError(f"URDF has no ee joint {ee_joint}")
    chain = [joint]
    while chain[-1].find("parent").get("link") in by_child:
        chain.append(by_child[chain[-1].find("parent").get("link")])
        if len(chain) > len(by_child):
            raise ValueError("URDF joints form a cycle")
    return chain[::-1]


def arm_reach(urdf: ET.Element, ee_joint: str, root_rotation: np.ndarray, root_position: np.ndarray) -> dict:
    """Arm base (world position) and all-links-straight reach to the ee joint origin."""
    chain = _chain(urdf, ee_joint)
    moving = [index for index, joint in enumerate(chain) if joint.get("type") in MOVING]
    if not moving:
        raise ValueError(f"no moving joint on the chain to {ee_joint}")
    if any(joint.get("type") not in MOVING + ("fixed",) for joint in chain):
        raise ValueError(f"unsupported joint type on the chain to {ee_joint}")
    rotation, position = root_rotation, np.asarray(root_position, dtype=float)
    for joint in chain[:moving[0]]:  # fixed joints place the arm base
        origin = joint.find("origin")
        position = position + rotation @ np.asarray(_floats(origin, "xyz", "0 0 0"))
        rotation = rotation @ _rpy(*_floats(origin, "rpy", "0 0 0"))
    offsets = []
    for joint in chain[moving[0]:]:
        origin = joint.find("origin")
        length = float(np.linalg.norm(_floats(origin, "xyz", "0 0 0")))
        if joint.get("type") == "prismatic":
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"prismatic joint {joint.get('name')} has no limit")
            length += max(abs(float(limit.get("lower", 0))), abs(float(limit.get("upper", 0))))
        offsets.append((joint.get("name"), length))
    return {"base_link": chain[moving[0]].find("parent").get("link"), "base": position,
            "offsets": offsets, "reach": sum(length for _, length in offsets),
            "joints": [joint.get("name") for joint in chain]}


def _task_table_bias(path: Path, default: float) -> tuple[float, str]:
    text = path.read_text(encoding="utf-8")
    calls = re.findall(r"_init_task_env_\((.*?)\)", text, flags=re.S)
    if not calls:
        raise ValueError(f"{path.name} does not call _init_task_env_")
    values = set()
    for call in calls:
        match = re.search(r"table_height_bias\s*=\s*([^,)\s]+)", call)
        if match is None:
            values.add(default)
            continue
        try:
            values.add(float(match.group(1)))
        except ValueError:
            raise ValueError(f"{path.name} sets table_height_bias to a non-literal {match.group(1)!r}") from None
    if len(values) != 1:
        raise ValueError(f"{path.name} sets more than one table_height_bias")
    value = values.pop()
    return value, "default" if all("table_height_bias" not in call for call in calls) else "task"


def derive_bounds(root, task_name: str, task_config: str, *, table_margin: float = TABLE_MARGIN,
                  slack: float = SLACK) -> dict:
    """Bounds for one RoboTwin task and task config; also the files they come from."""
    root = Path(root).resolve()
    files = {name: root / name for name in (f"task_config/{task_config}.yml", "task_config/_embodiment_config.yml",
                                             "envs/_base_task.py", "envs/utils/create_actor.py",
                                             "envs/robot/robot.py", f"envs/{task_name}.py")}
    for path in files.values():
        _pinned(path)
    task_cfg = yaml.safe_load(files[f"task_config/{task_config}.yml"].read_text(encoding="utf-8"))
    embodiment = task_cfg.get("embodiment")
    if not isinstance(embodiment, list) or len(embodiment) != 1:
        raise ValueError("only a single dual-arm embodiment entity is supported (task config embodiment [name])")
    random_height = (task_cfg.get("domain_randomization") or {}).get("random_table_height", 0)
    if isinstance(random_height, bool) or not isinstance(random_height, (int, float)) or random_height < 0:
        raise ValueError("random_table_height must be a nonnegative number")
    registry = yaml.safe_load(files["task_config/_embodiment_config.yml"].read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not isinstance(registry.get(embodiment[0]), dict):
        raise ValueError(f"_embodiment_config.yml has no entry for embodiment {embodiment[0]!r}")
    folder = (root / registry[embodiment[0]]["file_path"]).resolve()
    files["embodiment config"] = _pinned(folder / "config.yml")
    robot = yaml.safe_load(files["embodiment config"].read_text(encoding="utf-8"))
    files["urdf"] = _pinned(folder / robot["urdf_path"])
    urdf = ET.parse(files["urdf"]).getroot()
    source = {name: path.read_text(encoding="utf-8") for name, path in files.items() if path.suffix == ".py"}
    lines = {}
    for text_key, table in (("envs/_base_task.py", BASE_TASK), ("envs/utils/create_actor.py", CREATE_ACTOR),
                            ("envs/robot/robot.py", ROBOT)):
        for what, pattern in table.items():
            lines[what] = _find(source[text_key], pattern, what, files[text_key])
    table_top = float(lines["table height"][0].group(1))
    default_bias = float(lines["table_height_bias default"][0].group(1))
    task_bias, bias_from = _task_table_bias(files[f"envs/{task_name}.py"], default_bias)
    lowest_top = table_top + task_bias - float(random_height)

    pose = robot.get("robot_pose")
    if not isinstance(pose, list) or len(pose) not in (1, 2) or any(len(item) != 7 for item in pose):
        raise ValueError("robot_pose must hold one or two [x, y, z, qw, qx, qy, qz] poses")
    bias = float(robot["gripper_bias"])
    point_offset = abs(bias - 0.12)
    joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
    arms = {}
    for index, arm in enumerate(("left", "right")):
        ee_joint, move_group = robot["ee_joints"][index], robot["move_group"][index]
        joint = joints.get(ee_joint)
        if joint is None or joint.find("child").get("link") != move_group:
            raise ValueError(f"{arm} ee joint {ee_joint} does not end at move_group link {move_group}")
        entry = pose[0 if len(pose) == 1 else index]
        data = arm_reach(urdf, ee_joint, _rotation(entry[3:7]), np.asarray(entry[:3], dtype=float))
        reach = data["reach"] + point_offset + slack
        base = data["base"]
        low_z = max(lowest_top - table_margin, base[2] - reach)
        box = [[_down(base[0] - reach), _up(base[0] + reach)], [_down(base[1] - reach), _up(base[1] + reach)],
               [_down(low_z), _up(base[2] + reach)]]
        arms[arm] = {**data, "box": box, "ee_joint": ee_joint, "move_group": move_group,
                     "kinematic_low_z": float(base[2] - data["reach"] - point_offset)}

    def cite(name):
        return f"{Path(files[name]).relative_to(root).as_posix()} sha256={_sha(files[name])}"

    def line(what, name):
        return f"{Path(files[name]).name}:{lines[what][1]}"

    parts = [
        "oeh_policy.robotwin_bounds v1: RoboTwin world frame at the EE16 action/get_arm_pose point "
        f"(ee joint origin + (gripper_bias - 0.12) along the gripper x; {cite('envs/robot/robot.py')} "
        f"lines {lines['observed offset'][1]}-{lines['observed point'][1]}, {lines['action offset'][1]}; "
        f"get_arm_pose {line('arm pose', 'envs/_base_task.py')})",
        f"embodiment {embodiment[0]} ({cite(f'task_config/{task_config}.yml')}; "
        f"{cite('task_config/_embodiment_config.yml')}; {cite('embodiment config')}: robot_pose {pose}, "
        f"gripper_bias {bias}, ee_joints {robot['ee_joints']}, move_group {robot['move_group']}; {cite('urdf')})",
    ]
    for arm, data in arms.items():
        offsets = ", ".join(f"{name} {length:.4f}" for name, length in data["offsets"])
        parts.append(
            f"{arm}: base {data['base_link']} at [{data['base'][0]:.4f}, {data['base'][1]:.4f}, {data['base'][2]:.4f}], "
            f"reach {data['reach']:.4f} m = joint-origin offsets ({offsets}) + |gripper_bias - 0.12| "
            f"{point_offset:.4f} + slack {slack}; box {data['box']}")
    parts.append(
        f"z low = lowest table top {lowest_top:.4f} (table_height {table_top} at {line('table height', 'envs/_base_task.py')} "
        f"+ table_height_bias {task_bias} from the {bias_from} ({cite(f'envs/{task_name}.py')}; default at "
        f"{line('table_height_bias default', 'envs/_base_task.py')}) - random_table_height {random_height} "
        f"({line('table_z_bias', 'envs/_base_task.py')}); top surface at the table pose z, "
        f"{cite('envs/utils/create_actor.py')} {line('table top at the pose z', 'envs/utils/create_actor.py')}; "
        f"{cite('envs/_base_task.py')}) - margin {table_margin} (a scene bound: the all-links-straight sphere, "
        f"which knows no table, would allow z down to {min(data['kinematic_low_z'] for data in arms.values()):.3f}); "
        f"z high = base z + reach")
    return {"left_xyz_bounds": arms["left"]["box"], "right_xyz_bounds": arms["right"]["box"],
            "bounds_source": "; ".join(parts), "files": [Path(path) for path in files.values()],
            "arms": arms}


def apply_pinned_bounds(cfg: dict) -> list[Path]:
    """Fill a template's workspace bounds from the pinned files; typed values must equal the derivation.

    The blocks that hold them (``BOUNDED``): the hybrid's slow `primitives` and review mode's `review_workspace`
    (oeh_policy.robotwin_review), the same box for the same task. Returns the files the bounds come from, for the
    frozen identity. A config with neither (the fast condition) is left alone.
    """
    blocks = [name for name in BOUNDED if cfg.get(name) is not None]
    if not blocks:
        return []
    for name in blocks:
        if not isinstance(cfg[name], dict):
            raise ValueError(f"{name} must be an object")
    native = cfg["robotwin"]
    derived = derive_bounds(native["root"], native["task_name"], native["task_config"])
    for name in blocks:
        block = cfg[name]
        for key in KEYS:
            value = block.get(key)
            if isinstance(value, str) and PLACEHOLDER in value:
                block[key] = derived[key]
            elif value != derived[key]:  # never typed by hand: a frozen value must be the derivation itself
                raise ValueError(f"{name}.{key} differs from the pinned derivation; leave it {PLACEHOLDER}")
    return derived["files"]
