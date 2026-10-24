"""Policy-side cuRobo planner for the two X5 arms.

The migration Worker plans with a rewrite of the benchmark's cuRobo wrapper (env/planner_manager/curobo_planner.py),
and the leaderboard forbids benchmark patches. PolicyPlanner is that rewrite moved into the policy and renamed. It
does not inherit from the official CuroboPlanner, has no batch planner and no pose planner (plan_path: the worker
never calls it). Kept from the rewrite, with the same maths:

  * validate_joint_trajectory: preflight of the measured start plus the final commanded samples: joint bounds and
    complete self collision of the cuRobo model, and the table slab with the per-sphere non-deepening escape rule
    (migration/fixture_clearance.py). Task objects, the other arm and swept volumes are not checked.
  * forward_ee_pose_base: link6 pose in the arm base frame from the IK robot model.
  * solve_ik_to_joint: 32 IK seeds, all candidates returned; the valid candidate nearest the current joints wins
    (migration.kinematics.nearest_solution, equivalent angles within the joint limits).
  * plan_joint: cuRobo cspace planning (4 trajopt seeds), result interpolated at 0.04 s, the official 25 Hz control
    period (env_cfg/arx_x5.yml observation.collect_freq).
  * The table: one axis-aligned slab named "table", TABLE_SLAB_XY wide and centred under the arm base as in the
    rewrite, with centre height and thickness from the public scene config. env_cfg/scene/default.yml has Table
    default_pos z = 0.74 (centre) and scale z = 0.05 (thickness), so the top is at 0.765 m, the plane the arm bases
    stand on (the benchmark's own LayoutManager.select_table computes the same top). The official wrapper places the
    top at 0.74 m, 25 mm too low.
  * The fixed-root cost manager (migration/fixture_solver.py): the world cost of IK and trajectory optimisation
    ignores the base_link spheres, which rest on the table; self collision and joint bounds keep the full model.
  * JointLimitModel: every output stays within the joint limits the simulator enforces
    (SIM_JOINT_LIMITS_DEG, read offline from the public ARX.usd the evaluator spawns; they equal X5A.urdf's) and
    within the wrist table WRIST_TABLE (joint4 against joint5: link5 kept 5 mm clear of link3, a self-collision that
    PhysX enforces and cuRobo's spheres miss; derived from public meshes). solve_ik_to_joint chooses only among
    allowed candidates (nearest_solution's choice is unchanged when it was allowed) and fails with a reason when
    there is none; plan_joint refuses a plan that leaves them; validate_joint_trajectory reports `limit_violation`.
    A measured start a hair outside (tracking overshoot) is tolerated as long as the samples go no further out.

Inputs come only from public files of the benchmark this process runs against (env.global_configs.ROOT_DIR):
    Assets/Robots/<robot>/curobo.yml            the robot's public cuRobo model
    Assets/Robots/<robot>/robot_config.yml      arm joint names
    env_cfg/robot/dual_x5.yml                   arm base poses (only for planners built without a live robot)
    task/RoboDojo/config/_task.yml              scene config of a task (default or conveyor)
    env_cfg/scene/<scene>.yml                   Table centre height and thickness
    env/robot_manager/robot_config/<robot>.py   fix_root_link of the articulation, read as text (importing it
                                                needs Isaac Sim)
    Assets/Robots/x5/ARX.usd                    only its sha256, against the one the joint-limit constants were
                                                read from (SIM_JOINT_LIMITS_SOURCE)
Imports from the benchmark package are read-only: env.global_configs (paths) and utils.transformer
(calculate_target_pose, the same helper the official wrapper uses). Nothing here imports Isaac.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import transforms3d as t3d
import yaml

import curobo.runtime as _curobo_runtime
from curobo._src.geom.types import SceneCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

from env.global_configs import BENCHMARK, ROOT_DIR  # read only: where this benchmark's public files are
from utils.transformer import calculate_target_pose  # read only: public helper, same maths as the official planner

from migration.fixture_clearance import first_fixture_violation, sphere_box_clearance
from migration.fixture_solver import (
    FixedRootCostManagerCfg, FixedX5Table, assert_solver_installation, bind_single_planner_config,
    bind_solver_config, preflight_components,
)
from migration.kinematics import nearest_solution

# As in the benchmark's wrapper: let one solver switch between its CUDA graphs (IK, cspace planning).
_curobo_runtime.cuda_graph_reset = True

CONTROL_DT = 0.04            # s, official control period (25 Hz)
IK_SEEDS = 32
TRAJOPT_SEEDS = 4
TABLE_SLAB_XY = (3.0, 3.0)   # m, footprint of the planner's table slab, centred under the arm base (as the rewrite)
IDENTITY_POSE = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

# ----------------------------------------------------------------------------- joint limits
# The joint limits the simulator enforces: physics:lowerLimit / physics:upperLimit (degrees) of /X5A/joints/joint1..6
# in the public robot USD the evaluator spawns (env/robot_manager/robot_config/x5.py: UsdFileCfg(usd_path=
# f"{ROBOTS_PATH}/x5/ARX.usd"), no joint-limit override; ARX.usd composes no other layer). Read offline with the USD
# libraries of .venv-sim, never from a running simulator. They equal X5A.urdf's (+-10 rad, joint6 +-3.14 rad), which cuRobo reads;
# configuration/ARX_physics.usd has the arm's tighter design limits, but it is not part of ARX.usd's composition and
# the recorded joint states go beyond them (observed_stops.py), so the simulator does not use them.
SIM_JOINT_LIMITS_DEG = {
    "joint1": (-572.957763671875, 572.957763671875),
    "joint2": (-572.957763671875, 572.957763671875),
    "joint3": (-572.957763671875, 572.957763671875),
    "joint4": (-572.957763671875, 572.957763671875),
    "joint5": (-572.957763671875, 572.957763671875),
    "joint6": (-179.9087371826172, 179.9087371826172),
}
SIM_JOINT_LIMITS_SOURCE = {
    "file": "Assets/Robots/x5/ARX.usd",
    "sha256": "5723a835bf5627a935a8dabb3a44447725bd1465243ce791e221b3e990121d63",
    "prims": "/X5A/joints/joint1 .. /X5A/joints/joint6",
    "attributes": "physics:lowerLimit, physics:upperLimit (degrees)",
    "spawned_by": "env/robot_manager/robot_config/x5.py "
                  "(sha256 13d35630f833d1b0a1e3df0bea05f843e88aae1248b59af8aadfefa9cad29bbb)",
    "derivation": "read offline from ARX.usd with the USD libraries of .venv-sim",
}
# What stopped some simulator motions about 11 mm short is not a joint limit but self-collision. The articulation is
# spawned with enabled_self_collisions=True and PhysX filters only parent-child pairs, so link5 (collider: convex hull)
# meets link3 (collider: convex decomposition) when the wrist pitches far: with joint5 near 0, joint4 stops at about
# -2.22 rad whatever joints 1-3 do. cuRobo's spheres (self_collision_buffer -0.01) miss this contact. The wrist table
# (the joint4 interval as a function of joint5 in which link5's hull keeps 5 mm from link3's mesh) is derived from the
# public URDF, meshes and USD collider approximations only (offline); the planner reads the extract WRIST_TABLE, checked against WRIST_TABLE_SHA256. The recorded
# stops (observed_stops.py) validate it; they did not derive it.
WRIST_TABLE = Path(__file__).resolve().with_name("policy_planner_wrist.json")
WRIST_TABLE_SHA256 = "92626fd66599c6ed329cf86d28ae368eeb4a596796b1f291641b3cb042bb6225"
LIMIT_TOLERANCE = 1e-9       # rad


def sim_joint_limits_rad(names):
    """(lower, upper) of the simulator's joint limits for the joints `names`, in rad."""
    lower = np.array([math.radians(SIM_JOINT_LIMITS_DEG[n][0]) for n in names])
    upper = np.array([math.radians(SIM_JOINT_LIMITS_DEG[n][1]) for n in names])
    return lower, upper


class WristTable:
    """The joint4 interval as a function of joint5 in which link5 stays clear of link3 (see WRIST_TABLE)."""

    def __init__(self, joint5, lower, upper, provenance=None):
        self.joint5 = np.asarray(joint5, float)
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        if (self.joint5.ndim != 1 or len(self.joint5) < 2 or self.lower.shape != self.joint5.shape
                or self.upper.shape != self.joint5.shape or np.any(np.diff(self.joint5) <= 0)
                or np.any(self.lower >= self.upper) or not np.all(np.isfinite(self.lower + self.upper))):
            raise ValueError("wrist table: increasing joint5 rows, each with a finite lower < upper")
        self.provenance = dict(provenance or {})

    @classmethod
    def load(cls, path=None, sha256=None):
        path = Path(path) if path is not None else WRIST_TABLE
        data = path.read_bytes()
        expected = WRIST_TABLE_SHA256 if sha256 is None else sha256
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"{path} has sha256 {digest}, expected {expected}: the wrist table changed")
        table = json.loads(data)
        rows = table["rows"]
        return cls([r["joint5"] for r in rows], [r["lower"] for r in rows], [r["upper"] for r in rows],
                   {k: v for k, v in table.items() if k != "rows"})

    def margin(self, joint4, joint5):
        """Signed distance (rad) inside the allowed wrist region, >= 0 allowed; vectorised. Between two table rows the
        tighter of their bounds applies (conservative); outside the table's joint5 range nothing is allowed."""
        q4, q5 = np.asarray(joint4, float), np.asarray(joint5, float)
        i0, i1 = self._rows(q5)
        lower = np.maximum(self.lower[i0], self.lower[i1])
        upper = np.minimum(self.upper[i0], self.upper[i1])
        inside = np.minimum(q4 - lower, upper - q4)
        return np.minimum(inside, np.minimum(q5 - self.joint5[0], self.joint5[-1] - q5))

    def info(self):
        return {"file": str(WRIST_TABLE.name), "sha256": WRIST_TABLE_SHA256, "rows": len(self.joint5),
                "joint5_range_rad": [float(self.joint5[0]), float(self.joint5[-1])],
                "joint4_at_joint5_0_rad": [float(v) for v in self.bounds(0.0)], **self.provenance}

    def _rows(self, joint5):
        """The two table rows around joint5 (the same row twice when joint5 is on a row)."""
        last = len(self.joint5) - 1
        i0 = np.clip(np.searchsorted(self.joint5, joint5, side="right") - 1, 0, last)
        i1 = np.clip(np.searchsorted(self.joint5, joint5, side="left"), 0, last)
        return i0, i1

    def bounds(self, joint5):
        """joint4 bounds that apply at joint5 (inside the table's joint5 range)."""
        i0, i1 = (int(i) for i in self._rows(joint5))
        return max(self.lower[i0], self.lower[i1]), min(self.upper[i0], self.upper[i1])


class JointLimitModel:
    """What PolicyPlanner keeps its IK, plan and preflight outputs within: the simulator's joint limits (a box) and the
    wrist table (joint4 against joint5). margins() > 0 inside, < 0 outside (rad)."""

    def __init__(self, active_joints, root=None, wrist=None):
        usd = robodojo_root(root) / SIM_JOINT_LIMITS_SOURCE["file"]
        digest = _sha256(usd)
        if digest != SIM_JOINT_LIMITS_SOURCE["sha256"]:
            raise ValueError(f"{usd} has sha256 {digest}, not the one the joint-limit constants were read from "
                             f"({SIM_JOINT_LIMITS_SOURCE['sha256']}): re-derive them")
        self.names = list(active_joints)
        self.lower, self.upper = sim_joint_limits_rad(self.names)
        self.wrist = wrist if wrist is not None else WristTable.load()
        self.i4, self.i5 = self.names.index("joint4"), self.names.index("joint5")

    def margins(self, q):
        q = np.atleast_2d(np.asarray(q, float))
        box = np.minimum(q - self.lower, self.upper - q).min(axis=1)
        return np.minimum(box, self.wrist.margin(q[:, self.i4], q[:, self.i5]))

    def check_path(self, start, points):
        """The first sample of `points` outside the limits and deeper than the measured `start` is, or None.

        A measured start may lie a hair outside (tracking overshoot at a bound); motions from it are allowed as long
        as they never go further out than it, so the arm can always move back in."""
        start_margin = float(self.margins(start)[0])
        allowed = min(0.0, start_margin) - LIMIT_TOLERANCE
        points = np.atleast_2d(np.asarray(points, float))
        margins = self.margins(points)
        bad = np.flatnonzero(margins < allowed)
        if len(bad) == 0:
            return None
        i = int(bad[0])
        q = points[i]
        box = bool(np.any(q < self.lower - LIMIT_TOLERANCE) or np.any(q > self.upper + LIMIT_TOLERANCE))
        return {"sample_index": i, "joints_rad": q.tolist(), "margin_rad": float(margins[i]),
                "start_margin_rad": start_margin,
                "checks": "simulator joint limits (ARX.usd)" if box else
                          "wrist table (link5 clear of link3, joint4 against joint5)",
                "wrist_joint4_bounds_rad": [float(v) for v in self.wrist.bounds(q[self.i5])]}

    def info(self):
        return {"sim_joint_limits_rad": {n: [float(lo), float(hi)] for n, lo, hi in
                                         zip(self.names, self.lower, self.upper)},
                "sim_joint_limits_source": SIM_JOINT_LIMITS_SOURCE, "wrist_table": self.wrist.info(),
                "tolerance_rad": LIMIT_TOLERANCE}


def shifted_candidates(positions, success, joint_names, active_names, current, lower, upper):
    """The candidates migration.kinematics.nearest_solution chooses from: each IK candidate's active joints shifted by
    whole turns towards `current` within [lower, upper], and which of them are valid (same arithmetic)."""
    values = np.asarray(positions, dtype=float)
    candidates = values.reshape(-1, values.shape[-1])
    columns = [joint_names.index(name) for name in active_names]
    active = candidates[:, columns]
    valid = np.asarray(success, dtype=bool).reshape(-1) & np.isfinite(active).all(axis=1)
    tau = 2 * np.pi
    safe = np.where(np.isfinite(active), active, 0)
    minimum_turn = np.ceil((np.asarray(lower) - safe) / tau)
    maximum_turn = np.floor((np.asarray(upper) - safe) / tau)
    valid &= (minimum_turn <= maximum_turn).all(axis=1)
    turns = np.clip(np.rint((np.asarray(current) - safe) / tau), minimum_turn, maximum_turn)
    return safe + tau * turns, valid


# ----------------------------------------------------------------------------- public configuration

def robodojo_root(root=None):
    """Root of the benchmark checkout: `root`, else the one whose `env` package this process imported."""
    return Path(os.path.normpath(root if root is not None else ROOT_DIR))


def public_curobo_yml(robot_name="x5", root=None):
    path = robodojo_root(root) / "Assets" / "Robots" / robot_name / "curobo.yml"
    if not path.is_file():
        raise FileNotFoundError(f"public cuRobo config not found: {path}")
    return path


def public_arm_joints(robot_name="x5", root=None):
    config = yaml.safe_load((robodojo_root(root) / "Assets" / "Robots" / robot_name / "robot_config.yml").read_text())
    return [str(name) for name in config["arm_joints_name"]]


def public_arm_bases(robot_config="dual_x5", root=None):
    """[x, y, z, qw, qx, qy, qz] of the target arms in config order (RobotManager: first = left_arm)."""
    config = yaml.safe_load((robodojo_root(root) / "env_cfg" / "robot" / f"{robot_config}.yml").read_text())
    return [list(map(float, r["default_root_pos"])) + list(map(float, r["default_root_rot"]))
            for r in config["robots"] if r.get("robot_type") == "arm" and r.get("type", "target") == "target"]


def public_fix_root_link(robot_name="x5", root=None):
    """`fix_root_link` of the robot's articulation config, read from the public source file without importing it."""
    path = robodojo_root(root) / "env" / "robot_manager" / "robot_config" / f"{robot_name}.py"
    values = [keyword.value for node in ast.walk(ast.parse(path.read_text()))
              if isinstance(node, ast.Call) for keyword in node.keywords if keyword.arg == "fix_root_link"]
    if len(values) != 1 or not isinstance(values[0], ast.Constant) or not isinstance(values[0].value, bool):
        raise ValueError(f"expected exactly one literal fix_root_link in {path}")
    return values[0].value


def task_scene(task=None, root=None):
    """Scene config name of a task, resolved as utils.pipeline_utils.process_config does (default: 'default')."""
    info = yaml.safe_load((robodojo_root(root) / "task" / BENCHMARK / "config" / "_task.yml").read_text())
    task_info = ((info.get("tasks") or {}).get(task) or {}) if task else {}
    return task_info.get("scene_config", (info.get("common") or {}).get("scene_config", "default"))


@dataclass(frozen=True)
class TableSlab:
    """The planner's table: an axis-aligned slab whose top is the supporting table surface."""
    center_z_world: float
    thickness: float
    root_z_world: float
    scene: str
    source: str

    @property
    def top_z_world(self):
        return self.center_z_world + self.thickness / 2

    @property
    def center_z_base(self):
        # Arm bases are yaw-only (checked in public_table), so base z = world z - base height.
        return self.center_z_world - self.root_z_world

    def cuboid(self):
        return {"dims": [float(TABLE_SLAB_XY[0]), float(TABLE_SLAB_XY[1]), float(self.thickness)],
                "pose": [0.0, 0.0, float(self.center_z_base), 1.0, 0.0, 0.0, 0.0]}

    def as_dict(self):
        return {"center_z_world": self.center_z_world, "thickness": self.thickness, "top_z_world": self.top_z_world,
                "root_z_world": self.root_z_world, "center_z_base": self.center_z_base, "scene": self.scene,
                "source": self.source, "cuboid_base": self.cuboid()}


def public_table(root_pose, task=None, root=None):
    """Table slab of a task's scene for an arm based at `root_pose` = [x, y, z, qw, qx, qy, qz] (world)."""
    root_pose = [float(v) for v in root_pose]
    if len(root_pose) != 7 or abs(root_pose[4]) > 1e-9 or abs(root_pose[5]) > 1e-9:
        raise ValueError("the planner's table slab needs a yaw-only arm base pose [x, y, z, qw, 0, 0, qz]")
    scene = task_scene(task, root)
    scenes = robodojo_root(root) / "env_cfg" / "scene"
    table = (yaml.safe_load((scenes / f"{scene}.yml").read_text()) or {}).get("Table")
    if table is not None:
        center, thickness = float(table["default_pos"][2]), float(table["scale"][2])
        source = f"env_cfg/scene/{scene}.yml Table: default_pos[2] = centre, scale[2] = thickness"
    else:
        # No table in the scene (conveyor tasks): keep the slab top at the plane the arm base stands on, which is
        # what the rewrite did for every scene, with the default table's thickness.
        thickness = float(yaml.safe_load((scenes / "default.yml").read_text())["Table"]["scale"][2])
        center = root_pose[2] - thickness / 2
        source = f"env_cfg/scene/{scene}.yml has no Table: slab top at the arm mount plane (base z)"
    return TableSlab(center, thickness, root_pose[2], scene, source)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- planner

class PolicyPlanner:
    """cuRobo IK, cspace planning, FK and preflight for one X5 arm type, in the arm's base frame."""

    @classmethod
    def from_public_config(cls, robot_name="x5", root_pose=None, task=None, *, active_joints_name=None,
                           root=None, interpolation_dt=CONTROL_DT):
        """Build from the public files; `root_pose` defaults to the first target arm of env_cfg/robot/dual_x5.yml."""
        if root_pose is None:
            root_pose = public_arm_bases(root=root)[0]
        joints = list(active_joints_name) if active_joints_name is not None else public_arm_joints(robot_name, root)
        return cls(joints, public_curobo_yml(robot_name, root), public_table(root_pose, task, root),
                   fix_root_link=public_fix_root_link(robot_name, root), interpolation_dt=interpolation_dt,
                   limits=JointLimitModel(joints, root))

    def __init__(self, active_joints_name, yml_path, table, *, fix_root_link, all_joints=None,
                 robot_origin_pose=IDENTITY_POSE, interpolation_dt=CONTROL_DT, limits=None):
        started = time.perf_counter()
        self.timings = {}
        self.yml_path = str(yml_path)
        self.table = table
        # outputs stay within the simulator's joint limits and the wrist table (JointLimitModel)
        self.limits = limits if limits is not None else JointLimitModel(active_joints_name)
        self.fix_root_link = fix_root_link
        self.interpolation_dt = float(interpolation_dt)
        # The planner works in the arm's base frame; targets arrive in world with the arm's base pose.
        self.robot_origin_pose = list(robot_origin_pose)
        self.active_joints_name = list(active_joints_name)
        self.all_joints = list(all_joints) if all_joints is not None else list(self.active_joints_name)
        with open(self.yml_path) as f:
            yml_data = yaml.safe_load(f)
        self.frame_bias = yml_data["planner"]["frame_bias"]

        self.robot_cfg, self.scene_model = self._build_robot_and_scene_cfg(yml_data, table)
        cspace_cfg = self.robot_cfg["robot_cfg"]["kinematics"].get("cspace", {})
        self.cspace_joint_names = list(cspace_cfg.get("joint_names", list(self.active_joints_name)))
        retract_config = cspace_cfg.get("default_joint_position", None)
        if retract_config is None:
            self.cspace_retract_config = np.zeros(len(self.cspace_joint_names), dtype=np.float32)
        else:
            self.cspace_retract_config = np.asarray(retract_config, dtype=np.float32).reshape(-1)
        if self.cspace_retract_config.shape[0] != len(self.cspace_joint_names):
            raise ValueError(
                "curobo cspace config mismatch: "
                f"{self.cspace_retract_config.shape[0]} retract values for "
                f"{len(self.cspace_joint_names)} joints in {self.yml_path}"
            )
        self._active_to_cspace_idx = []
        for joint_name in self.active_joints_name:
            if joint_name not in self.cspace_joint_names:
                raise ValueError(
                    f"Active joint {joint_name} not found in curobo cspace joints {self.cspace_joint_names}"
                )
            self._active_to_cspace_idx.append(self.cspace_joint_names.index(joint_name))

        self.device_cfg = DeviceCfg()
        self.use_cuda_graph = True
        self.installation = self._build_installation()
        self._mark("config_s", started)

        phase = time.perf_counter()
        self.motion_planner = MotionPlanner(self._build_motion_planner_cfg())
        assert_solver_installation(self.motion_planner.ik_solver, self.installation)
        assert_solver_installation(self.motion_planner.trajopt_solver, self.installation)
        self._mark("motion_planner_s", phase)

        planner_joint_names = list(self.motion_planner.joint_names)
        if planner_joint_names != self.cspace_joint_names:
            new_retract = np.zeros(len(planner_joint_names), dtype=np.float32)
            for i, name in enumerate(planner_joint_names):
                if name in self.cspace_joint_names:
                    new_retract[i] = self.cspace_retract_config[self.cspace_joint_names.index(name)]
            self.cspace_joint_names = planner_joint_names
            self.cspace_retract_config = new_retract
            self._active_to_cspace_idx = [planner_joint_names.index(n) for n in self.active_joints_name]

        self.tool_frames = list(self.motion_planner.tool_frames)
        if not self.tool_frames:
            raise ValueError("No tool_frames found in curobo robot configuration.")

        phase = time.perf_counter()
        self.ee_link = self.tool_frames[0]
        self.ik_solver = InverseKinematics(self._build_ik_cfg())
        assert_solver_installation(self.ik_solver, self.installation)
        self._mark("ik_solver_s", phase)

        # The same immutable geometry drives optimizer, IK and final samples.
        phase = time.perf_counter()
        self.fixture_center = np.asarray(self.installation.center)
        self.fixture_dimensions = np.asarray(self.installation.dimensions)
        scene_checker = self.motion_planner.scene_collision_checker
        if scene_checker is None:
            raise ValueError('Trajectory validation requires the original scene checker')
        self.trajectory_checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=self.robot_cfg, device_cfg=self.device_cfg,
            scene_collision_checker=scene_checker, collision_activation_distance=0.0))
        params = self.trajectory_checker.kinematics.config.kinematics_config
        self.installation.check_scene(scene_checker.scene_model)
        self._world_keep = self.installation.keep_indices(params)
        sphere_links = params.link_sphere_idx_map.detach().cpu().numpy().reshape(-1)
        link_names = {idx: name for name, idx in params.link_name_to_idx_map.items()}
        self._sphere_link_names = [link_names[int(idx)] for idx in sphere_links]
        self._moving_spheres = np.zeros(len(sphere_links), dtype=bool)
        self._moving_spheres[self._world_keep.cpu().numpy()] = True
        self._mark("trajectory_checker_s", phase)

        # Same warm-up calls in the same order as the rewrite: the solvers' seed sequences stay aligned with it.
        phase = time.perf_counter()
        self.motion_planner.warmup(enable_graph=False, num_warmup_iterations=5)
        self._mark("warmup_s", phase)
        phase = time.perf_counter()
        if self.use_cuda_graph:
            self._prewarm_alternate_modes()
        self._mark("prewarm_alternate_modes_s", phase)
        self.timings["total_s"] = time.perf_counter() - started

    def _mark(self, name, since):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings[name] = time.perf_counter() - since

    def close(self):
        """Release the cuRobo solvers (also done on garbage collection)."""
        for name in ("ik_solver", "motion_planner"):
            solver = self.__dict__.pop(name, None)
            if solver is not None:
                try:
                    solver.destroy()
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def info(self):
        """Configuration summary for session provenance."""
        return {"class": f"{type(self).__module__}.{type(self).__qualname__}", "yml_path": self.yml_path,
                "yml_sha256": _sha256(self.yml_path), "table": self.table.as_dict(),
                "scene_model": self.scene_model, "fix_root_link": self.fix_root_link,
                "interpolation_dt_s": self.interpolation_dt, "ik_seeds": IK_SEEDS, "trajopt_seeds": TRAJOPT_SEEDS,
                "active_joints": self.active_joints_name, "frame_bias": list(self.frame_bias),
                "joint_limits": self.limits.info(), "timings_s": dict(self.timings)}

    # ------------------------------------------------------------------ joint states
    def _build_cspace_joint_values(self, active_joint_values):
        active_joint_values = self._extract_active_joint_values(active_joint_values)
        full_joint_values = self.cspace_retract_config.copy()
        for src_idx, dst_idx in enumerate(self._active_to_cspace_idx):
            full_joint_values[dst_idx] = active_joint_values[src_idx]
        return full_joint_values

    def _extract_active_joint_values(self, joint_values):
        joint_array = np.asarray(joint_values, dtype=np.float32).reshape(-1)
        if joint_array.shape[0] == len(self.active_joints_name):
            return joint_array
        if joint_array.shape[0] == len(self.all_joints):
            indices = [self.all_joints.index(n) for n in self.active_joints_name]
            return joint_array[indices]
        raise ValueError(
            f"Joint state size {joint_array.shape[0]} does not match active joints "
            f"{len(self.active_joints_name)} or all joints {len(self.all_joints)} for planner."
        )

    def _build_joint_state(self, joint_pos):
        cspace_pos = self._build_cspace_joint_values(joint_pos)
        device = self.device_cfg.device
        joint_pos_tensor = torch.as_tensor(cspace_pos, dtype=torch.float32, device=device).reshape(1, -1)
        return JointState.from_position(joint_pos_tensor, joint_names=self.cspace_joint_names)

    # ------------------------------------------------------------------ public API used by the worker
    def validate_joint_trajectory(self, start, points):
        """Preflight measured start and final commanded prefix, with named joints.

        Full native bounds/self plus the solver's installed table geometry.
        No task objects, inter-arm or swept-mesh claim. Initial table overlap
        retains the per-sphere nondeepening escape rule at this boundary.
        """
        start = np.asarray(start, dtype=float)
        points = np.asarray(points, dtype=float)
        if (start.shape != (len(self.active_joints_name),) or points.ndim != 2
                or points.shape[1:] != start.shape or len(points) == 0
                or not np.isfinite(start).all() or not np.isfinite(points).all()):
            raise ValueError('Expected finite active-joint start and nonempty trajectory')
        full = np.stack([self._build_cspace_joint_values(q) for q in np.vstack([start, points])])
        if not np.isfinite(full).all():
            raise ValueError('Joint trajectory exceeds model numeric range')
        names = list(self.trajectory_checker.kinematics.joint_names)
        if any(name not in self.cspace_joint_names for name in names):
            raise ValueError('Trajectory checker requires unprovided joints')
        js = JointState.from_position(
            torch.as_tensor(full, dtype=torch.float32, device=self.device_cfg.device),
            joint_names=self.cspace_joint_names).reorder(names)
        q = js.position.reshape(1, len(full), len(names))
        with torch.no_grad():
            self.installation.check_scene(self.trajectory_checker.scene_model.scene_model)
            valid, world_valid, spheres = preflight_components(
                self.trajectory_checker, q, self._world_keep)
            valid = valid.detach().cpu().numpy()
            world_valid = world_valid.detach().cpu().numpy()
            spheres = spheres.detach().cpu().numpy()
        if valid.shape != (1, len(full)) or valid.dtype != np.dtype(bool):
            raise ValueError('Invalid trajectory checker feasibility mask')
        if spheres.shape != (1, len(full), len(self._sphere_link_names), 4):
            raise ValueError('Collision sphere/link map mismatch')
        spheres = spheres[0]
        bad = first_fixture_violation(spheres[1:], spheres[0], self._moving_spheres,
                                      self.fixture_center, self.fixture_dimensions)
        if bad is not None:
            bad['link'] = self._sphere_link_names[bad['sphere_index']]
            bad['joints_rad'] = points[bad['sample_index']].tolist()
        invalid = np.flatnonzero(~valid[0])
        legacy_bad = None if len(invalid) == 0 else {
            'sample_index': int(invalid[0]) - 1,  # -1 denotes measured start
            'joints_rad': np.vstack([start, points])[int(invalid[0])].tolist(),
            'checks': 'joint bounds / original complete self collision'}
        gaps = sphere_box_clearance(spheres, self.fixture_center, self.fixture_dimensions)
        # the commanded samples stay within the simulator's joint limits and the wrist table (never deeper
        # outside than the measured start)
        limit_bad = self.limits.check_path(start, points)
        return {'ok': legacy_bad is None and bad is None and limit_bad is None,
                'scope': 'sampled robot model and installed static fixture only',
                'checked_samples': len(points), 'start_checked': True,
                'legacy_violation': legacy_bad, 'fixture_violation': bad, 'limit_violation': limit_bad,
                'strict_world_valid': world_valid[0].tolist(),
                'minimum_moving_fixture_clearance_m': float(gaps[:, self._moving_spheres].min())}

    def forward_ee_pose_base(self, joint_pos):
        """Configured tool pose in robot base, using the same IK robot model."""
        state = self.ik_solver.compute_kinematics(self._build_joint_state(joint_pos))
        tool = state.tool_poses[self.ee_link]
        position = tool.position.detach().cpu().numpy().reshape(3) - np.asarray(self.frame_bias)
        quaternion = tool.quaternion.detach().cpu().numpy().reshape(4)
        return np.r_[position, quaternion]

    def plan_joint(self, start_joint_pos, goal_joint_pos):
        start_joint_state = self._build_joint_state(start_joint_pos)
        goal_joint_state = self._build_joint_state(goal_joint_pos)

        max_attempts = 5
        result = self.motion_planner.plan_cspace(
            current_state=start_joint_state,
            goal_state=goal_joint_state,
            max_attempts=max_attempts,
            enable_graph_attempt=0,
        )

        res_result = dict()
        if result is None or not self._is_success(result.success):
            res_result["status"] = "Fail"
            return res_result

        interpolated = result.get_interpolated_plan()
        jnames = getattr(interpolated, "joint_names", None)
        res_result["status"] = "Success"
        res_result["position"] = self._extract_trajectory(interpolated.position, batched=False, traj_joint_names=jnames)
        res_result["velocity"] = self._extract_trajectory(interpolated.velocity, batched=False, traj_joint_names=jnames)
        res_result["acceleration"] = self._extract_trajectory(
            interpolated.acceleration, batched=False, traj_joint_names=jnames
        )
        if res_result["velocity"] is None and res_result["position"] is not None:
            res_result["velocity"] = np.zeros_like(res_result["position"])
        if res_result["acceleration"] is None and res_result["position"] is not None:
            res_result["acceleration"] = np.zeros_like(res_result["position"])
        if res_result["position"] is not None:
            # refuse a plan that leaves the joint limits or the wrist table (deeper than its start)
            violation = self.limits.check_path(self._extract_active_joint_values(start_joint_pos),
                                               res_result["position"])
            if violation is not None:
                return {"status": "Fail", "reason": "plan leaves the simulator's joint limits or the wrist range",
                        "limit_violation": violation}
        return res_result

    def solve_ik_to_joint(self, curr_joint_pos, target_ee_pose, real_robot_pose, num_seeds: int = IK_SEEDS):
        """IK for a world link6 pose of the arm based at `real_robot_pose`; nearest valid branch to the current joints.

        `num_seeds` is accepted for API compatibility only: the solver's seed count is fixed at construction
        (CUDA graphs are sized against it)."""
        target_pose = deepcopy(target_ee_pose)
        target_pose = calculate_target_pose(real_robot_pose, self.robot_origin_pose, target_pose)
        world_base_pose = np.array(self.robot_origin_pose, dtype=np.float32)
        world_target_pose = np.array(target_pose, dtype=np.float32)
        target_pose_p, target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)
        target_pose_p[0] += self.frame_bias[0]
        target_pose_p[1] += self.frame_bias[1]
        target_pose_p[2] += self.frame_bias[2]
        goal_tool_poses = self._build_goal_pose(target_pose_p, target_pose_q)
        current_state = self._build_joint_state(curr_joint_pos)

        # Return every seed's candidate so the caller can keep the branch nearest the prior sample.
        ik_result = self.ik_solver.solve_pose(
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            return_seeds=IK_SEEDS,
        )

        if ik_result.success is None or not self._is_success(ik_result.success):
            return {
                "status": "Fail",
                "joint_names": self.active_joints_name,
            }

        # cuRobo v2's standalone InverseKinematics fills js_solution with the FULL robot DOF (incl. locked
        # fingers/grippers like joint7/joint8), in the optimizer's internal order -- which is neither active-joint
        # nor cspace order. Always do a name-based lookup against js_solution.joint_names when it's attached; fall
        # back to the by-length heuristic only if the names are missing.
        limits = self.motion_planner.kinematics.get_joint_limits()
        limit_columns = [limits.joint_names.index(n) for n in self.active_joints_name]
        position_limits = limits.position.detach().cpu().numpy()[:, limit_columns]
        # choose only among candidates inside the simulator's joint limits (ARX.usd) and the wrist table. The
        # shifts by whole turns are nearest_solution's own; when its unconstrained choice is allowed it is unchanged.
        lower = np.maximum(position_limits[0], self.limits.lower)
        upper = np.minimum(position_limits[1], self.limits.upper)
        positions = np.asarray(ik_result.js_solution.position.detach().cpu())
        success = np.asarray(ik_result.success.detach().cpu())
        names = list(ik_result.js_solution.joint_names)
        current = curr_joint_pos            # exactly what nearest_solution was always given
        shifted, valid = shifted_candidates(positions, success, names, self.active_joints_name, current, lower,
                                            upper)
        allowed = valid & (self.limits.margins(shifted) >= -LIMIT_TOLERANCE)
        if not allowed.any():
            return {
                "status": "Fail",
                "joint_names": self.active_joints_name,
                "reason": f"none of {int(valid.sum())} IK solutions lies within the simulator's joint limits and "
                          f"the wrist range",
            }
        joint_solution = nearest_solution(
            positions, allowed.reshape(success.shape), names,
            self.active_joints_name, current, lower, upper,
        )
        dof = joint_solution.shape[0]
        n_active = len(self.active_joints_name)
        n_cspace = len(self.cspace_joint_names)
        js_names = list(getattr(ik_result.js_solution, "joint_names", None) or [])

        if len(js_names) == dof:
            try:
                indices = [js_names.index(n) for n in self.active_joints_name]
            except ValueError as exc:
                raise ValueError(
                    f"IK solution joint_names {js_names} do not contain all "
                    f"active joints {self.active_joints_name}: {exc}"
                ) from None
            joint_solution = joint_solution[indices]
        elif dof == n_cspace:
            joint_solution = joint_solution[self._active_to_cspace_idx]
        elif dof != n_active:
            raise ValueError(
                "Unexpected IK solution dimension from curobo: "
                f"got {dof}, expected either {n_active} active joints or "
                f"{n_cspace} cspace joints (and no usable joint_names "
                f"on js_solution)."
            )

        return {
            "status": "Success",
            "joint_names": self.active_joints_name,
            "joint_value": joint_solution,
        }

    # ------------------------------------------------------------------ helpers
    def _build_goal_pose(self, target_pose_p, target_pose_q):
        device = self.device_cfg.device
        pos = torch.as_tensor(target_pose_p, dtype=torch.float32, device=device).reshape(1, 3)
        quat = torch.as_tensor(target_pose_q, dtype=torch.float32, device=device).reshape(1, 4)
        return GoalToolPose.from_poses(
            {self.ee_link: Pose(position=pos, quaternion=quat)},
            ordered_tool_frames=self.tool_frames,
            num_goalset=1,
        )

    def _trans_from_world_to_base(self, base_pose, target_pose):
        """
        transform target pose from world frame to base frame
        base_pose: np.array([x, y, z, qw, qx, qy, qz])
        target_pose: np.array([x, y, z, qw, qx, qy, qz])
        """
        base_p, base_q = base_pose[0:3], base_pose[3:]
        target_p, target_q = target_pose[0:3], target_pose[3:]
        rel_p = target_p - base_p
        wRb = t3d.quaternions.quat2mat(base_q)
        wRt = t3d.quaternions.quat2mat(target_q)
        result_p = wRb.T @ rel_p
        result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
        return result_p, result_q

    def _build_robot_and_scene_cfg(self, yml_data, table):
        """Translate a Maniverse-style yml into v2 robot_cfg + scene_model (one table slab in the base frame)."""
        kinematics = deepcopy(yml_data.get("robot_cfg", {}).get("kinematics", {}))
        ee_link = kinematics.get("ee_link", None)
        tool_frames = kinematics.get("tool_frames", None)
        if tool_frames is None:
            if ee_link is None:
                raise ValueError(f"robot_cfg.kinematics in {self.yml_path} requires ee_link or tool_frames.")
            tool_frames = [ee_link]
        kinematics["tool_frames"] = list(tool_frames)
        kinematics["format_version"] = 2.0

        cspace = deepcopy(kinematics.get("cspace", {}))
        if "default_joint_position" not in cspace and "retract_config" in cspace:
            cspace["default_joint_position"] = cspace["retract_config"]
        cspace.pop("retract_config", None)
        cspace_allowed = {
            "joint_names",
            "default_joint_position",
            "cspace_distance_weight",
            "null_space_weight",
            "null_space_maximum_distance",
            "max_acceleration",
            "max_jerk",
            "velocity_scale",
            "acceleration_scale",
            "jerk_scale",
            "position_limit_clip",
        }
        cspace = {k: v for k, v in cspace.items() if k in cspace_allowed}
        if "joint_names" not in cspace:
            cspace["joint_names"] = list(self.active_joints_name)
        if "default_joint_position" not in cspace:
            cspace["default_joint_position"] = [0.0 for _ in cspace["joint_names"]]
        kinematics["cspace"] = cspace

        kinematics_allowed = {
            "format_version",
            "base_link",
            "tool_frames",
            "collision_link_names",
            "collision_spheres",
            "collision_sphere_buffer",
            "extra_collision_spheres",
            "self_collision_ignore",
            "self_collision_buffer",
            "asset_root_path",
            "mesh_link_names",
            "lock_joints",
            "extra_links",
            "cspace",
            "urdf_path",
            "use_global_cumul",
            "grasp_contact_link_names",
        }
        kinematics = {k: v for k, v in kinematics.items() if k in kinematics_allowed}
        robot_cfg_v2 = {
            "robot_cfg": {"kinematics": kinematics},
            "load_dynamics": bool(yml_data.get("load_dynamics", False)),
        }
        # Fixed base/table contact is handled by the world-only cost adapter (migration/fixture_solver.py).
        scene_model = {"cuboid": {"table": table.cuboid()}}
        return robot_cfg_v2, scene_model

    def _build_installation(self):
        family = Path(self.yml_path).parent.name
        if family != "x5":
            raise ValueError("Fixed-table solver supports only the X5 installation")
        return FixedX5Table.from_scene(
            SceneCfg.create(self.scene_model), robot_family=family,
            fix_root_link=self.fix_root_link,
            root=self.robot_cfg["robot_cfg"]["kinematics"]["base_link"],
        )

    def _build_motion_planner_cfg(self):
        cfg = MotionPlannerCfg.create(
            robot=self.robot_cfg,
            scene_model=self.scene_model,
            device_cfg=self.device_cfg,
            num_ik_seeds=IK_SEEDS,
            num_trajopt_seeds=TRAJOPT_SEEDS,
            self_collision_check=True,
            cost_manager_config_instance_type=FixedRootCostManagerCfg,
            use_cuda_graph=self.use_cuda_graph,
            max_batch_size=1,
            multi_env=False,
            max_goalset=1,
        )
        cfg.trajopt_solver_config.interpolation_dt = self.interpolation_dt
        bind_single_planner_config(cfg, self.installation)
        return cfg

    def _build_ik_cfg(self):
        cfg = InverseKinematicsCfg.create(
            robot=self.robot_cfg,
            scene_model=self.scene_model,
            device_cfg=self.device_cfg,
            num_seeds=IK_SEEDS,
            seed_solver_num_seeds=IK_SEEDS,
            position_tolerance=0.001,
            orientation_tolerance=0.02,
            self_collision_check=True,
            cost_manager_config_instance_type=FixedRootCostManagerCfg,
            use_cuda_graph=self.use_cuda_graph,
            max_batch_size=1,
            multi_env=False,
            max_goalset=1,
        )
        bind_solver_config(cfg, self.installation)
        return cfg

    def _prewarm_alternate_modes(self):
        """Warm single cspace and standalone IK modes skipped by warmup.

        Contract checks run before warmup and are never swallowed here.
        Mode-specific warmup remains best effort, as in the original wrapper.
        """
        try:
            single_retract = self.motion_planner.default_joint_state.clone().unsqueeze(0)
            single_js = JointState.from_position(
                single_retract.position,
                joint_names=list(self.motion_planner.joint_names),
            )
            goal_state = single_js.clone()
            goal_state.position[..., 0] += 0.05
            self.motion_planner.plan_cspace(
                goal_state=goal_state,
                current_state=single_js,
                max_attempts=1,
            )
        except Exception:
            pass

        try:
            ik_retract = self.ik_solver.default_joint_state.clone().unsqueeze(0)
            ik_js = JointState.from_position(
                ik_retract.position,
                joint_names=list(self.ik_solver.joint_names),
            )
            ik_kin = self.ik_solver.compute_kinematics(ik_js)
            ik_goal = GoalToolPose.from_poses(
                ik_kin.tool_poses.to_dict(),
                ordered_tool_frames=list(self.ik_solver.tool_frames),
                num_goalset=1,
            )
            self.ik_solver.solve_pose(
                goal_tool_poses=ik_goal,
                current_state=ik_js,
                return_seeds=1,
            )
        except Exception:
            pass

    def _is_success(self, success):
        if success is None:
            return False
        if isinstance(success, torch.Tensor):
            return bool(success.any().item())
        return bool(np.asarray(success).any())

    def _extract_trajectory(self, tensor, batched, traj_joint_names=None):
        """Pull active-joint columns out of an interpolated trajectory tensor.

        cuRobo v2 returns trajectories with ALL robot joints (incl. gripper) in a robot-specific order that may
        differ from cspace.joint_names. `traj_joint_names` (from interp.joint_names) is used to locate active
        joints by name when the trajectory DOF exceeds the active DOF.
        """
        if tensor is None:
            return None
        arr = np.asarray(tensor.detach().cpu())
        target_ndim = 3 if batched else 2
        while arr.ndim > target_ndim:
            arr = arr[:, 0] if batched else arr[0]
        if not batched and arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if batched and arr.ndim == 2:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])

        dof = arr.shape[-1]
        n_active = len(self.active_joints_name)
        n_cspace = len(self.cspace_joint_names)

        if dof == n_cspace:
            return arr[..., self._active_to_cspace_idx]
        if dof == n_active:
            return arr
        if traj_joint_names is not None and len(traj_joint_names) == dof:
            traj_names = list(traj_joint_names)
            try:
                indices = [traj_names.index(n) for n in self.active_joints_name]
                return arr[..., indices]
            except ValueError:
                raise ValueError(
                    f"Trajectory joint_names {traj_names} do not contain all active joints {self.active_joints_name}"
                ) from None
        return None
