# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired vs padded goalset IK demo on ``tri_ur10e.yml`` (3 arms).

2 envs of ``tri_ur10e`` (tool0 / tool1 / tool2). tool2 is **always
disabled** so the 3rd cube is grey and the third arm doesn't drive.

Two modes:

* ``--mode paired`` — paired group on ``[tool0, tool1]`` with ``G=8``.
  Per env, build 8 paired candidates (both arms perturbed with the
  same g-index). Paired kernel must pick the SAME ``g`` for both arms
  per env (printed each replan).

* ``--mode padded`` — caller-side ragged-G padding. Real intent is
  ``[G=8 on tool0, G=16 on tool1, tool2=disabled]`` but the kernel
  only supports one ``num_goalset``. Caller pads tool0 from 8 → 16
  by duplicating each candidate twice (``[c0..c7, c0..c7]``). Runs
  unpaired with ``num_goalset=16``. After solve, ``g % 8`` recovers
  the real tool0 candidate. Picked g values per arm are printed each
  replan (independent per arm).

Solve-gating, IK apply pattern, and cube setup mirror
``per_env_disable_ik_dual_ur10e.py``. tool2's cube is rendered grey to
visually confirm it doesn't drive anything.

Goalset candidates are visualized with :class:`VisualSphere` clouds —
one sphere per (env, active_frame, real_candidate) at
``anchor + offsets[g]``. Offsets are sampled on a 3D meshgrid (port of
the ``ik_reachability.get_pose_grid`` pattern). After each solve the
picked sphere is recolored bright green and enlarged. Sphere positions
follow the dragged anchor cube every tick — drag the cube and the
candidate cloud trails it.

Run::

    python -m curobo.examples.isaacsim.goalset_groups_tri_ur10e --mode paired
    python -m curobo.examples.isaacsim.goalset_groups_tri_ur10e --mode padded
"""
from __future__ import annotations

# ---- Isaac Sim bootstrap (must come before any curobo / pxr imports) ------
from curobo.examples.isaacsim import bootstrap  # noqa: F401

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import argparse
from typing import Dict, List, Optional

import torch

_ = torch.zeros(4, device="cuda:0")

parser = argparse.ArgumentParser(
    description="cuRobo v2 paired/padded goalset IK demo on tri_ur10e (3 arms)"
)
parser.add_argument("--headless_mode", type=str, default=None)
parser.add_argument(
    "--mode",
    type=str,
    choices=["paired", "padded"],
    default="paired",
    help="paired -> [tool0, tool1] paired G=8; "
         "padded -> tool0 padded 8->16, tool1 G=16, both unpaired.",
)
parser.add_argument("--n_envs", type=int, default=2)
parser.add_argument("--offset_y", type=float, default=2.5)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Exit after N simulator steps (smoke test in headless mode).",
)
args = parser.parse_args()

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# ---- Post-SimulationApp imports ------------------------------------------
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from pxr import Gf, UsdGeom

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.solver.solver_core_cfg import enable_paired_tool_pose
from curobo.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path
from curobo.examples.isaacsim.helper import add_extensions, add_robot_to_scene
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.logging import log_and_raise, setup_logger
from curobo.types import GoalToolPose, JointState, Pose


ROBOT_YAML = "tri_ur10e.yml"
# Always-disabled frame in this demo. tool0 + tool1 are always tracked;
# tool2 is rendered grey and not part of any active goalset.
DISABLED_FRAME = "tool2"
ACTIVE_FRAMES = ("tool0", "tool1")
# Mode-specific kernel sizing.
G_PAIRED = 8                  # paired mode: same G across both active arms
G_PADDED_REAL_TOOL0 = 8       # padded mode: tool0's real candidates
G_PADDED_TOTAL = 16           # padded mode: tool1 real + tool0 padded → 16


# ---- Helpers --------------------------------------------------------------


def _apply_disable_pattern(
    solver: InverseKinematics,
    device_cfg,
    n_envs: int,
    tool_frames: List[str],
) -> None:
    """tool0 / tool1 tracked; tool2 disabled — broadcast across all envs."""
    track = ToolPoseCriteria.track_position_and_orientation(
        xyz=[1.0, 1.0, 1.0], rpy=[1.0, 1.0, 1.0]
    )
    track.device_cfg = device_cfg
    track.__post_init__()
    disabled = ToolPoseCriteria.disabled()
    disabled.device_cfg = device_cfg
    disabled.__post_init__()
    for env_idx in range(n_envs):
        criteria: Dict[str, ToolPoseCriteria] = {}
        for frame in tool_frames:
            criteria[frame] = disabled if frame == DISABLED_FRAME else track
        solver.update_tool_pose_criteria_per_env(env_idx, criteria)


def _scene_setup(n_envs: int, offset_y: float):
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    stage.DefinePrim("/World", "Xform").GetStage().SetDefaultPrim(
        stage.GetPrimAtPath("/World")
    )
    stage.DefinePrim("/curobo", "Xform")

    robot_cfg = load_yaml(join_path(str(get_robot_configs_path()), ROBOT_YAML))["robot_cfg"]

    robots = []
    env_offsets: List[np.ndarray] = []
    for i in range(n_envs):
        env_origin = np.array([0.0, i * offset_y, 0.0], dtype=np.float32)
        env_offsets.append(env_origin)
        env_root = UsdGeom.Xform.Define(stage, f"/World/world_{i}")
        if i > 0:
            env_root.AddTranslateOp().Set(Gf.Vec3d(*env_origin.tolist()))
        robot, _ = add_robot_to_scene(
            robot_cfg,
            my_world,
            subroot=f"/World/world_{i}/",
            robot_name=f"robot_{i}",
            position=env_origin,
            initialize_world=False,
        )
        robots.append(robot)
    my_world.initialize_physics()
    return my_world, robots, env_offsets, robot_cfg


def _make_anchor_cubes(
    n_envs: int,
    tool_frames: List[str],
    retract_tool_poses: Dict[str, Pose],
    env_offsets: List[np.ndarray],
):
    """One draggable VisualCuboid per (env, tool_frame).

    tool0 = red, tool1 = yellow, tool2 = grey (disabled).
    The user drags the active anchors; on settle the demo regenerates
    the goalset around each anchor and solves.
    """
    color_for = {"tool0": np.array([1.0, 0.0, 0.0]),
                 "tool1": np.array([0.5, 0.5, 0.0]),
                 "tool2": np.array([0.4, 0.4, 0.4])}
    cubes: List[Dict[str, "cuboid.VisualCuboid"]] = []
    for env_id in range(n_envs):
        per_env: Dict[str, "cuboid.VisualCuboid"] = {}
        for frame in tool_frames:
            pose = retract_tool_poses[frame]
            pos = pose.position.view(3).detach().cpu().numpy() + env_offsets[env_id]
            quat = pose.quaternion.view(4).detach().cpu().numpy()
            per_env[frame] = cuboid.VisualCuboid(
                f"/World/world_{env_id}/target_{frame}",
                position=pos,
                orientation=quat,
                color=color_for.get(frame, np.array([0.4, 0.4, 0.4])),
                size=0.05,
            )
        cubes.append(per_env)
    return cubes


def _read_target_poses(cubes, tool_frames, n_envs, device, dtype) -> Dict[str, Pose]:
    pose_dict: Dict[str, Pose] = {}
    for frame in tool_frames:
        pos_stack = np.stack([cubes[e][frame].get_local_pose()[0] for e in range(n_envs)])
        quat_stack = np.stack([cubes[e][frame].get_local_pose()[1] for e in range(n_envs)])
        pose_dict[frame] = Pose(
            position=torch.as_tensor(pos_stack, device=device, dtype=dtype),
            quaternion=torch.as_tensor(quat_stack, device=device, dtype=dtype),
        )
    return pose_dict


def _read_full_js(robots, device, dtype):
    sim_js_names = robots[0].dof_names
    positions_np = []
    velocities_np = []
    for r in robots:
        s = r.get_joints_state()
        if s is None:
            return None, None
        positions_np.append(s.positions)
        velocities_np.append(s.velocities)
    positions = torch.as_tensor(np.stack(positions_np), device=device, dtype=dtype)
    velocities = torch.as_tensor(np.stack(velocities_np), device=device, dtype=dtype)
    full_js = JointState(
        position=positions,
        velocity=velocities * 0.0,
        acceleration=torch.zeros_like(positions),
        jerk=torch.zeros_like(positions),
        joint_names=sim_js_names,
    )
    return full_js, sim_js_names


# ---- Goal builders --------------------------------------------------------


def _grid_offsets_3d(
    n_x: int, n_y: int, n_z: int,
    ext_x: float, ext_y: float, ext_z: float,
    device, dtype,
):
    """Port of ``ik_reachability.get_pose_grid`` to torch — shape ``(n_x*n_y*n_z, 3)``.

    Z is one-sided (``[0, ext_z]``) so candidates only float ABOVE the
    anchor cube. xy are centered (``[-ext, +ext]``). This matches the
    v1/v2 reachability sample layout — predictable and easy to read in
    the viewport.
    """
    x = torch.linspace(-ext_x, ext_x, n_x, device=device, dtype=dtype)
    y = torch.linspace(-ext_y, ext_y, n_y, device=device, dtype=dtype)
    z = torch.linspace(0.0, ext_z, n_z, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)


# Single source of truth for offset grids — both the goal builder AND
# the sphere visualization read from these so the spheres always sit
# exactly at the candidate positions the kernel is evaluating.
def _paired_offsets(device, dtype):
    """G=8 paired (2x2x2 grid). One offset tensor used by BOTH active arms."""
    return _grid_offsets_3d(2, 2, 2, 0.04, 0.04, 0.06, device, dtype)


def _padded_real_tool0_offsets(device, dtype):
    """tool0 real 8 candidates (2x2x2 grid)."""
    return _grid_offsets_3d(2, 2, 2, 0.04, 0.04, 0.06, device, dtype)


def _padded_tool1_offsets(device, dtype):
    """tool1 real 16 candidates (4x2x2 grid) — wider in x to look distinct."""
    return _grid_offsets_3d(4, 2, 2, 0.06, 0.04, 0.05, device, dtype)


def _resolve_offsets_for_active_frame(
    mode: str, frame: str, device, dtype,
):
    """Returns the (real_count, real_offsets) tuple for the given active frame.

    ``real_offsets`` is the deduplicated tensor of unique candidate
    positions to display as spheres. The padded representation needed
    by the kernel is built separately by ``_build_goal``.
    """
    if mode == "paired":
        offs = _paired_offsets(device, dtype)
        return offs.shape[0], offs
    # padded mode
    if frame == "tool0":
        offs = _padded_real_tool0_offsets(device, dtype)
        return offs.shape[0], offs
    if frame == "tool1":
        offs = _padded_tool1_offsets(device, dtype)
        return offs.shape[0], offs
    raise ValueError(f"unexpected active frame {frame}")


def _build_goal(
    mode: str,
    pose_dict: Dict[str, Pose],
    tool_frames: List[str],
    n_envs: int,
    device,
    dtype,
) -> GoalToolPose:
    """Unified goal builder for both modes — kernel sees uniform G.

    Paired: G=8, both active arms use the same offsets[g] (paired
    candidate g is jointly reachable). Padded: G=16, tool0 padded
    ``[c0..c7, c0..c7]`` from 8 reals, tool1 uses 16 distinct reals.
    tool2 (disabled) gets anchor-broadcast filler (kernel weight is
    zero — value doesn't matter).
    """
    if mode == "paired":
        G = G_PAIRED
        offs_active = _paired_offsets(device, dtype)
        # Same offsets for both active arms → "paired by index".
        offs_per_frame = {"tool0": offs_active, "tool1": offs_active}
    else:  # padded
        G = G_PADDED_TOTAL
        real_tool0 = _padded_real_tool0_offsets(device, dtype)
        # Pad: slot g reads real candidate (g % 8).
        idx = torch.arange(G, device=device) % G_PADDED_REAL_TOOL0
        offs_per_frame = {
            "tool0": real_tool0[idx],                          # (16, 3)
            "tool1": _padded_tool1_offsets(device, dtype),     # (16, 3)
        }

    L = len(tool_frames)
    position = torch.zeros((n_envs, 1, L, G, 3), device=device, dtype=dtype)
    quaternion = torch.zeros((n_envs, 1, L, G, 4), device=device, dtype=dtype)
    for ti, frame in enumerate(tool_frames):
        anchor = pose_dict[frame]
        if frame == DISABLED_FRAME:
            # tool2 filler — broadcast anchor across G; weight is zero.
            position[:, 0, ti] = anchor.position.unsqueeze(1).expand(n_envs, G, 3)
            quaternion[:, 0, ti] = anchor.quaternion.unsqueeze(1).expand(n_envs, G, 4)
        else:
            offs = offs_per_frame[frame]                         # (G, 3)
            position[:, 0, ti] = anchor.position.unsqueeze(1) + offs.unsqueeze(0)
            quaternion[:, 0, ti] = anchor.quaternion.unsqueeze(1).expand(n_envs, G, 4)
    return GoalToolPose(
        tool_frames=tool_frames, position=position, quaternion=quaternion,
    )


# ---- Goalset candidate sphere visualization ------------------------------


# Each (env, active_frame) gets a row of small spheres, one per UNIQUE
# candidate (8 for tool0/tool1 in paired mode; 8 for tool0 + 16 for tool1
# in padded mode). Padded mode does NOT show duplicate slots because
# they share a position with the original real slot — visually
# overlapping spheres are confusing. The "picked" sphere is recolored +
# enlarged after each solve so you can see which candidate the kernel
# chose (paired mode: same index on both arms; padded mode: independent).
DEFAULT_RADIUS = 0.012
PICKED_RADIUS = 0.024
DEFAULT_COLOR_TOOL0 = np.array([0.9, 0.4, 0.4])    # pinkish red
DEFAULT_COLOR_TOOL1 = np.array([0.85, 0.85, 0.3])  # pale yellow
PICKED_COLOR = np.array([0.1, 0.95, 0.2])           # bright green


def _set_sphere_color(sph: "sphere.VisualSphere", rgb: np.ndarray) -> None:
    """Update ``displayColor`` on a VisualSphere in-place.

    ``set_color`` isn't on the public surface for VisualSphere across
    every Isaac Sim version, so we poke the USD attribute directly.
    """
    prim = sph.prim
    attr = prim.GetAttribute("primvars:displayColor")
    if not attr:
        attr = UsdGeom.Sphere(prim).GetDisplayColorAttr()
    from pxr import Vt
    attr.Set(Vt.Vec3fArray([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))]))


def _make_candidate_spheres(
    mode: str,
    n_envs: int,
    tool_frames: List[str],
    retract_tool_poses: Dict[str, Pose],
    env_offsets: List[np.ndarray],
    device,
    dtype,
):
    """Create one VisualSphere per (env, active_frame, real_candidate).

    Returns ``spheres[env_id][frame] = list[VisualSphere]`` for active
    frames only. Disabled frame gets no spheres.
    """
    color_for = {"tool0": DEFAULT_COLOR_TOOL0, "tool1": DEFAULT_COLOR_TOOL1}
    out: List[Dict[str, List["sphere.VisualSphere"]]] = []
    for env_id in range(n_envs):
        per_env: Dict[str, List["sphere.VisualSphere"]] = {}
        for frame in ACTIVE_FRAMES:
            anchor_pose = retract_tool_poses[frame]
            anchor_pos_world = (
                anchor_pose.position.view(3).detach().cpu().numpy()
                + env_offsets[env_id]
            )
            real_count, real_offsets = _resolve_offsets_for_active_frame(
                mode, frame, device, dtype,
            )
            offsets_np = real_offsets.detach().cpu().numpy()
            sph_list: List["sphere.VisualSphere"] = []
            for g in range(real_count):
                pos = anchor_pos_world + offsets_np[g]
                sph_list.append(
                    sphere.VisualSphere(
                        prim_path=f"/World/world_{env_id}/cand_{frame}_{g}",
                        position=pos,
                        radius=DEFAULT_RADIUS,
                        color=color_for[frame],
                    )
                )
            per_env[frame] = sph_list
        out.append(per_env)
    return out


def _refresh_candidate_spheres(
    spheres,
    mode: str,
    n_envs: int,
    pose_dict: Dict[str, Pose],
    env_offsets: List[np.ndarray],
    device,
    dtype,
) -> None:
    """Reposition each candidate sphere to follow its anchor cube.

    Anchor cubes are draggable; sphere positions = anchor + offsets[g]
    so the candidate cloud always trails the anchor in real time.
    """
    for frame in ACTIVE_FRAMES:
        _, real_offsets = _resolve_offsets_for_active_frame(
            mode, frame, device, dtype,
        )
        offsets_np = real_offsets.detach().cpu().numpy()
        anchors = pose_dict[frame].position.detach().cpu().numpy()  # (n_envs, 3)
        for env_id in range(n_envs):
            anchor_world = anchors[env_id] + env_offsets[env_id]
            for g, sph in enumerate(spheres[env_id][frame]):
                sph.set_world_pose(position=anchor_world + offsets_np[g])


def _highlight_picked_sphere(
    spheres,
    n_envs: int,
    picked_real_g: Dict[str, List[int]],
) -> None:
    """Reset every candidate to the default look, then mark each env's
    pick with PICKED_COLOR + a larger radius. ``picked_real_g[frame]``
    is the list of REAL candidate indices (post mod-8 translation for
    padded mode), one per env, or -1 for unsolved.
    """
    color_for = {"tool0": DEFAULT_COLOR_TOOL0, "tool1": DEFAULT_COLOR_TOOL1}
    for env_id in range(n_envs):
        for frame in ACTIVE_FRAMES:
            for g, sph in enumerate(spheres[env_id][frame]):
                sph.set_radius(DEFAULT_RADIUS)
                _set_sphere_color(sph, color_for[frame])
            picked = picked_real_g[frame][env_id]
            if 0 <= picked < len(spheres[env_id][frame]):
                sel = spheres[env_id][frame][picked]
                sel.set_radius(PICKED_RADIUS)
                _set_sphere_color(sel, PICKED_COLOR)


# ---- Main -----------------------------------------------------------------


def main() -> None:
    setup_logger("warn")
    n_envs = int(args.n_envs)
    if n_envs < 1:
        log_and_raise(f"--n_envs must be >= 1, got {n_envs}")

    my_world, robots, env_offsets, robot_cfg = _scene_setup(n_envs, float(args.offset_y))

    G = G_PAIRED if args.mode == "paired" else G_PADDED_TOTAL
    cfg = InverseKinematicsCfg.create(
        robot=ROBOT_YAML,
        num_seeds=20,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        self_collision_check=True,
        use_cuda_graph=False,
        max_batch_size=n_envs,
        multi_env=True,
        max_goalset=G,
    )
    cfg.use_lm_seed = False
    if args.mode == "paired":
        # Must come AFTER the factory auto-enabled per_env (which it
        # did because multi_env=True). Paired dispatch requires per_env.
        enable_paired_tool_pose(cfg.core_cfg)
    ik = InverseKinematics(cfg)
    tool_frames = list(ik.tool_frames)
    if tool_frames != ["tool0", "tool1", "tool2"]:
        log_and_raise(f"expected tri_ur10e tool_frames=[tool0,tool1,tool2], got {tool_frames}")

    _apply_disable_pattern(ik, ik.device_cfg, n_envs, tool_frames)
    print(
        f"[goalset_groups/{args.mode}] n_envs={n_envs} tool_frames={tool_frames} "
        f"G={G} (real_tool0={G_PADDED_REAL_TOOL0 if args.mode == 'padded' else G}) "
        f"disabled={DISABLED_FRAME}"
    )

    # Anchor cubes at retract FK.
    default_js_1 = ik.default_joint_state.clone().unsqueeze(0)
    kin = ik.compute_kinematics(default_js_1)
    retract_tool_poses = {n: kin.tool_poses.get_link_pose(n) for n in tool_frames}
    cubes = _make_anchor_cubes(n_envs, tool_frames, retract_tool_poses, env_offsets)
    # Candidate cloud — one VisualSphere per (env, active_frame, real_g).
    # Placed at retract anchor + offset; refreshed every tick to follow
    # the dragged anchor cube; recolored after each solve to show pick.
    cand_spheres = _make_candidate_spheres(
        args.mode, n_envs, tool_frames, retract_tool_poses, env_offsets,
        ik.device_cfg.device, ik.device_cfg.dtype,
    )

    add_extensions(simulation_app, args.headless_mode)
    if args.headless_mode is not None:
        my_world.play()

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = (
        robot_cfg["kinematics"]["cspace"].get("default_joint_position")
        or robot_cfg["kinematics"]["cspace"].get("retract_config")
    )

    per_env_idx_list: List[List[int]] = [[] for _ in range(n_envs)]
    _cmd_idx_list: List[List[int]] = [[] for _ in range(n_envs)]
    step_wait = 0
    device = ik.device_cfg.device
    dtype = ik.device_cfg.dtype

    # Solve-gating mirrors per_env_disable_ik_dual_ur10e.py — only call
    # solve_pose when a primary cube has moved + settled + robots
    # static + no pending plan, otherwise tick-to-tick IK branch
    # picking causes visible flail.
    primary_frame = ACTIVE_FRAMES[0]
    prev_primary_goal: Optional[Pose] = None
    past_primary_goal: Optional[Pose] = None
    cmd_plan: List[Optional[torch.Tensor]] = [None] * n_envs

    while simulation_app.is_running():
        my_world.step(render=True)
        if args.max_steps is not None and my_world.current_time_step_index >= args.max_steps:
            print(f"Reached --max_steps={args.max_steps}, exiting.")
            break
        if not my_world.is_playing():
            if step_wait % 100 == 0:
                print("**** Click Play to start simulation *****")
            step_wait += 1
            continue

        step = my_world.current_time_step_index
        if step <= 10:
            for env_id, robot in enumerate(robots):
                robot._articulation_view.initialize()
                per_env_idx_list[env_id] = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, per_env_idx_list[env_id])
        if step < 20:
            continue

        pose_dict = _read_target_poses(cubes, tool_frames, n_envs, device, dtype)
        full_js, sim_js_names = _read_full_js(robots, device, dtype)
        if full_js is None:
            continue

        # Repaint candidate cloud every tick — anchor may have moved
        # (user dragging the cube). Cheap: just set_world_pose per sphere.
        _refresh_candidate_spheres(
            cand_spheres, args.mode, n_envs, pose_dict, env_offsets, device, dtype,
        )

        # ---- Solve gating ------------------------------------------------
        primary_goal = pose_dict[primary_frame]
        if prev_primary_goal is None:
            prev_primary_goal = primary_goal.clone()
        if past_primary_goal is None:
            past_primary_goal = primary_goal.clone()

        prev_d = primary_goal.distance(prev_primary_goal)
        past_d = primary_goal.distance(past_primary_goal)
        any_moved = bool((prev_d[0] > 1e-3).any() or (prev_d[1] > 1e-3).any())
        all_settled = bool((past_d[0] < 1e-6).all() and (past_d[1] < 1e-6).all())
        all_static = bool(full_js.velocity.abs().max() < 0.2)
        no_pending = all(c is None for c in cmd_plan)
        should_solve = any_moved and all_settled and all_static and no_pending

        if should_solve:
            goal = _build_goal(args.mode, pose_dict, tool_frames, n_envs, device, dtype)

            current_state = full_js.reorder(ik.joint_names)
            result = ik.solve_pose(goal, current_state=current_state)
            prev_primary_goal.copy_(primary_goal)

            if result is not None and bool(result.success.any().item()):
                # Translate IK solution back into per-env teleport.
                # ``js_solution`` carries the column joint_names directly,
                # so we don't have to assume cspace order matches.
                js_sol = result.js_solution
                sol_positions = (
                    js_sol.position[:, 0, :] if js_sol.position.dim() == 3
                    else js_sol.position
                )
                sol_joint_names: List[str] = list(js_sol.joint_names)
                for env_id in range(n_envs):
                    if not bool(result.success[env_id].any().item()):
                        cmd_plan[env_id] = None
                        _cmd_idx_list[env_id] = []
                        continue
                    cur_idx: List[int] = []
                    cur_vals: List[float] = []
                    for x in sim_js_names:
                        if x in sol_joint_names:
                            cur_idx.append(robots[env_id].get_dof_index(x))
                            cur_vals.append(
                                float(sol_positions[env_id, sol_joint_names.index(x)])
                            )
                    cmd_plan[env_id] = np.asarray(cur_vals, dtype=np.float32)
                    _cmd_idx_list[env_id] = cur_idx

                # Visibility: print picked g per arm + recolor sphere.
                # Paired mode -> tool0 and tool1 picked g must match per
                # env. Padded mode -> tool0 picked padded_g, real_g is
                # padded_g % 8; tool1 picked g is real already.
                gi = result.goalset_index
                tool0_col = tool_frames.index("tool0")
                tool1_col = tool_frames.index("tool1")
                picked_real_g: Dict[str, List[int]] = {"tool0": [], "tool1": []}
                for e in range(n_envs):
                    if not bool(result.success[e].any().item()):
                        print(f"  env {e}: solve FAILED")
                        picked_real_g["tool0"].append(-1)
                        picked_real_g["tool1"].append(-1)
                        continue
                    g0 = int(gi[e, 0, tool0_col].item())
                    g1 = int(gi[e, 0, tool1_col].item())
                    if args.mode == "paired":
                        match = "MATCH" if g0 == g1 else "MISMATCH(BUG?)"
                        print(f"  env {e}: tool0_g={g0} tool1_g={g1} -> {match}")
                        picked_real_g["tool0"].append(g0)
                        picked_real_g["tool1"].append(g1)
                    else:
                        real_g0 = g0 % G_PADDED_REAL_TOOL0
                        print(
                            f"  env {e}: tool0_padded_g={g0} (real_g={real_g0}) "
                            f"tool1_g={g1}"
                        )
                        # Sphere viz only carries the 8 real tool0
                        # spheres — translate via mod-8.
                        picked_real_g["tool0"].append(real_g0)
                        picked_real_g["tool1"].append(g1)
                _highlight_picked_sphere(cand_spheres, n_envs, picked_real_g)

        # ---- Apply pending per-env cmd_plan (teleport) -------------------
        for env_id, robot in enumerate(robots):
            cmd = cmd_plan[env_id]
            if cmd is None:
                continue
            robot.set_joint_positions(cmd, _cmd_idx_list[env_id])
            cmd_plan[env_id] = None
            _cmd_idx_list[env_id] = []

        past_primary_goal.copy_(primary_goal)

    simulation_app.close()


if __name__ == "__main__":
    main()
