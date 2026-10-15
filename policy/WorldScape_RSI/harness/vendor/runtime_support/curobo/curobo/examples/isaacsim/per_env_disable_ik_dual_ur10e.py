# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-env tool-pose disable smoke test for :class:`InverseKinematics`.

4 envs of ``dual_ur10e.yml`` in one batched ``solve_pose`` call with a
per-env disable pattern:

::

    env 0: disable tool0   (drive tool1 only)
    env 1: disable tool1   (drive tool0 only)
    env 2: track both
    env 3: track both

IK apply pattern follows :mod:`curobo.examples.isaacsim.ik_batched_env`:
``solve_pose`` returns a joint config per env; we teleport each robot
with :meth:`Robot.set_joint_positions`. No trajectory, no physics
stepping through waypoints.

Two sub-modes via ``--mode``:

* ``single`` (default) — one goal per env per tool frame, pulled from a
  draggable :class:`VisualCuboid`. ``max_goalset=1``.
* ``goalset`` — sub-option A. Each env picks ONE arm and gets G goal
  candidates on it; the other arm is disabled per env.
  ``max_goalset=G``; chosen arm's candidates arranged in a ring around
  the cube anchor.

Disabled cubes are rendered grey so you can visually confirm they do
not drive the arm.

Run::

    python -m curobo.examples.isaacsim.per_env_disable_ik_dual_ur10e --mode single
    python -m curobo.examples.isaacsim.per_env_disable_ik_dual_ur10e --mode goalset
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
    description="cuRobo v2 per-env disable IK smoke test on dual_ur10e"
)
parser.add_argument("--headless_mode", type=str, default=None)
parser.add_argument(
    "--mode",
    type=str,
    choices=["single", "goalset"],
    default="single",
    help="single -> one goal per (env, frame); goalset -> G candidates on the chosen arm per env.",
)
parser.add_argument("--n_envs", type=int, default=4)
parser.add_argument("--offset_y", type=float, default=2.5)
parser.add_argument(
    "--num_goalset",
    type=int,
    default=8,
    help="Goalset size G for --mode goalset (chosen-arm only).",
)
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
from omni.isaac.core.objects import cuboid
from pxr import Gf, UsdGeom

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path
from curobo.examples.isaacsim.helper import add_extensions, add_robot_to_scene
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.logging import log_and_raise, setup_logger
from curobo.types import GoalToolPose, JointState, Pose


ROBOT_YAML = "dual_ur10e.yml"


# ---- Helpers --------------------------------------------------------------


def _disable_pattern_for(n_envs: int, tool_frames: List[str]) -> List[List[str]]:
    """Per-env list of frames to disable.

    Hardcoded for n_envs == 4 to match the test scenario; for other
    values we alternate.
    """
    if n_envs == 4:
        return [
            [tool_frames[1]],   # env 0: disable second frame, drive first
            [tool_frames[0]],   # env 1: disable first frame, drive second
            [],                 # env 2: track both
            [],                 # env 3: track both
        ]
    out: List[List[str]] = []
    for e in range(n_envs):
        out.append([tool_frames[1] if e % 2 == 0 else tool_frames[0]])
    return out


def _arm_per_env_for(n_envs: int) -> List[int]:
    """For --mode goalset: which tool_frame index each env's goalset targets."""
    if n_envs == 4:
        return [0, 1, 0, 1]
    return [0 if e % 2 == 0 else 1 for e in range(n_envs)]


def _apply_per_env_disable(
    solver: InverseKinematics,
    device_cfg,
    pattern: List[List[str]],
    tool_frames: List[str],
) -> None:
    """Disable listed frames per env; track the rest."""
    track = ToolPoseCriteria.track_position_and_orientation(
        xyz=[1.0, 1.0, 1.0], rpy=[1.0, 1.0, 1.0]
    )
    track.device_cfg = device_cfg
    track.__post_init__()
    disabled = ToolPoseCriteria.disabled()
    disabled.device_cfg = device_cfg
    disabled.__post_init__()
    for env_idx, disable_set in enumerate(pattern):
        criteria: Dict[str, ToolPoseCriteria] = {}
        ds = set(disable_set)
        for frame in tool_frames:
            criteria[frame] = disabled if frame in ds else track
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


def _make_target_cubes(
    n_envs: int,
    tool_frames: List[str],
    retract_tool_poses: Dict[str, Pose],
    env_offsets: List[np.ndarray],
    pattern: List[List[str]],
):
    """One VisualCuboid per (env, tool_frame). Disabled cubes are grey."""
    cubes: List[Dict[str, "cuboid.VisualCuboid"]] = []
    for env_id in range(n_envs):
        per_env: Dict[str, "cuboid.VisualCuboid"] = {}
        disable_set = set(pattern[env_id])
        for ti, frame in enumerate(tool_frames):
            pose = retract_tool_poses[frame]
            pos = pose.position.view(3).detach().cpu().numpy() + env_offsets[env_id]
            quat = pose.quaternion.view(4).detach().cpu().numpy()
            if frame in disable_set:
                color = np.array([0.4, 0.4, 0.4])
            elif ti == 0:
                color = np.array([1.0, 0.0, 0.0])
            else:
                color = np.array([0.5, 0.5, 0.0])
            per_env[frame] = cuboid.VisualCuboid(
                f"/World/world_{env_id}/target_{frame}",
                position=pos,
                orientation=quat,
                color=color,
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


def _build_single_goal(pose_dict: Dict[str, Pose], tool_frames: List[str]) -> GoalToolPose:
    """(B, 1, L, 1, 3/4) — one pose per (env, tool_frame).

    IMPORTANT: do NOT call ``pose.unsqueeze(1)`` here. ``Pose.unsqueeze``
    is in-place (mutates ``self.position``/``self.quaternion``) and we
    re-read the same ``pose_dict[primary_frame]`` reference across ticks
    for the solve-gating distance check. ``from_poses`` with
    ``num_goalset=1`` already handles the final ``(batch, 1, 3)`` view
    internally from ``(batch, 3)`` inputs.
    """
    return GoalToolPose.from_poses(
        pose_dict,
        ordered_tool_frames=tool_frames,
        num_goalset=1,
    )


def _build_goalset(
    pose_dict: Dict[str, Pose],
    tool_frames: List[str],
    arm_per_env: List[int],
    n_envs: int,
    G: int,
    device,
    dtype,
) -> GoalToolPose:
    """(N, 1, L, G, 3/4) — chosen arm gets G candidates around its cube
    anchor; other arm gets its cube pose broadcast over G (weight 0,
    value irrelevant).
    """
    L = len(tool_frames)
    position = torch.zeros((n_envs, 1, L, G, 3), device=device, dtype=dtype)
    quaternion = torch.zeros((n_envs, 1, L, G, 4), device=device, dtype=dtype)
    # Ring offsets around the anchor.
    grid_offsets = torch.zeros((G, 3), device=device, dtype=dtype)
    for g in range(G):
        ang = 2.0 * np.pi * g / G
        grid_offsets[g, 0] = 0.05 * np.cos(ang)
        grid_offsets[g, 1] = 0.05 * np.sin(ang)
        grid_offsets[g, 2] = 0.02 * (g % 2)
    for e in range(n_envs):
        chosen_li = arm_per_env[e]
        anchor = pose_dict[tool_frames[chosen_li]]
        position[e, 0, chosen_li] = anchor.position[e].unsqueeze(0).expand(G, 3) + grid_offsets
        quaternion[e, 0, chosen_li] = anchor.quaternion[e].unsqueeze(0).expand(G, 4)
        other_li = 1 - chosen_li
        other = pose_dict[tool_frames[other_li]]
        position[e, 0, other_li] = other.position[e].unsqueeze(0).expand(G, 3)
        quaternion[e, 0, other_li] = other.quaternion[e].unsqueeze(0).expand(G, 4)
    return GoalToolPose(
        tool_frames=tool_frames,
        position=position,
        quaternion=quaternion,
    )


# ---- Main -----------------------------------------------------------------


def main() -> None:
    setup_logger("warn")
    n_envs = int(args.n_envs)
    if n_envs < 2:
        log_and_raise(f"--n_envs must be >= 2, got {n_envs}")

    my_world, robots, env_offsets, robot_cfg = _scene_setup(n_envs, float(args.offset_y))

    G = int(args.num_goalset) if args.mode == "goalset" else 1
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
    ik = InverseKinematics(cfg)
    tool_frames = list(ik.tool_frames)

    # Pick per-env disable pattern based on mode.
    if args.mode == "goalset":
        arm_per_env = _arm_per_env_for(n_envs)
        pattern = [[tool_frames[1 - arm_per_env[e]]] for e in range(n_envs)]
    else:
        arm_per_env = None
        pattern = _disable_pattern_for(n_envs, tool_frames)

    _apply_per_env_disable(ik, ik.device_cfg, pattern, tool_frames)
    print(
        f"[ik/{args.mode}] tool_frames={tool_frames} "
        f"per-env disable={pattern} "
        f"{f'arm_per_env={arm_per_env}' if arm_per_env else ''} "
        f"G={G}"
    )

    # Cubes at retract FK.
    default_js_1 = ik.default_joint_state.clone().unsqueeze(0)
    kin = ik.compute_kinematics(default_js_1)
    retract_tool_poses = {n: kin.tool_poses.get_link_pose(n) for n in tool_frames}
    cubes = _make_target_cubes(n_envs, tool_frames, retract_tool_poses, env_offsets, pattern)

    add_extensions(simulation_app, args.headless_mode)
    if args.headless_mode is not None:
        my_world.play()

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = (
        robot_cfg["kinematics"]["cspace"].get("default_joint_position")
        or robot_cfg["kinematics"]["cspace"].get("retract_config")
    )

    # per_env_idx_list is used only during step<=10 teleport to default.
    # Trajectory/solution-apply uses a separate _cmd_idx_list, built per
    # plan, paired with the JointState's own joint_names — see below.
    per_env_idx_list: List[List[int]] = [[] for _ in range(n_envs)]
    _cmd_idx_list: List[List[int]] = [[] for _ in range(n_envs)]
    step_wait = 0
    device = ik.device_cfg.device
    dtype = ik.device_cfg.dtype

    # Solve-gating state (matches ``ik_batched_env.py``):
    # only call ``solve_pose`` when a cube has moved since the last solve
    # AND is settled (same pose as last tick) AND all robots are static
    # AND there's no pending per-env cmd. Without this, calling solve_pose
    # every tick picks a different IK branch frame-to-frame → visible flail.
    primary_frame = tool_frames[0]
    prev_primary_goal: Optional[Pose] = None
    past_primary_goal: Optional[Pose] = None
    # cmd_plan[env_id] = (position_np, joint_names) pending teleport.
    # Populated when solve_pose succeeds; cleared after the next tick
    # applies it via ``set_joint_positions(position_np, idx_list)``.
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

        # ---- Solve gating ------------------------------------------------
        primary_goal = pose_dict[primary_frame]
        if prev_primary_goal is None:
            prev_primary_goal = primary_goal.clone()
        if past_primary_goal is None:
            past_primary_goal = primary_goal.clone()

        prev_d = primary_goal.distance(prev_primary_goal)   # vs last solved
        past_d = primary_goal.distance(past_primary_goal)   # vs last tick
        any_moved = bool((prev_d[0] > 1e-3).any() or (prev_d[1] > 1e-3).any())
        all_settled = bool((past_d[0] < 1e-6).all() and (past_d[1] < 1e-6).all())
        all_static = bool(full_js.velocity.abs().max() < 0.2)
        no_pending = all(c is None for c in cmd_plan)
        should_solve = any_moved and all_settled and all_static and no_pending

        if should_solve:
            if args.mode == "goalset":
                goal = _build_goalset(
                    pose_dict, tool_frames, arm_per_env, n_envs, G, device, dtype
                )
            else:
                goal = _build_single_goal(pose_dict, tool_frames)

            current_state = full_js.reorder(ik.joint_names)
            result = ik.solve_pose(goal, current_state=current_state)
            prev_primary_goal.copy_(primary_goal)

            if result is not None and bool(result.success.any().item()):
                # IMPORTANT: use ``result.js_solution`` (which carries
                # ``joint_names``) rather than ``result.solution`` whose
                # column order is the solver's internal joint order — that
                # may NOT equal the cspace ``j_names`` we built
                # ``per_env_idx_list`` from at init. Mirroring
                # ``ik_batched_env.py:444-470``: per env, walk
                # ``sim_js_names`` and pair each with its dof index +
                # the matching column in the solution.
                js_sol = result.js_solution
                # js_sol.position shape: (B, return_seeds=1, dof)
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
                    # Build (position values, dof indices) pair aligned
                    # to sim_js_names ∩ sol_joint_names.
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

                if args.mode == "goalset":
                    gi = result.goalset_index
                    picks = [
                        int(gi[e, 0, arm_per_env[e]].item())
                        if bool(result.success[e].any().item())
                        else -1
                        for e in range(n_envs)
                    ]
                    print(
                        f"[ik/goalset step {step}] success={result.success.flatten().tolist()}"
                        f" picked g={picks}"
                    )
                else:
                    print(
                        f"[ik/{args.mode} step {step}] "
                        f"success={result.success.flatten().tolist()}"
                    )

        # ---- Apply pending per-env cmd_plan (teleport) -------------------
        # We apply on the tick AFTER solve so the ``full_js`` read next
        # tick reflects the new robot state (else still_vs_last_frame
        # would immediately re-trigger). ``_cmd_idx_list[env_id]`` is the
        # per-plan idx list paired with cmd_plan[env_id]; do NOT use the
        # init ``per_env_idx_list`` here (that's built from cspace
        # ``j_names``, which may differ from the IK solution's joint
        # order — see the build site above).
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
