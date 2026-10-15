# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-env tool-pose disable smoke test for :class:`BatchMotionPlanner`.

4 envs of ``dual_ur10e.yml`` in one batched ``plan_pose`` call with a
per-env disable pattern:

::

    env 0: disable tool0   (drive tool1 only)
    env 1: disable tool1   (drive tool0 only)
    env 2: track both
    env 3: track both

Unlike the IK variant (which teleports with ``set_joint_positions``),
MotionGen returns an interpolated trajectory per env. We replay each
env's trajectory through :class:`ArticulationAction` one waypoint per
sim tick — same apply pattern as :mod:`batched_multi_arm_reacher`. A
replan is triggered when all envs' primary cube has moved, all robots
are static, and there is nothing pending in any env's cmd_plan.

Disabled-arm cubes are rendered grey so you can visually confirm they
do not drive the arm during the plan.

Run::

    python -m curobo.examples.isaacsim.per_env_disable_motiongen_dual_ur10e
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
    description="cuRobo v2 per-env disable MotionGen smoke test on dual_ur10e"
)
parser.add_argument("--headless_mode", type=str, default=None)
parser.add_argument("--n_envs", type=int, default=4)
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
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, UsdGeom

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path
from curobo.examples.isaacsim.helper import add_extensions, add_robot_to_scene
from curobo.logging import log_and_raise, setup_logger
from curobo.motion_planner import MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose


ROBOT_YAML = "dual_ur10e.yml"


def _disable_pattern_for(n_envs: int, tool_frames: List[str]) -> List[List[str]]:
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


def _apply_per_env_disable(
    planner: BatchMotionPlanner,
    device_cfg,
    pattern: List[List[str]],
    tool_frames: List[str],
) -> None:
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
        planner.update_tool_pose_criteria_per_env(env_idx, criteria)


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
    cubes: List[Dict[str, "cuboid.VisualCuboid"]] = []
    for env_id in range(n_envs):
        per_env: Dict[str, "cuboid.VisualCuboid"] = {}
        disable_set = set(pattern[env_id])
        for ti, frame in enumerate(tool_frames):
            pose = retract_tool_poses[frame]
            pos = pose.position.view(3).detach().cpu().numpy() + env_offsets[env_id]
            quat = pose.quaternion.view(4).detach().cpu().numpy()
            if frame in disable_set:
                color = np.array([0.4, 0.4, 0.4])  # grey: disabled
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


def main() -> None:
    setup_logger("warn")
    n_envs = int(args.n_envs)
    if n_envs < 2:
        log_and_raise(f"--n_envs must be >= 2, got {n_envs}")

    my_world, robots, env_offsets, robot_cfg = _scene_setup(n_envs, float(args.offset_y))

    cfg = MotionPlannerCfg.create(
        robot=ROBOT_YAML,
        scene_model=[{} for _ in range(n_envs)],
        collision_cache={"cuboid": 10, "mesh": 10},
        max_batch_size=n_envs,
        multi_env=True,
        max_goalset=1,
        num_trajopt_seeds=12,
        num_ik_seeds=32,
        use_cuda_graph=False,
        self_collision_check=True,
        optimizer_collision_activation_distance=0.025,
    )
    planner = BatchMotionPlanner(cfg)
    tool_frames = list(planner.tool_frames)
    pattern = _disable_pattern_for(n_envs, tool_frames)
    _apply_per_env_disable(planner, planner.device_cfg, pattern, tool_frames)
    print(f"[motiongen] tool_frames={tool_frames} per-env disable={pattern}")

    print("warming up planner...")
    planner.warmup(enable_graph=False, num_warmup_iterations=3)

    default_js_1 = planner.default_joint_state.clone().unsqueeze(0)
    kin = planner.compute_kinematics(default_js_1)
    retract_tool_poses = {n: kin.tool_poses[n] for n in planner.tool_frames}
    cubes = _make_target_cubes(n_envs, tool_frames, retract_tool_poses, env_offsets, pattern)

    add_extensions(simulation_app, args.headless_mode)
    if args.headless_mode is not None:
        my_world.play()

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = (
        robot_cfg["kinematics"]["cspace"].get("default_joint_position")
        or robot_cfg["kinematics"]["cspace"].get("retract_config")
    )

    art_controllers = [r.get_articulation_controller() for r in robots]
    # per_env_idx_list[env_id] is used ONLY during step<=10 teleport to the
    # default config (via set_joint_positions with j_names order). It is
    # NOT used during trajectory apply — that path rebuilds a separate
    # (idx_list, common_js_names) pair per env, synced to the trajectory's
    # own joint order. See ``_cmd_idx_list`` below.
    per_env_idx_list: List[List[int]] = [[] for _ in range(n_envs)]
    # Trajectory-application indices: parallel to ``cmd_plan[s].position``.
    # Rebuilt every time a new plan is installed for env s; never reused
    # from the init ``per_env_idx_list``.
    _cmd_idx_list: List[List[int]] = [[] for _ in range(n_envs)]
    cmd_plan: List[Optional[JointState]] = [None] * n_envs
    cmd_idx = 0
    prev_first_goal: Optional[Pose] = None
    past_first_goal: Optional[Pose] = None
    step_wait = 0
    device = planner.device_cfg.device
    dtype = planner.device_cfg.dtype

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
                robot._articulation_view.set_max_efforts(
                    values=np.array([5000 for _ in range(len(per_env_idx_list[env_id]))]),
                    joint_indices=per_env_idx_list[env_id],
                )
        if step < 20:
            continue

        pose_dict = _read_target_poses(cubes, tool_frames, n_envs, device, dtype)
        full_js, sim_js_names = _read_full_js(robots, device, dtype)
        if full_js is None:
            continue

        first_goal = pose_dict[tool_frames[0]]
        if prev_first_goal is None:
            prev_first_goal = first_goal.clone()
        if past_first_goal is None:
            past_first_goal = first_goal.clone()

        prev_d = first_goal.distance(prev_first_goal)
        past_d = first_goal.distance(past_first_goal)
        any_moved = bool((prev_d[0] > 1e-2).any() or (prev_d[1] > 1e-2).any())
        all_settled = bool((past_d[0] == 0.0).all() and (past_d[1] == 0.0).all())
        all_static = bool(full_js.velocity.abs().max() < 0.2)
        no_pending = all(c is None for c in cmd_plan)

        if any_moved and all_settled and all_static and no_pending:
            print(f"[motiongen plan] triggered at step {step}")
            full_js_active = full_js.reorder(planner.kinematics.joint_names)
            goal_tool_poses = GoalToolPose.from_poses(
                pose_dict,
                ordered_tool_frames=tool_frames,
                num_goalset=1,
            )
            result = planner.plan_pose(
                goal_tool_poses,
                full_js_active,
                max_attempts=2,
                enable_graph_attempt=0,
            )
            prev_first_goal.copy_(first_goal)

            if result is not None and bool(result.success.any().item()):
                interp = result.interpolated_trajectory           # (B, 1, max_H, dof_full)
                last = result.interpolated_last_tstep             # (B, 1)
                for s in range(result.success.shape[0]):
                    if not bool(result.success[s].any().item()):
                        print(f"[motiongen plan] env_{s} failed")
                        cmd_plan[s] = None
                        continue
                    env_traj = interp[s].squeeze(0)               # (max_H, dof_full)
                    env_traj = trim_joint_state_trajectory(
                        env_traj, 0, int(last[s].item())
                    )
                    # IMPORTANT: build ``(common_js_names, idx_list)`` as a
                    # paired tuple so the apply-time ``joint_indices`` is
                    # aligned with the trajectory's joint order. If we
                    # reused ``per_env_idx_list[s]`` (which is built from
                    # ``j_names`` at init — cuRobo joint order), the
                    # position values and joint indices would be in
                    # different orders → wrong joint gets each target →
                    # arm drives to the wrong final pose. Matches
                    # ``batched_multi_arm_reacher.py:412-418``.
                    common_js_names: List[str] = []
                    idx_list_env: List[int] = []
                    for x in sim_js_names:
                        if x in env_traj.joint_names:
                            common_js_names.append(x)
                            idx_list_env.append(robots[s].get_dof_index(x))
                    cmd_plan[s] = env_traj.reorder(common_js_names)
                    _cmd_idx_list[s] = idx_list_env
                cmd_idx = 0
            else:
                print("[motiongen plan] batch plan failed")

        # Animate each env's trajectory one waypoint per tick.
        for s in range(n_envs):
            plan = cmd_plan[s]
            if plan is None:
                continue
            if cmd_idx >= len(plan.position):
                cmd_plan[s] = None
                continue
            cmd_state = plan[cmd_idx]
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy() if cmd_state.velocity is not None else None,
                joint_indices=_cmd_idx_list[s],
            )
            art_controllers[s].apply_action(art_action)
        cmd_idx += 1

        past_first_goal.copy_(first_goal)

        for _ in range(2):
            my_world.step(render=False)

    simulation_app.close()


if __name__ == "__main__":
    main()
