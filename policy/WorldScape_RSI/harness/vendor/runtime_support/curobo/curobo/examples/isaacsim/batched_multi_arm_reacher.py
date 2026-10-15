# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched multi-arm motion-planning reacher — combines:

* :mod:`curobo.examples.isaacsim.multi_arm_reacher`: a robot YAML with more
  than one ``tool_frames`` entry (defaults to ``dual_ur10e.yml`` →
  ``["tool1", "tool0"]``) and one draggable target cube per tool frame.
* :mod:`curobo.examples.isaacsim.batch_motion_gen_reacher`: ``N`` envs at
  ``y`` offsets, each under its own ``/World/world_i`` Xform, all solved in
  a single :meth:`BatchMotionPlanner.plan_pose` call via
  ``max_batch_size=N, multi_env=True``.

So every plan solves ``N × L`` constraints simultaneously (``N`` envs times
``L`` tool frames per env). No obstacles are loaded — the collision worlds
are initialised from empty per-env scene dicts so the solver sees only
self-collision.

v1 → v2 call shape:

* ``GoalToolPose.position`` is ``(N, 1, L, 1, 3)``. Build by stacking each
  env's per-tool-frame :class:`Pose` → one :class:`Pose` per tool frame at
  shape ``(N, 3)`` and then calling
  :meth:`GoalToolPose.from_poses` with ``ordered_tool_frames=planner.tool_frames``.
* ``current_state.position`` is ``(N, dof)``.
* Result iteration mirrors :mod:`batch_motion_gen_reacher`: iterate
  ``result.success[s]`` and slice
  ``result.interpolated_trajectory[s].squeeze(0)`` trimmed by
  ``result.interpolated_last_tstep[s]``.

Initial cube placement matches the single-env ``multi_arm_reacher``: each
target cube starts at the forward-kinematics pose of the robot's retract
joint configuration, plus the env offset.

Run (inside the Isaac Sim Python env)::

    python -m curobo.examples.isaacsim.batched_multi_arm_reacher
    python -m curobo.examples.isaacsim.batched_multi_arm_reacher --n_envs 3
"""
from __future__ import annotations

# ---- Isaac Sim bootstrap ---------------------------------------------------
from curobo.examples.isaacsim import bootstrap  # noqa: F401

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import argparse
from typing import Dict, List

import torch

_ = torch.zeros(4, device="cuda:0")

parser = argparse.ArgumentParser(
    description="cuRobo v2 batched multi-arm motion planner reacher"
)
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket, webrtc].",
)
parser.add_argument(
    "--robot",
    type=str,
    default="magicsim_vega1p_sharpa_mobile.yml",
    help="cuRobo v2 robot YAML. Must declare tool_frames with len >= 2.",
)
parser.add_argument(
    "--n_envs",
    type=int,
    default=2,
    help="Number of parallel environments (= max_batch_size).",
)
parser.add_argument(
    "--offset_y",
    type=float,
    default=2.5,
    help="World-Y offset between adjacent env origins (metres).",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="Render each robot's collision spheres under its env root.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Exit after N simulator steps. Useful for smoke-testing in headless mode.",
)
args = parser.parse_args()

# ---- SimulationApp ---------------------------------------------------------
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# ---- Post-SimulationApp imports -------------------------------------------
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, UsdGeom

from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path
from curobo.examples.isaacsim.helper import add_extensions, add_robot_to_scene
from curobo.logging import log_and_raise, setup_logger
from curobo.motion_planner import MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose


def _resolve_robot_cfg(robot: str) -> dict:
    return load_yaml(join_path(str(get_robot_configs_path()), robot))["robot_cfg"]


def main() -> None:
    setup_logger("warn")

    n_envs = int(args.n_envs)
    offset_y = float(args.offset_y)
    if n_envs < 1:
        log_and_raise(f"--n_envs must be >= 1, got {n_envs}")

    # ---- Scene / stage setup -----------------------------------------------
    my_world = World(stage_units_in_meters=1.0)
    # my_world.scene.add_default_ground_plane()
    stage = my_world.stage
    stage.DefinePrim("/World", "Xform").GetStage().SetDefaultPrim(
        stage.GetPrimAtPath("/World")
    )
    stage.DefinePrim("/curobo", "Xform")

    # ---- Robot config -------------------------------------------------------
    robot_cfg = _resolve_robot_cfg(args.robot)
    tool_frames_yaml = robot_cfg["kinematics"].get("tool_frames")
    if not tool_frames_yaml or len(tool_frames_yaml) < 2:
        log_and_raise(
            "batched_multi_arm_reacher requires a robot YAML with tool_frames of "
            f"length >= 2; got tool_frames={tool_frames_yaml} in {args.robot}."
        )

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = (
        robot_cfg["kinematics"]["cspace"].get("default_joint_position")
        or robot_cfg["kinematics"]["cspace"].get("retract_config")
    )
    if default_config is None:
        log_and_raise("cspace.default_joint_position (or retract_config) missing from robot YAML")

    # ---- Per-env roots + robots (cubes placed after FK is available) -------
    robot_list = []
    env_offsets: List[np.ndarray] = []
    for i in range(n_envs):
        env_origin = np.array([0.0, i * offset_y, 0.0], dtype=np.float32)
        env_offsets.append(env_origin)

        # Same reasoning as batch_motion_gen_reacher: build the env root
        # directly with the USD API because UsdWriter.add_subroot misplaces
        # paths that start with ``/`` (join_usd_path strips the leading slash).
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
        robot_list.append(robot)

    my_world.initialize_physics()

    # ---- Planner: batch_env + multi-EEF, no obstacles ---------------------
    planner_cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        # Empty per-env scenes. multi_env=True still allocates one collision
        # world per slot; each is empty so the solver only has self-collision
        # to worry about. That's what "先不用导入障碍物" asked for.
        scene_model=[{} for _ in range(n_envs)],
        collision_cache={"cuboid": 10, "mesh": 10},
        max_batch_size=n_envs,
        multi_env=True,
        max_goalset=1,
        num_trajopt_seeds=12,
        num_ik_seeds=32,
        use_cuda_graph=True,
        self_collision_check=True,
        optimizer_collision_activation_distance=0.025,
    )
    planner = BatchMotionPlanner(planner_cfg)

    print("warming up...")
    planner.warmup(enable_graph=False, num_warmup_iterations=3)
    print(
        f"cuRobo v2 batched multi-arm planner ready "
        f"(n_envs={n_envs}, tool_frames={planner.tool_frames}, "
        f"action_dim={planner.action_dim})"
    )

    add_extensions(simulation_app, args.headless_mode)

    device = planner.device_cfg.device
    dtype = planner.device_cfg.dtype

    # ---- Retract FK per tool frame → per-env cube start pose --------------
    retract_js = planner.default_joint_state.clone().unsqueeze(0)
    retract_kin = planner.compute_kinematics(retract_js)
    retract_tool_poses = {name: retract_kin.tool_poses[name] for name in planner.tool_frames}

    # target_cubes[env_id][tool_frame] = VisualCuboid
    target_cubes: List[Dict[str, "cuboid.VisualCuboid"]] = []
    primary_frame = planner.tool_frames[0]
    for env_id in range(n_envs):
        per_env: Dict[str, "cuboid.VisualCuboid"] = {}
        for ti, frame in enumerate(planner.tool_frames):
            pose = retract_tool_poses[frame]
            # Retract FK is in robot-base frame. The per-env Xform carries the
            # env offset, so the cube's *world* position = retract + offset.
            pos = pose.position.view(3).detach().cpu().numpy() + env_offsets[env_id]
            quat = pose.quaternion.view(4).detach().cpu().numpy()
            # Distinct colour for the primary cube (so you can tell which is
            # which when both arms move).
            color = (
                np.array([1.0, 0.0, 0.0]) if ti == 0 else np.array([0.5, 0.5, 0.0])
            )
            per_env[frame] = cuboid.VisualCuboid(
                f"/World/world_{env_id}/target_{frame}",
                position=pos,
                orientation=quat,
                color=color,
                size=0.05,
            )
        target_cubes.append(per_env)

    if args.headless_mode is not None:
        my_world.play()

    # ---- Runtime state ----------------------------------------------------
    art_controllers = [r.get_articulation_controller() for r in robot_list]
    idx_list: list[int] = []
    cmd_plan: List[JointState | None] = [None] * n_envs
    cmd_idx = 0
    prev_primary_goal: Pose | None = None
    past_primary_goal: Pose | None = None
    spheres_per_env: List[list | None] = [None] * n_envs
    step_wait = 0

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

        step_index = my_world.current_time_step_index
        if step_index <= 10:
            for robot in robot_list:
                robot._articulation_view.initialize()
                idx_list = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, idx_list)
                robot._articulation_view.set_max_efforts(
                    values=np.array([5000 for _ in range(len(idx_list))]),
                    joint_indices=idx_list,
                )
        if step_index < 20:
            continue

        # ---- Read every cube's local pose (per env, per tool frame) --------
        # local_pose_per_env[env_id][frame] = (pos_np3, quat_np4)
        local_pose_per_env: List[Dict[str, tuple[np.ndarray, np.ndarray]]] = []
        for env_id in range(n_envs):
            per_env: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for frame in planner.tool_frames:
                p, q = target_cubes[env_id][frame].get_local_pose()
                per_env[frame] = (np.asarray(p), np.asarray(q))
            local_pose_per_env.append(per_env)

        # ---- Build (B, dof) current_state from all robots -----------------
        sim_js_names = robot_list[0].dof_names
        positions_np = []
        velocities_np = []
        ok = True
        for r in robot_list:
            s = r.get_joints_state()
            if s is None:
                ok = False
                break
            positions_np.append(s.positions)
            velocities_np.append(s.velocities)
        if not ok:
            continue

        positions = torch.as_tensor(np.stack(positions_np), device=device, dtype=dtype)
        velocities = torch.as_tensor(np.stack(velocities_np), device=device, dtype=dtype)
        full_js = JointState(
            position=positions,
            velocity=velocities * 0.0,
            acceleration=torch.zeros_like(positions),
            jerk=torch.zeros_like(positions),
            joint_names=sim_js_names,
        )

        if args.visualize_spheres and step_index % 2 == 0:
            active_js = full_js.reorder(planner.kinematics.joint_names)
            sph_list = planner.kinematics.get_robot_as_spheres(active_js.position)
            for env_idx, env_spheres in enumerate(sph_list):
                if spheres_per_env[env_idx] is None:
                    spheres_per_env[env_idx] = []
                    for si, s in enumerate(env_spheres):
                        spheres_per_env[env_idx].append(
                            sphere.VisualSphere(
                                prim_path=f"/World/world_{env_idx}/curobo_sphere_{si}",
                                position=np.ravel(s.position),
                                radius=float(s.radius),
                                color=np.array([0, 0.8, 0.2]),
                            )
                        )
                else:
                    for si, s in enumerate(env_spheres):
                        if not np.isnan(s.position[0]):
                            spheres_per_env[env_idx][si].set_world_pose(
                                position=np.ravel(s.position)
                            )
                            spheres_per_env[env_idx][si].set_radius(float(s.radius))

        # ---- Build goal_tool_poses: (B, 1, L, 1, 3/4) ---------------------
        # For each tool frame, stack per-env (local) poses into one Pose of
        # shape (B, 3) / (B, 4); then GoalToolPose.from_poses packs the L
        # per-frame Poses into the 5-D tensor.
        pose_dict: Dict[str, Pose] = {}
        for frame in planner.tool_frames:
            pos_stack = np.stack([local_pose_per_env[e][frame][0] for e in range(n_envs)])
            quat_stack = np.stack([local_pose_per_env[e][frame][1] for e in range(n_envs)])
            pose_dict[frame] = Pose(
                position=torch.as_tensor(pos_stack, device=device, dtype=dtype),
                quaternion=torch.as_tensor(quat_stack, device=device, dtype=dtype),
            )

        ik_goal_primary = pose_dict[primary_frame]            # (B, 3) / (B, 4)
        if prev_primary_goal is None:
            prev_primary_goal = ik_goal_primary.clone()
        if past_primary_goal is None:
            past_primary_goal = ik_goal_primary.clone()

        # ---- Replan trigger -----------------------------------------------
        # Track the primary cube per env (matches the single-env
        # multi_arm_reacher trigger pattern). Fire when any env's primary cube
        # moved since the last plan, all envs' primary cubes are settled since
        # the previous tick, all robots are static, and there is nothing
        # pending in any env's cmd_plan.
        prev_d = ik_goal_primary.distance(prev_primary_goal)
        past_d = ik_goal_primary.distance(past_primary_goal)
        any_moved = bool((prev_d[0] > 1e-2).any() or (prev_d[1] > 1e-2).any())
        all_settled = bool((past_d[0] == 0.0).all() and (past_d[1] == 0.0).all())
        all_static = bool(full_js.velocity.abs().max() < 0.2)
        no_pending = all(c is None for c in cmd_plan)

        if any_moved and all_settled and all_static and no_pending:
            print(f"[plan] triggered at step {step_index}")
            full_js_active = full_js.reorder(planner.kinematics.joint_names)
            goal_tool_poses = GoalToolPose.from_poses(
                pose_dict,
                ordered_tool_frames=planner.tool_frames,
                num_goalset=1,
            )                                                 # (B, 1, L, 1, 3/4)

            result = planner.plan_pose(
                goal_tool_poses,
                full_js_active,
                max_attempts=2,
                enable_graph_attempt=0,
            )
            prev_primary_goal.copy_(ik_goal_primary)

            if result is not None and bool(result.success.any().item()):
                interp = result.interpolated_trajectory       # (B, 1, max_H, dof_full)
                last = result.interpolated_last_tstep         # (B, 1)
                status = getattr(result, "status", None)
                for s in range(result.success.shape[0]):
                    if not bool(result.success[s].any().item()):
                        reason = status if status else "planner returned success=False"
                        print(
                            f"[plan] env_{s} failed — {reason} "
                            f"(likely start/goal self-collision or unreachable IK)"
                        )
                        cmd_plan[s] = None
                        continue
                    env_traj = interp[s].squeeze(0)           # (max_H, dof_full)
                    env_traj = trim_joint_state_trajectory(
                        env_traj, 0, int(last[s].item())
                    )
                    common_js_names: list[str] = []
                    idx_list = []
                    for x in sim_js_names:
                        if x in env_traj.joint_names:
                            idx_list.append(robot_list[s].get_dof_index(x))
                            common_js_names.append(x)
                    cmd_plan[s] = env_traj.reorder(common_js_names)
                cmd_idx = 0
            else:
                status = getattr(result, "status", "no solution")
                print(f"[plan] batch plan failed: {status}")

        # ---- Step each env's trajectory one waypoint ----------------------
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
                joint_indices=idx_list,
            )
            art_controllers[s].apply_action(art_action)
        cmd_idx += 1

        past_primary_goal.copy_(ik_goal_primary)

        for _ in range(2):
            my_world.step(render=False)

    simulation_app.close()


if __name__ == "__main__":
    main()
