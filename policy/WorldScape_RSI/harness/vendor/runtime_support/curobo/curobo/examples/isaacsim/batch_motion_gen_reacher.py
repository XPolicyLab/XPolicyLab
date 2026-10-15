# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuRobo 2.0 port of ``MagicCurobo/examples/isaac_sim/batch_motion_gen_reacher.py``.

Batch-env motion planning: two Franka Panda robots at separate world offsets,
each with its **own** collision scene. A single call to
:meth:`curobo.batch_motion_planner.BatchMotionPlanner.plan_pose` solves both
problems in parallel; the per-env trajectories are then fed back to the two
robot articulations.

v1 → v2 delta (see ``MIGRATION_V1_TO_V2.md``):
- ``MotionGen / MotionGenConfig / MotionGenPlanConfig`` → ``BatchMotionPlanner / MotionPlannerCfg``.
- ``MotionGenConfig.load_from_robot_config(robot, [world0, world1], ...)`` →
  ``MotionPlannerCfg.create(robot=..., scene_model=[dict0, dict1],
  max_batch_size=N, multi_env=True, ...)``.
- ``motion_gen.plan_batch_env(full_js, ik_goal, plan_config)`` →
  ``planner.plan_pose(GoalToolPose.from_poses({link: stacked_pose}, num_goalset=1),
  full_js, max_attempts=…, enable_graph_attempt=0)``.
- ``result.get_paths()`` (v1 helper that returns ``List[JointState]``) has no
  v2 equivalent; iterate ``result.success[s]`` + slice
  ``result.interpolated_trajectory[s]`` + trim via
  :func:`trim_joint_state_trajectory` and ``result.interpolated_last_tstep``.
- ``UsdHelper`` (v1) → ``curobo.viewer.UsdWriter`` (public v2 re-export).

Scene YAMLs for the two envs live next to this script
(``scenes/collision_test.yml``, ``scenes/collision_thin_walls.yml``) — copied
verbatim from ``MagicCurobo/src/curobo/content/configs/world/`` so the two
collision worlds match the v1 example exactly.

Run (inside the Isaac Sim Python env):

    python -m curobo.examples.isaacsim.batch_motion_gen_reacher

Drag each red target cuboid; when both stop moving the planner solves both
problems in one batch and replays the two trajectories simultaneously.
"""
from __future__ import annotations

# ---- Isaac Sim bootstrap (must come before any curobo / pxr imports) -------
from curobo.examples.isaacsim import bootstrap  # noqa: F401

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import argparse

import torch

_ = torch.zeros(4, device="cuda:0")

parser = argparse.ArgumentParser(description="cuRobo v2 batch-env motion planner reacher")
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket, webrtc].",
)
parser.add_argument(
    "--robot",
    type=str,
    default="magicsim_franka_umi.yml",
    help="cuRobo v2 robot YAML (under content/configs/robot/ or absolute path).",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="Render each robot's collision spheres in its sub-root.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Exit after N simulator steps. Useful for smoke-testing in headless mode.",
)
args = parser.parse_args()

# ---- SimulationApp has to be created before any pxr / omni.isaac.core -----
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# ---- Imports that touch pxr / omni.isaac.* go AFTER SimulationApp ---------
import pathlib

import carb
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, UsdGeom

from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.config_io import load_yaml
from curobo.content import get_robot_configs_path
from curobo.examples.isaacsim.helper import (
    add_extensions,
    add_robot_to_scene,
)
from curobo.logging import log_warn, setup_logger
from curobo.motion_planner import MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState, Pose
from curobo.viewer import UsdWriter

_SCENES_DIR = pathlib.Path(__file__).parent / "scenes"


def _load_scene_dict(name: str) -> dict:
    """Load a scene YAML from our local ``scenes/`` dir (not cuRobo's content)."""
    path = _SCENES_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"scene YAML not found: {path}")
    return load_yaml(str(path))


def _resolve_robot_cfg(robot: str) -> dict:
    from curobo.config_io import join_path

    return load_yaml(join_path(str(get_robot_configs_path()), robot))["robot_cfg"]


def main() -> None:
    setup_logger("warn")

    n_envs = 2
    scene_files = ["collision_test.yml", "collision_thin_walls.yml"]
    offset_y = 2.5

    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    stage = my_world.stage
    stage.DefinePrim("/World", "Xform").GetStage().SetDefaultPrim(
        stage.GetPrimAtPath("/World")
    )
    stage.DefinePrim("/curobo", "Xform")

    # ---- Build per-env sub-roots, targets and robots ------------------------
    robot_cfg = _resolve_robot_cfg(args.robot)
    usd_writer = UsdWriter()
    usd_writer.load_stage(stage)

    target_list = []
    robot_list = []
    env_offsets = []                                 # world-space (x,y,z) per env
    for i in range(n_envs):
        env_origin = np.array([0.0, i * offset_y, 0.0], dtype=np.float32)
        env_offsets.append(env_origin)

        # Create the per-env root Xform directly. NOTE: ``UsdWriter.add_subroot``
        # cannot be used here because it calls ``join_usd_path(root, sub_root)``
        # which strips the leading ``/`` from ``sub_root`` (USD path convention),
        # so ``add_subroot("/World", "/World/world_0", pose)`` would actually
        # create ``/World/World/world_0``. We want ``/World/world_i`` to be a
        # real Xform at ``env_origin`` so child local poses correspond to
        # env-local coords (which is what the planner's collision world and
        # goal pose both operate in).
        env_root = UsdGeom.Xform.Define(stage, f"/World/world_{i}")
        if i > 0:
            env_root.AddTranslateOp().Set(Gf.Vec3d(*env_origin.tolist()))

        target_xyz = np.array([0.5, 0, 0.5]) + env_origin
        target = cuboid.VisualCuboid(
            f"/World/world_{i}/target",
            position=target_xyz,                      # world pose; parent Xform makes local = (0.5, 0, 0.5)
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        target_list.append(target)

        # Isaac Sim 5.1's URDF importer places the robot under /World/<name>
        # regardless of the ``subroot`` argument, but ``position`` correctly
        # transforms the robot root to world coords.
        robot, _robot_path = add_robot_to_scene(
            robot_cfg,
            my_world,
            subroot=f"/World/world_{i}/",             # retained for parity; ignored by the 5.1 URDF path
            robot_name=f"robot_{i}",
            position=env_origin,
            initialize_world=False,
        )
        robot_list.append(robot)

    my_world.initialize_physics()

    # ---- Per-env scene configs (v1 world_cfg_list → v2 list of dicts) -----
    scene_dicts = [_load_scene_dict(name) for name in scene_files]
    # v1 nudges each world's first object 2cm down "to avoid ground" and
    # randomizes colour; we keep the pose nudge for behavioural parity but
    # skip the colour step (purely cosmetic, not in v2's SceneCfg API).
    for sd in scene_dicts:
        objs = list((sd.get("cuboid") or {}).values()) + list(
            (sd.get("mesh") or {}).values()
        )
        if objs and "pose" in objs[0]:
            objs[0]["pose"][2] -= 0.02

    # Render each env's obstacles in the stage under its sub-root so they
    # match what the planner sees (v1 did the same via usd_help.add_world_to_stage).
    for i, sd in enumerate(scene_dicts):
        usd_writer.add_world_to_stage(
            Scene.create(sd), base_frame=f"/World/world_{i}",
        )

    # ---- Build the v2 batch motion planner (batch_env mode) -----------------
    planner_cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        scene_model=scene_dicts,               # list → per-env collision worlds
        collision_cache={"cuboid": 30, "mesh": 10},
        max_batch_size=n_envs,
        multi_env=True,                        # key flag: solve_batch_env mode
        max_goalset=1,
        num_trajopt_seeds=12,
        num_ik_seeds=32,
        use_cuda_graph=True,
        self_collision_check=True,
        optimizer_collision_activation_distance=0.025,
    )
    planner = BatchMotionPlanner(planner_cfg)

    print("warming up...")
    # BatchMotionPlanner.warmup disables graph seeding when multi_env=True.
    planner.warmup(enable_graph=False, num_warmup_iterations=3)
    print("cuRobo v2 batch motion planner is ready")

    add_extensions(simulation_app, args.headless_mode)

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = (
        robot_cfg["kinematics"]["cspace"].get("default_joint_position")
        or robot_cfg["kinematics"]["cspace"].get("retract_config")
    )

    # In headless mode there is no UI Play button; start the sim ourselves.
    if args.headless_mode is not None:
        my_world.play()

    art_controllers = [r.get_articulation_controller() for r in robot_list]
    idx_list: list[int] = []

    cmd_plan: list[JointState | None] = [None] * n_envs
    cmd_idx = 0
    prev_goal: Pose | None = None
    past_goal: Pose | None = None
    spheres_per_env: list[list | None] = [None] * n_envs
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

        # ---- Collect per-env target poses (robot-local, via get_local_pose) ----
        sp_buffer = []
        sq_buffer = []
        for t in target_list:
            local_pos, local_quat = t.get_local_pose()
            sp_buffer.append(local_pos)
            sq_buffer.append(local_quat)
        ik_goal = Pose(
            position=torch.as_tensor(np.stack(sp_buffer), device=device, dtype=dtype),
            quaternion=torch.as_tensor(np.stack(sq_buffer), device=device, dtype=dtype),
        )                                                # (B, 3) / (B, 4)

        if prev_goal is None:
            prev_goal = ik_goal.clone()
        if past_goal is None:
            past_goal = ik_goal.clone()

        # ---- Build batched current_state (B, dof) -----------------------------
        #
        # v1 did ``full_js.stack(cu_js)`` per-env, but v2 deprecated
        # ``JointState.stack`` in favour of ``stack_joint_states``. Cheapest
        # path is to just collect the per-env numpy tensors and build one
        # ``JointState`` with a ``(B, dof)`` position/velocity tensor.
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

        positions = torch.as_tensor(np.stack(positions_np), device=device, dtype=dtype)   # (B, dof)
        velocities = torch.as_tensor(np.stack(velocities_np), device=device, dtype=dtype) # (B, dof)
        full_js = JointState(
            position=positions,
            velocity=velocities * 0.0,
            acceleration=torch.zeros_like(positions),
            jerk=torch.zeros_like(positions),
            joint_names=sim_js_names,
        )

        if args.visualize_spheres and step_index % 2 == 0:
            # Render one sphere cloud per env under /World/world_i/curobo_spheres
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

        # ---- Decide whether to plan this frame --------------------------------
        # Replicate v1 trigger: any env's target moved since prev, no env's
        # target moved since past (i.e. target settled this tick), all robots
        # essentially static, and no pending trajectory.
        prev_distance = ik_goal.distance(prev_goal)       # [pos_err, rot_err] per env
        past_distance = ik_goal.distance(past_goal)
        any_moved = bool((prev_distance[0] > 1e-2).any() or (prev_distance[1] > 1e-2).any())
        all_settled = bool((past_distance[0] == 0.0).all() and (past_distance[1] == 0.0).all())
        all_static = bool(full_js.velocity.abs().max() < 0.2)
        no_pending = all(c is None for c in cmd_plan)

        if any_moved and all_settled and all_static and no_pending:
            full_js_active = full_js.reorder(planner.kinematics.joint_names)

            goal_tool_poses = GoalToolPose.from_poses(
                {planner.tool_frames[0]: ik_goal},
                ordered_tool_frames=planner.tool_frames,
                num_goalset=1,
            )                                            # (B, 1, L=1, G=1, 3/4)

            result = planner.plan_pose(
                goal_tool_poses,
                full_js_active,
                max_attempts=2,
                # graph seeding is skipped when multi_env=True, so this is a no-op.
                enable_graph_attempt=0,
            )
            prev_goal.copy_(ik_goal)

            if result is not None and bool(result.success.any().item()):
                # v1: result.get_paths()  →  v2: iterate per-env slices manually.
                # interpolated_trajectory.position: (B, 1, max_H, dof_full).
                interp = result.interpolated_trajectory
                last = result.interpolated_last_tstep     # (B, 1)
                status = getattr(result, "status", None)
                for s in range(result.success.shape[0]):
                    if not bool(result.success[s].any().item()):
                        # Per-env failure must not be silent — the batch
                        # overall succeeded (some env did), so the outer
                        # ``else`` branch wouldn't fire. Log why this
                        # specific env failed so the user isn't staring at
                        # a stationary robot with no output.
                        reason = status if status else "planner returned success=False"
                        carb.log_warn(
                            f"env_{s} plan failed — {reason} "
                            f"(likely start/goal self-collision or unreachable IK)"
                        )
                        cmd_plan[s] = None
                        continue
                    env_traj = interp[s].squeeze(0)       # (max_H, dof_full)
                    last_t = int(last[s].item())
                    env_traj = trim_joint_state_trajectory(
                        env_traj, 0, last_t
                    )                                     # (valid_H, dof_full)
                    if env_traj.position.shape[0] == 0:
                        # success=True but 0-waypoint trajectory → start ≈ goal.
                        # Robot will sit still. Warn so the user isn't left
                        # guessing why this env didn't animate.
                        carb.log_warn(
                            f"env_{s} plan returned 0 waypoints (last_tstep={last_t}) — "
                            f"start pose ≈ goal pose, robot stays put"
                        )
                        cmd_plan[s] = None
                        continue

                    # Reorder to sim dof indexing (and cache idx_list for both envs
                    # — assuming identical robot type across envs).
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
                carb.log_warn(f"Batch plan failed: {status}")

        # ---- Step each env's trajectory one waypoint -------------------------
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

        past_goal.copy_(ik_goal)

        for _ in range(2):
            my_world.step(render=False)

    simulation_app.close()


if __name__ == "__main__":
    main()
