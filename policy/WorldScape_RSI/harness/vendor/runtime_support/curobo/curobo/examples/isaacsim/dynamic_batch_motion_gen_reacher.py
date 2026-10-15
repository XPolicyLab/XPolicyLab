# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuRobo 2.0 port of
``MagicCurobo/examples/isaac_sim/dynamic_batch_motion_gen_reacher.py``.

Variable-subset batch-env planning: four Franka Panda robots live in four
different collision scenes. On a rotating schedule (every ~1000 sim steps)
the script switches which envs are being re-planned — 3 envs, then 2 envs,
then all 4 — while continuing to execute any in-flight trajectories for the
envs that were already running.

**Key v1 → v2 difference (important!).** v1's ``MotionGen.plan_batch_env``
accepts a runtime batch size smaller than ``n_collision_envs``. v2's
:class:`~curobo.batch_motion_planner.BatchMotionPlanner` requires the input
``JointState`` / ``GoalToolPose`` to match ``max_batch_size`` exactly; passing
a smaller batch raises a shape error. So we wrap the planner in
:class:`~curobo.examples.isaacsim.batch_planning_utils.VariableBatchPlanner`,
which restores v1-style call-site ergonomics:

- The main loop builds ``current_state`` and ``GoalToolPose`` at the actual
  ``B`` (size of the current cycle, 1…max_batch_size).
- The wrapper pads to ``max_batch_size`` internally by reusing an empty
  collision scene + the retract joint state + retract FK goal for each pad
  slot — a ``start == goal`` problem the optimizer solves in one iteration.
- The wrapper slices the result back to ``B`` so the caller sees a native
  ``B`` result (``success.shape == (B, 1)`` etc.).

The scene-swap mechanism for the *real* slots is unchanged: when the cycle
rotates, we reload those slots via
:meth:`SceneCollision.load_collision_model`.

v1 → v2 call map (see ``MIGRATION_V1_TO_V2.md`` for the full table):
- ``MotionGenConfig(..., n_collision_envs=max_batch_size, ...)`` →
  ``MotionPlannerCfg.create(..., max_batch_size=N, multi_env=True, max_goalset=1, ...)``.
- ``motion_gen.plan_batch_env(full_js, ik_goal, plan_config)`` →
  ``planner.plan_pose(goal_tool_poses, full_js, max_attempts=..., enable_graph_attempt=0)``.
- ``motion_gen.world_coll_checker.load_collision_model(world_cfg, env_idx=k, fix_cache_reference=False)`` →
  ``planner.scene_collision_checker.load_collision_model(Scene.create(dict), env_idx=k)``.
- ``result.get_paths()`` → iterate ``result.success`` + slice
  ``result.interpolated_trajectory[slot]`` + trim via ``trim_joint_state_trajectory``.
- ``UsdHelper.add_subroot`` — **not used here** (leading-``/`` quirk in v2's
  ``join_usd_path`` places the Xform at the wrong path); we create the per-env
  root Xform directly with ``UsdGeom.Xform.Define`` instead.

``use_cuda_graph`` is off because per-slot scene reloads invalidate cached
CUDA graphs (same caveat as v1).

Run (inside the Isaac Sim Python env):

    python -m curobo.examples.isaacsim.dynamic_batch_motion_gen_reacher
"""
from __future__ import annotations

# ---- Isaac Sim bootstrap (must come before any curobo / pxr imports) -------
from curobo.examples.isaacsim import bootstrap  # noqa: F401

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import argparse
import time

import torch

_ = torch.zeros(4, device="cuda:0")

parser = argparse.ArgumentParser(description="cuRobo v2 dynamic batch-env motion planner")
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket, webrtc].",
)
parser.add_argument(
    "--robot",
    type=str,
    default="franka.yml",
    help="cuRobo v2 robot YAML.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="Render each robot's collision spheres under its world root.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Exit after N simulator steps. Useful for smoke-testing in headless mode.",
)
parser.add_argument(
    "--cycle_switch_interval",
    type=int,
    default=1000,
    help="Rotate which env-subset gets re-planned every N sim steps.",
)
args = parser.parse_args()

# ---- SimulationApp -------------------------------------------------------
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# ---- Post-SimulationApp imports ------------------------------------------
import pathlib

import carb
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, UsdGeom

from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.config_io import load_yaml
from curobo.content import get_robot_configs_path
from curobo.examples.isaacsim.batch_planning_utils import VariableBatchPlanner
from curobo.examples.isaacsim.helper import add_extensions, add_robot_to_scene
from curobo.logging import setup_logger
from curobo.motion_planner import MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState, Pose
from curobo.viewer import UsdWriter

_SCENES_DIR = pathlib.Path(__file__).parent / "scenes"

# Planning cycle: rotate among these env subsets (matches v1 exactly).
PLANNING_CYCLE: list[list[int]] = [
    [0, 1, 2],        # 3 envs
    [1, 2],           # 2 envs
    [0, 1, 2, 3],     # 4 envs
]

# Scene YAML per env (v1 pattern: alternating test / thin_walls).
ENV_SCENE_FILES: list[str] = [
    "collision_test.yml",
    "collision_thin_walls.yml",
    "collision_test.yml",
    "collision_thin_walls.yml",
]


def _load_scene_dict(name: str) -> dict:
    path = _SCENES_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"scene YAML not found: {path}")
    return load_yaml(str(path))


def _resolve_robot_cfg(robot: str) -> dict:
    from curobo.config_io import join_path
    return load_yaml(join_path(str(get_robot_configs_path()), robot))["robot_cfg"]


def _nudge_scene_down_2cm(sd: dict) -> None:
    """Replicate v1's ``world_cfg.objects[0].pose[2] -= 0.02`` on the first obstacle."""
    objs = list((sd.get("cuboid") or {}).values()) + list(
        (sd.get("mesh") or {}).values()
    )
    if objs and "pose" in objs[0]:
        objs[0]["pose"][2] -= 0.02


def main() -> None:
    setup_logger("warn")

    n_envs = 4
    max_batch_size = 4
    offset_y = 2.5

    # ---- Scene + stage -----------------------------------------------------
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    stage = my_world.stage
    stage.DefinePrim("/World", "Xform").GetStage().SetDefaultPrim(
        stage.GetPrimAtPath("/World")
    )
    stage.DefinePrim("/curobo", "Xform")

    robot_cfg = _resolve_robot_cfg(args.robot)
    usd_writer = UsdWriter()
    usd_writer.load_stage(stage)

    # ---- Per-env roots, targets, robots -----------------------------------
    target_list = []
    robot_list = []
    env_offsets = []
    for i in range(n_envs):
        env_origin = np.array([0.0, i * offset_y, 0.0], dtype=np.float32)
        env_offsets.append(env_origin)

        # Bypass UsdWriter.add_subroot (join_usd_path strips leading "/" and
        # misplaces the Xform). Create the env root at the absolute path
        # directly and set its translate op; child local poses then match
        # what the planner's per-env collision world expects.
        env_root = UsdGeom.Xform.Define(stage, f"/World/world_{i}")
        if i > 0:
            env_root.AddTranslateOp().Set(Gf.Vec3d(*env_origin.tolist()))

        target_xyz = np.array([0.5, 0, 0.5]) + env_origin
        target = cuboid.VisualCuboid(
            f"/World/world_{i}/target",
            position=target_xyz,
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        target_list.append(target)

        robot, _robot_path = add_robot_to_scene(
            robot_cfg,
            my_world,
            subroot=f"/World/world_{i}/",
            robot_name=f"robot_{i}",
            position=env_origin,
            initialize_world=False,
        )
        robot_list.append(robot)

    my_world.initialize_physics()

    # ---- Per-env scene dicts (loaded once, reused when slotting) ---------
    scene_dicts = [_load_scene_dict(name) for name in ENV_SCENE_FILES[:n_envs]]
    for sd in scene_dicts:
        _nudge_scene_down_2cm(sd)

    # Visualise each env's obstacles under its own root (USD only — planner
    # collision worlds come from the per-slot ``load_collision_model`` below).
    for i, sd in enumerate(scene_dicts):
        usd_writer.add_world_to_stage(Scene.create(sd), base_frame=f"/World/world_{i}")

    # ---- Build the planner -----------------------------------------------
    #
    # We seed the scene list with env 0's scene for every slot; the runtime
    # cycle reloads slots as needed via load_collision_model(..., env_idx=k).
    initial_scenes = [scene_dicts[0]] * max_batch_size
    planner_cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        scene_model=initial_scenes,
        collision_cache={"cuboid": 30, "mesh": 10},
        max_batch_size=max_batch_size,
        multi_env=True,
        max_goalset=1,
        num_trajopt_seeds=12,
        num_ik_seeds=32,
        # Off because we reload per-slot scenes at runtime; CUDA graph capture
        # assumes static tensor sizes and addresses.
        use_cuda_graph=False,
        self_collision_check=True,
        optimizer_collision_activation_distance=0.025,
    )
    planner = BatchMotionPlanner(planner_cfg)

    print("warming up...")
    planner.warmup(enable_graph=False, num_warmup_iterations=3)

    # Wrap the planner with the variable-B adapter. It precomputes an empty
    # scene + retract-state / retract-FK dummy problem for the pad slots so
    # the wrapper can be called with any B in [1, max_batch_size].
    vbp = VariableBatchPlanner(planner)

    print("cuRobo v2 dynamic-batch motion planner is ready")

    add_extensions(simulation_app, args.headless_mode)

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = (
        robot_cfg["kinematics"]["cspace"].get("default_joint_position")
        or robot_cfg["kinematics"]["cspace"].get("retract_config")
    )

    if args.headless_mode is not None:
        my_world.play()

    device = planner.device_cfg.device
    dtype = planner.device_cfg.dtype

    # ---- Runtime state per env --------------------------------------------
    art_controllers = [r.get_articulation_controller() for r in robot_list]
    idx_list: list[int] = []

    prev_goal: list[Pose | None] = [None] * n_envs
    past_goal: list[Pose | None] = [None] * n_envs
    failed_plan_goal: list[Pose | None] = [None] * n_envs
    cmd_plan: list[JointState | None] = [None] * n_envs
    cmd_idx = 0

    cycle_index = 0
    cycle_step_counter = 0
    loaded_cycle_signature: tuple[int, ...] | None = None

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

        # ---- Rotate the cycle -----------------------------------------------
        cycle_step_counter += 1
        batch_switched = False
        if cycle_step_counter >= args.cycle_switch_interval:
            cycle_step_counter = 0
            cycle_index = (cycle_index + 1) % len(PLANNING_CYCLE)
            batch_switched = True
            print(
                f"[cycle] now planning {len(PLANNING_CYCLE[cycle_index])} env(s): "
                f"{PLANNING_CYCLE[cycle_index]}"
            )

        current_batch_env_ids = PLANNING_CYCLE[cycle_index]
        current_batch_size = len(current_batch_env_ids)    # the "real" B
        cycle_signature = tuple(current_batch_env_ids)

        # ---- Per-slot scene reload when the active subset changes ----------
        #
        # We only load the *real* slots 0..current_batch_size-1. The wrapper
        # (VariableBatchPlanner) takes care of the pad slots by pointing them
        # at an empty scene + retract-state / retract-FK goal, so those
        # slots resolve in ~1 iteration and never drag down the real problems.
        if batch_switched or loaded_cycle_signature != cycle_signature:
            print(f"[cycle] reloading {current_batch_size} real scene(s) "
                  f"(pad handled by wrapper)")
            t0 = time.time()
            for slot_idx, env_id in enumerate(current_batch_env_ids):
                planner.scene_collision_checker.load_collision_model(
                    Scene.create(scene_dicts[env_id]), env_idx=slot_idx,
                )
            print(f"[cycle] scene reload: {(time.time() - t0) * 1000:.2f} ms")
            loaded_cycle_signature = cycle_signature

        # ---- Collect per-env local target poses (needed for past/prev tracking)
        local_pos_per_env = []
        local_quat_per_env = []
        for t in target_list:
            p, q = t.get_local_pose()
            local_pos_per_env.append(p)
            local_quat_per_env.append(q)

        # ---- Build ik_goal at *actual* B -----------------------------------
        real_pos = [local_pos_per_env[env_id] for env_id in current_batch_env_ids]
        real_quat = [local_quat_per_env[env_id] for env_id in current_batch_env_ids]
        ik_goal_batch = Pose(
            position=torch.as_tensor(np.stack(real_pos), device=device, dtype=dtype),
            quaternion=torch.as_tensor(np.stack(real_quat), device=device, dtype=dtype),
        )                                                   # (B, 3) / (B, 4)

        # Seed per-env prev/past goals on first visit.
        for env_id in range(n_envs):
            env_goal = Pose(
                position=torch.as_tensor(local_pos_per_env[env_id], device=device, dtype=dtype).view(1, 3),
                quaternion=torch.as_tensor(local_quat_per_env[env_id], device=device, dtype=dtype).view(1, 4),
            )
            if prev_goal[env_id] is None:
                prev_goal[env_id] = env_goal.clone()
            if past_goal[env_id] is None:
                past_goal[env_id] = env_goal.clone()

        # ---- Build current_state at *actual* B -----------------------------
        sim_js_names = robot_list[0].dof_names
        pos_per_env = []
        vel_per_env = []
        ok = True
        for r in robot_list:
            s = r.get_joints_state()
            if s is None:
                ok = False
                break
            pos_per_env.append(s.positions)
            vel_per_env.append(s.velocities)
        if not ok:
            continue

        real_positions = np.stack([pos_per_env[env_id] for env_id in current_batch_env_ids])
        real_velocities = np.stack([vel_per_env[env_id] for env_id in current_batch_env_ids])
        positions_t = torch.as_tensor(real_positions, device=device, dtype=dtype)
        velocities_t = torch.as_tensor(real_velocities, device=device, dtype=dtype) * 0.0
        full_js_batch = JointState(
            position=positions_t,
            velocity=velocities_t,
            acceleration=torch.zeros_like(positions_t),
            jerk=torch.zeros_like(positions_t),
            joint_names=sim_js_names,
        )                                                   # (B, dof)

        # ---- Decide whether to replan -------------------------------------
        #
        # Single-env motion_gen_reacher trigger, replicated per active env:
        #   target moved since the last plan  (prev_goal)
        #   AND cube settled since the last tick  (past_goal)
        #   AND this env's robot is not moving.
        # If ANY active env is ready, we replan for the whole active cycle
        # (v2 requires a uniform batch shape; the envs whose targets didn't
        #  move just get a plan that re-solves to the same goal — cheap).
        need_replan = False
        for slot_idx in range(current_batch_size):
            env_id = current_batch_env_ids[slot_idx]
            env_goal = ik_goal_batch[slot_idx]             # Pose (1, 3/4)

            if failed_plan_goal[env_id] is not None:
                d = env_goal.distance(failed_plan_goal[env_id])
                if d[0] >= 1e-2 or d[1] >= 1e-2:
                    failed_plan_goal[env_id] = None
                else:
                    # Same failed goal, don't retry until user drags it again.
                    continue

            prev_d = env_goal.distance(prev_goal[env_id])
            past_d = env_goal.distance(past_goal[env_id])
            slot_vel = full_js_batch.velocity[slot_idx].abs().max()
            moved = bool((prev_d[0] > 1e-2 or prev_d[1] > 1e-2))
            settled = bool((past_d[0] == 0.0 and past_d[1] == 0.0))
            static = bool(slot_vel < 0.2)
            if moved and settled and static:
                need_replan = True
                break

        if need_replan:
            full_js_active = full_js_batch.reorder(planner.kinematics.joint_names)

            goal_tool_poses = GoalToolPose.from_poses(
                {planner.tool_frames[0]: ik_goal_batch},
                ordered_tool_frames=planner.tool_frames,
                num_goalset=1,
            )                                               # (B, 1, L=1, G=1, 3/4)

            t0 = time.time()
            # The wrapper accepts any B ≤ max_batch_size; pad slots get the
            # precomputed empty-scene + retract-FK dummy problem.
            result = vbp.plan_pose(
                goal_tool_poses,
                full_js_active,
                max_attempts=2,
                enable_graph_attempt=0,
            )
            print(f"[plan] B={current_batch_size}/{max_batch_size} took "
                  f"{(time.time() - t0) * 1000:.1f} ms")

            for slot_idx in range(current_batch_size):
                env_id = current_batch_env_ids[slot_idx]
                prev_goal[env_id].copy_(ik_goal_batch[slot_idx])

            if result is not None and bool(result.success.any().item()):
                interp = result.interpolated_trajectory     # (B, 1, max_H, dof_full)
                last = result.interpolated_last_tstep       # (B, 1)
                status = getattr(result, "status", None)
                for slot_idx in range(current_batch_size):
                    env_id = current_batch_env_ids[slot_idx]
                    if not bool(result.success[slot_idx].any().item()):
                        reason = status if status else "planner returned success=False"
                        print(
                            f"[plan] env {env_id} failed — {reason} "
                            f"(likely start/goal self-collision or unreachable IK)"
                        )
                        if failed_plan_goal[env_id] is None:
                            failed_plan_goal[env_id] = ik_goal_batch[slot_idx].clone()
                        continue

                    env_traj = interp[slot_idx].squeeze(0)   # (max_H, dof_full)
                    env_traj = trim_joint_state_trajectory(
                        env_traj, 0, int(last[slot_idx].item())
                    )                                        # (valid_H, dof_full)

                    common_js_names: list[str] = []
                    idx_list = []
                    for x in sim_js_names:
                        if x in env_traj.joint_names:
                            idx_list.append(robot_list[env_id].get_dof_index(x))
                            common_js_names.append(x)
                    cmd_plan[env_id] = env_traj.reorder(common_js_names)
                    failed_plan_goal[env_id] = None
                cmd_idx = 0
            else:
                status = getattr(result, "status", "no solution")
                carb.log_warn(f"Batch plan failed for cycle {current_batch_env_ids}: {status}")
                for slot_idx in range(current_batch_size):
                    env_id = current_batch_env_ids[slot_idx]
                    if failed_plan_goal[env_id] is None:
                        failed_plan_goal[env_id] = ik_goal_batch[slot_idx].clone()

        # ---- Step every env's in-flight plan (not only cycle envs) --------
        for env_id in range(n_envs):
            plan = cmd_plan[env_id]
            if plan is None:
                continue
            if cmd_idx >= len(plan.position):
                cmd_plan[env_id] = None
                continue
            cmd_state = plan[cmd_idx]
            # idx_list was set either in the init block or after a plan-success;
            # for an identical-robot batch the dof ordering is the same per env,
            # so we can reuse it. Recompute per-env to be safe.
            env_idx_list = [
                robot_list[env_id].get_dof_index(x) for x in plan.joint_names
            ]
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy() if cmd_state.velocity is not None else None,
                joint_indices=env_idx_list,
            )
            art_controllers[env_id].apply_action(art_action)
        cmd_idx += 1

        # Update past_goal for ALL envs every tick — even non-active ones —
        # so when an env comes back into the active cycle, its "settled
        # since last tick" check compares against a fresh observation, not
        # one from many cycles ago.
        for env_id in range(n_envs):
            past_goal[env_id].position.copy_(
                torch.as_tensor(local_pos_per_env[env_id], device=device, dtype=dtype).view(1, 3)
            )
            past_goal[env_id].quaternion.copy_(
                torch.as_tensor(local_quat_per_env[env_id], device=device, dtype=dtype).view(1, 4)
            )

        for _ in range(2):
            my_world.step(render=False)

    simulation_app.close()


if __name__ == "__main__":
    main()
