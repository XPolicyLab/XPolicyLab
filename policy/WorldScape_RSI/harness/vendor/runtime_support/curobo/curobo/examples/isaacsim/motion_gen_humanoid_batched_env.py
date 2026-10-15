# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuRobo 2.0 dual-arm humanoid motion planning — ``plan_batch_env`` with two tool frames.

Same scene + robot setup as :mod:`ik_humanoid_batched_env` but swaps the
one-shot IK solve for a full trajectory optimization via
:class:`curobo.batch_motion_planner.BatchMotionPlanner`.  Each env gets its
own interpolated joint trajectory that is played back frame-by-frame in
Isaac Sim.

Variant axes used:

* ``multi_env=True``   →  per-problem collision world (N envs)
* ``L > 1``            →  multi-tool-frame goal (right + left palms)
* trajopt (plan)       →  returns a time-parameterized path, not a single pose

v1 → v2 method map (see ``MIGRATION_V1_TO_V2.md``):

+----------------------------------------------+-----------------------------------------------------------+
| v1 ``MotionGen.plan_batch_env(...)``         | v2 ``BatchMotionPlanner.plan_pose(GoalToolPose, ...)``    |
+----------------------------------------------+-----------------------------------------------------------+
| v1 ``link_poses: Dict[str, List[Pose]]``     | folded into the single ``GoalToolPose (B, 1, L, 1, 3/4)`` |
+----------------------------------------------+-----------------------------------------------------------+
| v1 ``result.get_paths()``                    | iterate ``result.interpolated_trajectory.position[env]``  |
|                                              | (trimmed by ``result.interpolated_last_tstep[env]``)      |
+----------------------------------------------+-----------------------------------------------------------+

Robot config is identical to :mod:`ik_humanoid_batched_env` — both
``magicsim_g1_simple.yml`` (fixed base, 17 DOF) and
``magicsim_g1_simple_mobile.yml`` (floating base, 21 DOF) ship
``tool_frames: [right_hand_palm_link, left_hand_palm_link]`` in that order.

Usage::

    python -m curobo.examples.isaacsim.motion_gen_humanoid_batched_env
    python -m curobo.examples.isaacsim.motion_gen_humanoid_batched_env --mobile
    python -m curobo.examples.isaacsim.motion_gen_humanoid_batched_env --n_envs 2 --visualize_spheres
"""

# Pin pip's warp-lang 1.12 BEFORE any ``omni.*`` / ``isaacsim.*`` /
# ``curobo._src`` import. See ``bootstrap.py`` for why.
from curobo.examples.isaacsim import bootstrap  # noqa: F401

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

from typing import Dict, List, Optional

import torch

_ = torch.zeros(4, device="cuda:0")

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket].",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="Visualize robot collision spheres as USD VisualSpheres.",
    default=False,
)
parser.add_argument(
    "--n_envs",
    type=int,
    default=2,
    help="Number of parallel G1 environments.",
)
parser.add_argument(
    "--mobile",
    action="store_true",
    help=("Use the floating-base variant (``magicsim_g1_simple_mobile.yml``). "
          "Default is the fixed-base variant (``magicsim_g1_simple.yml``)."),
    default=False,
)
parser.add_argument(
    "--exec_skip",
    type=int,
    default=1,
    help=("Play back every Nth interpolated waypoint (1 = all). Bump to "
          "slow execution down if the interpolated trajectory is too dense."),
)
parser.add_argument(
    "--load_from_usd",
    action="store_true",
    help=("Reference a pre-built robot USD (``kinematics.usd_path``) instead of "
          "running the Isaac Sim URDF importer. Faster startup, but requires "
          "the USD file to already exist under the asset tree."),
    default=False,
)
args = parser.parse_args()


from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

import carb  # noqa: E402
import numpy as np  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import cuboid, sphere  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from curobo.batch_motion_planner import BatchMotionPlanner  # noqa: E402
from curobo.config_io import load_yaml  # noqa: E402
from curobo.content import get_robot_configs_path  # noqa: E402
from curobo.logging import setup_logger  # noqa: E402
from curobo.motion_planner import MotionPlannerCfg  # noqa: E402
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose  # noqa: E402

from curobo.examples.isaacsim.helper import (  # noqa: E402
    add_extensions,
    add_robot_to_scene,
)


############################################################
# Config knobs
############################################################


ENV_OFFSET_Y: float = 2.5  # world-frame spacing between the two G1 bases


############################################################
# Helpers
############################################################


def _expand_joint_state(js: JointState, batch_size: int) -> JointState:
    """Broadcast a ``(1, dof)`` JointState to ``(batch_size, dof)`` (no warning)."""
    if batch_size == 1:
        return js
    return JointState.from_position(
        js.position.expand(batch_size, -1).contiguous(),
        joint_names=js.joint_names,
    )


def _build_goal_tool_pose(
    planner: BatchMotionPlanner,
    link_positions: Dict[str, torch.Tensor],
    link_quaternions: Dict[str, torch.Tensor],
) -> GoalToolPose:
    """Wrap per-link ``(B, 3)`` / ``(B, 4)`` tensors into a ``(B, 1, L, 1, …)`` goal."""
    poses = {
        name: Pose(position=link_positions[name], quaternion=link_quaternions[name])
        for name in planner.tool_frames
    }
    return GoalToolPose.from_poses(
        poses,
        ordered_tool_frames=planner.tool_frames,
        num_goalset=1,
    )


def _extract_per_env_trajectory(
    result,
    env_idx: int,
) -> Optional[JointState]:
    """Pull env ``env_idx``'s interpolated trajectory out of a batched result.

    :meth:`TrajOptSolverResult.get_interpolated_plan` only handles a single
    problem; for batched results the tensor layout is
    ``interpolated_trajectory.position : (B, num_seeds, H, dof)`` and
    ``interpolated_last_tstep : (B, num_seeds)``.  We slice env ``env_idx``,
    take seed 0 (the solver has already ranked seeds so seed 0 is the best),
    and trim to the reported ``last_tstep``.
    """
    interp = result.interpolated_trajectory
    if interp is None:
        return None
    pos = interp.position
    # Normalize position to (B, num_seeds, H, dof): some code paths return
    # (B, H, dof) when num_seeds == 1, handle both.
    if pos.ndim == 3:
        pos = pos.unsqueeze(1)
    horizon = pos.shape[2]

    last_steps = result.interpolated_last_tstep
    if last_steps is not None:
        ls = last_steps
        # Shapes seen in practice: (B,), (B, 1), (B, num_seeds).  Pick env, seed 0.
        ls_env = ls[env_idx]
        if torch.is_tensor(ls_env) and ls_env.ndim >= 1:
            ls_env = ls_env.flatten()[0]
        last = int(ls_env.item())
    else:
        last = horizon
    last = max(1, min(horizon, last))

    return JointState.from_position(
        pos[env_idx, 0, :last, :].clone(),
        joint_names=interp.joint_names,
    )


############################################################
# Main
############################################################


def main():
    n_envs = max(1, int(args.n_envs))
    device_cfg = DeviceCfg()

    my_world = World(stage_units_in_meters=1.0)
    # No default ground plane — matches v1's ``multi_arm_humanoid_parallel_ik.py``
    # behaviour, keeps the free-floating legs from fighting a physics collider.
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    setup_logger("warn")

    # ------------------------------------------------------------------
    # Robot config: pick fixed-base or floating-base G1 variant.  Both ship
    # with tool_frames = [right_hand_palm_link, left_hand_palm_link] (right
    # first) and lock_joints pinning every non-arm DOF at its default.
    # ------------------------------------------------------------------
    robot_file = (
        "magicsim_g1_simple_mobile.yml" if args.mobile else "magicsim_g1_simple.yml"
    )
    robot_yaml = load_yaml(str(get_robot_configs_path() / robot_file))
    # Isaac Sim-only fields live at the YAML top level (peers of
    # ``kinematics:``); pop them out before the solver ever sees them.
    usd_path = robot_yaml.pop("usd_path", None)
    usd_robot_root = robot_yaml.pop("usd_robot_root", None)
    if "robot_cfg" not in robot_yaml:
        robot_cfg_full = {"robot_cfg": robot_yaml}
    else:
        robot_cfg_full = robot_yaml
    robot_cfg = robot_cfg_full["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = (
        robot_cfg["kinematics"]["cspace"].get("default_joint_position")
        or robot_cfg["kinematics"]["cspace"].get("retract_config")
    )

    # ------------------------------------------------------------------
    # Batch motion planner: multi_env=True, L=2, G=1.
    # ------------------------------------------------------------------
    # No ``scene_model`` — self-collision only; matches the humanoid-parallel
    # IK example. ``num_envs=max_batch_size`` still propagates to RobotCfg
    # for per-env self-collision link_spheres.
    planner_cfg = MotionPlannerCfg.create(
        robot=robot_cfg_full,
        num_ik_seeds=64,
        num_trajopt_seeds=4,
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        self_collision_check=True,
        use_cuda_graph=False,
        max_batch_size=n_envs,
        multi_env=True,
        max_goalset=1,
        device_cfg=device_cfg,
    )
    planner = BatchMotionPlanner(planner_cfg)

    retract_kin = planner.compute_kinematics(
        planner.default_joint_state.clone().unsqueeze(0)
    )
    retract_tool_poses = {
        name: retract_kin.tool_poses[name] for name in planner.tool_frames
    }

    # ------------------------------------------------------------------
    # Per-env stage graph: one subroot, two palm targets, one robot.
    # ------------------------------------------------------------------
    robots = []
    base_positions: List[np.ndarray] = []
    per_env_targets: List[Dict[str, "cuboid.VisualCuboid"]] = []

    for env_idx in range(n_envs):
        base_position = np.array([0.0, ENV_OFFSET_Y * env_idx, 0.0], dtype=np.float32)
        base_positions.append(base_position)

        subroot = f"/World/world_{env_idx}"
        env_root = UsdGeom.Xform.Define(stage, subroot)
        if env_idx > 0:
            env_root.AddTranslateOp().Set(Gf.Vec3d(*base_position.tolist()))

        env_targets: Dict[str, "cuboid.VisualCuboid"] = {}
        for link_idx, link_name in enumerate(planner.tool_frames):
            link_pos_robot = retract_tool_poses[link_name].position.view(3).cpu().numpy()
            link_quat = retract_tool_poses[link_name].quaternion.view(4).cpu().numpy()
            color = np.array([1.0 if link_idx == 0 else 0.2,
                              0.2 if link_idx == 0 else 1.0,
                              0.2], dtype=np.float32)
            env_targets[link_name] = cuboid.VisualCuboid(
                f"{subroot}/target_{link_name}",
                position=(link_pos_robot + base_position),
                orientation=link_quat,
                color=color,
                size=0.05,
            )
        per_env_targets.append(env_targets)

        robot, _prim = add_robot_to_scene(
            robot_cfg,
            my_world,
            subroot=subroot,
            robot_name=f"robot_{env_idx}",
            position=base_position,
            initialize_world=False,
            load_from_usd=args.load_from_usd,
            usd_path=usd_path,
            usd_robot_root=usd_robot_root,
        )
        robots.append(robot)

    my_world.initialize_physics()
    print(
        f"Loaded {n_envs} G1 envs ({robot_file}), "
        f"tool_frames={planner.tool_frames}, action_dim={planner.action_dim}"
    )

    # ------------------------------------------------------------------
    # Warmup: plan once so CUDA graphs + trajopt seeds are primed.  Must
    # use the same shape (B, dof) as the main loop to avoid triggering a
    # ``reset_cuda_graph`` on the first real plan.
    # ------------------------------------------------------------------
    print("Warming up BatchMotionPlanner...")
    default_js_1 = planner.default_joint_state.clone().unsqueeze(0)
    warmup_state = _expand_joint_state(default_js_1, n_envs)
    warmup_positions = {
        name: retract_tool_poses[name].position.view(1, 3).expand(n_envs, 3).contiguous()
        for name in planner.tool_frames
    }
    warmup_quaternions = {
        name: retract_tool_poses[name].quaternion.view(1, 4).expand(n_envs, 4).contiguous()
        for name in planner.tool_frames
    }
    warmup_goal = _build_goal_tool_pose(planner, warmup_positions, warmup_quaternions)
    planner.plan_pose(warmup_goal, warmup_state, max_attempts=1)

    print(
        f"cuRobo humanoid batch_env motion-gen ready "
        f"(n_envs={n_envs}, tool_frames={planner.tool_frames})"
    )
    add_extensions(simulation_app, args.headless_mode)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    past_goal_pos: Optional[torch.Tensor] = None
    target_goal_pos: Optional[torch.Tensor] = None
    cmd_plans: List[Optional[JointState]] = [None] * n_envs
    cmd_idxs: List[int] = [0] * n_envs
    robot_idx_lists: List[List[int]] = [[] for _ in range(n_envs)]
    spheres_vis = None
    waiting_message_idx = 0

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if waiting_message_idx % 100 == 0:
                print("**** Click Play to start simulation *****")
            waiting_message_idx += 1
            continue

        step_index = my_world.current_time_step_index
        if step_index <= 10:
            for robot in robots:
                try:
                    robot._articulation_view.initialize()
                except Exception:
                    pass
                idx_list: List[int] = []
                valid_defaults: List[float] = []
                for i, jn in enumerate(j_names):
                    try:
                        idx = robot.get_dof_index(jn)
                    except (KeyError, Exception):
                        continue
                    if idx is None:
                        continue
                    idx_list.append(idx)
                    valid_defaults.append(default_config[i])
                if not idx_list:
                    continue
                robot.set_joint_positions(np.array(valid_defaults, dtype=np.float32), idx_list)
                robot._articulation_view.set_max_efforts(
                    values=np.array([5000.0] * len(idx_list)),
                    joint_indices=idx_list,
                )
        if step_index < 20:
            continue

        # ---- Read per-env target poses (robot-base frame) ----
        positions_np = {name: np.zeros((n_envs, 3), dtype=np.float32) for name in planner.tool_frames}
        quaternions_np = {name: np.zeros((n_envs, 4), dtype=np.float32) for name in planner.tool_frames}
        for env_idx in range(n_envs):
            for link_name in planner.tool_frames:
                p, q = per_env_targets[env_idx][link_name].get_local_pose()
                positions_np[link_name][env_idx] = p
                quaternions_np[link_name][env_idx] = q
        link_positions = {
            name: device_cfg.to_device(positions_np[name]) for name in planner.tool_frames
        }
        link_quaternions = {
            name: device_cfg.to_device(quaternions_np[name]) for name in planner.tool_frames
        }

        primary_positions = link_positions[planner.tool_frames[0]]
        if target_goal_pos is None:
            target_goal_pos = primary_positions.clone()
        if past_goal_pos is None:
            past_goal_pos = primary_positions.clone()

        # ---- Read per-env joint states into a single (B, dof) JointState ----
        per_env_positions = []
        per_env_velocities = []
        sim_js_names = robots[0].dof_names
        for robot in robots:
            sjs = robot.get_joints_state()
            if sjs is None:
                per_env_positions = None
                break
            per_env_positions.append(device_cfg.to_device(sjs.positions).view(1, -1))
            per_env_velocities.append(device_cfg.to_device(sjs.velocities).view(1, -1))
        if per_env_positions is None:
            continue
        positions_batched = torch.cat(per_env_positions, dim=0)
        velocities_batched = torch.cat(per_env_velocities, dim=0)
        batched_js = JointState(
            position=positions_batched,
            velocity=velocities_batched * 0.0,
            acceleration=velocities_batched * 0.0,
            jerk=velocities_batched * 0.0,
            joint_names=sim_js_names,
        ).reorder(planner.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list_batch = planner.kinematics.get_robot_as_spheres(batched_js.position)
            if spheres_vis is None:
                spheres_vis = []
                for env_idx, sph_list in enumerate(sph_list_batch):
                    env_spheres = []
                    for si, s in enumerate(sph_list):
                        env_spheres.append(
                            sphere.VisualSphere(
                                prim_path=f"/curobo/robot_{env_idx}_sphere_{si}",
                                position=np.ravel(s.position),
                                radius=float(s.radius),
                                color=np.array([0, 0.8, 0.2]),
                            )
                        )
                    spheres_vis.append(env_spheres)
            else:
                for env_idx, sph_list in enumerate(sph_list_batch):
                    for si, s in enumerate(sph_list):
                        spheres_vis[env_idx][si].set_world_pose(position=np.ravel(s.position))
                        spheres_vis[env_idx][si].set_radius(float(s.radius))

        # Any env's target moved + all are still + robots static + no env
        # is still animating a previous plan → re-plan the whole batch.
        moved_vs_target = (primary_positions - target_goal_pos).norm(dim=-1) > 1e-2
        still_vs_last_frame = (primary_positions - past_goal_pos).norm(dim=-1) < 1e-4
        robots_static = batched_js.velocity.abs().max() < 0.2
        any_cmd_running = any(p is not None for p in cmd_plans)

        should_plan = (
            not any_cmd_running
            and bool(moved_vs_target.any().item())
            and bool(still_vs_last_frame.all().item())
            and bool(robots_static.item())
        )

        if should_plan:
            goal = _build_goal_tool_pose(planner, link_positions, link_quaternions)
            result = planner.plan_pose(
                goal,
                batched_js,
                use_implicit_goal=True,
                max_attempts=3,
            )
            target_goal_pos = primary_positions.clone()

            if result is None:
                carb.log_warn("BatchMotionPlanner.plan_pose returned None.")
            else:
                success_mask = result.success.any(dim=-1) if result.success.ndim > 1 else result.success
                success_flat = success_mask.view(n_envs).detach().cpu().numpy().astype(bool)
                print(
                    "plan_pose: envs=%d success=%d/%d  total_time=%.3fs"
                    % (n_envs, int(success_flat.sum()), n_envs, float(result.total_time or 0.0))
                )

                plan_status = getattr(result, "status", None)
                for env_idx in range(n_envs):
                    if not success_flat[env_idx]:
                        reason = plan_status if plan_status else "planner returned success=False"
                        carb.log_warn(
                            f"env_{env_idx} plan_pose failed — {reason} "
                            f"(likely start/goal self-collision or unreachable IK)"
                        )
                        cmd_plans[env_idx] = None
                        continue
                    traj = _extract_per_env_trajectory(result, env_idx)
                    if traj is None or traj.position.shape[0] == 0:
                        carb.log_warn(
                            f"env_{env_idx} plan succeeded but trajectory is empty "
                            f"(start ≈ goal or interpolator returned 0 waypoints)"
                        )
                        cmd_plans[env_idx] = None
                        continue
                    # Map the planner's active joints onto the Isaac Sim
                    # articulation DOF indices for this env.
                    env_robot = robots[env_idx]
                    idx_list_env: List[int] = []
                    common: List[str] = []
                    for x in sim_js_names:
                        if x in traj.joint_names:
                            try:
                                idx_list_env.append(env_robot.get_dof_index(x))
                            except (KeyError, Exception):
                                continue
                            common.append(x)
                    cmd_plans[env_idx] = traj.reorder(common)
                    robot_idx_lists[env_idx] = idx_list_env
                    cmd_idxs[env_idx] = 0

                if not success_flat.any():
                    carb.log_warn("No env converged in this plan_pose batch.")

        past_goal_pos = primary_positions.clone()

        # ---- Play back per-env trajectories, one waypoint per sim step ----
        exec_skip = max(1, int(args.exec_skip))
        for env_idx in range(n_envs):
            plan = cmd_plans[env_idx]
            if plan is None:
                continue
            if cmd_idxs[env_idx] >= plan.position.shape[0]:
                # Trajectory finished — stay at the last waypoint.
                cmd_plans[env_idx] = None
                continue
            cmd_state = plan[cmd_idxs[env_idx]]
            robots[env_idx].set_joint_positions(
                cmd_state.position.cpu().numpy().reshape(-1),
                robot_idx_lists[env_idx],
            )
            cmd_idxs[env_idx] += exec_skip

    simulation_app.close()


if __name__ == "__main__":
    main()
