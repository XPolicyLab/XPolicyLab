# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuRobo 2.0 ``solve_batch_env`` IK demo — two robots, two worlds, one solve.

Port of v1 ``batch_motion_gen_reacher.py`` restricted to IK (no trajopt) and
upgraded to the v2 API.  Two Franka arms live in two independent collision
environments (``/World/world_0`` with ``collision_test.yml`` and
``/World/world_1`` with ``collision_table.yml``); each has its own target
cube.  On every target update, both arms' IK queries are stacked into a
single ``(B=2, 1, L=1, G=1, 3/4)`` :class:`GoalToolPose` and solved together
with ``multi_env=True`` — v2's equivalent of v1's ``solve_batch_env``.

Mode selection map (see ``BATCH_INTERFACES.md``):

+---------------+------------------+-------------+-----------------+
| v1 name       | ``max_batch_size`` | ``multi_env`` | ``max_goalset`` |
+===============+==================+=============+=================+
| solve_batch_env | ``N=2``          | ``True``      | ``1``           |
+---------------+------------------+-------------+-----------------+

Runtime flow:

1. Build ``InverseKinematics`` with ``max_batch_size=2, multi_env=True`` and
   a *template* ``scene_model`` sized for the largest per-env scene (its
   only job here is to allocate the collision cache).
2. Replace env 0's obstacles via
   ``ik.scene_collision_checker.load_collision_model(scene_0, env_idx=0)``;
   same for env 1 — this is how v2 gives each problem its own world.
3. Warmup once with a ``(2, dof)`` state so the first real solve doesn't
   change the goal-buffer structure (would otherwise trigger
   ``reset_cuda_graph`` → ``CUDA graph reset is not available``).
4. Main loop: read each target cube's local pose (it's a child of
   ``/World/world_i`` so the local pose already sits in the robot's base
   frame), stack to ``(2, 3)`` + ``(2, 4)``, wrap as ``GoalToolPose``,
   read both robots' joint states into a single ``(2, dof)``
   :class:`JointState`, call :meth:`ik.solve_pose`, and apply per-env
   solutions via :meth:`Robot.set_joint_positions`.

Usage::

    python -m curobo.examples.isaacsim.ik_batched_env
    python -m curobo.examples.isaacsim.ik_batched_env --visualize_spheres
"""

# Pin pip's warp-lang 1.12 BEFORE any ``omni.*`` / ``isaacsim.*`` /
# ``curobo._src`` import. See ``bootstrap.py`` for why.
from curobo.examples.isaacsim import bootstrap  # noqa: F401

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

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
    "--robot",
    type=str,
    default="franka.yml",
    help="Robot configuration file name (under curobo/content/configs/robot/).",
)
parser.add_argument(
    "--n_envs",
    type=int,
    default=2,
    help="Number of parallel environments. The example only ships two distinct "
         "scenes; extra envs will be filled with the last scene in the list.",
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

from curobo.config_io import load_yaml  # noqa: E402
from curobo.content import get_robot_configs_path, get_scene_configs_path  # noqa: E402
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg  # noqa: E402
from curobo.logging import setup_logger  # noqa: E402
from curobo.scene import Scene  # noqa: E402
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose  # noqa: E402
from curobo.viewer import UsdWriter  # noqa: E402

from curobo.examples.isaacsim.helper import (  # noqa: E402
    add_extensions,
    add_robot_to_scene,
)


############################################################
# Helpers
############################################################


PER_ENV_SCENE_FILES = (
    "collision_test.yml",       # env 0 — table + box
    "collision_table.yml",      # env 1 — thicker table, no box
)

ENV_OFFSET_Y = 2.5              # world-frame spacing between robots


def _expand_joint_state(js: JointState, batch_size: int) -> JointState:
    """Broadcast a ``(1, dof)`` JointState to ``(batch_size, dof)``.

    Avoids :meth:`JointState.repeat`, which v2 marks ``@deprecated``.
    """
    if batch_size == 1:
        return js
    return JointState.from_position(
        js.position.expand(batch_size, -1).contiguous(),
        joint_names=js.joint_names,
    )


def _load_env_scene(file_name: str) -> Scene:
    """Load one per-env scene YAML into a :class:`Scene` instance.

    The shipped YAMLs key obstacles by name (``cuboid: {my_table: {dims, pose}}``)
    so we must go through :meth:`Scene.create`, not ``Scene(**yaml_dict)``.
    """
    path = get_scene_configs_path() / file_name
    return Scene.create(load_yaml(str(path)))


def _build_batch_goal(
    ik: InverseKinematics,
    batch_positions: torch.Tensor,
    batch_quaternions: torch.Tensor,
) -> GoalToolPose:
    """Wrap per-env target poses into a ``(B, 1, L, 1, 3/4)`` GoalToolPose."""
    batch_pose = Pose(position=batch_positions, quaternion=batch_quaternions)
    return GoalToolPose.from_poses(
        {ik.tool_frames[0]: batch_pose},
        ordered_tool_frames=ik.tool_frames,
        num_goalset=1,
    )


############################################################
# Main
############################################################


def main():
    n_envs = max(1, int(args.n_envs))
    device_cfg = DeviceCfg()

    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    setup_logger("warn")

    # ------------------------------------------------------------------
    # Build the per-env USD scene graph + robots + targets
    # ------------------------------------------------------------------
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(str(robot_cfg_path / args.robot))["robot_cfg"]
    cspace = robot_cfg["kinematics"]["cspace"]
    j_names = cspace["joint_names"]
    default_config = cspace.get("default_joint_position", cspace.get("retract_config"))

    writer = UsdWriter()
    writer.load_stage(stage)

    robots = []
    targets = []
    robot_base_positions = []
    env_scene_files = [
        PER_ENV_SCENE_FILES[i] if i < len(PER_ENV_SCENE_FILES) else PER_ENV_SCENE_FILES[-1]
        for i in range(n_envs)
    ]
    env_scenes: list[Scene] = []

    for env_idx in range(n_envs):
        base_position = np.array([0.0, ENV_OFFSET_Y * env_idx, 0.0], dtype=np.float32)
        robot_base_positions.append(base_position)

        subroot = f"/World/world_{env_idx}"
        # Define the env-root xform directly via the USD API rather than
        # ``UsdWriter.add_subroot``.  ``add_subroot`` joins via
        # ``join_usd_path`` which strips a leading ``/`` from the sub-root arg
        # (USD-path convention), so a v1-style call like
        # ``add_subroot("/World", "/World/world_1", pose)`` silently creates
        # ``/World/World/world_1`` and the "real" ``/World/world_1`` is left as
        # an implicit identity Xform auto-created under the child prims.
        # That breaks per-env obstacle placement and makes
        # ``target.get_local_pose()`` return world coords — the planner then
        # receives unreachable goals and env 1's arm never moves.  Using the
        # low-level ``UsdGeom.Xform.Define`` avoids the string-join ambiguity.
        env_root = UsdGeom.Xform.Define(stage, subroot)
        if env_idx > 0:
            env_root.AddTranslateOp().Set(Gf.Vec3d(*base_position.tolist()))

        # Target lives inside the subroot so ``get_local_pose`` is already in
        # the robot's base frame (robot base is at the subroot origin).
        target = cuboid.VisualCuboid(
            f"{subroot}/target",
            position=(np.array([0.5, 0.0, 0.5], dtype=np.float32) + base_position),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0.0, 0.0]),
            size=0.05,
        )
        targets.append(target)

        # Robot placement: Isaac Sim 4.5 ignores ``subroot=`` in the URDF
        # importer and assigns its own path, so we separate the two arms by
        # world-frame translation instead.
        robot, _prim = add_robot_to_scene(
            robot_cfg,
            my_world,
            subroot=subroot,
            robot_name=f"robot_{env_idx}",
            position=base_position,
            initialize_world=False,
        )
        robots.append(robot)

        # Per-env collision world — add to the USD stage for visualization and
        # keep a Scene object for handing to the collision checker below.
        scene_i = _load_env_scene(env_scene_files[env_idx])
        env_scenes.append(scene_i)
        writer.add_world_to_stage(
            scene_i,
            base_frame=subroot,
            obstacles_frame=f"obstacles_{env_idx}",
        )

    my_world.initialize_physics()
    print(f"Loaded {n_envs} envs, scenes: {env_scene_files}")

    # ------------------------------------------------------------------
    # IK solver: multi_env=True with a template scene for cache sizing
    # ------------------------------------------------------------------
    # Pick the template scene with the largest obstacle counts so the cache is
    # big enough for every env.  The ``load_collision_model`` calls below then
    # overwrite the allocated cache entries per env.
    cuboid_cache = max((len(s.cuboid) for s in env_scenes), default=1)
    mesh_cache = max((len(s.mesh) for s in env_scenes), default=1)

    ik_cfg = InverseKinematicsCfg.create(
        robot=args.robot,
        scene_model=env_scene_files[0],           # template for cache sizing
        num_seeds=20,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        self_collision_check=True,
        use_cuda_graph=True,
        collision_cache={"cuboid": max(4, cuboid_cache), "mesh": max(1, mesh_cache)},
        max_batch_size=n_envs,
        multi_env=True,
        max_goalset=1,
        device_cfg=device_cfg,
    )
    # v2 quirk: ``InverseKinematicsCfg.create(scene_model=<single file>,
    # multi_env=True, max_batch_size=N)`` only wires ``num_envs=N`` through to
    # the RobotCfg / kinematics side.  ``create_scene_collision_cfg`` builds a
    # ``SceneCollisionCfg`` whose ``__post_init__`` only bumps ``num_envs``
    # when ``scene_model`` is a *list*, so the collision cache is allocated
    # with shape ``(1, N_cuboids)`` and a later ``load_collision_model(..,
    # env_idx=1)`` trips
    #
    #     IndexError: index 1 is out of bounds for dimension 0 with size 1
    #
    # To get per-env storage we swap in the full list of ``Scene`` objects
    # before the solver is constructed; ``SceneCollision.from_config`` then
    # dispatches to ``SceneData.from_batch_scene_cfg(list, …)``, which
    # allocates ``(N, N_cuboids)`` and loads each env's obstacles at build
    # time.
    ik_cfg.core_cfg.scene_collision_cfg.scene_model = env_scenes
    ik_cfg.core_cfg.scene_collision_cfg.num_envs = n_envs
    ik = InverseKinematics(ik_cfg)

    # Warmup: MUST use the same goal-buffer structure as the main loop (same
    # batch size, same ``current_state`` presence). See ``ik_reachability.py``
    # for the full explanation.
    default_js_1 = ik.default_joint_state.clone().unsqueeze(0)
    kin = ik.compute_kinematics(default_js_1)
    retract_pose = kin.tool_poses[ik.tool_frames[0]]
    warmup_positions = retract_pose.position.view(1, 3).expand(n_envs, 3).contiguous()
    warmup_quaternions = retract_pose.quaternion.view(1, 4).expand(n_envs, 4).contiguous()
    warmup_goal = _build_batch_goal(ik, warmup_positions, warmup_quaternions)
    warmup_state = _expand_joint_state(default_js_1, n_envs)
    ik.solve_pose(warmup_goal, current_state=warmup_state)

    print(f"cuRobo batch_env IK ready ({n_envs} envs)")
    add_extensions(simulation_app, args.headless_mode)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    past_goal = None
    target_goal = None
    cmd_plans = [None] * n_envs
    cmd_idxs = [0] * n_envs
    robot_idx_lists: list[list[int]] = [[] for _ in range(n_envs)]
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
                idx_list = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, idx_list)
                robot._articulation_view.set_max_efforts(
                    values=np.array([5000.0] * len(idx_list)),
                    joint_indices=idx_list,
                )
        if step_index < 20:
            continue

        # Read all targets (already in robot-base frame via local pose).
        target_positions_np = np.zeros((n_envs, 3), dtype=np.float32)
        target_quaternions_np = np.zeros((n_envs, 4), dtype=np.float32)
        for i, t in enumerate(targets):
            p, q = t.get_local_pose()
            target_positions_np[i] = p
            target_quaternions_np[i] = q
        cur_goal = Pose(
            position=device_cfg.to_device(target_positions_np),
            quaternion=device_cfg.to_device(target_quaternions_np),
        )
        if target_goal is None:
            target_goal = cur_goal.clone()
        if past_goal is None:
            past_goal = cur_goal.clone()

        # Read each robot's joint state into a single (B, dof) JointState.
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
        ).reorder(ik.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list_batch = ik.kinematics.get_robot_as_spheres(batched_js.position)
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

        # Any env's target moved → re-solve for the whole batch.
        moved_vs_target = (cur_goal.position - target_goal.position).norm(dim=-1) > 1e-3
        still_vs_last_frame = (cur_goal.position - past_goal.position).norm(dim=-1) < 1e-6
        robots_static = batched_js.velocity.abs().max() < 0.2
        any_cmd_running = any(p is not None for p in cmd_plans)

        should_solve = (
            not any_cmd_running
            and bool(moved_vs_target.any().item())
            and bool(still_vs_last_frame.all().item())
            and bool(robots_static.item())
        )

        if should_solve:
            goal = _build_batch_goal(ik, cur_goal.position, cur_goal.quaternion)
            result = ik.solve_pose(goal, current_state=batched_js)
            target_goal.copy_(cur_goal)

            success_flat = result.success.view(n_envs).detach().cpu().numpy().astype(bool)
            print(
                "IK completed: envs=%d success=%d/%d solve_time=%.4fs"
                % (n_envs, int(success_flat.sum()), n_envs, float(result.solve_time))
            )

            # js_solution is (B, return_seeds=1, dof); take seed 0 per env.
            sol_positions = result.js_solution.position[:, 0, :]
            sol_joint_names = result.js_solution.joint_names

            ik_status = getattr(result, "status", None)
            for env_idx in range(n_envs):
                if not success_flat[env_idx]:
                    reason = ik_status if ik_status else "IK seed did not converge"
                    carb.log_warn(
                        f"env_{env_idx} IK failed — {reason} "
                        f"(goal likely unreachable or in collision)"
                    )
                    cmd_plans[env_idx] = None
                    continue
                single_env_js = JointState.from_position(
                    sol_positions[env_idx:env_idx + 1],
                    joint_names=sol_joint_names,
                )
                env_robot = robots[env_idx]
                idx_list_env: list[int] = []
                common: list[str] = []
                for x in sim_js_names:
                    if x in sol_joint_names:
                        idx_list_env.append(env_robot.get_dof_index(x))
                        common.append(x)
                robot_idx_lists[env_idx] = idx_list_env
                cmd_plans[env_idx] = single_env_js.reorder(common)
                cmd_idxs[env_idx] = 0

            if not success_flat.any():
                carb.log_warn("Batch IK did not converge for any env.")

        past_goal.copy_(cur_goal)

        # Apply whichever envs still have a plan to run.
        if step_index % 20 == 0:
            for env_idx in range(n_envs):
                plan = cmd_plans[env_idx]
                if plan is None:
                    continue
                if cmd_idxs[env_idx] >= len(plan.position):
                    cmd_plans[env_idx] = None
                    continue
                cmd_state = plan[cmd_idxs[env_idx]]
                robots[env_idx].set_joint_positions(
                    cmd_state.position.cpu().numpy(),
                    robot_idx_lists[env_idx],
                )
                cmd_idxs[env_idx] += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
