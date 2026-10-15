# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuRobo 2.0 port of the v1 ``ik_reachability.py`` Isaac Sim sample.

Two run modes, selected by ``--reachability``:

* **Single target (default).**  Solve one IK query to the cube target and
  snap the robot to the solution.  ``max_batch_size=1``.
* **Reachability grid.**  Build a 10×10×5 grid of target offsets and solve
  all 500 problems in a single batched call.  Results are drawn as
  green/red points and any successful solutions are cycled through to
  animate the robot.  ``max_batch_size=N`` where ``N`` is the grid size.

v1 → v2 surface change (see ``MIGRATION_V1_TO_V2.md`` for the full map):

* ``IKSolver``/``IKSolverConfig``        → ``InverseKinematics`` / ``InverseKinematicsCfg``
* ``solve_single`` / ``solve_batch``     → single entry ``solve_pose``
* ``Pose`` (primary EE only)             → ``GoalToolPose (B, 1, L, G, 3/4)``
* ``retract_config`` + ``seed_config``   → ``current_state: JointState``
* ``WorldConfig`` + ``UsdHelper``        → ``Scene`` + :class:`UsdSceneParser`
* ``ik_solver.kinematics.joint_names``   → ``ik.joint_names``

Usage:
    python ik_reachability.py                       # single-target mode
    python ik_reachability.py --reachability        # 500-pose reachability grid
    python ik_reachability.py --visualize_spheres   # draw collision spheres
"""

# Must stay above every ``omni.*`` / ``isaacsim.*`` / ``curobo._src`` import:
# pins pip's warp-lang 1.12 in ``sys.modules`` so Isaac Sim's bundled
# ``omni.warp.core 1.8.2`` cannot shadow it, and eagerly runs ``warp.init()``
# so ``warp.context.runtime.cuda_devices`` is populated before cuRobo's
# ``MeshData.__post_init__`` reads it.
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
    "--reachability",
    action="store_true",
    help="Run the 10x10x5 reachability-grid sweep instead of single-target IK.",
    default=False,
)
parser.add_argument(
    "--robot",
    type=str,
    default="franka.yml",
    help="Robot configuration file name (under curobo/content/configs/robot/).",
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

from curobo.config_io import load_yaml  # noqa: E402
from curobo.content import get_robot_configs_path  # noqa: E402
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg  # noqa: E402
from curobo.logging import setup_logger  # noqa: E402
from curobo.scene import Scene  # noqa: E402
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose  # noqa: E402

from curobo.examples.isaacsim.helper import (  # noqa: E402
    add_extensions,
    add_robot_to_scene,
    stage_obstacles_as_scene,
)


############################################################
# Helpers
############################################################


def get_pose_grid(n_x: int, n_y: int, n_z: int, max_x: float, max_y: float, max_z: float) -> np.ndarray:
    """Return an ``(n_x*n_y*n_z, 3)`` grid of position offsets."""
    x = np.linspace(-max_x, max_x, n_x)
    y = np.linspace(-max_y, max_y, n_y)
    z = np.linspace(0, max_z, n_z)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    out = np.zeros((n_x * n_y * n_z, 3))
    out[:, 0] = xx.flatten()
    out[:, 1] = yy.flatten()
    out[:, 2] = zz.flatten()
    return out


def draw_points(positions: torch.Tensor, success: torch.Tensor) -> None:
    """Draw per-goal success (green) / failure (red) points in the debug viewport."""
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    pts = positions.detach().cpu().numpy()
    flags = success.detach().cpu().numpy().reshape(-1)
    b = pts.shape[0]
    point_list = [tuple(pts[i]) for i in range(b)]
    colors = [
        (0, 1, 0, 0.25) if bool(flags[i]) else (1, 0, 0, 0.25)
        for i in range(b)
    ]
    sizes = [40.0] * b
    draw.draw_points(point_list, colors, sizes)


def _expand_joint_state(js: JointState, batch_size: int) -> JointState:
    """Broadcast a ``(1, dof)`` JointState to ``(batch_size, dof)``.

    Avoids :meth:`JointState.repeat`, which v2 marks ``@deprecated`` and emits
    a warning every call (would be once per solve in reachability mode).
    """
    if batch_size == 1:
        return js
    return JointState.from_position(
        js.position.expand(batch_size, -1).contiguous(),
        joint_names=js.joint_names,
    )


def build_goal_tool_pose(
    ik: InverseKinematics,
    ik_goal: Pose,
    position_grid_offset: torch.Tensor | None = None,
) -> GoalToolPose:
    """Build the v2 :class:`GoalToolPose` for either mode.

    Args:
        ik: Configured IK solver (used to pull ``tool_frames``).
        ik_goal: Primary EE target, ``position=(1,3)``, ``quaternion=(1,4)``.
        position_grid_offset: Optional ``(B,3)`` offsets added to ``ik_goal.position``.
            When provided, produces a ``(B,1,L,1,3/4)`` goal. Otherwise produces
            ``(1,1,L,1,3/4)``.
    """
    primary = ik.tool_frames[0]
    if position_grid_offset is None:
        return GoalToolPose.from_poses(
            {primary: ik_goal},
            ordered_tool_frames=ik.tool_frames,
            num_goalset=1,
        )

    batch = position_grid_offset.shape[0]
    positions = ik_goal.position.expand(batch, 3).clone()
    positions = positions + position_grid_offset
    quaternions = ik_goal.quaternion.expand(batch, 4).contiguous()
    batched = Pose(position=positions, quaternion=quaternions)
    return GoalToolPose.from_poses(
        {primary: batched},
        ordered_tool_frames=ik.tool_frames,
        num_goalset=1,
    )


############################################################
# Main
############################################################


GRID_SIZE = (10, 10, 5)
GRID_EXTENT = (0.5, 0.5, 0.5)


def main():
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_logger("warn")
    device_cfg = DeviceCfg()
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(str(robot_cfg_path / args.robot))["robot_cfg"]
    # v2 YAML renamed ``retract_config`` → ``default_joint_position`` under
    # ``kinematics.cspace``; fall back to the v1 name for older configs.
    cspace = robot_cfg["kinematics"]["cspace"]
    j_names = cspace["joint_names"]
    default_config = cspace.get("default_joint_position", cspace.get("retract_config"))

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    robot.get_articulation_controller()

    reachability_mode = bool(args.reachability)
    if reachability_mode:
        grid_offset_t = device_cfg.to_device(get_pose_grid(*GRID_SIZE, *GRID_EXTENT))
        max_batch_size = grid_offset_t.shape[0]
    else:
        grid_offset_t = None
        max_batch_size = 1

    ik_cfg = InverseKinematicsCfg.create(
        robot=args.robot,
        scene_model="collision_table.yml",
        num_seeds=20,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        self_collision_check=True,
        use_cuda_graph=True,
        collision_cache={"cuboid": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        max_batch_size=max_batch_size,
        max_goalset=1,
        multi_env=False,
        device_cfg=device_cfg,
    )
    ik = InverseKinematics(ik_cfg)

    # Warmup: solve at the retract pose so CUDA graphs + caches are primed.
    # Must use the EXACT shape the main loop will use (same ``current_state``
    # batch dim, same goal tensor shape). Any difference forces ``reset_shape``
    # → ``reset_cuda_graph`` on the first real call, which v2 disallows once
    # CUDA graphs are captured ("CUDA graph reset is not available").
    default_js_1 = ik.default_joint_state.clone().unsqueeze(0)          # (1, dof)
    kin = ik.compute_kinematics(default_js_1)
    retract_pose = kin.tool_poses[ik.tool_frames[0]]
    retract_pose = Pose(
        position=retract_pose.position.view(1, 3).clone(),
        quaternion=retract_pose.quaternion.view(1, 4).clone(),
    )
    warmup_goal = build_goal_tool_pose(ik, retract_pose, grid_offset_t)
    warmup_state = _expand_joint_state(default_js_1, max_batch_size)
    ik.solve_pose(warmup_goal, current_state=warmup_state)

    print("cuRobo IK ready (mode: %s, batch=%d)"
          % ("reachability" if reachability_mode else "single", max_batch_size))

    add_extensions(simulation_app, args.headless_mode)

    my_world.scene.add_default_ground_plane()

    past_pose = None
    target_pose = None
    cmd_plan = None
    cmd_idx = 0
    i = 0
    spheres = None

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index
        if step_index <= 10:
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000.0] * len(idx_list)),
                joint_indices=idx_list,
            )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 500 == 0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles: Scene = stage_obstacles_as_scene(
                my_world.stage,
                reference_prim_path=robot_prim_path,
                only_paths=("/World",),
                ignore_substring=(
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ),
            )
            print([o.name for o in obstacles.objects])
            ik.update_world(obstacles)
            carb.log_info("Synced cuRobo world from stage.")

        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position

        sim_js = robot.get_joints_state()
        if sim_js is None:
            continue
        sim_js_names = robot.dof_names
        zero_vel = device_cfg.to_device(sim_js.velocities) * 0.0
        cu_js = JointState(
            position=device_cfg.to_device(sim_js.positions),
            velocity=zero_vel,
            acceleration=zero_vel.clone(),
            jerk=zero_vel.clone(),
            joint_names=sim_js_names,
        ).reorder(ik.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = ik.kinematics.get_robot_as_spheres(cu_js.position)
            if spheres is None:
                spheres = []
                for si, s in enumerate(sph_list[0]):
                    spheres.append(
                        sphere.VisualSphere(
                            prim_path=f"/curobo/robot_sphere_{si}",
                            position=np.ravel(s.position),
                            radius=float(s.radius),
                            color=np.array([0, 0.8, 0.2]),
                        )
                    )
            else:
                for si, s in enumerate(sph_list[0]):
                    spheres[si].set_world_pose(position=np.ravel(s.position))
                    spheres[si].set_radius(float(s.radius))

        # Only re-solve when (a) the user has actually moved the cube, (b) the
        # cube is still (not mid-drag), (c) the robot is close to static, and
        # (d) we're not already cycling through a previous batch of solutions
        # — without the last clause, reachability mode re-triggers a full
        # 500-pose sweep every few frames during the cmd_plan animation even
        # though the target hasn't changed.
        target_moved = (
            cmd_plan is None
            and np.linalg.norm(cube_position - target_pose) > 1e-3
            and np.linalg.norm(past_pose - cube_position) == 0.0
            and np.linalg.norm(sim_js.velocities) < 0.2
        )
        if target_moved:
            ik_goal = Pose(
                position=device_cfg.to_device(cube_position).view(1, 3),
                quaternion=device_cfg.to_device(cube_orientation).view(1, 4),
            )
            goal = build_goal_tool_pose(ik, ik_goal, grid_offset_t)
            current_state = _expand_joint_state(cu_js.unsqueeze(0), max_batch_size)
            result = ik.solve_pose(goal, current_state=current_state)

            batch_positions = goal.position.view(max_batch_size, 3)
            print(
                "IK completed: poses=%d  solve_time=%.4fs  success=%d/%d"
                % (
                    max_batch_size,
                    float(result.solve_time),
                    int(result.success.sum().item()),
                    max_batch_size,
                )
            )

            if reachability_mode:
                draw_points(batch_positions, result.success)

            if bool(result.success.any().item()):
                good = result.success.view(-1).bool()
                # js_solution is (B, return_seeds, dof) with return_seeds=1; take seed 0
                # and keep only successful problems.
                js = JointState.from_position(
                    result.js_solution.position[:, 0, :][good],
                    joint_names=result.js_solution.joint_names,
                )

                idx_list = []
                common = []
                for x in sim_js_names:
                    if x in js.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common.append(x)
                cmd_plan = js.reorder(common)
                cmd_idx = 0
            else:
                carb.log_warn("Plan did not converge to a solution. No action is being taken.")
            target_pose = cube_position
        past_pose = cube_position

        if cmd_plan is not None and step_index % 20 == 0:
            cmd_state = cmd_plan[cmd_idx]
            robot.set_joint_positions(cmd_state.position.cpu().numpy(), idx_list)
            cmd_idx += 1
            if cmd_idx >= len(cmd_plan.position):
                # Done walking through this batch of solutions. Hold the last
                # applied joint position and wait for the next target change —
                # do NOT snap back to ``default_config``. v1's reset-to-home
                # was a visual cue for the reachability-grid sweep; in
                # single-target mode it manifests as an instant flash back to
                # the default pose.
                cmd_plan = None
                cmd_idx = 0

    simulation_app.close()


if __name__ == "__main__":
    main()
