# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuRobo 2.0 port of the v1 ``mpc_example.py`` Isaac Sim sample.

Reactive MPC tracking of a draggable target cube in Isaac Sim. The robot
continuously tracks a red cube target; move the cube in the viewport and MPC
re-plans online to follow it while avoiding the collision table.

v1 → v2 surface change (see ``MIGRATION_V1_TO_V2.md`` Chapter 4 for the full map):

* ``MpcSolver`` / ``MpcSolverConfig``              → ``ModelPredictiveControl`` / ``ModelPredictiveControlCfg``
* ``MpcSolverConfig.load_from_robot_config``       → ``ModelPredictiveControlCfg.create``
* ``step_dt``                                      → ``optimization_dt``
* ``collision_cache={"obb": N}``                   → ``collision_cache={"cuboid": N}``
* ``Goal(...) + setup_solve_single + update_goal`` → ``setup(current_state)`` + ``update_goal_tool_poses(GoalToolPose)``
* ``mpc.step(...).js_action``                      → ``mpc.optimize_next_action(...).next_action``
* ``mpc.rollout_fn.joint_names``                   → ``mpc.joint_names``
* ``mpc.rollout_fn.dynamics_model.retract_config`` → ``mpc.default_joint_position``
* ``mpc.rollout_fn.compute_kinematics``            → ``mpc.compute_kinematics``
* ``mpc.world_coll_checker.load_collision_model``  → ``mpc.scene_collision_checker.load_collision_model``
* ``UsdHelper.get_obstacles_from_stage``           → :func:`stage_obstacles_as_scene`

Usage:
    python mpc_example.py                         # default Franka, LBFGS MPC
    python mpc_example.py --robot ur10.yml        # another robot
    python mpc_example.py --use_mppi              # MPPI+LBFGS two-stage MPC;
                                                  # draws rollouts in the viewport
"""

# Pin the pip-installed warp (1.12+) into sys.modules before Isaac Sim's
# omni.warp.core extension can hijack it with its vendored 1.8.2 files, and
# call warp.init() so cuda_devices is populated before cuRobo reads it. Must
# run above SimulationApp(). See curobo/examples/isaacsim/bootstrap.py.
from curobo.examples.isaacsim import bootstrap  # noqa: F401,E402

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
    help="When True, visualizes robot spheres.",
    default=False,
)
parser.add_argument(
    "--robot",
    type=str,
    default="franka.yml",
    help="Robot configuration file name (under curobo/content/configs/robot/).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Exit after N simulator steps. Useful for smoke-testing in headless mode.",
)
parser.add_argument(
    "--use_mppi",
    action="store_true",
    help=(
        "Use the MPPI+LBFGS two-stage optimizer instead of the LBFGS-only "
        "default. Enables debug drawing of the top MPPI rollouts in the "
        "viewport."
    ),
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
from omni.isaac.core.objects import cuboid  # noqa: E402
from omni.isaac.core.utils.types import ArticulationAction  # noqa: E402

from curobo.config_io import load_yaml  # noqa: E402
from curobo.content import get_robot_configs_path  # noqa: E402
from curobo.logging import setup_logger  # noqa: E402
from curobo.model_predictive_control import (  # noqa: E402
    ModelPredictiveControl,
    ModelPredictiveControlCfg,
)
from curobo.scene import Scene  # noqa: E402
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose  # noqa: E402

from curobo.examples.isaacsim.helper import (  # noqa: E402
    add_extensions,
    add_robot_to_scene,
    stage_obstacles_as_scene,
)


def draw_points(ee_positions):
    """Draw an ``(N, H, 3)`` tensor of EE trajectories as debug points.

    Used in ``--use_mppi`` mode to draw the planned horizon from the MPC
    result's ``robot_state_sequence`` — v2 MPC does not expose MPPI's raw
    sample rollouts through :meth:`MPPI.get_rollouts` because its rollout
    state is a :class:`RobotState` and storing 3-D sample trajectories
    would require custom plumbing. The planned EE horizon is the closest
    direct equivalent of v1's ``get_visual_rollouts`` and is what MPPI is
    ultimately converging on.
    """
    if ee_positions is None:
        return
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    cpu = ee_positions.detach().cpu().numpy()
    if cpu.ndim != 3 or cpu.shape[-1] != 3:
        return
    b, h, _ = cpu.shape
    point_list = []
    colors = []
    for i in range(b):
        point_list += [(cpu[i, j, 0], cpu[i, j, 1], cpu[i, j, 2]) for j in range(h)]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)


def _planned_ee_trajectory(
    mpc: "ModelPredictiveControl", result
) -> "torch.Tensor | None":
    """Pull the planned horizon's EE positions out of an MPC result.

    Returns ``(B, H, 3)`` on the primary tool frame, or ``None`` if the result
    didn't populate a robot-state sequence (e.g. on the very first warmup
    call). Works with any optimizer stack; only gated by ``--use_mppi`` in
    the loop to match the user-requested "MPPI ⇒ show rollouts" behavior.
    """
    rss = getattr(result, "robot_state_sequence", None)
    if rss is None or rss.tool_poses is None:
        return None
    link_idx = 0
    # tool_poses.position shape: (B, H, L, 3)
    return rss.tool_poses.position[:, :, link_idx, :].contiguous()


def _goal_from_pose(mpc: ModelPredictiveControl, pose: Pose) -> GoalToolPose:
    """Wrap a single primary-EE Pose as a ``(1, 1, L, 1, 3/4)`` GoalToolPose.

    Extra tool frames (if the robot YAML declares more than one) are filled
    with the identity-at-origin pose; MPC only tracks the primary frame in
    this example.
    """
    return GoalToolPose.from_poses(
        {mpc.tool_frames[0]: pose},
        ordered_tool_frames=mpc.tool_frames,
        num_goalset=1,
    )


def main():
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    my_world.scene.add_default_ground_plane()

    # Target cube created later, after MPC is ready, so we can place it at the
    # retract-config EE pose (v1 ``multi_arm_reacher.py`` pattern). Putting it
    # there keeps the first few frame's ``cube - eef`` delta at zero; user
    # drags it incrementally and MPC tracks it frame-to-frame. Placing the
    # cube 35cm away from the retract EE (like v1's ``[0.5, 0, 0.5]``) would
    # exceed v2 MPC's per-cycle reach and the arm would stall at retract.
    target = None

    setup_logger("warn")
    device_cfg = DeviceCfg()
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(str(robot_cfg_path / args.robot))["robot_cfg"]
    cspace = robot_cfg["kinematics"]["cspace"]
    j_names = cspace["joint_names"]
    # v2 YAML renamed ``retract_config`` → ``default_joint_position``; fall
    # back to the v1 key so older robot configs still load.
    default_config = cspace.get("default_joint_position", cspace.get("retract_config"))
    if default_config is None:
        raise KeyError(
            "cspace.default_joint_position (or retract_config) missing from robot YAML"
        )

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = robot.get_articulation_controller()

    # Optimizer selection: LBFGS-only (default) vs MPPI+LBFGS two-stage.
    if args.use_mppi:
        optimizer_configs = ["mpc/mppi_mpc.yml", "mpc/lbfgs_mpc.yml"]
    else:
        optimizer_configs = ["mpc/lbfgs_mpc.yml"]

    mpc_cfg = ModelPredictiveControlCfg.create(
        robot=args.robot,
        scene_model="collision_table.yml",
        optimizer_configs=optimizer_configs,
        use_cuda_graph=True,
        self_collision_check=True,
        collision_cache={"cuboid": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimization_dt=0.02,
        interpolation_steps=4,
        device_cfg=device_cfg,
        max_batch_size=1,
        multi_env=False,
        max_goalset=1,
    )
    mpc = ModelPredictiveControl(mpc_cfg)

    # Seed MPC at the robot's retract config with zero velocity/acceleration.
    current_state = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0),
        joint_names=mpc.joint_names,
    )
    current_state.velocity = torch.zeros_like(current_state.position)
    current_state.acceleration = torch.zeros_like(current_state.position)

    mpc.setup(current_state)

    # Initial goal = FK at the retract config. We also place the draggable
    # cube here so the first user interaction (grab + drag) produces small
    # incremental targets — the only input pattern v2 MPC can actually track.
    kin = mpc.compute_kinematics(current_state)
    retract_pose = kin.tool_poses.to_dict()[mpc.tool_frames[0]]
    retract_pos_np = retract_pose.position.view(3).cpu().numpy()
    retract_quat_np = retract_pose.quaternion.view(4).cpu().numpy()
    target = cuboid.VisualCuboid(
        "/World/target",
        position=retract_pos_np,
        orientation=retract_quat_np,
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )
    mpc.update_goal_tool_poses(
        _goal_from_pose(
            mpc,
            Pose(
                position=retract_pose.position.view(1, 3).clone(),
                quaternion=retract_pose.quaternion.view(1, 4).clone(),
            ),
        ),
        run_ik=False,
    )

    # Warmup a first optimize call so CUDA graphs get captured.
    mpc.optimize_action_sequence(current_state)

    init_curobo = False
    step = 0
    add_extensions(simulation_app, args.headless_mode)

    # In headless there is no UI to hit "Play"; start the sim programmatically.
    # v1 did a 10-step ``my_world.step`` warmup before the main loop; we drop it
    # because (a) the MPC solver was already warmed up with
    # ``optimize_next_action`` above, and (b) with auto-play the pre-loop steps
    # would advance ``current_time_step_index`` past the ``step_index <= 10``
    # window that set_joint_positions relies on, leaving the robot at its URDF
    # zero pose instead of the retract config.
    if args.headless_mode is not None:
        my_world.play()

    while simulation_app.is_running():
        my_world.step(render=True)

        if args.max_steps is not None and my_world.current_time_step_index >= args.max_steps:
            print(f"Reached --max_steps={args.max_steps}, exiting.", flush=True)
            break

        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for _ in range(len(idx_list))]),
                joint_indices=idx_list,
            )

        if not init_curobo:
            init_curobo = True
        step += 1
        step_index = step

        if step_index % 1000 == 0:
            print("Updating world", flush=True)
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
            mpc.scene_collision_checker.load_collision_model(obstacles)

        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position + 1.0

        if np.linalg.norm(cube_position - past_pose) > 1e-3:
            ik_goal = Pose(
                position=torch.as_tensor(
                    cube_position, device=device_cfg.device, dtype=device_cfg.dtype
                ).view(1, 3),
                quaternion=torch.as_tensor(
                    cube_orientation, device=device_cfg.device, dtype=device_cfg.dtype
                ).view(1, 4),
            )
            # Pose-only tracking (same pattern as v2's
            # ``reactive_control --visualize`` and ``humanoid_retargeting --mpc``).
            # Using ``run_ik=True`` here would silently fail for any cube
            # position the internal 1-seed IK can't reach, leaving the goal
            # unchanged — see the discussion in ``BATCH_INTERFACES.md``.
            mpc.update_goal_tool_poses(_goal_from_pose(mpc, ik_goal), run_ik=False)
            past_pose = cube_position

        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names

        # Feed MPC its **own last planned state**, not the live sim joints.
        # PD controllers always lag a few ms behind the command, so reading
        # ``robot.get_joints_state()`` back into MPC makes each cycle re-plan
        # from a position just barely past the previous one → knot 7 only
        # advances a fraction of a millimetre per frame → arm looks frozen.
        # v2's own viser loop (``reactive_control.py:306-317``) does the same
        # thing: ``current_state`` is chained from ``action_sequence[:, -1]``
        # so the planner always projects forward in its own clean coordinate
        # frame. The ``sim_js`` read is only used for visualising drift or
        # falling back if MPC produced no action this tick.

        # Use ``optimize_action_sequence`` + command ``action_sequence[:, -1]``
        # (the last knot of the trimmed execution horizon). Commanding
        # ``next_action`` (knot 0) would keep the robot glued to the current
        # pose because the B-spline's first few knots all sit on the start
        # state (ease-in region).
        mpc_result = mpc.optimize_action_sequence(current_state)

        if args.use_mppi:
            draw_points(_planned_ee_trajectory(mpc, mpc_result))

        succ = (
            mpc_result.action_sequence is not None
            and mpc_result.action_sequence.position.shape[1] > 0
        )
        if not succ:
            carb.log_warn("MPC returned no action sequence; skipping command.")
            continue

        act_seq = mpc_result.action_sequence
        cmd_position = act_seq.position[:, -1, :]

        # Chain MPC's own view of the robot: next optimize() call will see the
        # planner's last predicted state, not the lagged sim state.
        current_state = JointState.from_position(
            cmd_position.clone(), joint_names=list(mpc.joint_names),
        )
        current_state.velocity = act_seq.velocity[:, -1, :].clone()
        current_state.acceleration = act_seq.acceleration[:, -1, :].clone()

        cmd_state_full = JointState.from_position(
            cmd_position.clone(), joint_names=list(mpc.joint_names),
        )

        idx_list = []
        common_js_names = []
        for x in sim_js_names:
            if x in mpc.joint_names:
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)

        cmd_state = cmd_state_full.reorder(common_js_names)

        art_action = ArticulationAction(
            cmd_state.position.view(-1).cpu().numpy(),
            joint_indices=idx_list,
        )

        if step_index % 1000 == 0:
            pose_err = mpc_result.position_error
            err_str = f"{pose_err.item():.4f}" if pose_err is not None else "N/A"
            print(f"pose_error={err_str}", flush=True)

        articulation_controller.apply_action(art_action)


if __name__ == "__main__":
    main()
    simulation_app.close()
