# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuRobo 2.0 port of ``MagicCurobo/examples/isaac_sim/multi_arm_reacher.py``.

Multi-tool-frame (multi-EEF) motion planning reacher: the robot YAML declares
more than one ``tool_frames`` entry (``dual_ur10e.yml`` defaults to
``["tool1", "tool0"]``) and every plan tracks **every** tool frame
simultaneously. One draggable target cuboid in the stage drives the primary
tool frame (``planner.tool_frames[0]``); one extra cuboid per additional
tool frame drives the rest. Every time any cube settles in a new pose,
:meth:`MotionPlanner.plan_pose` is re-invoked with a
``GoalToolPose`` of shape ``(1, 1, L, 1, 3/4)``.

v1 → v2 delta (see ``MIGRATION_V1_TO_V2.md`` §3.2 and
``BATCH_INTERFACES.md`` §2):

* ``MotionGen.plan_single(q, goal_pose=Pose, link_poses={...})`` →
  ``MotionPlanner.plan_pose(GoalToolPose, current_state)`` with a single
  goal tensor that packs every tracked link.
* v1's ``kinematics.ee_link`` + ``kinematics.link_names`` split is replaced by
  a single ``tool_frames`` list in the robot YAML. The first entry of
  ``planner.tool_frames`` plays the role of the old primary EE — that's the
  cube with the red outline.
* Scene loading uses :class:`curobo.scene.Scene` directly; the v1
  ``world_cfg_table`` + ``world_cfg1.get_mesh_world()`` merge drops away
  because the v2 Scene supports cuboids and meshes in one object.

Run (inside the Isaac Sim Python env)::

    python -m curobo.examples.isaacsim.multi_arm_reacher
    python -m curobo.examples.isaacsim.multi_arm_reacher --robot dual_ur10e.yml --visualize_spheres
"""
from __future__ import annotations

# ---- Isaac Sim bootstrap (must come before any curobo / pxr imports) -------
# Pins warp-lang from pip so Isaac Sim's older extscache warp doesn't shadow it.
# See curobo/examples/isaacsim/bootstrap.py for the full explanation.
from curobo.examples.isaacsim import bootstrap  # noqa: F401

try:
    import isaacsim  # noqa: F401  (side effect: registers extensions)
except ImportError:
    pass

import argparse
from typing import Dict, List

# Third Party
import torch

# Warmup CUDA before SimulationApp starts (matches the v1 script).
_ = torch.zeros(4, device="cuda:0")

parser = argparse.ArgumentParser(
    description="cuRobo v2 multi-arm (multi-EEF) motion planner reacher (Isaac Sim)"
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
    default="dual_ur10e.yml",
    help="cuRobo v2 robot YAML. Must declare tool_frames with len >= 2.",
)
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Override robot_cfg.kinematics.external_asset_path at runtime.",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Directory to search for --robot when it is not one of the bundled names.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="Render the robot's collision spheres as green spheres under /curobo.",
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="Reactive mode: skip warmup and limit plan_pose to a single attempt.",
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

# ---- Everything that touches pxr / omni.isaac.* goes after SimulationApp --
import carb
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

from curobo.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path, get_scene_configs_path
from curobo.examples.isaacsim.helper import (
    add_extensions,
    add_robot_to_scene,
    stage_obstacles_as_scene,
)
from curobo.logging import log_and_raise, log_warn, setup_logger
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState, Pose


def _resolve_robot_cfg(robot: str, external_robot_configs_path: str | None) -> dict:
    """Load robot YAML from bundled configs or an override directory."""
    if external_robot_configs_path is not None:
        path = join_path(external_robot_configs_path, robot)
    else:
        path = join_path(str(get_robot_configs_path()), robot)
    return load_yaml(path)["robot_cfg"]


def _make_scene_from_yaml(name: str) -> Scene:
    """Load a bundled scene YAML (``content/configs/scene/<name>``) as a v2 Scene."""
    scene_dict = load_yaml(join_path(str(get_scene_configs_path()), name))
    return Scene.create(scene_dict)


def _retract_tool_poses(planner: MotionPlanner) -> Dict[str, Pose]:
    """Return the per-link retract pose (robot-base frame) for every tool frame.

    Uses ``planner.compute_kinematics`` on the robot's default joint state;
    the returned :class:`ToolPose` is indexed by frame name and yields a
    ``Pose`` of shape ``(1, 3/4)`` suitable for seeding the visual cubes.
    """
    default_js = planner.default_joint_state.clone().unsqueeze(0)
    kin = planner.compute_kinematics(default_js)
    return {name: kin.tool_poses[name] for name in planner.tool_frames}


def main() -> None:
    setup_logger("warn")

    # ---- Scene / stage setup ------------------------------------------------
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # ---- Robot config (v2 YAML, but same dict layout as v1) -----------------
    robot_cfg = _resolve_robot_cfg(args.robot, args.external_robot_configs_path)
    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path

    tool_frames_yaml = robot_cfg["kinematics"].get("tool_frames")
    if not tool_frames_yaml or len(tool_frames_yaml) < 2:
        log_and_raise(
            "multi_arm_reacher requires a robot YAML with tool_frames of length >= 2; "
            f"got tool_frames={tool_frames_yaml} in {args.robot}."
        )

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = (
        robot_cfg["kinematics"]["cspace"].get("default_joint_position")   # v2
        or robot_cfg["kinematics"]["cspace"].get("retract_config")        # v1 fallback
    )
    if default_config is None:
        log_and_raise("cspace.default_joint_position (or retract_config) missing from robot YAML")

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = None

    # ---- Initial collision scene (table, same as v1) ------------------------
    world_cfg = _make_scene_from_yaml("collision_table.yml")
    # v1 nudges the table 2 cm down; mirror that for behavioural parity.
    if world_cfg.cuboid:
        world_cfg.cuboid[0].pose[2] -= 0.02

    # ---- Build the v2 motion planner ---------------------------------------
    planner_cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        scene_model=world_cfg,
        collision_cache={"cuboid": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        num_trajopt_seeds=12,
        num_ik_seeds=32,
        use_cuda_graph=True,
        self_collision_check=True,
        # max_batch_size=1, multi_env=False, max_goalset=1  (defaults)
    )
    planner = MotionPlanner(planner_cfg)

    if not args.reactive:
        print("warming up...")
        planner.warmup(enable_graph=True, num_warmup_iterations=5)

    print(
        f"cuRobo v2 multi-EEF motion planner ready "
        f"(tool_frames={planner.tool_frames}, action_dim={planner.action_dim})"
    )

    add_extensions(simulation_app, args.headless_mode)

    # v2 planner knobs that used to live in MotionGenPlanConfig are now kwargs
    # on plan_pose() directly.
    max_attempts = 1 if args.reactive else 10
    enable_graph_attempt = 4

    my_world.scene.add_default_ground_plane()

    # ---- Per-tool-frame target cubes (retract-pose initial positions) -------
    retract_tool_poses = _retract_tool_poses(planner)

    primary_frame = planner.tool_frames[0]
    extra_frames: List[str] = list(planner.tool_frames[1:])

    primary_pose = retract_tool_poses[primary_frame]
    primary_pos = primary_pose.position.view(3).detach().cpu().numpy()
    primary_quat = primary_pose.quaternion.view(4).detach().cpu().numpy()
    target = cuboid.VisualCuboid(
        "/World/target",
        position=primary_pos,
        orientation=primary_quat,
        color=np.array([1.0, 0.0, 0.0]),
        size=0.05,
    )

    target_links: Dict[str, "cuboid.VisualCuboid"] = {}
    ignore_names: List[str] = []
    for link_name in extra_frames:
        link_pose = retract_tool_poses[link_name]
        pos = link_pose.position.view(3).detach().cpu().numpy()
        quat = link_pose.quaternion.view(4).detach().cpu().numpy()
        prim_path = f"/World/target_{link_name}"
        target_links[link_name] = cuboid.VisualCuboid(
            prim_path,
            position=pos,
            orientation=quat,
            color=np.array([0.5, 0.5, 0.0]),
            size=0.05,
        )
        ignore_names.append(prim_path)

    # ---- Main loop ---------------------------------------------------------
    past_pose = None
    target_pose = None
    cmd_plan: JointState | None = None
    cmd_idx = 0
    spheres: list | None = None
    past_cmd: JointState | None = None
    idx_list: list[int] = []
    i = 0

    # In headless mode there is no UI to hit "Play", so start the simulation
    # programmatically.
    if args.headless_mode is not None:
        my_world.play()

    while simulation_app.is_running():
        my_world.step(render=True)
        if args.max_steps is not None and my_world.current_time_step_index >= args.max_steps:
            print(f"Reached --max_steps={args.max_steps}, exiting.")
            break
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for _ in range(len(idx_list))]),
                joint_indices=idx_list,
            )
        if step_index < 20:
            continue

        # Re-sync obstacles from the stage (cheap; matches v1 cadence).
        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = stage_obstacles_as_scene(
                my_world.stage,
                reference_prim_path=robot_prim_path,
                only_paths=("/World",),
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ]
                + ignore_names,
            )
            planner.update_world(obstacles)
            print(f"Updated World: {len(obstacles.objects)} obstacles")
            carb.log_info("Synced cuRobo world from stage.")

        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position

        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_and_raise("Isaac Sim returned NaN joint positions.")

        device = planner.device_cfg.device
        dtype = planner.device_cfg.dtype
        cu_js = JointState(
            position=torch.as_tensor(sim_js.positions, device=device, dtype=dtype),
            velocity=torch.as_tensor(sim_js.velocities, device=device, dtype=dtype),
            acceleration=torch.as_tensor(sim_js.velocities, device=device, dtype=dtype) * 0.0,
            jerk=torch.as_tensor(sim_js.velocities, device=device, dtype=dtype) * 0.0,
            joint_names=sim_js_names,
        )
        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0
        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration

        cu_js = cu_js.reorder(planner.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = planner.kinematics.get_robot_as_spheres(cu_js.position)
            if spheres is None:
                spheres = []
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path=f"/curobo/robot_sphere_{si}",
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))

        robot_static = (np.max(np.abs(sim_js.velocities)) < 0.5) or args.reactive
        primary_moved = np.linalg.norm(cube_position - target_pose) > 1e-3
        primary_settled = np.linalg.norm(past_pose - cube_position) == 0.0

        if primary_moved and primary_settled and robot_static:
            print("Planning")

            # Build a per-link dict { frame_name: Pose(position=(1,3), quaternion=(1,4)) }
            # and pack into a single GoalToolPose of shape (1, 1, L, 1, 3/4).
            pose_dict: Dict[str, Pose] = {
                primary_frame: Pose(
                    position=torch.as_tensor(cube_position, device=device, dtype=dtype).view(1, 3),
                    quaternion=torch.as_tensor(cube_orientation, device=device, dtype=dtype).view(1, 4),
                )
            }
            for link_name, link_cube in target_links.items():
                c_p, c_rot = link_cube.get_world_pose()
                pose_dict[link_name] = Pose(
                    position=torch.as_tensor(c_p, device=device, dtype=dtype).view(1, 3),
                    quaternion=torch.as_tensor(c_rot, device=device, dtype=dtype).view(1, 4),
                )

            goal_tool_poses = GoalToolPose.from_poses(
                pose_dict,
                ordered_tool_frames=planner.tool_frames,
                num_goalset=1,
            )

            result = planner.plan_pose(
                goal_tool_poses,
                cu_js.unsqueeze(0),
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
            )

            succ = result is not None and bool(result.success.any().item())
            if succ:
                # v2 returns (1, 1, H, dof_full); drop batch+seed dims. The
                # interpolated plan already includes locked joints, so no
                # ``get_full_js`` step is needed (v1 had to re-add them).
                interp = result.get_interpolated_plan()       # (1, 1, H, dof_full)
                cmd_plan = interp.squeeze(0).squeeze(0)        # (H, dof_full)

                common_js_names: List[str] = []
                idx_list = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                cmd_plan = cmd_plan.reorder(common_js_names)
                cmd_idx = 0
            else:
                status = getattr(result, "status", "no solution")
                carb.log_warn(f"Plan did not converge: {status}")

            target_pose = cube_position

        past_pose = cube_position

        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy() if cmd_state.velocity is not None else None,
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None

    simulation_app.close()


if __name__ == "__main__":
    main()
