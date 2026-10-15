# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched (multi-env) version of ``mpc_example.py``.

Two Franka robots on independent world offsets, each with its own collision
scene, tracked in parallel by a single batched
:class:`curobo.model_predictive_control.ModelPredictiveControl` call
(``max_batch_size=2, multi_env=True``). Target cubes start on each robot's
retract-config EE pose, same as the single-env ``mpc_example.py``; drag either
cube slowly and that robot follows incrementally — the other robot keeps
tracking its own cube.

Structural pattern copied from ``batch_motion_gen_reacher.py``:

* Per-env root Xform ``/World/world_{i}`` at a world-space offset. Cubes and
  robots are children, so ``get_local_pose()`` returns robot-local coords
  (which is what MPC operates in).
* ``scene_model=[scene_dict_0, scene_dict_1]`` + ``multi_env=True`` allocates
  per-env obstacle buffers; each robot sees its own collision world.
* ``current_state`` is a ``(B, dof)`` stacked :class:`JointState`. Goal poses
  are stacked into ``(B, 1, L=1, G=1, 3/4)`` via :meth:`GoalToolPose.from_poses`.

MPC specifics kept from the single-env example:

* Cube starts at FK(retract) in env-local coords, zero initial delta.
* ``run_ik=False`` — pose-only tracking (avoids MPC's silent-IK-fail bug).
* ``optimize_action_sequence`` + command ``action_sequence[:, -1]`` — later
  knot is past the B-spline ease-in region; ``next_action`` (knot 0) would
  just keep commanding the current state.
* ``current_state`` is **chained from MPC's own last planned state**, not
  read from the live sim — PD lag would otherwise break the accumulation.

Usage::

    python -m curobo.examples.isaacsim.batch_mpc_example
    python -m curobo.examples.isaacsim.batch_mpc_example --headless_mode native --max_steps 300
"""

# Warp shim must load before SimulationApp.
from curobo.examples.isaacsim import bootstrap  # noqa: F401,E402

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import torch

_ = torch.zeros(4, device="cuda:0")

import argparse


parser = argparse.ArgumentParser(description="cuRobo v2 batched MPC reacher")
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket].",
)
parser.add_argument(
    "--robot",
    type=str,
    default="franka.yml",
    help="Robot YAML (under curobo/content/configs/robot/).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Exit after N simulator steps. Useful for smoke-testing in headless mode.",
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

import pathlib  # noqa: E402

import carb  # noqa: E402
import numpy as np  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import cuboid  # noqa: E402
from omni.isaac.core.utils.types import ArticulationAction  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from curobo.config_io import load_yaml  # noqa: E402
from curobo.content import get_robot_configs_path  # noqa: E402
from curobo.logging import setup_logger  # noqa: E402
from curobo.model_predictive_control import (  # noqa: E402
    ModelPredictiveControl,
    ModelPredictiveControlCfg,
)
from curobo.scene import Scene  # noqa: E402
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose  # noqa: E402
from curobo.viewer import UsdWriter  # noqa: E402

from curobo.examples.isaacsim.helper import (  # noqa: E402
    add_extensions,
    add_robot_to_scene,
)


_SCENES_DIR = pathlib.Path(__file__).parent / "scenes"


def _load_scene_dict(name: str) -> dict:
    """Load a scene YAML from the example's local ``scenes/`` dir."""
    path = _SCENES_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"scene YAML not found: {path}")
    return load_yaml(str(path))


def _goal_from_stacked_pose(mpc: ModelPredictiveControl, pose: Pose) -> GoalToolPose:
    """Wrap a stacked ``(B, 3)/(B, 4)`` Pose as ``(B, 1, L=1, G=1, 3/4)``."""
    return GoalToolPose.from_poses(
        {mpc.tool_frames[0]: pose},
        ordered_tool_frames=mpc.tool_frames,
        num_goalset=1,
    )


def main():
    setup_logger("warn")
    device_cfg = DeviceCfg()

    n_envs = 2
    offset_y = 2.5
    scene_files = ["collision_test.yml", "collision_thin_walls.yml"]
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    stage = my_world.stage
    stage.DefinePrim("/World", "Xform").GetStage().SetDefaultPrim(
        stage.GetPrimAtPath("/World")
    )
    stage.DefinePrim("/curobo", "Xform")

    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(str(robot_cfg_path / args.robot))["robot_cfg"]
    cspace = robot_cfg["kinematics"]["cspace"]
    j_names = cspace["joint_names"]
    default_config = cspace.get("default_joint_position", cspace.get("retract_config"))
    if default_config is None:
        raise KeyError(
            "cspace.default_joint_position (or retract_config) missing from robot YAML"
        )

    # --- Per-env Xforms + robots (targets added after MPC is ready) ----------
    usd_writer = UsdWriter()
    usd_writer.load_stage(stage)

    env_offsets = []
    robot_list = []
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
        robot_list.append(robot)

    my_world.initialize_physics()

    # --- Per-env collision scenes (matches batch_motion_gen_reacher) ----------
    # Two distinct worlds (collision_test + collision_thin_walls) shipped under
    # this example's ``scenes/`` directory. v2 batch-env MPC accepts
    # ``scene_model`` as a list of scene dicts; the planner allocates per-env
    # obstacle buffers and each env is collision-checked independently. Nudge
    # each world's first object down 2cm so it clears the ground plane (same
    # cosmetic fix v1 applied).
    scene_dicts = [_load_scene_dict(name) for name in scene_files]
    for sd in scene_dicts:
        objs = list((sd.get("cuboid") or {}).values()) + list(
            (sd.get("mesh") or {}).values()
        )
        if objs and "pose" in objs[0]:
            objs[0]["pose"][2] -= 0.02

    # Render each env's obstacles in the stage under its sub-root so the user
    # can see what the planner is colliding against.
    for i, sd in enumerate(scene_dicts):
        usd_writer.add_world_to_stage(
            Scene.create(sd), base_frame=f"/World/world_{i}",
        )

    mpc_cfg = ModelPredictiveControlCfg.create(
        robot=args.robot,
        scene_model=scene_dicts,
        use_cuda_graph=True,
        self_collision_check=True,
        collision_cache={"cuboid": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimization_dt=0.02,
        interpolation_steps=4,
        device_cfg=device_cfg,
        max_batch_size=n_envs,
        multi_env=True,
        max_goalset=1,
    )
    mpc = ModelPredictiveControl(mpc_cfg)

    # --- Stacked (B, dof) current_state starting at retract ------------------
    default_t = torch.as_tensor(
        default_config, device=device_cfg.device, dtype=device_cfg.dtype,
    )[:mpc.action_dim]
    current_state = JointState.from_position(
        default_t.unsqueeze(0).repeat(n_envs, 1).clone(),
        joint_names=list(mpc.joint_names),
    )
    current_state.velocity = torch.zeros_like(current_state.position)
    current_state.acceleration = torch.zeros_like(current_state.position)

    mpc.setup(current_state)

    # FK at retract → initial goal and initial cube pose (identical per env,
    # because all robots share the retract config).
    kin = mpc.compute_kinematics(current_state)
    retract_pose = kin.tool_poses.to_dict()[mpc.tool_frames[0]]  # (B, 3)/(B, 4)

    # Spawn a cube at each env's retract EE. Cube is a child of that env's
    # root Xform, so ``get_local_pose`` returns robot-local coords directly.
    target_list = []
    local_pos_np = retract_pose.position.cpu().numpy()
    local_quat_np = retract_pose.quaternion.cpu().numpy()
    for i in range(n_envs):
        world_pos = local_pos_np[i] + env_offsets[i]
        target = cuboid.VisualCuboid(
            f"/World/world_{i}/target",
            position=world_pos,
            orientation=local_quat_np[i],
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        target_list.append(target)

    mpc.update_goal_tool_poses(
        _goal_from_stacked_pose(
            mpc,
            Pose(
                position=retract_pose.position.view(n_envs, 3).clone(),
                quaternion=retract_pose.quaternion.view(n_envs, 4).clone(),
            ),
        ),
        run_ik=False,
    )

    # Warmup CUDA graphs.
    mpc.optimize_action_sequence(current_state)

    articulation_controllers = [r.get_articulation_controller() for r in robot_list]
    add_extensions(simulation_app, args.headless_mode)

    past_pose = [None] * n_envs
    step = 0

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
            for robot in robot_list:
                robot._articulation_view.initialize()
                idx_list = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, idx_list)
                robot._articulation_view.set_max_efforts(
                    values=np.array([5000.0] * len(idx_list)),
                    joint_indices=idx_list,
                )
        step += 1

        # --- Collect per-env target poses (robot-local via parent Xform) -----
        any_moved = False
        stacked_pos = []
        stacked_quat = []
        for i, t in enumerate(target_list):
            local_pos, local_quat = t.get_local_pose()
            stacked_pos.append(local_pos)
            stacked_quat.append(local_quat)
            if past_pose[i] is None:
                past_pose[i] = local_pos + 1.0
            if np.linalg.norm(local_pos - past_pose[i]) > 1e-3:
                any_moved = True
            past_pose[i] = local_pos

        if any_moved:
            ik_goal = Pose(
                position=torch.as_tensor(
                    np.stack(stacked_pos), device=device_cfg.device, dtype=device_cfg.dtype,
                ),
                quaternion=torch.as_tensor(
                    np.stack(stacked_quat), device=device_cfg.device, dtype=device_cfg.dtype,
                ),
            )
            mpc.update_goal_tool_poses(_goal_from_stacked_pose(mpc, ik_goal), run_ik=False)

        # --- Batched MPC step ------------------------------------------------
        mpc_result = mpc.optimize_action_sequence(current_state)
        if mpc_result.action_sequence is None or mpc_result.action_sequence.position.shape[1] == 0:
            carb.log_warn("MPC returned no action sequence; skipping command.")
            continue

        act_seq = mpc_result.action_sequence
        cmd_position = act_seq.position[:, -1, :]     # (B, dof)

        # Chain current_state from MPC's own plan (viser-style). See
        # mpc_example.py for why feeding real sim joints back into MPC fails.
        current_state = JointState.from_position(
            cmd_position.clone(), joint_names=list(mpc.joint_names),
        )
        current_state.velocity = act_seq.velocity[:, -1, :].clone()
        current_state.acceleration = act_seq.acceleration[:, -1, :].clone()

        # --- Fan out per-env commands to the two articulations ---------------
        sim_js_names = robot_list[0].dof_names
        idx_list = []
        common_js_names = []
        for x in sim_js_names:
            if x in mpc.joint_names:
                idx_list.append(robot_list[0].get_dof_index(x))
                common_js_names.append(x)

        reordered = JointState.from_position(
            cmd_position.clone(), joint_names=list(mpc.joint_names),
        ).reorder(common_js_names)

        for i in range(n_envs):
            art_action = ArticulationAction(
                reordered.position[i].cpu().numpy(),
                joint_indices=idx_list,
            )
            articulation_controllers[i].apply_action(art_action)

        if step_index % 1000 == 0:
            pose_err = mpc_result.position_error
            err_str = (
                f"{pose_err.view(-1).cpu().numpy().tolist()}"
                if pose_err is not None else "N/A"
            )
            print(f"pose_error={err_str}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
