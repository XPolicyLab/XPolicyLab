# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuRobo 2.0 dual-arm humanoid IK — ``solve_batch_env`` with two tool frames.

Port of v1 ``multi_arm_humanoid_parallel_ik.py`` onto the v2 API.  Two
Unitree G1 robots, each in its own collision env, each solving a two-link IK
query (left and right palms) from its own pair of target cubes — all in one
``ik.solve_pose`` call.

This is the interesting corner of the variant table because it exercises
*both* axes at once:

* ``multi_env=True``   →  per-problem collision world (N=2 envs)
* ``L > 1``            →  multi-tool-frame goal (left + right hand palms)

v1 → v2 surface changes used here (see ``MIGRATION_V1_TO_V2.md``):

+-----------------------------------------------+--------------------------------------------------------+
| v1                                            | v2                                                     |
+===============================================+========================================================+
| ``IKSolver.solve_batch_env(goal_pose=Pose,``  | ``ik.solve_pose(GoalToolPose (B,1,L,1,3/4),``          |
| ``  link_poses={link: Pose, …},``             | ``             current_state: JointState (B,dof))``    |
| ``  retract_config=(B, dof),``                |                                                        |
| ``  seed_config=(B, 1, dof))``                |                                                        |
+-----------------------------------------------+--------------------------------------------------------+
| Pass ``List[WorldConfig]`` to                 | Pass a template ``scene_model`` + override             |
| ``IKSolverConfig.load_from_robot_config``     | ``ik_cfg.core_cfg.scene_collision_cfg.scene_model``    |
| for per-env worlds                            | with a ``List[Scene]`` before constructing the solver  |
+-----------------------------------------------+--------------------------------------------------------+
| ``ee_link`` + ``link_names`` in robot YAML    | ``tool_frames: [...]`` list in robot YAML (or          |
|                                               | injected at runtime via the ``robot`` dict below)      |
+-----------------------------------------------+--------------------------------------------------------+

Robot config.  Uses the v2 YAMLs derived from v1's ``magicsim_g1_*_simple*``
configs:

* ``magicsim_g1_simple.yml``        — fixed base, 14 active DOF (arms only).
* ``magicsim_g1_simple_mobile.yml`` — floating base (6 virtual joints kept
  active), 20 active DOF.

Both ship with ``tool_frames: [right_hand_palm_link, left_hand_palm_link]``
(right first, left second) and ``lock_joints`` pinning every non-arm joint
at its default — the v2 equivalent of v1's ``ignore_joints`` dict.

Usage::

    python -m curobo.examples.isaacsim.ik_humanoid_batched_env
    python -m curobo.examples.isaacsim.ik_humanoid_batched_env --mobile
    python -m curobo.examples.isaacsim.ik_humanoid_batched_env --n_envs 2 --visualize_spheres
"""

# Pin pip's warp-lang 1.12 BEFORE any ``omni.*`` / ``isaacsim.*`` /
# ``curobo._src`` import. See ``bootstrap.py`` for why.
from curobo.examples.isaacsim import bootstrap  # noqa: F401

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

from typing import Dict, List

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
    "--load_from_usd",
    action="store_true",
    help=("Reference a pre-built robot USD (``kinematics.usd_path``) instead of "
          "running the Isaac Sim URDF importer. Faster startup, but requires the "
          "USD file to already exist under the asset tree."),
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

from curobo.config_io import load_yaml  # noqa: E402
from curobo.content import get_robot_configs_path  # noqa: E402
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg  # noqa: E402
from curobo.logging import setup_logger  # noqa: E402
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
    ik: InverseKinematics,
    link_positions: Dict[str, torch.Tensor],
    link_quaternions: Dict[str, torch.Tensor],
) -> GoalToolPose:
    """Wrap per-link ``(B, 3)`` / ``(B, 4)`` tensors into a ``(B, 1, L, 1, …)`` goal."""
    poses = {
        name: Pose(position=link_positions[name], quaternion=link_quaternions[name])
        for name in ik.tool_frames
    }
    return GoalToolPose.from_poses(
        poses,
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
    # No default ground plane — v1's ``multi_arm_humanoid_parallel_ik.py``
    # intentionally leaves it out so the legs aren't penetrating a physics
    # collider that would fight the set_joint_positions-driven updates.
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
    # ``kinematics:``); pop them out before the solver ever sees them, since
    # ``KinematicsLoaderCfg`` is a strict dataclass.
    usd_path = robot_yaml.pop("usd_path", None)
    usd_robot_root = robot_yaml.pop("usd_robot_root", None)
    # v2 YAML style for these files is unwrapped (top-level ``kinematics:``)
    # — wrap with ``robot_cfg:`` so ``InverseKinematicsCfg.create`` handles
    # both shipping styles uniformly.
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
    # IK solver: multi_env=True, L=2, G=1
    # ------------------------------------------------------------------
    # (Built first so we can use ``ik.compute_kinematics(default_js)`` to
    #  place the initial target cubes at the retract pose of each arm.)
    #
    # Note: we do NOT pass ``scene_model`` — v1's humanoid parallel-IK demo
    # used empty ``WorldConfig()`` per env, so only self-collision matters.
    # With ``scene_model=None``, ``core_cfg.scene_collision_cfg`` stays
    # ``None`` and there's no per-env scene cache to override (or overflow).
    # ``num_envs=max_batch_size`` still flows through to ``RobotCfg.create``
    # so per-env self-collision link_spheres are allocated correctly.
    ik_cfg = InverseKinematicsCfg.create(
        robot=robot_cfg_full,
        num_seeds=64,                      # v1 used 50; we need headroom for 14 DOF
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        self_collision_check=True,
        use_cuda_graph=False,              # v1 also disabled for the humanoid IK
        max_batch_size=n_envs,
        multi_env=True,
        max_goalset=1,
        device_cfg=device_cfg,
    )
    ik = InverseKinematics(ik_cfg)
    retract_kin = ik.compute_kinematics(ik.default_joint_state.clone().unsqueeze(0))
    retract_tool_poses = {
        name: retract_kin.tool_poses[name] for name in ik.tool_frames
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
        for link_idx, link_name in enumerate(ik.tool_frames):
            # retract_tool_poses[link] position/quaternion are (1, 3/4) in
            # robot-base frame; add base_position to get world-frame placement
            # and keep the orientation identity-identical to the retract pose.
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
    print(f"Loaded {n_envs} G1 envs ({robot_file}), tool_frames={ik.tool_frames}")

    # Warmup with matching shape so the first real solve doesn't trigger
    # ``reset_cuda_graph`` on a shape mismatch.
    default_js_1 = ik.default_joint_state.clone().unsqueeze(0)
    warmup_state = _expand_joint_state(default_js_1, n_envs)

    warmup_positions = {
        name: retract_tool_poses[name].position.view(1, 3).expand(n_envs, 3).contiguous()
        for name in ik.tool_frames
    }
    warmup_quaternions = {
        name: retract_tool_poses[name].quaternion.view(1, 4).expand(n_envs, 4).contiguous()
        for name in ik.tool_frames
    }
    warmup_goal = _build_goal_tool_pose(ik, warmup_positions, warmup_quaternions)
    ik.solve_pose(warmup_goal, current_state=warmup_state)

    print(f"cuRobo humanoid batch_env IK ready "
          f"(n_envs={n_envs}, tool_frames={ik.tool_frames}, "
          f"action_dim={ik.action_dim})")
    add_extensions(simulation_app, args.headless_mode)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    past_goal_pos = None
    target_goal_pos = None
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
                # ``robot.get_dof_index`` raises KeyError (not returns None)
                # for joints absent from the articulation — this happens e.g.
                # when the cspace lists a v2-only virtual joint that the
                # URDF/Isaac Sim view doesn't expose.  Skip those.
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

        # ---- Read per-env target poses (local, i.e. robot-base frame) ----
        link_positions: Dict[str, torch.Tensor] = {}
        link_quaternions: Dict[str, torch.Tensor] = {}
        positions_np = {name: np.zeros((n_envs, 3), dtype=np.float32) for name in ik.tool_frames}
        quaternions_np = {name: np.zeros((n_envs, 4), dtype=np.float32) for name in ik.tool_frames}
        for env_idx in range(n_envs):
            for link_name in ik.tool_frames:
                p, q = per_env_targets[env_idx][link_name].get_local_pose()
                positions_np[link_name][env_idx] = p
                quaternions_np[link_name][env_idx] = q
        for name in ik.tool_frames:
            link_positions[name] = device_cfg.to_device(positions_np[name])
            link_quaternions[name] = device_cfg.to_device(quaternions_np[name])

        # Stack primary EE positions once for the move-detection guard below.
        primary_positions = link_positions[ik.tool_frames[0]]
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
        ).reorder(ik.joint_names)

        moved_vs_target = (primary_positions - target_goal_pos).norm(dim=-1) > 1e-2
        still_vs_last_frame = (primary_positions - past_goal_pos).norm(dim=-1) < 1e-4
        robots_static = batched_js.velocity.abs().max() < 0.2

        should_solve = (
            bool(moved_vs_target.any().item())
            and bool(still_vs_last_frame.all().item())
            and bool(robots_static.item())
        )

        if should_solve:
            goal = _build_goal_tool_pose(ik, link_positions, link_quaternions)
            result = ik.solve_pose(goal, current_state=batched_js)
            target_goal_pos = primary_positions.clone()

            success_flat = result.success.view(n_envs).detach().cpu().numpy().astype(bool)
            pos_err = result.position_error.view(n_envs, -1).detach().cpu().numpy()
            print(
                "IK: envs=%d success=%d/%d pos_err=%s solve_time=%.4fs"
                % (
                    n_envs,
                    int(success_flat.sum()),
                    n_envs,
                    np.array2string(pos_err, precision=4, suppress_small=True),
                    float(result.solve_time),
                )
            )

            if not success_flat.any():
                carb.log_warn(
                    "Dual-arm IK did not converge for any env "
                    f"(pos_err={pos_err}, rot_err="
                    f"{result.rotation_error.view(n_envs, -1).detach().cpu().numpy()})"
                )
            else:
                ik_status = getattr(result, "status", None)
                sol_positions = result.js_solution.position[:, 0, :]
                sol_joint_names = result.js_solution.joint_names
                for env_idx in range(n_envs):
                    if not success_flat[env_idx]:
                        reason = ik_status if ik_status else "IK seed did not converge"
                        carb.log_warn(
                            f"env_{env_idx} dual-arm IK failed — {reason} "
                            f"(goal likely unreachable or in collision)"
                        )
                        continue
                    env_robot = robots[env_idx]
                    idx_list_env = []
                    common = []
                    for x in sim_js_names:
                        if x in sol_joint_names:
                            idx_list_env.append(env_robot.get_dof_index(x))
                            common.append(x)
                    single_env_js = JointState.from_position(
                        sol_positions[env_idx:env_idx + 1],
                        joint_names=sol_joint_names,
                    ).reorder(common)
                    env_robot.set_joint_positions(
                        single_env_js.position.cpu().numpy().reshape(-1),
                        idx_list_env,
                    )

        past_goal_pos = primary_positions.clone()

    simulation_app.close()


if __name__ == "__main__":
    main()
