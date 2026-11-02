#!/usr/bin/env python3
"""Dump the camera calibration and static robot limits that Auriga's RoboDojo worker computes.

This is the provenance tool for ``assets/arx_x5_calibration.json``. It runs inside the RoboDojo
simulator environment (Isaac Sim + RoboDojo + the pinned Auriga checkout on PYTHONPATH), builds
the unmodified Auriga worker adapter (``auriga.robot.backends.robodojo.robot_worker.
RoboDojoAdapter``) for one task/layout, and writes one JSON file with:

- ``camera_calibration``: the worker's ``_camera_calibration()`` (K with fy := fx, head
  ``T_world_camera``, wrist ``T_link_camera`` / ``T_world_base``), plus the raw intrinsics;
- ``robot_model``: ``urdf_manifest`` of the URDF the worker reads;
- ``static``: control rate, physics dt, native ticks per action, step limit, joint/velocity
  limits, gripper constants, ``capabilities()``;
- ``samples``: worker observations (``robot_state`` + ``camera_pose_sample``) at the reset pose
  and after driving both arms to several joint configurations through the native action path.
  These are the ground truth for checking the policy-side FK (``auriga_xpl.calibration``).

Reproduce (inside a container of the evaluation image, one free GPU):

    cd /workspace/RoboDojo
    PYTHONPATH=<Auriga checkout>:/workspace/RoboDojo:/workspace/RoboDojo/XPolicyLab \
    CUDA_VISIBLE_DEVICES=0 ROBODOJO_RENDER_GPU=0 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
    /root/miniconda3/envs/RoboDojo/bin/python <adapter>/tools/dump_calibration.py \
        --task stack_bowls --seed 0 --layout-id 0 --out dumps/stack_bowls.json --run-dir runs/stack_bowls
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path


def _jsonable(value):
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return _jsonable(dict(value))
    except Exception:
        return repr(value)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strip_observation(observation):
    keep = {key: observation.get(key) for key in (
        "raw_observation_id", "sim_step", "step_domain", "control_dt_s", "sim_time_s",
        "robot_state", "camera_pose_sample")}
    keep["cameras"] = [{key: camera.get(key) for key in ("camera_id", "frame_id", "width", "height", "pixel_transform")}
                       for camera in observation.get("cameras", [])]
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robodojo-root", default="/workspace/RoboDojo")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout-id", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--poses", type=int, default=6, help="joint configurations to drive to after reset")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=12)
    parser.add_argument("--max-steps-per-pose", type=int, default=120)
    args = parser.parse_args()

    started = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": "auriga-xpl-calibration-dump-v1", "task": args.task, "seed": args.seed,
              "layout_id": args.layout_id, "started_unix": started, "status": "started"}

    import numpy as np
    from auriga.robot.backends.robodojo.robot_worker import RoboDojoAdapter
    import auriga

    namespace = argparse.Namespace(
        robodojo_root=args.robodojo_root, robot_model_id="arx-x5", gpu=args.gpu, task=args.task,
        run_dir=args.run_dir, seed=args.seed, layout_id=args.layout_id,
        observation_profile="rgb_proprio_calibrated_v1", tracking_guard=False, joint_replan=False,
        motion_safety_guard=True, trajectory_preview=True, motion_execution_mode="resampled",
        scene_truth_log=None)
    adapter = None
    try:
        adapter = RoboDojoAdapter(namespace)
        root = Path(args.robodojo_root)
        binding = adapter.robot_binding
        auriga_root = Path(auriga.__file__).resolve().parents[1]
        try:
            import subprocess
            auriga_commit = subprocess.check_output(["git", "-C", str(auriga_root), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            auriga_commit = None
        result["provenance"] = {
            "auriga_root": str(auriga_root), "auriga_commit": auriga_commit,
            "worker_versions": adapter.versions,
            "urdf_sha256": _sha256(root / binding.urdf_path),
            "robot_config_sha256": _sha256(root / binding.robot_config_path),
            "mesh_sha256": {p.name: _sha256(p) for p in sorted((root / binding.urdf_path).parent.joinpath("meshes").glob("*"))},
            "environment_config_sha256": _sha256(root / binding.environment_config_path),
            "camera_config_sha256": _sha256(root / "env_cfg/camera/camera_config.yml"),
        }
        result["camera_calibration"] = adapter.camera_calibration
        result["robot_model"] = adapter.robot_model

        # Raw intrinsics before the worker's fy := fx override, and per-camera config.
        manager = adapter.env.camera_manager
        raw = {}
        for index, camera_id in enumerate(manager.camera_names[0]):
            camera = manager.cameras[0][index]
            item = {"K_raw": _jsonable(camera.get_intrinsics_matrix()), "resolution": _jsonable(camera.get_resolution())}
            for getter in ("get_focal_length", "get_horizontal_aperture", "get_vertical_aperture", "get_clipping_range"):
                try:
                    item[getter[4:]] = _jsonable(getattr(camera, getter)())
                except Exception as error:  # noqa: BLE001
                    item[getter[4:]] = f"unavailable: {type(error).__name__}"
            try:
                item["config"] = _jsonable(manager.camera_config[camera_id].camera)
            except Exception as error:  # noqa: BLE001
                item["config"] = f"unavailable: {type(error).__name__}"
            raw[camera_id] = item
        result["camera_raw"] = raw

        # Static limits exactly as the worker derives them.
        robots = adapter.robot
        joint_info = {}
        for robot, asset in zip(robots.robot_list, robots.robot_key):
            name = f"{getattr(robot, 'robot_name', '?')}:{getattr(robot, 'arm_name', '?')}:{getattr(robot, 'type', '?')}"
            entry = {"type": getattr(robot, "type", None), "joint_names_all": list(asset.joint_names),
                     "body_names": list(asset.body_names)}
            for field in ("joint_pos_limits", "soft_joint_pos_limits", "joint_vel_limits", "joint_effort_limits",
                          "default_joint_pos", "joint_stiffness", "joint_damping"):
                tensor = getattr(asset.data, field, None)
                if tensor is not None:
                    entry[field] = _jsonable(tensor[0])
            try:
                entry["root_pose_w_minus_env_origin"] = _jsonable(
                    (asset.data.root_link_pose_w[0, :3] - adapter.env.env_origins[0]).cpu())
                entry["root_quat_wxyz"] = _jsonable(asset.data.root_link_pose_w[0, 3:7])
            except Exception as error:  # noqa: BLE001
                entry["root_pose"] = f"unavailable: {type(error).__name__}"
            if robot.type == "target":
                entry.update(arm_joint_names=list(robot.arm_joints_name), gripper_joint_names=list(robot.gripper_joints_name),
                             arm_joint_indices=_jsonable(robot.arm_joint_indices), gripper_scale=_jsonable(robot.gripper_scale),
                             gripper_move=_jsonable(robot.gripper_move))
            joint_info[name] = entry
        result["static"] = {
            "control_hz": adapter.control_hz, "native_step_ticks": adapter.native_step_ticks,
            "physics_dt_s": float(adapter.env.dt), "native_action_budget_limit": adapter.native_action_budget_limit,
            "arm_step_limits": _jsonable(adapter.arm_step_limits),
            "arm_velocity_step_limits": _jsonable(adapter.arm_velocity_step_limits),
            "arm_joint_limits": _jsonable(adapter.arm_joint_limits), "arm_joint_names": adapter.arm_joint_names,
            "gripper_step_limit": adapter.gripper_step_limit, "eef_gripper_dwell_step": adapter.eef_gripper_dwell_step,
            "env_origin_0": _jsonable(adapter.env.env_origins[0]),
            "robots": joint_info,
        }
        result["capabilities"] = _jsonable(adapter.capabilities())

        samples = []

        def sample(tag, commanded=None):
            observation = adapter.observe()
            entry = {"tag": tag, "commanded": commanded, **_strip_observation(observation)}
            diagnostics = {}
            for camera_id in ("cam_left_wrist", "cam_right_wrist"):
                pose = adapter._camera_pose(camera_id)
                diagnostics[camera_id] = pose.get("native_usd_pose_diagnostic", {}).get("translation_residual_m")
            entry["native_usd_translation_residual_m"] = diagnostics
            # Extra link poses straight from the simulator for FK cross-checks.
            links = {}
            for robot in robots.robot_list:
                if robot.type != "target":
                    continue
                side = robot.arm_name.split("_")[0]
                links[side] = {link: adapter._matrix(robots.get_link_pose(robot, link, [0], is_relative=True)[0]).tolist()
                               for link in ("base_link", "link6", "camera")
                               if link in robots.robot_key[robots.robot_list.index(robot)].body_names}
            entry["sim_link_poses"] = links
            samples.append(entry)

        sample("reset")
        rng = np.random.default_rng(args.rng_seed)
        sides = ("left", "right")
        limits = {side: np.asarray(adapter.arm_joint_limits[side], dtype=float) for side in sides}
        amplitudes = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
        for index in range(args.poses):
            state = adapter.state()
            current = {side: np.asarray(state["arms"][side]["q_rad"], dtype=float) for side in sides}
            reset_q = {side: np.asarray(samples[0]["robot_state"]["arms"][side]["q_rad"], dtype=float) for side in sides}
            amplitude = amplitudes[index % len(amplitudes)]
            target = {}
            for side in sides:
                low, high = limits[side][:, 0] + 0.05, limits[side][:, 1] - 0.05
                delta = rng.uniform(-amplitude, amplitude, 6)
                delta[1] = abs(delta[1]) * 0.6   # shoulder pitch: bias up/away from the table
                target[side] = np.clip(reset_q[side] + delta, low, high)
            grip = {side: float(index % 2) for side in sides}
            travel = max(float(np.max(np.abs(target[s] - current[s]) / np.maximum(adapter.arm_step_limits[s], 1e-6))) for s in sides)
            steps = int(min(max(1, math.ceil(travel)), args.max_steps_per_pose))
            ended = False
            for step in range(1, steps + args.settle_steps + 1):
                fraction = min(1.0, step / steps)
                row = []
                for side in sides:
                    row += [*(current[side] + (target[side] - current[side]) * fraction).tolist(), grip[side]]
                if not adapter._native_action(row):
                    ended = True
                    break
            sample(f"pose{index}", {"target_q": _jsonable(target), "gripper_goal": grip, "steps": steps, "ended": ended})
            if ended:
                break
        result["samples"] = samples
        result["status"] = "ok"
    except Exception as error:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_s"] = time.time() - started
        out_path.write_text(json.dumps(_jsonable(result), indent=1, allow_nan=False))
        print(f"dump_calibration: {args.task} status={result['status']} -> {out_path}", flush=True)
        if adapter is not None:
            try:
                adapter.close()
            except Exception:  # noqa: BLE001
                pass
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
