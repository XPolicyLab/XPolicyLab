"""OFFLINE grasp probe for the pack task (oracle runtime only).

Physically tests the CAD grasp choices and a few variants per object: pinch, lift,
tilt about the object's long axis, hold, set back down. The analysis of the object
truth trace (tools/probe_report.py) tells which pinches pivot, wedge out or slip.
"""
import math

import numpy as np

from ..dual import grip_steps
from ..executor import _log
from . import pack_random as pr

PROBE_TILT = 0.6
# (long_shift, tcp_shift) relative to the scorer's choice.
VARIANTS = {
    "shoe": [(0.0, 0.0), (0.012, 0.014), (0.0, 0.008), (0.0, -0.006)],
    "car": [(0.0, 0.0), (0.0, 0.006), (0.0, 0.012)],
    "hammer": [(0.0, 0.0), (0.0, 0.006)],
    "electric_toothbrush": [(0.0, 0.0)],
}


def grasp_probe(ctx):
    scene = pr.perceive(ctx)
    planner = pr.PackPlanner(ctx, scene)
    executor = pr.PackExecutor(ctx, planner)
    kinds = [o["kind"] for o in scene["objects"]]
    sequence = [(kind, variant) for kind in pr.FAMILIES if kind in kinds for variant in VARIANTS[kind]]
    last_arm = None
    for index, (kind, variant) in enumerate(sequence):
        if ctx.steps > 1300 - 150:
            break
        pr.GRASP_VARIANT = variant
        try:
            used = yield from probe_one(ctx, executor, index, kind, variant, last_arm)
            if used is not None:
                last_arm = used
        except Exception as error:  # keep probing the other variants
            _log(ctx, "probe_error", index=index, kind=kind, error=f"{type(error).__name__}: {error}"[:160])
            for arm in ("left", "right"):
                if ctx.gripper_targets[arm] < 0.99:
                    yield from executor.run_arm(arm, grip_steps(ctx, arm, 1.0, 4))
            try:
                yield from executor.run_arm("left", executor.go_home("left", 12))
                yield from executor.run_arm("right", executor.go_home("right", 12))
            except Exception as home_error:
                _log(ctx, "probe_home_error", error=str(home_error)[:120])
        finally:
            pr.GRASP_VARIANT = None
    if last_arm is not None:
        yield from executor.run_arm(last_arm, executor.go_home(last_arm, 12))
    ctx.memory["phase"] = "probe_done"
    for _ in range(4):
        yield ctx.hold_action()


def probe_one(ctx, executor, index, kind, variant, last_arm):
    fresh = pr.perceive(ctx)
    obj = next((o for o in fresh["objects"] if o["kind"] == kind), None)
    if obj is None:
        _log(ctx, "probe_missing", index=index, kind=kind)
        return
    spec = pr.grasp_spec(obj["model_key"])
    arms = ("left", "right") if obj["bottom"][0] < 0 else ("right", "left")
    arm = yaw = None
    pitch = 0.0
    for candidate in arms:
        grasp = executor.choose_grasp(candidate, obj)
        if grasp is not None:
            yaw, pitch = grasp
            arm = candidate
            break
    if arm is None:
        _log(ctx, "probe_unreachable", index=index, kind=kind, variant=list(variant))
        return None
    if last_arm is not None and last_arm != arm:
        yield from executor.run_arm(last_arm, executor.go_home(last_arm, 12))
    obj = dict(obj, gripper_yaw=yaw, pitch=pitch)
    _log(ctx, "probe_begin", index=index, kind=kind, model=obj["model_key"], variant=list(variant), arm=arm, pitch=pitch,
         tcp_height=spec["tcp_height"], grasp_xy=spec["grasp_xy"].tolist(), width=spec["width"], long_axis=spec["long_axis"],
         patch=spec["patch_height"], clamped=spec["clamped"], pinch_z=spec["pinch_z"], com=spec["com"].tolist(),
         taper=spec["taper_deg"], bottom=obj["bottom"].tolist(), yaw=obj["yaw"])
    ctx.memory["phase"] = f"probe_{index}_{kind}"
    held = yield from executor.run_arm(arm, executor.pick(arm, obj))
    reading = float(ctx.state[f"{arm}_ee_joint_state"][0])
    _log(ctx, "probe_lifted", index=index, kind=kind, held=bool(held), reading=reading,
         gripper_yaw=obj["gripper_yaw"])
    if held:
        long_dir = pr.object_axes(spec, obj["yaw"])
        upright = executor.pick_poses(obj, obj["gripper_yaw"])["rotation"]
        try:
            for sign in (1.0, -1.0):
                tilted = pr.tilt_rotation(obj["gripper_yaw"], long_dir, sign * PROBE_TILT)
                if executor.models[arm].solve(executor.tcp(arm)[:3, 3], tilted, ctx.joint_targets[arm], prefer_small_roll=True) is not None:
                    break
            yield from executor.run_arm(arm, executor.rotate_in_place(arm, tilted))
            yield from executor.run_arm(arm, executor.settle(arm, 8))
            _log(ctx, "probe_tilted", index=index, kind=kind, reading=float(ctx.state[f"{arm}_ee_joint_state"][0]))
            yield from executor.run_arm(arm, executor.rotate_in_place(arm, upright))
            yield from executor.run_arm(arm, executor.settle(arm, 4))
            _log(ctx, "probe_upright", index=index, kind=kind, reading=float(ctx.state[f"{arm}_ee_joint_state"][0]))
        except RuntimeError as error:
            _log(ctx, "probe_tilt_skipped", index=index, kind=kind, error=str(error)[:100])
        poses = executor.pick_poses(obj, obj["gripper_yaw"])
        rotation = upright
        yield from executor.run_arm(arm, executor.linear_move(arm, poses["grasp"] + [0, 0, 0.012], rotation, step_m=0.025, allow_partial=True))
        yield from executor.run_arm(arm, grip_steps(ctx, arm, 1.0, 4))
        yield from executor.run_arm(arm, executor.settle(arm, 2))
        yield from executor.run_arm(arm, executor.linear_move(arm, poses["hover"], rotation, step_m=0.03, allow_partial=True))
    _log(ctx, "probe_end", index=index, kind=kind)
    return arm
