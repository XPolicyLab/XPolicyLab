"""Lossless XPolicyLab boundary for the archived C05 RoboTwin checkpoint.

The caller supplies get_robot_state_schema('joint', metadata) from XPolicyLab.
Normalization and block-relative action reconstruction belong to the frozen
native policy, not this boundary. No image decoding or color conversion here.
"""
import numpy as np

STATE_KEYS = ("left_arm_joint_state", "left_ee_joint_state",
              "right_arm_joint_state", "right_ee_joint_state")
CAMERAS = {"cam_head": "cam_high", "cam_left_wrist": "cam_left_wrist",
           "cam_right_wrist": "cam_right_wrist"}


def validate_schema(schema, checkpoint_contract):
    if tuple(schema) != STATE_KEYS:
        raise ValueError("C05 requires left arm/gripper, right arm/gripper packing order")
    if dict(schema) != checkpoint_contract["state_schema"]:
        raise ValueError("Runtime robot dimensions differ from the checkpoint contract")


def native_observation(obs, schema):
    instruction = obs.get("instruction")
    if instruction is None:
        instruction = obs.get("instructions")
        if isinstance(instruction, (list, tuple)) and len(instruction) == 1:
            instruction = instruction[0]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Expected one explicit, non-empty instruction")
    parts = []
    for key, width in schema.items():
        a = np.asarray(obs["state"][key])
        if a.shape != (width,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError("Invalid state field: " + key)
        parts.append(a)
    result = {"task": instruction, "observation.state": np.concatenate(parts).astype(np.float32)}
    for source, target in CAMERAS.items():
        a = obs["vision"][source]["color"]
        if not isinstance(a, np.ndarray) or a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError("Expected server-decoded uint8 HWC RGB: " + source)
        if min(a.shape[:2]) < 1:
            raise ValueError("Empty camera image")
        # Native preprocessing mutates its input dictionary; own the arrays too.
        result["observation.images." + target] = a.copy()
    return result


def action_chunk(native_result, schema, execute_steps):
    a = np.asarray(native_result["action"])
    if a.ndim != 2 or a.shape[1] != sum(schema.values()):
        raise ValueError("Native policy must return [time, total_joint_dimensions]")
    if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError("Non-finite or non-numeric policy action")
    if isinstance(execute_steps, bool) or not isinstance(execute_steps, int) or not 1 <= execute_steps <= len(a):
        raise ValueError("Execution length exceeds the available native chunk")
    result = []
    for row in a[:execute_steps]:
        step, start = {}, 0
        for key, width in schema.items():
            step[key] = row[start:start + width].copy()
            start += width
        result.append(step)
    return result
