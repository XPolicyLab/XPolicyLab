"""Observation contracts: what a robot connection lets a policy see.

A simulator worker can report more than an evaluation host hands a policy. The
XPolicyLab contract is what RoboDojo's official client gives (checked 2026-10-08,
RoboDojo cb2a37d): camera colors, measured arm joints, link6 poses, the instruction,
and for each gripper only the value it was last commanded; no camera calibration, no
depth, no step limit. A worker served with a contract (`harness serve --contract
xpolicylab --embodiment arx_x5`) reports only that, so development runs see what an
evaluated policy will:

- grippers: the last commanded value instead of the measured opening;
- cameras: intrinsics and camera_to_world from the embodiment's calibration (the head
  camera fixed in the world, wrist cameras fixed on link6), not the simulator's;
- the step limit is left out of the profile and the state;
- the task is not named: a system under this contract has no task descriptions
  (config/systems.py), the instruction is all it is told.

The profile says which contract applies (observation_contract, gripper_feedback), so
skills that need a measured gripper can check differently. Standard library only.
"""

import copy
import math


CONTRACTS = ("native", "xpolicylab")


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def pose_matrix(pose):
    """The 4x4 homogeneous transform (nested lists) of a pose [x, y, z, qw, qx, qy, qz];
    the quaternion is normalized first."""
    x, y, z, w, i, j, k = (float(v) for v in pose)
    norm = math.sqrt(w * w + i * i + j * j + k * k)
    w, i, j, k = w / norm, i / norm, j / norm, k / norm
    return [[1 - 2 * (j * j + k * k), 2 * (i * j - w * k), 2 * (i * k + w * j), x],
            [2 * (i * j + w * k), 1 - 2 * (i * i + k * k), 2 * (j * k - w * i), y],
            [2 * (i * k - w * j), 2 * (j * k + w * i), 1 - 2 * (i * i + j * j), z],
            [0.0, 0.0, 0.0, 1.0]]


class ContractFilter:
    """Applied by the worker server to everything its environment returns."""

    def __init__(self, contract: str, embodiment: dict):
        if contract not in CONTRACTS:
            raise ValueError(f"contract must be one of {CONTRACTS}")
        self.contract = contract
        self.embodiment = embodiment
        self.grippers = [int(group["gripper"]) for group in embodiment["groups"].values()]
        self.commanded = {}  # gripper index -> last commanded value
        self.calibration = embodiment.get("calibration", {})

    @property
    def active(self) -> bool:
        """Whether the filter changes anything (every contract but native)."""
        return self.contract != "native"

    def apply(self, method: str, arguments: dict, result):
        """What worker method `method` returned, as the contract shows it (unchanged under
        native); resets and executed ticks update the last commanded gripper values. States
        report commanded grippers and no step limit, camera views the embodiment's
        calibration, and the profile names the contract; other methods pass unchanged."""
        if not self.active:
            return result
        result = copy.deepcopy(result)
        if method == "reset":
            # An episode starts with the grippers where they are (RoboDojo's first
            # observation reports the reset position as the last command).
            state = result.get("state") or {}
            positions = state.get("joint_position") or []
            self.commanded = {index: float(positions[index]) for index in self.grippers if index < len(positions)}
            result["state"] = self._state(state)
            return result
        if method in ("profile", "action_spec"):
            result.pop("step_limit", None)
            result.update(observation_contract=self.contract, gripper_feedback="commanded",
                          calibration_source="embodiment")
            return result
        if method == "tick":
            if result.get("executed"):
                command = (result.get("shield") or {}).get("command") or arguments.get("command") or []
                for index in self.grippers:
                    if index < len(command):
                        self.commanded[index] = float(command[index])
            result["state"] = self._state(result.get("state"))
            if result.get("observation") is not None:
                result["observation"] = self._observe(result["observation"])
            return result
        if method == "state":
            return self._state(result)
        if method == "observe":
            return self._observe(result)
        if method == "policy_obs":
            result["robot_state"] = self._state(result.get("robot_state"))
            return result
        return result

    def _state(self, state):
        if not isinstance(state, dict):
            return state
        state = dict(state)
        state.pop("step_limit", None)
        positions = list(state.get("joint_position") or [])
        for index, value in self.commanded.items():
            if index < len(positions):
                positions[index] = value
        if positions:
            state["joint_position"] = positions
        return state

    def _observe(self, data):
        data = dict(data)
        state = self._state(data.get("state"))
        data["state"] = state
        intrinsics, extrinsics = dict(data.get("intrinsics") or {}), dict(data.get("camera_to_world") or {})
        for camera in list(intrinsics):
            calibration = self.calibration.get(camera)
            if calibration is None:
                raise ValueError(f"The embodiment has no calibration for {camera}")
            intrinsics[camera] = calibration["intrinsics"]
            if "camera_to_world" in calibration:
                extrinsics[camera] = calibration["camera_to_world"]
            else:
                side = calibration["link"].split("_")[0]
                extrinsics[camera] = _matmul(pose_matrix(state[f"{side}_ee_pose"]), calibration["camera_on_link6"])
        data["intrinsics"], data["camera_to_world"] = intrinsics, extrinsics
        return data
