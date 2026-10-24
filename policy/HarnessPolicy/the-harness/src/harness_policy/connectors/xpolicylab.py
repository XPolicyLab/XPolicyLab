"""The harness as an XPolicyLab policy: evaluated by RoboDojo's (or RoboTwin's, or a real
robot's) own client through XPolicyLab's policy-server protocol.

XPolicyLab drives: its client calls reset(), then update_obs(obs) and get_action() ->
a chunk of actions, executes them, and sends the observation after each. The harness
drives too: the model's programs ask for one control tick at a time. The bridge here
meets the two halves:

- To the session and the driver it is a robot worker (the Robot API: profile, state,
  tick, observe, policy_obs, fk, ik, plan, outcome) on a local port, so kernels,
  skills, isolation and traces work as with a simulator worker.
- To XPolicyLab it is the Model: update_obs() delivers the observation a waiting tick
  needs; get_action() hands out the action a tick asked for, one per call (the
  reactive layer sees every observation).
- While the model thinks or a model loads, get_action() returns an empty chunk after
  poll_seconds (under the client's 120 s request limit). XPolicyLab's standard
  episode loop then takes no action and asks again: the simulator does not advance,
  as with a simulator worker.
- After the program finishes (and the arms are home), get_action() holds the arms
  until the client ends the episode; that the episode goes on is not read as an
  outcome.

What the policy sees is the XPolicyLab contract (core/contract.py): the observation the
client sends, the gripper values it reports (the last commands), camera calibration
and kinematics from the embodiment (a kinematics service: RoboDojo's planner without
the simulator); neither the task nor its step limit. The shield checks every action
before it leaves.
"""

import base64
import json
import secrets
import threading
import time
from urllib.request import Request, urlopen

import os
from pathlib import Path
import sys

from ..core.contract import _matmul, pose_matrix
from ..core.contracts import Observation
from ..runtime.shield import Shield, ShieldProfile


ARM_KEYS = {"left_arm": ("left_arm_joint_state", "left_ee_joint_state"),
            "right_arm": ("right_arm_joint_state", "right_ee_joint_state")}
POSE_KEYS = {"left": "left_ee_pose", "right": "right_ee_pose"}


def _data_url(image, fmt=".jpg"):
    import cv2
    import numpy as np

    array = np.asarray(image, dtype=np.uint8)
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if fmt == ".jpg" else []
    ok, encoded = cv2.imencode(fmt, cv2.cvtColor(array, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise RuntimeError("could not encode an image")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(encoded.tobytes()).decode()


def _vector(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


class Bridge:
    """The rendezvous between XPolicyLab's calls and the session's ticks."""

    def __init__(self, embodiment: dict, *, kinematics_url: str | None, poll_seconds: float = 90.0,
                 shield: dict | None = None, obs_timeout: float = 600.0):
        from ..embodiments import profile as embodiment_profile

        self.embodiment = embodiment
        self.profile = {**embodiment_profile(embodiment), "observation_contract": "xpolicylab",
                        "gripper_feedback": "commanded", "calibration_source": "embodiment"}
        self.groups = {name: group for name, group in embodiment["groups"].items()}
        self.size = 1 + max(max(g["joints"] + [g["gripper"]]) for g in self.groups.values())
        self.kinematics_url = kinematics_url
        self.poll_seconds = float(poll_seconds)
        self.obs_timeout = float(obs_timeout)
        options = {**embodiment.get("shield", {}), **(shield or {})}
        shield_profile = {"groups": {n: {"joints": g["joints"], "gripper": g["gripper"]} for n, g in self.groups.items()},
                          "control_hz": float(embodiment["control_hz"]), **options}
        self.shield = Shield(ShieldProfile.from_dict(shield_profile, embodiment["action_low"], embodiment["action_high"]))
        self.lock = threading.Condition()
        self.episode = 0           # increments at every host reset
        self.episode_id = None     # the id the session uses
        self.obs = None            # latest observation from the host
        self.last_obs = None       # the last episode's last observation (its programs may still ask)
        self.obs_seq = 0
        self.step = 0
        self.pending = None        # action the session asked for, not yet handed out
        self.handed = 0            # obs_seq when the last action was handed out
        self.commanded = None      # last command sent (for holding after the program ends)
        self.idle = True           # no tick will come for this episode (program over)
        self.token = None
        self.closed = False
        self._episode_of_id = None

    # -- the XPolicyLab side -------------------------------------------------------

    def host_reset(self):
        """The client started a new episode: the session's episode ends, the observation and
        any pending action are dropped, and the shield forgets its last command."""
        with self.lock:
            self.episode += 1
            self.last_obs = self.obs if self.obs is not None else self.last_obs
            self.obs, self.pending, self.step = None, None, 0
            self.idle = False
            self.commanded = None
            self.shield.reset()
            self.lock.notify_all()

    def host_update_obs(self, obs):
        """The client's latest observation (an XPolicyLab obs dict); it wakes a tick, or the
        session's reset, waiting for it."""
        with self.lock:
            self.obs = obs
            self.obs_seq += 1
            self.lock.notify_all()

    def host_status(self):
        """``action`` when a tick is waiting to be taken, ``idle`` when the program is
        over, ``wait`` while it is still computing. Does not take the action."""
        with self.lock:
            if self.pending is not None:
                return "action"
            if self.idle or self.closed:
                return "idle"
            return "wait"

    def host_get_action(self):
        """One action, or [] when none is ready within poll_seconds."""
        deadline = time.monotonic() + self.poll_seconds
        with self.lock:
            while self.pending is None and not self.idle and not self.closed:
                left = deadline - time.monotonic()
                if left <= 0:
                    return []
                self.lock.wait(timeout=min(left, 1.0))
            if self.pending is not None:
                action, self.pending = self.pending, None
                self.handed = self.obs_seq
                self.step += 1
                self.lock.notify_all()
                return [action]
            hold = self._hold_action()
            return [hold] if hold is not None else []

    def _hold_action(self):
        if self.obs is None:
            return None
        command = list(self.commanded) if self.commanded is not None else self._joints(self.obs)
        state_joints = self._joints(self.obs)
        for group in self.groups.values():
            for index in group["joints"]:
                command[index] = state_joints[index]
        return self._action(command)

    # -- conversions ---------------------------------------------------------------

    def _latest(self):
        obs = self.obs if self.obs is not None else self.last_obs
        if obs is None:
            raise RuntimeError("No observation: the episode has not started")
        return obs

    def _joints(self, obs):
        state = obs["state"]
        joints = [0.0] * self.size
        for name, group in self.groups.items():
            arm_key, gripper_key = ARM_KEYS[name]
            for index, value in zip(group["joints"], _vector(state[arm_key])):
                joints[index] = value
            joints[group["gripper"]] = _vector(state[gripper_key])[0]
        return joints

    def _action(self, command):
        import numpy as np

        action = {}
        for name, group in self.groups.items():
            arm_key, gripper_key = ARM_KEYS[name]
            action[arm_key] = np.asarray([command[i] for i in group["joints"]], dtype=np.float32)
            action[gripper_key] = np.asarray([command[group["gripper"]]], dtype=np.float32)
        return action

    def _state(self, obs=None, ended=False):
        obs = obs if obs is not None else self.obs if self.obs is not None else self.last_obs
        if obs is None:
            return {"step": self.step, "ended": True}
        state = {"step": self.step, "ended": bool(ended), "joint_position": self._joints(obs)}
        for side, key in POSE_KEYS.items():
            state[key] = _vector(obs["state"][key])[:7]
        return state

    def _calibration(self, camera, state):
        calibration = self.embodiment["calibration"][camera]
        if "camera_to_world" in calibration:
            return calibration["intrinsics"], calibration["camera_to_world"]
        side = calibration["link"].split("_")[0]
        return calibration["intrinsics"], _matmul(pose_matrix(state[POSE_KEYS[side]]), calibration["camera_on_link6"])

    def _kinematics(self, route, body):
        if not self.kinematics_url:
            raise RuntimeError("No kinematics service: fk, ik and plan are not available")
        request = Request(self.kinematics_url.rstrip("/") + route, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    def _fk_positions(self, command):
        return {name: pose[:3] for name, pose in self._kinematics("/fk", {"command": list(command)}).items()}

    # -- the session side: a robot worker ------------------------------------------

    name = "xpolicylab-bridge"
    methods = ("profile", "state", "tick", "observe", "policy_obs", "fk", "ik", "plan", "claim_outcome", "outcome")

    def tools(self):
        """No tool schemas: the session uses the Robot API methods through `call`."""
        return []

    def reset(self, task, seed):
        """The episode the host started: waits for its first observation."""
        deadline = time.monotonic() + self.obs_timeout
        with self.lock:
            while self.obs is None and not self.closed:
                if time.monotonic() > deadline:
                    raise RuntimeError("the XPolicyLab client sent no observation")
                self.lock.wait(timeout=1.0)
            self.episode_id = f"xpl-{self.episode}-{secrets.token_hex(4)}"
            self._episode_of_id = self.episode
            self.token = None
            obs = self.obs
            images = {camera: _data_url(item["color"]) for camera, item in (obs.get("vision") or {}).items()
                      if "color" in item}
            return Observation(self.episode_id, 0, str(obs.get("instruction") or ""), self._state(obs), images)

    def call(self, name, arguments):
        """Run Robot API method `name` (one of `methods`) with `arguments` for the session."""
        if name not in self.methods:
            raise ValueError(f"Unknown runtime method: {name}")
        return getattr(self, f"_call_{name}")(**arguments)

    def _call_profile(self):
        return dict(self.profile)

    def _call_state(self, episode_id):
        with self.lock:
            ended = episode_id != self.episode_id or self._episode_of_id != self.episode
            return self._state(ended=ended)

    def _call_tick(self, episode_id, step, command, cameras=None):
        with self.lock:
            if episode_id != self.episode_id or self._episode_of_id != self.episode:
                return {"executed": False, "shield": {"verdict": "rejected", "reasons": [{"check": "episode_over"}],
                                                      "command": None}, "state": self._state(ended=True)}
            if step != self.step:
                raise ValueError(f"Stale tick: step {step}, the episode is at {self.step}")
            current = self._joints(self.obs)
        decision = self.shield.check([float(v) for v in command], current,
                                     fk=self._fk_positions if self.kinematics_url else None)
        if decision.command is None:
            with self.lock:
                return {"executed": False, "shield": decision.to_dict(), "state": self._state()}
        with self.lock:
            self.commanded = list(decision.command)
            self.pending = self._action(decision.command)
            self.lock.notify_all()
            deadline = time.monotonic() + self.obs_timeout
            # The host takes the action (get_action), executes it and sends the next
            # observation; or the episode ends (the host resets, or the server closes).
            while self._episode_of_id == self.episode and not self.closed:
                if self.pending is None and self.obs_seq > self.handed:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("the XPolicyLab client stopped sending observations")
                self.lock.wait(timeout=1.0)
            ended = self._episode_of_id != self.episode or self.closed
            state = self._state(ended=ended)
        result = {"executed": True, "shield": decision.to_dict(), "state": state}
        if cameras and not ended:
            result["observation"] = self._call_observe(episode_id, cameras=cameras)
        return result

    def _call_observe(self, episode_id, cameras=None, image_format="jpeg", quality=90, raw_state=False):
        with self.lock:
            obs = self.obs if self.obs is not None else self.last_obs
            if obs is None:
                raise RuntimeError("No observation: the episode has not started")
            ended = episode_id != self.episode_id or self._episode_of_id != self.episode
            state = self._state(obs, ended=ended)
        vision = obs.get("vision") or {}
        cameras = list(cameras or self.embodiment["cameras"])
        images, intrinsics, extrinsics = {}, {}, {}
        for camera in cameras:
            if camera not in vision:
                raise RuntimeError(f"The observation has no camera {camera}")
            images[camera] = _data_url(vision[camera]["color"], ".jpg" if image_format == "jpeg" else ".png")
            intrinsics[camera], extrinsics[camera] = self._calibration(camera, state)
        size = list(vision[cameras[0]]["color"].shape[1::-1]) if cameras else None
        return {"state": state, "images": images, "intrinsics": intrinsics, "camera_to_world": extrinsics,
                "camera_convention": "opencv", "image_size": size}

    def _call_policy_obs(self, episode_id):
        with self.lock:
            obs = self.obs if self.obs is not None else self.last_obs
            if obs is None:
                raise RuntimeError("No observation: the episode has not started")
            state = self._state(obs)
        vision = {camera: _data_url(item["color"], ".png") for camera, item in (obs.get("vision") or {}).items()
                  if "color" in item}
        return {"vision": vision, "state": {key: _vector(value) for key, value in (obs.get("state") or {}).items()},
                "instruction": obs.get("instruction"), "additional_info": obs.get("additional_info") or {},
                "data_format_version": obs.get("data_format_version"), "robot_state": state}

    def _call_fk(self, command=None):
        if command is None:
            with self.lock:
                command = self._joints(self._latest())
        return self._kinematics("/fk", {"command": list(command)})

    def _call_ik(self, episode_id, arm, pose):
        with self.lock:
            current = self._joints(self._latest())
        return self._kinematics("/ik", {"arm": arm, "pose": list(pose), "current": current})

    def _call_plan(self, episode_id, arm, pose=None, joints=None, raw=False):
        with self.lock:
            current = self._joints(self._latest())
        return self._kinematics("/plan", {"arm": arm, "current": current, "pose": pose, "joints": joints, "raw": raw})

    def _call_claim_outcome(self, episode_id):
        self.token = secrets.token_hex(16)
        return {"token": self.token}

    def _call_outcome(self, token):
        # The host evaluates; a policy never learns the outcome.
        return {"success": None, "reward": None, "process_score": None, "terminated": None, "truncated": None,
                "steps": self.step, "host_scored": True}

    def program_over(self, episode=None):
        """The driver of host episode `episode` (default: the current one) finished: hold
        the arms until the host ends the episode. A driver that outlived its episode
        changes nothing."""
        with self.lock:
            if episode is None or episode == self.episode:
                self.idle = True
                self.lock.notify_all()

    def close(self):
        """Close the bridge; calls waiting for an observation or an action return."""
        with self.lock:
            self.closed = True
            self.lock.notify_all()

