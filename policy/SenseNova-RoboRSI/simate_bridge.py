"""Simate history and joint actions behind the controller's unchanged wire API."""
from __future__ import annotations

import inspect
import re
import threading

import numpy as np

CHANNELS = (
    "left_arm_joint_state", "left_ee_joint_state",
    "right_arm_joint_state", "right_ee_joint_state",
)
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def session_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("A unique episode UUID is required")
    return value


def step_index(value):
    if type(value) is not int or value < 0:
        raise ValueError("control_step must be a nonnegative integer")
    return value


class SimateBridge:
    """Share immutable weights; isolate history by explicitly owned episode UUID."""

    def __init__(self, policy):
        if policy.exec_chunk_size != 50:
            raise ValueError("The adaptive controller requires all 50 Simate predictions")
        self.policy = policy
        self.steps = {}
        self.lock = threading.RLock()

    def request(self, payload):
        with self.lock:
            scope = session_id(payload.get("evaluation_id"))
            operation = payload.get("op", "infer")
            if operation == "release":
                self.policy.reset_evaluation({"evaluation_id": scope})
                self.steps.pop(scope, None)
                return {"released": True}
            step = step_index(payload.get("control_step"))
            if operation == "observe":
                previous = self.steps.get(scope)
                if previous == step:
                    return {"observed_step": step, "duplicate": True}
                if step != (0 if previous is None else previous + 1):
                    raise ValueError("Simate history requires every consecutive native action step")
                image = np.asarray(payload["image"])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError("History requires a decoded H,W,3 uint8 RGB head image")
                self.policy.update_obs({
                    "evaluation_id": scope, "env_idx": 0,
                    "vision": {"cam_head": {"color": image}},
                })
                self.steps[scope] = step
                return {"observed_step": step}
            if operation != "infer":
                raise ValueError("Unsupported Simate operation")
            if self.steps.get(scope) != step:
                raise ValueError("Inference must use the latest acknowledged native observation")
            state = np.asarray(payload["state"], dtype=np.float32)
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError("Simate requires the 14 native joints and grippers")
            images = {}
            for camera, wire in zip(CAMERAS, ("cam_high", *CAMERAS[1:]), strict=True):
                image = np.asarray(payload["images"][wire])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3:
                    raise ValueError("VLA input requires C,H,W uint8 RGB images")
                images[camera] = {"color": image.transpose(1, 2, 0)}
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("A controller subgoal is required")
            # The observation was already appended by observe. Changing its prompt
            # and current views must never append history a second time.
            self.policy.observations[scope, 0] = {
                "evaluation_id": scope, "env_idx": 0, "instruction": prompt,
                "vision": images,
                "state": dict(zip(CHANNELS, (state[:6], state[6:7], state[7:13], state[13:]), strict=True)),
            }
            chunk = self.policy.get_action({"evaluation_id": scope})
            actions = np.asarray([
                np.concatenate([np.asarray(row[name]).reshape(-1) for name in CHANNELS])
                for row in chunk
            ], dtype=np.float32)
            if actions.shape != (50, 14) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid Simate trajectory: {actions.shape}")
            return {"actions": actions}


class SimateObserver:
    """One lightweight observation websocket per episode, with synchronous ACKs."""

    def __init__(self, port, evaluation_id):
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy

        self.codec = msgpack_numpy
        self.evaluation_id = session_id(evaluation_id)
        # websockets 15 enables proxy discovery; 14 has no proxy argument and
        # does not discover proxies. Always keep this loopback socket direct.
        options = {"proxy": None} if "proxy" in inspect.signature(connect).parameters else {}
        self.connection = connect(
            f"ws://127.0.0.1:{port}", compression=None, max_size=None,
            open_timeout=30, **options,
        )
        metadata = self.codec.unpackb(self.connection.recv(timeout=30))
        if metadata.get("policy") != "Simate_beta":
            self.connection.close()
            raise ValueError("Shared VLA service is not Simate_beta")

    def _request(self, payload):
        self.connection.send(self.codec.Packer().pack({**payload, "evaluation_id": self.evaluation_id}))
        response = self.connection.recv(timeout=300)
        if isinstance(response, str):
            raise RuntimeError(response)
        return self.codec.unpackb(response)

    def observe(self, obs, step):
        return self._request({"op": "observe", "control_step": step, "image": obs["vision"]["cam_head"]["color"]})

    def close(self):
        try:
            self._request({"op": "release"})
        finally:
            self.connection.close()


class SimateConnection:
    """Add session identity without changing the pinned proposal implementation."""

    def __init__(self, connection, evaluation_id, control_step):
        self.connection = connection
        self.evaluation_id = session_id(evaluation_id)
        self.control_step = control_step

    def send(self, message):
        from openpi_client import msgpack_numpy

        payload = msgpack_numpy.unpackb(message)
        payload.update(evaluation_id=self.evaluation_id, control_step=self.control_step(), op="infer")
        return self.connection.send(msgpack_numpy.Packer().pack(payload))

    def __getattr__(self, name):
        return getattr(self.connection, name)
