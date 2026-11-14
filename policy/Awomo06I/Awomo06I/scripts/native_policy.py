# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoboTwin native qpos execution using the five-domain HTTP inference service."""
import io
import os
import urllib.request
from collections import deque

import numpy as np


class RobotTwinPolicy:
    """Execute the established eight-step prefix through the native controller."""

    def __init__(self) -> None:
        self.pending = deque()
        self.count = 0
        self.url = "http://" + os.environ["ROBOTWIN_MODEL_HOST"] + ":" + str(18700 + int(os.environ["RW_WORKER"]))

    def should_request_observation(self) -> bool:
        return not self.pending

    def step(self, task_env, observation) -> None:
        if not self.pending:
            assert observation is not None
            cams = observation["observation"]
            buffer = io.BytesIO()
            np.savez(buffer, state=np.asarray(observation["joint_action"]["vector"], np.float32),
                     head=cams["head_camera"]["rgb"], left=cams["left_camera"]["rgb"],
                     right=cams["right_camera"]["rgb"], instruction=np.asarray(task_env.get_instruction()))
            request = urllib.request.Request(self.url, data=buffer.getvalue(), method="POST")
            with urllib.request.urlopen(request, timeout=900) as response:
                output = np.load(io.BytesIO(response.read()), allow_pickle=False)
            assert output.shape == (16, 14) and np.isfinite(output).all()
            self.pending.extend(output[:8])
            self.count += 1
            print("ROBOTWIN_FIVE_FINITE_FORWARD", self.count, flush=True)
        task_env.take_action(self.pending.popleft(), action_type="qpos")

    def reset(self) -> None:
        self.pending.clear()
        self.count = 0


def get_model(usr_args) -> RobotTwinPolicy:
    return RobotTwinPolicy()


def eval(task_env, model, observation) -> None:
    model.step(task_env, observation)


def reset_model(model) -> None:
    model.reset()
