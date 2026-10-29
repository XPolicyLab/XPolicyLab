import json
import time

import numpy as np
from PIL import Image


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    # Upload images at model input size (bilinear, as the policy resizes): raw 3x640x480 RGB uploads
    # dominated rollout time over the SSH tunnel.
    camera_sizes = model_client.call(func_name="get_camera_sizes")
    started = time.perf_counter()
    timing = {"observation_s": 0.0, "update_rpc_s": 0.0, "action_rpc_s": 0.0, "simulation_s": 0.0}

    while not TASK_ENV.is_episode_end():
        t0 = time.perf_counter()
        obs = TASK_ENV.get_obs()
        for name, size_wh in camera_sizes.items():
            camera = obs["vision"][name]
            # Encoded colors pass through unchanged for the policy server to decode.
            if isinstance(camera["color"], np.ndarray) and camera["color"].ndim == 3:
                camera["color"] = np.asarray(Image.fromarray(camera["color"]).resize(tuple(size_wh), Image.BILINEAR))
                camera["shape"] = camera["color"].shape
        t1 = time.perf_counter()
        model_client.call(func_name="update_obs", obs=obs)
        t2 = time.perf_counter()
        actions = model_client.call(func_name="get_action")
        t3 = time.perf_counter()
        timing["observation_s"] += t1 - t0
        timing["update_rpc_s"] += t2 - t1
        timing["action_rpc_s"] += t3 - t2

        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break
        timing["simulation_s"] += time.perf_counter() - t3

    timing["rollout_s"] = time.perf_counter() - started
    timing["policy"] = model_client.call(func_name="get_timing")
    print("[RoboDojo timing] " + json.dumps(timing), flush=True)
