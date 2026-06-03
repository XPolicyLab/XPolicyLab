# Copyright (C) 2026 Xiaomi Corporation.
import os
import sys
import collections
import dataclasses
import json
import logging
import math
import pickle
import socket
import struct
import hashlib
from multiprocessing import Pool, Manager
from functools import partial

import numpy as np
import imageio
from PIL import Image
from pathlib import Path
import tyro

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

sys.path.append(str(Path(__file__).resolve().parents[1]))
from deploy.client import Client

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training数据


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 10086
    resize_size: int = 224
    replan_steps: int = 10

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    debug: bool = False  # Print debug info and visualize environment.
    num_workers: int = 10
    task_id: int = -1  # Single task ID to evaluate (-1 means all tasks)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = None  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def hash_data_to_seed(data, max_bytes=4):
    """
    Computes a stable SHA256 hash for a dictionary containing Numpy arrays
    and PIL Images, ensuring consistency across different times and machines.
    """

    def custom_encoder(obj):
        """
        Serializer for non-JSON serializable objects.
        """
        # Handle Numpy arrays
        if isinstance(obj, np.ndarray):
            return {"__type__": "numpy", "dtype": str(obj.dtype), "shape": obj.shape, "data": obj.tobytes().hex()}

        # Handle PIL Images
        if isinstance(obj, Image.Image):
            # Calculate a digest of the raw image bytes to keep the JSON small
            img_hash = hashlib.md5(obj.tobytes()).hexdigest()
            return {"__type__": "PIL.Image", "mode": obj.mode, "size": obj.size, "content_hash": img_hash}

        # Handle Sets
        if isinstance(obj, set):
            return sorted(list(obj))

        # Let it crash if type is unknown (as requested, no try-except)
        raise TypeError(f"Type {type(obj)} is not JSON serializable")

    # Generate canonical JSON string
    json_str = json.dumps(
        data,
        default=custom_encoder,
        sort_keys=True,  # Enforce deterministic key order
        separators=(",", ":"),  # Remove whitespace for compact representation
        ensure_ascii=False,  # Preserve non-ASCII characters
    )

    hex_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    seed_int = int(hex_hash, 16)
    if max_bytes > 0:
        seed_int = seed_int % (2 ** (8 * max_bytes))

    return seed_int


def run_trial(trial_data, shared_dict, client_lock, counter_lock, args, client):
    task_id, episode_idx, initial_states, task_description, max_steps = trial_data
    try:
        env, _ = _get_libero_env(task_suite.get_task(task_id), LIBERO_ENV_RESOLUTION, args.seed)
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])

        action_plan = collections.deque()
        t = 0
        replay_images = []
        done = False

        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = convert_to_uint8(img)
            wrist_img = convert_to_uint8(wrist_img)
            replay_images.append(img)

            if len(action_plan) <= 0:
                base_obs = Image.fromarray(img)
                left_wrist_obs = Image.fromarray(wrist_img)
                state = np.concatenate(
                    [
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                        np.array([0.0] * 24),
                    ]
                )
                instruction = str(task_description).capitalize()

                model_inputs = {
                    "task_id": f"libero_all",
                    "state": state,
                    "base": base_obs,
                    "wrist_left": left_wrist_obs,
                    "language": instruction,
                }
                temp_seed = hash_data_to_seed(model_inputs)
                model_inputs["seed"] = temp_seed
                model_inputs["language"] = model_inputs["language"] + "."
                with client_lock:
                    action_chunk = client(**model_inputs)[0, :, :-1]
                action_plan.extend(action_chunk[0 : args.replan_steps, 0:7])

            action = action_plan.popleft().tolist()
            obs, _, done, _ = env.step(action)
            if done:
                break
            t += 1

        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_")
        task_video_dir = Path(args.video_out_path) / task_segment
        task_video_dir.mkdir(parents=True, exist_ok=True)
        video_path = task_video_dir / f"rollout_{task_segment}_{suffix}_{episode_idx}.mp4"
        imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

        with counter_lock:
            shared_dict["total_episodes"] += 1
            if done:
                shared_dict["total_successes"] += 1
            current_total = shared_dict["total_episodes"]
            current_successes = shared_dict["total_successes"]
            success_rate = current_successes / current_total if current_total > 0 else 0
            logging.info(
                f"Task_id {task_id}: Episode {current_total}: {'Success' if done else 'Failure'} | "
                f"Current Success Rate: {success_rate:.2%} ({current_successes}/{current_total})"
            )

        return done
    except Exception as e:
        logging.error(f"Error in trial {episode_idx}: {e}")
        return False
    finally:
        if "env" in locals():
            env.close()


def eval_libero(args: Args) -> None:
    global task_suite
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = Client(args.host, args.port)

    manager = Manager()
    shared_dict = manager.dict({"total_episodes": 0, "total_successes": 0})
    client_lock = manager.Lock()
    counter_lock = manager.Lock()

    total_episodes, total_successes = 0, 0

    # Determine which tasks to run
    if args.task_id >= 0:
        task_ids = [args.task_id]
        if args.task_id >= num_tasks_in_suite:
            raise ValueError(f"Task ID {args.task_id} is out of range for suite {args.task_suite_name}")
    else:
        task_ids = range(num_tasks_in_suite)

    all_task_results = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language

        trials = [(task_id, i, initial_states, task_description, max_steps) for i in range(args.num_trials_per_task)]

        if not args.debug:
            with Pool(processes=args.num_workers) as pool:
                func = partial(run_trial, shared_dict=shared_dict, client_lock=client_lock, counter_lock=counter_lock, args=args, client=client)
                results = list(pool.imap(func, trials))
        else:
            results = []
            for trial in trials:
                result = run_trial(trial, shared_dict, client_lock, counter_lock, args, client)
                results.append(result)

        successes = sum(results)
        total_episodes += len(results)
        total_successes += successes

        success_rate = successes / len(results) if len(results) > 0 else 0

        task_result = {"task_id": task_id, "instruction": task_description, "success_rate": success_rate}
        all_task_results.append(task_result)

        task_segment = task_description.replace(" ", "_")

        logging.info(f"Task {task_id} Success Rate: {success_rate:.2%}")

        json_result_file = Path(args.video_out_path) / f"result_{task_id}.json"
        with open(json_result_file, "w") as f:
            json.dump(all_task_results, f, indent=2)

    if args.task_id < 0:
        logging.info(f"Overall Success Rate: {total_successes / total_episodes:.2%}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
