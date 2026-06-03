# Copyright (C) 2026 Xiaomi Corporation.
import sys
from pathlib import Path
import json
import collections
import dataclasses
import logging
import pathlib
import time
import hashlib
from typing import Literal
import pandas as pd

import torch
import torchvision.transforms.functional as F
import numpy as np
import imageio
from PIL import Image
import tyro
from transforms3d.euler import euler2axangle, mat2euler, quat2mat
from transforms3d.quaternions import mat2quat

sys.path.append(str(Path(__file__).resolve().parents[1]))
from deploy.client import Client
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.env.env_builder import build_maniskill2_env


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 10086
    replan_steps: int = 4

    #################################################################################################################
    # SimplerEnv environment-specific parameters
    #################################################################################################################
    dataset_name: Literal["fractal", "bridge"] = "fractal"  # Dataset name. Choose from fractal and bridge
    worker_id: int = 0  # Worker ID for distributed evaluation
    num_workers: int = 1  # Total number of workers

    repeat_times: int = 1  # Repeat times for each episode
    record_video: bool = True  # Record video

    #################################################################################################################
    # Utils
    #################################################################################################################
    experiment_root: str = "./logs/"  # Path to save experiment results

    seed: int = 42  # Random Seed (for reproducibility)


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


def preprocess_proprio_fractal(proprio: np.array) -> np.array:
    """convert wxyz quat from simpler to xyzw used in fractal"""
    gripper_quat_wxyz = proprio[3:7]
    gripper_rotm = quat2mat(gripper_quat_wxyz)
    gripper_xyzw = mat2quat(gripper_rotm)[[1, 2, 3, 0]]
    gripper_width = proprio[7]  # from simpler, 0 for close, 1 for open
    gripper_closedness = 1 - gripper_width

    raw_proprio = np.concatenate(
        (
            proprio[:3],
            gripper_xyzw,
            [gripper_closedness],
        )
    )
    return raw_proprio


def postprocess_gripper_fractal(action: float) -> float:
    # trained with [0, 1], 0 for close, 1 for open
    # convert to -1 open, 1 close for simpler
    action = (action * 2) - 1  # [0, 1] -> [-1, 1] -1 close, 1 open

    # without sticky
    relative_gripper_action = -action
    relative_gripper_action = np.clip(relative_gripper_action, -1, 1)
    return relative_gripper_action


def preprocess_proprio_bridge(proprio: np.array) -> np.array:
    # convert ee rotation to the frame of top-down
    default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    rm_bridge = quat2mat(proprio[3:7])
    rpy_bridge_converted = mat2euler(rm_bridge @ default_rot.T)
    gripper_openness = proprio[7]
    raw_proprio = np.concatenate(
        [
            proprio[:3],
            rpy_bridge_converted,
            [gripper_openness],
        ]
    )
    return raw_proprio


def postprocess_gripper_bridge(action: float) -> float:
    # trained with [0, 1], 0 for close, 1 for open
    # convert to -1 close, 1 open for simpler
    action_gripper = 2.0 * (action > 0.5) - 1.0
    return action_gripper


def client_process(task_id, state, base_obs, instruction):
    if "bridge" in task_id:
        base_obs = base_obs.resize((256, 256))
        instruction = f"<|im_start|>user\nThe following observations are captured from multiple views.\n# Base View\n<|vision_start|><|image_pad|><|vision_end|>\nGenerate robot actions for the task:\n{instruction} /no_cot<|im_end|>\n<|im_start|>assistant\n<cot></cot><|im_end|>\n"
        zero_state = np.zeros_like(state)[..., -1:]
        state = np.concatenate([state[..., :-1], zero_state, state[..., -1:]], axis=-1)
    else:
        base_obs = base_obs.resize((320, 256))
        instruction = f"<|im_start|>user\nThe following observations are captured from multiple views.\n# Ego View\n<|vision_start|><|image_pad|><|vision_end|>\nGenerate robot actions for the task:\n{instruction} /no_cot<|im_end|>\n<|im_start|>assistant\n<cot></cot><|im_end|>\n"

    # center crop
    crop_ratio = 0.95
    h, w = base_obs.size[1], base_obs.size[0]
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    crop_y = (h - crop_h) // 2
    crop_x = (w - crop_w) // 2
    base_obs = F.crop(base_obs, crop_y, crop_x, crop_h, crop_w)
    assert (base_obs.size[0], base_obs.size[1]) == (
        crop_w,
        crop_h,
    ), f"{(base_obs.size[0], base_obs.size[1])} != ({crop_w}, {crop_h})"
    base_obs = F.resize(base_obs, (h, w))
    assert (base_obs.size[0], base_obs.size[1]) == (w, h), f"{(base_obs.size[0], base_obs.size[1])} != ({w}, {h})"

    state = torch.from_numpy(state)[None, None]
    state = torch.nn.functional.pad(state, (0, 32 - state.shape[-1]))
    model_inputs = {
        "task_id": task_id,
        "state": state.numpy(),
        "language": instruction,
        "base": base_obs,
    }

    return model_inputs


def eval_simplerenv(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    print(f"Running dataset: {args.dataset_name} with {args.num_workers} workers")
    print(f"Worker ID: {args.worker_id}")

    total_indices = args.num_workers
    global_index = args.worker_id

    video_out_path = None
    if args.experiment_root:
        exp_root = pathlib.Path(args.experiment_root)
        exp_root.mkdir(parents=True, exist_ok=True)
        log_dir = exp_root / "results"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{global_index}.csv"
        end_dir = exp_root / "end"
        end_dir.mkdir(parents=True, exist_ok=True)
        end_path = end_dir / f"{global_index}.end"
        if args.record_video:
            video_out_path = exp_root / f"videos"
            video_out_path.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        f.write(f"exp_name,dataset_name,task_suite_name,global_index,total_indices,success_episodes,total_episodes\n")

    with open(f"eval_simplerenv/env_options/{args.dataset_name}.json", "r") as f:
        env_configs = json.load(f)

    options_list = []
    for i, config in enumerate(env_configs):
        options_list.extend([config] * args.repeat_times)

    client = Client(args.host, args.port)

    # Start evaluation
    total_episodes_dict = collections.defaultdict(int)
    total_successes_dict = collections.defaultdict(int)
    start_time = time.time()
    for i, config in enumerate(options_list):
        if i % total_indices != global_index:
            continue

        env_kwargs = config["env_kwargs"]
        env_reset_options = config["env_reset_options"]
        env_kwargs["max_episode_steps"] = int(env_kwargs["max_episode_steps"])
        success = run_single_episode(i, config["task_suite_name"], args, env_kwargs, env_reset_options, client, video_out_path)
        total_successes_dict[config["task_suite_name"]] += success
        total_episodes_dict[config["task_suite_name"]] += 1

        time_elapsed = time.time() - start_time
        time_remaining = (time_elapsed / (i + 1)) * (len(options_list) - i - 1)
        print(
            f"[Worker {args.worker_id} / {args.num_workers}] {config['task_suite_name']} - {i}/{len(options_list)}, success: {success}, time elapsed: {time_elapsed:.2f}s, time remaining: {time_remaining:.2f}s"
        )
        with open(log_path, "a") as f:
            f.write(f"debug,{args.dataset_name},{config['task_suite_name']},{global_index},{total_indices},{int(success)},1\n")

    with open(end_path, "w") as f:
        f.write(f"")

    while True:
        end_files = end_dir.glob("*.end")
        if len(list(end_files)) >= total_indices:
            break
        print(f"[Worker {args.worker_id} / {args.num_workers}] Waiting for all end files... ({len(list(end_files))}/{total_indices}) files present. Sleeping for 30 seconds.")
        time.sleep(30)

    if global_index == 0:
        csv_files = list(log_dir.glob("*.csv"))
        dataframes = [pd.read_csv(f) for f in csv_files]
        data = pd.concat(dataframes, ignore_index=True)

        grouped = data.groupby(["exp_name", "dataset_name", "task_suite_name"], as_index=False)[["success_episodes", "total_episodes"]].sum()
        grouped["success_rate"] = grouped["success_episodes"] / grouped["total_episodes"]
        grouped.to_csv(exp_root / "all_results.csv", index=False)


def run_single_episode(config_id, task_suite_name, args: Args, env_kwargs, env_reset_options, client, video_out_path=None):
    # Create environment
    env = build_maniskill2_env(**env_kwargs)
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    instruction = env.get_language_instruction()

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs)
    done, truncated = False, False

    if video_out_path:
        (video_out_path / task_suite_name).mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            video_out_path / task_suite_name / f"{config_id:04d}.mp4",
            fps=30,
        )
        writer.append_data(image)

    # Reset environment
    action_plan = collections.deque()

    while not truncated:
        if len(action_plan) <= 0:
            # Prepare observations dict
            base_obs = Image.fromarray(image, mode="RGB")
            if "bridge" in args.dataset_name:
                state = preprocess_proprio_bridge(obs["agent"]["eef_pos"])
                task_id = "bridge_delta"
            else:
                state = preprocess_proprio_fractal(obs["agent"]["eef_pos"])
                task_id = "fractal_delta"

            instruction = instruction[0].upper() + instruction[1:] + "."
            model_inputs = client_process(task_id, state, base_obs, instruction)
            temp_seed = hash_data_to_seed(model_inputs)
            model_inputs["seed"] = temp_seed
            action_chunk = client(**model_inputs)[0]
            assert (
                args.replan_steps <= action_chunk.shape[1]
            ), f"Replan steps must be less than or equal to the number of steps in the action chunk. {args.replan_steps} > {action_chunk.shape[1]}"

            action_chunk = action_chunk[: args.replan_steps, :7].cpu().numpy()
            action_plan.extend(action_chunk)

        raw_action = action_plan.popleft()

        roll, pitch, yaw = raw_action[3:6]
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle

        if "bridge" in args.dataset_name:
            action_gripper = postprocess_gripper_bridge(raw_action[-1])
        else:
            action_gripper = postprocess_gripper_fractal(raw_action[-1])

        action = np.concatenate(
            [
                raw_action[:3],
                action_rotation_axangle,
                [action_gripper],
            ]
        )

        # Execute action in environment
        obs, reward, done, truncated, info = env.step(action)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        instruction = env.get_language_instruction()
        if video_out_path:
            writer.append_data(image)

        if done:
            break

    if video_out_path:
        writer.close()

    return done


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_simplerenv)
