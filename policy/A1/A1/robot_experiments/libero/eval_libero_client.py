"""LIBERO Evaluation Client - Connects to inference server for actions.

Usage:
    python eval_libero_client.py --url http://localhost:7778 --task_suite libero_spatial

This script evaluates LIBERO tasks by connecting to a remote inference server
instead of loading the model locally.
"""

import os
import json
import logging
import argparse
from collections import deque
import random
import time
from pathlib import Path
from typing import Optional
import base64
from io import BytesIO

import numpy as np
from PIL import Image
import requests
import tqdm

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from robot_experiments.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from robot_experiments.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from robot_experiments.vla_utils import resize_image_for_policy
from a1.vla.constants import NUM_ACTIONS_CHUNK


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_action_from_server(
    url: str,
    task_description: str,
    full_image: np.ndarray,
    wrist_image: np.ndarray,
    state: Optional[np.ndarray] = None,
    action_nums: int = 8,
    use_proprio: bool = True,
) -> list:
    """Query inference server for actions.

    Args:
        url: Inference server URL
        task_description: Task description text
        full_image: Primary camera image
        wrist_image: Wrist camera image
        state: Proprioception state (optional)
        action_nums: Number of actions to request
        use_proprio: Whether to send proprio data

    Returns:
        List of action arrays
    """
    # Encode images to base64
    full_image_b64 = encode_image_to_base64(full_image)
    wrist_image_b64 = encode_image_to_base64(wrist_image)

    # Prepare request payload (matching server API format)
    payload = {
        "instruction": task_description,
        "images": [full_image_b64, wrist_image_b64],
    }

    if use_proprio and state is not None:
        # Pad state to fixed dimension (32) if needed
        if state.ndim == 1:
            state = np.pad(state, (0, 32 - state.shape[0]), "constant", constant_values=0)
        elif state.ndim == 2:
            state = np.pad(
                state, ((0, 0), (0, 32 - state.shape[-1])), "constant", constant_values=0
            )
        else:
            pad_width = [(0, 0)] * (state.ndim - 1) + [(0, 32 - state.shape[-1])]
            state = np.pad(state, pad_width, "constant", constant_values=0)
        # Server expects proprio_data as list of lists
        if state.ndim == 1:
            payload["proprio_data"] = [state.tolist()]
        else:
            payload["proprio_data"] = state.tolist()

    # Send request to server
    try:
        response = requests.post(
            f"{url}/inference",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        # Parse actions from response (server returns 'predicted_actions')
        actions = result.get("predicted_actions", [])
        if not actions:
            logger.warning("Server returned empty actions, using dummy action")
            return [get_libero_dummy_action("a1")]

        # Server returns a list of actions, we may need to slice based on action_nums
        if len(actions) > action_nums:
            actions = actions[:action_nums]

        return [np.array(a) for a in actions]

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get action from server: {e}")
        raise


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img


def process_action(action, model_family="a1"):
    """Process action before sending to environment."""
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "a1":
        action = invert_gripper_action(action)
    return action


def run_episode(
    env,
    task_description: str,
    server_url: str,
    resize_size,
    num_open_loop_steps: int = 8,
    num_steps_wait: int = 10,
    max_steps: int = 220,
    use_proprio: bool = True,
    action_nums: int = 8,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()
    obs = env.get_observation()

    # Initialize action queue
    action_queue = deque(maxlen=num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    success = False

    while t < max_steps + num_steps_wait:
        # Do nothing for the first few timesteps to let objects stabilize
        if t < num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action("a1"))
            t += 1
            continue

        # Prepare observation
        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)

        # If action queue is empty, query server
        if len(action_queue) == 0:
            actions = get_action_from_server(
                url=server_url,
                task_description=task_description,
                full_image=observation["full_image"],
                wrist_image=observation["wrist_image"],
                state=observation["state"],
                action_nums=action_nums,
                use_proprio=use_proprio,
            )
            action_queue.extend(actions)

        # Get action from queue
        action = action_queue.popleft()

        # Process action
        action = process_action(action)

        # Execute action in environment
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    return success, replay_images


def run_task(
    task_suite,
    task_id: int,
    server_url: str,
    resize_size: int,
    num_trials_per_task: int = 30,
    num_open_loop_steps: int = 8,
    num_steps_wait: int = 10,
    use_proprio: bool = True,
    action_nums: int = 8,
    save_rollout_video: bool = True,
    save_path: str = "./eval_logs",
    total_episodes: int = 0,
    total_successes: int = 0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # Initialize environment and get task description
    task_description = task.language
    env = OffScreenRenderEnv(
        task_file=task.task_file,
        camera_heights=256,
        camera_widths=256,
    )

    task_start_time = time.time()
    task_episodes, task_successes = 0, 0

    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    task_suite_name = task_suite.name.lower().replace(" ", "_")
    max_steps = max_steps_map.get(task_suite_name, 220)

    for episode_idx in tqdm.tqdm(range(num_trials_per_task)):
        logger.info(f"\nTask {task_id}: {task_description}")
        episode_start_time = time.time()

        # Set initial state
        initial_state = initial_states[episode_idx]
        env.reset()
        env.set_init_state(initial_state)

        logger.info(f"Starting episode {task_episodes + 1}/{num_trials_per_task}...")

        # Run episode
        success, replay_images = run_episode(
            env=env,
            task_description=task_description,
            server_url=server_url,
            resize_size=resize_size,
            num_open_loop_steps=num_open_loop_steps,
            num_steps_wait=num_steps_wait,
            max_steps=max_steps,
            use_proprio=use_proprio,
            action_nums=action_nums,
            log_file=log_file,
        )

        episode_duration = time.time() - episode_start_time

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        if save_rollout_video:
            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                save_path=save_path,
                log_file=log_file,
            )

        # Log results
        current_total_time = time.time() - task_start_time
        avg_time_per_episode = current_total_time / task_episodes
        remaining_episodes = num_trials_per_task - task_episodes
        estimated_remaining_time = avg_time_per_episode * remaining_episodes
        num_tasks = task_suite.n_tasks
        estimated_total_evaluation_time = avg_time_per_episode * (num_trials_per_task * num_tasks - total_episodes)

        logger.info(f"Success: {success}")
        logger.info(f"# episodes completed so far: {total_episodes}/{num_trials_per_task * num_tasks}")
        logger.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        logger.info(f"Episode duration: {episode_duration:.2f}s")
        logger.info(f"Current task time so far: {current_total_time:.2f}s ({current_total_time/60:.1f}min)")
        logger.info(f"Estimated remaining time for this task: {estimated_remaining_time:.2f}s ({estimated_remaining_time/60:.1f}min)")
        logger.info(f"Estimated remaining time for evaluation: {estimated_total_evaluation_time:.2f}s ({estimated_total_evaluation_time/60:.1f}min)")

    env.close()

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    logger.info(f"Current task success rate: {task_success_rate}")
    logger.info(f"Current total success rate: {total_success_rate}")

    return total_episodes, total_successes


def main():
    parser = argparse.ArgumentParser(description="Evaluate LIBERO tasks using remote inference server")

    # Server configuration
    parser.add_argument("--url", type=str, required=True, help="Inference server URL, e.g., http://localhost:7778")

    # Task configuration
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite name",
    )
    parser.add_argument("--num_trials_per_task", type=int, default=30, help="Number of rollouts per task")
    parser.add_argument("--num_steps_wait", type=int, default=10, help="Steps to wait for objects to stabilize")

    # Inference configuration
    parser.add_argument("--num_open_loop_steps", type=int, default=8, help="Number of actions to execute open-loop")
    parser.add_argument("--action_nums", type=int, default=8, help="Number of actions to request from server")
    parser.add_argument("--use_proprio", action="store_true", default=True, help="Whether to send proprio data")
    parser.add_argument("--no_proprio", action="store_true", help="Disable proprioception")

    # Image configuration
    parser.add_argument("--image_size", type=int, default=336, help="Image size for policy input")

    # Logging configuration
    parser.add_argument("--save_rollout_video", action="store_true", default=True, help="Save rollout videos")
    parser.add_argument("--save_path", type=str, default="./eval_logs", help="Path to save rollout videos")
    parser.add_argument("--seed", type=int, default=666, help="Random seed")

    args = parser.parse_args()

    # Handle --no_proprio flag
    if args.no_proprio:
        args.use_proprio = False

    # Set random seed
    set_seed_everywhere(args.seed)

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)

    # Setup logging to file
    log_filepath = os.path.join(args.save_path, f"EVAL-{args.task_suite_name}-{DATE_TIME}.txt")
    log_file = open(log_filepath, "w")
    logger.info(f"Logging to file: {log_filepath}")

    # Test server connection
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"Successfully connected to server at {args.url}")
        else:
            logger.warning(f"Server health check returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to server at {args.url}: {e}")
        raise

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks

    logger.info(f"Task suite: {args.task_suite_name}")
    logger.info(f"Number of tasks: {num_tasks}")
    logger.info(f"Trials per task: {args.num_trials_per_task}")
    logger.info(f"Server URL: {args.url}")
    logger.info(f"Use proprio: {args.use_proprio}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_ids = list(range(num_tasks))
    random.shuffle(task_ids)

    for task_id in tqdm.tqdm(task_ids):
        total_episodes, total_successes = run_task(
            task_suite=task_suite,
            task_id=task_id,
            server_url=args.url,
            resize_size=args.image_size,
            num_trials_per_task=args.num_trials_per_task,
            num_open_loop_steps=args.num_open_loop_steps,
            num_steps_wait=args.num_steps_wait,
            use_proprio=args.use_proprio,
            action_nums=args.action_nums,
            save_rollout_video=args.save_rollout_video,
            save_path=args.save_path,
            total_episodes=total_episodes,
            total_successes=total_successes,
            log_file=log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    logger.info("=" * 50)
    logger.info("Final results:")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Total successes: {total_successes}")
    logger.info(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)")
    logger.info("=" * 50)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    main()
