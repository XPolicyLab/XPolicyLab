# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
(Parallelized Version)
"""
import concurrent.futures
import json
import logging
import os
import sys
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import wandb

from .libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from BeingH.benchmark.utils.policy import Policy_Libero as Policy 
from BeingH.benchmark.utils.policy import Obses_to_Policy_Obs_Dict
from BeingH.benchmark.utils.utils import DATE_TIME, set_seed_everywhere

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:

    # Configuration for controlling the number of parallel processes
    max_workers: Optional[int] = 8                # Number of parallel workers, None means using all available cores

    # fmt: off
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "bagel"                    # Model family
    port: int = 5555
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    action_type: str = ""   # "eef_delta", "eef_relative", "world_delta", "world_relative", "world_abs", "camera_abs"
    data_config_name: str = ""

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    num_save_videos_per_task: int = 50               # Number of videos to save per task
    log_interval: int = 10                           # Interval (in episodes) to log intermediate results
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 512                           # Resolution for environment images (not policy input resolution)

    video_dir: str = "results/rollouts"

    # task_ids_to_eval: Optional[List[int]] = field(default_factory=list)
    task_ids_to_eval: Optional[str] = None
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./results/experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 42                                    # Random Seed (for reproducibility)

    # fmt: on

def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""

    policy = Policy(port=cfg.port, exec_chunk_size=cfg.num_open_loop_steps, action_type=cfg.action_type)
    print("cfg.num_open_loop_steps", cfg.num_open_loop_steps)

    return policy

def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    run_id = f"EVAL-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        # log_message("Using default initial states", log_file)
        return initial_states, None

# run_episode function remains mostly unchanged, just no longer needs log_file parameter since logging is handled by main process
def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    policy,
    initial_state=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # NOTE!!!
    policy.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    libero_obses_to_policy_obs_dict = Obses_to_Policy_Obs_Dict[cfg.data_config_name]

    success = False
    # try:
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        obs_dict = libero_obses_to_policy_obs_dict(obs, task_description)
        # print(obs_dict['video.top_view'].shape)
        # print(asd)
        replay_images.append(obs_dict['video.top_view'].reshape(*obs_dict['video.top_view'].shape[1:]))

        action = policy.get_action(obs_dict)
        action[-1] = -2 * action[-1] + 1

        obs, reward, done, info = env.step(action)
        if done:
            success = True
            break
        t += 1

    return success, replay_images

# ##################################################################
#  Parallel Execution Worker Function
# ##################################################################
def run_trial_worker(args):
    """
    An independent worker function that runs a complete trial (episode) in a single process.
    It initializes all necessary components (environment, policy, etc.), executes the trial, and returns results.
    """
    cfg, task_id, trial_idx, seed = args

    # 1. Initialize within the worker
    # Each worker must have its own independent random seed
    set_seed_everywhere(seed)

    # Each worker creates its own policy client instance
    policy = initialize_model(cfg)

    # Each worker creates its own environment instance
    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task, resolution=cfg.env_img_res)
    env.seed(seed) # Ensure environment also uses the correct seed

    # 2. Get initial state (this logic is migrated from the original run_task function)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id)
    initial_state = None
    should_skip = False

    if cfg.initial_states_path == "DEFAULT":
        initial_state = initial_states[trial_idx]
    else:
        initial_states_task_key = task_description.replace(" ", "_")
        episode_key = f"demo_{trial_idx}"
        if not all_initial_states[initial_states_task_key][episode_key]["success"]:
            should_skip = True # Skip this trial if expert demonstration failed
        else:
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

    if should_skip:
        return {
            "task_id": task_id,
            "task_description": task_description,
            "success": False,
            "skipped": True, # Add a flag to indicate skipped
        }

    # 3. Run single episode
    success, replay_images = run_episode(
        cfg,
        env,
        task_description,
        policy,
        initial_state,
    )

    if trial_idx < cfg.num_save_videos_per_task:
        # Call function to save video
        save_rollout_video(replay_images,
            trial_idx,
            task_id,
            success=success,
            task_description=task_description,
            video_dir=cfg.video_dir)

    env.close()

    # 5. Return serializable (pickleable) result
    return {
        "task_id": task_id,
        "task_description": task_description,
        "success": success,
        "skipped": False,
    }


# ##################################################################
#  Refactored: Main Evaluation Function
# ##################################################################
@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Note: Policy model service needs to be started and listening on the specified port before this function runs
    # policy = initialize_model(cfg) # No longer initialize policy client in main process

    log_file, local_log_filepath, run_id = setup_logging(cfg)
    log_message(f"Starting evaluation run: {run_id}", log_file)

    # benchmark_dict = benchmark.get_benchmark_dict()
    # task_suite = benchmark_dict[cfg.task_suite_name]()

    if cfg.task_ids_to_eval:
        task_ids_to_run = [int(x) for x in cfg.task_ids_to_eval.split()]
        log_message(f"Evaluating on a specific subset of tasks: {task_ids_to_run}", log_file)
        # Still need to validate task ID validity, so we create a temporary object to get total task count
        num_tasks_total = benchmark.get_benchmark_dict()[cfg.task_suite_name]().n_tasks
        for task_id in task_ids_to_run:
            if not (0 <= task_id < num_tasks_total):
                raise ValueError(f"Invalid task ID {task_id}. Must be between 0 and {num_tasks_total - 1} for {cfg.task_suite_name}.")
    else:
        # Get total task count in a stateless way: create a temporary object, get attribute, then let it be garbage collected.
        # This ensures no LIBERO rendering objects are alive in the main process before forking the process pool.
        num_tasks_total = benchmark.get_benchmark_dict()[cfg.task_suite_name]().n_tasks
        task_ids_to_run = list(range(num_tasks_total))
        log_message(f"Evaluating on all {num_tasks_total} tasks in {cfg.task_suite_name}", log_file)

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # 1. Create task list for all trials
    jobs = []
    job_counter = 0
    for task_id in task_ids_to_run:
        for trial_idx in range(cfg.num_trials_per_task):
            # Assign a unique, deterministic seed to each job
            job_seed = cfg.seed + job_counter
            jobs.append((cfg, task_id, trial_idx, job_seed))
            job_counter += 1

    log_message(f"Created {len(jobs)} trials to run in parallel.", log_file)

    # 2. Use ProcessPoolExecutor to execute all tasks in parallel
    task_results = defaultdict(lambda: {"successes": 0, "episodes": 0})
    total_successes = 0
    total_episodes = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
        # Use executor.map to distribute tasks, and tqdm to show progress
        results_iterator = executor.map(run_trial_worker, jobs)

        for result in tqdm.tqdm(results_iterator, total=len(jobs), desc="Running Trials"):
            if result is None: # Worker may fail due to unknown error
                continue

            if result["skipped"]:
                log_message(f"Skipped trial for task {result['task_id']} due to failed expert demo.", log_file)
                continue

            # 3. Collect and aggregate results
            total_episodes += 1
            task_id = result["task_id"]

            task_results[task_id]["description"] = result["task_description"]
            task_results[task_id]["episodes"] += 1
            if result["success"]:
                total_successes += 1
                task_results[task_id]["successes"] += 1

            if total_episodes % cfg.log_interval == 0:
                current_success_rate = (total_successes / total_episodes) if total_episodes > 0 else 0

                # Print formatted intermediate results
                log_message(f"  So far: {total_successes} successes out of {total_episodes} valid episodes.", log_file)
                log_message(f"  Current Overall Success Rate: {current_success_rate:.4f} ({current_success_rate * 100:.1f}%)", log_file)

    # 4. Calculate final success rate and log
    final_success_rate = (total_successes / total_episodes) if total_episodes > 0 else 0

    log_message("\n" + "=" * 50, log_file)
    log_message("--- DETAILED RESULTS PER TASK ---", log_file)
    log_message("=" * 50, log_file)

    # Output sorted by task_id
    for task_id in sorted(task_results.keys()):
        res = task_results[task_id]
        success_rate = (res['successes'] / res['episodes']) if res['episodes'] > 0 else 0
        log_message(
            f"  - Task ID {task_id:02d} ({res['description']}): "
            f"Success Rate = {success_rate:.4f} "
            f"({res['successes']}/{res['episodes']})",
            log_file,
        )
        if cfg.use_wandb:
            wandb.log({
                f"success_rate/{res['description']}": success_rate,
                f"num_episodes/{res['description']}": res['episodes'],
            })

    log_message("\n--- FINAL SUMMARY ---", log_file)
    log_message(f"Total episodes evaluated: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if cfg.use_wandb:
        wandb.log({
            "success_rate/total": final_success_rate,
            "num_episodes/total": total_episodes,
        })
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    # In multiprocessing environment, main execution logic needs to be under if __name__ == "__main__":
    import multiprocessing
    # Force use of 'spawn' start method before all other code
    # 'force=True' ensures it can be set successfully even if a context already exists
    multiprocessing.set_start_method("spawn", force=True)

    eval_libero()