# Copyright (C) 2026 Xiaomi Corporation.
import argparse
import copy
import json
import logging
import os
import sys
import pickle
import random
import dataclasses
import hashlib

from datetime import datetime
from multiprocessing import Manager, Pool
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import hydra
import tyro
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as IF

from PIL import Image
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401  # reserved


from calvin_agent.evaluation.utils import (  # noqa: E402,F401
    count_success,
    get_env_state_for_initial_condition,
)
from calvin_env.envs.play_table_env import get_env as make_calvin_env  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
from deploy.client import Client


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Dataset & split settings
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 10086
    rank: int = 0  # Local Rank.
    world_size: int = 8  # Number of ranks.

    split: str = "abcd"  # Dataset split to evaluate on: 'abc' or 'abcd'.
    dataset_path: str = "/path/to/Calvin/task_{split}_D"  # Path to the dataset root directory.

    #################################################################################################################
    # Action / control settings
    #################################################################################################################
    num_sequences: int = 1000
    replan_steps: int = 10  # Number of future steps predicted per planning call.

    #################################################################################################################
    # Vision / crop settings
    #################################################################################################################
    crop_ratio: float = 0.95  # Crop ratio used for image preprocessing in the model.

    #################################################################################################################
    # Parallel / debug settings
    #################################################################################################################
    debug: bool = False  # Print debug info and visualize environment.
    num_workers: int = 8  # Number of multiprocessing workers per rank.

    #################################################################################################################
    # IO / logging settings
    #################################################################################################################
    CACHE_ROOT: str = "path/to/save"  # Root directory for evaluation logs and cache.
    save_file: str = "results_calvin.json"  # Filename for aggregated evaluation results.

    def __post_init__(self):
        self.dataset_path = self.dataset_path.format(split=self.split.upper())

        self.ACT_CHUNK = self.replan_steps
        self.EP_LEN = 360 // self.ACT_CHUNK


def to_pil(frames):
    pil_frames = []
    for frame in frames:
        if not isinstance(frame, Image.Image):
            pil_img = Image.fromarray(frame)
        else:
            pil_img = frame
        pil_frames.append(pil_img)
    return pil_frames


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


def random_crop(images, crop_ratio=0.98, center_crop=True):
    def _crop(image, crop_ratio=0.98, center_crop=False):
        h, w = image.size[1], image.size[0]
        crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)

        if center_crop:
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
        else:
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)

        cropped = IF.crop(image, top=top, left=left, height=crop_h, width=crop_w)
        resized = IF.resize(cropped, size=(h, w))
        return resized

    return [_crop(image, crop_ratio=crop_ratio, center_crop=center_crop) for image in images]


def make_env(dataset_path: str):
    """
    Create a CALVIN validation environment.

    Args:
        dataset_path: Root path of the CALVIN dataset (task_XXX_D).

    Returns:
        A Gym-like CALVIN environment instance.
    """
    val_folder = Path(dataset_path) / "validation"
    return make_calvin_env(val_folder, show_gui=False)


def get_eval_log_dir(args) -> str:
    """
    Build eval_log_dir from args.CACHE_ROOT
    Format:
        <args.CACHE_ROOT>

    Args:
        args: Parsed arguments (must contain CACHE_ROOT attribute).

    Returns:
        Full path to evaluation log directory with datetime suffix.
    """
    log_dir = Path(args.CACHE_ROOT)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_vis_dir = log_dir / "visualize"
    log_vis_dir.mkdir(parents=True, exist_ok=True)

    return str(log_dir)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with a specific name.

    Args:
        name: The name of the logger.

    Returns:
        logger: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def evaluate_policy(
    rank: int,
    world_size: int,
    model: Client,
    eval_sr_path: str,
    eval_log_dir: str | None = None,
    debug: bool = False,
) -> list[int]:
    """
    Evaluate a model on CALVIN multi-step sequences with multiprocessing.

    Each sequence (initial_state + instruction list) is assigned to a worker
    process. Inside each worker, a fresh CALVIN env is created. A shared
    process-safe list and Lock are used to aggregate results and serialize
    model client calls.

    Args:
        rank: Distributed rank of this process.
        world_size: Total number of ranks.
        model: Remote policy client or callable model.
        eval_sr_path: Path to the success-rate log file (append mode).
        eval_log_dir: Directory to store GIFs and meta logs.
        debug: If True, prints verbose debug info and saves more visualizations.

    Returns:
        List of per-sequence success counts for this rank.
    """
    # Configs
    task_cfg = OmegaConf.load(Path("eval_calvin/config/new_playtable_tasks.yaml"))
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(Path("eval_calvin/config/new_playtable_validation.yaml"))

    original_random_state = random.getstate()
    random.seed(42)

    # Load sequences and shard by rank
    with open("eval_calvin/config/eval_sequences.json", "r") as f:
        eval_sequences = json.load(f)
        random.shuffle(eval_sequences)
    eval_sequences = eval_sequences[: args.num_sequences]
    eval_sequences = eval_sequences[rank::world_size]
    random.setstate(original_random_state)

    # Local Python list for returning results to caller
    results: list[int] = []

    # Shared manager objects across worker processes
    manager = Manager()
    lock = manager.Lock()
    lock_list = manager.Lock()
    shared_results = manager.list()

    # Build jobs for this rank
    jobs = []
    for local_idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        jobs.append(
            (
                model,
                task_oracle,
                initial_state,
                eval_sequence,
                val_annotations,
                debug,
                eval_log_dir,
                lock,
                lock_list,
                shared_results,
                eval_sr_path,
                rank,
            )
        )

    logger.info("Rank %d: starting evaluation of %d sequences.", rank, len(jobs))

    if not args.debug:
        # Run jobs with multiprocessing Pool
        with Pool(processes=args.num_workers) as pool:
            futures = [pool.apply_async(evaluate_sequence, job_args) for job_args in jobs]

            # We still gather return values here, but logging and SR updates
            # are done inside `evaluate_sequence` when each sequence finishes.
            for future in futures:
                result = future.get()
                results.append(result)
    else:
        for job_args in jobs:
            result = evaluate_sequence(*job_args)
            results.append(result)

    logger.info("Rank %d: finished evaluation of %d sequences (shared_results_len=%d).", rank, len(results), len(shared_results))
    return results


def evaluate_sequence(
    model: Client,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    debug: bool,
    eval_log_dir: str,
    lock,
    lock_list,
    shared_results,
    eval_sr_path: str,
    rank: int,
) -> int:
    """
    Evaluate a single multi-step instruction sequence.

    A sequence consists of an initial environment state and an ordered list
    of language subtasks (instructions). This function creates a new CALVIN
    env instance, restores the initial state, and rolls out one subtask at
    a time. When the sequence is finished, it appends the result to the
    shared list and logs the current success statistics.

    Args:
        model: Remote policy client or callable model.
        task_checker: Task oracle used to determine task success.
        initial_state: Serialized initial state for CALVIN env reset.
        eval_sequence: List of language instructions (subtasks).
        val_annotations: Mapping from subtask to language strings.
        debug: Whether to print debug logs and save GIFs.
        eval_log_dir: Directory for GIF outputs.
        lock: Multiprocessing lock used to serialize model calls & logging.
        shared_results: Manager list used to accumulate success counts.
        eval_sr_path: Path to text file where success rates are appended.
        rank: Rank id used only for logging.

    Returns:
        Number of successfully completed subtasks in this sequence.
    """
    # Each worker process creates its own env instance
    env = make_env(args.dataset_path)

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")

    # Evaluate subtasks in order; stop at first failure
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(
            env=env,
            model=model,
            task_oracle=task_checker,
            subtask=subtask,
            val_annotations=val_annotations,
            debug=debug,
            eval_log_dir=eval_log_dir,
            subtask_i=subtask_i,
            rank=rank,
            lock=lock,
        )
        if success:
            success_counter += 1
        else:
            break

    # Update shared_results and log progress (protected by lock)
    with lock_list:
        shared_results.append(success_counter)
        # Copy to a normal list for count_success
        current_results = list(shared_results)

    success_list = count_success(current_results)
    local_i = len(current_results) - 1

    # Log current success rate via logger instead of tqdm
    sr_str = " ".join([f"{i + 1}/{len(eval_sequence)}: {v * 100:.1f}% |" for i, v in enumerate(success_list)])
    logger.info(
        "Rank %d finished local seq %d with success=%d | %s with avg_len=%.3f",
        rank,
        local_i,
        success_counter,
        sr_str,
        np.mean(current_results),
    )

    return success_counter


def rollout(
    env,
    model: Client,
    task_oracle,
    subtask: str,
    val_annotations,
    debug: bool,
    eval_log_dir: str,
    subtask_i: int,
    rank: int,
    lock,
) -> bool:
    """
    Roll out a single language subtask inside the CALVIN environment.

    This function runs a short horizon control loop conditioned on one
    language instruction. It queries the model for an action plan, converts
    the action representation into low-level robot commands, and steps the
    environment until success or time-out.

    Args:
        env: CALVIN environment instance.
        model: Remote policy client or callable model.
        task_oracle: Oracle that checks whether a subtask is solved.
        subtask: Current language instruction.
        val_annotations: Mapping from subtask to annotation strings.
        debug: Whether to print debug logs and save GIFs.
        eval_log_dir: Directory to store GIFs.
        subtask_i: Index of the current subtask in the sequence.
        rank: Rank id used only for logging.
        lock: Multiprocessing lock guarding model calls.
    Returns:
        True if the subtask was completed successfully, False otherwise.
    """
    if debug:
        print(f"{subtask} ", end="")
        img_list = []

    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0].capitalize()

    start_info = env.get_info()
    for _ in range(args.EP_LEN):
        # process state
        start_state = obs["robot_obs"][:7]
        start_state = np.concatenate([start_state, np.zeros([25])], axis=-1)

        # process images
        rgb_static, rgb_gripper = to_pil([obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"]])
        rgb_static, rgb_gripper = random_crop([rgb_static, rgb_gripper], args.crop_ratio)

        #
        model_inputs = {
            "task_id": f"calvin_{args.split}_orig",
            "state": start_state,
            "base": rgb_static,
            "wrist_left": rgb_gripper,
            "language": lang_annotation,
        }
        temp_seed = hash_data_to_seed(model_inputs)
        model_inputs["seed"] = temp_seed
        model_inputs["language"] = model_inputs["language"] + "."
        with lock:
            action = model(**model_inputs)

        # (B, T, D+1) -> use first B=0, and truncate to ACT_CHUNK, discard last dim (prob)
        action = action[0, 0 : args.ACT_CHUNK, :7].cpu().numpy()
        # gripper sign -> {-1, 1}
        action[:, -1:] = np.where(action[:, -1:] > 0, 1, -1)

        # Step through each low-level action
        for single_action in action:
            obs, _, _, current_info = env.step(single_action)
            if debug:
                img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
                img_list.append(img_copy)
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if debug:
                    print(colored("success", "green"), end=" ")
                    clip = ImageSequenceClip(img_list, fps=30)
                    gif_path = Path(eval_log_dir) / "visualize" / f"{rank}-{subtask_i}-{subtask}-succ.gif"
                    clip.write_gif(str(gif_path), fps=30)
                return True

    if debug:
        print(colored("fail", "red"), end=" ")
        clip = ImageSequenceClip(img_list, fps=30)
        gif_path = Path(eval_log_dir) / "visualize" / f"{rank}-{subtask_i}-{subtask}-fail.gif"
        clip.write_gif(str(gif_path), fps=30)
    return False


# ===============================
# Main entry
# ===============================
def main(args: Args):
    """Main entry point for CALVIN evaluation."""
    eval_log_dir = get_eval_log_dir(args)
    os.makedirs(eval_log_dir, exist_ok=True)

    logger.info(
        "Starting evaluation with rank=%d, world_size=%d, num_workers=%d",
        args.rank,
        args.world_size,
        args.num_workers,
    )

    model_client = Client(host="localhost", port=10086 + args.rank)

    sr_path = Path(eval_log_dir) / "success_rate_calvin.txt"
    sr_path_str = str(sr_path)
    # Run evaluation for this rank
    results = evaluate_policy(
        rank=args.rank,
        world_size=args.world_size,
        model=model_client,
        eval_sr_path=sr_path_str,
        eval_log_dir=eval_log_dir,
        debug=args.debug,
    )

    # For merging across ranks later we also store the subset of sequences for this rank
    with open("eval_calvin/config/eval_sequences.json", "r") as f:
        eval_sequences = json.load(f)
    eval_sequences = eval_sequences[: args.num_sequences]
    eval_sequences = eval_sequences[args.rank :: args.world_size]

    # save `results` and `eval_sequences` as pickle for later merge.
    pickle_path = Path(eval_log_dir) / f"rank_{args.rank}_results.pkl"
    with open(str(pickle_path), "wb") as f:
        pickle.dump({"results": results, "sequences": eval_sequences}, f)

    logger.info("Rank %d: evaluation finished, got %d result entries.", args.rank, len(results))


if __name__ == "__main__":
    logger = get_logger(__name__)

    args = tyro.cli(Args)

    main(args)
