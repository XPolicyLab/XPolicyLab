import logging
import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Union, List
from datetime import timedelta

import numpy as np

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import torch
import torch.distributed as dist
from transformers.utils.versions import require_version

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from galaxea_fm.processors.base_processor import BaseProcessor
from galaxea_fm.models.base_policy import BasePolicy
from galaxea_fm.utils.logging_config import (
    log_allocated_gpu_memory,
    log_amp_config,
    setup_logging,
)
from galaxea_fm.utils.pytorch_utils import dict_apply, set_global_seed
from galaxea_fm.utils.train_utils import init_experiment_tracker
from galaxea_fm.utils.load_pretrained_resumed import load_checkpoint_for_eval
from galaxea_fm.utils.config_resolvers import register_default_resolvers
from galaxea_fm.utils.tqdm import tqdm

from collections import deque
from galaxea_fm.utils.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    binarize_gripper_open,
    invert_gripper_action,
    save_rollout_video,
    LIBERO_ENV_RESOLUTION
)

register_default_resolvers()
logger = get_logger(__name__)
require_version("datasets==3.6.0", "To fix: uv pip install datasets==3.6.0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
num_gpus_visible = torch.cuda.device_count()
if num_gpus_visible > 1:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)

from libero.libero import benchmark, get_libero_path

from concurrent.futures import ThreadPoolExecutor, as_completed


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def get_libero_sample(obs, task_description, processor, device):
    imgs = get_libero_image(obs)
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )

    sample = {
        # Convert torch.Size([1, 256, 256, 3]) to torch.Size([1, 3, 256, 256])
        "images": {
            "image": torch.from_numpy(np.expand_dims(
                imgs["image"], axis=0
            )).permute(0, 3, 1, 2),  # (H, W, C), dtype=uint8, range(0-255)
            "wrist_image": torch.from_numpy(np.expand_dims(
                imgs["wrist_image"], axis=0
            )).permute(0, 3, 1, 2),  # (H, W, C)
        },
        "state": {
            "default": torch.from_numpy(np.expand_dims(state, axis=0)).to(torch.float32),
        },
        "task": str(task_description),
        "state_is_pad": torch.tensor([False]),
        "image_is_pad": torch.tensor([False]),
        "action_is_pad": torch.tensor([False] * 32),
        "idx": torch.tensor(0),
    }

    if processor is not None:
        sample = processor.preprocess(sample)

    batch = dict_apply(sample, lambda x: x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else x)

    return batch, imgs


def get_libero_batch(env_idx, obs_all_env, task_description, processor, device):
    sample, imgs = get_libero_sample(obs_all_env[env_idx], task_description, processor, device)
    return env_idx, sample


def predict_action(batch, processor, policy, binarize_gripper=False):
    """
    Predict action from sample using processor and policy.

    Args:
        batch: input batch for prediction
        processor: data pre/post-processor
        policy: policy model for action prediction
    """

    with torch.no_grad():
        batch = policy.predict_action(batch)

    batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
    batch = processor.postprocess(batch)
    cur_pd_action = dict_apply(batch["action"], lambda x: x.cpu().numpy())

    action = cur_pd_action['default']
    assert len(action.shape) == 3, f"action.shape {action.shape} should be (B, chunk_size, 7)"

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)

    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])

    return action


def run_single_task_parallels(
    task,
    initial_state,
    policy: BasePolicy,
    processor: BaseProcessor,
    cfg: DictConfig,
    task_suite_name: str,
    task_idx: int,
    video_dir: Path,
    verbose: bool = True,
) -> dict:
    """Run a single evaluation episode.

    Args:
        task: LIBERO task instance
        initial_state: initial states for all trials
        policy: policy model
        processor: data processor
        cfg: configuration object
        task_suite_name: name of the task suite
        task_idx: index of the current task within the suite
        video_dir: directory to save rollout videos
        verbose: whether to show progress bar

    Returns:
        dict: evaluation results including success count and episode indices
    """
    env_num = cfg.libero_eval.env_num if hasattr(cfg.libero_eval, "env_num") else 1
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"), env_num=env_num)
    results = {
        'successes': 0,
        'failure_episodes': [],
        'success_episodes': [],
        'task_description': task_description,
    }

    if task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {task_suite_name}")

    replan_steps = cfg.libero_eval.get('replan_steps', 5)

    # Reset environment
    env.reset()

    # Setup
    t = 0
    replay_images = [[] for _ in range(env_num)]
    full_actions = []

    if verbose:
        pbar = tqdm.tqdm(total=max_steps + cfg.libero_eval.num_steps_wait, desc=f"Episode")
    else:
        pbar = None

    done_result = np.array([False] * env_num)

    while t < max_steps + cfg.libero_eval.num_steps_wait:
        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
        # and we need to wait for them to fall
        dummy_actions = np.array([get_libero_dummy_action() for _ in range(env_num)])
        if t < cfg.libero_eval.num_steps_wait:
            obs_all_env, reward, done, info = env.step(dummy_actions)
            if pbar is not None:
                pbar.update(1)
            t += 1
            continue

        # IMPORTANT: rotate 180 degrees to match train preprocessing
        sample_list = [None] * env_num

        # Use ThreadPoolExecutor to parallelize preprocessing across environments
        with ThreadPoolExecutor(max_workers=min(8, env_num)) as executor:
            future_to_env = {
                executor.submit(get_libero_batch, env_idx, obs_all_env, task_description, processor, policy.device): env_idx
                for env_idx in range(env_num)
            }

            for future in as_completed(future_to_env):
                env_idx, sample = future.result()
                sample_list[env_idx] = sample

        batch = {}
        keys = sample_list[0].keys()

        for key in keys:
            values = [sample[key] for sample in sample_list]
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.cat(values, dim=0)
            else:
                batch[key] = values

        batch = dict_apply(batch, lambda x: x.to(policy.device) if isinstance(x, torch.Tensor) else x)

        actions = predict_action(
            batch, processor, policy,
            binarize_gripper=cfg.libero_eval.get('binarize_gripper', False)
        )
        full_actions.append(actions.copy())

        # Execute replan_steps actions sequentially
        for i in range(replan_steps):
            obs_all_env, reward, done, info = env.step(actions[:, i, :])
            for env_idx in range(env_num):
                img = np.ascontiguousarray(obs_all_env[env_idx]["agentview_image"][::-1, ::-1])
                replay_images[env_idx].append(img)
            if pbar is not None:
                pbar.update(1)
            t += 1

        done_result = done | done_result
        if np.sum(done_result) == env_num:
            break

    results['successes'] = np.sum(done_result)
    results['failure_episodes'] = np.where(done_result == False)[0].tolist()
    results['success_episodes'] = np.where(done_result == True)[0].tolist()

    # Save rollout videos
    for env_idx in range(env_num):
        save_rollout_video(
            video_dir / task_suite_name / "videos",
            replay_images[env_idx],
            f"task{task_idx}_trial{env_idx}",
            success=done_result[env_idx],
            task_description=task_description
        )

    if pbar is not None:
        pbar.close()
    return results


def create_task_queue(task_suite_names, benchmark_dict, num_trials):
    """Create a task queue ordered by task ID, without splitting into individual trials."""
    task_queue = []

    for task_suite_name in task_suite_names:
        task_suite = benchmark_dict[task_suite_name]()

        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)

            # Ensure enough initial states by cycling if necessary
            if len(initial_states) < num_trials:
                extended_states = []
                for i in range(num_trials):
                    extended_states.append(initial_states[i % len(initial_states)])
                initial_states = extended_states

            task_queue.append({
                'task_suite_name': task_suite_name,
                'task_id': task_id,
                'task': task,
                'initial_states': initial_states  # All trials' initial states for this task
            })

    return task_queue


def run_evaluation_on_gpu(gpu_id, all_tasks, policy, processor, cfg, video_dir, accelerator):
    """Run evaluation tasks assigned to a specific GPU."""
    local_results = []

    total_gpus = accelerator.num_processes
    gpu_tasks = all_tasks[gpu_id::total_gpus]
    logger.info(f"GPU {gpu_id} will process {len(gpu_tasks)} tasks", main_process_only=False)

    for idx, task_info in enumerate(gpu_tasks):
        task_suite_name = task_info['task_suite_name']
        task_id = task_info['task_id']
        task = task_info['task']
        initial_states = task_info['initial_states']

        logger.info(f"GPU {gpu_id}: Processing task {task_suite_name}/task{task_id} ({idx+1}/{len(gpu_tasks)})", main_process_only=False)

        results = run_single_task_parallels(
            task=task,
            initial_state=initial_states,
            policy=policy,
            processor=processor,
            cfg=cfg,
            task_suite_name=task_suite_name,
            task_idx=task_id,
            video_dir=video_dir,
            verbose=accelerator.is_main_process,
        )

        success_rate = results['successes'] / (len(results.get('success_episodes', [])) + len(results.get('failure_episodes', [])))
        logger.info(f"GPU {gpu_id}: Task {task_suite_name}/task{task_id} success rate: {success_rate:.2f}", main_process_only=False)

        results['gpu_id'] = gpu_id
        results['task_suite_name'] = task_suite_name
        results['task_id'] = task_id
        local_results.append(results)

    return local_results


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def eval(cfg: DictConfig) -> Optional[float]:
    start_time = time.time()
    OmegaConf.resolve(cfg)
    output_dir = Path(cfg.output_dir)

    assert cfg.ckpt_path is not None, "cfg.ckpt_path must not be None!"
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    project_config = ProjectConfiguration(project_dir=str(output_dir))
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.model.enable_bf16_training else "no",
        project_config=project_config,
        kwargs_handlers=[init_process_group_kwargs],
        log_with=cfg.logger.type,
    )
    torch.cuda.set_device(device_id := accelerator.local_process_index)
    torch.cuda.empty_cache()
    device_id = accelerator.local_process_index

    setup_logging(log_level=logging.INFO, is_main_process=True)
    logger.info(f"Output directory: {output_dir}")
    log_amp_config(logger, accelerator)
    tracker_type = init_experiment_tracker(cfg, accelerator, output_dir)

    # Load model (supports both legacy .pt and new directory formats)
    model: BasePolicy = instantiate(cfg.model.model_arch)
    model, dataset_stats = load_checkpoint_for_eval(cfg.ckpt_path, model, device="cpu")
    if cfg.model.use_torch_compile:
        model = torch.compile(model, mode="default")
    policy = model.to(device_id).eval()
    log_allocated_gpu_memory(logger, stage="loading model", device=0)
    if hasattr(policy, 'action_tokenizer'):
        policy.action_tokenizer.to(device_id)

    if cfg.get("seed"):
        set_global_seed(cfg.seed, get_worker_init_fn=False)

    processor: BaseProcessor = instantiate(cfg.data.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    # Set tokenizer for Pi0FastPolicy (autoregressive models need tokenizer for action decoding)
    if hasattr(policy, 'set_tokenizer') and hasattr(processor, 'tokenizer'):
        policy.set_tokenizer(processor.tokenizer)

    # Create video output directories
    if accelerator.is_main_process:
        video_output_dir = Path(cfg.libero_eval.output_dir)
        os.makedirs(video_output_dir, exist_ok=True)
        for task_suite_name in cfg.libero_eval.task_suite_names:
            os.makedirs(video_output_dir / task_suite_name / "videos", exist_ok=True)
    else:
        video_output_dir = None
    container = [video_output_dir]
    dist.broadcast_object_list(container, src=0)
    video_output_dir = container[0]

    # Initialize task queue
    benchmark_dict = benchmark.get_benchmark_dict()
    all_tasks = create_task_queue(cfg.libero_eval.task_suite_names, benchmark_dict, cfg.libero_eval.num_trials)
    logger.info(f"Created task queue with {len(all_tasks)} total tasks across all suites")

    # Run evaluation on assigned GPU
    gpu_id = accelerator.local_process_index
    local_results = run_evaluation_on_gpu(gpu_id, all_tasks, policy, processor, cfg, video_output_dir, accelerator)

    # Gather results from all GPUs
    all_results = accelerator.gather_for_metrics(local_results)

    total_successes = 0
    total_trials = 0

    if accelerator.is_main_process:
        for result in all_results:
            if 'successes' in result:
                total_successes += result['successes']
                total_trials += len(result.get('success_episodes', [])) + len(result.get('failure_episodes', []))

        output_dir = os.path.join(cfg.libero_eval.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"all_gpu_results.json")

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=4, cls=NumpyEncoder)

        # Calculate and generate markdown report
        # Group results by task_suite_name and task_id
        task_suite_results = {}
        task_id_results = {}
        
        for result in all_results:
            task_suite_name = result['task_suite_name']
            task_id = result['task_id']
            
            if task_suite_name not in task_suite_results:
                task_suite_results[task_suite_name] = {'successes': 0, 'trials': 0}
            
            if (task_suite_name, task_id) not in task_id_results:
                task_id_results[(task_suite_name, task_id)] = {'successes': 0, 'trials': 0}
            
            task_suite_results[task_suite_name]['successes'] += result['successes']
            task_suite_results[task_suite_name]['trials'] += len(result.get('success_episodes', [])) + len(result.get('failure_episodes', []))
            
            task_id_results[(task_suite_name, task_id)]['successes'] += result['successes']
            task_id_results[(task_suite_name, task_id)]['trials'] += len(result.get('success_episodes', [])) + len(result.get('failure_episodes', []))
        
        # Generate markdown report
        markdown_content = f"# Evaluation Results Report\n\n"
        markdown_content += f"## Summary\n\n"
        markdown_content += f"- Overall Success Rate: {total_successes}/{total_trials} = {total_successes/total_trials:.2f} ({(total_successes/total_trials)*100:.2f}%)\n\n"
        
        # Task ID level results
        markdown_content += f"## Success Rates by Task ID\n\n"
        markdown_content += f"| Task Suite | Task ID | Successes | Trials | Success Rate |\n"
        markdown_content += f"|----------|---------|----------|--------|--------------|\n"
        
        # Sort by task suite and task id for consistent output
        sorted_task_ids = sorted(task_id_results.keys(), key=lambda x: (x[0], x[1]))
        for (task_suite_name, task_id) in sorted_task_ids:
            task_result = task_id_results[(task_suite_name, task_id)]
            successes = task_result['successes']
            trials = task_result['trials']
            success_rate = successes / trials if trials > 0 else 0
            markdown_content += f"| {task_suite_name} | {task_id} | {successes} | {trials} | {success_rate:.2f} ({success_rate*100:.2f}%) |\n"
        
        markdown_content += f"\n"
        
        # Task Suite level results
        markdown_content += f"## Success Rates by Task Suite\n\n"
        markdown_content += f"| Task Suite | Successes | Trials | Success Rate |\n"
        markdown_content += f"|----------|----------|--------|--------------|\n"
        
        for task_suite_name in sorted(task_suite_results.keys()):
            suite_result = task_suite_results[task_suite_name]
            successes = suite_result['successes']
            trials = suite_result['trials']
            success_rate = successes / trials if trials > 0 else 0
            markdown_content += f"| {task_suite_name} | {successes} | {trials} | {success_rate:.2f} ({success_rate*100:.2f}%) |\n"
        
        markdown_content += f"\n"
        
        # Overall results
        markdown_content += f"## Overall Results\n\n"
        markdown_content += f"| Metric | Value |\n"
        markdown_content += f"|--------|-------|\n"
        markdown_content += f"| Total Successes | {total_successes} |\n"
        markdown_content += f"| Total Trials | {total_trials} |\n"
        markdown_content += f"| Overall Success Rate | {total_successes/total_trials:.2f} ({(total_successes/total_trials)*100:.2f}%) |\n"
        
        # Write markdown report
        markdown_file = os.path.join(output_dir, f"evaluation_results.md")
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Overall results: {total_successes}/{total_trials} successes : {total_successes/total_trials:.2f}")
        logger.info(f"Markdown report saved to: {markdown_file}")

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Time taken: {duration:.2f} seconds")
        logger.info(f"Output directory: {output_dir}")

    return total_successes / total_trials if total_trials > 0 else 0


if __name__ == "__main__":
    eval()