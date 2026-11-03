"""Evaluate a LIBERO task manifest while keeping one policy instance loaded.

This entrypoint is intended for single-GPU sweeps where model loading dominates
runtime.  It preserves the single-task evaluator's per-task RNG starting point,
creates a fresh environment for every task, and writes the same result JSON
schema as ``eval_libero_single.py``.  Benchmark objects may optionally be cached
for very large suites whose construction is expensive.
"""

import copy
import json
import logging
import random
import time
from pathlib import Path
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from recoverwam.libero.eval_helpers import (
    build_trial_initial_states,
    hash_libero_trial_seed,
    _load_model_checkpoint,
    _resolve_action_horizon,
    resolve_max_steps,
    resolve_reset_mode,
    resolve_trial_seed_mode,
)
from recoverwam.libero.eval_libero_single import (
    NumpyEncoder,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    run_single_task,
)
from recoverwam.datasets.lerobot.processors.recoverwam_processor import RecoverWAMProcessor
from recoverwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from recoverwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark


def load_task_manifest(path: str | Path) -> list[tuple[str, int]]:
    """Load unique ``suite,task_id`` entries in their declared order."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Task manifest does not exist: {manifest_path}")
    tasks: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0]:
            raise ValueError(
                f"Invalid task manifest line {line_number}: {raw_line!r}; expected suite,task_id."
            )
        try:
            task_id = int(fields[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid task id on manifest line {line_number}: {fields[1]!r}."
            ) from exc
        if task_id < 0:
            raise ValueError(
                f"Task id must be non-negative on manifest line {line_number}: {task_id}."
            )
        task_key = (fields[0], task_id)
        if task_key in seen:
            raise ValueError(f"Duplicate task in manifest: {task_key}.")
        seen.add(task_key)
        tasks.append(task_key)
    if not tasks:
        raise ValueError(f"Task manifest is empty: {manifest_path}")
    return tasks


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _completed_result_matches(
    output_file: Path,
    *,
    task_suite_name: str,
    task_id: int,
    num_trials: int,
    reset_mode: str,
    trial_seed_mode: str,
    trial_seed_id: int | None,
    max_steps_per_episode: int,
    init_state_indices: list[int],
    trial_seeds: list[int | None],
) -> bool:
    if not output_file.is_file():
        return False
    try:
        result = json.loads(output_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    good, bad = result.get("success_episodes", []), result.get("failure_episodes", [])
    if not isinstance(good, list) or not isinstance(bad, list):
        return False
    if len(good + bad) != num_trials or set(good + bad) != set(range(num_trials)):
        return False
    if result.get("successes") != len(good):
        return False
    return (
        result.get("task_suite") == task_suite_name
        and int(result.get("task_id", -1)) == task_id
        and (int(result.get("total_episodes", -1)) == num_trials)
        and (result.get("reset_mode") == reset_mode)
        and (result.get("trial_seed_mode", "sequential") == trial_seed_mode)
        and (result.get("trial_seed_id") == trial_seed_id)
        and (int(result.get("max_steps_per_episode", -1)) == max_steps_per_episode)
        and (result.get("init_state_indices") == init_state_indices)
        and (result.get("trial_seeds") == trial_seeds)
        and (
            len(result.get("success_episodes", [])) + len(result.get("failure_episodes", []))
            == num_trials
        )
    )


def eval_task_manifest(cfg: DictConfig) -> list[dict]:
    partial_state = PartialState()
    partial_state.config = cfg
    manifest_path = cfg.EVALUATION.get("task_manifest")
    if manifest_path is None:
        raise ValueError("Pass +EVALUATION.task_manifest=/path/to/tasks.txt.")
    tasks = load_task_manifest(str(manifest_path))
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)
    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError("The manifest evaluator requires EVALUATION.env_num=1.")
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(
        model,
        str(cfg.ckpt),
        video_checkpoint_override=cfg.EVALUATION.get("video_checkpoint_override"),
    )
    model = model.to(model_device).eval()
    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: RecoverWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)
    action_horizon = _resolve_action_horizon(cfg)
    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h, input_w = (int(video_size[0]), int(video_size[1]))
    output_root = Path(cfg.EVALUATION.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    num_trials = int(cfg.EVALUATION.num_trials)
    init_state_start = int(cfg.EVALUATION.get("init_state_start", 0))
    init_state_indices = list(range(init_state_start, init_state_start + num_trials))
    reset_mode = resolve_reset_mode(cfg)
    trial_seed_mode = resolve_trial_seed_mode(cfg)
    trial_seed_id_value = cfg.EVALUATION.get("trial_seed_id")
    trial_seed_id = None if trial_seed_id_value is None else int(trial_seed_id_value)
    trial_seed_start = int(cfg.EVALUATION.get("trial_seed_start", 1))
    gpu_id = int(cfg.gpu_id)
    benchmark_dict = benchmark.get_benchmark_dict()
    unknown_suites = sorted({suite for suite, _ in tasks if suite not in benchmark_dict})
    if unknown_suites:
        raise ValueError(f"Unknown LIBERO benchmark suites: {unknown_suites}")
    clean_task_rng_state = _capture_rng_state()
    all_results: list[dict] = []
    reuse_task_suite = bool(cfg.EVALUATION.get("reuse_task_suite", False))
    task_suite_cache: dict[str, object] = {}
    for task_suite_name, task_id in tasks:
        if trial_seed_mode == "benchmark_seed_sequence":
            expected_trial_seeds: list[int | None] = list(
                range(trial_seed_start, trial_seed_start + num_trials)
            )
        elif trial_seed_mode == "task_episode_hash":
            expected_trial_seeds = [
                hash_libero_trial_seed(task_suite_name, task_id, episode_index, int(trial_seed_id))
                for episode_index in range(num_trials)
            ]
        else:
            expected_trial_seeds = [None] * num_trials
        output_dir = output_root / task_suite_name
        output_file = output_dir / f"gpu{gpu_id}_task{task_id}_results.json"
        max_steps_per_episode = resolve_max_steps(task_suite_name)
        if _completed_result_matches(
            output_file,
            task_suite_name=task_suite_name,
            task_id=task_id,
            num_trials=num_trials,
            reset_mode=reset_mode,
            trial_seed_mode=trial_seed_mode,
            trial_seed_id=trial_seed_id,
            max_steps_per_episode=max_steps_per_episode,
            init_state_indices=init_state_indices,
            trial_seeds=expected_trial_seeds,
        ):
            logging.info("Skipping completed task %s,%s", task_suite_name, task_id)
            continue
        _restore_rng_state(clean_task_rng_state)
        with open_dict(cfg.EVALUATION):
            cfg.EVALUATION.task_suite_name = task_suite_name
            cfg.EVALUATION.task_id = task_id
        task_start = time.time()
        if reuse_task_suite:
            task_suite = task_suite_cache.get(task_suite_name)
            if task_suite is None:
                task_suite = benchmark_dict[task_suite_name]()
                task_suite_cache[task_suite_name] = task_suite
        else:
            task_suite = benchmark_dict[task_suite_name]()
        if task_id >= int(task_suite.n_tasks):
            raise ValueError(
                f"Task id {task_id} is out of range for {task_suite_name} (n_tasks={task_suite.n_tasks})."
            )
        task = task_suite.get_task(task_id)
        initial_states = build_trial_initial_states(
            task_suite, task_id, num_trials, reset_mode, init_state_start=init_state_start
        )
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        predicted_video_dir = output_dir / "predicted_videos"
        if bool(cfg.EVALUATION.get("visualize_future_video", False)) or bool(
            cfg.EVALUATION.get("save_one_step_video", False)
        ):
            predicted_video_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "task_suite": task_suite_name,
            "task_id": task_id,
            "task_description": None,
            "successes": 0,
            "total_episodes": num_trials,
            "gpu_id": gpu_id,
            "success_episodes": [],
            "failure_episodes": [],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": 0,
            "reset_mode": reset_mode,
            "policy_seed": None if cfg.get("seed") is None else int(cfg.seed),
            "max_steps_per_episode": max_steps_per_episode,
            "init_state_indices": init_state_indices,
            "policy_backend": str(cfg.EVALUATION.get("policy_backend", "recoverwam_native")),
        }
        task_results = run_single_task(
            task=task,
            initial_states=initial_states,
            model=model,
            processor=processor,
            cfg=cfg,
            video_dir=video_dir,
            predicted_video_dir=predicted_video_dir,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        results.update(task_results)
        results["duration"] = time.time() - task_start
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_output = output_file.with_suffix(".json.tmp")
        temporary_output.write_text(
            json.dumps(results, indent=4, cls=NumpyEncoder), encoding="utf-8"
        )
        temporary_output.replace(output_file)
        all_results.append(copy.deepcopy(results))
        print(
            f"Task {task_suite_name},{task_id} completed: {results['successes']}/{num_trials} successes",
            flush=True,
        )
    return all_results
