# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Literal

from robo_orchard_core.utils.cli import SettingConfig, pydantic_from_argparse
from robo_orchard_core.utils.ray import RayRemoteClassConfig

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projects.holobrain.policy.robotwin_policy import (  # noqa: E402
    HoloBrainRoboTwinPolicy,
    HoloBrainRoboTwinPolicyCfg,
)
from robo_orchard_lab.envs.robotwin.env import (  # noqa: E402
    config_robotwin_path,
)
from robo_orchard_lab.models.holobrain.pipeline import (  # noqa: E402
    HoloBrainInferencePipeline,
)
from robo_orchard_lab.policy.evaluator.benchmark.robotwin import (  # noqa: E402
    SEM_TASKS_16,
    RoboTwinBenchmarkEvaluatorCfg,
    RoboTwinLocalBenchmarkBackendCfg,
    RoboTwinRemoteBenchmarkBackendCfg,
)
from robo_orchard_lab.utils.env import set_env  # noqa: E402


class Config(SettingConfig):
    model_dir: str
    inference_prefix: str = "inference"
    model_prefix: str = "model"
    task_names: list[str] = list(SEM_TASKS_16)
    episode_num: int = 100
    device: str | None = None
    output_path: str = "eval_result/robotwin_eval/eval_result.json"
    mode: Literal["local", "ray"] = "local"
    config_type: Literal["demo_clean", "demo_randomized"] = "demo_clean"
    seed: int = 0
    use_action_chunk_size: int = 32
    save_video: bool = False
    gpu_ids: list[int] | None = None
    workers_per_gpu: int = 1
    ray_temp_dir: str | None = None


def artifact_root_dir() -> str:
    if os.environ.get("CLUSTER") is not None:
        return "/job_data"
    return "eval_result"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=Config.__doc__)
    try:
        return pydantic_from_argparse(Config, parser)
    except SystemExit as e:
        if e.code == 2:
            parser.print_help()
        raise


def evaluate_tasks_locally(
    policy_or_cfg,
    task_names: list[str],
    episode_num: int,
    device: str,
    config_type: Literal["demo_clean", "demo_randomized"] = "demo_clean",
    seed: int = 0,
    save_video: bool = False,
) -> dict:
    """Evaluate tasks locally with the benchmark evaluator replacement."""

    result = RoboTwinBenchmarkEvaluatorCfg(
        task_names=task_names,
        episode_num=episode_num,
        config_type=config_type,
        start_seed=seed,
        format_datatypes=True,
        fail_fast=True,
        backend=RoboTwinLocalBenchmarkBackendCfg(),
        artifact_root_dir=artifact_root_dir() if save_video else None,
    )().evaluate(policy_or_cfg, device=device)
    return result.metrics


def evaluate_tasks_remote(
    policy_or_cfg,
    task_names: list[str],
    episode_num: int,
    device: str,
    gpu_ids: list[int],
    workers_per_gpu: int,
    config_type: Literal["demo_clean", "demo_randomized"] = "demo_clean",
    seed: int = 0,
    ray_temp_dir: str | None = None,
    save_video: bool = False,
) -> dict:
    if len(gpu_ids) == 0:
        raise ValueError("Ray evaluation requires at least one gpu_id.")
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be >= 1.")

    ray_inst_gpus = 1.0 / workers_per_gpu
    cuda_devices_env = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    runtime_env_vars = {"RoboTwin_PATH": config_robotwin_path()}
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        runtime_env_vars["PYTHONPATH"] = pythonpath
    if ray_temp_dir is None:
        ray_temp_dir = os.path.abspath(".ray")
    os.makedirs(ray_temp_dir, exist_ok=True)

    with set_env(CUDA_VISIBLE_DEVICES=cuda_devices_env):
        result = RoboTwinBenchmarkEvaluatorCfg(
            task_names=task_names,
            episode_num=episode_num,
            config_type=config_type,
            start_seed=seed,
            format_datatypes=True,
            backend=RoboTwinRemoteBenchmarkBackendCfg(
                num_parallel_envs=len(gpu_ids) * workers_per_gpu,
                ray_init_config={"_temp_dir": ray_temp_dir},
                remote_class_config=RayRemoteClassConfig(
                    num_cpus=1,
                    num_gpus=ray_inst_gpus,
                    memory=16 * 1024**3,
                    runtime_env={"env_vars": runtime_env_vars},
                ),
            ),
            artifact_root_dir=artifact_root_dir() if save_video else None,
        )().evaluate(policy_or_cfg, device=device)
    return result.metrics


def run(cfg: Config) -> dict:
    resolved_device = cfg.device or ("cuda" if cfg.mode == "ray" else "cpu")

    if cfg.mode == "local":
        pipeline = HoloBrainInferencePipeline.load_pipeline(
            directory=cfg.model_dir,
            inference_prefix=cfg.inference_prefix,
            device=resolved_device,
            load_weights=True,
            load_impl="native",
            model_prefix=cfg.model_prefix,
        )
        pipeline.model.eval()
        policy_or_cfg = HoloBrainRoboTwinPolicy(
            cfg=HoloBrainRoboTwinPolicyCfg(
                use_action_chunk_size=cfg.use_action_chunk_size
            ),
            pipeline=pipeline,
        )
        metrics = evaluate_tasks_locally(
            policy_or_cfg=policy_or_cfg,
            task_names=cfg.task_names,
            episode_num=cfg.episode_num,
            device=resolved_device,
            config_type=cfg.config_type,
            seed=cfg.seed,
            save_video=cfg.save_video,
        )
    elif cfg.mode == "ray":
        if cfg.gpu_ids is None:
            raise ValueError("Ray evaluation requires gpu_ids to be set.")
        policy_or_cfg = HoloBrainRoboTwinPolicyCfg(
            model_dir=cfg.model_dir,
            inference_prefix=cfg.inference_prefix,
            model_prefix=cfg.model_prefix,
            use_action_chunk_size=cfg.use_action_chunk_size,
        )
        metrics = evaluate_tasks_remote(
            policy_or_cfg=policy_or_cfg,
            task_names=cfg.task_names,
            episode_num=cfg.episode_num,
            device=resolved_device,
            gpu_ids=cfg.gpu_ids,
            workers_per_gpu=cfg.workers_per_gpu,
            config_type=cfg.config_type,
            seed=cfg.seed,
            ray_temp_dir=cfg.ray_temp_dir,
            save_video=cfg.save_video,
        )
    else:
        raise ValueError(f"Unsupported evaluation mode: {cfg.mode}")

    output_dir = os.path.dirname(cfg.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=4)
    return metrics


def main() -> None:
    cfg = parse_args()
    metrics = run(cfg)
    print(
        json.dumps(
            {"config": cfg.model_dump(), "metrics": metrics},
            indent=4,
        )
    )


if __name__ == "__main__":
    main()
