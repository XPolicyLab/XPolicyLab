# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Use the pinned RoboDojo simulator and run_eval without rewriting its scorer."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roborsi.evaluation.messages import Task
from robots.robodojo.roborsi_adapter import ACTION_SCHEMA


class RoboRSIPolicyClient:
    """RoboRSI owns Agent reset; native reset must not contact another policy server."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def call(self, func_name: str, **kwargs: Any) -> None:
        if func_name != "reset" or kwargs:
            raise ValueError("RoboRSI only replaces the native policy reset transport")

    def close(self) -> None:
        pass


def actual_task(env: Any, obs: dict, native_limit: int) -> Task:
    """Record the actual loaded layout, instruction, cameras and action budget."""
    if env.num_envs != 1 or env.robot_action_dim_info != {
        "arm_dim": [6, 6],
        "ee_dim": [1, 1],
    }:
        raise ValueError(
            "RoboRSI RoboDojo requires one scene and two six-joint target arms"
        )
    layout_id = int(env.env_seeds[0])
    layout = Path(env.seed_manager.seed_info[layout_id]["scene_layout"])
    return Task(
        benchmark="robodojo",
        task_id=env.task_name,
        seed=layout_id,
        instruction=str(obs["instruction"]),
        max_steps=int(env.step_lim),
        native={
            "config": env.config_name,
            "eval_seed": int(env.eval_seed),
            "native_step_limit": native_limit,
            "layout_sha256": hashlib.sha256(layout.read_bytes()).hexdigest(),
        },
        capabilities={
            "action_schema": ACTION_SCHEMA,
            "observation_channels": sorted(obs["vision"]),
        },
    )


class NativeRuntime:
    """Initialize Isaac once per process and run the official evaluator once."""

    def __init__(self) -> None:
        self._app = None

    def close(self) -> None:
        """Exit Isaac only after the worker has delivered the episode reply."""
        app, self._app = self._app, None
        if app is not None:
            # Isaac 5.1 terminates Python here, including pending RPC replies.
            app.close()

    def query(self, env: Any, method: str, arguments: dict) -> dict:
        """Read EEF limits or plan on the worker's native owner thread."""
        from robots.robodojo.eef_contracts import InfrastructureFailure
        from robots.robodojo.eef_queries import query

        try:
            return query(env, method, arguments)
        except InfrastructureFailure as exc:
            # The source intentionally derives this failure from BaseException.
            # RPC dispatch catches Exception; preserve it as an operational error.
            raise RuntimeError(str(exc)) from exc

    def run(self, specification: dict, interact: Callable) -> dict:
        from robots.robodojo.benchmark_service import NATIVE_ROOT, verify_native_source

        verify_native_source()
        # EEF bounds must come from the same tree that the worker just verified.
        os.environ["ROBODOJO_ROOT"] = str(NATIVE_ROOT)
        # Isaac requires AppLauncher before imports that create simulator types.
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
        from isaaclab.app import AppLauncher
        from src.eval_client.physx_warning_monitor import get_monitor

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        arguments = parser.parse_args(
            ["--headless", "--enable_cameras", "--device", "cuda:0"]
        )
        add_zero_delay_kit_args(arguments)
        # Match native main.py: articulation tasks enable the PhysX monitor.
        from env.global_configs import ROOT_DIR
        from task.RoboDojo import task_registry
        from utils.load_file import load_yaml

        config_path = task_registry.task_config_path(
            str(Path(ROOT_DIR) / "task/RoboDojo/config"), specification["task_id"]
        )
        monitor_enabled = bool((load_yaml(config_path) or {}).get("Articulation"))
        get_monitor().start(enabled=monitor_enabled)
        app = None
        env = None
        try:
            app = self._app = AppLauncher(arguments).app
            env = self._environment(specification, app, monitor_enabled)
            native_limit = int(env.step_lim)
            requested_limit = specification.get("max_steps")
            if requested_limit is not None:
                if (
                    type(requested_limit) is not int
                    or not 1 <= requested_limit <= native_limit
                ):
                    raise ValueError(
                        "RoboDojo max_steps must not exceed its native budget"
                    )
                env.step_lim = requested_limit
            env.reset(seed=[specification["layout_id"]])
            if env.env_seeds != [specification["layout_id"]] or env.unstable_envs:
                raise ValueError("RoboDojo did not load the requested stable layout")

            def policy():
                obs = env.get_obs()
                task = actual_task(env, obs, native_limit)
                interact(env, task, obs)

            # This is the policy boundary. run_eval still owns reward setup,
            # support-arm setup, final scoring, exclusions and native output.
            env.eval_one_episode = policy
            env.run_eval()
            return {"source": str((Path(env.save_dir) / "_result.json").resolve())}
        finally:
            try:
                if env is not None:
                    try:
                        env.model_client.close()
                    finally:
                        env.close()
            finally:
                get_monitor().shutdown()

    def _environment(self, specification: dict, app: Any, monitor_enabled: bool) -> Any:
        from env.global_configs import ENV_CONFIG_PATH, ROOT_DIR
        from omegaconf import OmegaConf
        from src.eval_client import eval_env
        from task.RoboDojo import task_registry
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        task_id = specification["task_id"]
        root = Path(ENV_CONFIG_PATH)
        evaluation = load_yaml(str(root / "arx_x5.yml"))
        evaluation.update(
            {
                "task_name": task_id,
                "num_envs": 1,
                "device_id": 0,
                "eval_batch": False,
                "policy_name": "RoboRSI",
                "additional_info": "roborsi",
                "seed": specification["eval_seed"],
                "physx_monitor_enabled": monitor_enabled,
            }
        )
        config = OmegaConf.create(
            {
                **{
                    key: load_yaml(
                        str(root / key / (evaluation["config"][key] + ".yml"))
                    )
                    for key in ("sim", "scene", "camera", "robot")
                },
                "task_env": load_yaml(
                    task_registry.task_config_path(
                        str(Path(ROOT_DIR) / "task/RoboDojo/config"), task_id
                    )
                ),
                "eval_cfg": evaluation,
                "deploy_cfg": {
                    "policy_name": "RoboRSI",
                    "port": 1,
                    "host": "127.0.0.1",
                },
            }
        )
        config.sim.scene.num_envs = 1
        config = process_randomization(config)
        config, count = process_config(config, task_name=task_id)
        config.eval_cfg.eval_num = count
        config.camera.default_frequency = evaluation["observation"].get(
            "collect_freq", 0
        )
        config.sim.seed = [0]
        original = eval_env.WsModelClient
        try:
            # Replace only the policy transport in this dedicated process.
            # No native task, control, seed or evaluator implementation is changed.
            eval_env.WsModelClient = RoboRSIPolicyClient
            return eval_env.create_eval_env(config, app)
        finally:
            eval_env.WsModelClient = original


def make_runtime() -> NativeRuntime:
    return NativeRuntime()
