"""Build the official RoboDojo evaluation environment without a policy server.

Must be imported after the Isaac Sim app has been launched. Mirrors
src/eval_client/main.py of RoboDojo; the only differences are that the policy
client is a stub and that camera intrinsics and extrinsics are switched on.
"""

import os

from omegaconf import OmegaConf

POLICY_NAME = "RoboBigWater"


class NullModelClient:
    """Stands in for the websocket policy client: there is no policy server."""

    def __init__(self, *_, **__):
        pass

    def call(self, *_, **__):
        return None

    def close(self):
        pass


def build_env(simulation_app, task_name, seed, env_cfg_type="arx_x5", device_id=0, additional_info="roboshell", depth=True):
    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    import src.eval_client.eval_env as eval_env_module
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    eval_env_module.WsModelClient = NullModelClient

    import importlib

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    benchmark_path = os.path.join(ROOT_DIR, "task", BENCHMARK)

    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, env_cfg_type + ".yml"))
    eval_cfg.update(
        task_name=task_name,
        num_envs=1,
        device_id=device_id,
        eval_batch=False,
        policy_name=POLICY_NAME,
        additional_info=additional_info,
        seed=seed,
        physx_monitor_enabled=False,
    )
    eval_cfg["observation"]["vision"]["intrinsic_matrix"] = True
    eval_cfg["observation"]["vision"]["extrinsic_matrix"] = True
    eval_cfg["observation"]["vision"]["depth"] = bool(depth)

    run_id = os.environ["ROBODOJO_RUN_ID"]
    deploy_cfg = dict(
        policy_name=POLICY_NAME,
        port=0,
        host="localhost",
        protocol="ws",
        policy_server_url="ws://localhost:0",
        evaluation_id=run_id,
        trial_id=f"{task_name}-{run_id}",
        action_case_id=f"{task_name}_case",
        repeat_index=None,
    )

    def cfg(kind):
        return load_yaml(os.path.join(ENV_CONFIG_PATH, kind, eval_cfg["config"][kind] + ".yml"))

    env_cfg = OmegaConf.create(
        {
            "sim": cfg("sim"),
            "scene": cfg("scene"),
            "camera": cfg("camera"),
            "robot": cfg("robot"),
            "task_env": load_yaml(task_registry.task_config_path(os.path.join(benchmark_path, "config"), task_name)),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, eval_num = process_config(env_cfg, task_name=task_name)
    if depth:
        # the rendered depth (distance to the image plane) of every camera; off in the stock camera config
        for camera_name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            OmegaConf.update(env_cfg, f"camera.annotator.{camera_name}.distance_to_image_plane_capture",
                             {"type": "distance_to_image_plane", "device": "cpu"}, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.eval_num", eval_num, force_add=True)
    OmegaConf.update(
        env_cfg, "camera.default_frequency", eval_cfg["observation"].get("collect_freq", 0), force_add=True
    )
    env_cfg.sim.seed = [0]
    return eval_env_module.create_eval_env(env_cfg, simulation_app)


def start_episode(env, layout):
    """Same order as the official loop: reset, then register the success checks and the progress score."""
    env.reset(seed=[int(layout)])
    env.run_reward()
    if hasattr(env, "get_score"):
        env.get_score()
