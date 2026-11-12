"""Run the upstream RoboDojo evaluator in the Isaac Sim subprocess."""

from pathlib import Path
from tempfile import TemporaryDirectory


def _layout_seeds(value):
    seeds = [value] if isinstance(value, int) else value
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("layout_seed must be a nonnegative integer or a list of such integers.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("layout_seed contains duplicate directories.")
    return seeds


def _combine_layouts(manager, seeds):
    """Keep RoboDojo's task filtering and numeric file ordering in each directory."""
    entries = []
    for seed in seeds:
        source = manager
        if seed != seeds[0]:
            source = type(manager)({**manager.config, "seed": seed})
            source.init_eval()
        if not source.seed_list:
            raise FileNotFoundError(f"No layouts for {manager.task_name} in layout_seed={seed}.")
        entries.extend(source.seed_info[i] for i in source.seed_list)
    manager.seed_info = dict(enumerate(entries))
    manager.seed_list = list(manager.seed_info)
    manager.ed_idx = len(entries)
    return len(entries)


class _RolloutClient:
    """The evaluator resets its policy client. RLinf already owns that policy."""

    def __init__(self, **connection_options):
        pass

    def call(self, func_name):
        if func_name != "reset":
            raise RuntimeError(f"Unexpected RoboDojo policy request: {func_name}")


class SimulatorBackend:
    def __init__(self, cfg):
        # AppLauncher must run before importing any app-dependent env modules.
        from isaaclab.app import AppLauncher

        from env.camera_manager.capture.render_sync import zero_delay_kit_args

        self.app = AppLauncher(
            headless=True,
            enable_cameras=True,
            device=cfg.get("device", "cuda:0"),
            multi_gpu=False,
            kit_args="--/renderer/multiGpu/enabled=false " + zero_delay_kit_args(),
        ).app
        self.env = None
        self._output = TemporaryDirectory(prefix="robodojo-rlinf-")
        try:
            self._initialize(cfg)
        except BaseException:
            import traceback

            traceback.print_exc()
            self.close()
            raise

    def _initialize(self, cfg):
        import importlib

        from omegaconf import OmegaConf

        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from src.eval_client import eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        task_name = cfg.get("task_name", "stack_bowls")
        if not isinstance(task_name, str) or not task_name.isidentifier():
            raise ValueError(f"Expected a RoboDojo task module name, got {task_name!r}.")
        root = Path(ENV_CONFIG_PATH)
        task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
        eval_cfg = load_yaml(str(root / f"{cfg['env_cfg_type']}.yml"))
        layout_seeds = _layout_seeds(cfg.get("layout_seed", 0))
        eval_cfg.update(
            task_name=task_name,
            num_envs=cfg["num_envs"],
            device_id=0,
            eval_batch=True,
            policy_name="RLinf",
            additional_info="rl",
            seed=layout_seeds[0],
            physx_monitor_enabled=False,
        )
        parts = {
            k: load_yaml(str(root / k / f"{eval_cfg['config'][k]}.yml")) for k in ("sim", "scene", "camera", "robot")
        }
        config = OmegaConf.create(
            {
                **parts,
                "eval_cfg": eval_cfg,
                "deploy_cfg": {"policy_name": "RLinf", "port": 0},
                "task_env": load_yaml(
                    task_registry.task_config_path(str(Path(ROOT_DIR) / "task" / BENCHMARK / "config"), task_name)
                ),
            }
        )
        config.sim.scene.num_envs = cfg["num_envs"]
        # Match the native evaluator's CPU tensor/physics pipeline. Its rigid
        # object reset API cannot clear velocities in the GPU tensor pipeline.
        # Rendering and CuRobo still use the selected CUDA device.
        config.sim.device = "cpu"
        config.sim.seed = [cfg.get("seed", 0) + i for i in range(cfg["num_envs"])]
        config, _ = process_config(process_randomization(config), task_name=task_name)
        config.camera.default_frequency = eval_cfg["observation"].get("collect_freq", 25)
        layout_dir = (
            Path(ROOT_DIR) / "Assets" / "Eval_Layout" / BENCHMARK / eval_cfg["config_name"] / str(eval_cfg["seed"])
        )
        if not layout_dir.is_dir():
            raise FileNotFoundError(f"Missing evaluation layout directory for {task_name}: {layout_dir}")
        # The upstream factory constructs its WebSocket client eagerly. Bind the
        # local reset endpoint only during construction in this dedicated child.
        client = eval_env.WsModelClient
        try:
            eval_env.WsModelClient = _RolloutClient
            self.env = eval_env.create_eval_env(
                config, self.app, resume_state={"save_dir": self._output.name}
            )
        finally:
            eval_env.WsModelClient = client
        self.env._stream_vision = lambda env_idx, frame: None
        self._observe_rewards()
        self.layout_count = _combine_layouts(self.env.seed_manager, layout_seeds)
        self.env.step_lim = min(self.env.step_lim, int(cfg["max_episode_steps"]))

    def _observe_rewards(self):
        # is_episode_end calls get_reward with the native final_check flag.
        # A second call can consume check/score state, so capture that result.
        native_get_reward = self.env.reward_manager.get_reward

        def get_reward(*args, **kwargs):
            self._step_rewards = native_get_reward(*args, **kwargs)
            return self._step_rewards

        self.env.reward_manager.get_reward = get_reward

    def metadata(self):
        from env.camera_manager.capture.render_sync import zero_delay_active

        return {
            "task_name": self.env.task_name,
            "max_episode_steps": self.env.step_lim,
            "num_layouts": self.layout_count,
            "robot_action_dim_info": self.env.robot_action_dim_info,
            "physics_device": self.env.device,
            "render_zero_delay": zero_delay_active(),
        }

    def reset(self, layout_ids):
        self.env.reset(seed=layout_ids)
        if any(self.env.end_flag):
            raise RuntimeError("Unstable/invalid initial layout; do not train on an invalid reset.")
        self.env.run_reward()
        if hasattr(self.env, "get_score"):
            self.env.get_score()
        if getattr(self.env, "interact", False) and hasattr(self.env, "query_support_arm_traj"):
            for env_idx in self.env.get_running_env_idx_list():
                self.env.query_support_arm_traj(env_idx=env_idx)
        return self._snapshot([0.0] * self.env.num_envs)

    def step(self, actions, env_indices):
        self._step_rewards = None
        self.env.take_action_batch(actions, env_idx_list=env_indices)
        if self._step_rewards is None:
            raise RuntimeError("RoboDojo completed an action without evaluating its reward.")
        return self._snapshot(self._step_rewards)

    def _snapshot(self, rewards):
        import numpy as np

        env = self.env
        # Native observations are RGB uint8; return CPU-only serializable data.
        def cpu(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [cpu(v) for v in value]
            return value

        done = np.asarray(env.end_flag, dtype=bool)
        success = done & np.asarray(env.success, dtype=bool)
        timeout = done & ~success & (np.asarray(env.take_action_cnt) >= env.step_lim)
        snapshot = {
            "observations": cpu(env.get_obs_batch(last_frame=True)),
            "reward": rewards,
            "success": success,
            "failed": done & ~success & ~timeout,
            "truncated": timeout,
        }
        if hasattr(env, "get_score"):
            snapshot["official_score"] = env.reward_manager.get_score()
        return snapshot

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            try:
                self.app.close()
            finally:
                self._output.cleanup()
