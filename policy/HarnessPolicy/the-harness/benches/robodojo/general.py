"""RoboDojo worker for any task in the pinned checkout, one task per process.

``adapter.py`` stays the pinned ``stack_bowls`` worker. This module reuses its
action layout, observation, motion video and execution code, and replaces only
the parts that name one task: task and layout selection, the ``_task.yml``
scene/robot overrides, and stability handling for layouts and scripted support arms.

``layout_source`` picks what ``reset(task, seed)`` replays. ``"eval"`` (default)
is layout ``seed`` of the official eval pack ``layout_pack_seed``.
``"procedural"`` is development layout ``seed`` from ``dev_layout_dir``
(default ``<checkout>/harness_dev_layouts/arx_x5``), generated on first use by
``devlayouts.py`` and disjoint from every eval pack.
Both go through RoboDojo's own replay path.

Importing this module does not import Isaac, torch, or RoboDojo.
"""

import os
from pathlib import Path
import sys
import uuid

from harness_policy.core.contracts import Action, Observation, Transition

from .adapter import RoboDojoEnvironment, _checkout, _discard_eval_video, _NoPolicyClient
from .codec import DEFAULT_INSTRUCTION, ENV_CONFIG, EpisodeCursor
from .devlayouts import MAX_UNSTABLE_SKIPS, discard_dev_layout, ensure_dev_layout, validate_dev_seed
from .tasks import (
    BENCHMARK,
    DEFAULT_TASK,
    layout_count,
    layout_dir,
    task_catalog,
    task_settings,
    validate_layout,
    validate_layout_pack_seed,
    validate_task,
)


def create_environment(**options):
    if os.environ.get("HARNESS_ROBODOJO_RUNTIME") != "1":
        raise RuntimeError(
            "RoboDojo adapter must run in the RoboDojo Python runtime with "
            "HARNESS_ROBODOJO_RUNTIME=1. See benches/robodojo/README.md."
        )
    unknown = set(options) - {"device", "headless", "checkout", "task", "layout_pack_seed",
                              "layout_source", "dev_layout_dir"}
    if unknown:
        raise ValueError(f"Unknown RoboDojo options: {sorted(unknown)}")
    device = options.get("device", "cuda:0")
    if device != "cuda:0":
        raise ValueError("RoboDojo worker uses cuda:0 inside CUDA_VISIBLE_DEVICES")
    return RoboDojoTaskEnvironment(
        task=options.get("task", DEFAULT_TASK),
        layout_pack_seed=options.get("layout_pack_seed", 0),
        device=device,
        headless=options.get("headless", True),
        checkout=options.get("checkout"),
        layout_source=options.get("layout_source", "eval"),
        dev_layout_dir=options.get("dev_layout_dir"),
    )


LAYOUT_SOURCES = ("eval", "procedural")
# Development layouts of a checkout when no dev_layout_dir is given, like RoboTwin's harness_seeds.
DEV_LAYOUT_SUBDIR = "harness_dev_layouts"


def default_dev_layout_dir(checkout):
    return Path(checkout) / DEV_LAYOUT_SUBDIR / ENV_CONFIG


def validate_layout_source(layout_source, dev_layout_dir, default_dir=None):
    if layout_source not in LAYOUT_SOURCES:
        raise ValueError(f"layout_source must be one of {list(LAYOUT_SOURCES)}")
    if layout_source == "procedural":
        if dev_layout_dir is None and default_dir is not None:
            return layout_source, Path(default_dir).absolute()
        if not isinstance(dev_layout_dir, str) or not dev_layout_dir:
            raise ValueError("layout_source='procedural' needs dev_layout_dir")
        return layout_source, Path(dev_layout_dir).absolute()
    if dev_layout_dir is not None:
        raise ValueError("dev_layout_dir is only used with layout_source='procedural'")
    return layout_source, None


class RoboDojoTaskEnvironment(RoboDojoEnvironment):
    """One sequential RoboDojo task behind the harness worker.

    The task is fixed when the worker starts. Changing tasks means restarting
    the worker, as the upstream evaluator runs one task per process.
    """

    name = "robodojo-isaac51-general"

    def __init__(self, task=DEFAULT_TASK, layout_pack_seed=0, device="cuda:0", headless=True, checkout=None,
                 layout_source="eval", dev_layout_dir=None):
        # Validate everything that can be checked on disk before Kit starts.
        # _boot changes directory before extending sys.path, so a relative root would break imports.
        self.checkout = _checkout(checkout).absolute()
        self._layout_source, self._dev_layout_dir = validate_layout_source(
            layout_source, dev_layout_dir, default_dev_layout_dir(self.checkout))
        self.device = device
        self.headless = bool(headless)
        self._task = validate_task(task, task_catalog(self.checkout))
        self._layout_pack_seed = validate_layout_pack_seed(layout_pack_seed)
        self._layout_count = layout_count(
            layout_dir(self.checkout, ENV_CONFIG, self._layout_pack_seed), self._task
        )
        if self._layout_count < 1:
            raise RuntimeError(
                f"No {ENV_CONFIG}/{self._layout_pack_seed} evaluation layouts for task {self._task!r}"
            )
        self._settings = None
        self._app = None
        self._env = None
        self._bounds = None
        self._layout = None
        self._tools = None
        self._seed = None
        self._cursor = EpisodeCursor()
        try:
            self._boot()
        except Exception as error:
            if self._app is not None:
                print(
                    f"RoboDojo worker failed during startup: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                self.close()
            except Exception:
                pass
            if isinstance(error, RuntimeError):
                raise
            raise RuntimeError(
                f"RoboDojo worker failed during startup: {type(error).__name__}: {error}"
            ) from error

    def reset(self, task: str, seed: int) -> Observation:
        if task != self._task:
            raise ValueError(
                f"This RoboDojo worker serves task={self._task!r}; restart it with task={task!r} in --options"
            )
        if self._layout_source == "procedural":
            validate_dev_seed(seed)
        else:
            validate_layout(seed, self._layout_count)
        self._reseed(seed)
        self._cursor.start(uuid.uuid4().hex)
        return self._observe()

    def execute(self, action: Action, episode_id: str, step: int) -> Transition:
        transition = super().execute(action, episode_id, step)
        # A scripted support arm (make_kong, play_tic_tac_toe, imitate_sorting_sequence)
        # can flag the episode unstable. Upstream drops it from the count; it is
        # neither a policy success nor a failure, so report it as an error.
        if 0 in getattr(self._env, "unstable_envs", ()):
            self._cursor.invalidate()
            raise RuntimeError(
                f"RoboDojo marked task={self._task} layout={self._seed} unstable after step {step}; "
                "the action was executed and the episode is not scorable"
            )
        return transition

    def _boot(self):
        runtime = Path(os.environ.get("TMPDIR", "/tmp")) / "harness-robodojo"
        runtime.mkdir(parents=True, exist_ok=True)
        os.chdir(runtime)
        os.environ.setdefault("ROBODOJO_RUN_ID", "harness-worker")
        for entry in (str(self.checkout / "XPolicyLab"), str(self.checkout)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        self._launch_kit()
        self._open()
        # Startup only needs a loaded scene to read the action spec. An evaluation layout that
        # does not settle on this machine must not take the worker down with it: try the next.
        attempts = min(6, self._layout_count) if self._layout_source == "eval" else 1
        for layout in range(attempts):
            try:
                self._reseed(layout)
                break
            except RuntimeError as error:
                if "unstable after reset" not in str(error) or layout == attempts - 1:
                    raise
                print(f"startup: {error}; trying layout {layout + 1}", file=sys.stderr, flush=True)
        else:
            raise RuntimeError(f"No RoboDojo layout of task {self._task} loaded at startup")
        self._read_action_spec()
        print(
            f"RoboDojo worker ready task={self._task} config={ENV_CONFIG} "
            f"robot={self._settings['robot_config']} scene={self._settings['scene_config']} "
            f"layouts={self._layout_count} layout_pack_seed={self._layout_pack_seed} "
            f"layout_source={self._layout_source} dev_layout_dir={self._dev_layout_dir} device={self.device}",
            file=sys.stderr,
            flush=True,
        )

    def _open(self):
        import importlib

        from omegaconf import OmegaConf

        from env.global_configs import BENCHMARK as UPSTREAM_BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from src.eval_client import eval_env as eval_env_module
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        if UPSTREAM_BENCHMARK != BENCHMARK:
            raise RuntimeError(f"Expected RoboDojo benchmark {BENCHMARK}, got {UPSTREAM_BENCHMARK}")
        task = self._task
        benchmark_config = os.path.join(ROOT_DIR, "task", BENCHMARK, "config")
        self._settings = task_settings(task, load_yaml(os.path.join(benchmark_config, "_task.yml")))

        eval_env_module.WsModelClient = _NoPolicyClient
        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, f"{ENV_CONFIG}.yml"))
        eval_cfg["task_name"] = task
        eval_cfg["num_envs"] = 1
        eval_cfg["device_id"] = 0
        eval_cfg["eval_batch"] = False
        eval_cfg["policy_name"] = "harness"
        eval_cfg["additional_info"] = "harness"
        eval_cfg["seed"] = self._layout_pack_seed
        eval_cfg["eval_num"] = 1
        eval_cfg["physx_monitor_enabled"] = False
        deploy_cfg = {
            "policy_name": "harness",
            "port": 1,
            "host": "127.0.0.1",
            "protocol": "ws",
            "policy_server_url": "ws://127.0.0.1:1",
            "evaluation_id": os.environ.get("ROBODOJO_RUN_ID", "harness-worker"),
            "trial_id": f"{task}-harness",
            "action_case_id": f"{task}_case",
            "repeat_index": None,
        }
        task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
        env_cfg = OmegaConf.create({
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", f"{eval_cfg['config']['sim']}.yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", f"{eval_cfg['config']['scene']}.yml")),
            "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", f"{eval_cfg['config']['camera']}.yml")),
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", f"{eval_cfg['config']['robot']}.yml")),
            "task_env": load_yaml(str(task_registry.task_config_path(benchmark_config, task))),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        })
        OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
        env_cfg = process_randomization(env_cfg)
        # process_config swaps in the task's scene and robot configs from _task.yml.
        env_cfg, _eval_num = process_config(env_cfg, task_name=task)
        if env_cfg.robot is None:
            raise RuntimeError(f"Robot config {self._settings['robot_config']} did not load")
        OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.eval_num", 1, force_add=True)
        OmegaConf.update(
            env_cfg,
            "camera.default_frequency",
            int(eval_cfg["observation"].get("collect_freq", 0)),
            force_add=True,
        )
        env_cfg.sim.seed = [0]
        self._env = eval_env_module.create_eval_env(env_cfg, self._app)
        self._env._stream_vision = _discard_eval_video
        loaded = len(getattr(self._env.seed_manager, "seed_info", {}))
        if loaded != self._layout_count:
            raise RuntimeError(
                f"RoboDojo loaded {loaded} layouts for {task}, expected {self._layout_count}"
            )
        if self._layout_source == "procedural":
            # Only dev layouts registered by _reseed can be replayed from here on.
            self._env.seed_manager.seed_info = {}
        self._install_motion_video()

    def _reseed(self, seed: int):
        from utils.cluttered_generator import UnStableError

        if self._layout_source == "procedural":
            # A generated layout can pass placement but settle unstably in physics.
            # Then the next acceptable layout in the seed's deterministic sequence is
            # used, so a dev seed always maps to the same stable layout.
            for skip in range(MAX_UNSTABLE_SKIPS):
                path = ensure_dev_layout(self.checkout, ENV_CONFIG, self._dev_layout_dir, self._task, seed,
                                         skip=skip)
                self._env.seed_manager.seed_info[seed] = {"scene_layout": str(path)}
                # Settling is not perfectly repeatable, so a layout gets a second reset
                # before it is replaced; a seed then keeps its layout across runs.
                for attempt in range(2):
                    try:
                        super()._reseed(seed)
                        return
                    except UnStableError as error:
                        print(f"dev layout {path} (skip {skip}, reset {attempt}) unstable: {error}",
                              file=sys.stderr, flush=True)
                discard_dev_layout(path)
            raise RuntimeError(
                f"No stable dev layout for {self._task} seed {seed} in {MAX_UNSTABLE_SKIPS} candidates")
        try:
            super()._reseed(seed)
        except UnStableError as error:
            raise RuntimeError(
                f"RoboDojo layout {seed} of task {self._task} is unstable after reset: {error}"
            ) from error

    def _observe(self):
        observation = super()._observe()
        # The pinned stack_bowls worker falls back to its own instruction. Any
        # other task must bring its own text from the observation manager.
        if self._task != DEFAULT_TASK and observation.instruction == DEFAULT_INSTRUCTION:
            raise RuntimeError(f"RoboDojo observation for {self._task} has no instruction")
        observation.state.update({
            "robot_config": self._settings["robot_config"],
            "scene_config": self._settings["scene_config"],
            "layout_source": self._layout_source,
            "layout_pack_seed": self._layout_pack_seed if self._layout_source == "eval" else None,
            "layout_count": self._layout_count if self._layout_source == "eval" else None,
        })
        return observation
