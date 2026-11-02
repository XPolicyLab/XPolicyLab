"""Boot of the native RoboDojo evaluator, mirroring ``src/eval_client/main.py``.

Every simulator import happens inside a function, after the RoboDojo root is
on ``sys.path`` and the working directory is that root (RoboDojo resolves
assets and outputs relative to it). The only deviation from the official boot
is the policy transport: the WebSocket model client is replaced by a stub,
because actions come from the KnowinAgent runtime instead of a policy server.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

POLICY_NAME = "KnowinAgent"
ADDITIONAL_INFO = "knowin_agent"
EVAL_CFG_NAME = "arx_x5"
KIT_ARGS_ENV = "KNOWIN_AGENT_ROBODOJO_KIT_ARGS"
OFFICIAL_KIT_ARGS = "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
DEPTH_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
DEPTH_ANNOTATOR = {"type": "distance_to_image_plane", "device": "cpu"}


class LayoutUnstable(RuntimeError):
    """The layout failed the native stability check; the official evaluator skips such layouts."""


class NoPolicyClient:
    """Stands in for XPolicyLab's WsModelClient; only the native reset notification is accepted."""

    def __init__(self, **config: Any):
        self.config = config

    def call(self, func_name: Optional[str] = None, **_: Any) -> None:
        if func_name == "reset":
            return None
        raise RuntimeError(f"RoboDojo asked the policy client for {func_name!r}; KnowinAgent supplies actions directly")

    def close(self) -> None:
        pass


@contextlib.contextmanager
def robodojo_cwd(root: Optional[Path]) -> Iterator[None]:
    if root is None:
        yield
        return
    previous = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def prepare_process(root: Path) -> str:
    """PYTHONPATH as scripts/eval_policy.sh sets it, and a run id unique to this process."""
    for path in (str(root / "XPolicyLab"), str(root)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    if not os.environ.get("ROBODOJO_RUN_ID"):
        os.environ["ROBODOJO_RUN_ID"] = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_knowin_agent{os.getpid()}"
    return os.environ["ROBODOJO_RUN_ID"]


def task_has_articulation(root: Path, task: str) -> bool:
    """main.py's rule for enabling the PhysX log monitor; unreadable config means enabled."""
    try:
        import yaml

        with open(root / "task" / "RoboDojo" / "config" / f"{task}.yml", encoding="utf-8") as handle:
            return bool((yaml.safe_load(handle) or {}).get("Articulation"))
    except Exception:
        return True


def start_physx_monitor(root: Path, task: str) -> Any:
    """Start the native PhysX warning monitor before Kit launches, when the task needs it."""
    if not task_has_articulation(root, task):
        return None
    from src.eval_client.physx_warning_monitor import get_monitor

    monitor = get_monitor()
    monitor.start(enabled=True)
    return monitor


def launch_app(kit_args: Optional[str] = None) -> tuple[Any, dict]:
    """AppLauncher with the flags scripts/eval_policy.sh passes to main.py."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless", "--enable_cameras"])
    if kit_args is None:
        kit_args = os.environ.get(KIT_ARGS_ENV, OFFICIAL_KIT_ARGS)
    args.kit_args = kit_args
    try:
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
    except ImportError:
        add_zero_delay_kit_args = None
    if add_zero_delay_kit_args is not None:
        add_zero_delay_kit_args(args)
    app = AppLauncher(args).app
    return app, {"kit_args": args.kit_args, "zero_delay_kit_args": add_zero_delay_kit_args is not None}


def build_env_cfg(task: str, bank: int, *, native_depth: bool, monitor_enabled: bool, run_id: str) -> Any:
    """The OmegaConf config main.py's ``main()`` assembles, for one env and one layout bank."""
    from omegaconf import OmegaConf

    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, EVAL_CFG_NAME + ".yml"))
    eval_cfg.update(
        task_name=task,
        num_envs=1,
        device_id=0,
        eval_batch=False,
        policy_name=POLICY_NAME,
        additional_info=ADDITIONAL_INFO,
        seed=bank,
        physx_monitor_enabled=monitor_enabled,
    )
    deploy_cfg = {
        "policy_name": POLICY_NAME,
        "port": 0,
        "host": "localhost",
        "protocol": "ws",
        "policy_server_url": "",
        "evaluation_id": run_id,
        "trial_id": f"{task}-{run_id}",
        "action_case_id": f"{task}_case",
        "repeat_index": None,
    }
    parts = {
        name: load_yaml(os.path.join(ENV_CONFIG_PATH, name, eval_cfg["config"][name] + ".yml"))
        for name in ("sim", "scene", "camera", "robot")
    }
    task_cfg = task_registry.task_config_path(os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), task)
    env_cfg = OmegaConf.create({**parts, "task_env": load_yaml(task_cfg), "eval_cfg": eval_cfg, "deploy_cfg": deploy_cfg})
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, _ = process_config(env_cfg, task_name=task)
    OmegaConf.update(
        env_cfg, "camera.default_frequency", eval_cfg["observation"].get("collect_freq", 0), force_add=True
    )
    env_cfg.sim.seed = [0]
    if native_depth:
        enable_native_depth(env_cfg)
    return env_cfg


def enable_native_depth(env_cfg: Any) -> None:
    """Attach distance_to_image_plane to the three cameras and expose it as vision depth."""
    from omegaconf import OmegaConf

    annotators = env_cfg.camera.get("annotator")
    annotators = OmegaConf.to_container(annotators, resolve=True) if annotators is not None else {}
    common = annotators.get("common") or {"enabled": True, "rgb_capture": {"type": "rgb", "device": "cpu"}}
    for name in DEPTH_CAMERAS:
        entry = copy.deepcopy(annotators.get(name) or common)
        entry["enabled"] = True
        entry["distance_to_image_plane_capture"] = dict(DEPTH_ANNOTATOR)
        annotators[name] = entry
    env_cfg.camera.annotator = annotators
    OmegaConf.update(env_cfg, "eval_cfg.observation.vision.depth", True, force_add=True)


def _install_client_stub() -> None:
    modules = {name: types.ModuleType(name) for name in ("client_server", "client_server.ws", "client_server.ws.model_client")}
    modules["client_server"].ws = modules["client_server.ws"]
    modules["client_server.ws"].model_client = modules["client_server.ws.model_client"]
    modules["client_server.ws.model_client"].WsModelClient = NoPolicyClient
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


def import_eval_env() -> Any:
    """src.eval_client.eval_env, with a stub transport if XPolicyLab is not checked out."""
    try:
        return importlib.import_module("src.eval_client.eval_env")
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("client_server"):
            raise
    _install_client_stub()
    return importlib.import_module("src.eval_client.eval_env")


def create_native(env_cfg: Any, app: Any) -> Any:
    eval_env = import_eval_env()
    original = eval_env.WsModelClient
    eval_env.WsModelClient = NoPolicyClient
    try:
        return eval_env.create_eval_env(env_cfg, app, resume_state=None)
    finally:
        eval_env.WsModelClient = original


def begin_episode(native: Any, ordinal: int, monitor: Any = None) -> None:
    """main.py's per-batch reset followed by the preamble of ``EvalEnv.run_eval``."""
    if monitor is not None:
        monitor.reset()
    try:
        native.reset(seed=[ordinal])
    except Exception as exc:
        if type(exc).__name__ == "UnStableError":
            raise LayoutUnstable(f"layout ordinal {ordinal} failed the native stability check") from exc
        raise
    if native.episode_nums != 1 or native.end_flag[0]:
        raise LayoutUnstable(f"layout ordinal {ordinal} is not a valid native episode")
    native.run_reward()
    if hasattr(native, "get_score"):
        native.get_score()
    if getattr(native, "interact", False) and hasattr(native, "query_support_arm_traj"):
        native.query_support_arm_traj(env_idx=0)
