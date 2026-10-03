"""Dedicated native simulator entrypoint; takes a trusted JSON run request.

Use the release-patched simulator and core process supervision. This worker
does not execute generated Python policies or provide their OS isolation.
"""

import builtins
import importlib
import importlib.abc
import importlib.machinery
import os
import runpy
import sys
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, read_json

from .native_layouts import bind_layouts
from .native_observer import NativeObserver
from .native_initialization import (
    InitializationObserver,
    InitializationFinished,
    ResetOnlyClient,
)


class EnvironmentHook(importlib.abc.MetaPathFinder):
    def __init__(self, observer):
        self.observer = observer

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "src.eval_client.eval_env":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError("Native eval environment is unavailable")
        original, observer = spec.loader, self.observer

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return original.create_module(spec)

            def exec_module(self, module):
                original.exec_module(module)
                factory = module.create_eval_env
                if getattr(observer, "initialization", False):
                    module.WsModelClient = ResetOnlyClient

                def create(*args, **kwargs):
                    if kwargs.get("resume_state") is not None:
                        raise ValueError("Native run must start from a fresh cohort")
                    if getattr(observer, "initialization", False):
                        app = args[1] if len(args) > 1 else kwargs["app"]
                        observer.app = app
                        return observer.attach(factory(*args, **kwargs), app=app)
                    env = factory(*args, **kwargs)
                    return observer.attach(env)

                module.create_eval_env = create

        spec.loader = Loader()
        return spec


def run(request, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "request.json", request)
    num_envs = request.get("num_envs", 1)
    if type(num_envs) is not int or not 1 <= num_envs <= 10:
        raise ValueError("Native worker supports one to ten environments")
    run_id = "physicalrsi-" + os.urandom(16).hex()
    atomic_json(output / "session.json", dict(run_id=run_id))
    sim = Path(request["simulator_root"]).resolve(strict=True)
    assets = Path(request["assets_root"]).resolve(strict=True)
    main = sim / "src/eval_client/main.py"
    source = main.read_text()
    if "ROBODOJO_RESULT_ROOT" not in source or "multi_gpu=False" not in source:
        raise ValueError("Native simulator requires the release client patches")
    if "src.eval_client.eval_env" in sys.modules:
        raise RuntimeError("Native worker requires a fresh process")
    sys.path.insert(0, str(sim))
    globals_module = importlib.import_module("env.global_configs")
    globals_module.ASSETS_PATH = str(assets)
    globals_module.OBJECTS_PATH = str(assets / "Object/RoboDojo")
    globals_module.ROBOTS_PATH = str(assets / "Robots")
    seeds = importlib.import_module("env.seed_manager.seed_manager")
    seeds.SeedManager = bind_layouts(
        seeds.SeedManager,
        task=request["task"],
        split=request["split"],
        root=request["layout_root"],
        rows=request["layouts"],
    )
    mode = request.get("mode", "rollout")
    if mode not in {"rollout", "initialize"}:
        raise ValueError("Unknown native worker mode")
    if mode == "initialize" and len(request["layouts"]) != num_envs:
        raise ValueError("Initialization requires one exact batch without padding")
    calibration = request.get("camera_calibration", False)
    if type(calibration) is not bool or (calibration and mode != "initialize"):
        raise ValueError("Camera diagnostics require reset-only initialization")
    robot_calibration = request.get("robot_calibration", False)
    if type(robot_calibration) is not bool or (
        robot_calibration and mode != "initialize"
    ):
        raise ValueError("Robot diagnostics require reset-only initialization")
    observer = (
        InitializationObserver(
            output,
            len(request["layouts"]),
            camera_calibration=calibration,
            **({"robot_calibration": True} if robot_calibration else {}),
        )
        if mode == "initialize"
        else NativeObserver(output, len(request["layouts"]))
    )
    sys.meta_path.insert(0, EnvironmentHook(observer))
    os.environ.update(
        ROBODOJO_RESULT_ROOT=str(output / "native"),
        ROBODOJO_RUN_ID=run_id,
        EVAL_NUM=str(len(request["layouts"])),
        # Native restart would bypass this entrypoint. Fail and let the
        # controller retain/reconcile the attempt instead of replaying it.
        ROBODOJO_FATAL_RESTART_COUNT="3",
    )

    def reject_exec(event, args):
        if event == "os.exec":
            raise RuntimeError("Native self-restart requires controller reconciliation")

    sys.addaudithook(reject_exec)

    def reject_prompt(prompt=""):
        raise RuntimeError(
            "Unattended native execution requires prior configuration: " + str(prompt)
        )

    # Core process supervision keeps stdin open for its lifecycle handshake.
    # An upstream setup/license prompt must fail promptly, never wait on it.
    builtins.input = reject_prompt
    os.chdir(sim)
    sys.argv = [
        str(main),
        "--task_name",
        request["task"],
        "--env_cfg_type",
        "arx_x5",
        "--num_envs",
        str(request.get("num_envs", 1)),
        "--enable_cameras",
        "--device_id",
        str(request.get("gpu", 0)),
        "--device",
        f"cuda:{request.get('gpu', 0)}",
        "--policy_name",
        request["policy_name"],
        "--port",
        str(request["port"]),
        "--host",
        request.get("host", "127.0.0.1"),
        "--protocol",
        "ws",
        "--policy_server_url",
        request["endpoint"],
        "--additional_info",
        "physicalrsi",
        "--seed",
        str(request.get("seed", 0)),
        "--headless",
    ]
    if mode == "initialize":
        try:
            runpy.run_path(str(main), run_name="__main__")
        except InitializationFinished:
            pass
        finally:
            observer.close()
        atomic_json(
            output / "initialized.json",
            dict(
                state="completed",
                initializations=observer.finish(),
                physical_qualification=False,
            ),
        )
    else:
        runpy.run_path(str(main), run_name="__main__")
        outcomes = observer.finish()
        atomic_json(
            output / "completed.json", dict(state="completed", outcomes=outcomes)
        )


if __name__ == "__main__":
    request_path, output_path = sys.argv[1:]
    output_path = str(Path(output_path).resolve())
    if Path(output_path).exists():
        raise FileExistsError("Native execution output must be new")
    try:
        run(read_json(request_path), output_path)
    except BaseException as error:
        # The parent must additionally check the supervised process exit code.
        if Path(output_path).is_dir():
            atomic_json(Path(output_path) / "failed.json", dict(error=repr(error)))
        raise
