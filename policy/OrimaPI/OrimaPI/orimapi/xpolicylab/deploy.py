# coding=utf-8
"""Run XPolicyLab's public server and a separately installed RoboDojo client."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import uuid

DEFAULTS = {
    "root": None,
    "simulator_root": None,
    "simulator_python": None,
    "env_cfg_type": "arx_x5",
    "action_type": "joint",
    "seed": 0,
    "host": "127.0.0.1",
    "port": 0,
    "request_timeout_s": 1800.0,
    "evaluation_timeout_s": 3600.0,
    "shutdown_timeout_s": 10.0,
    "headless": True,
    "simulator_gpu": 0,
    "camera_map": None,
}


def load_settings(value: dict, base: Path) -> dict:
    if not isinstance(value, dict) or value.keys() - DEFAULTS.keys():
        raise ValueError("Invalid or unknown settings in xpolicylab")
    settings = {**DEFAULTS, **value}
    for name in ("root", "simulator_root", "simulator_python"):
        value = settings[name]
        if value is not None:
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"xpolicylab.{name} must be a path or null")
            path = Path(value).expanduser()
            path = path if path.is_absolute() else base / path
            # Resolving a venv's Python symlink loses the selected environment.
            settings[name] = (
                path.absolute() if name == "simulator_python" else path.resolve()
            )
    for name in ("request_timeout_s", "evaluation_timeout_s", "shutdown_timeout_s"):
        value = settings[name]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"xpolicylab.{name} must be positive and finite")
    for name, low, high in (
        ("port", 0, 65535),
        ("seed", 0, 2**31 - 1),
        ("simulator_gpu", 0, 1024),
    ):
        if type(settings[name]) is not int or not low <= settings[name] <= high:
            raise ValueError(f"Invalid xpolicylab.{name}")
    if not isinstance(settings["host"], str) or settings["host"] not in {
        "127.0.0.1",
        "localhost",
    }:
        raise ValueError("This launcher requires a loopback host")
    if settings["action_type"] not in {"joint", "ee"}:
        raise ValueError("xpolicylab.action_type must be joint or ee")
    if type(settings["headless"]) is not bool:
        raise ValueError("xpolicylab.headless must be a boolean")
    if (
        not isinstance(settings["env_cfg_type"], str)
        or not settings["env_cfg_type"].strip()
    ):
        raise ValueError("xpolicylab.env_cfg_type is required")
    cameras = settings["camera_map"]
    if cameras is not None and (
        not isinstance(cameras, dict)
        or not cameras
        or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in cameras.items()
        )
    ):
        raise ValueError(
            "xpolicylab.camera_map must map internal camera names to observed names"
        )
    return settings


def _validate_installation(settings):
    for name in ("root", "simulator_root", "simulator_python"):
        if settings[name] is None:
            raise ValueError(f"Configure xpolicylab.{name} before running evaluation")
    root, simulator = settings["root"], settings["simulator_root"]
    if (
        root.name != "XPolicyLab"
        or not (root / "model_template.py").is_file()
        or not (root / "client_server/ws/model_server.py").is_file()
    ):
        raise ValueError(
            "xpolicylab.root must point to the installed XPolicyLab checkout"
        )
    if (simulator / "XPolicyLab").resolve() != root or not (root / "policy").is_dir():
        raise ValueError(
            "RoboDojo/XPolicyLab must resolve to xpolicylab.root and contain policy/"
        )
    entry = simulator / "src/eval_client/main.py"
    if not entry.is_file():
        raise ValueError("RoboDojo src/eval_client/main.py is missing")
    source = entry.read_text(encoding="utf-8")
    if "ROBODOJO_RUN_ID" not in source or "EVAL_NUM" not in source:
        raise ValueError(
            "Use a RoboDojo client supporting ROBODOJO_RUN_ID and EVAL_NUM"
        )
    if not settings["simulator_python"].is_file() or not os.access(
        settings["simulator_python"], os.X_OK
    ):
        raise ValueError(
            "xpolicylab.simulator_python must be an executable Python interpreter"
        )


def _server_types(root):
    # Official XPolicyLab uses both XPolicyLab.* and top-level client_server.* imports.
    for path in (root.parent, root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        module = importlib.import_module("client_server.ws.model_server")
    except ImportError as error:
        raise RuntimeError(
            "Install the XPolicyLab transport dependencies in the policy Python environment"
        ) from error
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("XPolicyLab was already imported from a different checkout")
    return module.PolicyServer, module.PolicyServerConfig


def _register_policy(settings, output, port, task, *, name=None, config_path=None):
    import yaml

    resuming = name is not None
    name = name or "orimapi_" + uuid.uuid4().hex
    directory = settings["root"] / "policy" / name
    if resuming and directory.is_dir():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        (directory / "__init__.py").write_text("# coding=utf-8\n")
        (directory / "model.py").write_text(
            "# coding=utf-8\nfrom orimapi.xpolicylab.model import Model\n"
        )
        (directory / "deploy.py").write_text(
            "# coding=utf-8\nfrom functools import partial\n"
            "from orimapi.xpolicylab.evaluation import eval_one_episode as _evaluate\n"
            f"eval_one_episode = partial(_evaluate, request_timeout_s={settings['request_timeout_s']!r})\n"
        )
        config = {
            "policy_name": name,
            "protocol": "ws",
            "host": settings["host"],
            "port": port,
            "bench_name": "RoboDojo",
            "task_name": task,
            "ckpt_name": "policy-library",
            "env_cfg_type": settings["env_cfg_type"],
            "action_type": settings["action_type"],
            "seed": settings["seed"],
            "eval_batch": False,
            "request_timeout_s": settings["request_timeout_s"],
            "obs_transform_pipeline": "xspark-v1.0",
            "result_dir": str(output / "standalone-policy"),
            "orimapi_config": str(config_path or output.parent / "config.json"),
            "camera_map": settings["camera_map"],
        }
        (directory / "deploy.yml").write_text(yaml.safe_dump(config, sort_keys=False))
        (output / "deploy.yml").write_text(yaml.safe_dump(config, sort_keys=False))
        return name, directory
    except BaseException:
        shutil.rmtree(directory)
        raise


def archive_native_results(simulator: Path, output: Path, run_id: str) -> None:
    # Official RoboDojo writes benchmark/task/policy/config/seed_info/run_id.
    # Both policy and run ID are unique to this invocation; ignore other runs.
    root = simulator / "eval_result"
    for directory in sorted(root.glob(f"*/*/{run_id}/*/*/{run_id}")):
        if directory.is_dir():
            shutil.copytree(directory, output / "native" / directory.relative_to(root))


def collect_results(output: Path, episodes: int) -> dict:
    rows, artifacts, identities = [], [], set()
    for path in sorted((output / "native").rglob("_result.json")):
        raw = path.read_bytes()
        document = json.loads(raw)
        details = document.get("details")
        if not isinstance(details, dict):
            raise ValueError(f"Invalid native result details: {path}")
        for entry in details.values():
            if not isinstance(entry, dict) or type(entry.get("success")) is not bool:
                raise ValueError("Native result is missing an official success flag")
            score, layout = entry.get("score"), entry.get("layout_id")
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("Native result is missing a finite score in [0, 1]")
            if type(layout) is not int or layout in identities:
                raise ValueError("Native result has a missing or duplicate layout ID")
            identities.add(layout)
            rows.append({**entry, "artifact": str(path.relative_to(output))})
        artifacts.append(
            {
                "path": str(path.relative_to(output)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if len(rows) != episodes:
        raise RuntimeError(
            f"Expected {episodes} official episode results, found {len(rows)}; process exit alone is not a task result"
        )
    return {
        "episodes": rows,
        "episode_count": episodes,
        "success_rate": sum(row["success"] for row in rows) / episodes,
        "mean_score": sum(row["score"] for row in rows) / episodes,
        "artifacts": artifacts,
    }


async def _stop_process(process, timeout):
    if process is None:
        return
    # The simulator owns a process group, including any renderer children.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            await asyncio.wait_for(process.wait(), timeout)
            return
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except asyncio.TimeoutError:
            if sig == signal.SIGKILL:
                raise RuntimeError("Simulator process did not exit after SIGKILL")


async def run_simulator(
    task,
    episodes,
    settings,
    output,
    name,
    port,
    *,
    run_id=None,
    simulator_device=None,
    min_free_bytes=0,
):
    """Own one simulator process group, independently of the policy server."""
    failure = output / "callback-failures.jsonl"
    simulator_failure = output / "simulator-failure.json"
    command = [
        str(settings["simulator_python"]),
        "-u",
        "-m",
        "orimapi.xpolicylab.simulator",
        "src/eval_client/main.py",
        "--task_name",
        task,
        "--env_cfg_type",
        settings["env_cfg_type"],
        "--policy_name",
        name,
        "--protocol",
        "ws",
        "--host",
        settings["host"],
        "--port",
        str(port),
        "--seed",
        str(settings["seed"]),
        "--num_envs",
        "1",
        "--device_id",
        str(settings["simulator_gpu"]),
        "--additional_info",
        "ckpt_name=policy-library,action_type=" + settings["action_type"],
        "--enable_cameras",
    ]
    if settings["headless"]:
        command.append("--headless")
    env = dict(os.environ)
    if simulator_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = simulator_device
        # AppLauncher parses known arguments before registering --device; argparse
        # otherwise treats it as an abbreviation of RoboDojo's --device_id.
        # Expose only the selected GPU and let Kit use its default logical device 0.
        command[command.index("--device_id") + 1] = "0"
    else:
        # RoboDojo's --device_id selects its tensors, but AppLauncher otherwise
        # defaults rendering/physics to GPU 0. Kit overrides avoid the ambiguous
        # early parsing of --device as --device_id and keep all three aligned.
        device = settings["simulator_gpu"]
        command.extend(
            [
                "--kit_args",
                f"--/renderer/activeGpu={device} --/physics/cudaDevice={device} "
                "--/renderer/multiGpu/enabled=false",
            ]
        )
    if env.get("ORIMAPI_KIT_OFFLINE") == "1":
        from .kit_cache import isolated_kit_environment, kit_cache_arguments

        env = isolated_kit_environment(
            env, env.get("ORIMAPI_KIT_HOME") or output / "kit-home"
        )
        # The batch coordinator has already prepared this experience serially.
        # Workers must never unpack extensions into the shared cache concurrently.
        offline_arg = "--/app/extensions/registryEnabled=false " + kit_cache_arguments(
            env
        )
        if "--kit_args" in command:
            command[command.index("--kit_args") + 1] += " " + offline_arg
        else:
            command.extend(["--kit_args", offline_arg])
    env.update(
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        EVAL_NUM=str(episodes),
        ROBODOJO_RUN_ID=run_id or name,
        ROBODOJO_FATAL_RESTART_COUNT="3",
        ORIMAPI_EVALUATION_FAILURE=str(failure),
        ORIMAPI_SIMULATOR_FAILURE=str(simulator_failure),
    )
    paths = [
        str(Path(__file__).resolve().parents[2]),
        str(settings["root"].parent),
        str(settings["root"]),
        str(settings["simulator_root"]),
    ]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    launch = {
        "command": command,
        "cwd": str(settings["simulator_root"]),
        "server": f"ws://{settings['host']}:{port}",
        "run_id": run_id or name,
        "simulator_visible_device": simulator_device,
    }
    from ..evaluation.storage import atomic_json

    atomic_json(output / "launch.json", launch)
    process = None
    try:
        with (output / "simulator.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=settings["simulator_root"],
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            launch.update(pid=process.pid, start_ticks=process_start_ticks(process.pid))
            atomic_json(output / "launch.json", launch)
            try:
                deadline = time.monotonic() + settings["evaluation_timeout_s"]
                while True:
                    if failure.exists():
                        raise RuntimeError(
                            f"RoboDojo evaluation callback failed; see {failure}"
                        )
                    if (
                        min_free_bytes
                        and min(
                            shutil.disk_usage(output).free,
                            shutil.disk_usage(settings["simulator_root"]).free,
                        )
                        < min_free_bytes
                    ):
                        raise RuntimeError(
                            "Disk free space below suite reserve during evaluation"
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    try:
                        code = await asyncio.wait_for(
                            process.wait(), min(1.0, remaining)
                        )
                        break
                    except asyncio.TimeoutError:
                        if time.monotonic() >= deadline:
                            raise
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"RoboDojo evaluation timed out after {settings['evaluation_timeout_s']} seconds"
                ) from None
        if simulator_failure.exists():
            try:
                detail = json.loads(simulator_failure.read_text())["error"]
            except (ValueError, KeyError, TypeError):
                detail = "Invalid simulator failure record"
            raise RuntimeError(
                f"RoboDojo exited with code {code}: {detail}; see {output / 'simulator.log'}"
            )
        if code != 0:
            raise RuntimeError(
                f"RoboDojo exited with code {code}; see {output / 'simulator.log'}"
            )
        if failure.exists():
            raise RuntimeError(f"RoboDojo evaluation callback failed; see {failure}")
        return code
    finally:
        await _stop_process(process, settings["shutdown_timeout_s"])


def process_start_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


async def _run(runtime_factory, task, episodes, settings, output, *, raw_images=False):
    server_class, config_class = _server_types(settings["root"])
    from .model import Model

    model = Model(
        {
            "camera_map": settings["camera_map"],
            "action_type": settings["action_type"],
            "raw_images": raw_images,
        },
        runtime=runtime_factory(),
    )
    registration, server = None, None
    try:
        server = server_class(
            model, config_class(host=settings["host"], port=settings["port"])
        )
        await server.start()
        from urllib.parse import urlsplit

        port = urlsplit(server.url).port
        resume_id = os.environ.get("ORIMAPI_RESUME_RUN_ID")
        name, registration = _register_policy(
            settings, output, port, task, name=resume_id
        )
        model.runtime.trace.write(
            "evaluation_started",
            task=task,
            episodes=episodes,
            server=server.url,
            raw_images=raw_images,
        )
        try:
            await run_simulator(task, episodes, settings, output, name, port)
        finally:
            archive_native_results(settings["simulator_root"], output, name)
        result = collect_results(output, episodes)
        model.runtime.trace.write("official_results", result=result)
        return result
    finally:
        try:
            if server is not None:
                await server.stop()
        finally:
            try:
                model.close()
            finally:
                if registration is not None:
                    shutil.rmtree(registration)


def run_evaluation(
    *,
    runtime_factory,
    task: str,
    episodes: int,
    settings=None,
    output_root=None,
    raw_images: bool = False,
) -> dict:
    if type(raw_images) is not bool:
        raise ValueError("raw_images must be a boolean")
    settings = load_settings(settings or {}, Path.cwd())
    if (
        type(episodes) is not int
        or episodes < 1
        or not isinstance(task, str)
        or not task.strip()
    ):
        raise ValueError("A task and positive episode count are required")
    _validate_installation(settings)
    if output_root is None:
        raise ValueError("An evaluation output_root is required")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "native").mkdir()
    return asyncio.run(
        _run(runtime_factory, task, episodes, settings, output, raw_images=raw_images)
    )
