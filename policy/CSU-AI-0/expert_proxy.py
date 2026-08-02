"""Process-isolated expert bridge used by the CSU-AI-0 ModelTemplate adapter."""

from __future__ import annotations

import atexit
import concurrent.futures
import ctypes
import inspect
import json
import os
import pathlib
import queue
import signal
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Callable

from client_server.ws.model_client import WsModelClient


HERE = pathlib.Path(__file__).resolve().parent
POLICY_ROOT = HERE.parent


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _parent_death_sigterm() -> None:
    """Ask Linux to terminate the expert if the outer policy process dies."""

    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(1, signal.SIGTERM)
    except Exception:
        # Best effort only; normal close() still terminates the owned child.
        pass


class _WsWorker:
    """Own WsModelClient and its asyncio loop on exactly one thread."""

    def __init__(
        self,
        *,
        url: str,
        request_timeout_s: float,
        client_factory: Callable[..., Any] = WsModelClient,
    ):
        self._url = url
        self._request_timeout_s = request_timeout_s
        self._client_factory = client_factory
        self._requests: queue.Queue[
            tuple[str, Any, concurrent.futures.Future[Any]] | None
        ] = queue.Queue()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="csu-ai-0-expert-ws", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        client = None
        try:
            token = uuid.uuid4().hex
            client_kwargs = dict(
                url=self._url,
                evaluation_id=f"csu-internal-{token}",
                trial_id=f"csu-trial-{token}",
                action_case_id=f"csu-case-{token}",
                ws_ping_interval_s=60.0,
                ws_ping_timeout_s=600.0,
            )
            signature = inspect.signature(self._client_factory)
            supports_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if supports_kwargs or "request_timeout_s" in signature.parameters:
                client_kwargs["request_timeout_s"] = self._request_timeout_s
            client = self._client_factory(**client_kwargs)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        while True:
            item = self._requests.get()
            if item is None:
                break
            operation, payload, future = item
            if future.cancelled():
                continue
            try:
                if operation == "prepare_case":
                    if isinstance(payload, dict) and payload.get("action_case_id"):
                        client.action_case_id = str(payload["action_case_id"])
                    result = client.call("prepare_case", obs=payload)
                elif operation == "reset":
                    result = client.call("reset")
                elif operation == "infer":
                    result = client.call("get_action", obs=payload)
                elif operation == "infer_batch":
                    client.call("update_obs_batch", obs=payload)
                    result = client.call("get_action_batch")
                elif operation == "trial_end":
                    result = client.call("trial_end", obs=payload)
                else:
                    raise ValueError(f"unknown proxy operation: {operation}")
                future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)

        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def wait_ready(self, timeout_s: float) -> None:
        if not self._ready.wait(timeout_s):
            raise TimeoutError("timed out creating the internal websocket client")
        if self._startup_error is not None:
            raise RuntimeError("internal websocket client failed") from self._startup_error

    def call(self, operation: str, payload: Any, timeout_s: float) -> Any:
        if not self._thread.is_alive():
            if self._startup_error is not None:
                raise RuntimeError("internal websocket client is unavailable") from self._startup_error
            raise RuntimeError("internal websocket worker has stopped")
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._requests.put((operation, payload, future))
        return future.result(timeout=timeout_s)

    def close(self, timeout_s: float = 10.0) -> None:
        if self._thread.is_alive():
            self._requests.put(None)
            self._thread.join(timeout_s)


class ExpertProcessProxy:
    """Start only the routed expert and forward policy calls over localhost."""

    def __init__(self, route_result: dict[str, Any], model_cfg: dict[str, Any]):
        self.route_result = dict(route_result)
        self.model_cfg = dict(model_cfg)
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None
        self._worker: _WsWorker | None = None
        self._log_stream = None

        self.request_timeout_s = _positive_float(
            os.environ.get(
                "CSU_EXPERT_REQUEST_TIMEOUT_S",
                self.model_cfg.get("expert_request_timeout_s", 900),
            ),
            "expert_request_timeout_s",
        )
        self.start_timeout_s = _positive_float(
            os.environ.get(
                "CSU_EXPERT_START_TIMEOUT_S",
                self.model_cfg.get("expert_start_timeout_s", 1200),
            ),
            "expert_start_timeout_s",
        )
        try:
            self._launch()
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _target_dir(self) -> pathlib.Path:
        raw = str(self.route_result.get("policy_dir") or "").strip()
        target = pathlib.Path(raw).expanduser() if raw else POLICY_ROOT / str(
            self.route_result["policy_name"]
        )
        return target.resolve()

    def _runtime_dir(self) -> pathlib.Path:
        configured = os.environ.get("CSU_RUNTIME_DIR") or self.model_cfg.get(
            "runtime_dir"
        )
        if configured:
            root = pathlib.Path(str(configured)).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            runtime = root / f"{self.route_result['task_name']}-{uuid.uuid4().hex[:12]}"
            runtime.mkdir(parents=True, exist_ok=False)
            return runtime
        return pathlib.Path(tempfile.mkdtemp(prefix="csu-ai-0-"))

    def _expert_environment(self) -> dict[str, str]:
        env = dict(os.environ)
        policy_env = str(self.route_result.get("policy_env") or "").strip()
        conda = env.get("CSU_CONDA_EXE") or shutil.which("conda", path=env.get("PATH"))
        if not conda:
            for candidate in (
                "/root/miniconda3/bin/conda",
                "/root/anaconda3/bin/conda",
                "/opt/conda/bin/conda",
            ):
                if pathlib.Path(candidate).is_file():
                    conda = candidate
                    break
        if conda:
            conda_bin = str(pathlib.Path(conda).resolve().parent)
            env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
            env.setdefault("CONDA_EXE", str(pathlib.Path(conda).resolve()))
        checkpoint = str(self.route_result["checkpoint_path"])
        expert = str(self.route_result["selected_expert"])
        if expert == "xiaomi":
            env["XIAOMI_MODEL_DIR"] = checkpoint
            hf_home = str(self.route_result.get("hf_home") or "").strip()
            if hf_home:
                env["HF_HOME"] = hf_home
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        elif expert == "spatial_forcing":
            if self.runtime_dir is None:
                raise RuntimeError("runtime directory must exist before expert launch")
            env.setdefault("OPENPI_DATA_HOME", str(self.runtime_dir / "openpi-cache"))
            env["HF_HUB_OFFLINE"] = "1"
            env["HF_DATASETS_OFFLINE"] = "1"
        elif expert == "hunyuan":
            env["HY_VLA_CKPT_PATH"] = checkpoint
            if policy_env:
                expanded_policy_env = pathlib.Path(policy_env).expanduser()
                if expanded_policy_env.is_absolute():
                    env.setdefault("HY_VLA_ROOT", str(expanded_policy_env.resolve()))
        else:
            raise RuntimeError(f"unsupported expert: {expert!r}")
        env["CSU_PROXY_CHILD"] = "1"
        return env

    def _launch_command(self, target_dir: pathlib.Path, port: int) -> list[str]:
        script = target_dir / "setup_eval_policy_server.sh"
        if not script.is_file():
            raise FileNotFoundError(f"expert entry point is missing: {script}")

        expert = str(self.route_result["selected_expert"])
        checkpoint_arg = str(self.route_result["checkpoint_name"])
        if expert in {"xiaomi", "spatial_forcing"}:
            # Both resolvers accept an absolute checkpoint directory as ckpt_name.
            # This also works with upstream launchers that do not read our env vars.
            checkpoint_arg = str(self.route_result["checkpoint_path"])

        gpu_id = self.model_cfg.get(
            "gpu_id", self.model_cfg.get("policy_gpu_id", 0)
        )
        policy_env = str(self.route_result["policy_env"])
        return [
            "bash",
            str(script),
            str(self.model_cfg.get("bench_name", "RoboDojo")),
            str(self.route_result["task_name"]),
            checkpoint_arg,
            str(self.model_cfg.get("env_cfg_type", "arx_x5")),
            str(self.route_result["action_type"]),
            str(self.model_cfg.get("seed", 0)),
            str(gpu_id),
            policy_env,
            str(port),
            "127.0.0.1",
        ]

    def _launch(self) -> None:
        target_dir = self._target_dir()
        if not target_dir.is_dir():
            raise FileNotFoundError(f"expert policy directory is missing: {target_dir}")

        runtime_dir = self._runtime_dir()
        route_path = runtime_dir / "route.json"
        route_path.write_text(
            json.dumps(self.route_result, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        log_path = runtime_dir / "expert.log"
        self.runtime_dir = runtime_dir
        self.log_path = log_path
        self._log_stream = log_path.open("ab", buffering=0)

        port = _reserve_local_port()
        command = self._launch_command(target_dir, port)
        self._process = subprocess.Popen(
            command,
            cwd=str(target_dir),
            env=self._expert_environment(),
            stdin=subprocess.DEVNULL,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            preexec_fn=_parent_death_sigterm if os.name == "posix" else None,
        )

        deadline = time.monotonic() + self.start_timeout_s
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"expert server exited with code {return_code}; log={log_path}"
                )
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                    break
            except OSError:
                time.sleep(1.0)
        else:
            raise TimeoutError(
                f"expert server did not listen within {self.start_timeout_s}s; log={log_path}"
            )

        self._worker = _WsWorker(
            url=f"ws://127.0.0.1:{port}", request_timeout_s=self.request_timeout_s
        )
        self._worker.wait_ready(min(30.0, self.start_timeout_s))

    def _call(self, operation: str, payload: Any = None) -> Any:
        if self._closed or self._worker is None:
            raise RuntimeError("expert proxy is closed")
        if self._process is None or self._process.poll() is not None:
            code = None if self._process is None else self._process.returncode
            raise RuntimeError(
                f"expert server is not running (returncode={code}); log={self.log_path}"
            )
        return self._worker.call(operation, payload, self.request_timeout_s)

    def reset(self) -> Any:
        return self._call("reset")

    def get_action(self, obs: Any) -> Any:
        return self._call("infer", obs)

    def get_action_batch(self, obs_list: list[Any]) -> Any:
        return self._call("infer_batch", list(obs_list))

    def prepare_case(self, case_meta: dict[str, Any]) -> Any:
        return self._call("prepare_case", dict(case_meta))

    def trial_end(self, result: dict[str, Any]) -> Any:
        return self._call("trial_end", dict(result))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10.0)
        self._process = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None


__all__ = ["ExpertProcessProxy"]
