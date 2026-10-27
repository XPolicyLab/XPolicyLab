"""Supervise one isolated Simate_beta interpreter and local websocket service."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

POLICY_DIR = Path(__file__).resolve().parent


def model_environment():
    env = dict(os.environ)
    for name in ("CODEX_API_KEY", "OPENAI_API_KEY", "TOKENHUB_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(name, None)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def runtime_python(policy_dir=POLICY_DIR):
    settings = Path(policy_dir) / "simate_backend.json"
    installed = json.loads(settings.read_text()) if settings.is_file() else {}
    return Path(os.environ.get("ROBORSI_SIM_MODEL_PYTHON") or installed.get("model_python") or Path(policy_dir) / "runtime/simate-env/bin/python")


def check_simate_runtime(policy_dir=POLICY_DIR, checkpoint_dir=None):
    policy_dir = Path(policy_dir)
    try:
        from .simate_assets import validate_source
    except ImportError:
        from simate_assets import validate_source
    source = validate_source(policy_dir)
    python = runtime_python(policy_dir)
    if not python.is_file():
        raise ValueError(f"Simate Python environment is missing: {python}")
    checkpoint = Path(checkpoint_dir or os.environ.get(
        "ROBORSI_SIM_CHECKPOINT", policy_dir / "checkpoints/simate-beta-robodojo-eval"
    ))
    for name in ("model.safetensors", "tokenizer.model"):
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Simate checkpoint asset is missing: {path}")
    probe = subprocess.run(
        [str(python), "-c", "import json,torch,websockets,safetensors,sentencepiece; print(json.dumps({'torch':torch.__version__,'python':__import__('platform').python_version()}))"],
        check=True, capture_output=True, text=True, env=model_environment(),
    )
    return {
        "policy": "Simate_beta", "python": str(python), "source": source,
        "checkpoint": str(checkpoint.resolve()),
        "runtime": json.loads(probe.stdout),
        "exec_chunk_size": 50, "history_frames": 25, "history_interval": 20,
        "dtype": "float32", "controller": "adaptive_2_50 from the installed authenticated profile; this bridge does not configure controller steps",
    }


class SimateServer:
    def __init__(self, checkpoint_dir, output_dir, gpu_id="0", port=0, policy_dir=POLICY_DIR, *, startup_timeout_s=900, seed=0):
        if not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", str(gpu_id)):
            raise ValueError("Simate requires one explicit policy GPU")
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Invalid Simate local port")
        self.policy_dir, self.checkpoint_dir, self.output_dir = map(Path, (policy_dir, checkpoint_dir, output_dir))
        self.gpu_id, self.port, self.startup_timeout_s = str(gpu_id), port, startup_timeout_s
        self.seed = int(seed)
        self.process = None
        self.streams = []

    def start(self):
        if self.process is not None:
            raise ValueError("Simate service already started")
        check_simate_runtime(self.policy_dir, self.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        env = model_environment()
        env.update(CUDA_VISIBLE_DEVICES=self.gpu_id, PYTHONNOUSERSITE="1")
        # Remove inherited PI/JAX source overrides from the model interpreter.
        env.pop("PYTHONPATH", None)
        command = [str(runtime_python(self.policy_dir)), "-u", str(self.policy_dir / "simate_server.py"),
                   "--checkpoint-dir", str(self.checkpoint_dir.resolve()), "--port", str(self.port), "--seed", str(self.seed)]
        self.streams = [(self.output_dir / name).open("w") for name in ("server.stdout.log", "server.stderr.log")]
        ready_read, ready_write = os.pipe()
        try:
            os.set_blocking(ready_read, False)
            requested = self.port
            try:
                self.process = subprocess.Popen(
                    [*command, "--ready-fd", str(ready_write)], env=env, stdin=subprocess.DEVNULL,
                    stdout=self.streams[0], stderr=self.streams[1], pass_fds=(ready_write,), start_new_session=True,
                )
            finally:
                os.close(ready_write)
            announcement = b""
            deadline = time.monotonic() + self.startup_timeout_s
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("Simate model service exited during load; see server.stderr.log")
                if not self.port or not announcement.endswith(b"\n"):
                    try:
                        chunk = os.read(ready_read, 129)
                    except BlockingIOError:
                        chunk = None
                    if chunk == b"":
                        raise RuntimeError("Simate closed readiness pipe before binding")
                    if chunk:
                        announcement += chunk
                    if len(announcement) > 128:
                        raise RuntimeError("Invalid Simate readiness announcement")
                    if announcement.endswith(b"\n"):
                        bound = json.loads(announcement)["port"]
                        if type(bound) is not int or not 1 <= bound <= 65535 or (requested and requested != bound):
                            raise RuntimeError("Invalid Simate bound port")
                        self.port = bound
                if announcement.endswith(b"\n"):
                    try:
                        with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{self.port}/healthz", timeout=1) as response:
                            if response.status == 200:
                                return self
                    except (OSError, URLError):
                        pass
                time.sleep(0.1)
            raise TimeoutError("Simate service did not become ready before its startup deadline")
        except BaseException:
            self.close()
            raise
        finally:
            os.close(ready_read)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        for stream in self.streams:
            stream.close()
