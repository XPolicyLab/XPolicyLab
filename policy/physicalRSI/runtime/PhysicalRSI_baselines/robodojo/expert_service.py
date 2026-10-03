"""Owned expert services with pinned code/checkpoint inventories.

Inventory validation establishes local artifact identity, not task competence.
The caller must include model-library dependencies in the supplied code roots.
"""

import math
import time
from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.processes import ManagedProcess
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json

from .process_env import with_project_path


def inventory(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Expert artifact root must be a directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        if path.is_symlink():
            raise ValueError("Expert artifact inventory requires concrete files")
        if path.is_file():
            files[str(path.relative_to(root))] = file_digest(path)
    if not files:
        raise ValueError("Expert artifact inventory cannot be empty")
    return dict(root=str(root), files=files)


def verify_spec(spec):
    if spec.get("schema") != "physicalrsi.robodojo.expert/v1":
        raise ValueError("Unknown expert service specification")
    if spec["expert"]["name"] not in {"pi05", "pi05-sparse-memory"}:
        raise ValueError("Expected a fixed VLA expert")
    if not spec["expert"]["revision"] or not spec["code"]:
        raise ValueError("Expert revision and model code are required")
    for expected in [spec["checkpoint"], *spec["code"]]:
        if inventory(expected["root"]) != expected:
            raise ValueError("Expert artifact inventory changed")
    if Path(spec["deploy"]["model_path"]).resolve() != Path(spec["checkpoint"]["root"]):
        raise ValueError("Model configuration does not select the pinned checkpoint")
    module, sep, factory = spec["entrypoint"].partition(":")
    if not sep or not module or not factory.isidentifier():
        raise ValueError("Expert factory must be installed.module:Model")
    return digest(spec)


def freeze_spec(*, expert, entrypoint, checkpoint_root, code_roots, deploy):
    """Build the content manifest without copying code or model weights."""
    checkpoint = inventory(checkpoint_root)
    config = deepcopy(deploy)
    config["model_path"] = checkpoint["root"]
    for key in ("checkpoint_path", "ckpt_name", "checkpoint_num"):
        if config.get(key) is not None:
            raise ValueError(
                "Use one exact model_path checkpoint, without competing selectors"
            )
    spec = dict(
        schema="physicalrsi.robodojo.expert/v1",
        expert=deepcopy(expert),
        entrypoint=entrypoint,
        checkpoint=checkpoint,
        code=[inventory(root) for root in code_roots],
        deploy=config,
    )
    verify_spec(spec)
    return spec


class ExpertService:
    """One owned model instance per service; no implicit cross-actor state sharing."""

    def __init__(self, spec, *, python, environment, output, startup_timeout_s=600):
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("Expert startup timeout must be positive")
        self.spec = deepcopy(spec)
        self.identity = verify_spec(spec)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        atomic_json(self.output / "spec.json", spec)
        self.timeout = startup_timeout_s
        self.process = ManagedProcess(
            spec["expert"]["name"],
            [
                str(python),
                "-m",
                "PhysicalRSI_baselines.robodojo.expert_worker",
                str(self.output / "spec.json"),
                str(self.output / "ready.json"),
            ],
            env_overrides=dict(
                with_project_path(environment), PYTHONDONTWRITEBYTECODE="1"
            ),
            cwd=self.output,
            log_path=self.output / "service.log",
        )
        self.ready = None

    def _record(self, state, **extra):
        atomic_json(
            self.output / "service.json",
            dict(state=state, pid=self.process.pid, identity=self.identity, **extra),
        )

    def __enter__(self):
        try:
            self.process.start()
            self._record("starting")
            deadline = time.monotonic() + self.timeout
            ready_path = self.output / "ready.json"
            while not ready_path.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "Expert exited before readiness; inspect service.log"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("Expert startup deadline reached")
                time.sleep(0.05)
            self.ready = read_json(ready_path)
            if (
                self.ready.get("identity") != self.identity
                or self.ready.get("pid") != self.process.pid
                or self.process.poll() is not None
            ):
                raise ValueError("Expert readiness does not match its owned process")
            self._record("ready", endpoint=self.ready["endpoint"])
            return self
        except BaseException as error:
            self._record("failed", error=repr(error))
            self.close()
            raise

    def close(self):
        try:
            self.process.stop()
        except BaseException as error:
            self._record("cleanup_failed", error=repr(error))
            raise
        atomic_json(
            self.output / "cleanup.json",
            dict(state="stopped", returncode=self.process.poll()),
        )
        status = self.output / "service.json"
        if status.exists() and read_json(status)["state"] == "ready":
            self._record("stopped")

    def __exit__(self, *args):
        self.close()
