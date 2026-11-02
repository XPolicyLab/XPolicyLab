"""Shared-storage boundaries for direct Python asset preparation entry points."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sys
import tempfile


def require_work_path(path, work, label="Output"):
    path = Path(path).expanduser().resolve()
    if path == work or not path.is_relative_to(work):
        raise ValueError(f"{label} must be inside RAP_WAM_WORK_DIR: {path}")
    return path


def configure_workspace():
    raw = os.environ.get("RAP_WAM_WORK_DIR", "")
    if not raw or not Path(raw).is_absolute():
        raise ValueError("Set RAP_WAM_WORK_DIR to an absolute workspace on shared storage")
    work = Path(raw).resolve()
    if work == Path(work.anchor):
        raise ValueError("RAP_WAM_WORK_DIR must be a dedicated workspace")
    relative_paths = {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "HF_HOME": "cache/hf", "HF_HUB_CACHE": "cache/hf/hub",
        "HUGGINGFACE_HUB_CACHE": "cache/hf/hub", "HF_XET_CACHE": "cache/hf/xet",
        "HF_ASSETS_CACHE": "cache/hf/assets", "TRANSFORMERS_CACHE": "cache/hf/transformers",
        "XDG_CACHE_HOME": "cache/xdg", "XDG_CONFIG_HOME": "config/xdg",
        "XDG_DATA_HOME": "data/xdg", "XDG_STATE_HOME": "state/xdg",
        "TORCH_HOME": "cache/torch", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor", "TRITON_CACHE_DIR": "cache/triton",
        "CUDA_CACHE_PATH": "cache/cuda", "NUMBA_CACHE_DIR": "cache/numba",
        "MPLCONFIGDIR": "config/matplotlib", "PYTHONUSERBASE": "python_user",
    }
    paths = {key: require_work_path(work / value, work, key) for key, value in relative_paths.items()}
    # Validate every path before creating any, then replace inherited home/system caches.
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    os.environ["RAP_WAM_WORK_DIR"] = str(work)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    for key, path in paths.items():
        os.environ[key] = str(path)
    # tempfile may have been imported before this direct entry point was invoked.
    tempfile.tempdir = str(paths["TMPDIR"])
    return work


@contextmanager
def staged_output(output, work):
    """Publish a new directory only on success; remove only this call's staging files."""
    output = require_work_path(output, work)
    if output.exists():
        raise FileExistsError(f"Will not overwrite an existing bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = require_work_path(output.with_name(output.name + ".prepare.lock"), work, "Lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Preparation lock exists: {lock_path}; check for another process before removing a stale lock") from exc
    stage = None
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(f"pid={os.getpid()}\n")
        if output.exists():
            raise FileExistsError(output)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
        yield stage
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        lock_path.unlink(missing_ok=True)
