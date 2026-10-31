"""Install the pinned independent VLA environment and own its local service."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

try:
    from .pi05_assets import (
        CONFIG_NAME,
        Pi05CheckpointError,
        _HTTPSRedirect,
        _sha256,
        install_pi05_checkpoint,
        validate_pi05_checkpoint,
    )
    from .pi05_metadata import (
        METADATA_MANIFEST_SHA256,
        check_runtime_distributions,
        expected_runtime_distributions,
        install_lerobot_metadata,
    )
    from .pi05_source import validate_pi05_source
    from .pi05_tokenizer import install_pi05_tokenizer, validate_pi05_tokenizer
except ImportError:
    from pi05_assets import (
        CONFIG_NAME,
        Pi05CheckpointError,
        _HTTPSRedirect,
        _sha256,
        install_pi05_checkpoint,
        validate_pi05_checkpoint,
    )
    from pi05_metadata import (
        METADATA_MANIFEST_SHA256,
        check_runtime_distributions,
        expected_runtime_distributions,
        install_lerobot_metadata,
    )
    from pi05_source import validate_pi05_source
    from pi05_tokenizer import install_pi05_tokenizer, validate_pi05_tokenizer

POLICY_DIR = Path(__file__).resolve().parent
PYTHON_METADATA_SHA256 = (
    "96236fa9810453fb8c39d5a0af1712fa7dfdbb41a60661548fd8c72609bff0f0"
)
_BOOTSTRAP = {
    "pip": "24.0",
    "setuptools": "79.0.1",
    "wheel": "0.48.0",
    "packaging": "26.3",
}
_IMPORT_CHECK_TIMEOUT_S = 120
# Compiler resources already shipped by the hash-locked Linux Triton 3.3.1 wheel.
# The historical host's system CUDA compiler version is unknown; this is an
# explicit adapter runtime dependency, without changing the frozen Python pins.
_CUDA_FILES = {
    "bin/ptxas": {
        "source": "bin/ptxas",
        "size": 30309984,
        "sha256": "5032aa5edc0a9960fb409df1c14445c385c78678c6342eba5bc436d1e3d3520b",
    },
    "nvvm/libdevice/libdevice.10.bc": {
        "source": "lib/libdevice.10.bc",
        "size": 473728,
        "sha256": "5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee",
    },
}


def _cuda_toolchain(policy_dir, *, install=False):
    """Expose and verify the locked wheel's compiler in the CUDA SDK layout."""
    runtime = Path(policy_dir) / "runtime"
    source = runtime / "pi05-env/lib/python3.11/site-packages/triton/backends/nvidia"
    target = runtime / "pi05-cuda"
    for relative, item in _CUDA_FILES.items():
        original = source / item["source"]
        if (
            original.is_symlink()
            or not original.is_file()
            or original.stat().st_size != item["size"]
            or _sha256(original) != item["sha256"]
        ):
            raise Pi05CheckpointError(
                f"pi0.5 pinned Triton CUDA resource differs: {item['source']}"
            )
        destination = target / relative
        for parent in (target, *destination.relative_to(target).parents):
            candidate = parent if parent == target else target / parent
            if candidate.is_symlink():
                raise Pi05CheckpointError("pi0.5 CUDA toolchain directory is linked")
        if install:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() and not destination.is_symlink():
                destination.symlink_to(original.resolve())
        if not destination.is_symlink() or destination.resolve() != original.resolve():
            raise Pi05CheckpointError(
                "pi0.5 CUDA toolchain is missing or changed; run bash install.sh"
            )
    if not os.access(target / "bin/ptxas", os.X_OK):
        raise Pi05CheckpointError("pi0.5 pinned CUDA assembler is not executable")
    return {
        "root": str(target.resolve()),
        "source_distribution": "triton==3.3.1",
        "ptxas_version": "12.4.99",
        "files": {name: item["sha256"] for name, item in _CUDA_FILES.items()},
    }


def _python_metadata(policy_dir):
    path = policy_dir / "pi05/python.json"
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256(path) != PYTHON_METADATA_SHA256
    ):
        raise Pi05CheckpointError("Committed pi0.5 Python metadata changed")
    metadata = json.loads(path.read_text())
    if (
        metadata.get("version") != "3.11.17"
        or metadata.get("archive_prefix") != "python"
        or metadata.get("executable") != "bin/python3.11"
    ):
        raise Pi05CheckpointError("Unsupported pi0.5 Python identity")
    return metadata


def _download_python(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".python-download-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            try:
                with build_opener(_HTTPSRedirect()).open(
                    Request(metadata["url"]), timeout=60
                ) as response:
                    remaining = metadata["size"]
                    while remaining:
                        block = response.read(min(4 * 1024 * 1024, remaining))
                        if not block:
                            raise Pi05CheckpointError("Truncated pi0.5 Python archive")
                        stream.write(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise Pi05CheckpointError(
                            "pi0.5 Python archive exceeds pinned size"
                        )
            except (HTTPError, URLError, OSError) as error:
                raise Pi05CheckpointError(
                    f"pi0.5 Python download failed ({type(error).__name__})"
                ) from None
            stream.flush()
            _verify_python_archive(temporary, metadata)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _verify_python_archive(path, metadata):
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != metadata["size"]
        or _sha256(path) != metadata["sha256"]
    ):
        raise Pi05CheckpointError("pi0.5 Python archive size or SHA-256 mismatch")


def _extract_python(archive, destination, metadata):
    if destination.exists() or destination.is_symlink():
        raise Pi05CheckpointError(
            "pi0.5 Python destination already exists without a verified installation"
        )
    with tempfile.TemporaryDirectory(
        prefix=".pi05-python-", dir=destination.parent
    ) as temporary:
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            for member in members:
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != metadata["archive_prefix"]
                ):
                    raise Pi05CheckpointError("Unsafe path in pi0.5 Python archive")
                if not (
                    member.isfile()
                    or member.isdir()
                    or member.issym()
                    or member.islnk()
                ):
                    raise Pi05CheckpointError("Unsupported pi0.5 Python archive member")
            source.extractall(temporary, filter="data")
        installed = Path(temporary) / metadata["archive_prefix"]
        if not (installed / metadata["executable"]).is_file():
            raise Pi05CheckpointError("pi0.5 Python archive has no pinned interpreter")
        (installed / ".roborsi-python.json").write_text(
            json.dumps(
                {
                    "archive_sha256": metadata["sha256"],
                    "executable_sha256": _sha256(installed / metadata["executable"]),
                }
            )
        )
        installed.rename(destination)


def _run(argv, *, env=None, cwd=None):
    subprocess.run([str(arg) for arg in argv], env=env, cwd=cwd, check=True)


def _public_install_env():
    # Retain configured proxy/CA/package-index transport, never model credentials.
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    for name in tuple(environment):
        upper = name.upper()
        if upper in {
            "CODEX_API_KEY",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
        } or upper.startswith(("CODEX_", "ROBORSI_BENCHMARK_", "RPENT_BENCHMARK_")):
            environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _uv(policy_dir):
    for path in (policy_dir / "runtime/uv/bin/uv", policy_dir / ".venv/bin/uv"):
        if path.is_file():
            return str(path)
    executable = shutil.which("uv")
    if executable is None:
        raise Pi05CheckpointError("uv is required by the policy installer")
    return executable


def _bootstrap_lock(lock, build_lock=None):
    build_lock = (
        lock.with_name("build-requirements.lock") if build_lock is None else build_lock
    )
    selected = {}
    for source in (lock, build_lock):
        if source.is_symlink() or not source.is_file():
            raise Pi05CheckpointError("pi0.5 bootstrap hash lock is missing or linked")
        for line in source.read_text().replace("\\\n", " ").splitlines():
            match = re.match(r"([A-Za-z0-9_-]+)==([^\s]+)", line)
            if match and match[1] in _BOOTSTRAP:
                if (
                    match[1] in selected
                    or match[2] != _BOOTSTRAP[match[1]]
                    or "--hash=sha256:" not in line
                ):
                    raise Pi05CheckpointError(
                        "pi0.5 bootstrap versions or artifact hashes changed"
                    )
                selected[match[1]] = line
    if set(selected) != set(_BOOTSTRAP):
        raise Pi05CheckpointError("pi0.5 hash lock is missing bootstrap packages")
    return "\n".join(selected[name] for name in sorted(selected)) + "\n"


def _apply_overrides(policy_dir, site):
    identity = validate_pi05_source(policy_dir)
    metadata = json.loads((policy_dir / "vendor/pi05-source.json").read_text())
    source = Path(identity["source_overrides"])
    lerobot = site / "lerobot"
    if lerobot.is_symlink():
        raise Pi05CheckpointError("pi0.5 LeRobot destination must not be linked")
    if lerobot.exists():
        # This is the installer's dedicated environment; retain replaced source
        # in its runtime directory rather than merging a different LeRobot tree.
        backup = Path(
            tempfile.mkdtemp(prefix="pi05-lerobot-backup-", dir=policy_dir / "runtime")
        )
        lerobot.rename(backup / "lerobot")
    shutil.copytree(source / "lerobot", lerobot)
    for relative in metadata["source_overrides"]["replace_regular_files"]:
        target = site / relative
        if target.is_symlink() or any(
            parent.is_symlink()
            for parent in target.parents
            if parent != site and parent.is_relative_to(site)
        ):
            raise Pi05CheckpointError("pi0.5 source override destination is linked")
        target.parent.mkdir(parents=True, exist_ok=True)
        # uv may hardlink an installed wheel to its cache. Replace the directory
        # entry so the historical source override cannot modify that cache.
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".pi05-source-", delete=False
        ) as stream:
            staging = Path(stream.name)
        try:
            shutil.copy2(source / relative, staging)
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
    _check_overrides(policy_dir, site)


def _check_overrides(policy_dir, site):
    metadata = json.loads((policy_dir / "vendor/pi05-source.json").read_text())
    records = {
        name.removeprefix("site-packages/"): record
        for name, record in metadata["files"].items()
        if name.startswith("site-packages/")
    }
    expected_lerobot = {name for name in records if name.startswith("lerobot/")}
    actual = {
        path.relative_to(site).as_posix()
        for path in (site / "lerobot").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual != expected_lerobot:
        raise Pi05CheckpointError("pi0.5 LeRobot source tree differs")
    for name, record in records.items():
        path = site / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["size"]
            or _sha256(path) != record["sha256"]
        ):
            raise Pi05CheckpointError(
                f"pi0.5 installed source override differs: {name}"
            )


def resolve_pi05_checkpoint(policy_dir=POLICY_DIR):
    import yaml

    workspace = str(Path(policy_dir).resolve().parents[2])
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

    cfg = yaml.safe_load((policy_dir / "deploy.yml").read_text())
    return resolve_checkpoint_root(
        cfg, policy_dir / "checkpoints", policy_dir=policy_dir, must_exist=False
    )


def _runtime_state_root(policy_dir):
    """Keep shared installations read-only when caller-private state is selected."""
    override = os.environ.get("ROBORSI_RUNTIME_STATE_DIR", "")
    if not override:
        return Path(policy_dir) / "runtime"
    selected = Path(override).expanduser()
    if not selected.is_absolute():
        raise Pi05CheckpointError("ROBORSI_RUNTIME_STATE_DIR must be an absolute path")
    return selected


def _jax_cache(policy_dir):
    """Keep compilation artifacts across episodes within the selected state root."""
    state_root = _runtime_state_root(policy_dir)
    cache = state_root / "pi05-jax-cache"
    if state_root.is_symlink() or cache.is_symlink():
        raise Pi05CheckpointError("pi0.5 persistent JAX cache must not be linked")
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    if cache.is_symlink() or cache.resolve().parent != state_root.resolve():
        raise Pi05CheckpointError("pi0.5 JAX cache escapes its runtime state root")
    return cache.resolve()


def _service_env(policy_dir, data_home, output_dir, gpu_id):
    paths = validate_pi05_source(policy_dir)
    state_root = _runtime_state_root(policy_dir)
    if state_root != Path(policy_dir) / "runtime":
        # OpenPI chmods its cache root even for a cache hit. Copy only the
        # verified public tokenizer into caller-private state, without downloads.
        shared_tokenizer = validate_pi05_tokenizer(data_home, policy_dir=policy_dir)
        data_home = state_root / "pi05-data"
        install_pi05_tokenizer(
            data_home,
            cache_file=Path(shared_tokenizer["path"]),
            offline=True,
            policy_dir=policy_dir,
        )
    venv = policy_dir / "runtime/pi05-env"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "home").mkdir(exist_ok=True)
    libraries = sorted((venv / "lib/python3.11/site-packages/nvidia").glob("*/lib"))
    cuda = policy_dir / "runtime/pi05-cuda"
    return {
        "PATH": f"{cuda / 'bin'}:{venv / 'bin'}:/usr/local/cuda/bin:/usr/bin:/bin",
        "HOME": str(output_dir / "home"),
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(
            (
                paths["server_package_root"],
                paths["openpi_src"],
                paths["openpi_client_src"],
            )
        ),
        "HF_HUB_OFFLINE": "1",
        "OPENPI_DATA_HOME": str(Path(data_home).resolve()),
        "JAX_COMPILATION_CACHE_DIR": str(_jax_cache(policy_dir)),
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.6",
        "XLA_FLAGS": f"--xla_gpu_cuda_data_dir={cuda}",
        "LD_LIBRARY_PATH": ":".join(
            [*map(str, libraries), "/usr/local/cuda/lib64", "/usr/local/nvidia/lib64"]
        ),
        "NO_PROXY": "localhost,127.0.0.1",
        "no_proxy": "localhost,127.0.0.1",
    }


def _run_import_check(python, program, environment, output_dir):
    """Bound imports and version checks, retaining logs and owning their group."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [str(python), "-c", program],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=_IMPORT_CHECK_TIMEOUT_S)
    except subprocess.TimeoutExpired as error:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired as cleanup_error:
            stdout = cleanup_error.stdout or error.stdout or ""
            stderr = cleanup_error.stderr or error.stderr or ""

        def decoded(value):
            return (
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else value
            )

        (output_dir / "import.stdout.log").write_text(decoded(stdout))
        (output_dir / "import.stderr.log").write_text(
            decoded(stderr)
            + f"\npi0.5 import/version check exceeded {_IMPORT_CHECK_TIMEOUT_S}s; "
            + (
                "owned process cleanup incomplete"
                if process.poll() is None
                else "owned process stopped"
            )
            + "\n"
        )
        raise Pi05CheckpointError(
            f"pi0.5 import/version check timed out after {_IMPORT_CHECK_TIMEOUT_S}s; "
            f"see {output_dir / 'import.stderr.log'}"
        ) from None
    (output_dir / "import.stdout.log").write_text(stdout)
    (output_dir / "import.stderr.log").write_text(stderr)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def check_pi05_runtime(
    policy_dir=POLICY_DIR, checkpoint_dir=None, *, check_assets=True
):
    policy_dir = Path(policy_dir).resolve()
    validate_pi05_source(policy_dir)
    metadata = _python_metadata(policy_dir)
    runtime = policy_dir / "runtime"
    receipt_path = runtime / "pi05-installed.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise Pi05CheckpointError("pi0.5 environment is missing; run bash install.sh")
    receipt = json.loads(receipt_path.read_text())
    lock = policy_dir / "pi05/requirements.lock"
    python = runtime / "pi05-env/bin/python"
    base = runtime / "pi05-python" / metadata["executable"]
    if (
        receipt.get("requirements_sha256") != _sha256(lock)
        or receipt.get("build_requirements_sha256")
        != _sha256(policy_dir / "pi05/build-requirements.lock")
        or receipt.get("lerobot_metadata_manifest_sha256") != METADATA_MANIFEST_SHA256
        or receipt.get("python_archive_sha256") != metadata["sha256"]
        or _sha256(base) != receipt.get("python_executable_sha256")
    ):
        raise Pi05CheckpointError("pi0.5 installed environment identity changed")
    site = runtime / "pi05-env/lib/python3.11/site-packages"
    _check_overrides(policy_dir, site)
    distributions = check_runtime_distributions(policy_dir, site)
    if receipt.get("runtime_distributions") != distributions:
        raise Pi05CheckpointError("pi0.5 installed 157-package distribution receipt changed")
    toolchain = _cuda_toolchain(policy_dir)
    check_dir = _runtime_state_root(policy_dir) / "pi05-check"
    env = _service_env(policy_dir, runtime / "pi05-data", check_dir, "0")
    required = expected_runtime_distributions(policy_dir)
    program = (
        "import sys,json,re,importlib.metadata as md,jax,flax,orbax.checkpoint,numpy,torch; from openpi.training import config; from server.openpi_batch_server import create_policy; required="
        + repr(required)
        + "; dists=list(md.distributions(path=[sys.prefix+'/lib/python3.11/site-packages'])); actual={re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in dists}; assert len(dists)==len(actual)==len(required)==157 and actual==required,(actual,required); c=config.get_config('"
        + CONFIG_NAME
        + "'); assert sys.version_info[:3]==(3,11,17); assert c.model.pi05; print(json.dumps({'python':sys.version.split()[0],'jax':jax.__version__,'torch':torch.__version__,'pinned_versions_checked':len(required),'service_imports':True}))"
    )
    result = _run_import_check(python, program, env, check_dir)
    if result.returncode:
        raise Pi05CheckpointError(
            f"pi0.5 source imports or pinned versions failed; see {check_dir / 'import.stderr.log'}"
        )
    imports = json.loads(result.stdout.strip().splitlines()[-1])
    if check_assets:
        checkpoint = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else resolve_pi05_checkpoint(policy_dir)
        )
        validate_pi05_checkpoint(checkpoint, policy_dir=policy_dir)
        validate_pi05_tokenizer(runtime / "pi05-data", policy_dir=policy_dir)
    return {
        **receipt,
        "cuda_toolchain": toolchain,
        "imports": imports,
        "model_load_verified": False,
        "inference_verified": False,
        "dependency_metadata": "Historical exact pins contain upstream metadata conflicts; dependency resolver/pip check closure is not claimed.",
    }


def install_pi05_runtime(
    policy_dir=POLICY_DIR,
    checkpoint_dir=None,
    python_archive=None,
    checkpoint_cache=None,
):
    policy_dir = Path(policy_dir).resolve()
    validate_pi05_source(policy_dir)
    metadata = _python_metadata(policy_dir)
    runtime = policy_dir / "runtime"
    runtime.mkdir(exist_ok=True)
    base = runtime / "pi05-python"
    if not base.exists():
        archive = (
            Path(python_archive)
            if python_archive is not None
            else runtime / "assets/cpython-3.11.17.tar.gz"
        )
        if python_archive is None and not archive.exists():
            _download_python(archive, metadata)
        _verify_python_archive(archive, metadata)
        _extract_python(archive, base, metadata)
    elif base.is_symlink():
        raise Pi05CheckpointError("pi0.5 Python installation must not be linked")
    base_python = base / metadata["executable"]
    marker = base / ".roborsi-python.json"
    if not marker.is_file() or marker.is_symlink():
        raise Pi05CheckpointError(
            "pi0.5 Python has no verified archive installation receipt"
        )
    base_identity = json.loads(marker.read_text())
    if base_identity.get("archive_sha256") != metadata["sha256"] or base_identity.get(
        "executable_sha256"
    ) != _sha256(base_python):
        raise Pi05CheckpointError(
            "pi0.5 Python differs from its verified archive installation"
        )
    version = subprocess.run(
        [
            str(base_python),
            "-I",
            "-c",
            "import sys; print('.'.join(map(str,sys.version_info[:3])))",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != metadata["version"]:
        raise Pi05CheckpointError("Installed pi0.5 Python version differs")
    env = _public_install_env()
    venv = runtime / "pi05-env"
    if venv.is_symlink():
        raise Pi05CheckpointError("pi0.5 dedicated environment must not be linked")
    uv = _uv(policy_dir)
    if not (venv / "bin/python").is_file():
        _run([uv, "venv", "--no-config", "--python", base_python, venv], env=env)
    if (venv / "bin/python").resolve() != base_python.resolve():
        raise Pi05CheckpointError(
            "pi0.5 environment must use the pinned standalone Python"
        )
    lock = policy_dir / "pi05/requirements.lock"
    expected_runtime_distributions(policy_dir)
    bootstrap = runtime / "pi05-bootstrap.lock"
    bootstrap.write_text(_bootstrap_lock(lock))
    flags = [
        uv,
        "pip",
        "install",
        "--no-config",
        "--python",
        venv / "bin/python",
        "--no-deps",
        "--require-hashes",
        "--no-build-isolation",
    ]
    _run([*flags, "-r", bootstrap], env=env)
    _run([*flags, "-r", lock], env=env)
    site = venv / "lib/python3.11/site-packages"
    _apply_overrides(policy_dir, site)
    install_lerobot_metadata(policy_dir, site)
    # wheel is needed only while building the locked source distributions.
    _run(
        [
            uv,
            "pip",
            "uninstall",
            "--no-config",
            "--python",
            venv / "bin/python",
            "wheel",
        ],
        env=env,
    )
    distributions = check_runtime_distributions(policy_dir, site)
    toolchain = _cuda_toolchain(policy_dir, install=True)
    checkpoint = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else resolve_pi05_checkpoint(policy_dir)
    )
    install_pi05_checkpoint(
        checkpoint, cache_dir=checkpoint_cache, policy_dir=policy_dir
    )
    install_pi05_tokenizer(runtime / "pi05-data", policy_dir=policy_dir)
    receipt = {
        "schema_version": 1,
        "python_version": version,
        "python_archive_sha256": metadata["sha256"],
        "python_executable_sha256": _sha256(base_python),
        "requirements_sha256": _sha256(lock),
        "build_requirements_sha256": _sha256(
            policy_dir / "pi05/build-requirements.lock"
        ),
        "lerobot_metadata_manifest_sha256": METADATA_MANIFEST_SHA256,
        "runtime_distributions": distributions,
        "source_overrides": 112,
        "cuda_toolchain": toolchain,
        "checkpoint": str(checkpoint.resolve()),
        "dependency_install": "exact release artifacts; no automatic dependency upgrades; original source overrides applied",
    }
    (runtime / "pi05-installed.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return check_pi05_runtime(policy_dir, checkpoint, check_assets=False)


class Pi05Server:
    """Own one original, localhost-only pi0.5 service and its process group."""

    def __init__(
        self,
        checkpoint_dir,
        data_home,
        output_dir,
        gpu_id="0",
        port=0,
        policy_dir=POLICY_DIR,
        *,
        startup_timeout_s=600,
    ):
        self.policy_dir = Path(policy_dir).resolve()
        self.checkpoint_dir, self.data_home, self.output_dir = (
            Path(checkpoint_dir),
            Path(data_home),
            Path(output_dir),
        )
        if not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", str(gpu_id)):
            raise ValueError("pi0.5 requires one explicit policy GPU")
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Invalid pi0.5 local port")
        self.gpu_id, self.port, self.startup_timeout_s = (
            str(gpu_id),
            port,
            startup_timeout_s,
        )
        self.process = None
        self._stdout = self._stderr = None

    def _ready(self):
        try:
            with build_opener(ProxyHandler({})).open(
                f"http://127.0.0.1:{self.port}/healthz", timeout=1
            ) as reply:
                return reply.status == 200 and len(reply.read(1024)) < 1024
        except (OSError, URLError):
            return False

    def start(self):
        if self.process is not None:
            raise ValueError("pi0.5 service already started")
        stderr_path = self.output_dir.resolve() / "server.stderr.log"
        check_pi05_runtime(self.policy_dir, self.checkpoint_dir, check_assets=False)
        validate_pi05_checkpoint(self.checkpoint_dir, policy_dir=self.policy_dir)
        validate_pi05_tokenizer(self.data_home, policy_dir=self.policy_dir)
        env = _service_env(
            self.policy_dir, self.data_home, self.output_dir, self.gpu_id
        )
        python = self.policy_dir / "runtime/pi05-env/bin/python"
        command = [
            str(python),
            "-u",
            str(self.policy_dir / "pi05_server.py"),
            "--config-name",
            CONFIG_NAME,
            "--checkpoint-dir",
            str(self.checkpoint_dir.resolve()),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--max-batch-size",
            "1",
            "--batch-wait-ms",
            "10",
        ]
        self._stdout = (self.output_dir / "server.stdout.log").open("w")
        self._stderr = (self.output_dir / "server.stderr.log").open("w")
        ready_read, ready_write = os.pipe()
        try:
            os.set_blocking(ready_read, False)
            requested_port = self.port
            try:
                self.process = subprocess.Popen(
                    [*command, "--ready-fd", str(ready_write)],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=self._stdout,
                    stderr=self._stderr,
                    start_new_session=True,
                    pass_fds=(ready_write,),
                )
            finally:
                os.close(ready_write)
            announcement = b""
            bound = False
            deadline = time.monotonic() + self.startup_timeout_s
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"pi0.5 model service exited during load (exit {self.process.returncode}); "
                        f"see {stderr_path}"
                    )
                if not bound:
                    try:
                        chunk = os.read(ready_read, 129)
                    except BlockingIOError:
                        chunk = None
                    if chunk == b"":
                        raise RuntimeError(
                            "pi0.5 service closed its readiness pipe before binding; "
                            f"see {stderr_path}"
                        )
                    if chunk:
                        announcement += chunk
                    if len(announcement) > 128:
                        raise RuntimeError(
                            f"Invalid pi0.5 readiness announcement; see {stderr_path}"
                        )
                    if b"\n" in announcement:
                        try:
                            port = json.loads(announcement)["port"]
                        except (ValueError, KeyError, TypeError):
                            raise RuntimeError(
                                f"Invalid pi0.5 readiness announcement; see {stderr_path}"
                            ) from None
                        if (
                            type(port) is not int
                            or not 1 <= port <= 65535
                            or (requested_port and port != requested_port)
                        ):
                            raise RuntimeError(f"Invalid pi0.5 bound port; see {stderr_path}")
                        self.port, bound = port, True
                if bound and self._ready():
                    return self
                time.sleep(0.1)
            raise TimeoutError(
                "pi0.5 model service did not become ready before its startup deadline; "
                f"see {stderr_path}"
            )
        except BaseException:
            self.close()
            raise
        finally:
            os.close(ready_read)

    def close(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        for stream in (self._stdout, self._stderr):
            if stream is not None:
                stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("install", "check"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--python-archive", type=Path)
    parser.add_argument("--checkpoint-cache", type=Path)
    args = parser.parse_args()
    try:
        receipt = (
            install_pi05_runtime(
                checkpoint_dir=args.checkpoint_dir,
                python_archive=args.python_archive,
                checkpoint_cache=args.checkpoint_cache,
            )
            if args.mode == "install"
            else check_pi05_runtime(checkpoint_dir=args.checkpoint_dir)
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"pi0.5 runtime failed ({type(error).__name__})", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
