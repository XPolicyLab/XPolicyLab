"""Validate the independent installer and local service lifecycle without a GPU."""

from __future__ import annotations

from importlib import import_module

import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

runtime = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_runtime")
pi05_tokenizer = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_tokenizer")
Pi05CheckpointError = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_assets").Pi05CheckpointError

P = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_state_environment(monkeypatch):
    monkeypatch.delenv("ROBORSI_RUNTIME_STATE_DIR", raising=False)


def test_python_and_bootstrap_pins_are_exact_release_artifacts():
    metadata = runtime._python_metadata(P)
    assert metadata["version"] == "3.11.17"
    assert (
        metadata["sha256"]
        == "c624af93ad62a596806bbd2404e1fb80744a407ca7279854445ede16d93858b8"
    )
    selected = runtime._bootstrap_lock(P / "pi05/requirements.lock")
    assert {line.split("==")[0] for line in selected.splitlines()} == {
        "pip",
        "setuptools",
        "wheel",
        "packaging",
    }
    assert all("--hash=sha256:" in line for line in selected.splitlines())


@pytest.mark.parametrize("member", ["../escape", "/absolute", "other/bin/python3.11"])
def test_python_archive_cannot_write_outside_its_prefix(tmp_path, member):
    archive = tmp_path / "python.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        entry = tarfile.TarInfo(member)
        entry.size = 1
        source.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(Pi05CheckpointError, match="Unsafe"):
        runtime._extract_python(
            archive,
            tmp_path / "installed",
            {"archive_prefix": "python", "executable": "bin/python3.11"},
        )
    assert not (tmp_path / "escape").exists()


def test_explicit_python_archive_is_hashed_before_use(tmp_path):
    archive = tmp_path / "python.tar.gz"
    archive.write_bytes(b"fixture")
    runtime._verify_python_archive(
        archive, {"size": 7, "sha256": hashlib.sha256(b"fixture").hexdigest()}
    )
    with pytest.raises(Pi05CheckpointError, match="SHA-256 mismatch"):
        runtime._verify_python_archive(archive, {"size": 7, "sha256": "0" * 64})


def test_all_original_dependency_overrides_install_byte_exact(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    cached = tmp_path / "cached-original.py"
    cached.write_bytes(b"original cached release fixture")
    (site / "augmax").mkdir()
    (site / "augmax/__init__.py").hardlink_to(cached)
    runtime._apply_overrides(P, site)
    runtime._check_overrides(P, site)
    assert cached.read_bytes() == b"original cached release fixture"
    assert sum(path.is_file() for path in (site / "lerobot").rglob("*")) == 100
    changed = site / "augmax/__init__.py"
    changed.write_bytes(b"modified")
    with pytest.raises(Pi05CheckpointError, match="override differs"):
        runtime._check_overrides(P, site)


def test_vla_environment_has_no_model_or_hub_credentials(tmp_path, monkeypatch):
    for name in ("CODEX_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.setenv(name, "fixture-secret")
    env = runtime._service_env(P, tmp_path / "data", tmp_path / "service", "2")
    assert "fixture-secret" not in repr(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["OPENPI_DATA_HOME"] == str(tmp_path / "data")
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.6"
    assert "vendor/pi05/openpi/src" in env["PYTHONPATH"]
    assert env["PATH"].split(":")[0] == str(P / "runtime/pi05-cuda/bin")
    assert env["XLA_FLAGS"] == f"--xla_gpu_cuda_data_dir={P / 'runtime/pi05-cuda'}"
    assert "unsafe_fallback" not in repr(env)


def test_cuda_compiler_resources_are_verified_and_exposed_in_sdk_layout(
    tmp_path, monkeypatch
):
    source = (
        tmp_path
        / "runtime/pi05-env/lib/python3.11/site-packages/triton/backends/nvidia"
    )
    resources = {
        "bin/ptxas": ("bin/ptxas", b"fixture assembler"),
        "nvvm/libdevice/libdevice.10.bc": ("lib/libdevice.10.bc", b"fixture libdevice"),
    }
    pins = {}
    for destination, (relative, content) in resources.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o755)
        pins[destination] = {
            "source": relative,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    monkeypatch.setattr(runtime, "_CUDA_FILES", pins)
    receipt = runtime._cuda_toolchain(tmp_path, install=True)
    assert runtime._cuda_toolchain(tmp_path) == receipt
    assert set(receipt["files"]) == set(resources)
    assembler = tmp_path / "runtime/pi05-cuda/bin/ptxas"
    assert assembler.resolve() == source / "bin/ptxas"
    assembler.unlink()
    assembler.symlink_to(source / "lib/libdevice.10.bc")
    with pytest.raises(Pi05CheckpointError, match="missing or changed"):
        runtime._cuda_toolchain(tmp_path)
    assembler.unlink()
    runtime._cuda_toolchain(tmp_path, install=True)
    (source / "bin/ptxas").write_bytes(b"corrupt assembler")
    with pytest.raises(Pi05CheckpointError, match="resource differs"):
        runtime._cuda_toolchain(tmp_path)


def test_jax_cache_persists_across_episodes_and_rejects_escape(tmp_path):
    policy = tmp_path / "policy"
    expected = policy / "runtime/pi05-jax-cache"
    assert runtime._jax_cache(policy) == expected
    (expected / "compiled-fixture").write_bytes(b"cache fixture")
    assert runtime._jax_cache(policy) == expected
    assert (expected / "compiled-fixture").read_bytes() == b"cache fixture"
    (expected / "compiled-fixture").unlink()
    expected.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    expected.symlink_to(outside, target_is_directory=True)
    with pytest.raises(Pi05CheckpointError, match="must not be linked"):
        runtime._jax_cache(policy)


def test_shared_state_copies_tokenizer_offline_and_persists_private_jax_cache(
    tmp_path, monkeypatch
):
    policy = tmp_path / "read-only-policy"
    data = policy / "runtime/pi05-data"
    tokenizer = data / "big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    content = b"public-tokenizer-fixture"
    tokenizer.write_bytes(content)
    tokenizer.chmod(0o444)
    metadata = {
        "cache_relative_path": "big_vision/paligemma_tokenizer.model",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr(
        pi05_tokenizer, "load_pi05_tokenizer_metadata", lambda _policy: metadata
    )
    monkeypatch.setattr(
        runtime,
        "validate_pi05_source",
        lambda _policy: {
            "server_package_root": "server-source",
            "openpi_src": "openpi-source",
            "openpi_client_src": "client-source",
        },
    )
    state = tmp_path / "caller-private"
    monkeypatch.setenv("ROBORSI_RUNTIME_STATE_DIR", str(state))
    protected = [policy, policy / "runtime", data, tokenizer.parent]
    for directory in protected:
        directory.chmod(0o555)
    before = tokenizer.stat()
    try:
        first = runtime._service_env(policy, data, state / "episode-one", "0")
        cache = Path(first["JAX_COMPILATION_CACHE_DIR"])
        (cache / "compiled-fixture").write_bytes(b"persistent")
        second = runtime._service_env(policy, data, state / "episode-two", "0")
        assert first["JAX_COMPILATION_CACHE_DIR"] == second["JAX_COMPILATION_CACHE_DIR"]
        assert cache == state / "pi05-jax-cache"
        assert (cache / "compiled-fixture").read_bytes() == b"persistent"
        assert first["OPENPI_DATA_HOME"] == str(state / "pi05-data")
        private_tokenizer = state / "pi05-data" / metadata["cache_relative_path"]
        assert private_tokenizer.read_bytes() == content
        assert private_tokenizer.stat().st_ino != tokenizer.stat().st_ino
        assert tokenizer.read_bytes() == content
        assert tokenizer.stat().st_mode == before.st_mode
        assert tokenizer.stat().st_mtime_ns == before.st_mtime_ns
        assert list((policy / "runtime").iterdir()) == [data]
    finally:
        for directory in protected:
            directory.chmod(0o755)


@pytest.mark.parametrize("linked", ["state-root", "cache"])
def test_private_jax_cache_rejects_links(tmp_path, monkeypatch, linked):
    state = tmp_path / "private-state"
    outside = tmp_path / "outside"
    outside.mkdir()
    if linked == "state-root":
        state.symlink_to(outside, target_is_directory=True)
    else:
        state.mkdir()
        (state / "pi05-jax-cache").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("ROBORSI_RUNTIME_STATE_DIR", str(state))
    with pytest.raises(Pi05CheckpointError, match="must not be linked"):
        runtime._jax_cache(tmp_path / "policy")
    assert not list(outside.iterdir())


def test_relative_runtime_state_fails_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORSI_RUNTIME_STATE_DIR", "relative-state")
    with pytest.raises(Pi05CheckpointError, match="must be an absolute path"):
        runtime._service_env(P, tmp_path / "data", tmp_path / "out", "0")
    assert not (tmp_path / "out").exists()


def test_public_installer_removes_inherited_python_environment(monkeypatch):
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CODEX_API_KEY",
        "HF_TOKEN",
    ):
        monkeypatch.setenv(name, "fixture-value")
    monkeypatch.setenv("HTTPS_PROXY", "http://fixture-proxy.invalid:8080")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/fixture/ca.pem")
    env = runtime._public_install_env()
    assert (
        not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CODEX_API_KEY", "HF_TOKEN"}
        & env.keys()
    )
    assert env["HTTPS_PROXY"] == "http://fixture-proxy.invalid:8080"
    assert env["REQUESTS_CA_BUNDLE"] == "/fixture/ca.pem"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_import_timeout_preserves_partial_logs_and_stops_its_process(
    tmp_path, monkeypatch
):
    original = runtime.subprocess.Popen
    children = []

    def tracked(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(runtime.subprocess, "Popen", tracked)
    monkeypatch.setattr(runtime, "_IMPORT_CHECK_TIMEOUT_S", 0.2)
    program = (
        "import sys,time; print('started',flush=True); "
        "print('diagnostic-marker',file=sys.stderr,flush=True); time.sleep(30)"
    )
    with pytest.raises(Pi05CheckpointError, match="import/version check timed out"):
        runtime._run_import_check(
            sys.executable, program, {"PATH": os.defpath}, tmp_path
        )
    assert children[0].returncode is not None
    assert (tmp_path / "import.stdout.log").read_text() == "started\n"
    error = (tmp_path / "import.stderr.log").read_text()
    assert "diagnostic-marker" in error
    assert "owned process stopped" in error


@pytest.mark.parametrize("ready", [True, False])
def test_server_start_and_timeout_clean_only_the_owned_process(
    tmp_path, monkeypatch, ready
):
    monkeypatch.setattr(runtime, "check_pi05_runtime", lambda *_a, **_k: {})
    monkeypatch.setattr(runtime, "validate_pi05_checkpoint", lambda *_a, **_k: {})
    monkeypatch.setattr(runtime, "validate_pi05_tokenizer", lambda *_a, **_k: {})
    calls = []
    fake = SimpleNamespace(pid=999999, poll=lambda: None, wait=lambda **_kwargs: 0)
    def launch(argv, **kwargs):
        calls.append((argv, kwargs))
        os.write(kwargs["pass_fds"][0], b'{"port":19803}\n')
        return fake

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(
        runtime.os, "killpg", lambda pid, signum: calls.append((pid, signum))
    )

    server = runtime.Pi05Server(
        tmp_path / "checkpoint",
        tmp_path / "data",
        tmp_path / "out",
        port=0,
        startup_timeout_s=0 if not ready else 2,
    )
    monkeypatch.setattr(server, "_ready", lambda: ready)
    if ready:
        assert server.start() is server
        server.close()
    else:
        with pytest.raises(TimeoutError):
            server.start()
    argv, kwargs = calls[0]
    assert server.port == (19803 if ready else 0)
    assert argv[argv.index("--port") + 1] == "0"
    assert argv[argv.index("--max-batch-size") + 1] == "1"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert (
        argv[argv.index("--config-name") + 1]
        == "pi05_base_aloha_full_sim_arx-x5_seed_0"
    )
    assert kwargs["start_new_session"] is True
    assert calls[-1][0] == 999999
    assert server._stdout.closed and server._stderr.closed
