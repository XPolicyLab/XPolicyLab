"""Portable vendor launch settings must not alter controller decisions or budgets."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

POLICY = Path(__file__).resolve().parents[1]
SOURCE = POLICY / "vendor/robodojo_adaptive/source"
PROXY_NAMES = (
    "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
    "https_proxy", "http_proxy", "all_proxy", "no_proxy",
)


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE / "code"))
    for name in ("ROBORSI_CODEX_BASE_URL", "ROBORSI_CODEX_MODEL", *PROXY_NAMES):
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location("test_local_runtime", SOURCE / "local_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["ROBORSI_CODEX_BASE_URL", "ROBORSI_CODEX_MODEL"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_standalone_requires_explicit_deployment(runtime, monkeypatch, name, value):
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", "https://provider.example.invalid/v1/")
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", "fixture-model")
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        runtime.resolve_runtime_config({"model": "ignored-config-model"})


@pytest.mark.parametrize("value", [
    "file:///service", "https://user:secret@example.invalid/v1",
    "https://example.invalid/v1?key=secret", "https://example.invalid/v1#secret",
    "https://example.invalid:invalid/v1", "https://example.invalid/a b",
])
def test_standalone_rejects_invalid_endpoint_without_echo(runtime, monkeypatch, value):
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", value)
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", "fixture-model")
    with pytest.raises(ValueError, match="ROBORSI_CODEX_BASE_URL") as error:
        runtime.resolve_runtime_config({})
    assert value not in str(error.value)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("model", ["two words", "line\nbreak", "control\x7f"])
def test_standalone_rejects_invalid_model(runtime, monkeypatch, model):
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", model)
    with pytest.raises(ValueError, match="ROBORSI_CODEX_MODEL"):
        runtime.resolve_runtime_config({})


def test_standalone_only_changes_deployment_and_preserves_cli_contract(runtime, monkeypatch):
    original = json.loads((SOURCE.parent / "resources/fresh-configs/align_blocks-s0.json").read_text())
    endpoint, model = "https://provider.example.invalid/v1/", "fixture-model/route"
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", endpoint)
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", model)
    cfg = runtime.resolve_runtime_config(original)
    assert cfg["model"] == model and cfg["model_base_url"] == endpoint
    assert original["model"] == ""
    assert original["options"]["approved_steps_min"] == 2
    assert cfg["options"]["approved_steps_min"] == 5
    assert {k: v for k, v in cfg.items() if k not in {"model", "model_base_url", "options"}} == {
        k: v for k, v in original.items() if k not in {"model", "options"}
    }
    assert {k: v for k, v in cfg["options"].items() if k != "approved_steps_min"} == {
        k: v for k, v in original["options"].items() if k != "approved_steps_min"
    }
    base = runtime.ORIGINAL_COMMAND(cfg, "same-session")
    command = runtime.command(cfg, "same-session")
    assert command[:len(base) - 1] == base[:-1]
    assert command[-1] == "-"
    assert command[command.index("--model") + 1] == model
    assert 'model_provider="runtime"' in command
    assert 'model_providers.runtime.name="Runtime"' in command
    assert 'model_providers.runtime.env_key="CODEX_API_KEY"' in command
    assert "model_providers.runtime.base_url=" + json.dumps(endpoint) in command
    assert 'model_providers.runtime.stream_max_retries=8' in command
    assert 'model_providers.runtime.request_max_retries=8' in command


@pytest.mark.parametrize("proxies", [{}, {"HTTPS_PROXY": "http://proxy.example.invalid:8080", "NO_PROXY": ""}, {
    "http_proxy": "http://proxy.example.invalid:8080", "no_proxy": "localhost,.example.invalid",
}])
def test_extracted_sandbox_preserves_explicit_proxy_settings(runtime, tmp_path, proxies):
    # Match the adapter's AST-only function extraction with no resolver globals.
    tree = ast.parse((SOURCE / "local_runtime.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sandbox_command")
    namespace = {
        "Path": Path,
        "os": SimpleNamespace(environ={"CODEX_API_KEY": "test-key", **proxies}),
        "BWRAP": tmp_path / "bwrap",
        "codex": SimpleNamespace(native_binary=lambda _: tmp_path / "bin/codex"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<sandbox>", "exec"), namespace)
    cfg = {"codex_binary": "codex", "agent_python": str(tmp_path / "venv/bin/python"), "training_set": str(tmp_path / "training")}
    argv = namespace["sandbox_command"](cfg, tmp_path / "run", tmp_path / "home", ["/codex", "exec", "-"])
    env = {argv[i + 1]: argv[i + 2] for i, arg in enumerate(argv[:-2]) if arg == "--setenv"}
    assert {k: env[k] for k in PROXY_NAMES if k in env} == proxies
    assert env["CODEX_API_KEY"] == "test-key"
    assert argv[-4:] == ["--", "/codex", "exec", "-"]
    assert "--clearenv" in argv and "--unshare-pid" in argv


def test_all_fresh_templates_have_only_portable_deployment_defaults():
    configs = sorted((SOURCE.parent / "resources/fresh-configs").glob("*.json"))
    assert len(configs) == 42
    for path in configs:
        cfg = json.loads(path.read_text())
        assert cfg["model"] == "", path.name
        assert cfg["options"]["physical_gpu"] == 0, path.name
        assert cfg["reasoning_effort"] == "max", path.name
        assert cfg["options"]["approved_steps_min"] == 2, path.name
        assert cfg["options"]["approved_steps_max"] == 50, path.name
        assert cfg["options"]["approved_steps_default"] == 5, path.name
        assert cfg["options"]["vla_horizon"] == 50, path.name


def test_native_benchmark_source_digest_is_unchanged():
    source = POLICY / "vendor/roborsi_source"
    spec = importlib.util.spec_from_file_location("test_runtime_source", source / "roborsi/benchmarks/runtime_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.benchmark_source_digest(source, "robodojo") == "faf3ee08f1c0f8d760df76fdd5cfabc54cb24de5ce23afd926dcd0ae6227a0b2"


@pytest.mark.parametrize("expected", [None, "fixture-model"])
def test_runtime_preserves_matching_prepared_identity(runtime, monkeypatch, expected):
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", "https://provider.example.invalid/v1")
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", "fixture-model")
    assert runtime.resolve_runtime_config({}, expected_model=expected)["model"] == "fixture-model"


def test_runtime_rejects_model_drift_after_preparation(runtime, monkeypatch):
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", "https://provider.example.invalid/v1")
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", "changed-model")
    with pytest.raises(ValueError, match="prepared model identity"):
        runtime.resolve_runtime_config({"model": "prepared-model"}, expected_model="prepared-model")


@pytest.mark.parametrize("env_model", [None, "", "different-model", "fixture-model"])
def test_legacy_factory_checks_model_identity_before_runtime_creation(tmp_path, env_model):
    source = POLICY / "vendor/roborsi_source/robots/robodojo/agents/icl_pi05/factory.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_case")
    # The typed configuration and binary preflight are isolated; execute the
    # complete factory body up to the first preflight after identity validation.
    allowed = {"instance"}
    class ReachedBinaryPreflight(Exception):
        pass
    def preflight(_):
        raise ReachedBinaryPreflight
    namespace = {
        "Path": Path, "AgentConfig": object, "_ALLOWED": allowed,
        "os": SimpleNamespace(environ={} if env_model is None else {"ROBORSI_CODEX_MODEL": env_model}),
        "CodexCliConfig": SimpleNamespace(from_dict=preflight),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<factory>", "exec"), namespace)
    options = {"planner": "codex_cli", "model": "fixture-model", "reasoning_effort": "max", "planner_timeout_s": 0}
    config = SimpleNamespace(document={"agent": options}, parameters={"instance": "fixture-s0"})
    if env_model == "fixture-model":
        with pytest.raises(ReachedBinaryPreflight):
            namespace["prepare_case"](tmp_path / "case", config=config)
    else:
        with pytest.raises(ValueError, match="ROBORSI_CODEX_MODEL must match"):
            namespace["prepare_case"](tmp_path / "case", config=config)
    assert not (tmp_path / "case").exists()
