"""Check endpoint and private runtime paths without opening a connection."""

from importlib import import_module
from importlib import util

import shutil
from pathlib import Path

import pytest

runtime_config = import_module("XPolicyLab.policy.SenseNova-RoboRSI.runtime_config")

P = Path(__file__).resolve().parents[1]


def test_direct_preflight_import_can_resolve_adaptive_limits():
    spec = util.spec_from_file_location("_direct_roborsi_runtime_config", P / "runtime_config.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.resolve_adaptive_steps({}) == {
        "approved_steps_min": 5,
        "approved_steps_max": 50,
        "approved_steps_default": 5,
    }


@pytest.fixture
def policy_copy(tmp_path, monkeypatch):
    policy = tmp_path / "shared-policy"
    policy.mkdir()
    for name in (
        "agent.yaml",
        "submission.json",
        "sense_nova_roborsi_factory.py",
        "hybrid_agent.py",
        "hybrid_launcher.py",
        "hybrid_backend.py",
    ):
        shutil.copyfile(P / name, policy / name)
    monkeypatch.setattr(runtime_config, "POLICY_DIR", policy)
    monkeypatch.delenv("ROBORSI_RUNTIME_STATE_DIR", raising=False)
    return policy


def test_base_url_is_required_and_ignores_sdk_environment(monkeypatch):
    monkeypatch.delenv("ROBORSI_CODEX_BASE_URL", raising=False)
    monkeypatch.setenv("CODEX_BASE_URL", "https://ignored-sdk-setting.example/v1")
    with pytest.raises(ValueError, match="ROBORSI_CODEX_BASE_URL"):
        runtime_config.resolve_base_url()
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", "http://127.0.0.1:8080/v1")
    assert runtime_config.resolve_base_url() == "http://127.0.0.1:8080/v1"


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "   ",
        "not-a-url",
        "https://user:private-value@example.com/v1",
        "https://example.com/v1?api_key=private-value",
        "https://example.com/v1#private-value",
        "https://example.com:not-a-port/v1",
        "https://example.com/bad path",
    ],
)
def test_invalid_base_url_fails_without_echoing_value(monkeypatch, endpoint):
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", endpoint)
    with pytest.raises(ValueError, match="ROBORSI_CODEX_BASE_URL") as stopped:
        runtime_config.resolve_base_url()
    assert "private-value" not in str(stopped.value)


@pytest.mark.parametrize("shared", [False, True])
def test_preflight_state_defaults_or_uses_private_override(
    policy_copy, tmp_path, monkeypatch, shared
):
    state = policy_copy / "runtime"
    if shared:
        state = tmp_path / "caller-private"
        monkeypatch.setenv("ROBORSI_RUNTIME_STATE_DIR", str(state))
        policy_copy.chmod(0o555)
    try:
        config = runtime_config.load_agent_config({}, require_key=False)
    finally:
        policy_copy.chmod(0o755)
    agent = config.document["agent"]
    assert agent["working_dir"] == str(state / "preflight/workspace")
    assert agent["codex_cli"]["env"]["CODEX_HOME"] == str(
        state / "preflight/codex-home"
    )
    assert agent["codex_cli"]["executable"] == str(
        policy_copy / "runtime/codex/bin/codex"
    )
    for name in ("workspace", "codex-home"):
        assert (state / "preflight" / name).stat().st_mode & 0o777 == 0o700
    if shared:
        assert not (policy_copy / "runtime").exists()


def test_episode_state_takes_precedence_over_preflight_override(
    policy_copy, tmp_path, monkeypatch
):
    monkeypatch.setenv("ROBORSI_RUNTIME_STATE_DIR", "relative-unused-for-episode")
    episode = tmp_path / "episode"
    config = runtime_config.load_agent_config(
        {}, require_key=False, episode_dir=episode
    )
    assert config.document["agent"]["working_dir"] == str(episode / "workspace")
    assert not (policy_copy / "runtime").exists()


def test_relative_preflight_state_fails_before_writing(policy_copy, monkeypatch):
    monkeypatch.setenv("ROBORSI_RUNTIME_STATE_DIR", "relative-state")
    with pytest.raises(ValueError, match="must be an absolute path"):
        runtime_config.load_agent_config({}, require_key=False)
    assert not (policy_copy / "runtime").exists()


@pytest.mark.parametrize("model", [None, "", "   ", "bad route", "bad\nroute", "bad\x7froute"])
def test_model_is_required_without_echoing_invalid_value(monkeypatch, model):
    if model is None:
        monkeypatch.delenv("ROBORSI_CODEX_MODEL", raising=False)
    else:
        monkeypatch.setenv("ROBORSI_CODEX_MODEL", model)
    monkeypatch.setenv("CODEX_MODEL", "ignored-sdk-model")
    with pytest.raises(ValueError, match="ROBORSI_CODEX_MODEL") as stopped:
        runtime_config.resolve_model()
    assert "bad" not in str(stopped.value)


@pytest.mark.parametrize("name", ["ROBORSI_CODEX_BASE_URL", "ROBORSI_CODEX_MODEL"])
def test_missing_deployment_config_fails_before_creating_state(policy_copy, monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match=name):
        runtime_config.load_agent_config({}, require_key=False)
    assert not (policy_copy / "runtime").exists()


def test_custom_routing_preserves_protocol_budgets_and_trajectory_header(policy_copy, monkeypatch):
    endpoint = "https://provider.example.invalid/custom/v1/"
    route = "model-family/deployment-a/route-b"
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", endpoint)
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", route)
    monkeypatch.setenv("CODEX_API_KEY", "synthetic-offline-secret")
    config = runtime_config.load_agent_config({}, trajectory_id="fixture-trajectory")
    agent = config.document["agent"]
    provider = agent["codex_cli"]["config_overrides"]
    assert agent["model"] == route
    assert agent["planner"] == "codex_cli" and agent["reasoning_effort"] == "max"
    assert agent["max_turns"] is None and agent["max_tokens"] is None
    assert agent["codex_cli"]["tools"] == "external"
    assert provider["model_provider"] == "runtime"
    assert provider["model_providers.runtime.base_url"] == endpoint
    assert provider["model_providers.runtime.wire_api"] == "responses"
    assert provider["model_providers.runtime.env_key"] == "CODEX_API_KEY"
    assert provider["model_providers.runtime.stream_max_retries"] == 8
    assert provider["model_providers.runtime.request_max_retries"] == 8
    assert provider["model_providers.runtime.env_http_headers.X-Trajectory-Id"] == "ROBORSI_TRAJECTORY_ID"
    assert "synthetic-offline-secret" not in repr(config.document)


def test_installation_check_does_not_require_or_reuse_deployment_settings(policy_copy, monkeypatch):
    for key in ("ROBORSI_CODEX_BASE_URL", "ROBORSI_CODEX_MODEL", "CODEX_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    config = runtime_config.load_agent_config({}, require_key=False, installation_only=True)
    assert config.document["agent"]["model"] == "installation-check"
    with pytest.raises(ValueError, match="ROBORSI_CODEX_BASE_URL"):
        runtime_config.load_agent_config({}, require_key=False)
    with pytest.raises(ValueError, match="CODEX_API_KEY"):
        runtime_config.load_agent_config({}, installation_only=True)
    monkeypatch.setenv("CODEX_API_KEY", "synthetic-offline-secret")
    with pytest.raises(ValueError, match="restricted to offline checks"):
        runtime_config.load_agent_config({}, installation_only=True)
