"""Check endpoint and private runtime paths without opening a connection."""

from importlib import import_module

import shutil
from pathlib import Path

import pytest

runtime_config = import_module("XPolicyLab.policy.SenseNova-RoboRSI.runtime_config")

P = Path(__file__).resolve().parents[1]


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


def test_base_url_default_ignores_sdk_environment(monkeypatch):
    monkeypatch.delenv("ROBORSI_CODEX_BASE_URL", raising=False)
    monkeypatch.setenv("CODEX_BASE_URL", "https://ignored-sdk-setting.example/v1")
    assert runtime_config.resolve_base_url() == "https://tokenhub.sensetime.com/v1"
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
