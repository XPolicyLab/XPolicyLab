"""Configuration/privacy checks shared by the formal hybrid Model boundary.

The original proposal/review lifecycle is exercised in test_model_hybrid.py.
These checks do not execute a remote model or native task.
"""

from __future__ import annotations

from importlib import import_module

import os

import numpy as np
import pytest
from roborsi.planner.codex_cli_config import CodexCliConfig

model_module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model")
runtime_config = import_module("XPolicyLab.policy.SenseNova-RoboRSI.runtime_config")
adapter_factory = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_model_hybrid").adapter_factory
observation = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_model_hybrid").observation
ToolRuntimeError = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tool_runtime").ToolRuntimeError


def test_missing_independent_tools_fails_before_vla_or_icl(monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "synthetic-unused-offline-key")

    def missing(_policy):
        raise ToolRuntimeError("Independent original tool runtime is missing")

    def must_not_start(*_args, **_kwargs):
        pytest.fail("Missing tools must fail before VLA startup or materialization")

    monkeypatch.setattr(model_module, "resolve_tool_python", missing)
    monkeypatch.setattr(model_module.Model, "_ensure_vla", must_not_start)
    monkeypatch.setattr(model_module, "prepare_icl_context", must_not_start)
    with pytest.raises(ToolRuntimeError, match="Independent original tool"):
        model_module.Model(
            {
                "bench_name": "RoboDojo",
                "env_cfg_type": "arx_x5",
                "action_type": "joint",
                "eval_batch": False,
            }
        )


def test_runtime_endpoint_is_explicit_cli_provider_configuration(
    adapter_factory, monkeypatch
):
    endpoint = "https://runtime-provider.example/v1"
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", endpoint)
    monkeypatch.setenv("CODEX_BASE_URL", "https://ignored-sdk-setting.example/v1")
    model = adapter_factory().model
    agent = model._agent_config.document["agent"]
    provider = agent["codex_cli"]["config_overrides"]
    assert provider["model_providers.runtime.base_url"] == endpoint
    assert provider["model_providers.runtime.wire_api"] == "responses"
    assert provider["model_providers.runtime.env_key"] == "CODEX_API_KEY"
    assert agent["codex_cli"]["expected_version"] == "codex-cli 0.160.1"
    assert agent["codex_cli"]["tools"] == "external"
    monkeypatch.delenv("ROBORSI_CODEX_BASE_URL")
    with pytest.raises(ValueError, match="ROBORSI_CODEX_BASE_URL"):
        runtime_config.load_agent_config(model.model_cfg)


def test_episode_id_is_isolated_without_mutating_parent_environment(
    adapter_factory, monkeypatch
):
    monkeypatch.setenv("ROBORSI_TRAJECTORY_ID", "unrelated-parent-episode")
    model = adapter_factory().model
    identities = []
    for _ in range(2):
        model.reset()
        identities.append(model._episode_id)
        cli = CodexCliConfig.from_dict(
            model._agent_config.document["agent"]["codex_cli"]
        )
        assert (
            cli.config_overrides[
                "model_providers.runtime.env_http_headers.X-Trajectory-Id"
            ]
            == "ROBORSI_TRAJECTORY_ID"
        )
        assert cli.subprocess_env()["ROBORSI_TRAJECTORY_ID"] == model._episode_id
        assert os.environ["ROBORSI_TRAJECTORY_ID"] == "unrelated-parent-episode"
    assert identities[0] != identities[1]


def test_multiple_environments_and_undecoded_or_non_rgb_images_are_rejected(
    adapter_factory,
):
    model = adapter_factory().model
    model.reset()
    with pytest.raises(ValueError, match="one environment"):
        model.update_obs_batch([observation(), observation()])
    with pytest.raises(ValueError, match="one environment"):
        model.get_action_batch([0, 1])
    for invalid in (
        b"encoded bytes require official server decoding",
        np.zeros((8, 9)),
    ):
        obs = observation()
        obs["vision"]["cam_left_wrist"]["color"] = invalid
        with pytest.raises(ValueError, match="decoded uint8 RGB"):
            model.update_obs(obs)


def test_private_login_file_is_rejected_before_agent_creation(adapter_factory):
    adapter = adapter_factory()
    (adapter.directory / "runtime/preflight/codex-home/auth.json").write_text("{}")
    with pytest.raises(ValueError, match="auth.json"):
        runtime_config.load_agent_config(adapter.model.model_cfg)


@pytest.mark.parametrize("missing", ["ROBORSI_CODEX_BASE_URL", "ROBORSI_CODEX_MODEL"])
def test_missing_route_fails_before_runtime_or_vla(monkeypatch, missing):
    monkeypatch.setenv("CODEX_API_KEY", "synthetic-unused-offline-key")
    monkeypatch.delenv(missing, raising=False)

    def must_not_start(*_args, **_kwargs):
        pytest.fail("Missing deployment config must fail before runtime or VLA startup")

    monkeypatch.setattr(model_module, "resolve_tool_python", must_not_start)
    monkeypatch.setattr(model_module.Model, "_ensure_vla", must_not_start)
    with pytest.raises(ValueError, match=missing):
        model_module.Model({"bench_name": "RoboDojo", "env_cfg_type": "arx_x5",
                            "action_type": "joint", "eval_batch": False})
