"""Exercise extension discovery and worker selection without starting a cluster."""

from types import SimpleNamespace
from unittest.mock import Mock

from omegaconf import OmegaConf
import pytest

from XPolicyLab.infra.RLinf.core import extensions
from XPolicyLab.infra.RLinf.envs.robodojo.extension import ENV_ID


def test_discover_new_components(tmp_path, monkeypatch):
    import XPolicyLab.policy as policies
    import XPolicyLab.infra.RLinf.envs as envs
    from XPolicyLab.infra.RLinf.core.configuration import config_dirs

    policy_root = tmp_path / "policy"
    adapter = tmp_path / "adapter"
    policy = policy_root / "Example" / "rlinf"
    simulator = adapter / "envs" / "example"
    for component in (policy, simulator):
        (component / "config").mkdir(parents=True)
    (policy / "extension.py").write_text("def configure_model(cfg, model_cfg): pass\n")
    (simulator / "extension.py").write_text(
        'NAME = "example"\nENV_KEY = ("isaaclab", "Example-v0")\n'
    )
    monkeypatch.setattr(extensions, "POLICIES", policy_root)
    monkeypatch.setattr(extensions, "ADAPTER", adapter)
    monkeypatch.setattr(policies, "__path__", [str(policy_root)])
    monkeypatch.setattr(envs, "__path__", [str(adapter / "envs")])
    extensions.simulator_extensions.cache_clear()
    extensions.policy_extensions.cache_clear()
    try:
        assert extensions.policy_extension(
            {"xpolicylab": {"policy_name": "Example"}}
        ).configure_model
        assert (
            extensions.simulator_extension(
                {"env_type": "isaaclab", "init_params": {"id": "Example-v0"}}
            ).NAME
            == "example"
        )
        assert {policy / "config", simulator / "config"} == set(config_dirs())
    finally:
        extensions.simulator_extensions.cache_clear()
        extensions.policy_extensions.cache_clear()


@pytest.mark.parametrize(
    "policy, backend, critic, only_eval, expected_env, expected_rollout",
    [
        ("Pi_05", "robotwin", False, True, "NativeEnv", "XPolicyLabRolloutWorker"),
        ("Pi_05", "isaaclab", True, False, "NativeEnv", "RoboDojoRolloutWorker"),
        ("Pi_05", "robotwin", True, False, "NativeEnv", "XPolicyLabRolloutWorker"),
    ],
)
def test_worker_selection(
    monkeypatch, policy, backend, critic, only_eval, expected_env, expected_rollout
):
    from rlinf.workers.env.env_worker import EnvWorker
    from XPolicyLab.infra.RLinf.workers.env import XPolicyLabEnvWorker
    from XPolicyLab.infra.RLinf.workers.rollout import rollout_worker

    model = {
        "xpolicylab": {"policy_name": policy, "required_mode": "eval" if only_eval else "ppo"},
        "add_value_head": critic,
    }
    env = {"env_type": backend, "init_params": {"id": ENV_ID}}
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": only_eval},
            "env": {"eval" if only_eval else "train": env},
            "rollout" if only_eval else "actor": {"model": model},
        }
    )
    worker = XPolicyLabEnvWorker.__new__(XPolicyLabEnvWorker)
    worker.model_cfg = extensions.active_model(cfg)
    monkeypatch.setattr(
        EnvWorker, "_setup_env_and_wrappers", lambda self, cls, cfg, count: cls
    )
    selected = worker._setup_env_and_wrappers(
        type("NativeEnv", (), {}), OmegaConf.create(env), 1
    )
    assert selected.__name__ == expected_env
    assert rollout_worker(cfg).__name__ == expected_rollout


@pytest.mark.parametrize("only_eval", [True, False])
def test_driver_uses_runner_setting(monkeypatch, only_eval):
    from XPolicyLab.infra.RLinf.scripts import run

    cfg = SimpleNamespace(runner=SimpleNamespace(only_eval=only_eval))
    monkeypatch.setattr(run, "register", Mock())
    monkeypatch.setattr(run, "validate_adapter_config", Mock())
    monkeypatch.setattr(run, "validate_cfg", lambda cfg: cfg)
    train, evaluate = Mock(), Mock()
    monkeypatch.setattr(run, "train", train)
    monkeypatch.setattr(run, "evaluate", evaluate)
    run.main.__wrapped__(cfg)
    (evaluate if only_eval else train).assert_called_once_with(cfg)
    (train if only_eval else evaluate).assert_not_called()
