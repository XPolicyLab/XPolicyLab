from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import Any

import pytest
from robodojo_fixtures import platform_dispatch

from robodojo.dispatch import build_trial_runs, run_dispatch
from robodojo.env_client import (
    EnvClientBaselineConfig,
    TrialRunnerError,
    make_dispatch_trial_runner,
    reset_idle_env,
    run_debug_trial,
    run_real_trial,
)
from robodojo.env_client.runner import _cleanup_env
from robodojo.schemas import DispatchPayload


def _baseline(**overrides: object) -> EnvClientBaselineConfig:
    return EnvClientBaselineConfig.model_validate(
        {
            "dataset_name": "demo_dataset",
            "task_name": "lift-cube",
            "env_cfg_type": "arx_x5",
            "policy_name": "demo_policy",
            "host": "localhost",
            "port": 19000,
            **overrides,
        }
    )


def test_make_dispatch_trial_runner_builds_deploy_cfg_and_runs_trial():
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]
    captured: list[dict[str, object]] = []

    def run_trial(
        deploy_cfg: dict[str, object],
        *,
        stop_check: object = None,
    ) -> dict[str, object]:
        captured.append(deploy_cfg)
        return {
            "status": "completed",
            "trial_id": deploy_cfg["trial_id"],
            "steps": 4,
            "eval_env": "debug",
            "policy_name": deploy_cfg.get("policy_name"),
        }

    runner = make_dispatch_trial_runner(_baseline(), run_trial=run_trial)
    result = runner(dispatch, trial_run, "eval-1")

    assert captured[0]["evaluation_id"] == "eval-1"
    assert captured[0]["trial_id"] == "case-1-r01"
    assert result == {
        "trial_id": "case-1-r01",
        "steps": 4,
        "eval_env": "debug",
        "policy_name": "demo_policy",
        "actions": [],
    }


def test_make_dispatch_trial_runner_raises_on_failed_status():
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]

    def run_trial(
        _deploy_cfg: dict[str, object],
        *,
        stop_check: object = None,
    ) -> dict[str, object]:
        return {
            "status": "failed",
            "error": {"code": "trial_failed", "message": "policy unreachable"},
        }

    runner = make_dispatch_trial_runner(_baseline(), run_trial=run_trial)

    with pytest.raises(TrialRunnerError, match="policy unreachable") as exc_info:
        runner(dispatch, trial_run, "eval-1")

    assert exc_info.value.error == {
        "code": "trial_failed",
        "message": "policy unreachable",
    }


def test_make_dispatch_trial_runner_real_baseline_includes_root_dir_and_skips_episode_override():
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]
    captured: list[dict[str, object]] = []

    def run_trial(
        deploy_cfg: dict[str, object],
        *,
        stop_check: object = None,
    ) -> dict[str, object]:
        captured.append(deploy_cfg)
        return {
            "status": "completed",
            "trial_id": deploy_cfg["trial_id"],
            "steps": 0,
            "eval_env": "real",
            "policy_name": deploy_cfg.get("policy_name"),
        }

    runner = make_dispatch_trial_runner(
        _baseline(eval_env="real", root_dir="/pipeline/root", action_type="ee"),
        run_trial=run_trial,
        eval_episode_num=1,
    )
    runner(dispatch, trial_run, "eval-1")

    assert captured[0]["root_dir"] == "/pipeline/root"
    assert captured[0]["eval_env"] == "real"
    assert captured[0]["eval_episode_num"] == 10


def test_make_dispatch_trial_runner_passes_stop_check_factory():
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]
    factory_inputs: list[dict[str, object]] = []
    captured_stop_checks: list[Callable[[], bool]] = []

    def run_trial(
        _deploy_cfg: dict[str, object],
        *,
        stop_check: Callable[[], bool] = lambda: False,
    ) -> dict[str, object]:
        captured_stop_checks.append(stop_check)
        return {"status": "completed", "trial_id": "case-1-r01", "steps": 0}

    runner = make_dispatch_trial_runner(
        _baseline(),
        run_trial=run_trial,
        stop_check_factory=lambda deploy_cfg: factory_inputs.append(deploy_cfg) or (
            lambda: False
        ),
    )
    runner(dispatch, trial_run, "eval-1")

    assert factory_inputs[0]["trial_id"] == "case-1-r01"
    assert captured_stop_checks[0]() is False


def test_run_debug_trial_stop_check_exits_before_eval_episode_num():
    episodes: list[str] = []
    completed_episodes = 0

    class FakeTestEnv:
        def __init__(self, deploy_cfg: dict[str, Any]):
            self.deploy_cfg = deploy_cfg
            self.episode_step = 0
            self.model_client = object()

        def reset(self) -> None:
            self.episode_step = 0
            episodes.append("reset")

        def eval_one_episode(self) -> None:
            self.episode_step = 3
            episodes.append("eval")

        def eval_one_episode_batch(self) -> None:
            raise AssertionError("batch path should not run")

        def finish_episode(self) -> None:
            nonlocal completed_episodes
            completed_episodes += 1
            episodes.append("finish")

    fake_module = types.ModuleType("debug_env_client")
    fake_module.TestEnv = FakeTestEnv
    previous = sys.modules.get("debug_env_client")
    sys.modules["debug_env_client"] = fake_module
    try:
        result = run_debug_trial(
            {
                **_baseline().model_dump(),
                "host": "127.0.0.1",
                "eval_episode_num": 3,
                "trial_id": "case-1-r01",
                "evaluation_id": "eval-1",
                "action_case_id": "case-1",
            },
            stop_check=lambda: completed_episodes >= 1,
        )
    finally:
        if previous is None:
            sys.modules.pop("debug_env_client", None)
        else:
            sys.modules["debug_env_client"] = previous

    assert episodes == ["reset", "eval", "finish", "reset"]
    assert result["steps"] == 3


def test_run_debug_trial_stop_check_exits_mid_episode():
    episodes: list[str] = []
    stop_requested = False

    class FakeTestEnv:
        def __init__(self, deploy_cfg: dict[str, Any]):
            self.deploy_cfg = deploy_cfg
            self.episode_step = 0
            self.model_client = object()
            self._stop_check = None

        def set_stop_check(self, stop_check: Callable[[], bool]) -> None:
            self._stop_check = stop_check

        def reset(self) -> None:
            self.episode_step = 0
            episodes.append("reset")

        def is_episode_end(self) -> bool:
            if self._stop_check is not None and self._stop_check():
                return True
            return self.episode_step >= 100

        def eval_one_episode(self) -> None:
            while not self.is_episode_end():
                self.episode_step += 1
                if self.episode_step >= 5:
                    nonlocal stop_requested
                    stop_requested = True
            episodes.append("eval")

        def eval_one_episode_batch(self) -> None:
            raise AssertionError("batch path should not run")

        def finish_episode(self) -> None:
            episodes.append("finish")

    fake_module = types.ModuleType("debug_env_client")
    fake_module.TestEnv = FakeTestEnv
    previous = sys.modules.get("debug_env_client")
    sys.modules["debug_env_client"] = fake_module
    try:
        result = run_debug_trial(
            {
                **_baseline().model_dump(),
                "host": "127.0.0.1",
                "eval_episode_num": 3,
                "trial_id": "case-1-r01",
                "evaluation_id": "eval-1",
                "action_case_id": "case-1",
            },
            stop_check=lambda: stop_requested,
        )
    finally:
        if previous is None:
            sys.modules.pop("debug_env_client", None)
        else:
            sys.modules["debug_env_client"] = previous

    assert episodes == ["reset", "eval", "finish", "reset"]
    assert result["steps"] == 5


def test_cleanup_env_closes_model_client_and_calls_env_cleanup():
    events: list[str] = []

    class FakeModelClient:
        def close(self) -> None:
            events.append("model_client.close")

    class FakeEnv:
        def __init__(self) -> None:
            self.model_client = FakeModelClient()

        def cleanup(self) -> None:
            events.append("env.cleanup")

    _cleanup_env(FakeEnv())
    assert events == ["model_client.close", "env.cleanup"]


def test_run_real_trial_wires_stop_check_into_episode_end():
    class FakeRealEnv:
        def __init__(self, deploy_cfg: dict[str, Any]):
            self.deploy_cfg = deploy_cfg
            self.episode_step = 0
            self.model_client = object()
            self._stop_check: Callable[[], bool] | None = None

        def set_stop_check(self, stop_check: Callable[[], bool]) -> None:
            self._stop_check = stop_check

        def is_episode_end(self) -> bool:
            return self._stop_check is not None and self._stop_check()

        def reset(self) -> None:
            self.episode_step = 0

        def eval_one_episode(self) -> None:
            while not self.is_episode_end():
                self.episode_step += 1

        def eval_one_episode_batch(self) -> None:
            raise AssertionError("batch path should not run")

        def finish_episode(self) -> None:
            return None

    fake_real_module = types.ModuleType("task_env.real_env_client")
    fake_real_module.RealEnv = FakeRealEnv
    previous_real = sys.modules.get("task_env.real_env_client")
    previous_task_env = sys.modules.get("task_env")
    sys.modules["task_env.real_env_client"] = fake_real_module
    if previous_task_env is None:
        sys.modules["task_env"] = types.ModuleType("task_env")
    try:
        result = run_real_trial(
            {
                **_baseline(eval_env="real", root_dir="/pipeline/root").model_dump(),
                "host": "127.0.0.1",
                "trial_id": "case-1-r01",
                "evaluation_id": "eval-1",
                "action_case_id": "case-1",
            },
            stop_check=lambda: True,
        )
    finally:
        if previous_real is None:
            sys.modules.pop("task_env.real_env_client", None)
        else:
            sys.modules["task_env.real_env_client"] = previous_real
        if previous_task_env is None:
            sys.modules.pop("task_env", None)

    assert result["steps"] == 0


def test_run_real_trial_loops_until_stop_check():
    episodes: list[str] = []
    completed_episodes = 0

    class FakeRealEnv:
        def __init__(self, deploy_cfg: dict[str, Any]):
            self.deploy_cfg = deploy_cfg
            self.episode_step = 0
            self.model_client = object()
            self._stop_check: Callable[[], bool] | None = None

        def set_stop_check(self, stop_check: Callable[[], bool]) -> None:
            self._stop_check = stop_check

        def reset(self) -> None:
            self.episode_step = 0
            episodes.append("reset")

        def eval_one_episode(self) -> None:
            self.episode_step = 2
            episodes.append("eval")

        def eval_one_episode_batch(self) -> None:
            raise AssertionError("batch path should not run")

        def finish_episode(self) -> None:
            nonlocal completed_episodes
            completed_episodes += 1
            episodes.append("finish")

    fake_real_module = types.ModuleType("task_env.real_env_client")
    fake_real_module.RealEnv = FakeRealEnv
    previous_real = sys.modules.get("task_env.real_env_client")
    previous_task_env = sys.modules.get("task_env")
    sys.modules["task_env.real_env_client"] = fake_real_module
    if previous_task_env is None:
        sys.modules["task_env"] = types.ModuleType("task_env")
    try:
        result = run_real_trial(
            {
                **_baseline(eval_env="real", root_dir="/pipeline/root").model_dump(),
                "host": "127.0.0.1",
                "trial_id": "case-1-r01",
                "evaluation_id": "eval-1",
                "action_case_id": "case-1",
            },
            stop_check=lambda: completed_episodes >= 2,
        )
    finally:
        if previous_real is None:
            sys.modules.pop("task_env.real_env_client", None)
        else:
            sys.modules["task_env.real_env_client"] = previous_real
        if previous_task_env is None:
            sys.modules.pop("task_env", None)

    assert episodes == [
        "reset",
        "eval",
        "finish",
        "reset",
        "eval",
        "finish",
        "reset",
    ]
    assert result == {
        "status": "completed",
        "trial_id": "case-1-r01",
        "steps": 4,
        "eval_env": "real",
        "policy_name": "demo_policy",
    }


def test_reset_idle_env_releases_env_resources_after_reset():
    events: list[str] = []

    class FakeModelClient:
        def close(self) -> None:
            events.append("model_client.close")

    class FakeTestEnv:
        def __init__(self, deploy_cfg: dict[str, Any]):
            self.deploy_cfg = deploy_cfg
            self.model_client = FakeModelClient()

        def reset(self) -> None:
            events.append("reset")

        def cleanup(self) -> None:
            events.append("cleanup")

    fake_module = types.ModuleType("debug_env_client")
    fake_module.TestEnv = FakeTestEnv
    previous = sys.modules.get("debug_env_client")
    sys.modules["debug_env_client"] = fake_module
    try:
        reset_idle_env(_baseline())
    finally:
        if previous is None:
            sys.modules.pop("debug_env_client", None)
        else:
            sys.modules["debug_env_client"] = previous

    assert events == ["reset", "model_client.close", "cleanup"]


def test_run_real_trial_requires_root_dir():
    result = run_real_trial(
        {
            **_baseline(eval_env="real").model_dump(),
            "host": "127.0.0.1",
            "trial_id": "case-1-r01",
            "evaluation_id": "eval-1",
            "action_case_id": "case-1",
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "missing_root_dir"


def test_run_dispatch_uses_injected_trial_runner(monkeypatch: pytest.MonkeyPatch):
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    captured: dict[str, object] = {}

    def fake_runner(_dispatch, trial_run, evaluation_id):
        captured["trial_id"] = trial_run["trial_id"]
        captured["evaluation_id"] = evaluation_id
        return {
            "trial_id": "case-1-r01",
            "steps": 3,
            "eval_env": "debug",
            "policy_name": "demo_policy",
            "actions": [],
        }

    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id="eval-1",
        artifact_dir=None,
        upload_s3=False,
        notify_webhook=False,
        run_policy_trials=True,
        trial_index=1,
        trial_runner=fake_runner,
    )

    assert exit_code == 0
    assert captured["evaluation_id"] == "eval-1"
    assert summary["policy_results"][0]["steps"] == 3
