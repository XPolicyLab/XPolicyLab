from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from robodojo.env_client import EnvClientBaselineConfig
from robodojo.env_client.runner import run_debug_trial
from robodojo_fixtures import platform_dispatch
from robodojo.schemas import DispatchPayload
from robodojo.servers.env_client_server import (
    EnvClientServerConfig,
    EnvClientServerState,
    baseline_from_args,
    create_server,
    session_dispatch_path,
    session_start_path,
    session_stop_path,
    _validate_startup_args,
)
from robodojo.servers.session_routes import parse_session_route


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


def _dispatch_payload() -> dict[str, Any]:
    return platform_dispatch()


def _completed_result(deploy_cfg: dict[str, Any], *, steps: int = 5) -> dict[str, Any]:
    return {
        "status": "completed",
        "trial_id": deploy_cfg["trial_id"],
        "steps": steps,
        "eval_env": "debug",
        "policy_name": deploy_cfg.get("policy_name", "demo_policy"),
    }


def _server_config(tmp_path, **overrides: object) -> EnvClientServerConfig:
    return EnvClientServerConfig(
        artifact_root=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=False,
        **overrides,
    )


@contextmanager
def _running_server(
    *,
    run_trial,
    tmp_path,
    deploy_yml: str | None = "/path/to/deploy.yml",
) -> Iterator[tuple[Any, EnvClientServerState]]:
    state = EnvClientServerState(
        baseline=_baseline(),
        config=_server_config(tmp_path),
        deploy_yml=deploy_yml,
        run_trial=run_trial,
    )
    server = create_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _get(port: int, path: str):
    with urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post(port: int, path: str, payload: dict[str, Any]):
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=2) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _start_trial(port: int, *, evaluation_id: str = "eval-1", trial_index: int = 1):
    _post(port, session_dispatch_path(evaluation_id), _dispatch_payload())
    return _post(port, session_start_path(evaluation_id, trial_index), {})


def _post_expect_http_error(port: int, path: str, expected_code: int) -> None:
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as exc_info:
        urlopen(request, timeout=2)
    assert exc_info.value.code == expected_code


def _wait_for_active_trial(
    state: EnvClientServerState,
    *,
    evaluation_id: str = "eval-1",
    trial_index: int = 1,
    timeout: float = 2,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline and not state.trial_control.is_active(
        evaluation_id, trial_index
    ):
        time.sleep(0.01)
    assert state.trial_control.is_active(evaluation_id, trial_index)


def test_health_reports_baseline_and_last_trial_id(tmp_path):
    with _running_server(
        run_trial=lambda deploy_cfg: _completed_result(deploy_cfg, steps=3),
        tmp_path=tmp_path,
    ) as (server, state):
        port = server.server_address[1]
        status, body = _get(port, "/v1/health")
        assert status == 200
        assert body == {
            "ok": True,
            "policy_name": "demo_policy",
            "eval_env": "debug",
            "deploy_yml": "/path/to/deploy.yml",
            "last_trial_id": None,
        }

        _start_trial(port)
        assert state.last_trial_id == "case-1-r01"

        status, body = _get(port, "/v1/health")
        assert status == 200
        assert body["last_trial_id"] == "case-1-r01"


def test_start_merges_dispatch_into_deploy_cfg(tmp_path):
    captured: list[dict[str, Any]] = []

    def run_trial(deploy_cfg: dict[str, Any]) -> dict[str, Any]:
        captured.append(deploy_cfg)
        return _completed_result(deploy_cfg)

    with _running_server(run_trial=run_trial, tmp_path=tmp_path) as (server, _state):
        port = server.server_address[1]
        status, body = _start_trial(port)

        assert status == 200
        assert body["status"] == "completed"
        assert body["trial_id"] == "case-1-r01"
        assert body["steps"] == 5
        assert body["eval_env"] == "debug"
        assert body["policy_name"] == "demo_policy"
        assert body["error"] is None
        assert body["exit_code"] == 0
        assert body["artifact_dir"].endswith("/artifacts/eval-1/trials/1")
        expected_baseline = _baseline().model_dump()
        expected_baseline.pop("action_type")  # unset baseline fields are omitted
        expected_baseline.pop("base_cfg")
        assert captured[0] == {
            **expected_baseline,
            "evaluation_id": "eval-1",
            "trial_id": "case-1-r01",
            "action_case_id": "case-1",
            "policy_server_url": "ws://127.0.0.1:19000",
            "host": "127.0.0.1",
            "port": 19000,
            "eval_episode_num": 1,
            "repeat_index": None,
        }


def test_start_returns_failed_response_on_runner_error(tmp_path):
    def run_trial(_deploy_cfg: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("policy server unavailable")

    with _running_server(run_trial=run_trial, tmp_path=tmp_path) as (server, _state):
        port = server.server_address[1]
        status, body = _start_trial(port)

        assert status == 200
        assert body["status"] == "failed"
        assert body["trial_id"] == "case-1-r01"
        assert body["exit_code"] == 1
        assert body["error"] == {
            "code": "internal",
            "message": "policy server unavailable",
        }


def test_start_requires_prior_dispatch(tmp_path):
    with _running_server(
        run_trial=lambda deploy_cfg: _completed_result(deploy_cfg, steps=0),
        tmp_path=tmp_path,
    ) as (server, _state):
        port = server.server_address[1]
        request = Request(
            f"http://127.0.0.1:{port}{session_start_path('eval-1', 1)}",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(request, timeout=2)
        assert exc_info.value.code == 404


def test_dispatch_rejects_invalid_payload(tmp_path):
    with _running_server(
        run_trial=lambda deploy_cfg: _completed_result(deploy_cfg, steps=0),
        tmp_path=tmp_path,
    ) as (server, _state):
        port = server.server_address[1]
        with pytest.raises(HTTPError) as exc_info:
            _post(port, session_dispatch_path("eval-1"), {"evaluation_id": "eval-1"})
        assert exc_info.value.code == 400


def test_dispatch_rejects_body_evaluation_id(tmp_path):
    with _running_server(
        run_trial=lambda deploy_cfg: _completed_result(deploy_cfg, steps=0),
        tmp_path=tmp_path,
    ) as (server, _state):
        port = server.server_address[1]
        payload = platform_dispatch()
        payload["evaluation_id"] = "eval-1"
        request = Request(
            f"http://127.0.0.1:{port}{session_dispatch_path('eval-1')}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(request, timeout=2)
        assert exc_info.value.code == 400


def test_validate_startup_args_requires_root_dir_for_real_eval_env():
    from argparse import ArgumentParser, Namespace

    parser = ArgumentParser()
    with pytest.raises(SystemExit) as exc_info:
        _validate_startup_args(
            parser,
            Namespace(
                no_policy_trials=False,
                no_webhook=False,
                eval_env="real",
                root_dir=None,
                action_type="ee",
            ),
        )
    assert exc_info.value.code == 2


def test_validate_startup_args_allows_missing_action_type_for_real_eval_env():
    from argparse import ArgumentParser, Namespace

    parser = ArgumentParser()
    _validate_startup_args(
        parser,
        Namespace(
            no_policy_trials=False,
            no_webhook=False,
            eval_env="real",
            root_dir="/pipeline/root",
            base_cfg="x-one",
            action_type=None,
        ),
    )


def test_baseline_from_args_includes_root_dir():
    from argparse import Namespace

    baseline = baseline_from_args(
        Namespace(
            dataset_name="demo_dataset",
            task_name="lift-cube",
            env_cfg_type="arx_x5",
            policy_name="demo_policy",
            protocol="robodojo_ws",
            host="localhost",
            port=19000,
            eval_batch=False,
            eval_episode_num=10,
            eval_env="real",
            root_dir="/pipeline/root",
            base_cfg="x-one",
            action_type="ee",
        )
    )
    assert baseline.eval_env == "real"
    assert baseline.root_dir == "/pipeline/root"
    assert baseline.base_cfg == "x-one"
    assert baseline.action_type == "ee"


def test_parse_session_route_recognizes_stop():
    assert parse_session_route("/sessions/eval-1/trials/2/stop") == (
        "eval-1",
        "stop",
        2,
    )


def test_reset_calls_idle_env_reset(tmp_path):
    reset_calls: list[str] = []

    def run_trial(deploy_cfg: dict[str, Any], *, stop_check=lambda: False):
        class _Env:
            def reset(self) -> None:
                reset_calls.append("reset")

        env = _Env()
        env.reset()
        return _completed_result(deploy_cfg, steps=0)

    with patch(
        "robodojo.servers.env_client_server.reset_idle_env",
        side_effect=lambda _baseline, **_kwargs: reset_calls.append("reset"),
    ):
        with _running_server(
            run_trial=run_trial,
            tmp_path=tmp_path,
        ) as (server, _state):
            port = server.server_address[1]
            status, body = _post(port, "/v1/reset", {})
            assert status == 200
            assert body == {"status": "reset"}
            assert reset_calls == ["reset"]


def test_reset_rejects_while_trial_is_active(tmp_path):
    def run_trial(_deploy_cfg: dict[str, Any], *, stop_check=lambda: False):
        while not stop_check():
            time.sleep(0.02)

    with _running_server(run_trial=run_trial, tmp_path=tmp_path) as (server, state):
        port = server.server_address[1]
        _post(port, session_dispatch_path("eval-1"), _dispatch_payload())

        start_thread = threading.Thread(
            target=lambda: _post(port, session_start_path("eval-1", 1), {})
        )
        start_thread.start()
        _wait_for_active_trial(state)

        _post_expect_http_error(port, "/v1/reset", 409)
        state.trial_control.request_stop("eval-1", 1)
        start_thread.join(timeout=5)


def test_start_rejects_second_trial_while_another_trial_is_active(tmp_path):
    def run_trial(deploy_cfg: dict[str, Any], *, stop_check=lambda: False):
        while not stop_check():
            time.sleep(0.02)
        return _completed_result(deploy_cfg, steps=1)

    with _running_server(run_trial=run_trial, tmp_path=tmp_path) as (server, state):
        port = server.server_address[1]
        _post(port, session_dispatch_path("eval-1"), _dispatch_payload())

        start_thread = threading.Thread(
            target=lambda: _post(port, session_start_path("eval-1", 1), {})
        )
        start_thread.start()
        _wait_for_active_trial(state)

        _post(port, session_dispatch_path("eval-2"), _dispatch_payload())
        _post_expect_http_error(port, session_start_path("eval-2", 1), 409)

        state.trial_control.request_stop("eval-1", 1)
        start_thread.join(timeout=5)
        assert not state.trial_control.has_active_trials()


def test_start_allows_another_trial_after_active_trial_clears(tmp_path):
    run_count = 0
    run_count_lock = threading.Lock()

    def run_trial(deploy_cfg: dict[str, Any], *, stop_check=lambda: False):
        nonlocal run_count
        with run_count_lock:
            run_count += 1
            current_run = run_count

        if current_run == 1:
            while not stop_check():
                time.sleep(0.02)

        return _completed_result(deploy_cfg, steps=current_run)

    with _running_server(run_trial=run_trial, tmp_path=tmp_path) as (server, state):
        port = server.server_address[1]
        _post(port, session_dispatch_path("eval-1"), _dispatch_payload())

        start_thread = threading.Thread(
            target=lambda: _post(port, session_start_path("eval-1", 1), {})
        )
        start_thread.start()
        _wait_for_active_trial(state)
        state.trial_control.request_stop("eval-1", 1)
        start_thread.join(timeout=5)
        assert not state.trial_control.has_active_trials()

        _post(port, session_dispatch_path("eval-2"), _dispatch_payload())
        status, body = _post(port, session_start_path("eval-2", 1), {})

        assert status == 200
        assert body["status"] == "completed"
        assert body["steps"] == 2


def test_stop_without_active_trial_returns_not_found(tmp_path):
    with _running_server(
        run_trial=lambda deploy_cfg: _completed_result(deploy_cfg, steps=0),
        tmp_path=tmp_path,
    ) as (server, _state):
        port = server.server_address[1]
        _post_expect_http_error(port, session_stop_path("eval-1", 1), 404)


def test_stop_triggers_runner_early_exit(tmp_path):
    steps_at_stop: list[int] = []

    def run_trial(_deploy_cfg: dict[str, Any], *, stop_check=lambda: False):
        steps = 0
        while not stop_check():
            steps += 1
            time.sleep(0.02)
            if steps >= 100:
                break
        steps_at_stop.append(steps)
        return {
            "status": "completed",
            "trial_id": "case-1-r01",
            "steps": steps,
            "eval_env": "debug",
            "policy_name": "demo_policy",
        }

    with _running_server(run_trial=run_trial, tmp_path=tmp_path) as (server, state):
        port = server.server_address[1]
        _post(port, session_dispatch_path("eval-1"), _dispatch_payload())

        start_result: tuple[int, dict[str, Any]] | None = None

        def start_trial() -> None:
            nonlocal start_result
            start_result = _post(port, session_start_path("eval-1", 1), {})

        start_thread = threading.Thread(target=start_trial)
        start_thread.start()
        _wait_for_active_trial(state)

        status, body = _post(port, session_stop_path("eval-1", 1), {})
        assert status == 200
        assert body == {"status": "stopping"}

        start_thread.join(timeout=5)
        assert start_result is not None
        assert start_result[1]["status"] == "completed"
        assert steps_at_stop[0] < 100
        assert steps_at_stop[0] >= 1
        assert not state.trial_control.is_active("eval-1", 1)


def test_duplicate_stop_returns_conflict(tmp_path):
    def run_trial(_deploy_cfg: dict[str, Any], *, stop_check=lambda: False):
        while not stop_check():
            time.sleep(0.02)
        time.sleep(0.2)
        return {
            "status": "completed",
            "trial_id": "case-1-r01",
            "steps": 0,
            "eval_env": "debug",
            "policy_name": "demo_policy",
        }

    with _running_server(run_trial=run_trial, tmp_path=tmp_path) as (server, state):
        port = server.server_address[1]
        _post(port, session_dispatch_path("eval-1"), _dispatch_payload())

        start_thread = threading.Thread(
            target=lambda: _post(port, session_start_path("eval-1", 1), {})
        )
        start_thread.start()
        _wait_for_active_trial(state)

        assert _post(port, session_stop_path("eval-1", 1), {})[1] == {"status": "stopping"}
        _post_expect_http_error(port, session_stop_path("eval-1", 1), 409)

        start_thread.join(timeout=5)


def test_rejects_non_integer_trial_route(tmp_path):
    with _running_server(
        run_trial=lambda deploy_cfg: _completed_result(deploy_cfg, steps=0),
        tmp_path=tmp_path,
    ) as (server, _state):
        port = server.server_address[1]
        request = Request(
            f"http://127.0.0.1:{port}/sessions/eval-1/trials/not-int/start",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(request, timeout=2)
        assert exc_info.value.code == 404


def test_run_debug_trial_executes_episode_loop():
    import sys
    import types

    episodes: list[str] = []

    class FakeTestEnv:
        def __init__(self, deploy_cfg: dict[str, Any]):
            self.deploy_cfg = deploy_cfg
            self.episode_step = 0
            self.model_client = object()

        def reset(self) -> None:
            self.episode_step = 0
            episodes.append("reset")

        def eval_one_episode(self) -> None:
            self.episode_step = 4
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
                "eval_episode_num": 2,
                "trial_id": "case-1-r01",
                "evaluation_id": "eval-1",
                "action_case_id": "case-1",
            }
        )
    finally:
        if previous is None:
            sys.modules.pop("debug_env_client", None)
        else:
            sys.modules["debug_env_client"] = previous

    assert episodes == [
        "reset",
        "eval",
        "reset",
        "finish",
        "reset",
        "eval",
        "reset",
        "finish",
    ]
    assert result == {
        "status": "completed",
        "trial_id": "case-1-r01",
        "steps": 8,
        "eval_env": "debug",
        "policy_name": "demo_policy",
    }


def test_handle_start_submits_publish_in_background(tmp_path, monkeypatch):
    publish_started = threading.Event()
    publish_release = threading.Event()
    publish_submit_seen = {"value": False}

    def fake_run_dispatch(*_args, publish_submit=None, **kwargs):
        assert publish_submit is not None
        publish_submit_seen["value"] = True

        def work():
            publish_started.set()
            publish_release.wait(timeout=2)
            return {"webhook": {"status_code": 200}}, "completed", None

        publish_submit(work, evaluation_id="eval-1", trial_index=1)
        return 0, {
            "status": "completed",
            "policy_results": [{"trial_id": "case-1-r01", "steps": 1}],
            "trial_runs": [{"trial_id": "case-1-r01"}],
        }

    monkeypatch.setattr(
        "robodojo.servers.env_client_server.run_dispatch",
        fake_run_dispatch,
    )

    with _running_server(
        run_trial=lambda deploy_cfg: _completed_result(deploy_cfg, steps=1),
        tmp_path=tmp_path,
    ) as (server, state):
        port = server.server_address[1]
        status, body = _start_trial(port)

        assert status == 200
        assert body["status"] == "completed"
        assert publish_submit_seen["value"]
        assert publish_started.wait(timeout=2)
        publish_release.set()
        state.shutdown_publish()
