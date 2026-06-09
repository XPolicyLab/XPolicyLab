import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from robodojo_fixtures import platform_dispatch

from robodojo.executor_server import ExecutorConfig, _run_and_store_result, create_server
from robodojo.schemas import DispatchPayload


def _post(port: int, path: str, payload: dict):
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=2) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _start_server(tmp_path, runner):
    config = ExecutorConfig(
        work_dir=tmp_path / "work",
        artifact_root=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=False,
    )
    server = create_server("127.0.0.1", 0, config, runner=runner)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, config


def _dispatch_payload_without_path_id() -> dict:
    return platform_dispatch()


def _wait_for_path(path):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"{path} was not written")


def test_executor_dispatch_persists_payload(tmp_path):
    def runner(evaluation_id, dispatch, artifact_dir, config):
        return 0, {"status": "completed"}

    server, thread, config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        status, body = _post(
            port,
            "/sessions/eval-1/dispatch",
            _dispatch_payload_without_path_id(),
        )

        assert status == 200
        assert body["status"] == "accepted"
        dispatch_path = config.work_dir / "eval-1" / "dispatch.json"
        saved = json.loads(dispatch_path.read_text(encoding="utf-8"))
        assert "evaluation_id" not in saved
        assert saved["policy_server_url"] == "ws://127.0.0.1:19000"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_executor_start_runs_stored_dispatch_in_background(tmp_path):
    calls = []

    def runner(evaluation_id, dispatch, artifact_dir, config):
        calls.append((evaluation_id, dispatch, artifact_dir, config))
        return 0, {
            "status": "completed",
            "evaluation_id": evaluation_id,
            "trial_index": config.trial_index,
        }

    server, thread, config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        _post(port, "/sessions/eval-1/dispatch", _dispatch_payload_without_path_id())

        status, body = _post(
            port,
            "/sessions/eval-1/trials/1/start",
            {},
        )

        assert status == 200
        assert body["status"] == "started"
        assert body["trial_index"] == 1
        assert body["artifact_dir"].endswith("/artifacts/eval-1/trials/1")
        result_path = config.work_dir / "eval-1" / "trials" / "1" / "result.json"
        _wait_for_path(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["exit_code"] == 0
        assert result["summary"]["status"] == "completed"
        assert result["summary"]["trial_index"] == 1
        assert calls[0][0] == "eval-1"
        assert calls[0][2] == config.artifact_root / "eval-1" / "trials" / "1"
        assert calls[0][3].trial_index == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_executor_start_requires_prior_dispatch(tmp_path):
    def runner(evaluation_id, dispatch, artifact_dir, config):
        return 0, {"status": "completed"}

    server, thread, _config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        request = Request(
            f"http://127.0.0.1:{port}/sessions/eval-1/trials/1/start",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request, timeout=2)
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("start unexpectedly succeeded without dispatch")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_executor_rejects_non_integer_trial_route(tmp_path):
    def runner(evaluation_id, dispatch, artifact_dir, config):
        return 0, {"status": "completed"}

    server, thread, _config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        request = Request(
            f"http://127.0.0.1:{port}/sessions/eval-1/trials/not-int/start",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request, timeout=2)
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("start unexpectedly accepted non-integer trial route")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_executor_writes_result_with_numpy_policy_actions(tmp_path):
    def runner(evaluation_id, dispatch, artifact_dir, config):
        return 0, {
            "status": "done",
            "policy_results": [
                {
                    "trial_id": "case-1-r01",
                    "actions": [
                        {
                            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
                        }
                    ],
                }
            ],
        }

    server, thread, config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        _post(port, "/sessions/eval-1/dispatch", _dispatch_payload_without_path_id())
        _post(
            port,
            "/sessions/eval-1/trials/1/start",
            {},
        )
        result_path = config.work_dir / "eval-1" / "trials" / "1" / "result.json"
        _wait_for_path(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["exit_code"] == 0
        assert result["summary"]["policy_results"][0]["actions"][0][
            "left_arm_joint_state"
        ] == [0.0] * 6
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_executor_emergency_webhook_when_runner_crashes(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_notify_trial_failure(dispatch, **kwargs):
        captured["trial_index"] = kwargs["trial_index"]
        captured["error"] = kwargs["error"]
        captured["finish_url"] = dispatch.evaluation_plan.trials[0].finish_url
        return {
            "finish_url": dispatch.evaluation_plan.trials[0].finish_url,
            "status_code": 200,
            "emergency": True,
        }

    monkeypatch.setattr(
        "robodojo.executor_server.notify_trial_failure",
        fake_notify_trial_failure,
    )

    dispatch = DispatchPayload.model_validate(platform_dispatch())
    executor_config = ExecutorConfig(
        work_dir=tmp_path / "work",
        artifact_root=tmp_path / "artifacts",
        upload_s3=False,
        notify_webhook=True,
        trial_index=1,
        webhook_secret="secret",
    )

    class ExplodingState:
        def __init__(self, run_config: ExecutorConfig) -> None:
            self.config = run_config

        def runner(self, _evaluation_id, _dispatch, _artifact_dir, _config):
            raise RuntimeError("runner exploded")

        def result_path(self, evaluation_id, trial_index):
            return self.config.work_dir / "eval-1" / "trials" / "1" / "result.json"

    _run_and_store_result(
        ExplodingState(executor_config),
        "eval-1",
        dispatch,
        executor_config.artifact_root / "eval-1" / "trials" / "1",
        executor_config,
    )

    result = json.loads(
        (
            executor_config.work_dir / "eval-1" / "trials" / "1" / "result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["exit_code"] == 1
    assert result["summary"]["error"] == {
        "code": "internal",
        "message": "runner exploded",
    }
    assert result["summary"]["published"]["webhook"]["emergency"] is True
    assert captured["trial_index"] == 1
    assert captured["error"] == result["summary"]["error"]
    assert str(captured["finish_url"]).endswith("/trials/1/finish/")


def test_executor_rejects_dispatch_body_evaluation_id(tmp_path):
    def runner(evaluation_id, dispatch, artifact_dir, config):
        return 0, {"status": "completed"}

    server, thread, _config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        payload = platform_dispatch()
        payload["evaluation_id"] = "eval-1"
        request = Request(
            f"http://127.0.0.1:{port}/sessions/eval-1/dispatch",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request, timeout=2)
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("dispatch unexpectedly accepted body evaluation_id")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_executor_start_ignores_body_identifiers(tmp_path):
    calls = []

    def runner(evaluation_id, dispatch, artifact_dir, config):
        calls.append(evaluation_id)
        return 0, {"status": "completed", "trial_index": config.trial_index}

    server, thread, config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        _post(port, "/sessions/eval-1/dispatch", _dispatch_payload_without_path_id())
        status, body = _post(
            port,
            "/sessions/eval-1/trials/1/start",
            {"evaluation_id": "other-eval", "trial_index": 99},
        )

        assert status == 200
        assert body["status"] == "started"
        assert body["evaluation_id"] == "eval-1"
        assert body["trial_index"] == 1
        result_path = config.work_dir / "eval-1" / "trials" / "1" / "result.json"
        _wait_for_path(result_path)
        assert calls == ["eval-1"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
