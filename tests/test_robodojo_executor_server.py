import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from robodojo_fixtures import platform_dispatch

from robodojo.executor_server import ExecutorConfig, create_server


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


def _wait_for_path(path):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"{path} was not written")


def test_executor_dispatch_persists_payload(tmp_path):
    def runner(dispatch, artifact_dir, config):
        return 0, {"status": "completed"}

    server, thread, config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        status, body = _post(port, "/sessions/eval-1/dispatch", platform_dispatch())

        assert status == 200
        assert body["status"] == "accepted"
        dispatch_path = config.work_dir / "eval-1" / "dispatch.json"
        saved = json.loads(dispatch_path.read_text(encoding="utf-8"))
        assert saved["evaluation_id"] == "eval-1"
        assert saved["policy_server_url"] == "ws://127.0.0.1:19000"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_executor_start_runs_stored_dispatch_in_background(tmp_path):
    calls = []

    def runner(dispatch, artifact_dir, config):
        calls.append((dispatch, artifact_dir, config))
        return 0, {
            "status": "completed",
            "evaluation_id": dispatch.evaluation_id,
            "trial_index": config.trial_index,
        }

    server, thread, config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        _post(port, "/sessions/eval-1/dispatch", platform_dispatch())

        status, body = _post(
            port,
            "/sessions/eval-1/trials/1/start",
            {"evaluation_id": "eval-1", "trial_index": 1},
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
        assert calls[0][0].evaluation_id == "eval-1"
        assert calls[0][1] == config.artifact_root / "eval-1" / "trials" / "1"
        assert calls[0][2].trial_index == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_executor_start_requires_prior_dispatch(tmp_path):
    def runner(dispatch, artifact_dir, config):
        return 0, {"status": "completed"}

    server, thread, _config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        request = Request(
            f"http://127.0.0.1:{port}/sessions/eval-1/trials/1/start",
            data=json.dumps({"evaluation_id": "eval-1", "trial_index": 1}).encode("utf-8"),
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
    def runner(dispatch, artifact_dir, config):
        return 0, {"status": "completed"}

    server, thread, _config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        request = Request(
            f"http://127.0.0.1:{port}/sessions/eval-1/trials/not-int/start",
            data=json.dumps({"evaluation_id": "eval-1", "trial_index": 1}).encode("utf-8"),
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


def test_executor_rejects_dispatch_evaluation_id_mismatch(tmp_path):
    def runner(dispatch, artifact_dir, config):
        return 0, {"status": "completed"}

    server, thread, _config = _start_server(tmp_path, runner)
    try:
        port = server.server_address[1]
        request = Request(
            f"http://127.0.0.1:{port}/sessions/other-eval/dispatch",
            data=json.dumps(platform_dispatch()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request, timeout=2)
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("dispatch unexpectedly accepted mismatched id")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
