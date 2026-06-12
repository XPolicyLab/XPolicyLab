from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from robodojo.env_client.api import EnvClientBaselineConfig
from robodojo.servers.env_client_server import (
    EnvClientServerConfig,
    EnvClientServerState,
    create_server,
    session_dispatch_path,
    session_start_path,
)
from robodojo.servers.preview_routes import MJPEG_BOUNDARY, parse_preview_route
from robodojo_fixtures import platform_dispatch

FAKE_JPEG = b"\xff\xd8fake-jpeg-bytes"
PLACEHOLDER_JPEG = b"\xff\xd8placeholder"


class FakePreviewManager:
    def __init__(self, *, with_frames: bool = True):
        self.events: list[str] = []
        self.with_frames = with_frames

    def roles(self):
        return ("head", "left_wrist", "right_wrist")

    def ensure_started(self):
        self.events.append("ensure_started")

    def pause(self):
        self.events.append("pause")

    def resume_async(self):
        self.events.append("resume")

    def frame(self, role):
        return (FAKE_JPEG, 1) if self.with_frames else None

    def wait_frame(self, role, after_timestamp_ns=0, timeout_s=1.0):
        return (FAKE_JPEG, after_timestamp_ns + 1) if self.with_frames else None

    def placeholder_jpeg(self):
        return PLACEHOLDER_JPEG

    def status(self):
        return {"active": self.with_frames, "paused": False, "last_error": None}


def _baseline() -> EnvClientBaselineConfig:
    return EnvClientBaselineConfig.model_validate(
        {
            "dataset_name": "demo_dataset",
            "task_name": "lift-cube",
            "env_cfg_type": "arx_x5",
            "policy_name": "demo_policy",
            "host": "localhost",
            "port": 19000,
        }
    )


@contextmanager
def _running_server(
    tmp_path,
    *,
    preview: Any | None,
    run_trial=None,
) -> Iterator[tuple[Any, EnvClientServerState]]:
    state = EnvClientServerState(
        baseline=_baseline(),
        config=EnvClientServerConfig(
            artifact_root=tmp_path / "artifacts",
            upload_s3=False,
            notify_webhook=False,
        ),
        run_trial=run_trial,
        preview=preview,
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


def _get_json(port: int, path: str):
    with urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_json(port: int, path: str):
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=2) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_parse_preview_route():
    assert parse_preview_route("/v1/preview/status") == ("status", None)
    assert parse_preview_route("/v1/preview/pause") == ("pause", None)
    assert parse_preview_route("/v1/preview/resume") == ("resume", None)
    assert parse_preview_route("/v1/preview/head.mjpeg") == ("stream", "head")
    assert parse_preview_route("/v1/preview/left_wrist.jpg") == (
        "snapshot",
        "left_wrist",
    )
    assert parse_preview_route("/v1/preview/head.mjpeg?x=1") == ("stream", "head")
    assert parse_preview_route("/v1/preview/.mjpeg") is None
    assert parse_preview_route("/v1/preview") is None
    assert parse_preview_route("/v1/health") is None


def test_preview_disabled_returns_503(tmp_path):
    with _running_server(tmp_path, preview=None) as (server, _state):
        port = server.server_address[1]
        with pytest.raises(HTTPError) as exc_info:
            _get_json(port, "/v1/preview/status")
        assert exc_info.value.code == 503


def test_preview_status_reports_roles_and_state(tmp_path):
    preview = FakePreviewManager()
    with _running_server(tmp_path, preview=preview) as (server, _state):
        port = server.server_address[1]
        status, body = _get_json(port, "/v1/preview/status")
        assert status == 200
        assert body["active"] is True
        assert body["roles"] == ["head", "left_wrist", "right_wrist"]
        assert "ensure_started" in preview.events


def test_preview_snapshot_serves_jpeg_with_cors(tmp_path):
    preview = FakePreviewManager()
    with _running_server(tmp_path, preview=preview) as (server, _state):
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/v1/preview/head.jpg", timeout=2) as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/jpeg"
            assert resp.headers["Access-Control-Allow-Origin"] == "*"
            assert resp.read() == FAKE_JPEG


def test_preview_snapshot_falls_back_to_placeholder(tmp_path):
    preview = FakePreviewManager(with_frames=False)
    with _running_server(tmp_path, preview=preview) as (server, _state):
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/v1/preview/head.jpg", timeout=2) as resp:
            assert resp.read() == PLACEHOLDER_JPEG


def test_preview_unknown_role_returns_404(tmp_path):
    preview = FakePreviewManager()
    with _running_server(tmp_path, preview=preview) as (server, _state):
        port = server.server_address[1]
        with pytest.raises(HTTPError) as exc_info:
            _get_json(port, "/v1/preview/torso.jpg")
        assert exc_info.value.code == 404


def test_preview_stream_emits_multipart_jpeg_frames(tmp_path):
    preview = FakePreviewManager()
    with _running_server(tmp_path, preview=preview) as (server, _state):
        port = server.server_address[1]
        response = urlopen(f"http://127.0.0.1:{port}/v1/preview/head.mjpeg", timeout=3)
        try:
            content_type = response.headers["Content-Type"]
            assert content_type.startswith("multipart/x-mixed-replace")
            assert MJPEG_BOUNDARY in content_type
            chunk = response.read(len(FAKE_JPEG) + 120)
            assert f"--{MJPEG_BOUNDARY}".encode("ascii") in chunk
            assert b"Content-Type: image/jpeg" in chunk
            assert FAKE_JPEG[:8] in chunk
        finally:
            response.close()


def test_preview_status_responds_while_manager_lock_is_held(tmp_path):
    """Camera open holds the manager lock for seconds; status must not block
    past the browser watchdog timeout (regression for BrokenPipeError storm)."""
    import sys
    from pathlib import Path

    robot_src = Path(__file__).resolve().parents[2] / "src"
    if not (robot_src / "robot").is_dir():
        pytest.skip("X-Robot-Pipeline src not available")
    sys.path.insert(0, str(robot_src))
    from robot.sensor.orbbec_preview import OrbbecPreviewManager

    manager = OrbbecPreviewManager.__new__(OrbbecPreviewManager)
    manager._lock = threading.RLock()
    manager._hub = type(
        "Hub", (), {"latest": staticmethod(lambda serial: None)}
    )()
    manager._serial_by_role = {"head": "SN1"}
    manager._active = False
    manager._paused = False
    manager._shutdown = False
    manager._last_error = None
    manager._last_open_attempt = 0.0

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        with manager._lock:
            lock_acquired.set()
            release_lock.wait(timeout=5)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert lock_acquired.wait(timeout=2)

    try:
        import time as _time

        started = _time.monotonic()
        manager.ensure_started()  # must skip, not block on the held lock
        body = manager.status()
        elapsed = _time.monotonic() - started
        assert elapsed < 0.5, f"status blocked for {elapsed:.2f}s on manager lock"
        assert body["active"] is False
    finally:
        release_lock.set()
        holder.join(timeout=2)


def test_preview_json_write_tolerates_client_disconnect():
    """Browser fetch abort mid-response must not crash the request thread."""
    from robodojo.servers.preview_routes import handle_preview_get

    class BrokenWfile:
        def write(self, _data):
            raise BrokenPipeError(32, "Broken pipe")

    class BrokenHandler:
        wfile = BrokenWfile()

        def send_response(self, _code):
            return

        def send_header(self, _k, _v):
            return

        def end_headers(self):
            return

    # Must not raise even though every write hits a dead socket.
    handle_preview_get(BrokenHandler(), FakePreviewManager(), "status", None)
    handle_preview_get(BrokenHandler(), FakePreviewManager(), "snapshot", "head")


def test_preview_pause_resume_endpoints(tmp_path):
    preview = FakePreviewManager()
    with _running_server(tmp_path, preview=preview) as (server, _state):
        port = server.server_address[1]
        status, body = _post_json(port, "/v1/preview/pause")
        assert (status, body["status"]) == (200, "paused")
        status, body = _post_json(port, "/v1/preview/resume")
        assert (status, body["status"]) == (200, "resuming")
        assert preview.events == ["pause", "resume"]


def test_trial_start_pauses_preview_then_resumes(tmp_path):
    preview = FakePreviewManager()

    def run_trial(deploy_cfg: dict[str, Any]) -> dict[str, Any]:
        preview.events.append("trial")
        return {
            "status": "completed",
            "trial_id": deploy_cfg["trial_id"],
            "steps": 1,
            "eval_env": "debug",
            "policy_name": deploy_cfg.get("policy_name", "demo_policy"),
        }

    with _running_server(tmp_path, preview=preview, run_trial=run_trial) as (
        server,
        _state,
    ):
        port = server.server_address[1]
        _post_json_payload(port, session_dispatch_path("eval-1"), platform_dispatch())
        status, body = _post_json(port, session_start_path("eval-1", 1))
        assert status == 200
        assert body["status"] == "completed"

    ordered = [e for e in preview.events if e in ("pause", "trial", "resume")]
    assert ordered == ["pause", "trial", "resume"]


def test_trial_start_resumes_preview_on_runner_error(tmp_path):
    preview = FakePreviewManager()

    def run_trial(_deploy_cfg: dict[str, Any]) -> dict[str, Any]:
        preview.events.append("trial")
        raise RuntimeError("boom")

    with _running_server(tmp_path, preview=preview, run_trial=run_trial) as (
        server,
        _state,
    ):
        port = server.server_address[1]
        _post_json_payload(port, session_dispatch_path("eval-1"), platform_dispatch())
        status, body = _post_json(port, session_start_path("eval-1", 1))
        assert status == 200
        assert body["status"] == "failed"

    ordered = [e for e in preview.events if e in ("pause", "trial", "resume")]
    assert ordered == ["pause", "trial", "resume"]


def _post_json_payload(port: int, path: str, payload: dict[str, Any]):
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))
