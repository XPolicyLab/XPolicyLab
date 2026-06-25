from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from robodojo.dispatch.status import STATUS_DONE
from robodojo.publish import state_store
from robodojo.publish.state_store import (
    PUBLISH_STATUS_DONE,
    PUBLISH_STATUS_FAILED,
    PUBLISH_STATUS_PENDING,
)
from robodojo.schemas import DispatchPayload
from robodojo.servers.env_client_server import (
    EnvClientServerState,
    resume_incomplete_publishes,
    session_republish_path,
)
from robodojo.servers.session_routes import parse_session_route
from robodojo_fixtures import platform_dispatch
from test_robodojo_env_client_server import (
    _baseline,
    _dispatch_payload,
    _post,
    _post_expect_http_error,
    _running_server,
    _server_config,
    _start_trial,
)


def _dispatch_model() -> DispatchPayload:
    return DispatchPayload.model_validate(_dispatch_payload())


def test_state_store_round_trip(tmp_path):
    artifact_root = tmp_path / "artifacts"
    dispatch = _dispatch_model()
    evaluation_id = "eval-abc"

    state_store.record_dispatch(artifact_root, evaluation_id, dispatch)
    state_store.record_pending(
        artifact_root,
        evaluation_id,
        1,
        "/tmp/trial.hdf5",
        STATUS_DONE,
    )

    loaded = state_store.load_dispatches(artifact_root)
    assert evaluation_id in loaded
    assert loaded[evaluation_id].policy_server_url == dispatch.policy_server_url

    record = state_store.load_publish_record(artifact_root, evaluation_id, 1)
    assert record is not None
    assert record.hdf5_path == "/tmp/trial.hdf5"
    assert record.publish_status == PUBLISH_STATUS_PENDING

    state_store.record_outcome(
        artifact_root,
        evaluation_id,
        1,
        PUBLISH_STATUS_DONE,
    )
    record = state_store.load_publish_record(artifact_root, evaluation_id, 1)
    assert record is not None
    assert record.publish_status == PUBLISH_STATUS_DONE
    assert state_store.load_incomplete(artifact_root) == []


def test_state_store_atomic_write(tmp_path):
    artifact_root = tmp_path / "artifacts"
    dispatch = _dispatch_model()
    state_store.record_dispatch(artifact_root, "eval-1", dispatch)
    dispatch_path = artifact_root / "eval-1" / "dispatch.json"
    assert dispatch_path.is_file()
    payload = json.loads(dispatch_path.read_text(encoding="utf-8"))
    assert "evaluation_plan" in payload


def test_parse_session_route_accepts_republish():
    parsed = parse_session_route("/sessions/eval-1/trials/2/republish")
    assert parsed == ("eval-1", "republish", 2)


def test_resume_incomplete_publishes_queues_pending_tasks(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    evaluation_id = "eval-resume"
    hdf5_path = tmp_path / "recording.hdf5"
    hdf5_path.write_bytes(b"hdf5")

    state_store.record_dispatch(artifact_root, evaluation_id, _dispatch_model())
    state_store.record_pending(
        artifact_root,
        evaluation_id,
        1,
        str(hdf5_path),
        STATUS_DONE,
    )

    submitted: list[tuple[str, int]] = []

    def capture_submit(work, *, evaluation_id, trial_index):
        submitted.append((evaluation_id, trial_index))
        return work()

    state = EnvClientServerState(
        baseline=_baseline(),
        config=_server_config(tmp_path),
    )

    def capture_submit(work, *, evaluation_id, trial_index):
        submitted.append((evaluation_id, trial_index))
        return work()

    state.submit_publish = capture_submit  # type: ignore[method-assign]

    queued = resume_incomplete_publishes(state)
    assert queued == 1
    assert submitted == [(evaluation_id, 1)]
    assert evaluation_id in state.dispatches


def test_resume_incomplete_publishes_marks_missing_hdf5_failed(tmp_path):
    artifact_root = tmp_path / "artifacts"
    evaluation_id = "eval-missing"

    state_store.record_dispatch(artifact_root, evaluation_id, _dispatch_model())
    state_store.record_pending(
        artifact_root,
        evaluation_id,
        1,
        str(tmp_path / "missing.hdf5"),
        STATUS_DONE,
    )

    state = EnvClientServerState(
        baseline=_baseline(),
        config=_server_config(tmp_path),
    )
    assert resume_incomplete_publishes(state) == 0
    record = state_store.load_publish_record(artifact_root, evaluation_id, 1)
    assert record is not None
    assert record.publish_status == PUBLISH_STATUS_FAILED


def test_handle_republish_returns_conflict_when_recording_missing(tmp_path):
    with _running_server(
        run_trial=lambda deploy_cfg: {
            "status": "completed",
            "trial_id": deploy_cfg["trial_id"],
            "steps": 1,
            "eval_env": "debug",
            "policy_name": "demo_policy",
        },
        tmp_path=tmp_path,
    ) as (server, _state):
        port = server.server_address[1]
        evaluation_id = "eval-republish-missing"
        _post(port, f"/sessions/{evaluation_id}/dispatch", _dispatch_payload())
        _post_expect_http_error(
            port,
            session_republish_path(evaluation_id, 1),
            expected_code=409,
        )


def test_handle_republish_queues_publish_work(tmp_path, monkeypatch):
    publish_started = threading.Event()
    publish_release = threading.Event()
    submitted: list[tuple[str, int]] = []

    def fake_build_republish_work(*_args, **_kwargs):
        def work():
            publish_started.set()
            publish_release.wait(timeout=2)
            return {"publish_status": PUBLISH_STATUS_DONE}, "completed", None

        return work

    monkeypatch.setattr(
        "robodojo.servers.env_client_server.build_republish_work",
        fake_build_republish_work,
    )

    original_submit = EnvClientServerState.submit_publish

    def capture_submit(self, work, *, evaluation_id, trial_index):
        submitted.append((evaluation_id, trial_index))
        return original_submit(self, work, evaluation_id=evaluation_id, trial_index=trial_index)

    monkeypatch.setattr(EnvClientServerState, "submit_publish", capture_submit)

    hdf5_path = tmp_path / "recording.hdf5"
    hdf5_path.write_bytes(b"hdf5")

    def fake_run_dispatch(*_args, **_kwargs):
        return 0, {
            "status": "completed",
            "policy_results": [{"trial_id": "case-1-r01", "hdf5_path": str(hdf5_path)}],
            "trial_runs": [{"trial_id": "case-1-r01"}],
        }

    monkeypatch.setattr(
        "robodojo.servers.env_client_server.run_dispatch",
        fake_run_dispatch,
    )

    with _running_server(
        run_trial=lambda deploy_cfg: {
            "status": "completed",
            "trial_id": deploy_cfg["trial_id"],
            "steps": 1,
            "eval_env": "debug",
            "policy_name": "demo_policy",
        },
        tmp_path=tmp_path,
    ) as (server, state):
        port = server.server_address[1]
        evaluation_id = "eval-republish"
        _start_trial(port, evaluation_id=evaluation_id, trial_index=1)
        state_store.record_pending(
            state.config.artifact_root,
            evaluation_id,
            1,
            str(hdf5_path),
            STATUS_DONE,
        )

        status, body = _post(
            port,
            session_republish_path(evaluation_id, 1),
            {},
        )
        assert status == 200
        assert body == {"status": "republishing"}
        assert submitted == [(evaluation_id, 1)]
        assert publish_started.wait(timeout=2)
        publish_release.set()
        state.shutdown_publish()

        record = state_store.load_publish_record(
            state.config.artifact_root,
            evaluation_id,
            1,
        )
        assert record is not None
        assert record.publish_status == PUBLISH_STATUS_DONE
