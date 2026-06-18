import json
from pathlib import Path

from robodojo_fixtures import platform_dispatch

from robodojo.publish import publish_trial_recording
from robodojo.schemas import ArtifactPayload, DispatchPayload

FINISH_URL = "https://example.test/api/v1/internal/eval/eval-1/trials/1/finish/"
VIDEO_KEY = "evaluations/eval-1/trial_1.mp4"
HDF5_KEY = "evaluations/eval-1/trial_1.hdf5"


def _dispatch_payload() -> DispatchPayload:
    return DispatchPayload.model_validate(
        platform_dispatch(
            callback={"hmac_secret_ref": "ROBODOJO_WEBHOOK_SECRET"},
            evaluation_plan={
                "repeat_count": 1,
                "trials": [
                    {
                        "trial_id": "case-1-r01",
                        "action_case_id": "case-1",
                        "trial_index": 1,
                        "finish_url": FINISH_URL,
                    }
                ],
            },
        )
    )


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def getcode(self):
        return 200


def _capturing_opener(captured: dict):
    def _open(request, timeout=30):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    return _open


def test_publish_trial_recording_uploads_video_and_hdf5(tmp_path, monkeypatch):
    dispatch = _dispatch_payload()
    hdf5_path = tmp_path / "recording.hdf5"
    hdf5_path.write_bytes(b"fake hdf5")

    def fake_encode(_hdf5_path: str) -> Path:
        mp4 = tmp_path / "recording.mp4"
        mp4.write_bytes(b"fake mp4")
        return mp4

    monkeypatch.setattr(
        "robodojo.publish.pipeline._encode_trial_video", fake_encode
    )

    uploads: list[str] = []

    def fake_upload(key: str, path: Path, content_type: str | None) -> None:
        uploads.append(key)

    captured: dict = {}
    published, status, error = publish_trial_recording(
        dispatch,
        finish_url=FINISH_URL,
        run_status="done",
        video_key=VIDEO_KEY,
        hdf5_key=HDF5_KEY,
        hdf5_path=str(hdf5_path),
        s3_client=object(),
        upload_file=fake_upload,
        webhook_secret="secret",
        webhook_opener=_capturing_opener(captured),
    )

    assert status == "completed"
    assert error is None
    assert VIDEO_KEY in uploads
    assert HDF5_KEY in uploads
    assert published["s3"]["video_s3_key"] == VIDEO_KEY
    assert published["s3"]["hdf5_s3_key"] == HDF5_KEY
    assert published["webhook"]["status_code"] == 200
    # The publish webhook only patches the trial artifact (phase=publish); the
    # robot trial outcome was already persisted from the /start response.
    assert captured["body"]["phase"] == "publish"
    assert captured["body"]["status"] == "done"
    assert captured["body"]["artifact"]["publish_status"] == "done"
    assert captured["body"]["artifact"]["video_s3_key"] == VIDEO_KEY
    assert captured["body"]["artifact"]["hdf5_s3_key"] == HDF5_KEY
    # The temporary mp4 is cleaned up after upload.
    assert not (tmp_path / "recording.mp4").exists()


def test_publish_trial_recording_skips_video_when_encoding_fails(tmp_path, monkeypatch):
    dispatch = _dispatch_payload()
    hdf5_path = tmp_path / "recording.hdf5"
    hdf5_path.write_bytes(b"fake hdf5")

    monkeypatch.setattr(
        "robodojo.publish.pipeline._encode_trial_video", lambda _hdf5_path: None
    )

    uploads: list[str] = []

    def fake_upload(key: str, path: Path, content_type: str | None) -> None:
        uploads.append(key)

    published, status, error = publish_trial_recording(
        dispatch,
        finish_url=FINISH_URL,
        run_status="done",
        video_key=VIDEO_KEY,
        hdf5_key=HDF5_KEY,
        hdf5_path=str(hdf5_path),
        s3_client=object(),
        upload_file=fake_upload,
        notify_webhook=False,
    )

    assert status == "completed"
    assert VIDEO_KEY not in uploads
    assert HDF5_KEY in uploads
    assert "video_s3_key" not in published["s3"]
    assert published["s3"]["hdf5_s3_key"] == HDF5_KEY


def test_publish_trial_recording_mirrors_robot_failure_status(tmp_path):
    dispatch = _dispatch_payload()

    captured: dict = {}
    published, status, error = publish_trial_recording(
        dispatch,
        finish_url=FINISH_URL,
        run_status="failed",
        video_key=VIDEO_KEY,
        hdf5_key=HDF5_KEY,
        hdf5_path=None,
        error={"code": "failed", "message": "trial blew up"},
        upload_s3=False,
        webhook_secret="secret",
        webhook_opener=_capturing_opener(captured),
    )

    # The returned status mirrors the robot trial outcome (already persisted on
    # web from /start); the publish webhook itself stays a publish-phase callback.
    assert status == "failed"
    assert error == {"code": "failed", "message": "trial blew up"}
    assert captured["body"]["phase"] == "publish"
    assert captured["body"]["status"] == "done"
    assert captured["body"]["artifact"]["publish_status"] == "done"


def test_publish_trial_recording_upload_error_sends_publish_failed_webhook(
    tmp_path, monkeypatch
):
    dispatch = _dispatch_payload()
    hdf5_path = tmp_path / "recording.hdf5"
    hdf5_path.write_bytes(b"fake hdf5")

    def fake_encode(_hdf5_path: str) -> Path:
        mp4 = tmp_path / "recording.mp4"
        mp4.write_bytes(b"fake mp4")
        return mp4

    monkeypatch.setattr(
        "robodojo.publish.pipeline._encode_trial_video", fake_encode
    )
    monkeypatch.setattr("robodojo.publish.s3.UPLOAD_RETRY_BACKOFF_S", (0, 0, 0))

    attempts: list[str] = []

    def boom(key: str, path: Path, content_type: str | None) -> None:
        attempts.append(key)
        raise OSError("tos unreachable")

    captured: dict = {}
    published, status, error = publish_trial_recording(
        dispatch,
        finish_url=FINISH_URL,
        run_status="done",
        video_key=VIDEO_KEY,
        hdf5_key=HDF5_KEY,
        hdf5_path=str(hdf5_path),
        s3_client=object(),
        upload_file=boom,
        webhook_secret="secret",
        webhook_opener=_capturing_opener(captured),
    )

    # The upload is retried before giving up.
    assert len(attempts) == 3
    # A failed upload never turns a successful robot trial into a failed one.
    assert status == "completed"
    assert error is None
    assert published["s3"]["error"]
    assert captured["body"]["phase"] == "publish"
    assert captured["body"]["status"] == "failed"
    assert captured["body"]["artifact"]["publish_status"] == "failed"
    # No video/hdf5 key is reported when the upload did not succeed.
    assert "video_s3_key" not in captured["body"]["artifact"]
    assert "error" in captured["body"]


def test_publish_trial_recording_retries_upload_then_succeeds(tmp_path, monkeypatch):
    dispatch = _dispatch_payload()
    hdf5_path = tmp_path / "recording.hdf5"
    hdf5_path.write_bytes(b"fake hdf5")

    def fake_encode(_hdf5_path: str) -> Path:
        mp4 = tmp_path / "recording.mp4"
        mp4.write_bytes(b"fake mp4")
        return mp4

    monkeypatch.setattr(
        "robodojo.publish.pipeline._encode_trial_video", fake_encode
    )
    monkeypatch.setattr("robodojo.publish.s3.UPLOAD_RETRY_BACKOFF_S", (0, 0, 0))

    attempts: list[str] = []

    def flaky_upload(key: str, path: Path, content_type: str | None) -> None:
        attempts.append(key)
        # Fail the first attempt for the video key, succeed afterwards.
        if attempts.count(VIDEO_KEY) == 1 and key == VIDEO_KEY:
            raise OSError("transient")

    captured: dict = {}
    published, status, error = publish_trial_recording(
        dispatch,
        finish_url=FINISH_URL,
        run_status="done",
        video_key=VIDEO_KEY,
        hdf5_key=HDF5_KEY,
        hdf5_path=str(hdf5_path),
        s3_client=object(),
        upload_file=flaky_upload,
        webhook_secret="secret",
        webhook_opener=_capturing_opener(captured),
    )

    assert status == "completed"
    assert published["s3"]["video_s3_key"] == VIDEO_KEY
    assert published["s3"]["hdf5_s3_key"] == HDF5_KEY
    assert captured["body"]["status"] == "done"
    assert captured["body"]["artifact"]["publish_status"] == "done"


def test_publish_trial_recording_falls_back_to_env_bucket(tmp_path, monkeypatch):
    dispatch = DispatchPayload.model_validate(
        platform_dispatch(
            artifact={"bucket": "", "prefix": "evaluations/eval-1/"},
            callback={"hmac_secret_ref": "ROBODOJO_WEBHOOK_SECRET"},
            evaluation_plan={
                "repeat_count": 1,
                "trials": [
                    {
                        "trial_id": "case-1-r01",
                        "action_case_id": "case-1",
                        "trial_index": 1,
                        "finish_url": FINISH_URL,
                    }
                ],
            },
        )
    )
    hdf5_path = tmp_path / "recording.hdf5"
    hdf5_path.write_bytes(b"fake hdf5")
    monkeypatch.setenv("S3_BUCKET", "robodojo-artifacts")
    monkeypatch.setattr(
        "robodojo.publish.pipeline._encode_trial_video", lambda _hdf5_path: None
    )

    uploads: list[tuple[str, str]] = []

    def fake_upload(key: str, path: Path, content_type: str | None) -> None:
        uploads.append((key, path.name))

    published, status, error = publish_trial_recording(
        dispatch,
        finish_url=FINISH_URL,
        run_status="done",
        video_key=VIDEO_KEY,
        hdf5_key=HDF5_KEY,
        hdf5_path=str(hdf5_path),
        s3_client=object(),
        upload_file=fake_upload,
        notify_webhook=False,
    )

    assert status == "completed"
    assert error is None
    assert uploads == [(HDF5_KEY, "recording.hdf5")]
    assert published["s3"]["hdf5_s3_key"] == HDF5_KEY


def test_encode_trial_video_builds_merged_three_view(tmp_path, monkeypatch):
    hdf5_path = tmp_path / "recording.hdf5"
    hdf5_path.write_bytes(b"fake hdf5")
    calls: list[tuple[str, list[str], str, int]] = []

    def fake_merged(path, camera_keys, save_path, fps=30):
        calls.append((path, list(camera_keys), save_path, fps))
        Path(save_path).write_bytes(b"merged mp4")

    monkeypatch.setattr(
        "robot.utils.base.data_handler.vis_merged_camera_video",
        fake_merged,
    )

    from robodojo.publish.pipeline import _encode_trial_video

    result = _encode_trial_video(str(hdf5_path))

    assert result == tmp_path / "recording.mp4"
    assert calls == [
        (
            str(hdf5_path),
            ["cam_head", "cam_left_wrist", "cam_right_wrist"],
            str(tmp_path / "recording.mp4"),
            25,
        )
    ]
