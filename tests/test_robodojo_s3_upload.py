from pathlib import Path

from robodojo.publish.s3 import normalize_s3_prefix, upload_file_to_key


def test_normalize_s3_prefix():
    assert normalize_s3_prefix("eval-1/") == "eval-1/"
    assert normalize_s3_prefix("eval-1") == "eval-1/"
    assert normalize_s3_prefix("") == ""


def test_upload_file_to_key_uploads_explicit_key(tmp_path):
    local = tmp_path / "trial_1.mp4"
    local.write_bytes(b"fake mp4")

    uploads: list[tuple[str, Path, str | None]] = []

    def fake_upload(key: str, path: Path, content_type: str | None) -> None:
        uploads.append((key, path, content_type))

    key = upload_file_to_key(
        local,
        bucket="robodojo-artifacts",
        key="evaluations/eval-1/trial_1.mp4",
        s3_client=object(),
        upload_file=fake_upload,
    )

    assert key == "evaluations/eval-1/trial_1.mp4"
    assert uploads == [
        ("evaluations/eval-1/trial_1.mp4", local, "video/mp4")
    ]


def test_upload_file_to_key_requires_existing_file(tmp_path):
    missing = tmp_path / "nope.mp4"

    try:
        upload_file_to_key(
            missing,
            bucket="robodojo-artifacts",
            key="evaluations/eval-1/trial_1.mp4",
            s3_client=object(),
            upload_file=lambda *args: None,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for missing file")
