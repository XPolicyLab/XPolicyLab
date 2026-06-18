"""Encode a trial recording and publish it to S3 / finish webhook."""

from __future__ import annotations

import os
import urllib.error
from pathlib import Path
from typing import Any

from robodojo.dispatch.errors import normalize_execution_error
from robodojo.dispatch.status import STATUS_COMPLETED, STATUS_DONE, STATUS_FAILED
from robodojo.publish.s3 import UploadFileFn, upload_file_to_key
from robodojo.publish.webhook import WebhookDeliveryError, notify_finish_webhook
from robodojo.schemas import DispatchPayload


def _publish_exception_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [
        OSError,
        urllib.error.URLError,
        KeyError,
        WebhookDeliveryError,
        RuntimeError,
        ValueError,
        ConnectionError,
    ]
    try:
        from botocore.exceptions import BotoCoreError, ClientError

        types.extend([BotoCoreError, ClientError])
    except ImportError:
        pass
    return tuple(types)


PUBLISH_ERRORS = _publish_exception_types()


def _encode_trial_video(hdf5_path: str) -> Path | None:
    """Encode the recorded HDF5 into a temporary mp4 sibling and return its path.

    Best-effort: any failure (missing robot deps, bad camera key, corrupt HDF5)
    returns None so the trial still publishes its HDF5 without a video, and the
    caller reports no ``video_s3_key``.
    """
    if not hdf5_path or not os.path.isfile(hdf5_path):
        return None
    try:
        from robot.utils.base.data_handler import vis_video

        camera_key = os.environ.get("ROBODOJO_VIDEO_CAMERA_KEY", "cam_head")
        fps = int(os.environ.get("ROBODOJO_VIDEO_FPS", "25"))
        video_out = Path(hdf5_path).with_suffix(".mp4")
        vis_video(hdf5_path, camera_key, str(video_out), fps=fps)
        return video_out if video_out.is_file() else None
    except Exception:  # noqa: BLE001 - never let encoding break publishing
        return None


def _webhook_status(run_status: str) -> str:
    if run_status in {STATUS_DONE, STATUS_COMPLETED}:
        return STATUS_COMPLETED
    if run_status == STATUS_FAILED:
        return STATUS_FAILED
    return run_status


def publish_trial_recording(
    dispatch: DispatchPayload,
    *,
    finish_url: str,
    run_status: str,
    video_key: str,
    hdf5_key: str,
    hdf5_path: str | None = None,
    error: dict[str, Any] | None = None,
    upload_s3: bool = True,
    notify_webhook: bool = True,
    s3_client: Any | None = None,
    upload_file: UploadFileFn | None = None,
    webhook_secret: str | None = None,
    webhook_opener: Any | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    """Encode the trial mp4, upload mp4 + hdf5 as flat keys, then fire the finish webhook.

    The mp4 is transcoded from ``hdf5_path`` into a temporary sibling file (no
    artifact directory / metrics.json indirection) and uploaded to ``video_key``;
    the HDF5 is uploaded to ``hdf5_key``. Only a successfully uploaded mp4 reports
    ``video_s3_key`` back to Django, so scoring never receives a key pointing at a
    missing object. Any upload/webhook failure still delivers a ``failed`` webhook
    so the trial leaves RUNNING.
    """
    published: dict[str, Any] = {}
    uploaded_video_key: str | None = None
    uploaded_hdf5_key: str | None = None
    video_path: Path | None = None
    try:
        if upload_s3:
            s3_published: dict[str, Any] = {}
            video_path = _encode_trial_video(hdf5_path) if hdf5_path else None
            if video_path is not None:
                upload_file_to_key(
                    video_path,
                    bucket=dispatch.artifact.bucket,
                    key=video_key,
                    s3_client=s3_client,
                    upload_file=upload_file,
                )
                uploaded_video_key = video_key
                s3_published["video_s3_key"] = video_key
            if hdf5_path and os.path.isfile(hdf5_path):
                upload_file_to_key(
                    Path(hdf5_path),
                    bucket=dispatch.artifact.bucket,
                    key=hdf5_key,
                    s3_client=s3_client,
                    upload_file=upload_file,
                )
                uploaded_hdf5_key = hdf5_key
                s3_published["hdf5_s3_key"] = hdf5_key
            published["s3"] = s3_published

        if notify_webhook:
            webhook_result = notify_finish_webhook(
                status=_webhook_status(run_status),
                finish_url=finish_url,
                metrics={},
                artifact=dispatch.artifact,
                hmac_secret_ref=dispatch.hmac_secret_ref,
                error=error,
                secret=webhook_secret,
                opener=webhook_opener,
                video_key=uploaded_video_key,
                hdf5_key=uploaded_hdf5_key,
            )
            published["webhook"] = {
                "finish_url": webhook_result.finish_url,
                "status_code": webhook_result.status_code,
            }
        return published, STATUS_COMPLETED, error
    except PUBLISH_ERRORS as exc:
        publish_error = normalize_execution_error(exc)
        failure_published: dict[str, Any] = {"error": publish_error["message"]}
        if notify_webhook:
            try:
                webhook_result = notify_finish_webhook(
                    status=STATUS_FAILED,
                    finish_url=finish_url,
                    metrics={},
                    artifact=dispatch.artifact,
                    hmac_secret_ref=dispatch.hmac_secret_ref,
                    error=publish_error,
                    secret=webhook_secret,
                    opener=webhook_opener,
                    video_key=uploaded_video_key,
                    hdf5_key=uploaded_hdf5_key,
                )
                failure_published["webhook"] = {
                    "finish_url": webhook_result.finish_url,
                    "status_code": webhook_result.status_code,
                }
            except PUBLISH_ERRORS as webhook_exc:
                failure_published["webhook_error"] = str(webhook_exc)
        return failure_published, STATUS_FAILED, publish_error
    finally:
        if video_path is not None and video_path.is_file():
            try:
                video_path.unlink()
            except OSError:
                pass
