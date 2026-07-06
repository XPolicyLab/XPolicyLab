"""Encode a trial recording and publish it to S3 / finish webhook."""

from __future__ import annotations

import os
import urllib.error
from pathlib import Path
from typing import Any

from station.dispatch.errors import normalize_execution_error
from station.dispatch.status import STATUS_COMPLETED, STATUS_FAILED
from station.publish.s3 import (
    UploadFileFn,
    resolve_artifact_payload,
    upload_file_to_key,
)
from station.publish.webhook import WebhookDeliveryError, notify_finish_webhook
from station.schemas import DispatchPayload


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

_TRIAL_MERGED_CAMERA_KEYS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
_TRIAL_VIDEO_FPS = 25


def _encode_trial_video(hdf5_path: str) -> Path | None:
    """Encode a three-view merged mp4 from the trial HDF5 (head + both wrists).

    Runs inside the background publish worker together with S3 upload and the
    finish webhook. Returns None when encoding fails so HDF5 upload can still
    proceed without a ``video_s3_key``.
    """
    if not hdf5_path or not os.path.isfile(hdf5_path):
        return None
    try:
        from robot.utils.base.data_handler import vis_merged_camera_video

        video_out = Path(hdf5_path).with_suffix(".mp4")
        vis_merged_camera_video(
            hdf5_path,
            list(_TRIAL_MERGED_CAMERA_KEYS),
            str(video_out),
            fps=_TRIAL_VIDEO_FPS,
        )
        return video_out if video_out.is_file() else None
    except Exception:  # noqa: BLE001 - never let encoding break publishing
        return None


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
    """Encode a three-view trial mp4, upload mp4 + hdf5, then fire the publish webhook.

    The robot trial outcome is already persisted synchronously from the ``/start``
    response, so this runs entirely in the background publish worker: it synthesizes
    the merged mp4 from ``hdf5_path`` (head + both wrists), uploads it plus the HDF5
    (with retry), then delivers a ``phase=publish`` webhook that only patches the
    trial artifact. A failed upload reports ``publish_status=failed`` without
    failing the trial or the parent session, so the collector can keep evaluating.
    """
    artifact = resolve_artifact_payload(dispatch.artifact)
    bucket = artifact.bucket.strip() if artifact.bucket else ""
    endpoint_url = str(getattr(artifact, "endpoint_url", "") or "").strip() or None
    region_name = str(getattr(artifact, "region", "") or "").strip() or None

    published: dict[str, Any] = {}
    uploaded_video_key: str | None = None
    uploaded_hdf5_key: str | None = None
    publish_error: dict[str, Any] | None = None
    video_path: Path | None = None
    try:
        if upload_s3:
            s3_published: dict[str, Any] = {}
            try:
                video_path = _encode_trial_video(hdf5_path) if hdf5_path else None
                if video_path is not None:
                    upload_file_to_key(
                        video_path,
                        bucket=bucket,
                        key=video_key,
                        s3_client=s3_client,
                        upload_file=upload_file,
                        endpoint_url=endpoint_url,
                        region_name=region_name,
                    )
                    uploaded_video_key = video_key
                    s3_published["video_s3_key"] = video_key
                if hdf5_path and os.path.isfile(hdf5_path):
                    upload_file_to_key(
                        Path(hdf5_path),
                        bucket=bucket,
                        key=hdf5_key,
                        s3_client=s3_client,
                        upload_file=upload_file,
                        endpoint_url=endpoint_url,
                        region_name=region_name,
                    )
                    uploaded_hdf5_key = hdf5_key
                    s3_published["hdf5_s3_key"] = hdf5_key
            except PUBLISH_ERRORS as exc:
                publish_error = normalize_execution_error(exc)
                s3_published["error"] = publish_error["message"]
            published["s3"] = s3_published

        if notify_webhook:
            publish_status = STATUS_FAILED if publish_error else STATUS_COMPLETED
            try:
                webhook_result = notify_finish_webhook(
                    phase="publish",
                    status=publish_status,
                    finish_url=finish_url,
                    metrics={},
                    artifact=artifact,
                    hmac_secret_ref=dispatch.hmac_secret_ref,
                    error=publish_error,
                    secret=webhook_secret,
                    opener=webhook_opener,
                    video_key=uploaded_video_key,
                    hdf5_key=uploaded_hdf5_key,
                )
                published["webhook"] = {
                    "finish_url": webhook_result.finish_url,
                    "status_code": webhook_result.status_code,
                }
            except PUBLISH_ERRORS as webhook_exc:
                published["webhook_error"] = str(webhook_exc)

        # The returned status mirrors the robot trial outcome, not the upload:
        # a failed upload never turns a successful trial into a failed one.
        if run_status == STATUS_FAILED:
            return published, STATUS_FAILED, error
        return published, STATUS_COMPLETED, error
    finally:
        if video_path is not None and video_path.is_file():
            try:
                video_path.unlink()
            except OSError:
                pass
