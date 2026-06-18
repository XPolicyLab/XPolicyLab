"""Write evaluation artifacts and publish them to S3 / webhook."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
from pathlib import Path
from typing import Any

from robodojo.dispatch.errors import normalize_execution_error
from robodojo.dispatch.status import (
    STATUS_COMPLETED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PLANNED,
)
from robodojo.publish.artifacts import ArtifactWriter
from robodojo.publish.s3 import (
    UploadFileFn,
    normalize_s3_prefix,
    resolve_artifact_payload,
    upload_artifact_directory,
    upload_file_to_key,
)
from robodojo.publish.webhook import WebhookDeliveryError, notify_finish_webhook
from robodojo.schemas import DispatchPayload


def _trial_id_from_metrics(artifact_paths: dict[str, str]) -> str:
    metrics_path = artifact_paths.get("metrics")
    if not metrics_path:
        return ""
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    trials = metrics.get("trials")
    if isinstance(trials, list) and trials and isinstance(trials[0], dict):
        return str(trials[0].get("trial_id") or "")
    return ""


def _resolve_staged_artifact(
    artifact_paths: dict[str, str], subdir: str, suffix: str
) -> Path | None:
    """Locate a staged per-trial artifact ({subdir}/{trial_id}{suffix}) for flat TOS delivery."""
    trial_id = _trial_id_from_metrics(artifact_paths)
    if not trial_id:
        return None
    path = Path(artifact_paths["artifact_dir"]) / subdir / f"{trial_id}{suffix}"
    return path if path.is_file() else None


def _stage_trial_recording(
    writer: ArtifactWriter,
    artifact_dir: Path,
    trial_id: str,
    hdf5_path: str | None,
) -> bool:
    """Convert the recorded HDF5 to mp4 and stage both files under the artifact dir.

    Returns True if an mp4 was produced (so the caller skips the placeholder).
    Best-effort: any failure (missing robot deps, bad camera key, corrupt HDF5)
    falls back to the placeholder so trial publishing never breaks.
    """
    if not hdf5_path or not os.path.isfile(hdf5_path):
        return False
    try:
        from robot.utils.base.data_handler import vis_merged_camera_video, vis_video

        fps = int(os.environ.get("ROBODOJO_VIDEO_FPS", "25"))
        video_out = artifact_dir / "videos" / f"{trial_id}.mp4"
        video_out.parent.mkdir(parents=True, exist_ok=True)
        camera_keys_raw = os.environ.get(
            "ROBODOJO_VIDEO_CAMERA_KEYS",
            "cam_head,cam_left_wrist,cam_right_wrist",
        )
        camera_keys = [key.strip() for key in camera_keys_raw.split(",") if key.strip()]
        if len(camera_keys) >= 2:
            vis_merged_camera_video(hdf5_path, camera_keys, str(video_out), fps=fps)
        else:
            camera_key = camera_keys[0] if camera_keys else "cam_head"
            vis_video(hdf5_path, camera_key, str(video_out), fps=fps)

        hdf5_out = artifact_dir / "recordings" / f"{trial_id}.hdf5"
        hdf5_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hdf5_path, hdf5_out)

        if video_out.is_file():
            writer.emit_event("trial_recording_staged", trial_id=trial_id)
            return True
    except Exception as exc:  # noqa: BLE001 - never let encoding break publishing
        writer.emit_event(
            "trial_recording_failed", trial_id=trial_id, error=str(exc)
        )
    return False


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


def write_artifacts(
    dispatch: DispatchPayload,
    trial_run: dict[str, object],
    artifact_dir: Path,
    *,
    evaluation_id: str,
    run_status: str = STATUS_PLANNED,
    policy_result: dict[str, Any] | None = None,
    error_summary: str | None = None,
    stage_recordings: bool = True,
) -> dict[str, str]:
    writer = ArtifactWriter(
        artifact_dir,
        evaluation_id=evaluation_id,
        dispatch=dispatch,
    )
    writer.setup()
    try:
        writer.emit_event("run_started")
        writer.register_trial(trial_run)
        trial_id = str(trial_run["trial_id"])
        if policy_result is not None:
            actions = policy_result.get("actions")
            action_count = len(actions) if isinstance(actions, list) else None
            writer.record_trial_start(trial_id)
            writer.record_trial_end(
                trial_id,
                status=STATUS_COMPLETED,
                metrics={
                    "success": True,
                    "steps": policy_result.get("steps", 1),
                    "action_count": action_count,
                },
            )
        if stage_recordings:
            hdf5_path = (policy_result or {}).get("hdf5_path") if policy_result else None
            video_written = _stage_trial_recording(
                writer, artifact_dir, trial_id, hdf5_path
            )
            if not video_written:
                writer.write_video_placeholder(trial_id)
        return writer.finalize(status=run_status, error_summary=error_summary)
    finally:
        writer.close()


def publish_artifacts(
    dispatch: DispatchPayload,
    artifact_paths: dict[str, str],
    *,
    run_status: str,
    upload_s3: bool,
    notify_webhook: bool,
    error: dict[str, Any] | None = None,
    finish_url: str = "",
    s3_client: Any | None = None,
    upload_file: UploadFileFn | None = None,
    webhook_secret: str | None = None,
    webhook_opener: Any | None = None,
    video_key: str | None = None,
    hdf5_key: str | None = None,
) -> dict[str, Any]:
    published: dict[str, Any] = {}
    artifact = resolve_artifact_payload(dispatch.artifact)

    if upload_s3:
        flat_delivery = bool(video_key or hdf5_key)
        if not flat_delivery:
            upload_result = upload_artifact_directory(
                Path(artifact_paths["artifact_dir"]),
                artifact,
                s3_client=s3_client,
                upload_file=upload_file,
            )
            published["s3"] = {
                "bucket": upload_result.bucket,
                "prefix": upload_result.prefix,
                "manifest_s3_key": upload_result.manifest_s3_key,
                "uploaded_count": len(upload_result.uploaded_keys),
            }
        else:
            published["s3"] = {
                "bucket": artifact.bucket,
                "prefix": normalize_s3_prefix(artifact.prefix),
                "uploaded_count": 0,
            }

        def _deliver_flat(local_path: Path | None, key: str | None, result_field: str) -> None:
            if not key or local_path is None:
                return
            upload_file_to_key(
                local_path,
                bucket=artifact.bucket,
                key=key,
                s3_client=s3_client,
                upload_file=upload_file,
            )
            published["s3"][result_field] = key

        _deliver_flat(
            _resolve_staged_artifact(artifact_paths, "videos", ".mp4"), video_key, "video_s3_key"
        )
        _deliver_flat(
            _resolve_staged_artifact(artifact_paths, "recordings", ".hdf5"),
            hdf5_key,
            "hdf5_s3_key",
        )
        if flat_delivery:
            published["s3"]["uploaded_count"] = sum(
                1
                for field in ("video_s3_key", "hdf5_s3_key")
                if published["s3"].get(field)
            )

    if notify_webhook:
        metrics = json.loads(
            Path(artifact_paths["metrics"]).read_text(encoding="utf-8")
        )
        webhook_result = notify_finish_webhook(
            status=run_status,
            finish_url=finish_url,
            metrics=metrics,
            artifact=artifact,
            hmac_secret_ref=dispatch.hmac_secret_ref,
            error=error,
            secret=webhook_secret,
            opener=webhook_opener,
            video_key=video_key,
            hdf5_key=hdf5_key,
        )
        published["webhook"] = {
            "finish_url": webhook_result.finish_url,
            "status_code": webhook_result.status_code,
        }

    return published


def publish_dispatch_artifacts(
    dispatch: DispatchPayload,
    artifact_paths: dict[str, str],
    *,
    run_status: str,
    upload_s3: bool,
    notify_webhook: bool,
    finish_url: str = "",
    error: dict[str, Any] | None = None,
    webhook_secret: str | None = None,
    video_key: str | None = None,
    hdf5_key: str | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    published: dict[str, Any] = {}
    if run_status in {STATUS_DONE, STATUS_COMPLETED}:
        webhook_status = STATUS_COMPLETED
    elif run_status == STATUS_FAILED:
        webhook_status = STATUS_FAILED
    else:
        webhook_status = run_status
    try:
        if upload_s3:
            published.update(
                publish_artifacts(
                    dispatch,
                    artifact_paths,
                    run_status=webhook_status,
                    upload_s3=True,
                    notify_webhook=False,
                    video_key=video_key,
                    hdf5_key=hdf5_key,
                )
            )
        if notify_webhook:
            published.update(
                publish_artifacts(
                    dispatch,
                    artifact_paths,
                    run_status=webhook_status,
                    upload_s3=False,
                    notify_webhook=True,
                    finish_url=finish_url,
                    error=error,
                    webhook_secret=webhook_secret,
                    video_key=video_key,
                    hdf5_key=hdf5_key,
                )
            )
        return published, STATUS_COMPLETED, error
    except PUBLISH_ERRORS as exc:
        publish_error = normalize_execution_error(exc)
        failure_published: dict[str, Any] = {"error": publish_error["message"]}
        s3_info = published.get("s3")
        if isinstance(s3_info, dict) and s3_info.get("manifest_s3_key"):
            failure_published["s3"] = {
                "manifest_s3_key": str(s3_info["manifest_s3_key"])
            }
        if notify_webhook:
            try:
                failure_published.update(
                    publish_artifacts(
                        dispatch,
                        artifact_paths,
                        run_status=STATUS_FAILED,
                        upload_s3=False,
                        notify_webhook=True,
                        finish_url=finish_url,
                        error=publish_error,
                        webhook_secret=webhook_secret,
                        video_key=video_key,
                        hdf5_key=hdf5_key,
                    )
                )
            except PUBLISH_ERRORS as webhook_exc:
                failure_published["webhook_error"] = str(webhook_exc)
        return failure_published, STATUS_FAILED, publish_error
