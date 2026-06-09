"""RoboDojo evaluation runner CLI."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from robodojo.artifacts import ArtifactWriter
from robodojo.env_client import RoboDojoModelClient
from robodojo.protocol.exceptions import WsError
from robodojo.s3_upload import UploadFileFn, upload_artifact_directory
from robodojo.schemas import DispatchPayload
from robodojo.serialization import to_jsonable
from robodojo.webhook import WebhookDeliveryError, notify_finish_webhook

STATUS_PLANNED = "planned"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_DONE = "done"


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


def normalize_execution_error(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, WsError):
        error: dict[str, Any] = {"code": exc.code.value, "message": exc.message}
        if exc.details:
            error["details"] = exc.details
        return error
    return {"code": "internal", "message": str(exc)}


def _dispatch_for_trial(dispatch: DispatchPayload, trial_index: int) -> DispatchPayload:
    if not dispatch.artifact.prefix:
        return dispatch
    base_prefix = dispatch.artifact.prefix.rstrip("/")
    return dispatch.model_copy(
        update={
            "artifact": dispatch.artifact.model_copy(
                update={"prefix": f"{base_prefix}/trials/{trial_index}/"}
            )
        }
    )


def notify_trial_failure(
    dispatch: DispatchPayload,
    *,
    trial_index: int,
    error: dict[str, Any],
    webhook_secret: str | None = None,
    webhook_opener: Any | None = None,
) -> dict[str, Any]:
    trial = next(
        (
            planned_trial
            for planned_trial in dispatch.evaluation_plan.trials
            if planned_trial.trial_index == trial_index
        ),
        None,
    )
    if trial is None or not trial.finish_url:
        raise ValueError(f"finish_url not found for trial_index {trial_index}")

    artifact = _dispatch_for_trial(dispatch, trial_index).artifact

    metrics: dict[str, Any] = {"summary": {}}
    if trial.trial_id:
        metrics["trials"] = [{"trial_id": trial.trial_id}]

    webhook_result = notify_finish_webhook(
        status=STATUS_FAILED,
        finish_url=trial.finish_url,
        metrics={"summary": {}},
        artifact=artifact,
        hmac_secret_ref=dispatch.hmac_secret_ref,
        error=error,
        secret=webhook_secret,
        opener=webhook_opener,
    )
    return {
        "finish_url": webhook_result.finish_url,
        "status_code": webhook_result.status_code,
        "emergency": True,
    }


def run_policy_trial(
    policy_server_url: str,
    evaluation_id: str,
    trial_run: dict[str, Any],
) -> dict[str, Any]:
    trial_id = str(trial_run["trial_id"])
    action_case_id = str(trial_run["action_case_id"])
    case_meta = dict(trial_run.get("case_meta") or {})
    obs = {
        "state": {},
        "instruction": case_meta.get("instruction", ""),
        "additional_info": {
            "trial_id": trial_id,
            "action_case_id": action_case_id,
        },
    }

    with RoboDojoModelClient(
        url=policy_server_url,
        evaluation_id=evaluation_id,
        trial_id=trial_id,
        action_case_id=action_case_id,
    ) as client:
        client.call(func_name="prepare_case", obs=case_meta)
        client.call(func_name="reset")
        actions = client.call(func_name="get_action", obs=obs)
        client.call(func_name="trial_end", obs={"result": "success", "steps": 1})

    return {"trial_id": trial_id, "actions": actions}


def build_trial_runs(
    dispatch: DispatchPayload, evaluation_id: str
) -> list[dict[str, object]]:
    trial_runs: list[dict[str, object]] = []
    for trial in dispatch.evaluation_plan.trials:
        action_case_id = trial.action_case_id
        trial_id = trial.trial_id or (
            f"{evaluation_id}:{action_case_id}:t{trial.trial_index:02d}"
        )
        trial_dump = trial.model_dump()
        case_meta = {
            key: value
            for key, value in trial_dump.items()
            if key not in {"trial_id", "repeat_index", "finish_url"}
            and value is not None
        }
        trial_runs.append(
            {
                "trial_id": str(trial_id),
                "action_case_id": action_case_id,
                "trial_index": trial.trial_index,
                "case_meta": case_meta,
                "finish_url": trial.finish_url,
            }
        )
    return trial_runs


def write_artifacts(
    dispatch: DispatchPayload,
    trial_run: dict[str, object],
    artifact_dir: Path,
    *,
    evaluation_id: str,
    run_status: str = STATUS_PLANNED,
    policy_result: dict[str, Any] | None = None,
    error_summary: str | None = None,
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
                    "steps": 1,
                    "action_count": action_count,
                },
            )
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
) -> dict[str, Any]:
    published: dict[str, Any] = {}

    if upload_s3:
        upload_result = upload_artifact_directory(
            Path(artifact_paths["artifact_dir"]),
            dispatch.artifact,
            s3_client=s3_client,
            upload_file=upload_file,
        )
        published["s3"] = {
            "bucket": upload_result.bucket,
            "prefix": upload_result.prefix,
            "manifest_s3_key": upload_result.manifest_s3_key,
            "uploaded_count": len(upload_result.uploaded_keys),
        }

    if notify_webhook:
        metrics = json.loads(
            Path(artifact_paths["metrics"]).read_text(encoding="utf-8")
        )
        webhook_result = notify_finish_webhook(
            status=run_status,
            finish_url=finish_url,
            metrics=metrics,
            artifact=dispatch.artifact,
            hmac_secret_ref=dispatch.hmac_secret_ref,
            error=error,
            secret=webhook_secret,
            opener=webhook_opener,
        )
        published["webhook"] = {
            "finish_url": webhook_result.finish_url,
            "status_code": webhook_result.status_code,
        }

    return published


def _publish_dispatch_artifacts(
    dispatch: DispatchPayload,
    artifact_paths: dict[str, str],
    *,
    run_status: str,
    upload_s3: bool,
    notify_webhook: bool,
    finish_url: str = "",
    error: dict[str, Any] | None = None,
    webhook_secret: str | None = None,
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
                    )
                )
            except PUBLISH_ERRORS as webhook_exc:
                failure_published["webhook_error"] = str(webhook_exc)
        return failure_published, STATUS_FAILED, publish_error


def _build_dispatch_summary(
    dispatch: DispatchPayload,
    *,
    evaluation_id: str,
    trial_run: dict[str, object],
    trial_index: int,
    run_status: str,
    policy_result: dict[str, Any] | None,
    error: dict[str, Any] | None,
    artifact_paths: dict[str, str] | None,
    published: dict[str, Any] | None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "evaluation_id": evaluation_id,
        "policy_server_url": dispatch.policy_server_url,
        "task_id": dispatch.task_id,
        "repeat_count": dispatch.evaluation_plan.repeat_count,
        "trial_count": len(dispatch.evaluation_plan.trials),
        "planned_trial_runs": 1,
        "trial_runs": [trial_run],
        "status": run_status,
        "trial_index": trial_index,
    }
    if policy_result is not None:
        summary["policy_results"] = [to_jsonable(policy_result)]
    if error is not None:
        summary["error_summary"] = error["message"]
        summary["error"] = error
    if artifact_paths is not None:
        summary["artifacts"] = artifact_paths
    if published is not None:
        summary["published"] = published
    return summary


def _fail_dispatch(
    dispatch: DispatchPayload,
    trial_run: dict[str, object],
    *,
    evaluation_id: str,
    trial_index: int,
    exc: BaseException,
    artifact_dir: Path | None,
    upload_s3: bool,
    notify_webhook: bool,
    webhook_secret: str | None = None,
) -> tuple[int, dict[str, object]]:
    error = normalize_execution_error(exc)
    run_status = STATUS_FAILED
    run_dispatch_payload = _dispatch_for_trial(dispatch, trial_index)

    artifact_paths: dict[str, str] | None = None
    published: dict[str, Any] | None = None
    if artifact_dir is not None:
        artifact_paths = write_artifacts(
            run_dispatch_payload,
            trial_run,
            artifact_dir,
            evaluation_id=evaluation_id,
            run_status=STATUS_FAILED,
            error_summary=error["message"],
        )
        if upload_s3 or notify_webhook:
            published, publish_status, publish_error = _publish_dispatch_artifacts(
                run_dispatch_payload,
                artifact_paths,
                run_status=run_status,
                upload_s3=upload_s3,
                notify_webhook=notify_webhook,
                finish_url=str(trial_run["finish_url"]),
                error=error,
                webhook_secret=webhook_secret,
            )
            if publish_status == STATUS_FAILED:
                error = publish_error

    summary = _build_dispatch_summary(
        dispatch,
        evaluation_id=evaluation_id,
        trial_run=trial_run,
        trial_index=trial_index,
        run_status=run_status,
        policy_result=None,
        error=error,
        artifact_paths=artifact_paths,
        published=published,
    )
    return 1, summary


def _execute_dispatch(
    dispatch: DispatchPayload,
    trial_run: dict[str, object],
    *,
    evaluation_id: str,
    trial_index: int,
    artifact_dir: Path | None,
    upload_s3: bool,
    notify_webhook: bool,
    run_policy_trials: bool,
    webhook_secret: str | None = None,
) -> tuple[int, dict[str, object]]:
    run_dispatch_payload = _dispatch_for_trial(dispatch, trial_index)

    artifact_paths: dict[str, str] | None = None
    published: dict[str, Any] | None = None
    run_status = STATUS_PLANNED
    error: dict[str, Any] | None = None
    policy_result: dict[str, Any] | None = None

    if run_policy_trials:
        run_status = STATUS_DONE
        try:
            policy_result = run_policy_trial(
                policy_server_url=dispatch.policy_server_url,
                evaluation_id=evaluation_id,
                trial_run=trial_run,
            )
        except Exception as exc:
            run_status = STATUS_FAILED
            error = normalize_execution_error(exc)

    if artifact_dir is not None:
        if notify_webhook and not run_policy_trials:
            raise ValueError("notify_webhook requires run_policy_trials")
        artifact_paths = write_artifacts(
            run_dispatch_payload,
            trial_run,
            artifact_dir,
            evaluation_id=evaluation_id,
            run_status=run_status,
            policy_result=policy_result,
            error_summary=error["message"] if error else None,
        )
        if upload_s3 or notify_webhook:
            published, publish_status, publish_error = _publish_dispatch_artifacts(
                run_dispatch_payload,
                artifact_paths,
                run_status=run_status,
                upload_s3=upload_s3,
                notify_webhook=notify_webhook,
                finish_url=str(trial_run["finish_url"]),
                error=error,
                webhook_secret=webhook_secret,
            )
            if publish_status == STATUS_FAILED:
                run_status = STATUS_FAILED
                error = publish_error
            elif run_status != STATUS_FAILED:
                run_status = STATUS_COMPLETED

    summary = _build_dispatch_summary(
        dispatch,
        evaluation_id=evaluation_id,
        trial_run=trial_run,
        trial_index=trial_index,
        run_status=run_status,
        policy_result=policy_result,
        error=error,
        artifact_paths=artifact_paths,
        published=published,
    )
    return (1 if run_status == STATUS_FAILED else 0), summary


def run_dispatch(
    dispatch: DispatchPayload,
    trial_index: int,
    evaluation_id: str,
    artifact_dir: Path | None = None,
    upload_s3: bool = True,
    notify_webhook: bool = True,
    run_policy_trials: bool = False,
    webhook_secret: str | None = None,
) -> tuple[int, dict[str, object]]:
    trial_run = next(
        (
            run
            for run in build_trial_runs(dispatch, evaluation_id=evaluation_id)
            if run["trial_index"] == trial_index
        ),
        None,
    )
    if trial_run is None:
        raise ValueError(f"trial_index {trial_index} not found in dispatch plan")

    try:
        return _execute_dispatch(
            dispatch=dispatch,
            trial_run=trial_run,
            evaluation_id=evaluation_id,
            trial_index=trial_index,
            artifact_dir=artifact_dir,
            upload_s3=upload_s3,
            notify_webhook=notify_webhook,
            run_policy_trials=run_policy_trials,
            webhook_secret=webhook_secret,
        )
    except ValueError:
        raise
    except Exception as exc:
        return _fail_dispatch(
            dispatch,
            trial_run,
            evaluation_id=evaluation_id,
            trial_index=trial_index,
            exc=exc,
            artifact_dir=artifact_dir,
            upload_s3=upload_s3,
            notify_webhook=notify_webhook,
            webhook_secret=webhook_secret,
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dispatch-payload",
        required=True,
        help="Path to dispatch JSON; use '-' to read from stdin",
    )
    parser.add_argument(
        "--evaluation-id",
        required=True,
        help="Evaluation id supplied by the control-plane route",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Directory for manifest.json, metrics.json, events.jsonl, videos/, logs/",
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip uploading artifacts to S3",
    )
    parser.add_argument(
        "--no-webhook",
        action="store_true",
        help="Skip finish webhook callback",
    )
    parser.add_argument(
        "--run-policy-trials",
        action="store_true",
        help="Connect to policy_server and run prepare/reset/infer/trial_end per trial",
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        required=True,
        help="Trial index to run from the dispatch plan",
    )
    args = parser.parse_args(argv)
    if args.artifact_dir and not args.run_policy_trials and not args.no_webhook:
        parser.error("--run-policy-trials is required unless --no-webhook is set")

    if args.dispatch_payload == "-":
        text = (stdin or sys.stdin).read()
    else:
        with open(args.dispatch_payload, "r", encoding="utf-8") as f:
            text = f.read()

    dispatch = DispatchPayload.model_validate_json(text)
    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id=args.evaluation_id,
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        upload_s3=not args.no_s3,
        notify_webhook=not args.no_webhook,
        run_policy_trials=args.run_policy_trials,
        trial_index=args.trial_index,
    )

    out = stdout or sys.stdout
    json.dump(summary, out, sort_keys=True)
    out.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
