"""Run a single dispatch trial: policy execution and publish."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from eval_station.dispatch.errors import normalize_execution_error
from eval_station.dispatch.planner import dispatch_for_trial, trial_run_for_index
from eval_station.dispatch.status import (
    STATUS_COMPLETED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PLANNED,
)
from eval_station.env_client.runner import TrialRunnerFn
import eval_station.publish.pipeline as publish_pipeline
from eval_station.publish.s3 import normalize_s3_prefix, resolve_artifact_payload
from eval_station.publish.webhook import notify_finish_webhook
from eval_station.schemas import ArtifactPayload, DispatchPayload
from eval_station.serialization import to_jsonable

PublishSubmitFn = Callable[[Callable[[], Any]], Any]


def _resolved_artifact(dispatch: DispatchPayload) -> ArtifactPayload:
    return resolve_artifact_payload(dispatch.artifact)


def _trial_video_key(dispatch: DispatchPayload, trial_index: int) -> str:
    """Flat TOS delivery key for the trial video, under the web-provided base prefix.

    e.g. robodojo/{team}/{model}/{robot}/{task}/{eval_id}/trial_{index}.mp4
    """
    prefix = normalize_s3_prefix(_resolved_artifact(dispatch).prefix)
    return f"{prefix}trial_{trial_index}.mp4"


def _trial_hdf5_key(dispatch: DispatchPayload, trial_index: int) -> str:
    """Flat TOS delivery key for the trial HDF5 recording (sibling of the mp4)."""
    prefix = normalize_s3_prefix(_resolved_artifact(dispatch).prefix)
    return f"{prefix}trial_{trial_index}.hdf5"


def _hdf5_path_from_policy_result(policy_result: dict[str, Any] | None) -> str | None:
    if not policy_result:
        return None
    hdf5_path = policy_result.get("hdf5_path")
    return str(hdf5_path) if hdf5_path else None


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

    artifact = resolve_artifact_payload(
        dispatch_for_trial(dispatch, trial_index).artifact
    )

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


def _build_dispatch_summary(
    dispatch: DispatchPayload,
    *,
    evaluation_id: str,
    trial_run: dict[str, object],
    trial_index: int,
    run_status: str,
    policy_result: dict[str, Any] | None,
    error: dict[str, Any] | None,
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
    if published is not None:
        summary["published"] = published
    return summary


def _publish_after_trial(
    dispatch: DispatchPayload,
    run_dispatch_payload: DispatchPayload,
    *,
    trial_run: dict[str, object],
    trial_index: int,
    run_status: str,
    policy_result: dict[str, Any] | None,
    error: dict[str, Any] | None,
    upload_s3: bool,
    notify_webhook: bool,
    webhook_secret: str | None,
    publish_submit: PublishSubmitFn | None,
) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    if not (upload_s3 or notify_webhook):
        if run_status == STATUS_DONE:
            return None, STATUS_COMPLETED, error
        return None, run_status, error

    def do_publish() -> tuple[dict[str, Any], str, dict[str, Any] | None]:
        return publish_pipeline.publish_trial_recording(
            run_dispatch_payload,
            finish_url=str(trial_run["finish_url"]),
            run_status=run_status,
            video_key=_trial_video_key(dispatch, trial_index),
            hdf5_key=_trial_hdf5_key(dispatch, trial_index),
            hdf5_path=_hdf5_path_from_policy_result(policy_result),
            error=error,
            upload_s3=upload_s3,
            notify_webhook=notify_webhook,
            webhook_secret=webhook_secret,
        )

    if publish_submit is not None:
        publish_submit(do_publish)
        if run_status == STATUS_DONE:
            return {"async": True}, STATUS_COMPLETED, error
        return {"async": True}, run_status, error

    published, publish_status, publish_error = do_publish()
    if publish_status == STATUS_FAILED:
        return published, STATUS_FAILED, publish_error
    if run_status == STATUS_DONE:
        return published, STATUS_COMPLETED, error
    return published, run_status, error


def _fail_dispatch(
    dispatch: DispatchPayload,
    trial_run: dict[str, object],
    *,
    evaluation_id: str,
    trial_index: int,
    exc: BaseException,
    should_publish: bool,
    upload_s3: bool,
    notify_webhook: bool,
    webhook_secret: str | None = None,
    publish_submit: PublishSubmitFn | None = None,
) -> tuple[int, dict[str, object]]:
    error = normalize_execution_error(exc)
    run_status = STATUS_FAILED
    run_dispatch_payload = dispatch_for_trial(dispatch, trial_index)

    published: dict[str, Any] | None = None
    if should_publish:
        published, run_status, error = _publish_after_trial(
            dispatch,
            run_dispatch_payload,
            trial_run=trial_run,
            trial_index=trial_index,
            run_status=run_status,
            policy_result=None,
            error=error,
            upload_s3=upload_s3,
            notify_webhook=notify_webhook,
            webhook_secret=webhook_secret,
            publish_submit=publish_submit,
        )

    summary = _build_dispatch_summary(
        dispatch,
        evaluation_id=evaluation_id,
        trial_run=trial_run,
        trial_index=trial_index,
        run_status=run_status,
        policy_result=None,
        error=error,
        published=published,
    )
    return 1, summary


def _execute_dispatch(
    dispatch: DispatchPayload,
    trial_run: dict[str, object],
    *,
    evaluation_id: str,
    trial_index: int,
    should_publish: bool,
    upload_s3: bool,
    notify_webhook: bool,
    run_policy_trials: bool,
    webhook_secret: str | None = None,
    trial_runner: TrialRunnerFn | None = None,
    publish_submit: PublishSubmitFn | None = None,
) -> tuple[int, dict[str, object]]:
    run_dispatch_payload = dispatch_for_trial(dispatch, trial_index)

    published: dict[str, Any] | None = None
    run_status = STATUS_PLANNED
    error: dict[str, Any] | None = None
    policy_result: dict[str, Any] | None = None

    if run_policy_trials:
        if trial_runner is None:
            raise ValueError("trial_runner is required when run_policy_trials=True")
        run_status = STATUS_DONE
        try:
            policy_result = trial_runner(dispatch, dict(trial_run), evaluation_id)
        except Exception as exc:
            run_status = STATUS_FAILED
            error = normalize_execution_error(exc)

    if should_publish:
        if notify_webhook and not run_policy_trials:
            raise ValueError("notify_webhook requires run_policy_trials")
        published, run_status, error = _publish_after_trial(
            dispatch,
            run_dispatch_payload,
            trial_run=trial_run,
            trial_index=trial_index,
            run_status=run_status,
            policy_result=policy_result,
            error=error,
            upload_s3=upload_s3,
            notify_webhook=notify_webhook,
            webhook_secret=webhook_secret,
            publish_submit=publish_submit,
        )

    summary = _build_dispatch_summary(
        dispatch,
        evaluation_id=evaluation_id,
        trial_run=trial_run,
        trial_index=trial_index,
        run_status=run_status,
        policy_result=policy_result,
        error=error,
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
    trial_runner: TrialRunnerFn | None = None,
    publish_submit: PublishSubmitFn | None = None,
) -> tuple[int, dict[str, object]]:
    trial_run = trial_run_for_index(
        dispatch,
        evaluation_id=evaluation_id,
        trial_index=trial_index,
    )
    should_publish = artifact_dir is not None

    try:
        return _execute_dispatch(
            dispatch=dispatch,
            trial_run=trial_run,
            evaluation_id=evaluation_id,
            trial_index=trial_index,
            should_publish=should_publish,
            upload_s3=upload_s3,
            notify_webhook=notify_webhook,
            run_policy_trials=run_policy_trials,
            webhook_secret=webhook_secret,
            trial_runner=trial_runner,
            publish_submit=publish_submit,
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
            should_publish=should_publish,
            upload_s3=upload_s3,
            notify_webhook=notify_webhook,
            webhook_secret=webhook_secret,
            publish_submit=publish_submit,
        )
