"""Run a single dispatch trial: policy execution, artifacts, and publish."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from robodojo.dispatch.errors import normalize_execution_error
from robodojo.dispatch.planner import build_trial_runs, dispatch_for_trial
from robodojo.dispatch.status import (
    STATUS_COMPLETED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PLANNED,
)
import robodojo.publish.pipeline as publish_pipeline
from robodojo.publish.webhook import notify_finish_webhook
from robodojo.schemas import DispatchPayload
from robodojo.serialization import to_jsonable
from robodojo.trial import run_policy_trial


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

    artifact = dispatch_for_trial(dispatch, trial_index).artifact

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
    run_dispatch_payload = dispatch_for_trial(dispatch, trial_index)

    artifact_paths: dict[str, str] | None = None
    published: dict[str, Any] | None = None
    if artifact_dir is not None:
        artifact_paths = publish_pipeline.write_artifacts(
            run_dispatch_payload,
            trial_run,
            artifact_dir,
            evaluation_id=evaluation_id,
            run_status=STATUS_FAILED,
            error_summary=error["message"],
        )
        if upload_s3 or notify_webhook:
            published, publish_status, publish_error = publish_pipeline.publish_dispatch_artifacts(
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
    eval_env: str | None = None,
    root_dir: str | None = None,
    sim_env_factory: str | None = None,
    episode_step_limit: int = 5,
) -> tuple[int, dict[str, object]]:
    run_dispatch_payload = dispatch_for_trial(dispatch, trial_index)

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
                dispatch=dispatch,
                eval_env=eval_env,
                root_dir=root_dir,
                sim_env_factory=sim_env_factory,
                episode_step_limit=episode_step_limit,
            )
        except Exception as exc:
            run_status = STATUS_FAILED
            error = normalize_execution_error(exc)

    if artifact_dir is not None:
        if notify_webhook and not run_policy_trials:
            raise ValueError("notify_webhook requires run_policy_trials")
        artifact_paths = publish_pipeline.write_artifacts(
            run_dispatch_payload,
            trial_run,
            artifact_dir,
            evaluation_id=evaluation_id,
            run_status=run_status,
            policy_result=policy_result,
            error_summary=error["message"] if error else None,
        )
        if upload_s3 or notify_webhook:
            published, publish_status, publish_error = publish_pipeline.publish_dispatch_artifacts(
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
    eval_env: str | None = None,
    root_dir: str | None = None,
    sim_env_factory: str | None = None,
    episode_step_limit: int = 5,
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
            eval_env=eval_env,
            root_dir=root_dir,
            sim_env_factory=sim_env_factory,
            episode_step_limit=episode_step_limit,
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


def capture_job_result(
    *,
    evaluation_id: str,
    trial_index: int,
    dispatch: DispatchPayload,
    notify_webhook: bool,
    webhook_secret: str | None,
    run: Callable[[], tuple[int, dict[str, object]]],
) -> tuple[int, dict[str, object]]:
    try:
        return run()
    except ValueError:
        raise
    except Exception as exc:
        error = normalize_execution_error(exc)
        summary: dict[str, object] = {
            "evaluation_id": evaluation_id,
            "trial_index": trial_index,
            "status": STATUS_FAILED,
            "error_summary": error["message"],
            "error": error,
        }
        if notify_webhook:
            try:
                summary["published"] = {
                    "webhook": notify_trial_failure(
                        dispatch,
                        trial_index=trial_index,
                        error=error,
                        webhook_secret=webhook_secret,
                    )
                }
            except Exception as webhook_exc:
                summary["webhook_error"] = str(webhook_exc)
        return 1, summary
