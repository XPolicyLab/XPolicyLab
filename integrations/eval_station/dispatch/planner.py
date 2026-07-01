"""Expand dispatch plans into per-trial run records."""

from __future__ import annotations

from eval_station.schemas import DispatchPayload


def dispatch_for_trial(dispatch: DispatchPayload, trial_index: int) -> DispatchPayload:
    """Return dispatch for a single trial run.

    Local artifact dirs already include ``trials/{trial_index}/``; the S3/TOS
    delivery prefix from dispatch must stay flat (``.../trial_{index}.mp4``).
    """
    _ = trial_index
    return dispatch


def build_trial_runs(
    dispatch: DispatchPayload, evaluation_id: str
) -> list[dict[str, object]]:
    trial_runs: list[dict[str, object]] = []
    task = dispatch.evaluation_plan.task
    env_cfg_type = task.env_cfg_type if task is not None else ""
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
                "env_cfg_type": env_cfg_type,
                "finish_url": trial.finish_url,
            }
        )
    return trial_runs


def trial_run_for_index(
    dispatch: DispatchPayload,
    *,
    evaluation_id: str,
    trial_index: int,
) -> dict[str, object]:
    for trial_run in build_trial_runs(dispatch, evaluation_id=evaluation_id):
        if trial_run["trial_index"] == trial_index:
            return trial_run
    raise ValueError(f"trial_index {trial_index} not found in dispatch plan")
