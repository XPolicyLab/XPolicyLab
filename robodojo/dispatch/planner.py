"""Expand dispatch plans into per-trial run records."""

from __future__ import annotations

from robodojo.schemas import DispatchPayload


def dispatch_for_trial(dispatch: DispatchPayload, trial_index: int) -> DispatchPayload:
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
