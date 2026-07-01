"""Dispatch orchestration: plan expansion, execution, and status."""

from eval_station.dispatch.errors import normalize_execution_error
from eval_station.dispatch.executor import notify_trial_failure, run_dispatch
from eval_station.dispatch.planner import build_trial_runs, dispatch_for_trial
from eval_station.dispatch.status import (
    STATUS_COMPLETED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PLANNED,
)

__all__ = [
    "STATUS_COMPLETED",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_PLANNED",
    "build_trial_runs",
    "dispatch_for_trial",
    "normalize_execution_error",
    "notify_trial_failure",
    "run_dispatch",
]
