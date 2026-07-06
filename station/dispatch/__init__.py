"""Dispatch orchestration: plan expansion, execution, and status."""

from station.dispatch.errors import normalize_execution_error
from station.dispatch.executor import notify_trial_failure, run_dispatch
from station.dispatch.planner import build_trial_runs, dispatch_for_trial
from station.dispatch.status import (
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
