"""Shared HTTP daemon state for eval-station environment clients."""

from __future__ import annotations

import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from station.env_client.api import EnvClientBaselineConfig
from station.env_client.runner import TrialRunnerFn, make_dispatch_trial_runner
from station.env_client.trial_control import TrialControlRegistry
from station.schemas import DispatchPayload


@dataclass(frozen=True)
class EnvClientServerConfig:
    artifact_root: Path
    upload_s3: bool = True
    notify_webhook: bool = True
    run_policy_trials: bool = True
    webhook_secret: str | None = None


@dataclass
class EnvClientServerState:
    baseline: EnvClientBaselineConfig
    config: EnvClientServerConfig
    deploy_yml: str | None = None
    run_trial: Any | None = None
    last_trial_id: str | None = None
    dispatches: dict[str, DispatchPayload] = field(default_factory=dict)
    trial_control: TrialControlRegistry = field(default_factory=TrialControlRegistry)
    preview: Any | None = None
    persistent_runtime: Any | None = None
    _publish_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="publish",
        ),
        repr=False,
        compare=False,
    )

    def submit_publish(self, work: Callable[[], Any]) -> Future[Any]:
        """Queue trial publishing on the background worker (fire-and-forget)."""
        future = self._publish_executor.submit(work)
        future.add_done_callback(_log_publish_failure)
        return future

    def shutdown_publish(self) -> None:
        """Block until every queued publish task has drained."""
        self._publish_executor.shutdown(wait=True)

    def pause_preview_for_trial(self) -> None:
        if self.preview is not None:
            self.preview.pause()

    def resume_preview_if_idle(self) -> None:
        if self.preview is not None and not self.trial_control.has_active_trials():
            self.preview.resume_async()

    def trial_runner_with_stop(
        self,
        evaluation_id: str,
        trial_index: int,
    ) -> TrialRunnerFn | None:
        stop_event = self.trial_control.register_if_idle(evaluation_id, trial_index)
        if stop_event is None:
            return None
        try:
            return make_dispatch_trial_runner(
                self.baseline,
                run_trial=self.run_trial,
                stop_check_factory=lambda _: stop_event.is_set,
            )
        except Exception:
            self.trial_control.clear(evaluation_id, trial_index)
            raise

    def artifact_dir(self, evaluation_id: str, trial_index: int) -> Path:
        return (
            self.config.artifact_root
            / quote(evaluation_id, safe="")
            / "trials"
            / str(trial_index)
        )


def _log_publish_failure(future: Future[Any]) -> None:
    exc = future.exception()
    if exc is not None:
        print(
            "background trial publish failed: "
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            file=sys.stderr,
        )
