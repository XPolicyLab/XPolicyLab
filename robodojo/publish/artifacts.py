"""RoboDojo evaluation artifact layout and writer."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robodojo.schemas import DispatchPayload, TrialRecord

MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.json"
EVENTS_NAME = "events.jsonl"
RUNNER_LOG_REL = "logs/runner.log"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactWriter:
    def __init__(
        self, root_dir: Path, *, evaluation_id: str, dispatch: DispatchPayload
    ):
        self.root_dir = root_dir
        self.evaluation_id = evaluation_id
        self.dispatch = dispatch
        self._trials: dict[str, TrialRecord] = {}
        self._run_started_at = _utc_now_iso()
        self._run_finished_at: str | None = None
        self._run_status = "running"
        self._run_error: str | None = None
        self._events_path = root_dir / EVENTS_NAME
        self._logger = logging.getLogger("robodojo.publish")
        self._log_handler: logging.Handler | None = None

    def setup(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "videos").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "logs").mkdir(parents=True, exist_ok=True)

        log_path = self.root_dir / RUNNER_LOG_REL
        self._log_handler = logging.FileHandler(log_path, encoding="utf-8")
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        self._logger.addHandler(self._log_handler)
        self._logger.setLevel(logging.INFO)

    def close(self) -> None:
        if self._log_handler is not None:
            self._logger.removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

    def emit_event(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "ts": _utc_now_iso(),
            "evaluation_id": self.evaluation_id,
            **fields,
        }
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
        self._logger.info("%s %s", event, fields)

    def register_trial(self, trial_run: dict[str, Any]) -> None:
        trial_id = trial_run["trial_id"]
        self._trials[trial_id] = TrialRecord(
            trial_id=trial_id,
            action_case_id=trial_run["action_case_id"],
            trial_index=trial_run["trial_index"],
            case_meta=trial_run["case_meta"],
        )
        self.emit_event(
            "trial_registered",
            trial_id=trial_id,
            action_case_id=trial_run["action_case_id"],
        )

    def record_trial_start(self, trial_id: str, **fields: Any) -> None:
        trial = self._trials[trial_id]
        trial.status = "running"
        trial.started_at = _utc_now_iso()
        self.emit_event("trial_started", trial_id=trial_id, **fields)

    def record_trial_end(
        self,
        trial_id: str,
        *,
        status: str,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
        **fields: Any,
    ) -> None:
        trial = self._trials[trial_id]
        trial.status = status
        trial.finished_at = _utc_now_iso()
        if metrics is not None:
            trial.metrics = metrics
        if error is not None:
            trial.error = error
        payload: dict[str, Any] = {
            "trial_id": trial_id,
            "status": status,
            **fields,
        }
        if error is not None:
            payload["error"] = error
        self.emit_event("trial_finished", **payload)

    def write_video_placeholder(self, trial_id: str) -> Path:
        path = self.root_dir / f"videos/{trial_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        return path

    def write_manifest(self) -> Path:
        manifest = {
            "evaluation_id": self.evaluation_id,
            "started_at": self._run_started_at,
            "finished_at": self._run_finished_at,
            "status": self._run_status,
            "policy_server_url": self.dispatch.policy_server_url,
            "task_id": self.dispatch.task_id,
            "evaluation_plan": self.dispatch.evaluation_plan.model_dump(),
            "artifact": self.dispatch.artifact.model_dump(),
            "callback": self.dispatch.callback.model_dump(),
            "trials": [t.to_manifest_entry() for t in self._trials.values()],
            "files": {
                "manifest": MANIFEST_NAME,
                "metrics": METRICS_NAME,
                "events": EVENTS_NAME,
                "logs": RUNNER_LOG_REL,
            },
        }
        if self._run_error is not None:
            manifest["error_summary"] = self._run_error
        path = self.root_dir / MANIFEST_NAME
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def write_metrics(self) -> Path:
        trials = list(self._trials.values())
        completed = sum(1 for t in trials if t.status == "completed")
        failed = sum(1 for t in trials if t.status == "failed")
        total = len(trials)
        executed = completed + failed
        success_rate = (completed / executed * 100.0) if executed else 0.0

        metrics_doc = {
            "evaluation_id": self.evaluation_id,
            "finished_at": self._run_finished_at,
            "summary": {
                "trial_count": total,
                "completed": completed,
                "failed": failed,
                "not_executed": sum(1 for t in trials if t.status == "not_executed"),
                "success_rate": success_rate,
            },
            "trials": [t.to_metrics_entry() for t in trials],
        }
        path = self.root_dir / METRICS_NAME
        path.write_text(
            json.dumps(metrics_doc, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def finalize(
        self, *, status: str, error_summary: str | None = None
    ) -> dict[str, str]:
        self._run_finished_at = _utc_now_iso()
        self._run_status = status
        self._run_error = error_summary
        finished_fields: dict[str, Any] = {"status": status}
        if error_summary is not None:
            finished_fields["error_summary"] = error_summary
        self.emit_event("run_finished", **finished_fields)
        manifest_path = self.write_manifest()
        metrics_path = self.write_metrics()
        return {
            "artifact_dir": str(self.root_dir),
            "manifest": str(manifest_path),
            "metrics": str(metrics_path),
            "events": str(self._events_path),
            "logs": str(self.root_dir / RUNNER_LOG_REL),
        }
