# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Case state kept separately from result artifacts, with live worker leases."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from roborsi.evaluation.messages import (
    EvaluationResult,
    Provenance,
    RoboRSIMessage,
    ServiceError,
    Task,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class CaseStatus(RoboRSIMessage):
    """Queryable execution state; completion does not imply task success."""

    run_id: str
    case_id: str
    episode_id: str | None = None
    state: Literal[
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "interrupted",
        "unknown",
    ]
    updated_at: str
    result_published: bool = False
    result_status: str | None = None
    error: ServiceError | None = None


class CasePublicationError(OSError):
    """Result output failed, but a scoreless error remains in the status journal."""

    def __init__(self, result: EvaluationResult) -> None:
        self.result = result
        super().__init__(result.error.message)


class CaseJournal:
    """Own small state records independently of a case's artifact directory.

    Identifiers are validated by ResultStore. Use a local POSIX filesystem for
    worker locks. A separate root may be placed on another writable volume.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, run_id: str, case_id: str) -> Path:
        return self.root / run_id / f"{case_id}.json"

    def read(self, run_id: str, case_id: str) -> dict:
        document = json.loads(self._path(run_id, case_id).read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {
            "status",
            "task",
            "provenance",
            "fallback_result",
        }:
            raise ValueError("invalid case status record")
        if any(
            not isinstance(document[key], dict)
            for key in ("status", "task", "provenance")
        ):
            raise ValueError("invalid case status record")
        status = CaseStatus.from_dict(document["status"])
        if (status.run_id, status.case_id) != (run_id, case_id):
            raise ValueError("case status identity differs")
        return document

    def _write(self, document: dict, *, exclusive: bool = False) -> None:
        status = CaseStatus.from_dict(document["status"])
        path = self._path(status.run_id, status.case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                os.chmod(temporary, 0o600)
                json.dump(document, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            if exclusive:
                os.link(temporary, path)
            else:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def create(
        self,
        run_id: str,
        case_id: str,
        task: Task,
        provenance: Provenance,
        episode_id: str | None,
    ) -> None:
        self._write(
            {
                "status": CaseStatus(
                    run_id=run_id,
                    case_id=case_id,
                    episode_id=episode_id,
                    state="pending",
                    updated_at=timestamp(),
                ).as_dict(),
                "task": task.as_dict(),
                "provenance": provenance.as_dict(),
                "fallback_result": None,
            },
            exclusive=True,
        )

    def status(self, run_id: str, case_id: str) -> CaseStatus:
        status = CaseStatus.from_dict(self.read(run_id, case_id)["status"])
        if status.state == "running" and not self.active(run_id, case_id):
            return status.model_copy(
                update={
                    "state": "interrupted",
                    "error": ServiceError(
                        code="infrastructure_error",
                        message="case worker exited before publishing a result",
                        retryable=True,
                        details={"stage": "worker.exit"},
                    ),
                }
            )
        return status

    def active(self, run_id: str, case_id: str) -> bool:
        import fcntl

        path = self._path(run_id, case_id).with_suffix(".lock")
        try:
            stream = path.open("rb")
        except FileNotFoundError:
            return False
        with stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream, fcntl.LOCK_UN)
        return False

    @contextmanager
    def execution(
        self,
        run_id: str,
        case_id: str,
        episode_id: str,
        *,
        published: Callable[[], bool],
    ) -> Iterator[None]:
        import fcntl

        path = self._path(run_id, case_id)
        with path.with_suffix(".lock").open("a+b") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("case already has an active worker") from exc
            try:
                document = self.read(run_id, case_id)
                status = CaseStatus.from_dict(document["status"])
                if status.state != "pending":
                    raise ValueError(
                        "case execution has already started; use a new run ID"
                    )
                document["status"] = status.model_copy(
                    update={
                        "episode_id": episode_id,
                        "state": "running",
                        "updated_at": timestamp(),
                    }
                ).as_dict()
                self._write(document)
                try:
                    yield
                except BaseException as exc:
                    if (
                        not published()
                        and self.read(run_id, case_id)["status"]["state"] == "running"
                    ):
                        self.failure(run_id, case_id, exc, "case.run")
                    raise
                else:
                    if (
                        not published()
                        and self.read(run_id, case_id)["status"]["state"] == "running"
                    ):
                        self.failure(
                            run_id,
                            case_id,
                            RuntimeError("case execution ended without a result"),
                            "case.run",
                        )
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def completed(self, run_id: str, case_id: str, result: EvaluationResult) -> None:
        document = self.read(run_id, case_id)
        state = (
            "cancelled"
            if result.status == "cancelled"
            else "failed"
            if result.error
            else "completed"
        )
        document["status"] = CaseStatus(
            run_id=run_id,
            case_id=case_id,
            episode_id=result.episode_id,
            state=state,
            updated_at=timestamp(),
            result_published=True,
            result_status=result.status,
            error=result.error,
        ).as_dict()
        document["fallback_result"] = None
        self._write(document)

    def failure(
        self,
        run_id: str,
        case_id: str,
        exc: BaseException,
        stage: str,
        *,
        previous: EvaluationResult | None = None,
    ) -> EvaluationResult | None:
        document = self.read(run_id, case_id)
        old = CaseStatus.from_dict(document["status"])
        error = ServiceError(
            code="infrastructure_error",
            message=str(exc) or type(exc).__name__,
            retryable=True,
            details={
                "stage": stage,
                "exception": type(exc).__name__,
                "previous_error": previous.error.as_dict()
                if previous and previous.error
                else None,
            },
        )
        episode_id = previous.episode_id if previous else old.episode_id
        result = (
            EvaluationResult(
                task=Task.from_dict(document["task"]),
                episode_id=episode_id,
                provenance=Provenance.from_dict(document["provenance"]),
                status="infrastructure_error",
                success=None,
                score=None,
                error=error,
            )
            if episode_id is not None
            else None
        )
        document["status"] = old.model_copy(
            update={
                "episode_id": episode_id,
                "state": "failed",
                "updated_at": timestamp(),
                "result_published": False,
                "result_status": "infrastructure_error",
                "error": error,
            }
        ).as_dict()
        document["fallback_result"] = result.as_dict() if result else None
        self._write(document)
        return result

    def fallback(self, run_id: str, case_id: str) -> EvaluationResult | None:
        document = self.read(run_id, case_id)
        value = document["fallback_result"]
        if value is None:
            return None
        status = CaseStatus.from_dict(document["status"])
        if (
            status.state != "failed"
            or status.result_published
            or status.result_status != "infrastructure_error"
            or not isinstance(value, dict)
        ):
            raise ValueError("invalid case failure record")
        result = EvaluationResult.from_dict(value)
        expected = EvaluationResult(
            task=Task.from_dict(document["task"]),
            episode_id=status.episode_id,
            provenance=Provenance.from_dict(document["provenance"]),
            status="infrastructure_error",
            success=None,
            score=None,
            error=status.error,
        )
        # This record cannot vouch for artifacts on the failed output volume.
        if result != expected:
            raise ValueError("invalid case failure record")
        return result
