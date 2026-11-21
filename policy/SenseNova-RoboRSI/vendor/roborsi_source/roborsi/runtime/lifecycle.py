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

"""Trace ownership at the existing runner and toolkit lifecycle boundary."""

from __future__ import annotations

import sys
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from roborsi.dashboard.events import DashboardEventSink, TraceUpdatedEvent
from roborsi.runtime.config import RuntimeConfig
from roborsi.runtime.events import RuntimeEvent
from roborsi.runtime.trace import TraceRecorder, exception_status


class _DashboardTraceSink:
    def __init__(self, output_dir: Path, events: DashboardEventSink) -> None:
        self.output_dir = output_dir
        self.events = events

    def emit(self, event: RuntimeEvent) -> None:
        self.events.emit(TraceUpdatedEvent(self.output_dir, event.run_id, event.seq))


def create_trace(
    output_dir: str | Path,
    runtime: RuntimeConfig | None,
    events: DashboardEventSink,
    *,
    toolkit: Any = None,
) -> TraceRecorder | None:
    """Start recording when configured and bind artifact collection to state."""
    if runtime is None or runtime.trace.mode == "off":
        return None
    output = Path(output_dir)
    recorder = TraceRecorder(
        output, runtime.trace, sinks=[_DashboardTraceSink(output, events)]
    )
    if toolkit is not None:
        toolkit.capture_video = runtime.trace.capture_video
        state = getattr(toolkit, "state", None)
        if state is not None:
            state.trace_recorder = recorder
    return recorder


def activate_trace(
    recorder: TraceRecorder | None,
) -> AbstractContextManager[TraceRecorder | None]:
    """Activate a configured recorder, leaving unconfigured runs unchanged."""
    return recorder.activate() if recorder is not None else nullcontext()


def finish_trace(
    recorder: TraceRecorder | None,
    *,
    error: BaseException | str | None = None,
    cancelled: bool = False,
) -> None:
    """Close after toolkit/media cleanup with the runner's final outcome."""
    if recorder is None:
        return
    from roborsi.evaluation.trajectory import TrajectoryReader

    try:
        reader = TrajectoryReader(recorder.output_dir)
        projection = reader.update()
        published = {
            event.get("payload", {}).get("artifact_ref")
            for event in projection.events
            if event["type"] in {"artifact_written", "artifact_failed"}
        }
        for media in reader.media():
            # Files predating trace activation still need a completion event.
            # Preserve emitted failures even if an older file remains on disk,
            # and keep the original tool/turn association on published media.
            if media["artifact_ref"] in published or media.get("status") == "failed":
                continue
            recorder.record_artifact(recorder.output_dir / media["artifact_ref"])
    except (OSError, ValueError) as exc:
        recorder.emit("artifact_failed", {"error": str(exc)})
    status = "cancelled" if cancelled else recorder.terminal_status
    if error is not None:
        status = (
            exception_status(error)
            if isinstance(error, BaseException)
            else ("timeout" if "timed out" in str(error).lower() else "error")
        )
    recorder.close(status=status, error=error)


def close_toolkit_trace(
    toolkit: Any,
    recorder: TraceRecorder | None,
    *,
    error: BaseException | str | None = None,
    cancelled: bool = False,
) -> None:
    """Drain toolkit-owned media before finishing a trace, including failures."""
    error = error or sys.exc_info()[1]
    try:
        toolkit.close()
    except BaseException as exc:
        finish_trace(recorder, error=exc)
        raise
    else:
        finish_trace(recorder, error=error, cancelled=cancelled)
