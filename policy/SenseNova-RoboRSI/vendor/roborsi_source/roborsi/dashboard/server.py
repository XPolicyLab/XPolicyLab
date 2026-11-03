# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dashboard HTTP server for the live monitor.

The FastAPI app runs on an OS-picked free port inside a daemon thread, so it can
sit alongside the agent run loop and keep serving after the run finishes, until
the process is stopped.

Routes mirror the fixed frontend contract in ``roborsi/dashboard/index.html``.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from roborsi.dashboard.interaction import (
    DashboardMessageConflictError,
    InteractionUnavailableError,
    UnknownDashboardMessageError,
)
from roborsi.dashboard.state import (
    DashboardState,
    PrimitiveArgumentError,
    PrimitiveConfigError,
)
from roborsi.evaluation.trajectory import TrajectoryReader
from roborsi.utils.daemon import pick_free_port
from roborsi.utils.logging import get_logger

logger = get_logger("dashboard_server")

if TYPE_CHECKING:
    from roborsi.llm.client import LLMConfig


class DashboardServer:
    """Threaded FastAPI server exposing one Dashboard Session."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        language: str = "en",
        state: DashboardState,
        planner: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self._state = state
        if llm_config is not None:
            from roborsi.planner.check import LlmCheckRequest

            model = LlmCheckRequest(llm_config=llm_config).resolved_model()
        self._planner_config = {"planner": planner, "model": model}
        # Server-side context the page cannot supply. The check keeps its own
        # short timeouts; the run's is never reused.
        self._llm_check_base_url = base_url
        self._llm_check_config = llm_config
        self._llm_check_lock = threading.Lock()
        self._trajectory_lock = threading.RLock()
        self._trajectory_reader: TrajectoryReader | None = None
        self._trajectory_identity: tuple[Path | None, str | None, int] | None = None
        dashboard_dir = Path(__file__).parent
        self._language = "zh-cn" if language == "zh-cn" else "en"
        self._index_html = (dashboard_dir / "index.html").read_text(encoding="utf-8")
        self._trajectory_html = (dashboard_dir / "trajectory.html").read_text(
            encoding="utf-8"
        )
        self._static_dir = dashboard_dir / "static"
        self._app = self._build_app()
        self._server: uvicorn.Server | None = None

    def start(self) -> str:
        """Launch uvicorn in a daemon thread; return the base URL once serving."""
        port = self.port or pick_free_port(self.host)
        config = uvicorn.Config(
            self._app, host=self.host, port=port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        threading.Thread(target=self._server.run, daemon=True).start()

        t0 = time.time()
        while not self._server.started and time.time() - t0 < 10:
            time.sleep(0.05)
        if not self._server.started:
            raise RuntimeError(f"dashboard server did not start on {self.host}:{port}")
        self.port = port
        return f"http://{self.host}:{port}"

    # -- routes ------------------------------------------------------------

    def _get_trajectory_reader(
        self, run_id: str | None, generation: int | None
    ) -> tuple[TrajectoryReader, tuple[Path | None, str | None, int]]:
        """Select a cached reader while its caller holds the trajectory lock."""
        identity = self._state.trajectory_source()
        directory, current_run, current_generation = identity
        if (run_id is not None and run_id != current_run) or (
            generation is not None and generation != current_generation
        ):
            raise HTTPException(
                status_code=409, detail="the selected trajectory has changed"
            )
        if directory is None or current_run is None:
            raise HTTPException(
                status_code=404, detail="no trajectory is available for this task"
            )
        if identity != self._trajectory_identity:
            self._trajectory_reader = TrajectoryReader(directory)
            self._trajectory_identity = identity
        assert self._trajectory_reader is not None
        return self._trajectory_reader, identity

    def _check_trajectory_identity(
        self, identity: tuple[Path | None, str | None, int]
    ) -> None:
        if self._state.trajectory_source() != identity:
            raise HTTPException(
                status_code=409, detail="the selected trajectory has changed"
            )

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="RoboRSI dashboard")
        app.mount(
            "/static",
            StaticFiles(directory=self._static_dir),
            name="dashboard-static",
        )

        @app.get("/")
        def index() -> HTMLResponse:
            html = self._index_html.replace(
                "__DASHBOARD_LANGUAGE__",
                self._language,
            )
            return HTMLResponse(html)

        @app.get("/trajectory")
        def trajectory() -> HTMLResponse:
            html = self._trajectory_html.replace(
                "__DASHBOARD_LANGUAGE__", self._language
            )
            return HTMLResponse(html)

        @app.get("/healthz")
        def healthz() -> JSONResponse:
            return JSONResponse({"ok": True})

        @app.get("/api/commands")
        def api_commands() -> JSONResponse:
            return JSONResponse(self._state.dashboard_spec)

        @app.get("/api/session/config")
        def api_session_config() -> JSONResponse:
            return JSONResponse(self._planner_config)

        @app.post("/api/llm/check")
        def api_llm_check(payload: dict[str, Any] = Body(default={})) -> JSONResponse:
            # Single-flight: the button is a diagnostic, not a load generator.
            if not self._llm_check_lock.acquire(blocking=False):
                return JSONResponse(
                    {"error": "a check is already running"}, status_code=409
                )
            try:
                # Imported here so the module's import cost stays unchanged.
                from roborsi.planner.check import LlmCheckRequest, check_llm

                configured = self._planner_config
                planner = str(
                    payload.get("planner") or configured.get("planner") or "api"
                )
                result = check_llm(
                    LlmCheckRequest(
                        planner=planner,
                        model=(payload.get("model") or configured.get("model") or None),
                        base_url=self._llm_check_base_url or None,
                        # A configured runtime model is authoritative; the browser
                        # receives only its identity and never supplies credentials.
                        llm_config=self._llm_check_config if planner == "api" else None,
                    )
                )
            finally:
                self._llm_check_lock.release()
            # A failed check is still a successful request; the body carries ok.
            return JSONResponse(result.as_dict())

        @app.post("/api/session/messages")
        def api_submit_message(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            try:
                self._state.submit_input(payload.get("text"))
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=422)
            except InteractionUnavailableError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            return JSONResponse({"ok": True}, status_code=202)

        @app.delete(
            "/api/session/messages/{message_id}",
        )
        def api_withdraw_message(
            message_id: str,
        ) -> JSONResponse:
            try:
                self._state.withdraw_message(message_id)
            except UnknownDashboardMessageError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except (DashboardMessageConflictError, InteractionUnavailableError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            return JSONResponse({"ok": True})

        @app.post("/api/session/interrupt")
        def api_interrupt() -> JSONResponse:
            try:
                result = self._state.request_interrupt()
            except InteractionUnavailableError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            if result == "noop":
                return JSONResponse(
                    {
                        "status": "noop",
                        "interrupt_requested": False,
                    }
                )
            return JSONResponse(
                {
                    "status": "requested",
                    "interrupt_requested": True,
                    "deduplicated": result == "duplicate",
                },
                status_code=202,
            )

        @app.get("/api/session/state")
        def api_session_state(timeline_since: int = 0) -> JSONResponse:
            return JSONResponse(
                self._state.session_detail(timeline_since=timeline_since)
            )

        @app.get("/api/session/trajectory")
        def api_trajectory(
            after_seq: int = Query(default=0, ge=0),
            run_id: str | None = None,
            generation: int | None = None,
        ) -> JSONResponse:
            with self._trajectory_lock:
                reader, identity = self._get_trajectory_reader(run_id, generation)
                try:
                    data = reader.index(after_seq)
                except (OSError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422, detail=f"cannot read trajectory: {exc}"
                    ) from exc
                self._check_trajectory_identity(identity)
                return JSONResponse(
                    {**data, "task_generation": identity[2]},
                    headers={"Cache-Control": "no-store"},
                )

        @app.get("/api/session/trajectory/turn/{turn_id}")
        def api_trajectory_turn(
            turn_id: str, run_id: str | None = None, generation: int | None = None
        ) -> JSONResponse:
            with self._trajectory_lock:
                reader, identity = self._get_trajectory_reader(run_id, generation)
                try:
                    reader.update()
                    data = reader.turn(turn_id)
                except KeyError as exc:
                    raise HTTPException(
                        status_code=404, detail="trajectory turn not found"
                    ) from exc
                except (OSError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422, detail=f"cannot read trajectory: {exc}"
                    ) from exc
                self._check_trajectory_identity(identity)
                return JSONResponse(
                    {**data, "run_id": identity[1], "task_generation": identity[2]},
                    headers={"Cache-Control": "no-store"},
                )

        @app.get("/api/session/trajectory/artifact")
        def api_trajectory_artifact(
            ref: str, run_id: str | None = None, generation: int | None = None
        ) -> Response:
            with self._trajectory_lock:
                reader, identity = self._get_trajectory_reader(run_id, generation)
                try:
                    path = reader.artifact_path(ref)
                except ValueError as exc:
                    raise HTTPException(status_code=403, detail=str(exc)) from exc
                if not path.is_file():
                    raise HTTPException(
                        status_code=404, detail="trajectory artifact not found"
                    )
                self._check_trajectory_identity(identity)
                media_type = (
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                )
                if media_type not in {
                    "image/png",
                    "image/jpeg",
                    "image/webp",
                    "image/gif",
                    "video/mp4",
                    "video/webm",
                    "application/json",
                }:
                    media_type = "application/octet-stream"
                return FileResponse(
                    path,
                    media_type=media_type,
                    headers={
                        "Cache-Control": "no-store",
                        "X-Content-Type-Options": "nosniff",
                    },
                )

        @app.get("/api/session/primitives")
        def api_primitives() -> JSONResponse:
            try:
                primitives = self._state.primitive_specs()
            except InteractionUnavailableError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            return JSONResponse({"primitives": primitives})

        @app.post("/api/session/primitive")
        def api_execute_primitive(
            payload: Any = Body(default=None),
        ) -> JSONResponse:
            if not isinstance(payload, dict):
                return JSONResponse(
                    {"error": "request body must be a JSON object"},
                    status_code=422,
                )
            name = payload.get("name")
            arguments = payload.get("arguments")
            if not isinstance(name, str) or not name.strip():
                return JSONResponse(
                    {"error": "name must be a non-empty string"},
                    status_code=422,
                )
            if not isinstance(arguments, dict):
                return JSONResponse(
                    {"error": "arguments must be a JSON object"},
                    status_code=422,
                )
            try:
                tool_result = self._state.execute_primitive(name, arguments)
            except InteractionUnavailableError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            except PrimitiveArgumentError as exc:
                return JSONResponse({"error": str(exc)}, status_code=422)
            except PrimitiveConfigError as exc:
                logger.error("Dashboard primitive configuration error: %s", exc)
                return JSONResponse({"error": str(exc)}, status_code=500)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)
            result = tool_result.result
            if isinstance(result, dict) and result.get("error") is not None:
                error = " ".join(str(result["error"]).split())[:500]
                return JSONResponse(
                    {"error": error or "primitive execution failed"},
                    status_code=422,
                )
            return JSONResponse({"ok": True})

        @app.get("/api/session/transcript")
        def api_transcript(since: int = 0) -> JSONResponse:
            return JSONResponse({"events": self._state.events_since(since)})

        @app.get("/api/session/frame")
        def api_frame(kind: str) -> Response:
            frame = self._state.frame(kind)
            if frame is None:
                return Response(status_code=404)
            return Response(frame, media_type="image/png")

        @app.get("/api/session/video")
        def api_video() -> Response:
            if not self._state.has_video():
                return Response(status_code=404)
            return FileResponse(
                self._state.video_path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/session/action-video")
        def api_action_video(step: int) -> Response:
            path = self._state.action_video_path(step)
            if path is None:
                return Response(status_code=404)
            return FileResponse(
                path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/session/stream")
        def api_stream() -> StreamingResponse:
            async def gen():
                version = -1
                while True:
                    version, snapshot = await asyncio.to_thread(
                        self._state.wait_for_snapshot,
                        version,
                        timeout=15.0,
                    )
                    if snapshot is None:
                        yield ": keepalive\n\n"
                    else:
                        yield f"data: {json.dumps(snapshot)}\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return app
