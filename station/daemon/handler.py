"""ThreadingHTTPServer request handler for the eval-station daemon."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from pydantic import ValidationError

from station.daemon.preview_routes import (
    handle_preview_get,
    handle_preview_post,
    parse_preview_route,
)
from station.daemon.session_routes import parse_session_route
from station.daemon.state import EnvClientServerState
from station.dispatch.errors import normalize_execution_error
from station.dispatch.executor import run_dispatch
from station.env_client.api import EnvClientBaselineConfig, HealthResponse, TrialRunResponse
from station.env_client.runner import TrialRunnerError, reset_idle_env
from station.env_client.trial_control import StopRequestResult
from station.schemas import DispatchPayload


def _first_record(items: object) -> dict[str, Any]:
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0]
    return {}


def _trial_id_from_summary(summary: dict[str, object], trial_index: int) -> str:
    policy_result = _first_record(summary.get("policy_results"))
    trial_run = _first_record(summary.get("trial_runs"))
    return str(
        policy_result.get("trial_id")
        or trial_run.get("trial_id")
        or f"trial-{trial_index}"
    )


def _summary_error(summary: dict[str, object]) -> dict[str, Any]:
    error = summary.get("error")
    if isinstance(error, dict):
        return error
    return {
        "code": "internal",
        "message": str(summary.get("error_summary", "trial failed")),
    }


def _start_response_from_summary(
    baseline: EnvClientBaselineConfig,
    summary: dict[str, object],
    *,
    exit_code: int,
    artifact_dir: Path,
    trial_index: int,
) -> dict[str, Any]:
    trial_id = _trial_id_from_summary(summary, trial_index)
    if exit_code != 0:
        response = TrialRunResponse(
            status="failed",
            trial_id=trial_id,
            eval_env_type=baseline.eval_env_type,
            policy_name=baseline.policy_name,
            error=_summary_error(summary),
        )
    else:
        policy_result = _first_record(summary.get("policy_results"))
        response = TrialRunResponse(
            status="completed",
            trial_id=trial_id,
            steps=policy_result.get("steps"),
            eval_env_type=policy_result.get("eval_env_type", baseline.eval_env_type),
            policy_name=policy_result.get("policy_name", baseline.policy_name),
        )

    body = response.model_dump(mode="json")
    body["exit_code"] = exit_code
    body["artifact_dir"] = str(artifact_dir)
    return body


_STOP_HTTP_RESPONSES: dict[
    StopRequestResult,
    tuple[HTTPStatus, dict[str, str]],
] = {
    "not_found": (HTTPStatus.NOT_FOUND, {"error": "no active trial"}),
    "already_stopping": (
        HTTPStatus.CONFLICT,
        {"error": "trial stop already requested"},
    ),
    "accepted": (HTTPStatus.OK, {"status": "stopping"}),
}


def make_handler(state: EnvClientServerState) -> type[BaseHTTPRequestHandler]:
    class EnvClientHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            preview_route = parse_preview_route(self.path)
            if preview_route is not None:
                action, role = preview_route
                if action in ("pause", "resume"):
                    self._write_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                    return
                handle_preview_get(self, state.preview, action, role)
                return

            if self._path != "/v1/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            self._write_model(
                HTTPStatus.OK,
                HealthResponse(
                    policy_name=state.baseline.policy_name,
                    eval_env_type=state.baseline.eval_env_type,
                    deploy_yml=state.deploy_yml,
                    last_trial_id=state.last_trial_id,
                ),
            )

        def do_POST(self) -> None:
            preview_route = parse_preview_route(self.path)
            if preview_route is not None:
                action, _ = preview_route
                if action not in ("pause", "resume"):
                    self._write_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                    return
                handle_preview_post(self, state.preview, action)
                return

            if self._path == "/v1/reset":
                self._handle_reset()
                return

            route = parse_session_route(self.path)
            if route is None:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return

            evaluation_id, action, trial_index = route
            if action == "dispatch":
                self._handle_dispatch(evaluation_id)
                return
            assert trial_index is not None
            match action:
                case "start":
                    self._handle_start(evaluation_id, trial_index)
                case "stop":
                    self._handle_stop(evaluation_id, trial_index)

        @property
        def _path(self) -> str:
            return urlparse(self.path).path

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _handle_dispatch(self, evaluation_id: str) -> None:
            body = self._read_json_body()
            if body is None:
                return

            try:
                dispatch = DispatchPayload.model_validate(body)
            except ValidationError:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid dispatch payload"},
                )
                return

            state.dispatches[evaluation_id] = dispatch
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "accepted",
                    "evaluation_id": evaluation_id,
                },
            )

        def _handle_start(self, evaluation_id: str, trial_index: int) -> None:
            dispatch = state.dispatches.get(evaluation_id)
            if dispatch is None:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "dispatch payload not found"},
                )
                return

            if not any(
                trial.trial_index == trial_index
                for trial in dispatch.evaluation_plan.trials
            ):
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "trial not found in dispatch payload"},
                )
                return

            artifact_dir = state.artifact_dir(evaluation_id, trial_index)
            try:
                trial_runner = state.trial_runner_with_stop(evaluation_id, trial_index)
            except TrialRunnerError as exc:
                body = TrialRunResponse(
                    status="failed",
                    trial_id=f"trial-{trial_index}",
                    eval_env_type=state.baseline.eval_env_type,
                    policy_name=state.baseline.policy_name,
                    error=exc.error or normalize_execution_error(exc),
                ).model_dump(mode="json")
                body["exit_code"] = 1
                body["artifact_dir"] = str(artifact_dir)
                self._write_json(HTTPStatus.OK, body)
                return
            if trial_runner is None:
                self._write_json(
                    HTTPStatus.CONFLICT,
                    {"error": "another trial is already executing"},
                )
                return
            try:
                exit_code, summary = run_dispatch(
                    dispatch,
                    trial_index=trial_index,
                    evaluation_id=evaluation_id,
                    artifact_dir=artifact_dir,
                    upload_s3=state.config.upload_s3,
                    notify_webhook=state.config.notify_webhook,
                    run_policy_trials=state.config.run_policy_trials,
                    webhook_secret=state.config.webhook_secret,
                    trial_runner=trial_runner,
                    publish_submit=state.submit_publish,
                )
            except Exception as exc:
                body = TrialRunResponse(
                    status="failed",
                    trial_id=f"trial-{trial_index}",
                    eval_env_type=state.baseline.eval_env_type,
                    policy_name=state.baseline.policy_name,
                    error=normalize_execution_error(exc),
                ).model_dump(mode="json")
                body["exit_code"] = 1
                body["artifact_dir"] = str(artifact_dir)
                self._write_json(HTTPStatus.OK, body)
                return
            finally:
                state.trial_control.clear(evaluation_id, trial_index)

            state.last_trial_id = _trial_id_from_summary(summary, trial_index)
            self._write_json(
                HTTPStatus.OK,
                _start_response_from_summary(
                    state.baseline,
                    summary,
                    exit_code=exit_code,
                    artifact_dir=artifact_dir,
                    trial_index=trial_index,
                ),
            )

        def _handle_stop(self, evaluation_id: str, trial_index: int) -> None:
            result = state.trial_control.request_stop(evaluation_id, trial_index)
            status_code, body = _STOP_HTTP_RESPONSES[result]
            self._write_json(status_code, body)

        def _handle_reset(self) -> None:
            if state.trial_control.has_active_trials():
                self._write_json(
                    HTTPStatus.CONFLICT,
                    {"error": "cannot reset while a trial is executing"},
                )
                return

            try:
                if state.persistent_runtime is not None:
                    state.persistent_runtime.reset_idle()
                else:
                    reset_kwargs: dict[str, object] = {}
                    if state.dispatches:
                        evaluation_id, dispatch = next(iter(state.dispatches.items()))
                        reset_kwargs = {
                            "dispatch": dispatch,
                            "evaluation_id": evaluation_id,
                        }
                    reset_idle_env(state.baseline, **reset_kwargs)
            except TrialRunnerError as exc:
                error = exc.error or {
                    "code": "reset_failed",
                    "message": str(exc),
                }
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"status": "failed", "error": error},
                )
                return
            except Exception as exc:
                error = normalize_execution_error(exc)
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"status": "failed", "error": error},
                )
                return

            self._write_json(HTTPStatus.OK, {"status": "reset"})

        def _read_json_body(self) -> dict[str, Any] | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid Content-Length"},
                )
                return None

            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"invalid JSON: {exc}"},
                )
                return None
            if not isinstance(body, dict):
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "request body must be a JSON object"},
                )
                return None
            return body

        def _write_model(self, status_code: HTTPStatus, model: Any) -> None:
            self._write_json(status_code, model.model_dump(mode="json"))

        def _write_json(self, status_code: HTTPStatus, body: dict[str, Any]) -> None:
            payload = json.dumps(body, sort_keys=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return EnvClientHandler


def create_server(
    host: str,
    port: int,
    state: EnvClientServerState,
) -> ThreadingHTTPServer:
    state.config.artifact_root.mkdir(parents=True, exist_ok=True)
    return ThreadingHTTPServer((host, port), make_handler(state))


def _session_path(evaluation_id: str, suffix: str) -> str:
    return f"/sessions/{quote(evaluation_id, safe='')}/{suffix}"


def session_dispatch_path(evaluation_id: str) -> str:
    return _session_path(evaluation_id, "dispatch")


def session_start_path(evaluation_id: str, trial_index: int) -> str:
    return _session_path(evaluation_id, f"trials/{trial_index}/start")


def session_stop_path(evaluation_id: str, trial_index: int) -> str:
    return _session_path(evaluation_id, f"trials/{trial_index}/stop")
