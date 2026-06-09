"""HTTP control-plane shim for RoboDojo evaluation execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse

from pydantic import ValidationError

from robodojo.eval_runner import (
    STATUS_FAILED,
    normalize_execution_error,
    notify_trial_failure,
    run_dispatch,
)
from robodojo.schemas import DispatchPayload
from robodojo.serialization import to_jsonable


@dataclass(frozen=True)
class ExecutorConfig:
    work_dir: Path
    artifact_root: Path
    run_policy_trials: bool = True
    upload_s3: bool = True
    notify_webhook: bool = True
    trial_index: int | None = None
    webhook_secret: str | None = None


RunnerFn = Callable[
    [str, DispatchPayload, Path, ExecutorConfig], tuple[int, dict[str, object]]
]


def default_runner(
    evaluation_id: str,
    dispatch: DispatchPayload,
    artifact_dir: Path,
    config: ExecutorConfig,
) -> tuple[int, dict[str, object]]:
    if config.trial_index is None:
        raise ValueError("trial_index is required")
    return run_dispatch(
        dispatch,
        evaluation_id=evaluation_id,
        artifact_dir=artifact_dir,
        upload_s3=config.upload_s3,
        notify_webhook=config.notify_webhook,
        run_policy_trials=config.run_policy_trials,
        trial_index=config.trial_index,
        webhook_secret=config.webhook_secret,
    )


class ExecutorState:
    def __init__(
        self,
        config: ExecutorConfig,
        *,
        runner: RunnerFn = default_runner,
    ) -> None:
        self.config = config
        self.runner = runner
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self.config.artifact_root.mkdir(parents=True, exist_ok=True)

    def dispatch_path(self, evaluation_id: str) -> Path:
        return self.session_dir(evaluation_id) / "dispatch.json"

    def result_path(self, evaluation_id: str, trial_index: int) -> Path:
        return (
            self.session_dir(evaluation_id)
            / "trials"
            / str(trial_index)
            / "result.json"
        )

    def artifact_dir(self, evaluation_id: str, trial_index: int) -> Path:
        return (
            self.config.artifact_root
            / quote(evaluation_id, safe="")
            / "trials"
            / str(trial_index)
        )

    def session_dir(self, evaluation_id: str) -> Path:
        return self.config.work_dir / quote(evaluation_id, safe="")


def make_handler(state: ExecutorState) -> type[BaseHTTPRequestHandler]:
    class ExecutorHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            route = _parse_session_route(self.path)
            if route is None:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "unknown endpoint"},
                )
                return

            evaluation_id, action, trial_index = route
            if action == "dispatch":
                self._handle_dispatch(evaluation_id)
                return
            assert trial_index is not None
            self._handle_start(evaluation_id, trial_index)

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

            session_dir = state.session_dir(evaluation_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            state.dispatch_path(evaluation_id).write_text(
                dispatch.model_dump_json(indent=2),
                encoding="utf-8",
            )

            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "accepted",
                    "evaluation_id": evaluation_id,
                    "dispatch_path": str(state.dispatch_path(evaluation_id)),
                },
            )

        def _handle_start(self, evaluation_id: str, trial_index: int) -> None:
            body = self._read_json_body()
            if body is None:
                return

            dispatch_path = state.dispatch_path(evaluation_id)
            if not dispatch_path.exists():
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "dispatch payload not found"},
                )
                return

            dispatch = DispatchPayload.model_validate_json(
                dispatch_path.read_text(encoding="utf-8")
            )
            if not any(
                trial.trial_index == trial_index
                for trial in dispatch.evaluation_plan.trials
            ):
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "trial not found in dispatch payload"},
                )
                return

            run_config = replace(state.config, trial_index=trial_index)
            artifact_dir = state.artifact_dir(evaluation_id, trial_index)
            thread = threading.Thread(
                target=_run_and_store_result,
                args=(state, evaluation_id, dispatch, artifact_dir, run_config),
                name=f"robodojo-executor-{evaluation_id}-trial-{trial_index}",
                daemon=True,
            )
            thread.start()

            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "started",
                    "evaluation_id": evaluation_id,
                    "trial_index": trial_index,
                    "artifact_dir": str(artifact_dir),
                },
            )

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

        def _write_json(self, status_code: HTTPStatus, body: dict[str, Any]) -> None:
            payload = json.dumps(body, sort_keys=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return ExecutorHandler


def _parse_session_route(path: str) -> tuple[str, str, int | None] | None:
    parts = urlparse(path).path.strip("/").split("/")
    match parts:
        case ["sessions", raw_evaluation_id, "dispatch"]:
            evaluation_id = unquote(raw_evaluation_id)
            return (evaluation_id, "dispatch", None) if evaluation_id else None
        case ["sessions", raw_evaluation_id, "trials", raw_trial_index, "start"]:
            evaluation_id = unquote(raw_evaluation_id)
            try:
                trial_index = int(raw_trial_index)
            except ValueError:
                return None
            if not evaluation_id or trial_index < 1:
                return None
            return evaluation_id, "start", trial_index
    return None


def _run_and_store_result(
    state: ExecutorState,
    evaluation_id: str,
    dispatch: DispatchPayload,
    artifact_dir: Path,
    config: ExecutorConfig,
) -> None:
    trial_index = config.trial_index
    if trial_index is None:
        raise ValueError("trial_index is required")

    try:
        exit_code, summary = state.runner(
            evaluation_id,
            dispatch,
            artifact_dir,
            config,
        )
    except Exception as exc:
        exit_code = 1
        error = normalize_execution_error(exc)
        summary = {
            "evaluation_id": evaluation_id,
            "trial_index": trial_index,
            "status": STATUS_FAILED,
            "error_summary": error["message"],
            "error": error,
        }
        if config.notify_webhook and not isinstance(exc, ValueError):
            try:
                summary["published"] = {
                    "webhook": notify_trial_failure(
                        dispatch,
                        trial_index=trial_index,
                        error=error,
                        webhook_secret=config.webhook_secret,
                    )
                }
            except Exception as webhook_exc:
                summary["webhook_error"] = str(webhook_exc)

    result_path = state.result_path(evaluation_id, trial_index)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            to_jsonable(
                {
                    "exit_code": exit_code,
                    "summary": summary,
                }
            ),
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_server(
    host: str,
    port: int,
    config: ExecutorConfig,
    *,
    runner: RunnerFn = default_runner,
) -> ThreadingHTTPServer:
    state = ExecutorState(config, runner=runner)
    return ThreadingHTTPServer((host, port), make_handler(state))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument(
        "--work-dir",
        default="/private/tmp/robodojo-executor",
        help="Directory for persisted dispatch payloads and run results",
    )
    parser.add_argument(
        "--artifact-root",
        default="/private/tmp/robodojo-artifacts",
        help="Directory where per-evaluation artifacts are written",
    )
    parser.add_argument(
        "--no-policy-trials",
        action="store_true",
        help="Only materialize planned artifacts; do not connect to policy_server",
    )
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 artifact upload")
    parser.add_argument(
        "--no-webhook",
        action="store_true",
        help="Skip finish webhook callback",
    )
    args = parser.parse_args(argv)
    if args.no_policy_trials and not args.no_webhook:
        parser.error("--no-policy-trials requires --no-webhook")

    config = ExecutorConfig(
        work_dir=Path(args.work_dir),
        artifact_root=Path(args.artifact_root),
        run_policy_trials=not args.no_policy_trials,
        upload_s3=not args.no_s3,
        notify_webhook=not args.no_webhook,
        webhook_secret=os.environ.get("EVAL_SERVER_WEBHOOK_SECRET") or None,
    )
    server = create_server(args.host, args.port, config)
    print(
        f"robodojo executor listening on http://{args.host}:{args.port}",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
