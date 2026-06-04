"""HTTP control-plane shim for RoboDojo evaluation execution."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse

from pydantic import ValidationError

from robodojo.eval_runner import run_dispatch
from robodojo.schemas import DispatchPayload


@dataclass(frozen=True)
class ExecutorConfig:
    work_dir: Path
    artifact_root: Path
    run_policy_trials: bool = True
    max_trials: int = 0
    upload_s3: bool = True
    notify_webhook: bool = True


RunnerFn = Callable[[DispatchPayload, Path, ExecutorConfig], tuple[int, dict[str, object]]]


def default_runner(
    dispatch: DispatchPayload,
    artifact_dir: Path,
    config: ExecutorConfig,
) -> tuple[int, dict[str, object]]:
    return run_dispatch(
        dispatch,
        artifact_dir=artifact_dir,
        upload_s3=config.upload_s3,
        notify_webhook=config.notify_webhook,
        run_policy_trials=config.run_policy_trials,
        max_trials=config.max_trials,
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

    def result_path(self, evaluation_id: str) -> Path:
        return self.session_dir(evaluation_id) / "result.json"

    def artifact_dir(self, evaluation_id: str) -> Path:
        return self.config.artifact_root / self.safe_id(evaluation_id)

    def session_dir(self, evaluation_id: str) -> Path:
        return self.config.work_dir / self.safe_id(evaluation_id)

    @staticmethod
    def safe_id(evaluation_id: str) -> str:
        return quote(evaluation_id, safe="")


def make_handler(state: ExecutorState) -> type[BaseHTTPRequestHandler]:
    class ExecutorHandler(BaseHTTPRequestHandler):
        server_version = "RoboDojoExecutor/0.1"

        def do_POST(self) -> None:
            route = _parse_session_route(self.path)
            if route is None:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "unknown endpoint"},
                )
                return

            evaluation_id, action = route
            if action == "dispatch":
                self._handle_dispatch(evaluation_id)
                return
            if action == "start":
                self._handle_start(evaluation_id)
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _handle_dispatch(self, evaluation_id: str) -> None:
            body = self._read_json_body()
            if body is None:
                return
            try:
                dispatch = DispatchPayload.model_validate(body)
            except ValidationError as exc:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid dispatch payload", "details": exc.errors()},
                )
                return

            if dispatch.evaluation_id != evaluation_id:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "evaluation_id does not match request path"},
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

        def _handle_start(self, evaluation_id: str) -> None:
            body = self._read_json_body()
            if body is None:
                return
            body_evaluation_id = body.get("evaluation_id")
            if body_evaluation_id is not None and body_evaluation_id != evaluation_id:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "evaluation_id does not match request path"},
                )
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
            artifact_dir = state.artifact_dir(evaluation_id)
            thread = threading.Thread(
                target=_run_and_store_result,
                args=(state, dispatch, artifact_dir),
                name=f"robodojo-executor-{evaluation_id}",
                daemon=True,
            )
            thread.start()

            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "started",
                    "evaluation_id": evaluation_id,
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


def _parse_session_route(path: str) -> tuple[str, str] | None:
    parts = urlparse(path).path.strip("/").split("/")
    if len(parts) != 3 or parts[0] != "sessions":
        return None
    evaluation_id = unquote(parts[1])
    action = parts[2]
    if not evaluation_id or action not in {"dispatch", "start"}:
        return None
    return evaluation_id, action


def _run_and_store_result(
    state: ExecutorState,
    dispatch: DispatchPayload,
    artifact_dir: Path,
) -> None:
    try:
        exit_code, summary = state.runner(dispatch, artifact_dir, state.config)
    except Exception as exc:
        exit_code = 1
        summary = {
            "evaluation_id": dispatch.evaluation_id,
            "status": "failed",
            "error_summary": str(exc),
        }

    result_path = state.result_path(dispatch.evaluation_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "exit_code": exit_code,
                "summary": summary,
            },
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
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="When policy trials run, limit trials (0 = all)",
    )
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 artifact upload")
    parser.add_argument(
        "--no-webhook",
        action="store_true",
        help="Skip finish webhook callback",
    )
    args = parser.parse_args(argv)

    config = ExecutorConfig(
        work_dir=Path(args.work_dir),
        artifact_root=Path(args.artifact_root),
        run_policy_trials=not args.no_policy_trials,
        max_trials=args.max_trials,
        upload_s3=not args.no_s3,
        notify_webhook=not args.no_webhook,
    )
    server = create_server(args.host, args.port, config)
    print(f"robodojo executor listening on http://{args.host}:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
