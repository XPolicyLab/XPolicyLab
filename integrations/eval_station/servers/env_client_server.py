"""HTTP daemon for eval-station environment clients."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse

from pydantic import ValidationError

from eval_station.dispatch.errors import normalize_execution_error
from eval_station.eval_env_type import is_real_world, resolve_eval_env_type
from eval_station.dispatch.executor import run_dispatch
from eval_station.env_client.api import EnvClientBaselineConfig, HealthResponse, TrialRunResponse
from eval_station.env_client.runner import (
    DebugTrialRunner,
    TrialRunnerError,
    TrialRunnerFn,
    _ensure_pipeline_paths,
    make_dispatch_trial_runner,
    reset_idle_env,
)
from eval_station.env_client.trial_control import StopRequestResult, TrialControlRegistry
from eval_station.schemas import DispatchPayload
from eval_station.servers.preview_routes import (
    handle_preview_get,
    handle_preview_post,
    parse_preview_route,
)
from eval_station.servers.session_routes import parse_session_route


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
    run_trial: DebugTrialRunner | None = None
    last_trial_id: str | None = None
    dispatches: dict[str, DispatchPayload] = field(default_factory=dict)
    trial_control: TrialControlRegistry = field(default_factory=TrialControlRegistry)
    preview: Any | None = None
    persistent_runtime: Any | None = None
    # Single-worker pool so trial recordings publish (encode + S3 upload +
    # finish webhook) one at a time off the /start thread, preserving order
    # while letting /start return as soon as the trial loop ends.
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
            # Runner construction failed (e.g. unsupported eval_env_type);
            # release the registration or every later /start returns 409.
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
    """Surface unexpected background publish crashes to stderr.

    ``publish_trial_recording`` already converts expected upload/webhook errors
    into a ``failed`` finish webhook, so this only fires on truly unexpected
    exceptions that would otherwise be swallowed by the worker thread.
    """
    exc = future.exception()
    if exc is not None:
        print(
            "background trial publish failed: "
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            file=sys.stderr,
        )


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


def add_debug_env_client_arguments(parser: argparse.ArgumentParser) -> None:
    from debug_env_client import str2bool

    parser.add_argument("--bench_name", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--env_cfg_type", type=str, default=None)
    parser.add_argument(
        "--policy_name",
        type=str,
        default=None,
        help="XPolicyLab module name for deployment "
        "(optional: auto-filled from dispatch payload)",
    )
    parser.add_argument(
        "--protocol",
        choices=("legacy_tcp", "ws"),
        default="ws",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="policy server host (optional: dispatch payload provides policy_server_url)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="policy server port (optional: dispatch payload provides policy_server_url)",
    )
    parser.add_argument("--policy_server_url", type=str)
    parser.add_argument("--evaluation_id", type=str, default="debug-eval")
    parser.add_argument("--action_case_id", type=str)
    parser.add_argument("--trial_id", type=str, default="debug-trial")
    parser.add_argument("--repeat_index", type=int)
    parser.add_argument(
        "--eval_episode_num",
        type=int,
        default=10,
        help="number of evaluation episodes",
    )
    parser.add_argument(
        "--eval_batch",
        type=str2bool,
        default=False,
        help="whether to run batch evaluation",
    )
    parser.add_argument(
        "--eval-env-type",
        dest="eval_env_type",
        type=str,
        default=None,
        help="evaluation environment type: debug, sim, or real_world (default: EVAL_ENV_TYPE or sim)",
    )
    parser.add_argument(
        "--root-dir",
        dest="root_dir",
        type=str,
        help="X-Robot-Pipeline root directory (required when eval_env_type=real_world)",
    )
    parser.add_argument(
        "--base-cfg",
        dest="base_cfg",
        type=str,
        help="Fixed robot base config for this eval station (config/{name}.yml)",
    )
    parser.add_argument(
        "--deploy-yml",
        dest="deploy_yml",
        type=str,
        help="deploy.yml path reported by /v1/health",
    )
    parser.add_argument(
        "--action-type",
        dest="action_type",
        choices=("joint", "ee"),
        help="robot action schema for RealEnv (must match policy output, e.g. ee for X_VLA)",
    )


def baseline_from_args(args: argparse.Namespace) -> EnvClientBaselineConfig:
    return EnvClientBaselineConfig(
        bench_name=args.bench_name,
        task_name=args.task_name,
        env_cfg_type=args.env_cfg_type,
        policy_name=args.policy_name,
        protocol=args.protocol,
        host=args.host,
        port=args.port,
        eval_batch=args.eval_batch,
        eval_episode_num=args.eval_episode_num,
        eval_env_type=resolve_eval_env_type(args.eval_env_type),
        root_dir=args.root_dir,
        action_type=args.action_type,
        base_cfg=args.base_cfg,
    )


def _validate_startup_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.no_policy_trials and not args.no_webhook:
        parser.error("--no-policy-trials requires --no-webhook")
    try:
        eval_env_type = resolve_eval_env_type(args.eval_env_type)
    except ValueError as exc:
        parser.error(str(exc))
    if is_real_world(eval_env_type) and not args.root_dir:
        parser.error("--root-dir is required when --eval-env-type=real_world")
    if is_real_world(eval_env_type) and not args.base_cfg:
        parser.error("--base-cfg is required when --eval-env-type=real_world")
    # action_type is resolved per trial (dispatch payload > startup arg) and
    # validated at trial start for real envs.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Artifact upload (S3 / Volcano TOS) env vars:\n"
            "  TOS_ENDPOINT_URL / S3_ENDPOINT_URL  S3-compatible endpoint, e.g. "
            "https://tos-s3-cn-beijing.volces.com (scheme optional; falls back to "
            "AWS_ENDPOINT_URL; unset = default AWS S3).\n"
            "  TOS_REGION / S3_REGION  e.g. cn-shanghai (falls back to AWS_REGION).\n"
            "  TOS_BUCKET / S3_BUCKET  default bucket when dispatch.artifact.bucket "
            "is omitted.\n"
            "  TOS_PREFIX / S3_PREFIX / ROBODOJO_ARTIFACT_PREFIX  default key prefix "
            "when dispatch.artifact.prefix is omitted.\n"
            "  S3_ADDRESSING_STYLE  S3 path style (default: virtual when an endpoint "
            "is set; required for Volcano TOS).\n"
            "  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  TOS access key / secret.\n"
            "  EVAL_SERVER_WEBHOOK_SECRET  HMAC secret for the finish webhook.\n"
            "Dispatch may still set artifact.bucket / artifact.prefix explicitly; env "
            "vars are used only as fallbacks when those fields are empty."
        ),
    )
    parser.add_argument("--serve-host", default="0.0.0.0")
    parser.add_argument("--serve-port", type=int, default=19200)
    parser.add_argument(
        "--artifact-root",
        default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "robodojo-artifacts"),
        help="Directory where per-evaluation artifacts are written",
    )
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 artifact upload")
    parser.add_argument(
        "--no-webhook",
        action="store_true",
        help="Skip finish webhook callback",
    )
    parser.add_argument(
        "--no-policy-trials",
        action="store_true",
        help="Only materialize planned artifacts; do not run trials",
    )
    add_debug_env_client_arguments(parser)
    args = parser.parse_args(argv)
    _validate_startup_args(parser, args)

    state = EnvClientServerState(
        baseline=baseline_from_args(args),
        config=EnvClientServerConfig(
            artifact_root=Path(args.artifact_root),
            upload_s3=not args.no_s3,
            notify_webhook=not args.no_webhook,
            run_policy_trials=not args.no_policy_trials,
            webhook_secret=os.environ.get("EVAL_SERVER_WEBHOOK_SECRET") or None,
        ),
        deploy_yml=args.deploy_yml,
    )
    if is_real_world(state.baseline.eval_env_type):
        _ensure_pipeline_paths(str(args.root_dir))
        from task_env.real_env_client import PersistentRealRobotRuntime

        state.persistent_runtime = PersistentRealRobotRuntime(
            root_dir=str(args.root_dir),
            base_cfg_name=str(args.base_cfg),
        )
        state.persistent_runtime.start()
        state.preview = state.persistent_runtime
        state.run_trial = state.persistent_runtime.run_trial
        print(
            f"persistent robot runtime enabled (base cfg: {args.base_cfg})",
            file=sys.stderr,
        )

    server = create_server(args.serve_host, args.serve_port, state)
    print(
        f"eval-station env client listening on http://{args.serve_host}:{args.serve_port}",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        # Drain queued trial publishes so their finish webhooks land before exit.
        state.shutdown_publish()
        if state.persistent_runtime is not None:
            # A second Ctrl+C during pipeline.stop() leaves the Orbbec
            # firmware half-stopped (stuck at STARTING, no frames until a
            # device reboot). Ignore SIGINT until cameras are released.
            import signal

            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                state.persistent_runtime.cleanup()
            finally:
                signal.signal(signal.SIGINT, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
