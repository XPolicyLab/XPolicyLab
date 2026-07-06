"""CLI entry point for the eval-station HTTP daemon."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from station.daemon.handler import create_server
from station.daemon.state import EnvClientServerConfig, EnvClientServerState
from station.env_client.api import EnvClientBaselineConfig
from station.env_client.runner import _ensure_pipeline_paths
from station.eval_env_type import is_real_world, resolve_eval_env_type


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
        state.shutdown_publish()
        if state.persistent_runtime is not None:
            import signal

            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                state.persistent_runtime.cleanup()
            finally:
                signal.signal(signal.SIGINT, previous)
    return 0
