"""RoboDojo evaluation runner CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from robodojo.dispatch.executor import run_dispatch
from robodojo.schemas import DispatchPayload
from robodojo.trial import add_trial_env_arguments


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dispatch-payload",
        required=True,
        help="Path to dispatch JSON; use '-' to read from stdin",
    )
    parser.add_argument(
        "--evaluation-id",
        required=True,
        help="Evaluation id supplied by the control-plane route",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Directory for manifest.json, metrics.json, events.jsonl, videos/, logs/",
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip uploading artifacts to S3",
    )
    parser.add_argument(
        "--no-webhook",
        action="store_true",
        help="Skip finish webhook callback",
    )
    parser.add_argument(
        "--run-policy-trials",
        action="store_true",
        help="Connect to policy_server and run prepare/reset/infer/trial_end per trial",
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        required=True,
        help="Trial index to run from the dispatch plan",
    )
    add_trial_env_arguments(parser)
    args = parser.parse_args(argv)
    if args.artifact_dir and not args.run_policy_trials and not args.no_webhook:
        parser.error("--run-policy-trials is required unless --no-webhook is set")

    if args.dispatch_payload == "-":
        text = (stdin or sys.stdin).read()
    else:
        with open(args.dispatch_payload, "r", encoding="utf-8") as f:
            text = f.read()

    dispatch = DispatchPayload.model_validate_json(text)
    exit_code, summary = run_dispatch(
        dispatch,
        evaluation_id=args.evaluation_id,
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        upload_s3=not args.no_s3,
        notify_webhook=not args.no_webhook,
        run_policy_trials=args.run_policy_trials,
        trial_index=args.trial_index,
        eval_env=args.eval_env,
        root_dir=args.root_dir,
        sim_env_factory=args.sim_env_factory,
        episode_step_limit=args.episode_step_limit,
    )

    out = stdout or sys.stdout
    json.dump(summary, out, sort_keys=True)
    out.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
