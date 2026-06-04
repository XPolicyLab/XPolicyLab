"""RoboDojo evaluation runner CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from robodojo.artifacts import ArtifactWriter
from robodojo.schemas import DispatchPayload


def build_trial_runs(dispatch: DispatchPayload) -> list[dict[str, object]]:
    trial_runs: list[dict[str, object]] = []
    for trial_index, trial in enumerate(dispatch.evaluation_plan.trials):
        for repeat_index in range(dispatch.evaluation_plan.repeat_count):
            trial_runs.append(
                {
                    "trial_id": (
                        f"{dispatch.evaluation_id}:{trial.action_case_id}"
                        f":repeat-{repeat_index}"
                    ),
                    "action_case_id": trial.action_case_id,
                    "trial_index": trial_index,
                    "repeat_index": repeat_index,
                    "case_meta": trial.model_dump(),
                }
            )
    return trial_runs


def write_artifacts(
    dispatch: DispatchPayload,
    trial_runs: list[dict[str, object]],
    artifact_dir: Path,
) -> dict[str, str]:
    writer = ArtifactWriter(artifact_dir, dispatch)
    writer.setup()
    try:
        writer.emit_event("run_started")
        writer.register_trials(trial_runs)
        for trial_run in trial_runs:
            trial_id = str(trial_run["trial_id"])
            writer.write_video_placeholder(trial_id)
        return writer.finalize(status="planned")
    finally:
        writer.close()


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
        "--artifact-dir",
        help="Directory for manifest.json, metrics.json, events.jsonl, videos/, logs/",
    )
    args = parser.parse_args(argv)

    if args.dispatch_payload == "-":
        text = (stdin or sys.stdin).read()
    else:
        with open(args.dispatch_payload, "r", encoding="utf-8") as f:
            text = f.read()

    dispatch = DispatchPayload.model_validate_json(text)
    trial_runs = build_trial_runs(dispatch)

    artifact_paths: dict[str, str] | None = None
    if args.artifact_dir:
        artifact_paths = write_artifacts(
            dispatch,
            trial_runs,
            Path(args.artifact_dir),
        )

    summary: dict[str, object] = {
        "evaluation_id": dispatch.evaluation_id,
        "policy_server_url": dispatch.policy_server.url,
        "connection_mode": dispatch.policy_server.connection_mode,
        "task": dispatch.evaluation_plan.task,
        "repeat_count": dispatch.evaluation_plan.repeat_count,
        "trial_count": len(dispatch.evaluation_plan.trials),
        "planned_trial_runs": len(trial_runs),
        "trial_runs": trial_runs,
        "status": "planned",
    }
    if artifact_paths is not None:
        summary["artifacts"] = artifact_paths

    out = stdout or sys.stdout
    json.dump(summary, out, sort_keys=True)
    out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
