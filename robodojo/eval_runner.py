"""RoboDojo evaluation runner CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TextIO

from robodojo.schemas import DispatchPayload


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
    args = parser.parse_args(argv)

    if args.dispatch_payload == "-":
        text = (stdin or sys.stdin).read()
    else:
        with open(args.dispatch_payload, "r", encoding="utf-8") as f:
            text = f.read()

    dispatch = DispatchPayload.model_validate_json(text)

    trial_runs = []
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

    out = stdout or sys.stdout
    json.dump(
        {
            "evaluation_id": dispatch.evaluation_id,
            "policy_server_url": dispatch.policy_server.url,
            "connection_mode": dispatch.policy_server.connection_mode,
            "task": dispatch.evaluation_plan.task,
            "repeat_count": dispatch.evaluation_plan.repeat_count,
            "trial_count": len(dispatch.evaluation_plan.trials),
            "planned_trial_runs": len(trial_runs),
            "trial_runs": trial_runs,
            "status": "planned",
        },
        out,
        sort_keys=True,
    )
    out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
