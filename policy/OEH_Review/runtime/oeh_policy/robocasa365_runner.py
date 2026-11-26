"""OEH condition's RoboCasa365 client: the pinned ``benchmarks/robocasa365/single_eval.py`` loop with the OEH policy.

    python -m oeh_policy.robocasa365_runner --config <runtime policy_config.yml> --task T --split pretrain --host H
        --port P

The loop runs exactly as the direct condition runs it (``oeh_eval.robocasa365_client.run``: the official ``main``,
its merged config, ``run_eval``, ``_rollout`` and printed result, plus the evaluator's evidence and native-episode
hooks); only the policy object in the loop's seat differs: the released client, built by the loop's own
``_build_policy``, wrapped in ``oeh_policy.robocasa365_entry``. The frozen OEH configuration is validated first, once;
one the policy refuses is reported to the campaign as a refusal to start.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from . import robocasa365_entry
from .robocasa365 import RoboCasa365Contract


def main(argv=None):
    from oeh_eval import robocasa365_client, runtime_config
    expected = json.loads(os.environ.get(runtime_config.EXPECTED_ENV) or "{}")
    try:
        path = robocasa365_entry.policy_config_path()
        cfg = RoboCasa365Contract().validate_config(json.loads(path.read_text(encoding="utf-8")))
        loop = Path(cfg["robocasa365"]["official_eval_path"]).resolve()
        if cfg["identity_files"].get(str(loop)) is None:
            raise ValueError("the official RoboCasa365 loop must be pinned")
        campaign = os.environ.get(robocasa365_client.LOOP_ENV)
        if campaign and Path(campaign).resolve() != loop:
            raise ValueError("the OEH configuration names another official RoboCasa365 loop than the campaign")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        robocasa365_client.refuse(f"OEH policy configuration rejected: {exc}", expected)
    robocasa365_entry.accept_validated(path, cfg)
    return robocasa365_client.run(sys.argv[1:] if argv is None else argv, loop_path=loop,
                                  policy_factory=robocasa365_entry.make_policy)


if __name__ == "__main__":
    raise SystemExit(main())
