"""OEH conditions' RoboTwin client: the pinned `script/eval_policy.py` loop with `oeh_policy.robotwin_entry`.

The loop runs exactly as condition 1 runs it (`oeh_eval.robotwin_client.run`: OpenWAM's
`eval_policy_wrapper` bootstrap and `ROBOTWIN_TEST_NUM=1` cap, RoboTwin's own `main`, expert
seed search, episode loop and `_result.txt`, plus the evaluator's evidence and native-episode
hooks); only `policy_name` differs. The frozen OEH configuration is validated first, once;
one the policy refuses is reported to the campaign as a refusal to start.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from . import robotwin_entry
from .robotwin import RoboTwinContract

POLICY_NAME = "oeh_policy.robotwin_entry"


def main():
    from oeh_eval import robotwin_client, runtime_config
    expected = json.loads(os.environ.get(runtime_config.EXPECTED_ENV) or "{}")
    try:
        path = robotwin_entry.policy_config_path({})
        cfg = RoboTwinContract().validate_config(json.loads(path.read_text(encoding="utf-8")))
        wrapper = Path(cfg["robotwin"]["official_wrapper_path"]).resolve()
        if cfg["identity_files"].get(str(wrapper)) is None:
            raise ValueError("published RoboTwin evaluation wrapper must be pinned")
        campaign = os.environ.get(robotwin_client.WRAPPER_ENV)
        if campaign and Path(campaign).resolve() != wrapper:
            raise ValueError("the OEH configuration names another OpenWAM wrapper than the campaign")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        robotwin_client.refuse(f"OEH policy configuration rejected: {exc}", expected)
    robotwin_entry.accept_validated(path, cfg)
    return robotwin_client.run(POLICY_NAME, wrapper_path=wrapper, direct=False)


if __name__ == "__main__":
    raise SystemExit(main())
