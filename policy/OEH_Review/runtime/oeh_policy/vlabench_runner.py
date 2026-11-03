"""OEH conditions' VLABench client: OpenWAM's pinned single_eval.py with `oeh_policy.vlabench_entry` as the agent.

The loop runs exactly as condition 1 runs it (`oeh_eval.vlabench_client.run`: single_eval.py's `main` with the argv
single_eval.sh passes, VLABench's Evaluator and its result files, plus the evaluator's evidence and native-episode
hooks); only the agent the Evaluator is given differs. The frozen OEH configuration is validated first, once; one the
policy refuses is reported to the campaign as a refusal to start.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from . import vlabench_entry
from .vlabench import VLABenchContract


def main():
    from oeh_eval import runtime_config, vlabench_client
    expected = json.loads(os.environ.get(runtime_config.EXPECTED_ENV) or "{}")
    try:
        path = vlabench_entry.policy_config_path()
        cfg = VLABenchContract().validate_config(json.loads(path.read_text(encoding="utf-8")))
        single_eval = Path(cfg["vlabench"]["single_eval_path"]).resolve()
        if cfg["identity_files"].get(str(single_eval)) is None:
            raise ValueError("OpenWAM's benchmarks/vlabench/single_eval.py must be pinned")
        campaign = os.environ.get(vlabench_client.SINGLE_EVAL_ENV)
        if not campaign or Path(campaign).resolve() != single_eval:
            raise ValueError("the OEH configuration names another single_eval.py than the campaign")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        vlabench_client.refuse(f"OEH policy configuration rejected: {exc}", expected)
    vlabench_entry.accept_validated(path, cfg)
    return vlabench_client.run(direct=False, make_policy=vlabench_entry.make_policy)


if __name__ == "__main__":
    raise SystemExit(main())
