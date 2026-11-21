"""OEH condition's EBench client: OpenWAM's released bridge loop with ``oeh_policy.ebench_entry`` as its driver.

The loop runs exactly as condition 1 runs it (``oeh_eval.ebench_client.run``: the local job submission, the released
``main`` with the case's arguments, the evaluator's evidence and native-episode hooks); only the driver differs. The
frozen OEH configuration is validated first, once; one the policy refuses is reported to the campaign as a refusal to
start.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from . import ebench_entry
from .ebench import EBenchContract


def main():
    from oeh_eval import ebench_client, runtime_config
    expected = json.loads(os.environ.get(runtime_config.EXPECTED_ENV) or "{}")
    try:
        path = ebench_entry.policy_config_path()
        cfg = EBenchContract().validate_config(json.loads(path.read_text(encoding="utf-8")))
        bridge = Path(cfg["ebench"]["official_bridge_path"]).resolve()
        if cfg["identity_files"].get(str(bridge)) is None:
            raise ValueError("the released EBench bridge must be pinned")
        settings = json.loads(os.environ.get(ebench_client.SETTINGS_ENV) or "{}")
        if settings.get("bridge_path") and Path(settings["bridge_path"]).resolve() != bridge:
            raise ValueError("the OEH configuration names another EBench bridge than the campaign")
        native = cfg["ebench"]
        for key in ("config_path", "task_name", "seed"):
            if key in settings and settings[key] != native[key]:
                raise ValueError(f"the OEH configuration's ebench.{key} differs from the campaign's case")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        ebench_client.refuse(f"OEH policy configuration rejected: {exc}", expected)
    ebench_entry.accept_validated(path, cfg)
    return ebench_client.run(direct=False, driver=ebench_entry.driver_class, bridge_path=bridge)


if __name__ == "__main__":
    raise SystemExit(main())
