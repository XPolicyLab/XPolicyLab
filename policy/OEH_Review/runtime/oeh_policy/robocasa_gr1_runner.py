"""The review condition's RoboCasa GR1 client: the pinned ``single_eval.py`` loop with ``oeh_policy.robocasa_gr1_entry``.

    python -m oeh_policy.robocasa_gr1_runner --openwam-root R -- --config C --env-id E --host H --port P

The loop runs exactly as condition 1 runs it (``oeh_eval.robocasa_gr1_client.run``: the pinned ``single_eval.py``'s
``main`` with the evaluator's evidence, seeding and native-episode hooks); only the policy object differs, the OEH
endpoint built from the released class and the loop's own arguments. The frozen OEH configuration is validated first,
once; one the policy refuses is reported to the campaign as a refusal to start.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from . import robocasa_gr1_entry
from .robocasa_gr1 import RoboCasaGR1Contract


def main(argv=None):
    from oeh_eval import robocasa_gr1_client, runtime_config
    from oeh_eval.libero_official import split_argv
    argv = sys.argv[1:] if argv is None else list(argv)
    expected = json.loads(os.environ.get(runtime_config.EXPECTED_ENV) or "{}")
    try:
        path = robocasa_gr1_entry.policy_config_path()
        cfg = RoboCasaGR1Contract().validate_config(json.loads(path.read_text(encoding="utf-8")))
        native = cfg["robocasa_gr1"]
        own, _ = split_argv(argv)
        if "--openwam-root" not in own or own.index("--openwam-root") + 1 >= len(own):
            raise ValueError("the client arguments name no --openwam-root")
        openwam = Path(own[own.index("--openwam-root") + 1]).resolve()
        if openwam != Path(native["openwam_root"]).resolve():
            raise ValueError("the OEH configuration names another OpenWAM root than the campaign")
        root = os.environ.get("ROBOCASA_GR1_PATH")
        if not root or Path(root).resolve() != Path(native["root"]).resolve():
            raise ValueError("the OEH configuration names another RoboCasa GR1 checkout than the campaign")
        runner = Path(native["official_runner_path"]).resolve()
        if runner != openwam / robocasa_gr1_client.CLIENT_DIR / "single_eval.py" or \
                cfg["identity_files"].get(str(runner)) is None:
            raise ValueError("the released single_eval.py must be pinned")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        robocasa_gr1_client.refuse(f"OEH policy configuration rejected: {exc}", expected)
    robocasa_gr1_entry.accept_validated(path, cfg)
    return robocasa_gr1_client.run(argv, factory=robocasa_gr1_entry.make_policy)


if __name__ == "__main__":
    raise SystemExit(main())
