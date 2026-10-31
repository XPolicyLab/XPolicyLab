"""Check optional trajectory configuration without launching a policy server."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

POLICY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("trajectory", [None, 'shared runs/quoted "path" $(literal)'])
def test_server_passes_trajectory_path_as_one_standard_override(tmp_path, trajectory):
    policy = tmp_path / "XPolicyLab/policy/SenseNova-RoboRSI"
    policy.mkdir(parents=True)
    shutil.copy2(POLICY / "setup_eval_policy_server.sh", policy)
    python = tmp_path / "environment/bin/python"
    python.parent.mkdir(parents=True)
    events = tmp_path / "calls.jsonl"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:2] == ['-c']:\n"
        "    sys.argv = [sys.argv[0], *sys.argv[3:]]\n"
        "    print(json.dumps(sys.argv[1]))\n"
        "else:\n"
        "    with Path(os.environ['TEST_SERVER_CALLS']).open('a') as stream:\n"
        "        stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    python.chmod(0o755)
    environment = {
        **os.environ,
        "CODEX_API_KEY": "synthetic-unused-test-key",
        "TEST_SERVER_CALLS": str(events),
    }
    environment.pop("ROBORSI_TRAJECTORY_DIR", None)
    if trajectory is not None:
        environment["ROBORSI_TRAJECTORY_DIR"] = trajectory
    subprocess.run(
        [
            "bash", str(policy / "setup_eval_policy_server.sh"),
            "RoboDojo", "align_blocks", "roborsi-codex-v1", "arx_x5", "joint",
            "0", "0", str(python.parent.parent), "12345",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in events.read_text().splitlines()]
    assert len(calls) == 2 and Path(calls[-1][0]).name == "serve.py"
    overrides = [value for value in calls[-1] if value.startswith("trajectory_dir=")]
    assert len(overrides) == (trajectory is not None)
    if trajectory is not None:
        assert json.loads(overrides[0].split("=", 1)[1]) == trajectory
