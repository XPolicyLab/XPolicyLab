"""Run the official server with graceful termination for owned CLI processes."""

from importlib import import_module

import runpy
import signal
from pathlib import Path


def _terminate(signum, frame):
    close_models = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model").close_models

    # Cancel before Python joins server worker threads that may be awaiting MCP.
    close_models()
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "setup_policy_server.py"),
        run_name="__main__",
    )
