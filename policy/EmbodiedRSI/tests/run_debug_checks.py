"""Exercise a detached submission in an official debug workspace, without an LLM.

Usage: python tests/run_debug_checks.py --env-cfg /official/RoboDojo/env_cfg --run-dir /new/check
Only public robot config is copied. No simulator, result, credential or vendor code is copied.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-cfg", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch", action="store_true", help="Also exercise parallel episode workspaces")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    from XPolicyLab.policy.EmbodiedRSI.runtime.workspace import POLICY_DIR
    policy = POLICY_DIR
    xpl = policy.parent.parent
    detached = run / "workspace/XPolicyLab"
    detached.mkdir(parents=True)
    # Only the official server/client, utilities and this submission package.
    for name in ("XPolicyLab.py", "model_template.py", "setup_policy_server.py"):
        shutil.copy2(xpl / name, detached / name)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "checkpoints")
    for name in ("client_server", "utils"):
        shutil.copytree(xpl / name, detached / name, ignore=ignore)
    shutil.copytree(policy, detached / "policy/EmbodiedRSI", ignore=ignore)
    config = run / "workspace/env_cfg"
    (config / "robot").mkdir(parents=True)
    for name in ("arx_x5.yml", "robot/_robot_info.json"):
        shutil.copy2(args.env_cfg / name, config / name)
    env = {**os.environ, "PYTHONPATH": str(detached), "PYTHONDONTWRITEBYTECODE": "1", "EVAL_ENV_TYPE": "debug"}
    copied = detached / "policy/EmbodiedRSI"
    if args.batch:
        config_path = copied / "deploy.yml"
        config_path.write_text(config_path.read_text().replace("eval_batch: false", "eval_batch: true"))
    results = {"diagnostic": True, "batch": args.batch, "model_called": False, "simulator_used": False, "runs": []}
    for encoded in ("0", "1"):
        episode_root = run / ("encoded" if encoded == "1" else "plain")
        command = ["bash", str(copied / "eval.sh"), "RoboDojo", "stack_bowls", "diagnostic",
                   "arx_x5", "ee", "1", "0", "0", sys.executable, sys.executable]
        case_env = {**env, "DEBUG_OBS_ENCODED": encoded, "EMBODIEDRSI_RUN_DIR": str(episode_root),
                    "EMBODIEDRSI_DEBUG_EPISODES": "2"}
        result = subprocess.run(command, env=case_env, cwd=copied, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        (run / f"debug_{encoded}.txt").write_text(output)
        if result.returncode or "[MAIN] eval finished" not in output or "Traceback" in output:
            raise RuntimeError(f"Debug {encoded} failed ({result.returncode}); inspect {run}/debug_{encoded}.txt")
        episodes = sorted((episode_root / "episodes").glob("episode_*"))
        import yaml
        batched = yaml.safe_load((copied / "deploy.yml").read_text())["eval_batch"]
        expected = 20 if batched else 2  # Official TestEnv Batch_Size=10, two rounds.
        if len(episodes) != expected:
            raise AssertionError(f"Expected {expected} independent episode workspaces, found {len(episodes)}")
        for episode in episodes:
            status = json.loads((episode / "runtime.json").read_text())
            assert status["test_reset_denied"] and status["persistent_globals"] and not status["model_called"]
            assert (episode / "ledger.jsonl").is_file()
            assert not (episode / "workspace/ledger.jsonl").exists()
        results["runs"].append({"encoded": encoded == "1", "episodes": len(episodes), "command": command, "passed": True})
    (run / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
