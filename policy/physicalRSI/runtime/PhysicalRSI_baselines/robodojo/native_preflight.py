"""Inspect native client wiring without importing Isaac or running a policy."""

import argparse
import importlib
import json
import os
import sys
from importlib.resources import files
from pathlib import Path

from PhysicalRSI_core.infra.batch import ProcessJob, run_batch
from PhysicalRSI_core.infra.storage import atomic_json, file_digest, read_json


def client_files(simulator_root):
    root = Path(simulator_root).resolve()
    main = root / "src/eval_client/main.py"
    source = main.read_text()
    if "ROBODOJO_RESULT_ROOT" not in source or "multi_gpu=False" not in source:
        raise ValueError("Native simulator requires release client patches")
    bundle = files("PhysicalRSI_baselines.robodojo.xpolicylab_policy")
    selected = [main]
    for name in ("__init__.py", "deploy.py", "deploy.yml"):
        target = root / "XPolicyLab/policy/physicalRSI" / name
        if target.read_bytes() != (bundle / name).read_bytes():
            raise ValueError("Native physicalRSI client differs from installed adapter")
        selected.append(target)
    return {str(path): file_digest(path) for path in selected}


def worker(root, output):
    dependencies = client_files(root)
    sys.path.insert(0, str(Path(root).resolve()))
    # This matches native_worker's import precedence, but deliberately does not
    # import the environment, AppLauncher or any policy/model factory.
    deploy = importlib.import_module("XPolicyLab.policy.physicalRSI.deploy")
    from . import evaluation

    if (
        deploy.eval_one_episode is not evaluation.eval_one_episode
        or deploy.eval_one_episode_batch is not evaluation.eval_one_episode_batch
    ):
        raise ValueError("Native import resolves to a different evaluation loop")
    for module in (deploy, evaluation):
        path = Path(module.__file__).resolve()
        dependencies[str(path)] = file_digest(path)
    atomic_json(
        output,
        dict(
            ready=True,
            scope="native client import and configuration only",
            physical_qualification=False,
            python=sys.executable,
            files=dependencies,
        ),
    )


def inspect_client(simulator_root, *, python, environment, output, timeout_s=30):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = root / "client.json"
    job = ProcessJob(
        "client-import",
        (
            str(python),
            "-m",
            "PhysicalRSI_baselines.robodojo.native_preflight",
            "--worker",
            str(Path(simulator_root).resolve()),
            str(report),
        ),
        root,
        environment,
        timeout_s,
    )
    run_batch([job], root / "processes", workers=1)
    process = read_json(root / "processes/client-import/process.json")
    if process.get("state") != "completed" or process.get("returncode") != 0:
        result = dict(
            ready=False,
            physical_qualification=False,
            error="Native client preflight failed; inspect processes/client-import/process.log",
        )
    else:
        result = read_json(report)
        for path, sha in result["files"].items():
            if file_digest(Path(path)) != sha:
                raise ValueError("Native client changed during inspection")
    atomic_json(root / "preflight.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pythonpath", default=os.environ.get("PYTHONPATH", ""))
    parser.add_argument("simulator_root")
    parser.add_argument("output")
    args = parser.parse_args()
    if args.worker:
        worker(args.simulator_root, args.output)
    else:
        result = inspect_client(
            args.simulator_root,
            python=args.python,
            environment={"PYTHONPATH": args.pythonpath},
            output=args.output,
        )
        print(json.dumps(result, indent=2))
        if not result["ready"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
