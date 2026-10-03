"""Assemble an installed native runtime into a CLI task package without rollout."""

import argparse
import json
from pathlib import Path

from jsonschema import validate

from PhysicalRSI_core.infra.storage import atomic_json, digest, identifier, read_json

from .native_runtime import NativeRuntime
from .workflow import create


def build(
    runtime,
    output,
    *,
    name,
    tasks,
    development_episodes,
    validation_episodes,
    minimum_vla_score,
    rollout_batch_size=1,
    proposal_mode="offline",
):
    """Preserve all dependency pins; incomplete policy assembly remains an explicit preflight failure."""
    identifier(name)
    configuration = dict(
        tasks=list(tasks),
        models={
            key: entry["spec"]["expert"]["revision"]
            for key, entry in runtime["experts"].items()
        },
        development_episodes=development_episodes,
        validation_episodes=validation_episodes,
        minimum_vla_score=minimum_vla_score,
        rollout_batch_size=rollout_batch_size,
        proposal_mode=proposal_mode,
        runtime_factory="PhysicalRSI_baselines.robodojo.native_runtime:create",
        runtime=runtime,
    )
    validate(configuration, create.configuration_schema)
    backend = NativeRuntime(runtime)
    report = backend.check()
    specification = dict(
        schema="physicalrsi.task/v1",
        name=name,
        entrypoint="PhysicalRSI_baselines.robodojo.workflow:create",
        configuration=configuration,
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "task.json", specification)
    receipt = dict(
        schema="physicalrsi.robodojo.native-package/v1",
        task_sha256=digest(specification),
        runtime_sha256=digest(runtime),
        preflight=report,
        physical_qualification=False,
        scope="installed artifact preflight only; no API, model or simulator launched",
    )
    atomic_json(output / "preflight.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--development-episodes", type=int, required=True)
    parser.add_argument("--validation-episodes", type=int, required=True)
    parser.add_argument("--minimum-vla-score", type=float, required=True)
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument(
        "--proposal-mode", choices=("offline", "api"), default="offline"
    )
    args = parser.parse_args()
    receipt = build(
        read_json(args.runtime),
        args.output,
        name=args.name,
        tasks=args.tasks,
        development_episodes=args.development_episodes,
        validation_episodes=args.validation_episodes,
        minimum_vla_score=args.minimum_vla_score,
        rollout_batch_size=args.rollout_batch_size,
        proposal_mode=args.proposal_mode,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["preflight"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
