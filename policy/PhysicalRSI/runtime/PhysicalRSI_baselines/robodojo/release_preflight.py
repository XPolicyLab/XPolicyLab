"""Run the offline, non-executing checks for a frozen RoboDojo task package."""

import argparse
import json
from pathlib import Path

from jsonschema import ValidationError, validate

from PhysicalRSI.tasks import configuration_schema, load_spec
from PhysicalRSI_core.infra.storage import atomic_json


WORKFLOW = "PhysicalRSI_baselines.robodojo.workflow:create"
NATIVE_RUNTIME = "physicalrsi.robodojo.native-runtime/v1"


def inspect_package(task_path, *, output=None):
    """Return a release report without creating a task session or running it."""
    binding = load_spec(task_path)
    specification = binding["specification"]
    report = dict(
        schema="physicalrsi.robodojo.release-preflight/v1",
        task=str(Path(task_path).expanduser().resolve()),
        task_sha256=binding["specification_sha256"],
        adapter_sha256=binding["adapter_sha256"],
        physical_qualification=False,
        execution_started=False,
        api_required=False,
    )
    schema = configuration_schema(binding)
    try:
        if schema is not None:
            validate(specification["configuration"], schema)
    except ValidationError as error:
        report.update(ready=False, error="Invalid task configuration: " + error.message)
    else:
        configuration = specification["configuration"]
        mode = configuration.get("proposal_mode")
        if mode == "api":
            report["api_required"] = True
        elif mode not in (None, "offline"):
            report.update(ready=False, error="Unknown proposal mode: " + str(mode))

        if report.get("ready", True) and specification["entrypoint"] == WORKFLOW:
            runtime_configuration = configuration.get("runtime", {})
            if runtime_configuration.get("schema") == NATIVE_RUNTIME:
                try:
                    from .native_runtime import NativeRuntime

                    native = NativeRuntime(runtime_configuration)
                    report["runtime"] = native.check()
                    report["ready"] = report["runtime"].get("ready") is True
                except (KeyError, OSError, TypeError, ValueError) as error:
                    report.update(
                        ready=False,
                        runtime={
                            "ready": False,
                            "physical_qualification": False,
                            "error": str(error),
                        },
                    )
            else:
                report.update(
                    ready=True,
                    runtime={
                        "ready": True,
                        "scope": "custom runtime; adapter check not executed",
                        "physical_qualification": False,
                    },
                )
        else:
            report.setdefault("ready", True)
    if output is not None:
        atomic_json(Path(output).resolve(), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_package(args.task, output=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ready") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
