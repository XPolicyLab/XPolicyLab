"""Export verified Playground knowledge into an XPolicyLab Test workspace."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from orchestrator.config import SESSION_PROTOCOL, config_from_dict
from orchestrator.files import hashes, run_lock, write_json
from orchestrator.session_runner import check_run


TEMPLATE = Path(__file__).resolve().parent.parent / "Workspace"


def export_workspace(run, output):
    run, output = Path(run).resolve(), Path(output).absolute()
    if not run.is_dir():
        raise ValueError(f"Missing Playground run: {run}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Export requires a new workspace: {output}")
    if output.resolve().is_relative_to(run):
        raise ValueError("Export outside the Playground run to preserve its evidence")
    with run_lock(run):
        cfg = config_from_dict(json.loads((run / "resolved.json").read_text()))
        state = check_run(cfg, run)
        result = json.loads((run / "playground/result.json").read_text())
        timing = json.loads((run / "playground/timing.json").read_text())
        if (state.get("protocol") != SESSION_PROTOCOL or state.get("phase") != "frozen"
                or state.get("playground_valid") is not True or cfg.baseline
                or cfg.simulator.backend != "robodojo" or cfg.playground is None
                or result.get("valid") is not True or result.get("failure")
                or timing.get("simulator_ready_at") is None
                or timing.get("agent_started_at") is None):
            raise ValueError("Export requires a completed, valid, frozen RoboDojo Playground")
        native_task = cfg.simulator.env_task
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", native_task):
            raise ValueError("Invalid Playground task name")
        task = native_task.removesuffix("_random")
        frozen = run / "frozen_harness"
        if (frozen.is_symlink() or not frozen.is_dir()
                or {p.name for p in frozen.iterdir()} != {"skills", "lessons"}
                or any(not (frozen / name).is_dir() for name in ("skills", "lessons"))):
            raise ValueError("Frozen knowledge must contain only skills/ and lessons/")
        expected = state.get("harness")
        if not isinstance(expected, dict) or hashes(frozen) != expected:
            raise ValueError("Frozen Playground knowledge checksum mismatch")
        recipe = json.loads((run / "recipe.json").read_text())
        if type(recipe.get("enabled")) is not bool:
            raise ValueError("Missing Playground recipe condition")
        recipe_name = f"{task}.md"
        recipe_data = (TEMPLATE / "recipes" / recipe_name).read_bytes()
        attribution = json.loads((TEMPLATE / "recipes/manifest.json").read_text())
        digest = hashlib.sha256(recipe_data).hexdigest()
        if attribution["files"].get(recipe_name) != digest:
            raise ValueError("Bundled recipe checksum mismatch")
        if recipe["enabled"] and (recipe.get("task") != task or recipe.get("sha256") != digest):
            raise ValueError("Playground and Test must use the same supplied recipe")

        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish only after every copied file has passed its checksum check.
        with tempfile.TemporaryDirectory(prefix=".embodiedevo-export-", dir=output.parent) as temporary:
            workspace = Path(temporary) / "Workspace"
            workspace.mkdir()
            for name in ("instruction.md", "primitives/primitives.md", "scripts/env.py",
                         "submission/solution.py", "recipes/LICENSE", "recipes/manifest.json",
                         f"recipes/{recipe_name}"):
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(TEMPLATE / name, target)
            for component in ("skills", "lessons"):
                shutil.copytree(frozen / component, workspace / component / task)
            copied = {f"{component}/{name}": digest
                      for component in ("skills", "lessons")
                      for name, digest in hashes(workspace / component / task).items()}
            if copied != expected or hashes(frozen) != expected:
                raise ValueError("Frozen knowledge changed during export")
            write_json(workspace / "experience-manifest.json", {
                "model": cfg.agent.model, "reasoning_effort": cfg.agent.reasoning_effort,
                "playground_seed": cfg.playground.seed, "source_run": run.name,
                "source_protocol_sha256": hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest(),
                "tasks": {task: {"native_variants": [native_task], "files": expected,
                                 "recipe": recipe}},
            })
            # mkdir is exclusive, including for an empty pre-existing directory.
            output.mkdir()
            try:
                for path in workspace.iterdir():
                    path.rename(output / path.name)
            except BaseException:
                shutil.rmtree(output)
                raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New Test workspace (ckpt_name path)")
    args = parser.parse_args(argv)
    print(export_workspace(args.run_dir, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
