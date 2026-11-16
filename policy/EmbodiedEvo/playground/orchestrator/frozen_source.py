"""Host-only provenance and immutable Skills/Lessons for external Test campaigns."""

import hashlib
import json
import shutil
from pathlib import Path

from orchestrator.config import SESSION_PROTOCOL, config_dict
from orchestrator.files import hashes, write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def references(campaign, tasks, *, task_mapping=None):
    """Identify producers without opening any scene list or unfinished workspace."""
    campaign = Path(campaign).resolve()
    manifest = read(campaign / "campaign.json")
    if (manifest.get("protocol") != SESSION_PROTOCOL or manifest.get("test_only")
            or manifest.get("suite") != "coverage"):
        raise ValueError("Source must be a Coverage Playground campaign")
    sources = {t["id"]: t for t in manifest["tasks"]}
    result = {}
    for task in tasks:
        source = sources.get(task_mapping[task["id"]] if task_mapping else task["id"])
        target_simulator = task["config"]["simulator"]
        source_simulator = source["config"]["simulator"] if source else {}
        if task_mapping:
            if source is None or source["id"] == task["id"]:
                raise ValueError("Cross-task transfer requires a distinct, existing source")
            source_simulator = {**source_simulator, "env_task": target_simulator["env_task"]}
        if (source is None or source["benchmark"] != task["benchmark"]
                or source["config"].get("baseline")
                or source_simulator != target_simulator
                or source["config"].get("playground", {}).get("seed") != task["config"]["test"]["excluded_seed"]):
            raise ValueError(f"Source task/environment mismatch: {task['id']}")
        plan = Path(task["config"]["test"]["shared_plan"])
        if not plan.is_file():
            raise ValueError(f"Test-only requires an existing shared scene plan: {plan}")
        # Hash bytes only; scene identities stay sealed until this task freezes.
        result[task["id"]] = {
            "campaign": str(campaign), "campaign_id": manifest["id"],
            "task_id": source["id"], "run": source["run"],
            "definition": {"config": source["config"]},
            "shared_plan": str(plan), "shared_plan_sha256": digest(plan),
        }
        if task_mapping:
            result[task["id"]]["transfer"] = {
                "target_task_id": task["id"], "target_simulator": target_simulator,
            }
    return result


def snapshot(source):
    campaign = Path(source["campaign"])
    manifest = read(campaign / "campaign.json")
    task = next((t for t in manifest.get("tasks", []) if t["id"] == source["task_id"]), {})
    if (manifest.get("id") != source["campaign_id"] or task.get("run") != source["run"]
            or task.get("config") != source["definition"]["config"]):
        raise ValueError("External Playground identity/configuration changed")
    run = Path(source["run"])
    protocol = read(run / "protocol.json")
    worker = read(campaign / "jobs" / f"{source['task_id']}_playground" / "worker.json")
    result = read(run / "playground/result.json")
    if (worker.get("status") != "finished" or worker.get("valid", True) is not True
            or protocol.get("phase") != "frozen" or protocol.get("playground_valid") is not True
            or result.get("valid") is not True or result.get("failure")):
        return None  # Failed/unknown sources remain dependencies, never rerun here.
    frozen = run / "frozen_harness"
    if (protocol.get("definition") != source["definition"]
            or protocol.get("harness") != hashes(frozen)
            or {p.name for p in frozen.iterdir()} != {"skills", "lessons"}
            or any(not (frozen / name).is_dir() for name in ("skills", "lessons"))):
        raise ValueError("External frozen harness or task definition changed")
    return {"harness": protocol["harness"],
            "protocol_sha256": digest(run / "protocol.json"),
            "result_sha256": digest(run / "playground/result.json")}


def prepare(cfg, run, source):
    """Return None while pending; otherwise pin only H and the new Test inputs."""
    from orchestrator import session_runner

    pinned = read(run / "protocol.json")
    target_simulator = config_dict(cfg)["simulator"]
    if source.get("transfer", {}).get("target_simulator", target_simulator) != target_simulator:
        raise ValueError("Cross-task target configuration changed")
    current = snapshot(source)
    if current is None:
        if pinned:
            raise ValueError("Previously pinned Playground is no longer frozen")
        return None
    plan = Path(source["shared_plan"])
    if str(plan) != cfg.test.shared_plan or not plan.is_file() or digest(plan) != source["shared_plan_sha256"]:
        raise ValueError("Existing shared Test plan changed or disappeared; refusing to resample")
    records = source.get("program_records")
    if records:
        from orchestrator.program_records import validate_bundle

        validate_bundle(records, source_snapshot=current)
    expected_harness = {} if records else current["harness"]
    if pinned:
        if (pinned.get("frozen_source") != source or pinned.get("source_snapshot") != current
                or pinned.get("execution_backend") != "xpolicylab" or pinned.get("phase") != "test"
                or pinned.get("harness") != expected_harness
                or hashes(run / "frozen_harness") != expected_harness
                or (records and read(run / "program-records.json") != records)
                or (run / "playground").exists()):
            raise ValueError("Pinned external knowledge/provenance changed")
        return pinned
    run.mkdir(parents=True, exist_ok=True)
    if any(p.name != ".protocol.lock" for p in run.iterdir()):
        raise ValueError("External Test requires a fresh task run or its pinned protocol")
    inputs = session_runner.prepare_inputs(cfg, run)
    if records:
        for name in ("skills", "lessons"):
            shutil.rmtree(run / "inputs" / name)
            (run / "inputs" / name).mkdir()
            (run / "frozen_harness" / name).mkdir(parents=True)
        instruction = Path(records["instruction"])
        if digest(instruction) != records["instruction_sha256"]:
            raise ValueError("C instruction changed")
        shutil.copyfile(instruction, run / "inputs/test.md")
        from orchestrator.recipes import append_recipe

        write_json(run / "recipe.json", append_recipe(run / "inputs", cfg.simulator.env_task))
        inputs = hashes(run / "inputs")
        write_json(run / "program-records.json", records)
    else:
        shutil.copytree(Path(source["run"]) / "frozen_harness", run / "frozen_harness")
    if hashes(run / "frozen_harness") != expected_harness or snapshot(source) != current:
        raise ValueError("External knowledge changed during pinning")
    state = {
        "protocol": SESSION_PROTOCOL, "phase": "test", "execution_backend": "xpolicylab",
        "definition": session_runner.definition(cfg), "inputs": inputs,
        "harness": expected_harness, "frozen_source": source, "source_snapshot": current,
    }
    write_json(run / "protocol.json", state)
    return state
